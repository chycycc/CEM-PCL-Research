### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from src.utils import config
from src.utils.constants import MAP_EMO

from sklearn.metrics import accuracy_score


# ================= [EPCL v6.2: 特征空间解耦 (Projection Head) 方案] =================
# 核心思想: 用非线性投影头隔离"对比学习空间"和"语言生成空间"，
# 使 alignment/uniformity 的梯度经过投影层衰减后才回传主干，保护 PPL 底盘。
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, num_prototypes, input_dim, temperature=0.3,
                 t_uniform=2.0, alpha_uni=1.0):  # v6.2: 恢复 1.0，投影头充当梯度缓冲器
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.t_uniform = t_uniform
        self.alpha_uni = alpha_uni

        # 【投影层构建】非线性映射: input_dim → bottleneck → input_dim
        # bottleneck=128 强制信息压缩，只保留情感判别性特征
        # ReLU 自动切断部分负梯度通道，形成天然减震器
        proj_hidden = 128
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, input_dim)
        )

        # 情感原型驻扎在投影后的对比子空间中
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def uniformity_loss(self, normalized_prototypes):
        """仅在原型之间计算排斥力，梯度 100% 只流向 self.prototypes"""
        sq_pdist = 2.0 - 2.0 * torch.matmul(
            normalized_prototypes, normalized_prototypes.T
        )
        mask = torch.eye(
            normalized_prototypes.size(0),
            device=normalized_prototypes.device
        ).bool()
        sq_pdist = sq_pdist.masked_fill(mask, float('inf'))
        return torch.logsumexp(-self.t_uniform * sq_pdist, dim=1).mean()

    def forward(self, features, labels, tau=None):
        current_tau = tau if tau is not None else self.temperature

        # 【空间转移】原始 features 原封不动保留给 Decoder（不截断主干计算图）
        # 投影层将 features 映射到对比专属子空间
        projected_features = self.projection_head(features)

        # 所有对比度量在投影子空间的超球面上进行
        proj_norm = F.normalize(projected_features, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Alignment Loss: 拉近投影后样本与目标原型
        logits = torch.matmul(proj_norm, proto_norm.T) / current_tau
        loss_align = F.cross_entropy(logits, labels)

        # Uniformity Loss: 原型间排斥力（梯度不经过投影头）
        loss_uni = self.uniformity_loss(proto_norm)

        return loss_align + self.alpha_uni * loss_uni
# ==============================================================================


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class CEM(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
    ):
        super(CEM, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        # === [新增] 初始化 EPCL Loss ===
        self.epcl_criterion = PrototypeContrastiveLoss(decoder_number, config.hidden_dim).to(config.device)
        # ================================

        self.word_freq = np.zeros(self.vocab_size)

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = self.make_encoder(config.emb_dim)
        self.emo_encoder = self.make_encoder(config.emb_dim)
        self.cog_encoder = self.make_encoder(config.emb_dim)
        self.emo_ref_encoder = self.make_encoder(2 * config.emb_dim)
        self.cog_ref_encoder = self.make_encoder(2 * config.emb_dim)

        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.emo_lin = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        if not config.woCOG:
            self.cog_lin = MLP()

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "CEM_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    # === [新增] 按 Accuracy 保存最优权重 ===
    def save_model_acc(self, best_acc, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "best_acc": best_acc,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "CEM_ACC_{}_{:.4f}".format(iter, best_acc),
        )
        torch.save(state, model_save_path)
    # ======================================

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)

    def forward(self, batch, need_rep=False):  # [修改] 增加 need_rep 参数
        ## Encode the context (Semantic Knowledge)
        enc_batch = batch["input_batch"]
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300

        # Commonsense relations
        cs_embs = []
        cs_masks = []
        cs_outputs = []
        for r in self.rels:
            emb = self.embedding(batch[r]).to(config.device)
            mask = batch[r].data.eq(config.PAD_idx).unsqueeze(1)
            cs_embs.append(emb)
            cs_masks.append(mask)
            if r != "x_react":
                enc_output = self.cog_encoder(emb, mask)
            else:
                enc_output = self.emo_encoder(emb, mask)
            cs_outputs.append(enc_output)

        cls_tokens = [c[:, 0].unsqueeze(1) for c in cs_outputs]

        # Shape: batch_size * 1 * 300
        cog_cls = cls_tokens[:-1]
        emo_cls = torch.mean(cs_outputs[-1], dim=1).unsqueeze(1)

        dim = [-1, enc_outputs.shape[1], -1]

        # Emotion
        if not config.woEMO:
            emo_concat = torch.cat([enc_outputs, emo_cls.expand(dim)], dim=-1)
            emo_ref_ctx = self.emo_ref_encoder(emo_concat, src_mask)
            # === [修改 START] ===
            emo_rep = emo_ref_ctx[:, 0]
            emo_logits = self.emo_lin(emo_rep)
            # === [修改 END] ===
        else:
            # === [修改 START] ===
            emo_rep = enc_outputs[:, 0]
            emo_logits = self.emo_lin(emo_rep)
            # === [修改 END] ===

        # Cognition
        cog_outputs = []
        for cls in cog_cls:
            cog_concat = torch.cat([enc_outputs, cls.expand(dim)], dim=-1)
            cog_concat_enc = self.cog_ref_encoder(cog_concat, src_mask)
            cog_outputs.append(cog_concat_enc)

        if config.woCOG:
            cog_ref_ctx = emo_ref_ctx
        else:
            if config.woEMO:
                cog_ref_ctx = torch.cat(cog_outputs, dim=-1)
            else:
                cog_ref_ctx = torch.cat(cog_outputs + [emo_ref_ctx], dim=-1)
            cog_contrib = nn.Sigmoid()(cog_ref_ctx)
            cog_ref_ctx = cog_contrib * cog_ref_ctx
            cog_ref_ctx = self.cog_lin(cog_ref_ctx)

        # === [修改返回值] ===
        if need_rep:
            return src_mask, cog_ref_ctx, emo_logits, emo_rep
        else:
            return src_mask, cog_ref_ctx, emo_logits

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # === [修改] 训练时开启 need_rep=True ===
        src_mask, ctx_output, emo_logits, emo_rep = self.forward(batch, need_rep=True)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(dec_emb, ctx_output, (src_mask, mask_trg))

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label).to(config.device)
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )

        # === [v6.3 异步截断退火 + 拓扑锚点调度] ===
        # 核心思想: 让分类任务在 14k 步"先下课"，保留微量锚点防止灾难性遗忘
        if not train:
            # 验证集保持 0.07 满载计算，确保 TensorBoard 曲线数值全程可比
            lambda_epcl = 0.07
        else:
            if iter < 3000:
                # 阶段 1 (0-3k): 升温期，对比学习平滑介入
                lambda_epcl = 0.07 * (iter / 3000.0)
            elif iter < 12000:
                # 阶段 2 (3k-12k): 甜区稳定期，全力撕裂情感边界
                lambda_epcl = 0.07
            elif iter < 15000:
                # 阶段 3 (12k-15k): 线性衰减到锚点值，无跳变
                decay_progress = (iter - 12000) / 3000.0  # 0→1
                lambda_epcl = 0.07 * (1 - decay_progress) + 0.005 * decay_progress
            else:
                # 阶段 4 (15k+): 拓扑锚点维持期，压制漂移不引发过拟合
                lambda_epcl = 0.005
        # ============================================

        if train:
            loss_epcl = self.epcl_criterion(emo_rep, emo_label)
        else:
            loss_epcl = 0.0
        # ================================

        if not (config.woDiv):
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(config.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
            # [修改] 加入 EPCL 损失
            loss = emo_loss + 1.5 * div_loss + ctx_loss + (lambda_epcl * loss_epcl)
        else:
            # [修改] 加入 EPCL 损失
            loss = emo_loss + ctx_loss + (lambda_epcl * loss_epcl)

        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = emo_logits.detach().cpu().numpy().argsort()[0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"
            for r in self.rels:
                txt = [[" ".join(t) for t in tm] for tm in batch[f"{r}_txt"]][0]
                comet_res[r] = txt

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent