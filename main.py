from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform_

from src.utils import config
from src.utils.common import set_seed
from src.models.MOEL.model import MOEL
from src.models.MIME.model import MIME
from src.models.EMPDG.model import EMPDG
from src.models.CEM.model import CEM
from src.models.Transformer.model import Transformer
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters, make_infinite


def make_model(vocab, dec_num):
    is_eval = config.test
    if config.model == "trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    if config.model == "multi-trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            is_multitask=True,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "moel":
        model = MOEL(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "mime":
        model = MIME(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "empdg":
        model = EMPDG(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "cem":
        model = CEM(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set):
    check_iter = 2000   # 【恢复】从 50 改回 2000 (每 2000 步正式验证一次)
    try:
        model.train()
        best_ppl = 1000
        best_acc = 0.0  # [新增] 追踪最佳 Accuracy
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(1000000)): # 【恢复】从 100 改回 1000000 (完整训练模式)
            if "cem" in config.model:
                loss, ppl, bce, acc, _, _ = model.train_one_batch(
                    next(data_iter), n_iter
                )
            else:
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)

            writer.add_scalars("loss", {"loss_train": loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            writer.add_scalars("bce", {"bce_train": bce}, n_iter)
            writer.add_scalars("accuracy", {"acc_train": acc}, n_iter)
            if config.noam:
                writer.add_scalars(
                    "lr", {"learning_rata": model.optimizer._rate}, n_iter
                )

            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.epoch = n_iter
                loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                writer.add_scalars("loss", {"loss_valid": loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                writer.add_scalars("bce", {"bce_valid": bce_val}, n_iter)
                writer.add_scalars("accuracy", {"acc_train": acc_val}, n_iter)
                model.train()
                if n_iter < 12000:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                # === [新增] 双轨保存：按 Accuracy 独立保存 ===
                if acc_val >= best_acc:
                    best_acc = acc_val
                    if hasattr(model, 'save_model_acc'):
                        model.save_model_acc(best_acc, n_iter)
                # ============================================
                if patient > 2:
                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, n_iter)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set, suffix=""):
    """在测试集上评估模型并保存结果。
    
    Args:
        suffix: 文件名后缀，如 " - ppl_best" 或 " - acc_best"
    """
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    file_summary = config.save_path + f"/results{suffix}.txt"
    with open(file_summary, "w", encoding='utf-8') as f:
        f.write("EVAL\tLoss\tPPL\tAccuracy\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                loss_test, ppl_test, bce_test, acc_test
            )
        )
        for r in results:
            f.write(r)
    print(f"[评估完成] 结果已保存至: {file_summary}")
    print(f"  PPL = {ppl_test:.4f}, Accuracy = {acc_test:.4f}")
    return ppl_test, acc_test


def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, dec_num)

    if config.test:
        test(model, test_set)
    else:
        weights_best = train(model, train_set, dev_set)

        # === [v6.4] 训练结束后自动双检查点评估 ===
        import glob
        import os

        # 1) 评估 PPL-best（从内存中的 weights_best 加载）
        print("\n" + "=" * 60)
        print("[自动评估] 加载 PPL-best 权重...")
        print("=" * 60)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, test_set, suffix=" - ppl_best")

        # 2) 评估 ACC-best（从磁盘检查点加载）
        acc_ckpts = sorted(
            glob.glob(os.path.join(config.save_path, "CEM_ACC_*")),
            key=os.path.getmtime
        )
        if acc_ckpts:
            acc_ckpt_path = acc_ckpts[-1]  # 最新的 ACC-best 检查点
            print("\n" + "=" * 60)
            print(f"[自动评估] 加载 ACC-best 权重: {os.path.basename(acc_ckpt_path)}")
            print("=" * 60)
            import torch
            state = torch.load(acc_ckpt_path, map_location=config.device)
            model.load_state_dict(state["model"])
            test(model, test_set, suffix=" - acc_best")
        else:
            print("[警告] 未找到 ACC-best 检查点，跳过 ACC-best 评估")

        print("\n" + "=" * 60)
        print("[完成] 所有评估已自动执行")
        print("=" * 60)


if __name__ == "__main__":
    main()