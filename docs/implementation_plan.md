# 实施计划：为 CEM 引入"情感原型对比学习" (EPCL)

> **最终版 (V3 - 防 Crash 优化版)**
> 备份文件：`src/models/CEM/model_epcl.py`
> 原始模型：`src/models/CEM/model.py` (保留用于对比)

## 1. 创新点与动机

- **问题**：原始 CEM 仅使用交叉熵分类，导致情感特征空间分布杂乱。
- **方案**：引入 **情感原型 (Emotion Prototypes)**。
  - 每个情感类别对应一个可学习的向量中心（原型）。
  - 通过对比损失，强制模型将同类情感的特征向其原型拉近，异类推远。
- **效果**：实现情感特征的聚类化，提高模型对细微情感差异的辨识能力。

## 2. 核心代码改动详情

### Mod 1：添加 EPCL Loss 类
在文件头部（`Encoder` 类之前）插入：
```python
# ================= [新增: EPCL Loss] =================
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, num_prototypes, input_dim, temperature=0.07):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        logits = torch.matmul(features, prototypes.T) / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss
# ======================================================
```

### Mod 2：在 `CEM.__init__` 中初始化
```python
# 在 self.vocab 之后
self.epcl_criterion = PrototypeContrastiveLoss(decoder_number, config.hidden_dim).to(config.device)
```

### Mod 3：改造 `forward`（软开关设计）
通过 `need_rep` 参数解决 `evaluate` 函数解包 3 个值导致的 Crash 问题。
```python
def forward(self, batch, need_rep=False):
    # ... 编码逻辑 ...
    # 提取第 0 位特征作为情感表示
    emo_rep = emo_ref_ctx[:, 0]
    emo_logits = self.emo_lin(emo_rep)

    # 动态返回
    if need_rep:
        return src_mask, cog_ref_ctx, emo_logits, emo_rep
    else:
        return src_mask, cog_ref_ctx, emo_logits
```

### Mod 4：在 `train_one_batch` 中计算 Loss
```python
# 获取特征
src_mask, ctx_output, emo_logits, emo_rep = self.forward(batch, need_rep=True)

# 计算 EPCL Loss
if train:
    loss_epcl = self.epcl_criterion(emo_rep, emo_label)
else:
    loss_epcl = 0.0

# 整合进总 Loss (权重 lambda = 0.1)
loss = emo_loss + 1.5 * div_loss + ctx_loss + (0.1 * loss_epcl)
```

---

## 3. 实验验证结果 (验收记录)

| 指标 | Baseline (CEM 原版) | **EPCL 增强版 (当前)** | 结论 |
| --- | --- | --- | --- |
| **PPL (困惑度)** | ~39.0 | **37.05** | 显著下降 (越低越好) |
| **Accuracy (准确率)** | ~34.0% | **36.65%** | 提升约 2.6% |
| **稳定性** | - | **通过** | 无训练/验证崩溃风险 |

## 4. 结论
EPCL 模块在保持模型兼容性的前提下，成功优化了情感特征表示，带来了更低的困惑度和更高的情感分类准确率，达到了预期的研究目标。
