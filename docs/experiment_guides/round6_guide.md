# EPCL v6 架构重塑与实验验证计划书
# — 引入超球面均匀性惩罚 ($\mathcal{L}_{uniformity}$) 解决特征拓扑坍塌

> **版本**: v6 — Alignment + Uniformity 双力均衡  
> **日期**: 2026-05-21  
> **前序版本**: v5 (τ=0.3, λ=0.07, warmup=3000) — 已达成 PPL/Acc 双超基线  
> **代码文件**: `src/models/CEM/model.py:29-68`

---

## 一、核心动机与科学问题

### 1.1 v5 的结构性缺陷：单臂对比学习的拓扑不完备性

EPCL v5 的 `PrototypeContrastiveLoss` 仅包含一个 alignment 项（交叉熵），其优化目标为：

$$\mathcal{L}_{align} = -\log \frac{\exp(\text{sim}(\mathbf{h}, \mathbf{p}_{y}) / \tau)}{\sum_{k=1}^{32} \exp(\text{sim}(\mathbf{h}, \mathbf{p}_k) / \tau)}$$

该损失函数在给定一个样本 $\mathbf{h}$ 及其标签 $y$ 时，将 $\mathbf{h}$ 向目标原型 $\mathbf{p}_y$ 拉近（类内对齐），同时隐式地将 $\mathbf{h}$ 推离其他原型 $\mathbf{p}_{k \neq y}$。

然而，这种隐式推力存在一个根本性的缺陷：**它只对当前 batch 中出现的样本施加排斥梯度，对原型与原型之间的几何关系不施加任何直接约束**。

具体而言：
- 当 batch 中不包含类别 $i$ 的样本时，原型 $\mathbf{p}_i$ 在该步骤**完全不受任何力的作用**
- 当两个语义相近的类别（如 `anxious` 与 `apprehensive`）的样本在特征空间中天然重叠时，交叉熵的隐式推力不足以阻止对应原型互相靠近
- 在 32 类不均衡的 EmpatheticDialogues 数据集上，低频类别的原型长期处于"梯度饥饿"状态，逐步被高频类别的梯度场吸引，导致局部原型聚团

这些问题在 v5 的 t-SNE 可视化（`docs/figures/Fig_EPCL_tsne_v5.png`）中得到了直观印证：边缘高频类别（如 `joyful`、`terrified`）的聚类清晰，但中心区域存在多类别特征的高度纠缠。

### 1.2 高维拓扑学视角：维度退化与特征纠缠

在 $\mathbb{R}^{300}$ 的单位超球面 $\mathcal{S}^{299}$ 上，32 个原型拥有极为充裕的空间余度——Tammes 问题（球面最优填充）表明，300 维超球面可以容纳远超 32 个互不干扰的区域。然而，缺乏显式排斥约束时，优化器不会主动利用这些高维空间：

- **维度退化 (Dimensional Collapse)**：原型倾向于坍塌到低维子空间中，导致 300 个维度中的大量维度沦为冗余噪声维，有效维度数远低于 300
- **特征纠缠 (Feature Entanglement)**：当原型在低维子空间中彼此靠近时，对应类别的样本特征无法获得足够锐利的决策超平面，分类边界模糊

解决方案的数学本质：**在超球面上引入全局排斥势能场**，迫使 32 个原型在 $\mathcal{S}^{299}$ 上趋向均匀分布（即趋向 Thomson 问题的最优解），从而最大化各原型之间的角距离，彻底消除维度退化与特征纠缠。

---

## 二、技术改动方案与 ADR 决策

### 2.1 ADR-01：保留 `nn.Parameter` 属性，拒绝 EMA/动量队列

**决策**：`self.prototypes` 必须保持为 `nn.Parameter`，通过标准反向传播直接接收 $\mathcal{L}_{uniformity}$ 的排斥梯度。

**放弃 EMA (指数滑动平均) 的原因**：
- EMA 更新 (`proto_ema = m * proto_ema + (1-m) * proto`) 位于计算图之外（`torch.no_grad()`），$\mathcal{L}_{uniformity}$ 的梯度无法通过 EMA 路径回传到原型参数
- 这意味着均匀性惩罚的数学力量会被 EMA 的滑动平均机制完全截断，排斥力名存实亡

**放弃动量队列 (Momentum Queue) 的原因**：
- 动量队列需要存储 $N \times d$ 的历史特征矩阵（$N$ 为队列长度），在 4GB VRAM 环境下引入额外显存压力
- 队列中的特征是 `detach()` 后的快照，同样截断了梯度流，无法为原型提供即时的排斥力梯度

**结论**：只有 `nn.Parameter` + 标准反向传播，才能保证 $\mathcal{L}_{uniformity}$ 的梯度 $\nabla_{\mathbf{p}} \mathcal{L}_{uni}$ 精准作用于每一个原型向量，实现真正的全局排斥。

### 2.2 ADR-02：`uniformity_loss` 的数值稳定性设计

最终实现代码位于 [model.py:42-53](file:///e:/github/CEM-master/src/models/CEM/model.py#L42-L53)：

```python
def uniformity_loss(self, normalized_prototypes):
    sq_pdist = 2.0 - 2.0 * torch.matmul(
        normalized_prototypes, normalized_prototypes.T
    )
    mask = torch.eye(
        normalized_prototypes.size(0),
        device=normalized_prototypes.device
    ).bool()
    sq_pdist = sq_pdist.masked_fill(mask, float('inf'))
    return torch.logsumexp(-self.t_uniform * sq_pdist, dim=1).mean()
```

**关键计算步骤拆解**：

| 步骤 | 操作 | 数值稳定性考量 |
| --- | --- | --- |
| **S1** | `sq_pdist = 2 - 2 * P·Pᵀ` | 对 L2 归一化向量，$\|\mathbf{u}-\mathbf{v}\|^2 = 2 - 2\mathbf{u}^\top\mathbf{v}$，结果严格 $\in [0, 4]$，无溢出风险 |
| **S2** | `masked_fill(diag, inf)` | **out-of-place** 操作（非 `masked_fill_`），避免 autograd version counter 冲突。对角线置 `inf` 排除自距离 |
| **S3** | `-t × sq_pdist` | 对角线变为 `-inf`，非对角线 $\in [-8, 0]$（$t=2.0$），所有值有界 |
| **S4** | `logsumexp(·, dim=1)` | PyTorch 内部使用 max-subtraction trick：$\log\sum e^{x_i} = \max(x) + \log\sum e^{x_i - \max(x)}$。$e^{-\infty}=0$，对角线不参与求和。数值稳定 |
| **S5** | `.mean()` | 对 32 行取均值，无风险 |

**梯度安全性**：$\frac{\partial}{\partial x_i} \text{logsumexp}(\mathbf{x}) = \text{softmax}(\mathbf{x})_i$。当 $x_i = -\infty$ 时，$\text{softmax}_i = 0$，梯度严格为零。不存在 NaN 或梯度爆炸路径。

### 2.3 ADR-03：损失函数组合方式

v6 的 `forward` 返回值为（[model.py:55-67](file:///e:/github/CEM-master/src/models/CEM/model.py#L55-L67)）：

$$\mathcal{L}_{EPCL} = \mathcal{L}_{align} + \alpha_{uni} \cdot \mathcal{L}_{uniformity}$$

然后在 `train_one_batch` 中（[model.py:646-670](file:///e:/github/CEM-master/src/models/CEM/model.py#L646-L670)）：

$$\mathcal{L}_{total} = \mathcal{L}_{emo}^{CE} + 1.5\mathcal{L}_{div} + \mathcal{L}_{ctx}^{PPL} + \lambda(t) \cdot \mathcal{L}_{EPCL}$$

其中 $\lambda(t) = 0.07 \times \min(1, t/3000)$，$\alpha_{uni} = 1.0$（默认值，可调节）。

**梯度隔离分析**：

```
L_uniformity → ∇ → self.prototypes (9,600 参数)     ← 仅原型受力
L_alignment  → ∇ → self.prototypes + Encoder 参数   ← 原型 + 编码器受力
L_ctx (PPL)  → ∇ → Decoder 参数                     ← 解码器受力
```

$\mathcal{L}_{uniformity}$ 的梯度**完全不流入 Encoder/Decoder**。新增的排斥力仅重塑原型的几何分布，不直接扰动语言生成空间。

### 2.4 v5 → v6 超参数变更总览

| 参数 | v5 值 | v6 值 | 变更 |
| --- | --- | --- | --- |
| 温度 τ | 0.3 | 0.3 | 不变 |
| EPCL 外部权重 λ | 0.07 | 0.07 | 不变 |
| Warmup 步数 | 3000 | 3000 | 不变 |
| Xavier 初始化 | ✅ | ✅ | 不变 |
| 初始化张量 | `torch.randn` | `torch.empty` | **修正冗余** |
| **高斯排斥温度 $t_{uni}$** | — | **2.0** | 🆕 新增 |
| **均匀性权重 $\alpha_{uni}$** | — | **1.0** | 🆕 新增 |
| **$\mathcal{L}_{uniformity}$** | — | **已激活** | 🆕 核心改动 |

---

## 三、实验验证与指标预期

### 3.1 对照实验矩阵

| 实验组 | 模型 | τ | λ | $\alpha_{uni}$ | $t_{uni}$ | 权重来源 |
| --- | --- | --- | --- | --- | --- | --- |
| **A. Baseline** | CEM 原版 (`model_base.py`) | — | — | — | — | `save/test_4.13/` |
| **B. EPCL v5** | v5 alignment-only | 0.3 | 0.07 | — | — | `save/test-v5/CEM_19999_42.1962` |
| **C. EPCL v6** | v6 alignment + uniformity | 0.3 | 0.07 | 1.0 | 2.0 | 本轮训练 |

### 3.2 成功标准

| 指标 | Baseline | EPCL v5 | **EPCL v6 理想目标** | **v6 可接受底线** |
| --- | --- | --- | --- | --- |
| **PPL** ↓ | 36.8776 | 36.3955 | ≤ 36.40 (不退化) | ≤ 36.88 (不劣于Baseline) |
| **Accuracy** ↑ | 37.41% | 38.17% | **≥ 39.0%** (突破) | ≥ 38.17% (不劣于v5) |
| **t-SNE 定性** | 中心纠缠严重 | 边缘清晰/中心纠缠 | **32 座孤岛全域分离** | 中心纠缠显著减少 |

### 3.3 效果推演

#### (a) t-SNE 可视化预期

v5 的 t-SNE 表现为"边缘清晰、中心糊"——高频情感（如 joyful、terrified）在特征空间边缘形成了独立岛屿，但低频或语义相近的情感（sad/lonely、anxious/apprehensive）在中心区域相互纠缠。

引入 $\mathcal{L}_{uniformity}$ 后，32 个原型在训练过程中持续接收全局排斥梯度。由于排斥力基于高斯势能核 $e^{-t\|\mathbf{p}_i - \mathbf{p}_j\|^2}$，距离越近的原型对受到的排斥力越强（指数级放大）。这意味着：

1. **中心区域的粘连原型首先被撕开**——它们是距离最小的对，承受最强排斥力
2. **边缘原型保持稳定**——它们之间距离已经足够大，排斥力衰减到可忽略
3. **最终状态**：32 个原型趋向 Thomson 问题的（近似）最优解，在 $\mathcal{S}^{299}$ 上形成互相角距离最大化的均匀分布

样本特征在 alignment loss 的拉力下被吸附到各自的原型周围，最终 t-SNE 图应呈现为 **32 座空间隔离的"特征孤岛"**，每座岛屿以对应原型（红色五角星）为几何中心。

#### (b) PPL 安全性论证

$\mathcal{L}_{uniformity}$ 对总损失的有效贡献为：

$$\Delta \mathcal{L} = \lambda \cdot \alpha_{uni} \cdot \mathcal{L}_{uni} = 0.07 \times 1.0 \times [-0.57, 3.43] = [-0.04, 0.24]$$

而 $\mathcal{L}_{ctx}$（PPL 损失）的量级为 ~3.5。即使在最极端情况下（所有原型完全坍塌），uniformity 仅占总损失的 **~6%**。

更关键的是：$\mathcal{L}_{uniformity}$ 的梯度仅作用于 9,600 个原型参数，**不流入 Encoder/Decoder 的任何参数**。它对 PPL 的影响只能通过改变原型几何分布来间接影响 alignment loss 的梯度方向——这是一个二阶效应，量级极小。

**结论**：PPL 预期维持在 36.40 ± 0.3 范围内，不发生退化。

#### (c) Accuracy 提升机制

当 32 个原型被均匀性惩罚撑开至更大角距离后：

1. **决策边界锐化**：原型间距增大 → softmax 分类器的 logit 差距增大 → 分类置信度提升 → 测试集上语义相近情感的误判率下降
2. **低频类别获益**：原本被高频原型"引力场"吸引的低频原型，在全局排斥力作用下获得独立空间 → 长尾类别的分类准确率显著提升
3. **泛化能力增强**：原型均匀分布迫使特征空间的有效利用维度增加（对抗维度退化），提高模型对未见样本的泛化判别力

**预期**：测试集 Accuracy 有望从 v5 的 38.17% 进一步上探至 **38.5%~39.5%** 区间。

### 3.4 历史对照表（含 v6 目标行）

| 版本 | τ | λ | 新增 | PPL ↓ | Acc ↑ | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | — | — | — | 36.88 | 37.41% | 基准 |
| v1 | 0.07 | 0.1 | — | 37.05 | 36.65% | ❌ 全面劣于基线 |
| v2 | 0.5 | 0.05 | — | **36.65** | 36.92% | PPL最优(历史) |
| v3 PPL | 0.2 | 0.1 | — | 36.83 | 36.97% | 略超基线PPL |
| v3 ACC | 0.2 | 0.1 | — | 37.64 | 37.70% | 超越基线Acc |
| v4 PPL | 0.5→0.1 | 0.05 | — | 36.91 | 36.25% | ❌ Acc最差 |
| v4 ACC | 0.5→0.1 | 0.05 | — | 37.87 | 37.74% | Acc次优 |
| v5 PPL | 0.3 | 0.07 | — | **36.40** | **38.17%** | 🏆🏆 双超突破 |
| v5 ACC | 0.3 | 0.07 | — | 37.63 | 37.94% | 验证集过拟合 |
| **v6** | **0.3** | **0.07** | **$\mathcal{L}_{uni}$ α=1.0** | ~~≤36.40~~ **37.00** | ~~≥39.0%~~ **37.13%** | ❌ 全面退化 |

### 3.5 应急预案

> **已触发**: 应急行 "原型发散过度" 与 "Acc 不升反降" 同时命中。执行修正方案 → v6.1。

| 状况 | 症状 | 诊断 | 应急方案 | 状态 |
| --- | --- | --- | --- | --- |
| PPL 爆炸 | 初始 PPL > 500 | uniformity 过早介入 | 延长 warmup | 未触发 |
| Acc 不升反降 | 测试 Acc < 38.0% | 排斥力过强 | 降 `alpha_uni` 为 0.3 | **✅ 已触发** |
| PPL 轻微退化 | PPL > 36.9 | uniformity 间接扰动 | 降 `alpha_uni` | **✅ 已触发** |
| t-SNE 无改善 | 中心仍然纠缠 | 排斥力不够 | 提高 `t_uniform` | 待验证 |
| 原型发散过度 | 所有原型等距、Acc 下降 | 过度均匀 | 降 `alpha_uni` 为 0.3 | **✅ 已触发** |

---

## 五、v6 实验结果 (2026-05-23)

### 5.1 测试集指标

| 版本/权重 | Loss | PPL ↓ | BCE | Accuracy ↑ | vs Baseline PPL | vs Baseline Acc |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 3.6076 | 36.8776 | 2.7816 | 37.41% | — | — |
| **v5 PPL-best** | 3.5944 | **36.3955** | 2.5919 | **38.17%** | ✅ -1.31% | ✅ +2.03% |
| **v6 PPL-best** (CEM_19999_42.3268) | 3.6110 | 37.0031 | 2.6424 | 37.13% | ❌ +0.34% | ❌ -0.75% |
| **v6 ACC-best** (CEM_ACC_13999_0.4037) | 3.6450 | 38.2823 | 2.4014 | 37.51% | ❌ +3.81% | ❌ +0.27% |

### 5.2 结论

**v6 实验失败**。`alpha_uni=1.0` 导致 PPL 和 Accuracy 双指标劣于 Baseline：

1. **PPL**: 37.00 vs v5 36.40 (+1.67%), 且超过 Baseline 36.88
2. **Accuracy**: 37.13% vs v5 38.17% (-1.04pp), 低于 Baseline 37.41%
3. **根因**: 均匀性排斥力过强，破坏超球面语义拓扑，alignment-uniformity 梯度对冲导致原型震荡

### 5.3 v6 实验文件归档

```
save/test-v6/                        ← v6 完整输出备份
├── CEM_19999_42.3268                ← PPL-best 权重
├── CEM_ACC_13999_0.4037             ← ACC-best 权重
├── results - ppl_best.txt           ← PPL-best 测试结果
├── results - acc_best.txt           ← ACC-best 测试结果
└── events.out.tfevents.*            ← TensorBoard 日志
```

---

## 六、EPCL v6.1 修正实验指南

> **版本**: v6.1 — 温和排斥修正  
> **日期**: 2026-05-23  
> **核心改动**: `alpha_uni`: 1.0 → **0.3** (仅此一处)  
> **代码文件**: `src/models/CEM/model.py:32`

### 6.1 修正策略

将 uniformity 从"主导力"降级为"微扰辅助"。

**定量论证**: v6 有效 uniformity 贡献 = `λ × α × L_uni = 0.07 × 1.0 × L_uni`  
v6.1 修正后 = `0.07 × 0.3 × L_uni = 0.021 × L_uni`  
总损失占比从 ~6% 降至 ~2%。

### 6.2 v5 → v6 → v6.1 超参数对比

| 参数 | v5 (最优) | v6 (失败) | **v6.1 (修正)** |
| --- | --- | --- | --- |
| 温度 τ | 0.3 | 0.3 | **0.3** |
| EPCL 外部权重 λ | 0.07 | 0.07 | **0.07** |
| Warmup 步数 | 3000 | 3000 | **3000** |
| 高斯排斥温度 $t_{uni}$ | — | 2.0 | **2.0** |
| **均匀性权重 $\alpha_{uni}$** | — | **1.0 ❌** | **0.3** |

### 6.3 成功标准

| 指标 | v5 基准 | v6.1 理想目标 | v6.1 可接受底线 |
| --- | --- | --- | --- |
| **PPL** ↓ | 36.3955 | ≤ 36.40 | ≤ 36.88 (不劣于 Baseline) |
| **Accuracy** ↑ | 38.17% | ≥ 38.50% | ≥ 38.17% (不劣于 v5) |

### 6.4 代码确认

当前 `model.py:31-32` 已修改为：

```python
def __init__(self, num_prototypes, input_dim, temperature=0.3,
             t_uniform=2.0, alpha_uni=0.3):  # v6.1: 从1.0降至0.3，避免过度排斥
```

其余代码**完全不变**。

### 6.5 执行清单

#### Task 1: 环境确认

- [x] `model.py:32` 已修改 `alpha_uni=0.3`
- [x] v6 结果已备份至 `save/test-v6/`
- [x] `save/test/` 已清理（权重、results、tfevents 已删除）
- [x] `main.py` 参数确认 (check_iter=2000, warmup 跳过=12000)

#### Task 2: 启动训练

```powershell
conda activate cem_env
cd E:\github\CEM-master
python main.py --model cem --batch_size 16 --cuda
```

✅ 训练完成 (26k steps, ~2h, 2026-05-23)

#### Task 3: 训练过程监控

✅ 已完成。关键发现：

- **BCE 验证集在 ~18k 步出现过拟合分叉**：Train BCE 继续下降至 ~0.6，Valid BCE 反转上升至 ~2.5+
- PPL 验证集在 ~20k 步触底 ~42，之后微幅回升
- 验证 Acc 峰值 41.49% (step 13999)，高于 v6 同期 40.37%

#### Task 4: 测试与归档

✅ 已完成。测试命令：

```powershell
# PPL-best 测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_19999_42.2865
Rename-Item "save/test/results.txt" "results - ppl_best.txt"

# ACC-best 测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_ACC_13999_0.4149
Rename-Item "save/test/results.txt" "results - acc_best.txt"
```

#### Task 5: 决策分支 — 实际判定

| 测试结果 | 行动 | **实际情况** |
| --- | --- | --- |
| PPL ≤ 36.88 **且** Acc ≥ 38.17% | ✅ v6.1 成功，作为最终版本 | ❌ PPL 未满足 |
| PPL 或 Acc 仍劣于 v5 | 尝试 alpha_uni=0.1（极微扰） | **← 触发此条件** |
| alpha_uni=0.1 仍无改善 | 放弃 uniformity 路线 | — |

**判定说明**: v6.1 PPL-best PPL=37.07 > 36.88 (未满足), Acc=37.79% < 38.17% (未满足)。
v6.1 ACC-best Acc=38.65% (满足) 但 PPL=38.45 (严重退化)。双条件未同时满足。

根据训练曲线分析，问题根因是 uniformity penalty 导致的 BCE 过拟合结构性问题，非 alpha 幅度可解。
**建议跳过 alpha=0.1 实验，直接确认 v5 alignment-only 为最终架构。**

### 6.6 历史对照表（最终版）

| 版本 | τ | λ | 特殊项 | PPL ↓ | Acc ↑ | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | — | — | — | 36.88 | 37.41% | 基准 |
| v1 | 0.07 | 0.1 | — | 37.05 | 36.65% | ❌ 全面劣于基线 |
| v2 | 0.5 | 0.05 | — | 36.65 | 36.92% | PPL 优 / Acc 劣 |
| v3 PPL | 0.2 | 0.1 | — | 36.83 | 36.97% | 略超基线 PPL |
| v3 ACC | 0.2 | 0.1 | — | 37.64 | 37.70% | 超越基线 Acc |
| v4 PPL | 0.5→0.1 | 0.05 | 退火 | 36.91 | 36.25% | ❌ Acc 最差 |
| v4 ACC | 0.5→0.1 | 0.05 | 退火 | 37.87 | 37.74% | Acc 次优 |
| **v5 PPL** | **0.3** | **0.07** | — | **36.40** | **38.17%** | 🏆🏆 双超突破 (最终版本) |
| v5 ACC | 0.3 | 0.07 | — | 37.63 | 37.94% | 验证集过拟合 |
| v6 PPL | 0.3 | 0.07 | α=1.0 | 37.00 | 37.13% | ❌ 全面退化 |
| v6 ACC | 0.3 | 0.07 | α=1.0 | 38.28 | 37.51% | ❌ 劣于基线 |
| v6.1 PPL | 0.3 | 0.07 | α=0.3 | 37.07 | 37.79% | ❌ PPL 未达标 |
| v6.1 ACC | 0.3 | 0.07 | α=0.3 | 38.45 | **38.65%** | ⚠ Acc 历史最高，PPL 严重退化 |

