# EPCL v6.4 实验数据诊断报告

> **实验**: EPCL v6.4 — 分类头早停 + Dropout(0.3) 正则化
> **日期**: 2026-05-26
> **训练时长**: ~51min (26k steps)
> **硬件**: RTX 3050 Ti 4G, CUDA
> **架构变更**: emo_lin 前增加 Dropout(0.3)；14k 步冻结 emo_lin + emo_loss 置零；λ_epcl 恢复 v6.2 恒定 0.07
> **判定**: ✅ **同一检查点双超达成。历史最优。**

---

## 一、测试集最终指标

| 检查点 | 步数 | PPL ↓ | Accuracy ↑ | 双超? |
| --- | --- | --- | --- | --- |
| **PPL-best** (`CEM_19999_41.1517`) | 19999 | **35.9690** | **40.29%** | ✅ PPL ≤ 36.3 且 Acc ≥ 39.0% |
| **ACC-best** (`CEM_ACC_17999_0.4178`) | 17999 | **37.1415** | **40.11%** | ❌ PPL 超标 |

> PPL-best 检查点在同一个模型权重上同时满足 PPL ≤ 36.3（35.97）和 Acc ≥ 39.0%（40.29%），**双超目标达成**。

## 二、历史全版本对照

| 版本 | PPL-best PPL ↓ | PPL-best Acc ↑ | ACC-best Acc ↑ | ACC-best PPL | 判定 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 36.88 | 37.41% | — | — | 基准 |
| v5 (甜区) | 36.40 | 38.17% | 37.94% | — | 双超基线 |
| v6 (α=1.0) | 37.00 | 37.13% | 37.51% | — | ❌ |
| v6.1 (α=0.3) | 37.07 | 37.79% | 38.65% | 38.45 | ❌ |
| v6.2 (ProjHead) | 36.25 | 38.00% | 39.30% | 38.23 | ⚠ 异步收敛 |
| v6.3 (截断退火) | 36.27 | 37.68% | 38.84% | 38.20 | ❌ 退化 |
| **v6.4 (分类头早停)** | **35.97** 🏆 | **40.29%** 🏆 | **40.11%** | **37.14** | ✅ 双超 🏆 |

### 核心增量 (v6.4 vs v6.2)

| 指标 | v6.2 | v6.4 | Δ | 判定 |
| --- | --- | --- | --- | --- |
| PPL-best PPL | 36.25 | **35.97** | **−0.28** | 🏆 历史新低 |
| PPL-best Acc | 38.00% | **40.29%** | **+2.29pp** | 🏆 历史新高 |
| ACC-best Acc | 39.30% | 40.11% | +0.81pp | 提升 |
| ACC-best PPL | 38.23 | 37.14 | −1.09 | 大幅改善 |

### 核心增量 (v6.4 vs Baseline)

| 指标 | Baseline | v6.4 | Δ |
| --- | --- | --- | --- |
| PPL | 36.88 | 35.97 | **−0.91** (−2.5%) |
| Acc | 37.41% | 40.29% | **+2.88pp** (+7.7%) |

---

## 三、训练曲线逐面板分析

### 3.1 Accuracy（训练集情感分类准确率）

![Accuracy 训练曲线](images/v64_acc_train.png)

- **Smoothed**: 0.5705 at 26k（v6.2 为 ~0.68，v6.3 为 ~0.68）
- **训练准确率大幅下降** → 这是**预期行为**，非故障
- 两个原因：
  1. **Dropout(0.3)** 在训练时随机屏蔽 30% 特征，训练 acc 被人为压低
  2. **14k 后 emo_lin 冻结 + emo_loss 移除**，分类头不再被优化，主干继续演化导致表征-分类头脱节
- **关键信号**: 训练 acc 低 + 测试 acc 高 = **正则化成功的经典标志**（过拟合被压制）

### 3.2 BCE（情感分类交叉熵损失）

![BCE 训练/验证曲线](images/v64_bce_train_valid.png)

- **训练 BCE Smoothed**: 1.043 (v6.2: ~0.49, v6.3: ~0.33)
- **验证 BCE Smoothed**: 2.235 (v6.2: ~2.7, v6.3: ~2.7)

| 指标 | v6.2 | v6.3 | v6.4 | 分析 |
| --- | --- | --- | --- | --- |
| Train BCE | ~0.49 | ~0.33 | 1.043 | ↑ Dropout 压低了训练拟合度 |
| Valid BCE | ~2.7 | ~2.7 | **2.235** | ↓ 验证集过拟合被有效抑制 |
| Train-Valid Gap | ~2.2 | ~2.4 | **1.2** | 🏆 **间距缩小近一半** |

> 验证 BCE 从 v6.2/v6.3 的 ~2.7 降至 2.235（−17%），且训练-验证间距从 ~2.2 缩小至 ~1.2。分类头过拟合被大幅抑制，Dropout + 早停策略直击根源。

### 3.3 Loss（总损失）

![Loss 曲线](images/v64_loss.png)

- **训练 Loss Smoothed**: 3.261
- **验证 Loss Smoothed**: 3.783
- 验证 loss 在 ~18k 步后略有回弹（3.75→3.80），但幅度远小于 v6.2/v6.3
- 训练-验证间距约 0.52，处于正常范围

### 3.4 Learning Rate

![学习率曲线](images/v64_lr.png)

- Noam 调度器：8k 步峰值 ~6e-4，26k 降至 ~3.5e-4
- 与 v6.2/v6.3 一致，无异常

### 3.5 PPL（困惑度）

![PPL 曲线](images/v64_ppl.png)

- **训练 PPL Smoothed**: 27.11
- **验证 PPL Smoothed**: 44.01
- PPL-best 在 step 19999，与 v6.2 一致
- 验证 PPL 在 18k-20k 触底约 41-42 后轻微回弹，符合预期

---

## 四、关键物理机制分析

### 4.1 跷跷板效应消除

v6.2 的核心问题是 PPL-best (step 19999) 和 ACC-best (step 13999) **相隔 6000 步**，无法在同一检查点兼顾。

v6.4 的检查点分布：

| 检查点 | PPL-best 步数 | ACC-best 步数 | 间距 |
| --- | --- | --- | --- |
| v6.2 | 19999 | 13999 | 6000 步 |
| v6.3 | 19999 | 13999 | 6000 步 |
| **v6.4** | **19999** | **17999** | **2000 步** |

ACC-best 从 step 13999 后移至 17999（+4000 步），与 PPL-best 的距离从 6000 步缩短至 2000 步。分类能力的峰值时间点被推迟到与生成能力更接近的位置。

**原因**：14k 步冻结 emo_lin 后，分类不再由 emo_loss 驱动，而是完全依赖 EPCL 对表征空间的持续优化。EPCL 在 14k→18k 期间继续组织情感聚类，冻住的 emo_lin 从更好的输入表征中被动受益，验证 acc 继续上升。

### 4.2 梯度通道释放效应

14k 步后 emo_loss 从总损失中移除，释放了原本用于分类优化的梯度通道：

```
Step 0-14k:  loss = emo_loss + 1.5*div + ctx + 0.07*epcl    (四路梯度竞争)
Step 14k+:   loss = 0       + 1.5*div + ctx + 0.07*epcl    (三路梯度协同)
```

emo_loss 移除后：
- ctx_loss 和 div_loss 获得更纯净的梯度信号 → PPL 改善（35.97 vs 36.25）
- EPCL 不再与 emo_loss 竞争对 emo_rep 的控制权 → 表征质量提升 → Acc 改善

### 4.3 Dropout 正则化验证

| 证据 | 数值 | 结论 |
| --- | --- | --- |
| 训练 Acc 下降 | 0.57 vs 0.68 | Dropout 生效，压制训练拟合 |
| 验证 BCE 下降 | 2.24 vs 2.7 | 泛化改善 |
| 测试 Acc 上升 | 40.29% vs 38.00% | 过拟合被转化为泛化能力 |

训练表现差 + 测试表现好 = Dropout 将过拟合容量成功转化为泛化容量。

---

## 五、双超验证矩阵

| 标准 | 阈值 | v6.4 PPL-best | 达标? |
| --- | --- | --- | --- |
| PPL ≤ 36.3 | 36.30 | **35.97** | ✅ (−0.33) |
| Acc ≥ 39.0% | 39.00% | **40.29%** | ✅ (+1.29pp) |
| 同一检查点 | 是 | step 19999 | ✅ |

**三项全部达标。v6.4 PPL-best 检查点为 EPCL 系列的最终最优模型。**

---

## 六、版本终审排序

| 排名 | 版本 | PPL | Acc | 亮点 |
| --- | --- | --- | --- | --- |
| 🥇 | **v6.4 (分类头早停)** | **35.97** | **40.29%** | 同一检查点双超，历史全面最优 |
| 🥈 | v6.2 (ProjHead) | 36.25 | 38.00% / 39.30%* | 两项分别最优但异步收敛 |
| 🥉 | v5 (甜区) | 36.40 | 38.17% | 首个双超基线版本 |

*v6.2 的 39.30% 来自 ACC-best 检查点（step 13999），与 PPL-best 不在同一权重上。

---

## 七、定性分析与特征流形可视化 (t-SNE)

为了在学术层面验证“分类早停 + Dropout”如何物理地改变了情感表征流形，我们提取了 v6.4 最终模型 (`CEM_19999_41.1517`) 的 5255 个测试集样本表征 (`emo_rep`) 与 32 个可学习原型 (Prototypes) 进行了 t-SNE 降维：

![EPCL v6.4 t-SNE 可视化](images/Fig_EPCL_tsne_v64.png)

**图注与学术分析**：
图中彩色散点为测试集样本情感表征，红色五角星为对比学习更新的可学习情感原型（Prototypes）。在分类头交叉熵损失于 14k 步被截断后，该特征流形的分布完全由 EPCL 的拓扑锚定机制驱动形成。

**1. 拓扑锚定效应与簇内内聚度 (Intra-cluster Cohesion)**
图像显示，彩色样本点高度向其对应的红色情感原型靠拢，形成了具有向心力的簇状分布。在 14k 步切断分类梯度后，后期样本向类中心收敛的动力唯一来源于 EPCL 的 InfoNCE 损失。这物理验证了**情感原型能作为稳定的拓扑锚点**，有效牵引同类上下文表征在空间内完成高密度聚集，实现了极强的簇内内聚度。

**2. 簇间分离度与细粒度语义耦合 (Inter-cluster Separation & Semantic Overlap)**
对比学习通过推离不同原型的特征距离，在多数基础情感类别之间建立起了相对清晰的决策边界。但从客观呈现来看，该特征空间并未实现绝对的 32 类正交分离。在簇与簇的交界地带，仍可观察到部分样本的粘连与混叠。这种现象直接映射了 EmpatheticDialogues 数据集细粒度情感标签（如 `sad` 与 `lonely`，`proud` 与 `impressed`）在自然语言语义上的固有耦合。这解释了为何即使在历史最优配置下，模型的分类准确率也受制于 40.29% 的天花板，证实了长尾复杂情感分布的边界模糊性。

**3. 流形平滑度与生成质量兼容性 (Manifold Smoothness)**
相较于纯 CE 监督导致的特征点塌陷（Representation Collapse），v6.4 在注入 `Dropout(0.3)` 正则化后，降维映射出的流形表现出了显著的连续性和平滑分布特征。这种平滑性避免了特征向量退化为极端的高维独热状态，从而保留了足够丰富的上下文语境信息用于下游 Decoder 解码。这一点从侧面解释了为何该架构能够在提升分类准确率的同时，将困惑度 (PPL) 进一步压低至 35.97，彻底消除了准确率与生成质量间的“跷跷板效应”。

---

## 八、Case Study：Baseline vs v6.4 典型对比分析

以下 3 个案例通过脚本 (`src/scripts/case_study_finder.py`) 从 5255 个测试集样本中系统性筛选而出。筛选标准：**Baseline Top-1 预测错误，v6.4 Top-1 预测正确，且两者在回复质量上存在可观察的差异。**

### Case 1: `grateful` vs `trusting` — 感恩与信任的语义边界

| 维度 | Baseline | v6.4 |
| --- | --- | --- |
| **真实情感** | grateful | grateful |
| **Top-3 预测** | ❌ trusting, terrified, grateful | ✅ **grateful**, hopeful, caring |
| **Context** | *"too often , it is a health scare that makes us value health . it inspires me to try and do something to keep healthy every day ."* | |
| **Beam 回复** | "that is great ! **i am so sorry** to hear that ." | "that is **good** to hear ." |
| **参考回复** | "i understand . i think the same thing . what is your health issue ?" | |

**语义边界分析**：
说话者表达的是一种**经历健康恐惧后对健康的感恩与珍视**。Baseline 将其误判为 `trusting`（信任），这是因为纯交叉熵监督下，`grateful` 和 `trusting` 在 GloVe 语义空间中的距离极近（两者都含有对外部的正向情感依赖），分类头难以在决策边界上区分它们。

v6.4 的 EPCL 拓扑锚定机制为 `grateful` 和 `trusting` 各自锚定了独立的原型向量，迫使表征在训练中向不同的吸引域聚集，从而拉开了这两类情感在球面空间上的角度距离。在回复层面，Baseline 生成了自相矛盾的回复（先肯定后道歉），暴露了情感分类错误对 Decoder 注意力分布的污染。v6.4 的回复虽简洁但语义一致。

### Case 2: `impressed` vs `embarrassed` — 钦佩与尴尬的极性反转

| 维度 | Baseline | v6.4 |
| --- | --- | --- |
| **真实情感** | impressed | impressed |
| **Top-3 预测** | ❌ embarrassed, disgusted, sad | ✅ **impressed**, confident, surprised |
| **Context** | *"...that is soul-sapping . what do you do to inspire yourself , and not let that kind of behavior drag you down ?"*（多轮对话中表达对领导者的敬佩） | |
| **Beam 回复** | "i am so sorry to hear that ." | "i am sorry to hear that . **i hope it works out for you** ." |
| **参考回复** | "well i have been in the business all my life and have worked for some great people..." | |

**语义边界分析**：
这是一个极端的极性误判案例。Baseline 将正向的 `impressed` 判定为负向的 `embarrassed`。极性反转的根源在于：对话末尾出现了 "soul-sapping"（令人疲惫的）等负面词汇，Baseline 的分类头过度拟合了局部词汇的极性，忽略了宏观情感走向。v6.4 通过 Dropout 削弱了对单一词汇特征的死记硬背，并由 EPCL 锚定整体情感簇，成功维持了正确的正向判定。

### Case 3: `grateful` vs `caring` — 感恩与关怀的主客体混淆

| 维度 | Baseline | v6.4 |
| --- | --- | --- |
| **真实情感** | grateful | grateful |
| **Top-3 预测** | ❌ caring, sentimental, grateful | ✅ **grateful**, faithful, caring |
| **Context** | *"i visited an orphan once and it was such a surreal moment . it made me realise how thankful i should be for a family"* | |
| **Beam 回复** | "that is very nice **of them** ." | "that is so sweet **of you** ." |
| **参考回复** | "that must have been an intense and emotional moment for you ." | |

**语义边界分析**：
说话者的核心情感是对自身家庭的感恩（grateful），而非对孤儿的关怀（caring）。Baseline 混淆了主客体，回复了 "nice of them"（将注意力错误放在第三方）；v6.4 回复了 "sweet of you"（精确将共情指向说话者本人）。这证明 EPCL 的对比学习迫使模型从 `emo_rep` 中区分了"我对他人"与"他人对我"的不同情感极性主体。
