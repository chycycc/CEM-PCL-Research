# EPCL v6.4 Attention Visualization & Interpretability (注意力与可解释性分析)

## 1. 探索动机
量化指标和人工评测说明了“模型变得更好”，但可解释性分析才能说明“为什么更好”。针对 Case Study 中提及的“主客体混淆”修正和局部负面词极性反转等现象，我们需要从内部注意力分布中寻找确凿证据。

## 2. 核心分析模块

### 2.1 Cross-Attention 热力图可视化
在 Decoder 生成每一阶段，提取 Transformer 的 Cross-Attention 权重矩阵，分析生成特定情绪反馈时，模型聚焦在了 Context 的哪些位置。
- **目标对比**: 在 `grateful` 语境下，Baseline 的注意力可能发散或错误聚焦在他人身上，而 v6.4 的注意力热力区应精确锁定在表达“我自身”感激之情的关键词（如 `i`, `my family`, `thankful`）上。

### 2.2 情感分类头梯度激活图 (CAM for NLP)
虽然我们截断了早期的梯度，但可以探索在最终推断时，哪些词汇对分类器的决策贡献最大。这有助于说明 Dropout 和 EPCL 如何成功抑制了模型对局部高频负面词汇（如 "soul-sapping"）的过度拟合。

## 3. 实现工具与脚本
- 开发 `src/scripts/attention_vis.py`。
- 修改 `model_epcl.py` 中 Decoder 返回值的逻辑，使其在 `evaluate` 模式下额外输出注意力权重张量。
- 使用 `matplotlib` 和 `seaborn` 将注意力分数与源 Context 的 token 序列进行二维热力图（Heatmap）映射。
