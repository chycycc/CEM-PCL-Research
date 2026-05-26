# EPCL v6.4 Paper Drafting Outline (学术论文框架草案)

## 1. Title Ideas (暂定论文标题)
- *Bridging Classification and Generation in Empathetic Dialogue via Prototype Contrastive Learning and Gradient Regularization*
- *Resolving the Seesaw Effect in Empathetic Responses with Early-Stopped Contrastive Prototypes*

## 2. Abstract (摘要核心点)
- **痛点 (The Problem)**: 现有的共情对话生成模型在联合训练分类器和生成器时，常陷入准确率与困惑度的“跷跷板效应”；且对近义情感存在严重的边界模糊。
- **创新 (The Method)**: 提出 EPCL（情感原型对比学习）加上基于 Dropout 和截断分类梯度的架构优化。
- **结果 (The Results)**: 在 EmpatheticDialogues 测试集上，打破了 PPL 和 Acc 的互斥瓶颈，在 PPL 达到 35.97 的同时，维持了 40.29% 的分类准确率（均为单模型历史最优水平）。

## 3. Structure (论文正文结构)

### 3.1 Introduction (引言)
- 引出同理心对话任务及其重要性。
- 指出多任务学习（分类+生成）中的梯度冲突和表征塌陷问题。
- 引入本文的核心思想：拓扑原型锚定与正则化解耦。

### 3.2 Methodology (方法论)
- **Problem Formulation**: 定义 Context、Emotion Label 和 Target Response。
- **EPCL Module**: 详细推导基于 InfoNCE 损失的原型牵引机制。
- **Architecture Tweaks**:
  - Dropout 正则化的引入及其在 NLP 流形平滑中的理论支撑。
  - 分类头 Early Stop 的启发式依据（从分类主导到生成主导的平滑过渡）。

### 3.3 Experiments (实验设计)
- **Dataset**: EmpatheticDialogues
- **Baselines**: 与原始 CEM、MIME、EmpDG 等进行系统对比。
- **Main Results**: PPL 和 Acc 的双维度超越（引用 v6.4 诊断报告的核心数据）。

### 3.4 Analysis & Ablation (消融与深入分析)
- **Ablation Studies**: 分析 w/o EPCL, w/o Dropout, w/o Early Stop 后的指标跌落（基于 `ablation_study_plan` 跑出的数据）。
- **Manifold Visualization (t-SNE)**: 插入 v6.4 最终生成的聚类散点图，分析簇内聚合和边界耦合情况。
- **Case Study**: 挑选 2-3 个代表性案例（如解决极性反转和主客体混淆），并辅以 Attention 热力图增强说服力。

### 3.5 Conclusion (结论)
- 总结贡献。
- 展望未来的方向（如如何彻底切分相似情感簇，迈入更复杂的语境感知）。
