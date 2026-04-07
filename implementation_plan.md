# 实施计划：为 CEM 引入"情感原型对比学习" (EPCL)

> **最终版 (V3 - 防 Crash 优化版)**
> 备份文件：`src/models/CEM/model_epcl.py`
> 原始模型：`src/models/CEM/model.py` (保留用于对比)

## 创新点与动机

- **问题**：原始 CEM 用简单的交叉熵做情感分类，不能保证情感特征在空间上结构化（相似情感不一定聚在一起）
- **方案**：引入**情感原型 (Emotion Prototypes)** + **对比损失 (Contrastive Loss)**
  - **拉近 (Pull)**：样本的情感向量靠近其对应的类别“情感原型”
  - **推远 (Push)**：远离其他不相关的情感原型
- **效果**：更好地区分相似情感，提升在共情回复生成任务中的表现

## 修改清单 (已应用于 `model_epcl.py`)

### Mod 1：添加 `PrototypeContrastiveLoss` 类
- **操作**：定义基于原型的对比损失计算逻辑

### Mod 2：在 `__init__` 中初始化
- **操作**：初始化 `self.epcl_criterion`

### Mod 3：改造 `forward`（软开关）
- **策略**：增加 `need_rep=False` 参数，训练时返回情感特征向量 `emo_rep`

### Mod 4：改造 `train_one_batch`
- **操作**：计算 `loss_epcl` 并以 `lambda_epcl=0.1` 的权重计入总损失

---

## ⚠️ 注意事项
1. **模型并行对比**：你可以通过切换 `model.py` 和 `model_epcl.py` 来对比 Baseline 和创新实验的结果。
2. **术语统一**：在论文中请统一使用 **EPCL** (Emotion Prototype Contrastive Learning)。
