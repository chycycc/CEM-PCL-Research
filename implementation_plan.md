# 实施计划：为 CEM 模型引入“原型对比学习” (Prototype-Guided Contrastive Learning)

## 目标
通过集成 **原型对比学习 (PGCL)** 来增强现有的 CEM（常识感知共情响应生成）模型。

## 创新点与动机 (为什么要这么做？)
*   **问题**：原始的 CEM 模型使用简单的**交叉熵损失 (Cross-Entropy Loss)** 进行情感分类。这虽然能保证模型预测出“正确”的标签，但不能保证学习到的情感特征在空间分布上是**结构化**的（即：相似的情感不一定聚在一起，不同的情感不一定分得够开）。
*   **解决方案**：引入 **情感原型 (Emotion Prototypes)**（即为每种情感学习一个聚类中心）并使用 **对比损失 (Contrastive Loss)**。
*   **核心机制**：
    1.  **拉近 (Pull)**：强制当前样本的情感向量 (`emo_rep`) 尽可能靠近其对应的“情感原型”。
    2.  **推远 (Push)**：强制其尽可能远离其他不相关的情感原型。
*   **预期效果**：更好地处理长尾（罕见）情感，并生成更稳健、语义更丰富的情感特征，从而生成更具共情能力的回复。

## 修改建议
我们将只修改一个文件：`src/models/CEM/model.py`。

### 1. 添加 `PrototypeContrastiveLoss` (原型对比损失) 类
在文件顶部插入此类。该类负责管理可学习的原型并计算对比损失。

```python
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, num_prototypes, input_dim, temperature=0.07):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        # 初始化可学习的情感原型矩阵
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        # 归一化处理
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def forward(self, features, labels):
        # 对特征和原型都进行归一化
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        # 计算相似度矩阵
        logits = torch.matmul(features, prototypes.T) / self.temperature
        # 使用交叉熵计算损失，拉近样本与目标原型的距离
        loss = F.cross_entropy(logits, labels)
        return loss
```

### 2. 在 `__init__` 中初始化 Loss
在 `CEM` 类的初始化函数中，实例化这个新的损失函数。

```python
self.pcl_criterion = PrototypeContrastiveLoss(decoder_number, config.hidden_dim).to(config.device)
```

### 3. 更新 `forward` 函数的返回值
修改 `forward` 函数，使其将用于计算损失的情感特征向量 (`emo_rep`) 返回出来。

```python
# 修改前
# return src_mask, cog_ref_ctx, emo_logits

# 修改后
return src_mask, cog_ref_ctx, emo_logits, emo_rep
```

### 4. 更新 `train_one_batch` (训练循环)
计算新的对比损失，并将其加到总损失中。

```python
# 1. 接收新的返回值
src_mask, ctx_output, emo_logits, emo_rep = self.forward(batch)

# 2. 计算对比损失
if train:
    emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
    loss_pcl = self.pcl_criterion(emo_rep, emo_label)
else:
    loss_pcl = 0.0

# 3. 加权计入总损失 (推荐权重 lambda = 0.1)
loss = ... + (0.1 * loss_pcl)
```

### 5. 修补解码器 (Patch Decoders)
更新 `decoder_greedy` 和 `decoder_topk` 函数，以接收 `forward` 返回的额外值，防止程序报错。

```python
# 使用下划线 _ 忽略多出来的返回值
src_mask, ctx_output, _, _ = self.forward(batch)
```

## 验证计划
1.  **代码静态检查**：确认变量名、矩阵维度是否匹配。（已完成）
2.  **试运行 (Dry Run)**：执行一次快速训练命令 (`python main.py --model cem ...`)，确保没有运行错误。
