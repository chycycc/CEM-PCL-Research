# EPCL v6.4 实验指导：分类头早停 + Dropout 正则化

## 版本定位

在 v6.3（截断退火）策略失败后，根据根因分析直击 BCE 过拟合的真正源头——分类头 `emo_lin` 的直接 BCE loss。

## 核心变更

### 变更 1：分类头 Dropout 正则化
- 在 `emo_lin` 前插入 `nn.Dropout(0.3)`
- 位置：`forward()` 中 `emo_logits = self.emo_lin(self.emo_dropout(emo_rep))`
- 目的：全程抑制分类头过拟合，增加泛化能力

### 变更 2：分类头早停（14k 步冻结）
- `iter >= 14000` 后：
  - 冻结 `emo_lin.weight` 的 `requires_grad = False`
  - `emo_loss` 从总损失中移除（`effective_emo_loss = 0`）
- 依据：v6.2 数据显示 ACC-best 在 step 13999 达峰，14k 后分类能力开始退化

### 变更 3：λ_epcl 恢复 v6.2 恒定调度
- 回退为 3k warmup + 0.07 满载
- 依据：v6.3 截断退火已证实打击对比学习（友军），无益于 BCE 过拟合控制

## 损失函数时间线

```
Step 0-3k:    loss = emo_loss + 1.5*div + ctx + warmup(λ)*epcl
Step 3k-14k:  loss = emo_loss + 1.5*div + ctx + 0.07*epcl    ← 分类学习期
Step 14k+:    loss = 0       + 1.5*div + ctx + 0.07*epcl    ← 分类下课
```

## 理论依据

| 参考文献 | 相关技术 | 与本实验关联 |
| --- | --- | --- |
| Howard & Ruder (2018) ULMFiT | 渐进式解冻/冻结 | 分类头早停是"逆向渐进冻结" |
| Chen et al. (2020) SimCLR | 投影头隔离 + 下游冻结 | EPCL 投影头已实现空间隔离，v6.4 进一步冻结分类头 |
| Srivastava et al. (2014) Dropout | 随机失活正则化 | 分类头前 Dropout(0.3) 抑制过拟合 |
| GradNorm (Chen et al., 2018) | 自适应任务权重 | v6.4 用硬切断替代动态权重，更激进但更可控 |

## 预期效果

1. **BCE 验证曲线**：14k 步后应停止上翘（分类梯度完全切断）
2. **PPL**：不受影响或略有改善（emo_loss 置零后释放的梯度通道归生成路径）
3. **Acc**：PPL-best 的 Acc 预期从 38.00% 提升至 ≥38.5%（Dropout 改善泛化）

## 成功标准

**同一检查点** PPL ≤ 36.3 **且** Accuracy ≥ 39.0%

## 运行命令

```bash
# 1. 备份 v6.3 结果
cp -r save/test save/test-v63

# 2. 清理旧检查点（保留 config）
rm save/test/CEM_*

# 3. 启动训练
python main.py --model cem --batch_size 16 --cuda
```

## 关键观察点

| TensorBoard 面板 | 期望信号 |
| --- | --- |
| bce_bce_valid | 14k 步后**平稳或下降**（非上翘） |
| ppl_ppl_valid | 与 v6.2 持平或略优 |
| acc_train | 14k 步后增速放缓（冻结分类头后训练 acc 不再上涨） |

## 决策矩阵

| 场景 | 行动 |
| --- | --- |
| PPL ≤ 36.3 且 Acc ≥ 39.0% | ✅ 双超达成，v6.4 为最终版本 |
| BCE 14k 后仍上翘 | 🔍 检查 emo_loss 是否确实被置零（log 验证） |
| Acc 退化但 PPL 改善 | 将冻结点后移至 16k |
| PPL 退化 | 检查 Dropout 是否影响了 emo_rep → decoder 路径 |
