# EPCL v4 实验指南 — 动态温度退火 (Dynamic Temperature Annealing)

> **日期**: 2026-04-24  
> **目标**: 同时保住 PPL ≤ 36.85 **和** Accuracy ≥ 37.5%，消灭 v2/v3 的帕累托互斥  
> **核心改动**: 动态温度 τ (0.5→0.1) + 权重 λ warmup 至 0.05 + 5000 步同步窗口

---

## 一、方案核心逻辑 — 为什么是 v4？

### 1.1 前三版的诊断总结

| 版本 | τ (温度) | λ (权重) | PPL | Accuracy | 诊断 |
| --- | --- | --- | --- | --- | --- |
| Baseline | — | — | 36.88 | 37.41% | 对照组 |
| v1 | 0.07 固定 | 0.1 固定 | 37.05 | 36.65% | 猛药致死：梯度爆炸 + 特征坍缩 |
| v2 | 0.5 固定 | 0→0.05 | **36.65** | 36.92% | 安慰剂：PPL 赢了，Accuracy 输了 |
| v3 | 0.2 固定 | 0→0.1 | 36.83 / 37.64 | 36.97% / **37.70%** | PPL 和 Accuracy 峰值错位 6000 步 |

**v3 的致命矛盾**：PPL 极小值（Step 19999）和 Accuracy 极大值（Step 13999）错位了 6000 步。

### 1.2 v4 的"绝杀"方案

**核心洞察：静态超参数的潜力已经被榨干。需要让温度随训练动态变化。**

| 参数 | v3 (静态) | v4 (动态退火) | 科学依据 |
| --- | --- | --- | --- |
| **温度 τ** | 固定 0.2 | `max(0.1, 0.5 - 0.4×min(1, iter/5000))` | 前期高温保护语言底盘，后期低温锐利切分情感边界 |
| **最终权重 λ** | 0.1 | **0.05** | v3 已证明 0.1 过度干扰 PPL，0.05 足以维持分类能力 |
| **Warmup 步数** | 5000 | **5000** | 保持不变，与温度退火同步 |

### 1.3 "有效力"时间线推演

对比学习对主任务的实际干扰强度 = λ / τ

| 步数 | τ | λ | 有效力 | 模型状态 |
| --- | --- | --- | --- | --- |
| 1000 | 0.42 | 0.01 | **0.024** | 语言婴儿期，几乎零干扰 |
| 2500 | 0.30 | 0.025 | 0.083 | 语言学步期，轻微引导 |
| 5000 | **0.10** | **0.05** | **0.50** | 语言成熟，全力切分情感 |
| 10000+ | 0.10 | 0.05 | 0.50 | 稳态维持 |

### 1.4 隐式课程学习效应

v4 天然构建了一套"由粗到细"的情感识别课程：
- **Step 0~2500（τ≈0.5~0.3）**：高温期。模型仅学习粗粒度区分（正面 vs 负面情感）
- **Step 2500~5000（τ≈0.3→0.1）**：急速冷却期。强迫区分细粒度相似情感（sad vs disappointed vs depressed）
- **Step 5000+（τ=0.1）**：极低温定型期。精修情感簇的最终决策边界

**这不是在调参，这是在让模型按照"先学说话，再学读心"的认知发展规律来训练。**

---

## 二、风险评估与预警

### ⚠️ τ=0.1 距离 v1 的致命 τ=0.07 只有一步之遥

**三重保险**：
1. ✅ Xavier 初始化（已验证有效）
2. ✅ λ=0.05 只有 v1 的一半
3. ✅ 5000 步的 warmup 完全保护了初始阶段

**关键监控窗口**：Step 4000~6000（τ 从 0.18 急降至 0.10 的过渡期）

如果 PPL 在此窗口反弹至 70+，立即中断，将下界提高至 0.15：
```python
current_tau = max(0.15, 0.5 - 0.35 * min(1.0, iter / 5000.0))
```

---

## 三、实验步骤

### 步骤 1：确认代码版本

打开 `src/models/CEM/model.py`（当前已是 v4 版本），确认：

- **第 30 行**：`class PrototypeContrastiveLoss`，`__init__` 中**无** `temperature` 参数
- **第 38 行**：`def forward(self, features, labels, tau)` — 接收动态 `tau`
- **第 618-628 行**：
  ```python
  current_tau = max(0.1, 0.5 - 0.4 * min(1.0, iter / 5000.0))
  lambda_epcl = 0.05 * min(1.0, iter / 5000.0) if train else 0.05
  if train:
      loss_epcl = self.epcl_criterion(emo_rep, emo_label, current_tau)
  ```

### 步骤 2：清空上一轮训练产物

```powershell
# 备份 Round 3 结果
Rename-Item E:\github\CEM-master\save\test E:\github\CEM-master\save\test_round3

# 创建新的输出目录
New-Item -ItemType Directory -Path E:\github\CEM-master\save\test -Force
```

### 步骤 3：启动训练

```powershell
cd E:\github\CEM-master
conda activate cem_env
python main.py --model cem --batch_size 16 --cuda
```

### 步骤 4：实时监控

```powershell
tensorboard --logdir save/test
```

---

## 四、关键监控节点

| 步数 | 当前 τ | 当前 λ | 应观察到 | 危险信号 |
| --- | --- | --- | --- | --- |
| 0~2000 | 0.50~0.34 | 0~0.02 | PPL 快速降至 ~80 | PPL > 200 |
| 2000~4000 | 0.34~0.18 | 0.02~0.04 | PPL 降至 ~55 | — |
| **4000~6000** | **0.18~0.10** | **0.04~0.05** | **⚡ 关键窗口：PPL 应稳定在 45~55** | **PPL 反弹至 70+** |
| 6000~12000 | 0.10 | 0.05 | PPL 缓降至 ~40，Accuracy 稳步上升 | BCE 验证集严重 U 型翘尾 |
| 12000+ | 0.10 | 0.05 | 进入 Early Stopping 区 | — |

---

## 五、成功标准

| 指标 | 理想目标 | 可接受底线 | Baseline 参考 |
| --- | --- | --- | --- |
| **PPL** | ≤ 36.70 | ≤ 36.90 | 36.8776 |
| **Accuracy** | ≥ 37.50% | ≥ 37.30% | 37.41% |

---

## 六、双轨保存机制

**本轮训练依然启用双轨保存**，训练完成后 `save/test/` 下会有两类权重：

| 文件前缀 | 含义 |
| --- | --- |
| `CEM_步数_PPL` | PPL 最优权重 |
| `CEM_ACC_步数_准确率` | Accuracy 最优权重 |

**测试命令**：
```powershell
# PPL 最优权重
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_XXXXX_XX.XXXX

# Accuracy 最优权重
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_ACC_XXXXX_X.XXXX
```
