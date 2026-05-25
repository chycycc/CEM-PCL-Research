# EPCL v6.3 实验指导: 异步截断退火 + 拓扑锚点

> **前置依赖**: v6.2 投影头架构 (Projection Head)
> **代码变更**: `model.py` 第 662-682 行 — λ_epcl 四阶段调度
> **目标**: 同一检查点 PPL ≤ 36.3 且 Accuracy ≥ 39.0%

---

## 一、问题诊断

v6.2 数据揭示的"异步收敛"现象:

| 时间点 | 事件 | 数值 |
| --- | --- | --- |
| ~14k 步 | 分类能力达峰，BCE 验证集开始上翘 | Acc=39.30% (ACC-best) |
| ~20k 步 | 语言生成达峰 | PPL=36.25 (PPL-best) |

分类任务收敛快、过拟合早；生成任务收敛慢、需要更多步数打磨。强行折中必毁其一。

## 二、架构方案

### 2.1 保留投影头 (v6.2 不变)

`Linear(300→128) → ReLU → Linear(128→300)`，继续作为梯度减震器保护主干。

### 2.2 λ_epcl 四阶段调度

```
λ_epcl
0.07 ┤──────────────────┐
     │   升温    甜区    │ 线性衰减
     │  /               │╲
     │ /                │  ╲
0.005┤                  │    ╲────────── 拓扑锚点
     │                  │
  0  ┼──┬──────┬────┬───┬──────────→ iter
     0  3k    12k  15k  26k
```

| 阶段 | iter | λ_epcl | 边界值 |
| --- | --- | --- | --- |
| 1. 升温 | 0 → 3k | `0.07 × (iter/3000)` | 0 → 0.07 |
| 2. 甜区 | 3k → 12k | `0.07` (恒定) | 0.07 |
| 3. 衰减 | 12k → 15k | 线性插值 → 锚点 | 0.07 → 0.005 |
| 4. 锚点 | 15k+ | `0.005` (恒定) | 0.005 |

### 2.3 验证集处理

验证集 `train=False` 时，λ_epcl 恒定 0.07 满载计算。确保 TensorBoard 曲线全程数值可比，无人为断层。

### 2.4 拓扑锚点物理解释

15k 步后 λ=0.005 的作用:
- **不足以引发 BCE 过拟合**: 仅为满载的 7.1%，经投影头衰减后对主干影响极微
- **足够锚定原型**: `self.prototypes` 的梯度**仅来源于 EPCL loss**，0.005 的权重确保原型持续微量更新，不被主网络的 PPL 梯度冲刷至漂移

### 2.5 审计修正记录

原方案阶段 3 使用 `0.07 × max(0.01, 1-(iter-12000)/3000)`，会导致:
- iter=14999: λ = 0.07 × 0.01 = **0.0007**
- iter=15000: λ = **0.005**
- 跳变 ×7.1，非单调

已修正为线性插值 `0.07×(1-p) + 0.005×p`，阶段 3→4 无缝对接。

---

## 三、代码变更清单

仅修改 `src/models/CEM/model.py` 中 `train_one_batch` 的 λ_epcl 计算（约第 662 行）。

**PrototypeContrastiveLoss 模块无改动**，沿用 v6.2 投影头 + α_uni=1.0。

---

## 四、执行清单

### Task 1: 环境准备

- [ ] 确认 `model.py` 已包含 v6.3 四阶段调度代码
- [ ] 备份 v6.2 结果: `Rename-Item save\test save\test-v62`
- [ ] 确认 `save\test\` 目录已清空

### Task 2: 启动训练

```powershell
conda activate cem_env
cd E:\github\CEM-master
python main.py --model cem --batch_size 16 --cuda
```

### Task 3: 训练过程监控

重点关注:

1. **Valid BCE 曲线** (核心指标):
   - v6.2 在 ~14k 步上翘 → v6.3 预期在 14k 后因 λ 衰减而**保持平稳或下降**
   - 如果仍然上翘，说明 0.005 锚点过强

2. **Valid PPL 曲线**:
   - 预期在 ~20k 步触底，且此检查点的 Acc 应显著高于 v6.2 PPL-best 的 38.00%
   - 因为分类边界在 14k 步被"冻结"而非被过度锐化

3. **λ_epcl 变化验证** (可在 TensorBoard 中观察 loss_epcl 的量级变化):
   - 12k 步前: loss_epcl 正常量级
   - 12k-15k: loss_epcl 贡献应逐步缩小
   - 15k 后: loss_epcl 对总 loss 的贡献极微 (~0.005 × 2.5 ≈ 0.0125)

### Task 4: 测试与归档

```powershell
# PPL-best 测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_[步数]_[PPL]
Rename-Item "save/test/results.txt" "results - ppl_best.txt"

# ACC-best 测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_ACC_13999_0.4088
Rename-Item "save/test/results.txt" "results - acc_best.txt"
```

### Task 5: 决策矩阵

| 测试结果 | 判定 | 行动 |
| --- | --- | --- |
| PPL-best: PPL ≤ 36.3 且 Acc ≥ 39.0% | 🏆 **双超成功** | v6.3 为最终架构，准备论文 |
| PPL-best: PPL ≤ 36.4 且 Acc ≥ 38.5% | ✅ 显著改善 | 超越 v5，可作为最终版本 |
| PPL 保持但 Acc 未超 v6.2 ACC-best | ⚠ 锚点不足 | 提高锚点至 0.01 |
| Valid BCE 仍在 14k 后上翘 | ❌ 截断不够早 | 将衰减窗口前移至 10k-13k |
| PPL 退化 > 37.0 | ❌ 异常 | 检查代码逻辑 |

---

## 五、历史对照表

| 版本 | 配置 | PPL-best PPL ↓ | PPL-best Acc ↑ | ACC-best Acc ↑ | 判定 |
| --- | --- | --- | --- | --- | --- |
| Baseline | — | 36.88 | 37.41% | — | 基准 |
| v5 | τ=0.3, λ=0.07 | 36.40 | 38.17% | 37.94% | 🏆 双超 |
| v6 | +uniformity α=1.0 | 37.00 | 37.13% | 37.51% | ❌ |
| v6.1 | +uniformity α=0.3 | 37.07 | 37.79% | 38.65% | ❌ |
| v6.2 | ProjHead + α=1.0 | 36.25 🏆 | 38.00% | 39.30% 🏆 | ⚠ 边缘 |
| **v6.3** | **+异步退火+锚点** | **≤36.3** | **≥39.0%** | — | 🎯 目标 |

