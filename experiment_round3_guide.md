# EPCL v3 实验指南 (Round 3)

> **日期**: 2026-04-24  
> **目标**: 在保持 PPL 不恶化的前提下，**将 Accuracy 拉升至 ≥ 37.5%**，超越 Baseline  
> **核心改动**: 温度 τ=0.2，权重 λ warmup 至 0.1，预热 5000 步

---

## 一、本轮参数变化总览

| 参数 | v2 (Round 2) | v3 (本轮) | 改变原因 |
| --- | --- | --- | --- |
| 温度 τ | 0.5 | **0.2** | v2 太平滑，情感簇边界模糊 |
| 最终权重 λ | 0.05 | **0.1** | v2 太低，模型不优化对比目标 |
| Warmup 步数 | 2000 | **5000** | 权重翻倍，需更长保护期 |
| Xavier 初始化 | ✅ | ✅ | 已验证有效，保留 |

---

## 二、实验步骤

### 步骤 1：确认代码版本

打开 `src/models/CEM/model_epcl.py`，确认以下关键行：

- **第 31 行**：`temperature=0.2`
- **第 35 行**：`nn.init.xavier_uniform_(self.prototypes)`
- **第 611-612 行**：
  ```python
  # 引入 Warm-up: 前 5000 step 线性增加到 0.1
  lambda_epcl = 0.1 * min(1.0, iter / 5000.0) if train else 0.1
  ```

### 步骤 2：文件重命名（切换到 EPCL 版本）

```powershell
cd E:\github\CEM-master\src\models\CEM

# 备份 baseline
Rename-Item model.py model_base_bak.py

# 启用 EPCL v3
Copy-Item model_epcl.py model.py
```

### 步骤 3：清空上一轮训练产物

```powershell
# 备份 Round 2 结果（如果还没备份的话）
# Rename-Item E:\github\CEM-master\save\test E:\github\CEM-master\save\test_round2

# 创建新的输出目录
New-Item -ItemType Directory -Path E:\github\CEM-master\save\test -Force
```

### 步骤 4：启动训练

```powershell
cd E:\github\CEM-master
conda activate cem_env
python main.py --model cem --batch_size 16 --cuda
```

### 步骤 5：实时监控（另开一个终端）

```powershell
cd E:\github\CEM-master
conda activate cem_env
tensorboard --logdir save/test
```

---

## 三、关键监控节点

| 步数 | 应该观察到的现象 | 危险信号 |
| --- | --- | --- |
| 0 ~ 2000 | PPL 从高位快速下降至 ~80，对比损失权重缓慢爬升 | PPL > 500（说明 τ=0.2 仍然太激进） |
| 2000 ~ 5000 | PPL 继续下降至 ~50，λ 线性增长中 | PPL 出现反弹或震荡加剧 |
| 5000 ~ 8000 | **关键窗口**：λ 达到最终值 0.1，PPL 应稳定在 40~50 | PPL 在 λ 到达 0.1 后突然飙升 |
| 8000 ~ 12000 | PPL 缓慢下降至 ~40，Accuracy 应明显上升 | BCE 验证集开始剧烈 U 型翘尾 |
| 12000+ | 进入 Early Stopping 监控区 | — |

---

## 四、成功标准

| 指标 | 目标值 | Baseline 参考 |
| --- | --- | --- |
| **PPL** | ≤ 37.0 | 36.8776 |
| **Accuracy** | **≥ 37.5%** | 37.41% |

---

## 四点五、双轨保存机制说明

本轮训练启用了**双轨保存**，训练完成后 `save/test/` 目录下会出现两类权重文件：

| 文件前缀 | 含义 | 示例 |
| --- | --- | --- |
| `CEM_步数_PPL` | PPL 最优权重（Early Stopping 保存） | `CEM_19999_36.6540` |
| `CEM_ACC_步数_准确率` | Accuracy 最优权重（独立保存） | `CEM_ACC_17999_0.3850` |

**测试时**：分别用两个权重各跑一次 test，选数据最好看的组合写论文。

```powershell
# 用 PPL 最优权重测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_19999_36.6540

# 用 Accuracy 最优权重测试
python main.py --model cem --batch_size 16 --cuda --test --model_path save/test/CEM_ACC_17999_0.3850
```

---

## 五、应急预案

**如果 Accuracy 依然 < 37.4%**：
→ 说明静态温度方案到极限了，需要换成"动态退火"：
```python
# 替换 model_epcl.py 第 31 行的固定温度
# 改为在 train_one_batch 中动态计算温度：
# tau = max(0.1, 0.5 - 0.4 * min(1.0, iter / 5000.0))
# 即：前 5000 步温度从 0.5 平滑降至 0.1
```

**如果 PPL > 37.5**：
→ 说明 τ=0.2 + λ=0.1 组合仍然对语义空间有轻微干扰，可尝试将 Warmup 延长至 8000 步。
