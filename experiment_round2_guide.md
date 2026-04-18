# 🔬 第二轮实验：改进版 EPCL 实验指南

> **创建时间**: 2026-04-18  
> **前置条件**: 已完成 Baseline 和初版 EPCL 的训练  
> **目标**: 用改进后的超参数重新训练 EPCL，验证修复效果

---

## 一、第一轮实验回顾与诊断总结

### 1.1 实验数据对比

| 模型 | PPL ↓ | Accuracy ↑ | 训练结果文件夹 |
| --- | --- | --- | --- |
| CEM Baseline（原版） | **36.8776** | **37.41%** | `save/test_4.13/` |
| CEM + EPCL v1（初版） | 37.0523 | 36.65% | `save/test_epcl/` |

**结论：初版 EPCL 不如 Baseline。** PPL 更高（生成质量更差），Accuracy 更低（情感分类更差）。

### 1.2 TensorBoard 图表诊断（根因分析）

通过对比两组 TensorBoard 曲线，我们定位了 **三个致命问题**：

#### 问题 1：PPL 初始大爆炸 💥
- **Baseline** 初始 PPL ≈ 130（正常范围）
- **EPCL v1** 初始 PPL ≈ **16,000**（极度异常！）
- **根因**：温度系数 τ=0.07 太小，导致对比损失在训练初期产生极大的梯度，直接撕裂了预训练的语义空间。模型后续 2 万步都在艰难"缝补"被破坏的空间。

#### 问题 2：BCE 验证集 Loss U 型反弹 📈
- **Baseline** 验证集 BCE Loss 在 12k 步后缓慢上扬（轻度过拟合）
- **EPCL v1** 验证集 BCE Loss 在 10k 步后**急剧飙升突破 3.0**（严重过拟合）
- **根因**：过于尖锐的对比目标（τ=0.07）迫使模型放弃泛化，转而死记硬背训练集。

#### 问题 3：虚假的高训练准确率 🎭
- 训练集 Accuracy 飙到 80%+，但测试集只有 36.65%
- **根因**：纯随机初始化的原型矩阵 + 过强的对比信号 = 特征坍缩

### 1.3 已实施的"外科手术"修改

我们在 `model_epcl.py` 中做了以下三处精确修改：

| 修改项 | 修改前（v1） | 修改后（v2） | 原理 |
| --- | --- | --- | --- |
| **温度系数 τ** | `0.07` | `0.5` | 放大温度，让对比惩罚更温和，不再撕裂语义空间 |
| **原型初始化** | `torch.randn` | `xavier_uniform_` | 给原型一个规范的初始分布，避免极端相似度偏差 |
| **λ 权重策略** | 固定 `0.1` | `0.05 * min(1.0, iter/2000)` | 前 2000 步缓慢上升，让主任务先稳住底盘 |

**代码已修改完毕，存放在 `src/models/CEM/model_epcl.py` 中。**

---

## 二、第二轮实验：零基础操作步骤

### 第 1 步：打开 Anaconda Prompt

在 Windows 搜索栏搜 **Anaconda Prompt**，点击打开。

### 第 2 步：进入项目目录并激活环境

```
E:
cd E:\github\CEM-master
conda activate cem_env
```

### 第 3 步：切换到改进版 EPCL 代码

当前 `model.py` 是 Baseline 版本，需要换成改进版 EPCL：

```
ren src\models\CEM\model.py model_base.py
ren src\models\CEM\model_epcl.py model.py
```

**验证**：

```
dir src\models\CEM\
```

你应该看到：
- `model.py` ≈ **27,089 字节**（改进版 EPCL，体积更大）✅
- `model_base.py` ≈ **25,841 字节**（原始 Baseline）✅

### 第 4 步：确认 `save/test/` 是空的

上次我们已经创建了空的 `save/test/` 文件夹。如果里面有东西，先清空：

```
dir save\test\
```

如果不是空的，把内容移走：
```
Move-Item save\test save\test_old
mkdir save\test
```

如果是空的，直接跳到第 5 步。

### 第 5 步：启动训练！

```
python main.py --model cem --batch_size 16 --cuda
```

### 第 6 步：观察训练过程（关键！）

训练启动后，**重点观察前 2000 步的 PPL 值**：

| 情况 | 判断 |
| --- | --- |
| 初始 PPL < 200 | ✅ 修复成功！语义空间没被破坏 |
| 初始 PPL = 200-1000 | ⚠️ 有改善但仍偏高，可继续观察 |
| 初始 PPL > 10000 | ❌ 修复失败，需要进一步降低 λ |

你也可以同时开另一个 Anaconda Prompt 窗口看 TensorBoard：
```
conda activate cem_env
tensorboard --logdir E:\github\CEM-master\save\test
```
然后打开浏览器访问 `http://localhost:6006`。

### 第 7 步：等待训练完成

预计 **1-2 小时**自动结束（Early Stopping）。

### 第 8 步：查看结果

```
notepad save\test\results.txt
```

第 2 行的数据格式：`Loss    PPL    BCE    Accuracy`

### 第 9 步：记录结果并恢复文件

把结果填入下表，然后恢复文件名：

```
ren src\models\CEM\model.py model_epcl.py
ren src\models\CEM\model_base.py model.py
```

---

## 三、预期结果对比表格

| 模型 | PPL ↓ | Accuracy ↑ | 备注 |
| --- | --- | --- | --- |
| CEM Baseline | 36.8776 | 37.41% | 原版，无修改 |
| CEM + EPCL v1 | 37.0523 | 36.65% | 初版，超参数过激 |
| **CEM + EPCL v2** | **（待填）** | **（待填）** | **改进版：τ=0.5, Xavier, Warm-up** |

### 成功标准

- ✅ **PPL < 36.88**（比 Baseline 更低）
- ✅ **Accuracy > 37.41%**（比 Baseline 更高）
- ✅ **TensorBoard 初始 PPL 不爆炸**（< 200）

---

## 四、如果 v2 仍然不如 Baseline 怎么办？

### 备选方案 A：进一步降低 λ
把 `model_epcl.py` 第 612 行的 `0.05` 改为 `0.02`：
```python
lambda_epcl = 0.02 * min(1.0, iter / 2000.0) if train else 0.02
```

### 备选方案 B：延长 Warm-up 期
把预热步数从 2000 改为 5000：
```python
lambda_epcl = 0.05 * min(1.0, iter / 5000.0) if train else 0.05
```

### 备选方案 C：冻结原型矩阵前 N 步
在 `train_one_batch` 中添加：
```python
if iter < 3000:
    self.epcl_criterion.prototypes.requires_grad = False
else:
    self.epcl_criterion.prototypes.requires_grad = True
```

---

**准备好了就开始跑吧！有任何问题随时来问。** 🚀
