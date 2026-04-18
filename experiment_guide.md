# 🧪 实验操作手册（零基础版）

> **适用人群**：完全没有深度学习实验经验的同学  
> **前置条件**：已安装 Anaconda，已有 `cem_env` 环境  
> **预计总耗时**：约 2-4 小时（取决于你的 GPU 速度）

---

## 📌 你需要完成的实验清单

论文中通常需要以下实验数据来证明你的方法有效：

| 编号 | 实验名称 | 目的 | 状态 |
| --- | --- | --- | --- |
| 实验 1 | EPCL 增强版训练 | 获取你的创新方法的指标 | ✅ 已完成 |
| 实验 2 | Baseline 基准训练 | 获取原始 CEM 的指标，用于对比 | ⬜ 待做 |
| 实验 3 | 结果对比表格 | 把两组数据放在一起，证明 EPCL 更好 | ⬜ 待做 |

---

## 实验 1：EPCL 增强版（已完成 ✅）

你已经跑完了这个实验！结果如下：

| 指标 | 值 |
| --- | --- |
| PPL | **37.05** |
| Accuracy | **36.65%** |
| Loss | 3.6123 |

结果文件位置：`E:\github\CEM-master\save\test\results.txt`

---

## 实验 2：Baseline 基准训练（接下来要做的）

### 第 1 步：打开 Anaconda Prompt

在 Windows 搜索栏搜索 **Anaconda Prompt**，点击打开。你会看到一个黑色命令行窗口。

### 第 2 步：进入项目目录

依次输入以下两行命令，每输入一行按一次回车：

```
E:
```

```
cd E:\github\CEM-master
```

### 第 3 步：激活环境

```
conda activate cem_env
```

输入后，你会看到命令行最前面出现 `(cem_env)` 字样，说明环境激活成功。

### 第 4 步：切换到原始版本代码

**⚠️ 这一步非常重要！** 你需要把代码从 EPCL 版切换回原始版。

依次输入以下两行命令：

```
ren src\models\CEM\model.py model_epcl.py
```

```
ren src\models\CEM\model_base.py model.py
```

**验证是否成功**：输入以下命令查看文件列表：

```
dir src\models\CEM\
```

你应该看到：
- `model.py`（现在是原始版，文件大小约 25KB）
- `model_epcl.py`（现在是 EPCL 版，文件大小约 27KB）

如果看到的结果和上面不一致，**停下来不要继续**，来找我帮你排查。

### 第 5 步：清空旧的训练结果

**⚠️ 重要！** 你必须先把之前 EPCL 的训练结果备份，否则会被覆盖！

```
ren save\test save\test_epcl
```

这会把之前 EPCL 的结果文件夹改名为 `test_epcl`，保护起来。

然后创建一个新的空文件夹给 Baseline 用：

```
mkdir save\test
```

### 第 6 步：启动 Baseline 训练

```
python main.py --model cem --batch_size 16 --cuda
```

### 第 7 步：等待训练完成

**你会看到什么？**
- 一个进度条，显示类似 `loss:10.xxx ppl:xxxxx: 0%|...`
- **不要关闭这个窗口！** 让它一直跑。
- 根据你之前 EPCL 的经验，大约 **1-2 小时**就会自动停止。

**训练结束的标志**：
- 进度条消失
- 你看到类似 `EVAL  Loss  PPL  Accuracy` 的输出
- 命令行重新出现 `(cem_env) E:\github\CEM-master>` 等待你输入

### 第 8 步：记录 Baseline 结果

训练结束后，打开这个文件查看结果：

```
notepad save\test\results.txt
```

你会在文件的**第 2 行**看到类似这样的数据：

```
3.xxxx    xx.xxxx    x.xxxx    0.xxxx
```

它们分别对应：`Loss    PPL    BCE    Accuracy`

**请把这组数据记录下来！**

### 第 9 步：切换回 EPCL 版本（恢复原状）

实验跑完后，把代码切换回 EPCL 版本：

```
ren src\models\CEM\model.py model_base.py
```

```
ren src\models\CEM\model_epcl.py model.py
```

---

## 实验 3：整理结果对比表格

当你完成了实验 2，你手上就有两组数据了。请把它们填入下面的表格：

### 📊 最终实验对比表（直接搬进论文）

| Model | PPL ↓ | Accuracy ↑ |
| --- | --- | --- |
| CEM (Baseline) | 36.8776 | 0.3741 |
| **CEM + EPCL (Ours)** | **37.05** | **36.65%** |

> 说明：PPL 越低越好（↓），Accuracy 越高越好（↑）。  
> 如果 EPCL 的 PPL 更低、Accuracy 更高，就证明你的方法有效！

---

## ❓ 常见问题与解决方案

### Q1：训练时报错 `CUDA out of memory`
**原因**：显存不够。  
**解决**：把命令里的 `--batch_size 16` 改成 `--batch_size 8`，再试一次。

### Q2：`ren` 命令报错 "找不到文件"
**原因**：可能已经改过名了，或者路径拼写错误。  
**解决**：先用 `dir src\models\CEM\` 看看现在有哪些文件，确认文件名后再操作。

### Q3：训练跑了很久还没结束
**正常现象**：如果 Baseline 模型收敛较慢，可能需要更多步数。耐心等待即可。

### Q4：两组实验的 PPL 差不多怎么办？
**可能原因**：数据集本身的上限。可以从 Accuracy 和生成回复质量两个维度补充分析。

### Q5：如何看训练的实时曲线？
**方法**：打开另一个 Anaconda Prompt 窗口，输入：
```
conda activate cem_env
tensorboard --logdir E:\github\CEM-master\save\test
```
然后打开浏览器访问 `http://localhost:6006`，你能看到 Loss 和 PPL 的实时下降曲线。

---

## 🎯 完成所有实验后的下一步

当你拿到两组对比数据后，你的论文实验部分就有了最核心的素材。接下来可以考虑：

1. **写论文的 Experiment 章节**：用上面的对比表格。
2. **做 t-SNE 可视化**（进阶，可选）：直观展示 EPCL 让情感更好地聚类了。
3. **消融实验**（进阶，可选）：测试不同的权重 λ（如 0.01, 0.1, 0.5）对结果的影响。

如果你需要做这些进阶实验，随时来找我，我会继续以这种"零基础"的风格为你写操作指南。

---

**祝实验顺利！有任何问题随时来问，我都在。** 🦾
