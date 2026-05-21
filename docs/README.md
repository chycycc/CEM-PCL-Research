# CEM-EPCL 研究文档索引

> **项目**: CEM + Emotion Prototype Contrastive Learning (EPCL)  
> **状态**: 实验进行中 🔬

---

## 📁 目录结构

```
docs/
├── README.md                    ← 本文件（文档索引）
├── experiment_status_sync.md    ← 实验状态全面同步报告（架构/轮次/指标全面对比）
├── development_log.md           ← 完整研发日志（核心文档，850+ 行）
├── implementation_plan.md       ← EPCL 技术方案与代码改动规范
├── task.md                      ← 任务清单与进度追踪
├── experiment_guides/           ← 各轮实验操作指南
│   ├── round1_guide.md          ← v1: τ=0.07, λ=0.1, 无warmup（失败）
│   ├── round2_guide.md          ← v2: τ=0.5, λ→0.05, warmup 2000（PPL突破）
│   ├── round3_guide.md          ← v3: τ=0.2, λ→0.1, warmup 5000（双轨保存）
│   ├── round4_guide.md          ← v4: τ 0.5→0.1 动态退火（Acc微超）
│   └── round5_guide.md          ← v5: τ=0.3, λ→0.07, warmup 3000（🏆 双超突破）
└── figures/                     ← 可视化图表
    └── Fig_EPCL_tsne_v5.png     ← v5 情感隐空间 t-SNE 可视化
```

## 📊 当前最优结果 (v5 PPL-best)

| 指标 | EPCL v5 | Baseline | 提升 |
| --- | --- | --- | --- |
| **PPL** | 36.3955 | 36.8776 | ↓ 1.31% |
| **Accuracy** | 38.17% | 37.41% | ↑ 2.03% |

## 🔗 相关代码文件

| 文件 | 说明 |
| --- | --- |
| `src/models/CEM/model.py` | 当前激活版本（EPCL v5） |
| `src/models/CEM/model_base.py` | CEM 原版 Baseline |
| `src/models/CEM/model_epcl.py` | EPCL 增强版独立备份 |
| `src/scripts/evaluate.py` | 自动评估脚本（Distinct-1/2） |
| `src/scripts/tsne_vis.py` | t-SNE 可视化脚本 |
| `main.py` | 训练/测试入口 |

## 💾 实验产物目录

| 目录 | 内容 |
| --- | --- |
| `save/test_4.13/` | Baseline 训练结果 |
| `save/test_epcl/` | EPCL v1 训练结果 |
| `save/test_round2/` | EPCL v2 训练结果 |
| `save/test_round3/` | EPCL v3 训练结果 |
| `save/test-v4/` | EPCL v4 训练结果 |
| `save/test-v5/` | EPCL v5 训练结果（备份） |
| `save/test/` | EPCL v5 训练结果（当前） |
| `release_v5/` | v5 最终发布权重与结果 |
