# 任务：在 CEM 中集成原型对比学习 (Prototype-Guided Contrastive Learning)

- [x] 撰写创新点文档和实施计划 (已中文化) <!-- id: 0 -->
- [ ] 将代码更改应用到 `src/models/CEM/model.py` <!-- id: 1 -->
    - [ ] 添加 `PrototypeContrastiveLoss` 类 <!-- id: 2 -->
    - [ ] 在 `__init__` 中初始化 Loss <!-- id: 3 -->
    - [ ] 更新 `forward` 以返回特征向量 <!-- id: 4 -->
    - [ ] 更新 `train_one_batch` 以计算并添加 PGCL Loss <!-- id: 5 -->
    - [ ] 更新 `decoder_greedy` 和 `decoder_topk` 以处理新的返回值 <!-- id: 6 -->
- [ ] 验证代码稳定性 <!-- id: 7 -->
