# EPCL v6.4 Human Evaluation Design (人工主观评估方案)

## 1. 评估目标
现有的自动化指标（PPL, BLEU, ROUGE 等）无法完美捕捉生成回复在深层次人类共情感知上的表现。我们需要引入标准化的盲评（Blind A/B Testing），证明 v6.4 在真实对话中的回应质量优于 Baseline。

## 2. 评估维度 (Evaluation Metrics)
参考 EmpatheticDialogues 原作及后续顶级会议论文标准，设立以下三个维度进行 1-5 分制打分：

1. **Empathy (共情度)**: 回复是否敏锐地识别了上下文中的情绪，并做出了恰当的情感呼应或安抚？
   - 1分: 完全无动于衷，甚至情感极性相反。
   - 5分: 像专业心理咨询师或知心好友，既理解了痛点/爽点，又给出了正向的情感反馈。
2. **Relevance (上下文相关性)**: 回复是否与话题（Context）在语义层面上紧密相连？
   - 1分: 答非所问，完全脱离当前话题。
   - 5分: 精确承接了上下文的细枝末节（如主体、客体、特定事件）。
3. **Fluency (流畅度)**: 回复是否符合人类自然语言习惯？有无语法错误或机器生成的机械感？

## 3. 采样与实验设计
- **采样规模**: 从包含复杂情感（如 grateful, impressed, embarrassed, trusting）的测试集中随机采样 100-200 对回复（Baseline vs v6.4）。
- **众包/专家评测**: 招募至少 3 位独立评测员。
- **盲评机制**: 屏蔽模型身份（A/B 乱序），评测员仅阅读 Context 及其生成的回复 A 和 回复 B。

## 4. 统计与分析
- 计算三人打分的 Krippendorff's alpha 或 Fleiss' kappa 验证一致性。
- 进行配对 T 检验（Paired t-test），验证 v6.4 在 Empathy 维度上相对 Baseline 的提升是否具有统计显著性 (p < 0.05)。
