"""
从 Baseline 和 v6.4 的预测日志中，找出 Baseline 预测错误但 v6.4 预测正确的案例，
并评估回复质量的差异。
"""
import re

def parse_results(filepath):
    """解析 results.txt，返回列表 [{emotion, pred_top1, context, beam, greedy, ref}, ...]"""
    cases = []
    current = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Emotion:'):
                current['emotion'] = line.split(':', 1)[1].strip()
            elif line.startswith('Pred Emotions:'):
                preds = [p.strip() for p in line.split(':', 1)[1].split(',')]
                current['pred_top1'] = preds[0] if preds else ''
                current['pred_top3'] = preds[:3]
            elif line.startswith('Context:'):
                current['context'] = line.split(':', 1)[1].strip()
            elif line.startswith('Beam:'):
                current['beam'] = line.split(':', 1)[1].strip()
            elif line.startswith('Greedy:'):
                current['greedy'] = line.split(':', 1)[1].strip()
            elif line.startswith('Ref:'):
                current['ref'] = line.split(':', 1)[1].strip()
            elif line.startswith('----'):
                if 'emotion' in current:
                    cases.append(current)
                current = {}
    return cases

# 解析两个日志
baseline = parse_results(r'save\test_4.13\results.txt')
v64 = parse_results(r'save\test\results - ppl_best.txt')

print(f"Baseline 样本数: {len(baseline)}")
print(f"v6.4 样本数: {len(v64)}")

# 找出 Baseline Top1 错误 且 v6.4 Top1 正确的案例
candidates = []
for i, (b, v) in enumerate(zip(baseline, v64)):
    if b['emotion'] != b['pred_top1'] and v['emotion'] == v['pred_top1']:
        candidates.append({
            'idx': i,
            'emotion': b['emotion'],
            'baseline_pred': b['pred_top1'],
            'baseline_top3': b.get('pred_top3', []),
            'v64_pred': v['pred_top1'],
            'v64_top3': v.get('pred_top3', []),
            'context': b['context'],
            'baseline_beam': b['beam'],
            'v64_beam': v['beam'],
            'ref': b['ref'],
            'baseline_greedy': b.get('greedy', ''),
            'v64_greedy': v.get('greedy', ''),
        })

print(f"\n找到 {len(candidates)} 个 Baseline 错误 → v6.4 正确 的案例")
