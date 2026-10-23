import json
from collections import Counter, defaultdict
import numpy as np

with open('reports/ciciot2023_integrated_validation_report.json') as f:
    data = json.load(f)

windows = data.get('windows', [])
print(f"Total Windows: {len(windows):,}")

per_class_actions = defaultdict(Counter)
per_class_probs = defaultdict(list)
per_class_fused_family = defaultdict(Counter)

window_gt = []
window_actions = []

for w in windows:
    gt = w.get('ground_truth', 'Benign')
    act = w.get('action') or w.get('level3', {}).get('action', 'ALLOW')
    f_prob = w.get('level2', {}).get('fused_attack_probability', 0.0)
    fam = w.get('level2', {}).get('fused_attack_family', 'Benign')
    
    window_gt.append(gt)
    window_actions.append(act)
    per_class_actions[gt][act] += 1
    per_class_probs[gt].append(f_prob)
    per_class_fused_family[gt][fam] += 1

print(f"{'Class Name':<26} | {'Total':>5} | {'ALLOW':>5} | {'FLAG':>5} | {'BLOCK':>5} | {'Detect%':>7} | {'Avg P_att':>10}")
print('-' * 80)
for cls in sorted(per_class_actions.keys(), key=lambda c: (c == 'Benign', c)):
    total_cls = sum(per_class_actions[cls].values())
    allow_cnt = per_class_actions[cls]['ALLOW']
    flag_cnt = per_class_actions[cls]['FLAG']
    block_cnt = per_class_actions[cls]['BLOCK']
    det_rate = (flag_cnt + block_cnt) / total_cls * 100 if total_cls > 0 else 0.0
    avg_prob = float(np.mean(per_class_probs[cls])) if per_class_probs[cls] else 0.0
    print(f"{cls:<26} | {total_cls:>5} | {allow_cnt:>5} | {flag_cnt:>5} | {block_cnt:>5} | {det_rate:>6.1f}% | {avg_prob:>10.4f}")

# Binary metrics
gt_binary = [0 if "Benign" in str(g) else 1 for g in window_gt]
pred_detected = [1 if a in ("BLOCK", "FLAG") else 0 for a in window_actions]

tp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 1)
fp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 1)
fn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 0)
tn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 0)

recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n" + "=" * 80)
print("INTEGRATED PIPELINE BINARY METRICS:")
print(f"Total Attacks:     {sum(gt_binary)}")
print(f"Total Benign:      {len(gt_binary) - sum(gt_binary)}")
print(f"Attacks Detected:  {tp} / {sum(gt_binary)} ({recall*100:.2f}%)")
print(f"Precision:         {precision*100:.2f}%")
print(f"False Positive Rt: {fpr*100:.2f}% ({fp}/{len(gt_binary) - sum(gt_binary)})")
print(f"F1-Score:          {f1*100:.2f}%")

print("\n" + "=" * 80)
print("FUSED ATTACK FAMILY CLASSIFICATION ACCURACY:")
print("=" * 80)
for cls in sorted(per_class_fused_family.keys()):
    top_fams = per_class_fused_family[cls].most_common(3)
    fam_str = ", ".join([f"{f}: {c}" for f, c in top_fams])
    print(f"{cls:<26} -> {fam_str}")

