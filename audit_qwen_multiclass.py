import json
from collections import defaultdict, Counter

with open("reports/comparison_with_qwen2.5_7b.json") as f:
    d = json.load(f)

class_family_map = {
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "Bot": "Bot",
    "SSH-Bruteforce": "BruteForce",
    "Infilteration": "Infiltration",
    "SQL Injection": "WebAttack",
    "Brute Force -XSS": "BruteForce",
    "Benign": None,
}

total_by_gt = defaultdict(int)
correct_family_by_gt = defaultdict(int)
qwen_families_by_gt = defaultdict(Counter)
model_fams_by_gt = defaultdict(lambda: {"packet": Counter(), "flow": Counter(), "time_window": Counter(), "session": Counter()})
actions_by_gt = defaultdict(Counter)

for w in d["windows"]:
    gt = w.get("ground_truth", "Unknown")
    total_by_gt[gt] += 1
    act = w.get("action")
    actions_by_gt[gt][act] += 1
    
    # Qwen attack family
    l3 = w.get("level3", {})
    q_fam = l3.get("attack_family")
    qwen_families_by_gt[gt][str(q_fam)] += 1
    
    # Level 1 models
    l1 = w.get("level1", {})
    for m in ["packet", "flow", "time_window", "session"]:
        m_pred = l1.get(m, {}).get("prediction", "Unknown")
        model_fams_by_gt[gt][m][m_pred] += 1
        
    expected_fam = class_family_map.get(gt, "Unknown")
    if expected_fam is None:
        if q_fam is None or q_fam == "None" or act == "ALLOW":
            correct_family_by_gt[gt] += 1
    else:
        if q_fam == expected_fam:
            correct_family_by_gt[gt] += 1

print("=" * 115)
print(f"{'Ground Truth Class':<24} | {'Win':<4} | {'Expected Family':<16} | {'Qwen Assigned Families':<42} | {'Family Acc':<10}")
print("=" * 115)
for gt, count in sorted(total_by_gt.items(), key=lambda x: x[1], reverse=True):
    exp = class_family_map.get(gt, "Unknown")
    fams = dict(qwen_families_by_gt[gt])
    acc = (correct_family_by_gt[gt] / count * 100) if count > 0 else 0.0
    print(f"{gt:<24} | {count:<4} | {str(exp):<16} | {str(fams):<42} | {acc:.1f}%")
print("=" * 115)

print("\n=== LEVEL 1 MODEL TOP PREDICTIONS FOR SQL INJECTION & INFILTRATION ===")
for tgt in ["SQL Injection", "Infilteration", "SSH-Bruteforce"]:
    print(f"\n--- Ground Truth: {tgt} (Total windows: {total_by_gt[tgt]}) ---")
    for m in ["packet", "flow", "time_window", "session"]:
        top_preds = model_fams_by_gt[tgt][m].most_common(3)
        print(f"  {m:<12}: {dict(top_preds)}")

