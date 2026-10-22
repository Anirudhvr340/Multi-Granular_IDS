import json
from collections import Counter

datasets = ['02_14', '02_15', '02_16', '02_21', '02_22']
total_win = 0
total_att = 0
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0

print("CROSS-FILE BENCHMARK SUMMARY TABLE:")
print(f"{'Dataset':10s} | {'Windows':8s} | {'Attacks':8s} | {'Recall':8s} | {'FPR':8s} | {'ALLOW':8s} | {'FLAG':8s} | {'BLOCK':8s}")
print("-" * 80)

for d in datasets:
    path = f"nids/validation_results/pipeline_report_optimized_{d}.json"
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    windows = rep["windows"]
    actions = [w["action"] for w in windows]
    c = Counter(actions)

    benign_w = [w for w in windows if w["ground_truth"] == "Benign"]
    attack_w = [w for w in windows if w["ground_truth"] != "Benign"]

    n_w = len(windows)
    n_a = len(attack_w)
    n_b = len(benign_w)

    tp = sum(1 for w in attack_w if w["action"] in ("FLAG", "BLOCK"))
    fp = sum(1 for w in benign_w if w["action"] in ("FLAG", "BLOCK"))
    tn = sum(1 for w in benign_w if w["action"] == "ALLOW")
    fn = sum(1 for w in attack_w if w["action"] == "ALLOW")

    rec = tp / n_a if n_a else 1.0
    fpr = fp / n_b if n_b else 0.0

    total_win += n_w
    total_att += n_a
    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn

    print(f"{d:10s} | {n_w:8d} | {n_a:8d} | {rec:8.2%} | {fpr:8.2%} | {c['ALLOW']:8d} | {c['FLAG']:8d} | {c['BLOCK']:8d}")

print("-" * 80)
agg_rec = total_tp / total_att
agg_fpr = total_fp / (total_fp + total_tn)
print(f"{'TOTAL':10s} | {total_win:8d} | {total_att:8d} | {agg_rec:8.2%} | {agg_fpr:8.2%}")
