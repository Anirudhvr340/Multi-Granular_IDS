"""
Validate the Integrated 4-Model NIDS Pipeline on realistic_synthetic_dataset.csv
Evaluating Detection Performance Across All 14 Classes / Attack Types.
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Add directories to path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from nids.pipeline import NIDSPipeline

CSV_FILE = HERE / "realistic_synthetic_dataset.csv"
OUT_DIR = HERE / "nids" / "validation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_14_attack_validation():
    print("=" * 80)
    print("VALIDATION OF INTEGRATED 4-MODEL PIPELINE ON REALISTIC SYNTHETIC DATASET")
    print(f"Dataset: {CSV_FILE.name}")
    print("=" * 80)

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Missing {CSV_FILE}")

    # Inspect dataset ground truth
    df = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"Total Flows: {len(df):,}")
    print("\nGround Truth Flow Class Distribution:")
    gt_flow_counts = df["Label"].value_counts()
    for lbl, count in gt_flow_counts.items():
        print(f"  - {lbl:<26}: {count:>5} flows ({count/len(df)*100:>5.2f}%)")

    # Initialize full 4-model pipeline
    pipeline = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=False,  # deterministic heuristic for reproducible benchmark
    )

    report_out = OUT_DIR / "realistic_synthetic_14_attacks_full_report.json"

    print("\n" + "=" * 80)
    print("Running Integrated Pipeline on All Windows...")
    print("=" * 80)

    t0 = time.perf_counter()
    report = pipeline.run(
        input_csv=str(CSV_FILE),
        window_seconds=30,
        report_path=str(report_out),
        no_llm=True,
    )
    total_time = time.perf_counter() - t0

    windows = report.get("windows", [])
    print(f"\nProcessing Complete in {total_time:.2f} seconds ({len(windows)} windows processed).")

    # Aggregate window performance across all classes
    window_gt = []
    window_actions = []
    window_rules = []
    window_sources = []
    window_fused_probs = []

    per_class_actions = defaultdict(lambda: Counter())
    per_class_probs = defaultdict(list)

    for w in windows:
        gt = w.get("ground_truth", "Benign")
        act = w.get("action") or w.get("level3", {}).get("action", "ALLOW")
        rule = w.get("level3", {}).get("rule_triggered")
        source = w.get("level3", {}).get("source", "heuristic")
        f_prob = w.get("level2", {}).get("fused_attack_probability", 0.0)

        window_gt.append(gt)
        window_actions.append(act)
        window_rules.append(rule if rule else "Dynamic_Policy_Agent")
        window_sources.append(source)
        window_fused_probs.append(f_prob)

        per_class_actions[gt][act] += 1
        per_class_probs[gt].append(f_prob)

    all_classes = sorted(per_class_actions.keys(), key=lambda c: (c == "Benign", c))

    # Binary performance overall
    gt_binary = [0 if "Benign" in str(g) else 1 for g in window_gt]
    pred_detected = [1 if a in ("BLOCK", "FLAG") else 0 for a in window_actions]
    pred_blocked = [1 if a == "BLOCK" else 0 for a in window_actions]

    tp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 0)

    total_attacks = sum(gt_binary)
    total_benign = len(gt_binary) - total_attacks

    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall) if (attack_precision + attack_recall) > 0 else 0.0

    print("\n" + "=" * 80)
    print("OVERALL DETECTION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total Windows Evaluated:      {len(windows):,}")
    print(f"Total Benign Windows:         {total_benign:,} ({total_benign/len(windows)*100:.1f}%)")
    print(f"Total Attack Windows:         {total_attacks:,} ({total_attacks/len(windows)*100:.1f}%)")
    print(f"Total Attacks Detected:       {tp:,} / {total_attacks:,}")
    print(f"Attack Recall (Detection Rt): {attack_recall*100:.2f}%")
    print(f"Detection Precision:          {attack_precision*100:.2f}%")
    print(f"False Positive Rate (FPR):    {fpr*100:.2f}%")
    print(f"Detection F1-Score:           {f1:.4f}")

    print("\n" + "=" * 80)
    print("OVERALL ACTION DISTRIBUTION")
    print("=" * 80)
    act_counts = Counter(window_actions)
    for act in ["ALLOW", "FLAG", "BLOCK"]:
        cnt = act_counts[act]
        pct = cnt / len(windows) * 100
        print(f"  - {act:<7}: {cnt:>5} windows ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("PERFORMANCE PER ATTACK TYPE (ALL 14 CLASSES)")
    print("=" * 80)
    print(f"{'Attack / Traffic Type':<26} | {'Total':>5} | {'ALLOW':>5} | {'FLAG':>5} | {'BLOCK':>5} | {'Detect%':>7} | {'Avg P(att)':>10}")
    print("-" * 80)

    per_class_summary = {}

    for cls in all_classes:
        total_cls = sum(per_class_actions[cls].values())
        allow_cnt = per_class_actions[cls]["ALLOW"]
        flag_cnt = per_class_actions[cls]["FLAG"]
        block_cnt = per_class_actions[cls]["BLOCK"]
        detected = flag_cnt + block_cnt
        det_rate = detected / total_cls * 100 if total_cls > 0 else 0.0
        avg_prob = float(np.mean(per_class_probs[cls])) if per_class_probs[cls] else 0.0

        per_class_summary[cls] = {
            "total_windows": total_cls,
            "allow": allow_cnt,
            "flag": flag_cnt,
            "block": block_cnt,
            "detection_rate_pct": round(det_rate, 2),
            "mean_attack_prob": round(avg_prob, 4),
        }

        print(f"{cls:<26} | {total_cls:>5} | {allow_cnt:>5} | {flag_cnt:>5} | {block_cnt:>5} | {det_rate:>6.1f}% | {avg_prob:>10.4f}")

    print("\n" + "=" * 80)
    print("DECISION ENGINE & RULE ACTIVATIONS")
    print("=" * 80)
    rule_counts = Counter(window_rules)
    for r_name, cnt in rule_counts.most_common():
        pct = cnt / len(windows) * 100
        print(f"  - {r_name:<26}: {cnt:>5} activations ({pct:>5.1f}%)")

    # Save summary json
    summary_out = OUT_DIR / "realistic_synthetic_14_attacks_summary.json"
    summary_data = {
        "dataset": CSV_FILE.name,
        "windows_evaluated": len(windows),
        "elapsed_seconds": round(total_time, 2),
        "overall_metrics": {
            "attack_recall": round(attack_recall, 4),
            "attack_precision": round(attack_precision, 4),
            "fpr": round(fpr, 4),
            "f1_score": round(f1, 4),
            "total_attacks": total_attacks,
            "attacks_detected": tp,
            "total_benign": total_benign,
            "benign_false_positives": fp,
        },
        "action_distribution": dict(act_counts),
        "rule_distribution": dict(rule_counts),
        "per_class_performance": per_class_summary,
    }

    summary_out.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Summary metrics written to: {summary_out}")
    print("=" * 80)


if __name__ == "__main__":
    run_14_attack_validation()

