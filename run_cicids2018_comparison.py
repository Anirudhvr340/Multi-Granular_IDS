#!/usr/bin/env python3
"""
=============================================================================
CICIDS2018 EVALUATION: FUSED PROBABILITY (4 MODELS) VS. HYBRID QWEN 2.5
=============================================================================
Runs side-by-side validation on the CICIDS2018 dataset:
1. Fused Probability Baseline (All 4 Models + Evidence Fusion + Rule Engine)
2. Hybrid Integration (4 Models + Evidence Fusion + Local Qwen 2.5 7B on Ambiguous Cases)

Outputs side-by-side comparison tables, per-class metrics, and saves:
reports/cicids2018_fused_vs_qwen_comparison.json
=============================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from nids.pipeline import NIDSPipeline


def run_evaluation(
    csv_path: str,
    max_windows: int = 150,
    window_seconds: int = 30,
    use_llm: bool = True,
    label: str = "with_llm",
) -> dict:
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {label.upper()}")
    print(f"Dataset:        {csv_path}")
    print(f"Max Windows:    {max_windows}")
    print(f"Agentic LLM:    {'ACTIVE (Qwen 2.5 7B Instruct on ambiguous outputs)' if use_llm else 'DISABLED (Fused Probability + Deterministic Fast-Path Only)'}")
    print("=" * 80)

    pipeline = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=use_llm,
    )

    t0 = time.perf_counter()
    report = pipeline.run(
        input_csv=csv_path,
        window_seconds=window_seconds,
        report_path=f"reports/cicids2018_{label}.json",
        no_llm=(not use_llm),
        max_windows=max_windows,
    )
    total_elapsed = time.perf_counter() - t0

    windows = report.get("windows", [])

    # Performance calculation
    window_gt = []
    window_actions = []
    window_sources = []
    fused_probs = []
    model_agreements = []
    per_class_stats = defaultdict(lambda: {"windows": 0, "detected": 0, "allowed": 0})

    for w in windows:
        gt = w.get("ground_truth", "Benign")
        act = w.get("action") or w.get("level3", {}).get("action", "ALLOW")
        src = w.get("source", "unknown")
        l2 = w.get("level2", {})
        fused_p = l2.get("fused_attack_probability", 0.0)
        agree_s = l2.get("model_agreement_score", 0.0)

        window_gt.append(gt)
        window_actions.append(act)
        window_sources.append(src)
        fused_probs.append(fused_p)
        model_agreements.append(agree_s)

        per_class_stats[gt]["windows"] += 1
        if act in ("FLAG", "BLOCK"):
            per_class_stats[gt]["detected"] += 1
        else:
            per_class_stats[gt]["allowed"] += 1

    # Binary metrics: Attack (FLAG/BLOCK) vs Benign (ALLOW)
    gt_binary = [0 if "Benign" in str(g) else 1 for g in window_gt]
    pred_binary = [1 if a in ("BLOCK", "FLAG") else 0 for a in window_actions]

    tp = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 0 and p == 0)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    action_counts = dict(Counter(window_actions))
    source_counts = dict(Counter(window_sources))

    metrics = {
        "label": label,
        "use_llm": use_llm,
        "total_windows": len(windows),
        "total_attacks": sum(gt_binary),
        "total_benign": len(gt_binary) - sum(gt_binary),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "mean_fused_attack_prob": round(float(np.mean(fused_probs)), 4) if fused_probs else 0.0,
        "mean_model_agreement": round(float(np.mean(model_agreements)), 4) if model_agreements else 0.0,
        "action_distribution": action_counts,
        "source_distribution": source_counts,
        "per_class": {k: dict(v) for k, v in per_class_stats.items()},
        "elapsed_seconds": round(total_elapsed, 2),
        "ms_per_window": round((total_elapsed / max(1, len(windows))) * 1000, 2),
        "stats": report.get("gemini_stats", {}),
    }

    return metrics, windows


def main():
    parser = argparse.ArgumentParser(description="Evaluate CICIDS2018 with Fused Probabilities vs. Qwen 2.5")
    parser.add_argument("--csv", type=str, default="cicids2018/cicids2018_benchmark.csv", help="Input CSV path")
    parser.add_argument("--max-windows", type=int, default=150, help="Max rolling windows to evaluate")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window length in seconds")
    args = parser.parse_args()

    csv_file = args.csv
    max_windows = args.max_windows
    window_seconds = args.window_seconds

    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV not found at {csv_file}!")
        print("Run 'python prepare_cicids2018_sample.py' first to generate the benchmark sample.")
        sys.exit(1)

    print("=" * 80)
    print("STARTING SIDE-BY-SIDE BENCHMARK ON CICIDS2018 DATASET")
    print(f"Dataset: {csv_file}")
    print(f"Max Windows: {max_windows}")
    print("=" * 80)

    # 1. Run WITHOUT LLM (Fused Probability of 4 models + Rule Engine + Heuristic Fallback)
    metrics_fused, _ = run_evaluation(
        csv_path=csv_file,
        max_windows=max_windows,
        window_seconds=window_seconds,
        use_llm=False,
        label="fused_probabilities_4models",
    )

    # 2. Run WITH LOCAL LLM Qwen 2.5 7B (Hybrid: 4 Models + Qwen 2.5 on Ambiguous Cases)
    metrics_hybrid, _ = run_evaluation(
        csv_path=csv_file,
        max_windows=max_windows,
        window_seconds=window_seconds,
        use_llm=True,
        label="hybrid_4models_with_qwen2.5",
    )

    # Side-by-side comparison table
    print("\n" + "=" * 92)
    print("COMPARATIVE EVALUATION: 4-MODEL FUSED PROBABILITIES vs. 4-MODEL + QWEN 2.5 HYBRID")
    print("=" * 92)

    headers = ["Metric / Dimension", "4 Models Fused Probabilities", "4 Models + Qwen 2.5 Hybrid", "Delta / Impact"]
    rows = [
        ["Total Windows Evaluated", f"{metrics_fused['total_windows']}", f"{metrics_hybrid['total_windows']}", "Identical traffic"],
        ["Total Attack Windows", f"{metrics_fused['total_attacks']}", f"{metrics_hybrid['total_attacks']}", "Ground truth"],
        ["Total Benign Windows", f"{metrics_fused['total_benign']}", f"{metrics_hybrid['total_benign']}", "Ground truth"],
        ["Attack Recall (Detection Rate)", f"{metrics_fused['recall']*100:.2f}%", f"{metrics_hybrid['recall']*100:.2f}%", f"{(metrics_hybrid['recall']-metrics_fused['recall'])*100:+.2f}%"],
        ["Detection Precision", f"{metrics_fused['precision']*100:.2f}%", f"{metrics_hybrid['precision']*100:.2f}%", f"{(metrics_hybrid['precision']-metrics_fused['precision'])*100:+.2f}%"],
        ["False Positive Rate (FPR)", f"{metrics_fused['fpr']*100:.2f}%", f"{metrics_hybrid['fpr']*100:.2f}%", f"{(metrics_hybrid['fpr']-metrics_fused['fpr'])*100:+.2f}%"],
        ["Detection F1-Score", f"{metrics_fused['f1']*100:.2f}%", f"{metrics_hybrid['f1']*100:.2f}%", f"{(metrics_hybrid['f1']-metrics_fused['f1'])*100:+.2f}%"],
        ["Mean Fused P(attack)", f"{metrics_fused['mean_fused_attack_prob']:.4f}", f"{metrics_hybrid['mean_fused_attack_prob']:.4f}", "Identical L2 evidence"],
        ["Mean Model Agreement", f"{metrics_fused['mean_model_agreement']*100:.1f}%", f"{metrics_hybrid['mean_model_agreement']*100:.1f}%", "Ensemble consensus"],
        ["Action: ALLOW", f"{metrics_fused['action_distribution'].get('ALLOW', 0)}", f"{metrics_hybrid['action_distribution'].get('ALLOW', 0)}", f"{metrics_hybrid['action_distribution'].get('ALLOW', 0) - metrics_fused['action_distribution'].get('ALLOW', 0):+d}"],
        ["Action: FLAG", f"{metrics_fused['action_distribution'].get('FLAG', 0)}", f"{metrics_hybrid['action_distribution'].get('FLAG', 0)}", f"{metrics_hybrid['action_distribution'].get('FLAG', 0) - metrics_fused['action_distribution'].get('FLAG', 0):+d}"],
        ["Action: BLOCK", f"{metrics_fused['action_distribution'].get('BLOCK', 0)}", f"{metrics_hybrid['action_distribution'].get('BLOCK', 0)}", f"{metrics_hybrid['action_distribution'].get('BLOCK', 0) - metrics_fused['action_distribution'].get('BLOCK', 0):+d}"],
        ["Ambiguous Dispatched to LLM", "0 (Bypassed)", f"{metrics_hybrid['stats'].get('confusing_cases_dispatched', 0)}", f"+{metrics_hybrid['stats'].get('confusing_cases_dispatched', 0)} calls"],
        ["Deterministic Fast-Path Handled", f"{metrics_fused['total_windows']}", f"{metrics_hybrid['stats'].get('deterministic_handled', 0)}", "0 ms LLM overhead"],
        ["Total Elapsed Time", f"{metrics_fused['elapsed_seconds']:.2f}s", f"{metrics_hybrid['elapsed_seconds']:.2f}s", f"{metrics_hybrid['elapsed_seconds']-metrics_fused['elapsed_seconds']:+.2f}s"],
        ["Latency (per window)", f"{metrics_fused['ms_per_window']:.1f} ms", f"{metrics_hybrid['ms_per_window']:.1f} ms", "Real-time edge rate"],
    ]

    col_widths = [32, 28, 28, 22]
    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    print(sep)
    header_str = "|" + "|".join([f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]) + "|"
    print(header_str)
    print(sep)
    for r in rows:
        row_str = "|" + "|".join([f" {c:<{col_widths[i]}} " for i, c in enumerate(r)]) + "|"
        print(row_str)
    print(sep)

    # Per-Class Comparison Table
    print("\n" + "=" * 80)
    print("PER-ATTACK CLASS DETECTION RATE: FUSED PROBABILITIES vs. HYBRID QWEN 2.5")
    print("=" * 80)
    all_classes = sorted(list(set(list(metrics_fused["per_class"].keys()) + list(metrics_hybrid["per_class"].keys()))))
    cls_headers = ["Attack Class", "Windows", "Fused 4-Models Rate", "Hybrid Qwen 2.5 Rate", "Impact"]
    cls_col_widths = [26, 10, 22, 22, 14]
    cls_sep = "+" + "+".join(["-" * (w + 2) for w in cls_col_widths]) + "+"
    print(cls_sep)
    print("|" + "|".join([f" {h:<{cls_col_widths[i]}} " for i, h in enumerate(cls_headers)]) + "|")
    print(cls_sep)
    for c in all_classes:
        f_stat = metrics_fused["per_class"].get(c, {"windows": 0, "detected": 0})
        h_stat = metrics_hybrid["per_class"].get(c, {"windows": 0, "detected": 0})
        n_win = f_stat["windows"]
        f_rate = (f_stat["detected"] / n_win * 100) if n_win > 0 else 0.0
        h_rate = (h_stat["detected"] / n_win * 100) if n_win > 0 else 0.0
        delta = h_rate - f_rate
        row = [c, str(n_win), f"{f_rate:.1f}%", f"{h_rate:.1f}%", f"{delta:+.1f}%"]
        print("|" + "|".join([f" {col:<{cls_col_widths[i]}} " for i, col in enumerate(row)]) + "|")
    print(cls_sep)

    # Save summary report
    out_report = {
        "dataset": csv_file,
        "metrics_fused_probabilities": metrics_fused,
        "metrics_hybrid_qwen2.5": metrics_hybrid,
        "comparison_table": rows,
    }
    report_path = "reports/cicids2018_fused_vs_qwen_comparison.json"
    with open(report_path, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"\n[OK] Complete report saved to {report_path}")


if __name__ == "__main__":
    main()

