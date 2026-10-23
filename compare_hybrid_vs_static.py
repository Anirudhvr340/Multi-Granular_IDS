#!/usr/bin/env python3
"""
=============================================================================
HYBRID IDS VALIDATION & COMPARISON: WITH VS. WITHOUT LOCAL LLM (QWEN 2.5 7B)
=============================================================================
Evaluates the integrated multi-granular NIDS on the converted CICIOT2023 dataset:
1. WITHOUT LLM (Deterministic Rules + Static Risk Heuristic Fallback)
2. WITH LOCAL LLM (Hybrid Integration: Deterministic Fast-Path + Qwen 2.5 7B Agentic AI on Confusing Outputs)
Compares precision, recall, F1, action distributions, rule vs agent activations, and latency.
=============================================================================
"""

import os
import sys
import time
import json
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
    print(f"Agentic LLM:    {'ACTIVE (Qwen 2.5 7B Instruct on confusing outputs)' if use_llm else 'DISABLED (Deterministic fallback only)'}")
    print("=" * 80)

    pipeline = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=use_llm,  # Enables Qwen 2.5 local reasoning
    )

    t0 = time.perf_counter()
    report = pipeline.run(
        input_csv=csv_path,
        window_seconds=window_seconds,
        report_path=f"reports/comparison_{label}.json",
        no_llm=(not use_llm),
        max_windows=max_windows,
    )
    total_elapsed = time.perf_counter() - t0

    windows = report.get("windows", [])

    # Performance calculation
    window_gt = []
    window_actions = []
    window_sources = []
    window_rules = []

    for w in windows:
        gt = w.get("ground_truth", "Benign")
        act = w.get("action") or w.get("level3", {}).get("action", "ALLOW")
        src = w.get("source", "unknown")
        rule = w.get("rule")

        window_gt.append(gt)
        window_actions.append(act)
        window_sources.append(src)
        window_rules.append(rule if rule else "Agentic_Reasoning")

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
        "action_distribution": action_counts,
        "source_distribution": source_counts,
        "elapsed_seconds": round(total_elapsed, 2),
        "ms_per_window": round((total_elapsed / max(1, len(windows))) * 1000, 2),
        "stats": report.get("gemini_stats", {}),
    }

    return metrics, windows


def main():
    csv_file = "ciciot2023/ciciot2023_converted_benchmark.csv"
    max_windows = 150

    print("=" * 80)
    print("STARTING SIDE-BY-SIDE BENCHMARK: HYBRID IDS (QWEN 2.5 7B) VS. WITHOUT LLM")
    print("=" * 80)

    # 1. Run WITHOUT LLM (Deterministic rules + static heuristic fallback)
    metrics_no_llm, windows_no_llm = run_evaluation(
        csv_path=csv_file,
        max_windows=max_windows,
        use_llm=False,
        label="without_llm",
    )

    # 2. Run WITH LOCAL LLM Qwen 2.5 7B (Hybrid: Rules + Agentic AI on Confusing Outputs)
    metrics_with_llm, windows_with_llm = run_evaluation(
        csv_path=csv_file,
        max_windows=max_windows,
        use_llm=True,
        label="with_qwen2.5_7b",
    )

    # Compile Comparison Table
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION: WITHOUT LLM vs. WITH LOCAL QWEN 2.5 (7B INSTRUCT)")
    print("=" * 80)

    headers = ["Metric", "Without LLM (Deterministic)", "With Qwen 2.5 7B (Hybrid Agent)", "Delta / Impact"]
    rows = [
        ["Total Windows Evaluated", f"{metrics_no_llm['total_windows']}", f"{metrics_with_llm['total_windows']}", "Identical windows"],
        ["Total Attack Windows", f"{metrics_no_llm['total_attacks']}", f"{metrics_with_llm['total_attacks']}", "—"],
        ["Total Benign Windows", f"{metrics_no_llm['total_benign']}", f"{metrics_with_llm['total_benign']}", "—"],
        ["Attack Recall (Detection Rate)", f"{metrics_no_llm['recall']*100:.2f}%", f"{metrics_with_llm['recall']*100:.2f}%", f"{(metrics_with_llm['recall']-metrics_no_llm['recall'])*100:+.2f}%"],
        ["Detection Precision", f"{metrics_no_llm['precision']*100:.2f}%", f"{metrics_with_llm['precision']*100:.2f}%", f"{(metrics_with_llm['precision']-metrics_no_llm['precision'])*100:+.2f}%"],
        ["False Positive Rate (FPR)", f"{metrics_no_llm['fpr']*100:.2f}%", f"{metrics_with_llm['fpr']*100:.2f}%", f"{(metrics_with_llm['fpr']-metrics_no_llm['fpr'])*100:+.2f}%"],
        ["Detection F1-Score", f"{metrics_no_llm['f1']*100:.2f}%", f"{metrics_with_llm['f1']*100:.2f}%", f"{(metrics_with_llm['f1']-metrics_no_llm['f1'])*100:+.2f}%"],
        ["Action: ALLOW", f"{metrics_no_llm['action_distribution'].get('ALLOW', 0)}", f"{metrics_with_llm['action_distribution'].get('ALLOW', 0)}", f"{metrics_with_llm['action_distribution'].get('ALLOW', 0) - metrics_no_llm['action_distribution'].get('ALLOW', 0):+d}"],
        ["Action: FLAG", f"{metrics_no_llm['action_distribution'].get('FLAG', 0)}", f"{metrics_with_llm['action_distribution'].get('FLAG', 0)}", f"{metrics_with_llm['action_distribution'].get('FLAG', 0) - metrics_no_llm['action_distribution'].get('FLAG', 0):+d}"],
        ["Action: BLOCK", f"{metrics_no_llm['action_distribution'].get('BLOCK', 0)}", f"{metrics_with_llm['action_distribution'].get('BLOCK', 0)}", f"{metrics_with_llm['action_distribution'].get('BLOCK', 0) - metrics_no_llm['action_distribution'].get('BLOCK', 0):+d}"],
        ["Confusing Dispatched to LLM", "0", f"{metrics_with_llm['stats'].get('confusing_cases_dispatched', 0)}", f"+{metrics_with_llm['stats'].get('confusing_cases_dispatched', 0)}"],
        ["Deterministic Handled", f"{metrics_no_llm['total_windows']}", f"{metrics_with_llm['stats'].get('deterministic_handled', 0)}", "Fast-path handled"],
        ["Elapsed Time (Total)", f"{metrics_no_llm['elapsed_seconds']:.2f}s", f"{metrics_with_llm['elapsed_seconds']:.2f}s", f"{metrics_with_llm['elapsed_seconds']-metrics_no_llm['elapsed_seconds']:+.2f}s"],
        ["Latency (per window)", f"{metrics_no_llm['ms_per_window']:.1f} ms", f"{metrics_with_llm['ms_per_window']:.1f} ms", "Real-time edge rate"],
    ]

    col_widths = [32, 28, 32, 20]
    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    print(sep)
    header_str = "|" + "|".join([f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]) + "|"
    print(header_str)
    print(sep)
    for r in rows:
        row_str = "|" + "|".join([f" {c:<{col_widths[i]}} " for i, c in enumerate(r)]) + "|"
        print(row_str)
    print(sep)

    # Save summary report
    comparison_summary = {
        "dataset": csv_file,
        "without_llm": metrics_no_llm,
        "with_qwen2.5_7b": metrics_with_llm,
        "comparison_table": rows,
    }
    with open("reports/qwen2.5_vs_static_comparison.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)

    print("\n[OK] Comparison saved to reports/qwen2.5_vs_static_comparison.json")


if __name__ == "__main__":
    main()

