#!/usr/bin/env python3
"""
=============================================================================
MULTI-GRANULAR NIDS: DAY-DATASET VALIDATION RUNNER
=============================================================================
Runs the integrated 4-model (Packet + Flow + Time-Window + Session) pipeline
on ANY day dataset in CICIDS-2018 (e.g., 02-14-2018.csv, 02-15-2018.csv, etc.)
and outputs a comprehensive security evaluation matrix.

Usage Examples:
---------------
1. Fast validation on test subset (50 windows, no API key needed):
   python validate_day.py 02-14-2018_test_run.csv --max-windows 50

2. Full validation on any day (deterministic heuristic fallback):
   python validate_day.py 02-14-2018.csv --no-llm

3. Validation with Gemini 2.5 Flash active reasoning:
   python validate_day.py 02-15-2018.csv --api-key "AIzaSy..." --max-windows 100
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Ensure repository root and time_window are in path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from nids.pipeline import NIDSPipeline


def validate_day_dataset(
    csv_file: str,
    window_seconds: int = 30,
    max_windows: int | None = None,
    attack_threshold: float = 0.35,
    no_llm: bool = False,
    api_key: str | None = None,
    gemini_model: str = "gemini-1.5-flash",
    report_file: str | None = None,
):
    csv_path = Path(csv_file).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print("=" * 80)
    print("MULTI-GRANULAR NIDS — DAY DATASET VALIDATION")
    print(f"Target Dataset: {csv_path.name}")
    print(f"Window Size:    {window_seconds} seconds")
    print(f"Max Windows:    {max_windows if max_windows else 'ALL'}")
    print(f"Attack Cutoff:  {attack_threshold}")
    print(f"Gemini LLM:     {'Enabled' if (api_key and not no_llm) else 'Disabled (Deterministic Fallback)'}")
    print("=" * 80)

    # Inspect dataset ground truth distribution
    print("\n[*] Scanning dataset labels...")
    df_sample = pd.read_csv(csv_path, usecols=["Label"], low_memory=False)
    print(f"Total Flows in File: {len(df_sample):,}")
    print("Ground Truth Flow Counts:")
    for lbl, count in df_sample["Label"].value_counts().items():
        print(f"  - {lbl:<26}: {count:>8,} flows ({count/len(df_sample)*100:>5.2f}%)")

    # Output paths
    if not report_file:
        stem = csv_path.stem
        out_dir = HERE / "nids" / "validation_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_file = str(out_dir / f"validation_{stem}_report.json")

    # Initialize full 4-model pipeline
    pipeline = NIDSPipeline(
        attack_threshold=attack_threshold,
        use_gemini=(not no_llm and bool(api_key)),
        gemini_api_key=api_key,
        gemini_model=gemini_model,
    )

    print("\n" + "=" * 80)
    print("Executing Integrated 4-Model Pipeline...")
    print("=" * 80)

    t0 = time.perf_counter()
    report = pipeline.run(
        input_csv=str(csv_path),
        window_seconds=window_seconds,
        report_path=report_file,
        no_llm=(no_llm or not api_key),
        max_windows=max_windows,
    )
    total_time = time.perf_counter() - t0

    windows = report.get("windows", [])
    print(f"\n[+] Pipeline finished in {total_time:.2f}s ({len(windows)} windows processed, {total_time/max(1, len(windows))*1000:.1f} ms/window)")

    # Compute evaluation statistics
    window_gt = []
    window_actions = []
    window_rules = []
    window_probs = []
    per_class_actions = defaultdict(lambda: Counter())
    per_class_probs = defaultdict(list)

    for w in windows:
        gt = w.get("ground_truth", "Benign")
        act = w.get("action") or w.get("level3", {}).get("action", "ALLOW")
        rule = w.get("level3", {}).get("rule_triggered")
        f_prob = w.get("level2", {}).get("fused_attack_probability", 0.0)

        window_gt.append(gt)
        window_actions.append(act)
        window_rules.append(rule if rule else "Dynamic_Policy_Agent")
        window_probs.append(f_prob)

        per_class_actions[gt][act] += 1
        per_class_probs[gt].append(f_prob)

    all_classes = sorted(per_class_actions.keys(), key=lambda c: (c == "Benign", c))

    # Binary performance
    gt_binary = [0 if "Benign" in str(g) else 1 for g in window_gt]
    pred_detected = [1 if a in ("BLOCK", "FLAG") else 0 for a in window_actions]

    tp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_binary, pred_detected) if g == 0 and p == 0)

    total_attacks = sum(gt_binary)
    total_benign = len(gt_binary) - total_attacks

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"PERFORMANCE SUMMARY FOR: {csv_path.name}")
    print("=" * 80)
    print(f"Total Windows Evaluated:      {len(windows):,}")
    print(f"Total Benign Windows:         {total_benign:,} ({total_benign/len(windows)*100:.1f}%)")
    print(f"Total Attack Windows:         {total_attacks:,} ({total_attacks/len(windows)*100:.1f}%)")
    print(f"Attacks Detected (BLOCK+FLAG):{tp:,} / {total_attacks:,}")
    print(f"Attack Recall (Detection Rt): {recall * 100:.2f}%")
    print(f"Detection Precision:          {precision * 100:.2f}%")
    print(f"False Positive Rate (FPR):    {fpr * 100:.2f}%")
    print(f"Detection F1-Score:           {f1:.4f}")

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)
    act_counts = Counter(window_actions)
    for act in ["ALLOW", "FLAG", "BLOCK"]:
        cnt = act_counts[act]
        pct = cnt / len(windows) * 100
        print(f"  - {act:<7}: {cnt:>5} windows ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("PER-CLASS ACTION BREAKDOWN")
    print("=" * 80)
    print(f"{'Class Name':<26} | {'Total':>5} | {'ALLOW':>5} | {'FLAG':>5} | {'BLOCK':>5} | {'Detect%':>7} | {'Avg P(att)':>10}")
    print("-" * 80)

    for cls in all_classes:
        total_cls = sum(per_class_actions[cls].values())
        allow_cnt = per_class_actions[cls]["ALLOW"]
        flag_cnt = per_class_actions[cls]["FLAG"]
        block_cnt = per_class_actions[cls]["BLOCK"]
        det_rate = (flag_cnt + block_cnt) / total_cls * 100 if total_cls > 0 else 0.0
        avg_prob = float(np.mean(per_class_probs[cls])) if per_class_probs[cls] else 0.0
        print(f"{cls:<26} | {total_cls:>5} | {allow_cnt:>5} | {flag_cnt:>5} | {block_cnt:>5} | {det_rate:>6.1f}% | {avg_prob:>10.4f}")

    print("\n" + "=" * 80)
    print("RULE & DECISION ENGINE ACTIVATIONS")
    print("=" * 80)
    rule_counts = Counter(window_rules)
    for r_name, cnt in rule_counts.most_common():
        pct = cnt / len(windows) * 100
        print(f"  - {r_name:<26}: {cnt:>5} activations ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print(f"[+] Full JSON results saved to: {report_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run validation of the Multi-Granular NIDS on any day CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to CICIDS-2018 CSV dataset (e.g. 02-14-2018.csv)")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window duration in seconds (default: 30)")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to process (default: all)")
    parser.add_argument("--attack-threshold", type=float, default=0.35, help="Fused attack threshold (default: 0.35)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use deterministic policy fallback")
    parser.add_argument("--api-key", default=None, help="Gemini API Key (optional)")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--report", default=None, help="Path for JSON report output")
    args = parser.parse_args()

    validate_day_dataset(
        csv_file=args.input_csv,
        window_seconds=args.window_seconds,
        max_windows=args.max_windows,
        attack_threshold=args.attack_threshold,
        no_llm=args.no_llm,
        api_key=args.api_key,
        gemini_model=args.gemini_model,
        report_file=args.report,
    )


if __name__ == "__main__":
    main()

