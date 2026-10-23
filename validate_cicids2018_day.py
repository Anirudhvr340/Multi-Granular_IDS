#!/usr/bin/env python3
"""
=============================================================================
VALIDATE ANY DAY OF THE CICIDS2018 DATASET (WITH VS. WITHOUT QWEN 2.5)
=============================================================================
Allows evaluating the multi-granular NIDS on ANY specific day of the CICIDS2018
benchmark dataset stored in the cicids2018/ folder.

Available Days:
  • 02-14-2018 (FTP-BruteForce, SSH-Bruteforce)
  • 02-15-2018 (DoS-GoldenEye, DoS-Slowloris)
  • 02-16-2018 (DoS-SlowHTTPTest, DoS-Hulk)
  • 02-20-2018 (DDoS-LOIC-HTTP)
  • 02-21-2018 (DDoS-HOIC, DDoS-LOIC-UDP)
  • 02-22-2018 (Brute Force -Web, Brute Force -XSS, SQL Injection)
  • 02-23-2018 (Brute Force -Web, Brute Force -XSS, SQL Injection)
  • 02-28-2018 (Infiltration)
  • 03-01-2018 (Infiltration)
  • 03-02-2018 (Botnet - Ares)

Compares:
1. 4-Model Fused Probabilities (Deterministic baseline without LLM)
2. Hybrid 4-Models + Qwen 2.5 7B Instruct (Agentic resolution on ambiguous cases)
=============================================================================
"""

import os
import sys
import time
import json
import glob
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from nids.pipeline import NIDSPipeline


DAY_CATALOG = {
    "02-14": {
        "file": "cicids2018/02-14-2018.csv",
        "date": "2018-02-14",
        "description": "Brute Force Day: FTP-BruteForce, SSH-Bruteforce",
        "attacks": ["FTP-BruteForce", "SSH-Bruteforce"],
        "aliases": ["02-14", "14", "ftp", "ssh", "bruteforce"],
    },
    "02-15": {
        "file": "cicids2018/02-15-2018.csv",
        "date": "2018-02-15",
        "description": "DoS Day 1: DoS attacks-GoldenEye, DoS attacks-Slowloris",
        "attacks": ["DoS attacks-GoldenEye", "DoS attacks-Slowloris"],
        "aliases": ["02-15", "15", "goldeneye", "slowloris"],
    },
    "02-16": {
        "file": "cicids2018/02-16-2018.csv",
        "date": "2018-02-16",
        "description": "DoS Day 2: DoS attacks-SlowHTTPTest, DoS attacks-Hulk",
        "attacks": ["DoS attacks-SlowHTTPTest", "DoS attacks-Hulk"],
        "aliases": ["02-16", "16", "hulk", "slowhttptest", "dos"],
    },
    "02-20": {
        "file": "cicids2018/02-20-2018.csv",
        "date": "2018-02-20",
        "description": "DDoS Day 1: DDoS attacks-LOIC-HTTP",
        "attacks": ["DDoS attacks-LOIC-HTTP"],
        "aliases": ["02-20", "20", "loic-http"],
    },
    "02-21": {
        "file": "cicids2018/02-21-2018.csv",
        "date": "2018-02-21",
        "description": "DDoS Day 2: DDOS attack-HOIC, DDOS attack-LOIC-UDP",
        "attacks": ["DDOS attack-HOIC", "DDOS attack-LOIC-UDP"],
        "aliases": ["02-21", "21", "hoic", "ddos"],
    },
    "02-22": {
        "file": "cicids2018/02-22-2018.csv",
        "date": "2018-02-22",
        "description": "Web Attacks Day 1: Brute Force -Web, Brute Force -XSS, SQL Injection",
        "attacks": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
        "aliases": ["02-22", "22", "web", "sql", "xss", "sqli"],
    },
    "02-23": {
        "file": "cicids2018/02-23-2018.csv",
        "date": "2018-02-23",
        "description": "Web Attacks Day 2: Brute Force -Web, Brute Force -XSS, SQL Injection",
        "attacks": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
        "aliases": ["02-23", "23"],
    },
    "02-28": {
        "file": "cicids2018/02-28-2018.csv",
        "date": "2018-02-28",
        "description": "Infiltration Day 1: Infilteration",
        "attacks": ["Infilteration"],
        "aliases": ["02-28", "28", "infiltration"],
    },
    "03-01": {
        "file": "cicids2018/03-01-2018.csv",
        "date": "2018-03-01",
        "description": "Infiltration Day 2: Infilteration",
        "attacks": ["Infilteration"],
        "aliases": ["03-01", "01"],
    },
    "03-02": {
        "file": "cicids2018/03-02-2018.csv",
        "date": "2018-03-02",
        "description": "Botnet Day: Bot (Ares Botnet)",
        "attacks": ["Bot"],
        "aliases": ["03-02", "02", "bot", "ares"],
    },
}


def resolve_day_file(day_arg: str) -> tuple[str, dict]:
    """Resolve a day shorthand, filename, or attack alias to the exact CSV path."""
    arg_lower = day_arg.strip().lower()

    # Check direct filename
    if os.path.exists(day_arg):
        base = os.path.basename(day_arg).replace(".csv", "")
        for k, v in DAY_CATALOG.items():
            if k in base:
                return day_arg, v
        return day_arg, {
            "date": base,
            "description": f"Custom dataset file: {day_arg}",
            "attacks": ["Custom"],
        }

    # Match aliases in catalog
    for key, info in DAY_CATALOG.items():
        if arg_lower == key.lower():
            return info["file"], info
        if arg_lower in [a.lower() for a in info["aliases"]]:
            return info["file"], info
        if f"{key}-2018" in arg_lower or f"{key}.csv" in arg_lower:
            return info["file"], info

    # Try glob matching
    matches = glob.glob(f"cicids2018/*{day_arg}*.csv")
    if matches:
        return matches[0], DAY_CATALOG.get(os.path.basename(matches[0])[:5], {
            "date": day_arg, "description": f"Matched {matches[0]}", "attacks": []
        })

    return None, None


def list_available_days():
    print("=" * 80)
    print("CICIDS2018 AVAILABLE DAILY DATASETS (IN cicids2018/ FOLDER)")
    print("=" * 80)
    print(f"{'Day Shorthand':<14} | {'File Size':<10} | {'Attack Classes Present in that Day':<45}")
    print("-" * 80)
    for key, info in sorted(DAY_CATALOG.items()):
        fpath = info["file"]
        size_str = "Missing"
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_mb/1024:.2f} GB"
        attacks_str = ", ".join(info["attacks"])
        print(f"{key:<14} | {size_str:<10} | {attacks_str:<45}")
    print("=" * 80)
    print("Example commands:")
    print("  python validate_cicids2018_day.py --day 02-14          # Validates FTP/SSH Brute Force")
    print("  python validate_cicids2018_day.py --day 02-22          # Validates Web & SQL Injection")
    print("  python validate_cicids2018_day.py --day 02-21 --max-windows 150 # Validates DDoS (HOIC/LOIC)")
    print("  python validate_cicids2018_day.py --day 03-02          # Validates Botnet")


def run_evaluation(
    csv_path: str,
    max_windows: int = 100,
    window_seconds: int = 30,
    use_llm: bool = True,
    label: str = "with_llm",
) -> tuple[dict, list]:
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {label.upper()}")
    print(f"Dataset File:   {csv_path}")
    print(f"Max Windows:    {max_windows}")
    print(f"Agentic LLM:    {'ACTIVE (Qwen 2.5 7B Instruct on ambiguous outputs)' if use_llm else 'DISABLED (Fused 4-Model Probabilities + Deterministic Fast-Path Only)'}")
    print("=" * 80)

    pipeline = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=use_llm,
    )

    t0 = time.perf_counter()
    report = pipeline.run(
        input_csv=csv_path,
        window_seconds=window_seconds,
        report_path=f"reports/temp_day_{label}.json",
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
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
    parser = argparse.ArgumentParser(description="Run Validation Demo on Any Specific Day of CICIDS2018")
    parser.add_argument("--day", type=str, default="02-22", help="Day code (e.g. 02-14, 02-15, 02-16, 02-20, 02-21, 02-22, 02-28, 03-02)")
    parser.add_argument("--max-windows", type=int, default=100, help="Max 30s windows to validate in demo (default: 100)")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window length in seconds (default: 30)")
    parser.add_argument("--list-days", action="store_true", help="List all available daily datasets and their attacks")
    args = parser.parse_args()

    if args.list_days:
        list_available_days()
        return

    csv_path, day_info = resolve_day_file(args.day)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"[ERROR] Could not resolve day dataset for '{args.day}'.")
        print("Use --list-days to see all available days.")
        sys.exit(1)

    max_windows = args.max_windows
    window_seconds = args.window_seconds
    day_name = day_info.get("date", os.path.basename(csv_path))
    attacks_desc = day_info.get("description", "Unknown")

    print("=" * 90)
    print(f"CICIDS2018 DAILY VALIDATION DEMO: {day_name.upper()}")
    print(f"Target File:         {csv_path} ({os.path.getsize(csv_path) / (1024*1024):.1f} MB)")
    print(f"Day Description:     {attacks_desc}")
    print(f"Evaluation Mode:     4-Model Fused Probabilities vs. Hybrid Qwen 2.5 7B")
    print(f"Max Windows:         {max_windows}")
    print("=" * 90)

    # 1. Run WITHOUT LLM (4-Model Fused Probability Baseline)
    metrics_fused, _ = run_evaluation(
        csv_path=csv_path,
        max_windows=max_windows,
        window_seconds=window_seconds,
        use_llm=False,
        label="fused_probabilities_4models",
    )

    # 2. Run WITH Qwen 2.5 (Hybrid Integration)
    metrics_hybrid, _ = run_evaluation(
        csv_path=csv_path,
        max_windows=max_windows,
        window_seconds=window_seconds,
        use_llm=True,
        label="hybrid_with_qwen2.5",
    )

    # Side-by-side comparison table
    print("\n" + "=" * 94)
    print(f"CICIDS2018 [{day_name}]: 4-MODEL FUSED PROBABILITIES vs. HYBRID QWEN 2.5")
    print("=" * 94)

    headers = ["Metric / Dimension", "4 Models Fused Probabilities", "4 Models + Qwen 2.5 Hybrid", "Delta / Impact"]
    rows = [
        ["Dataset File", os.path.basename(csv_path), os.path.basename(csv_path), day_name],
        ["Total Windows Evaluated", f"{metrics_fused['total_windows']}", f"{metrics_hybrid['total_windows']}", "Identical traffic"],
        ["Total Attack Windows", f"{metrics_fused['total_attacks']}", f"{metrics_hybrid['total_attacks']}", "Ground truth"],
        ["Total Benign Windows", f"{metrics_fused['total_benign']}", f"{metrics_hybrid['total_benign']}", "Ground truth"],
        ["Attack Recall (Detection Rate)", f"{metrics_fused['recall']*100:.2f}%", f"{metrics_hybrid['recall']*100:.2f}%", f"{(metrics_hybrid['recall']-metrics_fused['recall'])*100:+.2f}%"],
        ["Detection Precision", f"{metrics_fused['precision']*100:.2f}%", f"{metrics_hybrid['precision']*100:.2f}%", f"{(metrics_hybrid['precision']-metrics_fused['precision'])*100:+.2f}%"],
        ["False Positive Rate (FPR)", f"{metrics_fused['fpr']*100:.2f}%", f"{metrics_hybrid['fpr']*100:.2f}%", f"{(metrics_hybrid['fpr']-metrics_fused['fpr'])*100:+.2f}%"],
        ["Detection F1-Score", f"{metrics_fused['f1']*100:.2f}%", f"{metrics_hybrid['f1']*100:.2f}%", f"{(metrics_hybrid['f1']-metrics_fused['f1'])*100:+.2f}%"],
        ["Mean Fused P(attack)", f"{metrics_fused['mean_fused_attack_prob']:.4f}", f"{metrics_hybrid['mean_fused_attack_prob']:.4f}", "Ensemble evidence"],
        ["Mean Model Agreement", f"{metrics_fused['mean_model_agreement']*100:.1f}%", f"{metrics_hybrid['mean_model_agreement']*100:.1f}%", "Ensemble consensus"],
        ["Action: ALLOW", f"{metrics_fused['action_distribution'].get('ALLOW', 0)}", f"{metrics_hybrid['action_distribution'].get('ALLOW', 0)}", f"{metrics_hybrid['action_distribution'].get('ALLOW', 0) - metrics_fused['action_distribution'].get('ALLOW', 0):+d}"],
        ["Action: FLAG", f"{metrics_fused['action_distribution'].get('FLAG', 0)}", f"{metrics_hybrid['action_distribution'].get('FLAG', 0)}", f"{metrics_hybrid['action_distribution'].get('FLAG', 0) - metrics_fused['action_distribution'].get('FLAG', 0):+d}"],
        ["Action: BLOCK", f"{metrics_fused['action_distribution'].get('BLOCK', 0)}", f"{metrics_hybrid['action_distribution'].get('BLOCK', 0)}", f"{metrics_hybrid['action_distribution'].get('BLOCK', 0) - metrics_fused['action_distribution'].get('BLOCK', 0):+d}"],
        ["Ambiguous Dispatched to Qwen", "0 (Bypassed)", f"{metrics_hybrid['stats'].get('confusing_cases_dispatched', 0)}", f"+{metrics_hybrid['stats'].get('confusing_cases_dispatched', 0)} calls"],
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

    # Per-class detection table
    print("\n" + "=" * 80)
    print(f"PER-CLASS DETECTION RATE FOR DAY [{day_name}]")
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
        "day": day_name,
        "file": csv_path,
        "description": attacks_desc,
        "metrics_fused_probabilities": metrics_fused,
        "metrics_hybrid_qwen2.5": metrics_hybrid,
        "comparison_table": rows,
    }
    report_path = f"reports/cicids2018_{day_name}_comparison.json"
    with open(report_path, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"\n[OK] Complete report saved to {report_path}")


if __name__ == "__main__":
    main()

