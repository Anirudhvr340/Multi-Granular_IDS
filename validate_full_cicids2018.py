#!/usr/bin/env python3
"""
=============================================================================
VALIDATION DEMO ON FULL CICIDS2018 DATASET (cicids2018/cic.csv)
=============================================================================
Streams through the full 7 GB CICIDS2018 dataset in memory-efficient chunks,
aligns schema on-the-fly, forms 30-second temporal windows, and evaluates:
1. Fused Probability Baseline (All 4 Models + Evidence Fusion + Rule Engine)
2. Hybrid Integration (All 4 Models + Evidence Fusion + Local Qwen 2.5 7B)

Outputs live window-by-window telemetry, side-by-side comparison tables,
and saves the full report to reports/cicids2018_full_validation_demo.json.
=============================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

import step0_preprocessing as preprocessing
from nids.pipeline import NIDSPipeline
from prepare_cicids2018_sample import COLUMN_MAP, LABEL_MAP


def stream_windows_from_csv(
    csv_path: str,
    window_seconds: int = 30,
    chunk_size: int = 50_000,
    stride: int = 1,
    max_windows: int = 150,
):
    """
    Generator that streams cleaned, standardized 30-second windows directly
    from cicids2018/cic.csv without loading the entire 7 GB file into memory.
    """
    base_time = datetime(2018, 3, 1, 8, 0, 0)
    current_time = base_time
    window_buffer = defaultdict(list)
    completed_windows = 0
    chunk_idx = 0

    print(f"\n[Data Streamer] Streaming from {csv_path} (chunk_size={chunk_size:,}, stride={stride})...")

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk_idx += 1
        if stride > 1 and (chunk_idx % stride != 0):
            continue

        # Standardize column headers and labels
        chunk.rename(columns=COLUMN_MAP, inplace=True)
        chunk["Label"] = chunk["Label"].map(lambda x: LABEL_MAP.get(str(x).strip(), str(x).strip()))

        # Reconstruct realistic timestamps
        ts_list = []
        for dur in chunk["Flow Duration"]:
            dur_us = float(dur) if pd.notnull(dur) else 100000.0
            delta = max(0.01, min(1.0, dur_us / 1000000.0 if dur_us > 0 else 0.05))
            current_time += timedelta(seconds=delta)
            ts_list.append(current_time)

        chunk["Timestamp"] = ts_list
        chunk["window_start"] = chunk["Timestamp"].dt.floor(f"{window_seconds}s")

        # Group into windows
        for ws, group in chunk.groupby("window_start", sort=True):
            cleaned = preprocessing.clean_chunk(group)
            if cleaned.empty:
                continue

            yield ws, cleaned
            completed_windows += 1

            if max_windows > 0 and completed_windows >= max_windows:
                return


def main():
    parser = argparse.ArgumentParser(description="Full CICIDS2018 Dataset Validation Demo")
    parser.add_argument("--csv", type=str, default="cicids2018/cic.csv", help="Path to CICIDS2018 CSV")
    parser.add_argument("--max-windows", type=int, default=150, help="Max windows to evaluate (0 for unlimited)")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window size in seconds")
    parser.add_argument("--stride", type=int, default=1, help="Chunk stride to sample across the 7 GB file")
    args = parser.parse_args()

    csv_path = args.csv
    max_windows = args.max_windows
    window_seconds = args.window_seconds
    stride = args.stride

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    print("=" * 90)
    print("CICIDS2018 FULL DATASET VALIDATION DEMO")
    print(f"Target Dataset:      {csv_path} ({os.path.getsize(csv_path) / (1024**3):.2f} GB)")
    print(f"Evaluation Mode:     Side-by-Side: 4-Model Fused Probabilities vs. Hybrid Qwen 2.5 7B")
    print(f"Max Windows:         {'ALL (Whole Dataset)' if max_windows == 0 else f'{max_windows} Windows'}")
    print(f"Window Duration:     {window_seconds}s")
    print("=" * 90)

    # Initialize two pipelines with identical architectures
    # 1. Baseline Pipeline (Fused Probabilities only, no LLM)
    pipeline_fused = NIDSPipeline(attack_threshold=0.35, use_gemini=False)
    # 2. Hybrid Pipeline (Fused Probabilities + Qwen 2.5 on ambiguous cases)
    pipeline_hybrid = NIDSPipeline(attack_threshold=0.35, use_gemini=True)

    results_fused = []
    results_hybrid = []

    t_start = time.perf_counter()

    try:
        w_idx = 0
        stream = stream_windows_from_csv(
            csv_path=csv_path,
            window_seconds=window_seconds,
            stride=stride,
            max_windows=max_windows,
        )

        for window_start, window_df in stream:
            w_idx += 1
            window_id = f"W{w_idx:05d}_{window_start}"

            # 1. Process with Fused Probabilities Baseline (no LLM)
            res_fused = pipeline_fused._process_window(
                window_df=window_df,
                window_start=window_start,
                window_id=window_id,
                window_seconds=window_seconds,
                preprocessing=preprocessing,
                no_llm=True,
            )
            results_fused.append(res_fused)

            # 2. Process with Hybrid Qwen 2.5 Integration
            res_hybrid = pipeline_hybrid._process_window(
                window_df=window_df,
                window_start=window_start,
                window_id=window_id,
                window_seconds=window_seconds,
                preprocessing=preprocessing,
                no_llm=False,
            )
            results_hybrid.append(res_hybrid)

            # Live telemetry output
            gt = res_fused.get("ground_truth", "Unknown")
            fused_p = res_fused["level2"].get("fused_attack_probability", 0.0)
            act_fused = res_fused.get("action", "ALLOW")
            act_hybrid = res_hybrid.get("action", "ALLOW")
            src_hybrid = res_hybrid.get("source", "unknown")

            symbol = "[MATCH]" if (act_fused == act_hybrid) else "[QWEN IMPACT]"
            print(f"[{window_id}] GT: {gt:<22} | Fused P: {fused_p:.3f} | Fused Action: {act_fused:<5} | Hybrid Action: {act_hybrid:<5} ({src_hybrid}) {symbol}")

    except KeyboardInterrupt:
        print("\n[!] Execution paused by user. Generating report on processed windows...")

    total_time = time.perf_counter() - t_start

    # Compile Comparison Metrics
    def calc_metrics(res_list, label):
        gt_binary = [0 if "Benign" in str(r.get("ground_truth")) else 1 for r in res_list]
        pred_binary = [1 if r.get("action") in ("BLOCK", "FLAG") else 0 for r in res_list]

        tp = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 1 and p == 1)
        fp = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 0 and p == 1)
        fn = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 1 and p == 0)
        tn = sum(1 for g, p in zip(gt_binary, pred_binary) if g == 0 and p == 0)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        fused_probs = [r["level2"].get("fused_attack_probability", 0.0) for r in res_list]
        agreements = [r["level2"].get("model_agreement_score", 0.0) for r in res_list]
        actions = Counter([r.get("action") for r in res_list])
        sources = Counter([r.get("source") for r in res_list])

        per_class = defaultdict(lambda: {"windows": 0, "detected": 0})
        for r in res_list:
            c = r.get("ground_truth", "Unknown")
            per_class[c]["windows"] += 1
            if r.get("action") in ("FLAG", "BLOCK"):
                per_class[c]["detected"] += 1

        return {
            "label": label,
            "windows": len(res_list),
            "attacks": sum(gt_binary),
            "benign": len(gt_binary) - sum(gt_binary),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "fpr": round(fpr, 4),
            "f1": round(f1, 4),
            "mean_fused_prob": round(float(np.mean(fused_probs)), 4) if fused_probs else 0.0,
            "mean_agreement": round(float(np.mean(agreements)), 4) if agreements else 0.0,
            "actions": dict(actions),
            "sources": dict(sources),
            "per_class": {k: dict(v) for k, v in per_class.items()},
        }

    m_fused = calc_metrics(results_fused, "4-Model Fused Probabilities")
    m_hybrid = calc_metrics(results_hybrid, "Hybrid Qwen 2.5 7B")

    # Display Comparison Table
    print("\n" + "=" * 96)
    print("CICIDS2018 DATASET VALIDATION REPORT: FUSED PROBABILITIES vs. HYBRID QWEN 2.5")
    print("=" * 96)

    headers = ["Metric / Dimension", "4-Model Fused Probabilities", "Hybrid 4-Models + Qwen 2.5", "Delta / Impact"]
    rows = [
        ["Total Windows Evaluated", f"{m_fused['windows']}", f"{m_hybrid['windows']}", "Identical traffic"],
        ["Total Attack Windows", f"{m_fused['attacks']}", f"{m_hybrid['attacks']}", "Ground truth"],
        ["Total Benign Windows", f"{m_fused['benign']}", f"{m_hybrid['benign']}", "Ground truth"],
        ["Attack Recall (Detection Rate)", f"{m_fused['recall']*100:.2f}%", f"{m_hybrid['recall']*100:.2f}%", f"{(m_hybrid['recall']-m_fused['recall'])*100:+.2f}%"],
        ["Detection Precision", f"{m_fused['precision']*100:.2f}%", f"{m_hybrid['precision']*100:.2f}%", f"{(m_hybrid['precision']-m_fused['precision'])*100:+.2f}%"],
        ["False Positive Rate (FPR)", f"{m_fused['fpr']*100:.2f}%", f"{m_hybrid['fpr']*100:.2f}%", f"{(m_hybrid['fpr']-m_fused['fpr'])*100:+.2f}%"],
        ["Detection F1-Score", f"{m_fused['f1']*100:.2f}%", f"{m_hybrid['f1']*100:.2f}%", f"{(m_hybrid['f1']-m_fused['f1'])*100:+.2f}%"],
        ["Mean Fused P(attack)", f"{m_fused['mean_fused_prob']:.4f}", f"{m_hybrid['mean_fused_prob']:.4f}", "Ensemble probability"],
        ["Mean Model Agreement", f"{m_fused['mean_agreement']*100:.1f}%", f"{m_hybrid['mean_agreement']*100:.1f}%", "Ensemble consensus"],
        ["Action: ALLOW", f"{m_fused['actions'].get('ALLOW', 0)}", f"{m_hybrid['actions'].get('ALLOW', 0)}", f"{m_hybrid['actions'].get('ALLOW', 0) - m_fused['actions'].get('ALLOW', 0):+d}"],
        ["Action: FLAG", f"{m_fused['actions'].get('FLAG', 0)}", f"{m_hybrid['actions'].get('FLAG', 0)}", f"{m_hybrid['actions'].get('FLAG', 0) - m_fused['actions'].get('FLAG', 0):+d}"],
        ["Action: BLOCK", f"{m_fused['actions'].get('BLOCK', 0)}", f"{m_hybrid['actions'].get('BLOCK', 0)}", f"{m_hybrid['actions'].get('BLOCK', 0) - m_fused['actions'].get('BLOCK', 0):+d}"],
        ["Ambiguous Dispatched to Qwen", "0 (Bypassed)", f"{pipeline_hybrid.policy_agent.stats.get('confusing_cases_dispatched', 0)}", f"+{pipeline_hybrid.policy_agent.stats.get('confusing_cases_dispatched', 0)} calls"],
        ["Deterministic Fast-Path Handled", f"{m_fused['windows']}", f"{pipeline_hybrid.policy_agent.stats.get('deterministic_handled', 0)}", "0 ms LLM overhead"],
        ["Total Runtime", f"{total_time:.2f}s", f"{total_time:.2f}s", f"{(total_time / max(1, m_fused['windows']))*1000:.1f} ms/window"],
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

    # Per-class detection rate table
    print("\n" + "=" * 80)
    print("PER-ATTACK CLASS DETECTION RATE COMPARISON (CICIDS2018)")
    print("=" * 80)
    cls_names = sorted(list(set(list(m_fused["per_class"].keys()) + list(m_hybrid["per_class"].keys()))))
    cls_headers = ["Attack Class", "Windows", "Fused 4-Models Rate", "Hybrid Qwen 2.5 Rate", "Impact"]
    cls_col_widths = [26, 10, 22, 22, 14]
    cls_sep = "+" + "+".join(["-" * (w + 2) for w in cls_col_widths]) + "+"
    print(cls_sep)
    print("|" + "|".join([f" {h:<{cls_col_widths[i]}} " for i, h in enumerate(cls_headers)]) + "|")
    print(cls_sep)
    for c in cls_names:
        f_stat = m_fused["per_class"].get(c, {"windows": 0, "detected": 0})
        h_stat = m_hybrid["per_class"].get(c, {"windows": 0, "detected": 0})
        n_win = f_stat["windows"]
        f_rate = (f_stat["detected"] / n_win * 100) if n_win > 0 else 0.0
        h_rate = (h_stat["detected"] / n_win * 100) if n_win > 0 else 0.0
        delta = h_rate - f_rate
        row = [c, str(n_win), f"{f_rate:.1f}%", f"{h_rate:.1f}%", f"{delta:+.1f}%"]
        print("|" + "|".join([f" {col:<{cls_col_widths[i]}} " for i, col in enumerate(row)]) + "|")
    print(cls_sep)

    # Save summary report
    out_data = {
        "dataset": csv_path,
        "total_time_seconds": round(total_time, 2),
        "fused_metrics": m_fused,
        "hybrid_metrics": m_hybrid,
        "comparison_table": rows,
    }
    report_file = "reports/cicids2018_full_validation_demo.json"
    with open(report_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[OK] Validation report saved to {report_file}")


if __name__ == "__main__":
    main()
