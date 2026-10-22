#!/usr/bin/env python3
"""
Comprehensive 4-Model + Integrated Validation on Synthetic Dataset
===================================================================
1. Validates each of the 4 models (Flow XGBoost, Time-Window XGBoost,
   Packet 1D-CNN, Session CNN-LSTM) INDEPENDENTLY on realistic_synthetic_cicids2018_converted.csv.
2. Validates the FINAL INTEGRATED multi-granular agentic pipeline on the same dataset.
3. Outputs individual model metrics, sample predictions, and integrated action results.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

HERE = Path(__file__).resolve().parent
TIME_WINDOW_DIR = HERE / "time_window"
if str(TIME_WINDOW_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_WINDOW_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import step0_preprocessing as preprocessing
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.pipeline import NIDSPipeline

SYNTHETIC_CSV = HERE / "realistic_synthetic_cicids2018_converted.csv"
OUT_DIR = HERE / "nids" / "validation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_JSON_DEFAULT = lambda v: v.item() if isinstance(v, np.generic) else str(v)


def compute_metrics(ground_truth: list, predictions: list, attack_probs: list, threshold: float = 0.35) -> dict:
    gt_bin = [0 if "Benign" in str(x) else 1 for x in ground_truth]
    pred_bin = [1 if p >= threshold else 0 for p in attack_probs]

    acc = float(accuracy_score(ground_truth, predictions))
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(ground_truth, predictions, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(ground_truth, predictions, average="weighted", zero_division=0)

    tp = sum(1 for g, p in zip(gt_bin, pred_bin) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_bin, pred_bin) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_bin, pred_bin) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_bin, pred_bin) if g == 0 and p == 0)

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    cls_report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)

    return {
        "n_evaluated": len(ground_truth),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_mac), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "attack_recall": round(recall, 4),
        "attack_precision": round(precision, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "ground_truth_counts": dict(Counter(ground_truth)),
        "predicted_counts": dict(Counter(predictions)),
        "per_class": {k: v for k, v in cls_report.items() if isinstance(v, dict)},
    }


# ===========================================================================
# 1. INDEPENDENT TEST: FLOW MODEL
# ===========================================================================
def validate_flow_model(df: pd.DataFrame, n_samples: int = 1000) -> dict:
    print("\n" + "=" * 80)
    print("1. FLOW-LEVEL MODEL (XGBoost) — INDEPENDENT VALIDATION ON SYNTHETIC DATA")
    print("=" * 80)

    adapter = FlowModelAdapter()
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42).copy()

    ground_truth = sample_df["Label"].tolist()
    predictions = []
    confidences = []
    attack_probs = []
    latencies = []

    for i in range(len(sample_df)):
        row_df = sample_df.iloc[[i]]
        t0 = time.perf_counter()
        ev = adapter.predict(row_df, window_id=f"flow_{i}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = compute_metrics(ground_truth, predictions, attack_probs)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    print(f"[*] Samples Tested:    {len(sample_df)}")
    print(f"[*] Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"[*] Weighted F1-Score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"[*] Attack Recall:     {metrics['attack_recall'] * 100:.2f}%")
    print(f"[*] Attack Precision:  {metrics['attack_precision'] * 100:.2f}%")
    print(f"[*] False Positive Rt: {metrics['false_positive_rate'] * 100:.2f}%")
    print(f"[*] Mean Latency:      {metrics['mean_latency_ms']:.2f} ms/flow")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        if cls_name not in ("macro avg", "weighted avg"):
            print(f"  - {cls_name:<25} Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1-score']:.3f} | Support: {vals['support']}")

    return {"model": "Flow XGBoost", "granularity": "flow", "metrics": metrics}


# ===========================================================================
# 2. INDEPENDENT TEST: TIME-WINDOW MODEL
# ===========================================================================
def validate_time_window_model(df: pd.DataFrame, window_seconds: int = 30, max_windows: int = 150) -> dict:
    print("\n" + "=" * 80)
    print("2. TIME-WINDOW MODEL (268-feat XGBoost) — INDEPENDENT VALIDATION ON SYNTHETIC DATA")
    print("=" * 80)

    adapter = TimeWindowAdapter()
    adapter.reset_history()

    clean_df = preprocessing.clean_chunk(df.copy())
    clean_df["window_start"] = clean_df["Timestamp"].dt.floor(f"{window_seconds}s")

    ground_truth = []
    predictions = []
    confidences = []
    attack_probs = []
    latencies = []

    windows_grouped = list(clean_df.groupby("window_start", sort=True))[:max_windows]

    for w_idx, (window_start, window_df) in enumerate(windows_grouped):
        # Determine window ground truth
        lbls = window_df["Label"].tolist()
        non_benign = [l for l in lbls if "Benign" not in str(l)]
        w_gt = non_benign[0] if non_benign else "Benign"
        ground_truth.append(w_gt)

        aggregate = preprocessing.window_aggregate(window_df, window_start)
        t0 = time.perf_counter()
        ev = adapter.predict(aggregate, window_id=f"tw_{w_idx}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = compute_metrics(ground_truth, predictions, attack_probs)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    print(f"[*] Windows Tested:    {len(windows_grouped)}")
    print(f"[*] Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"[*] Weighted F1-Score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"[*] Attack Recall:     {metrics['attack_recall'] * 100:.2f}%")
    print(f"[*] Attack Precision:  {metrics['attack_precision'] * 100:.2f}%")
    print(f"[*] False Positive Rt: {metrics['false_positive_rate'] * 100:.2f}%")
    print(f"[*] Mean Latency:      {metrics['mean_latency_ms']:.2f} ms/window")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        if cls_name not in ("macro avg", "weighted avg"):
            print(f"  - {cls_name:<25} Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1-score']:.3f} | Support: {vals['support']}")

    return {"model": "Time-Window XGBoost", "granularity": "time_window", "metrics": metrics}


# ===========================================================================
# 3. INDEPENDENT TEST: PACKET MODEL
# ===========================================================================
def validate_packet_model(df: pd.DataFrame, n_samples: int = 1000) -> dict:
    print("\n" + "=" * 80)
    print("3. PACKET-LEVEL MODEL (1D-CNN) — INDEPENDENT VALIDATION ON SYNTHETIC DATA")
    print("=" * 80)

    adapter = PacketModelAdapter()
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42).copy()

    ground_truth = sample_df["Label"].tolist()
    predictions = []
    confidences = []
    attack_probs = []
    latencies = []

    for i in range(len(sample_df)):
        row_df = sample_df.iloc[[i]]
        t0 = time.perf_counter()
        ev = adapter.predict(row_df, window_id=f"pkt_{i}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = compute_metrics(ground_truth, predictions, attack_probs)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    print(f"[*] Packets/Flows:     {len(sample_df)}")
    print(f"[*] Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"[*] Weighted F1-Score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"[*] Attack Recall:     {metrics['attack_recall'] * 100:.2f}%")
    print(f"[*] Attack Precision:  {metrics['attack_precision'] * 100:.2f}%")
    print(f"[*] False Positive Rt: {metrics['false_positive_rate'] * 100:.2f}%")
    print(f"[*] Mean Latency:      {metrics['mean_latency_ms']:.2f} ms/packet")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        if cls_name not in ("macro avg", "weighted avg"):
            print(f"  - {cls_name:<25} Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1-score']:.3f} | Support: {vals['support']}")

    return {"model": "Packet 1D-CNN", "granularity": "packet", "metrics": metrics}


# ===========================================================================
# 4. INDEPENDENT TEST: SESSION MODEL
# ===========================================================================
def validate_session_model(df: pd.DataFrame, n_samples: int = 1500) -> dict:
    print("\n" + "=" * 80)
    print("4. SESSION-LEVEL MODEL (CNN-LSTM) — INDEPENDENT VALIDATION ON SYNTHETIC DATA")
    print("=" * 80)

    adapter = SessionModelAdapter()
    sample_df = df.iloc[:n_samples].copy()

    ground_truth = []
    predictions = []
    confidences = []
    attack_probs = []
    latencies = []

    chunk_size = 30
    for start_idx in range(0, len(sample_df), chunk_size):
        chunk_df = sample_df.iloc[start_idx:start_idx + chunk_size]
        lbls = chunk_df["Label"].values
        non_b = [l for l in lbls if "Benign" not in str(l)]
        gt_lbl = non_b[0] if non_b else "Benign"
        ground_truth.append(gt_lbl)

        t0 = time.perf_counter()
        ev = adapter.predict(chunk_df, window_id=f"sess_{start_idx}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = compute_metrics(ground_truth, predictions, attack_probs)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    print(f"[*] Sessions Tested:   {len(ground_truth)} (chunks of 30 flows)")
    print(f"[*] Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"[*] Weighted F1-Score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"[*] Attack Recall:     {metrics['attack_recall'] * 100:.2f}%")
    print(f"[*] Attack Precision:  {metrics['attack_precision'] * 100:.2f}%")
    print(f"[*] False Positive Rt: {metrics['false_positive_rate'] * 100:.2f}%")
    print(f"[*] Mean Latency:      {metrics['mean_latency_ms']:.2f} ms/session")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        if cls_name not in ("macro avg", "weighted avg"):
            print(f"  - {cls_name:<25} Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1-score']:.3f} | Support: {vals['support']}")

    return {"model": "Session CNN-LSTM", "granularity": "session", "metrics": metrics}


# ===========================================================================
# 5. INTEGRATED PIPELINE RUN & EVALUATION
# ===========================================================================
def validate_integrated_pipeline(csv_path: str, max_windows: int = 150) -> dict:
    print("\n" + "=" * 80)
    print("5. FINAL INTEGRATED 4-MODEL AGENTIC PIPELINE — VALIDATION RUN")
    print("=" * 80)

    pipe = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=False,  # deterministic heuristic fallback for reproducible benchmarking
    )

    report_path = OUT_DIR / "synthetic_validation_integrated_output.json"
    report = pipe.run(
        input_csv=csv_path,
        window_seconds=30,
        report_path=str(report_path),
        no_llm=True,
        max_windows=max_windows,
    )

    windows = report.get("windows", [])
    ground_truth = []
    actions = []
    fused_probs = []
    rule_counts = defaultdict(int)

    for w in windows:
        gt = w.get("ground_truth", "Benign")
        ground_truth.append(gt)

        # In pipeline output, window has 'action' at top level or in w['level3']['action']
        act = w.get("action") or w.get("level3", {}).get("action", "ALLOW")
        actions.append(act)

        f_prob = w.get("level2", {}).get("fused_attack_probability", 0.0)
        fused_probs.append(f_prob)

        rule = w.get("level3", {}).get("rule_triggered")
        source = w.get("level3", {}).get("source", "heuristic")
        if rule:
            rule_counts[rule] += 1
        elif source == "rule":
            rule_counts["Static_Rule_Engine"] += 1
        else:
            rule_counts["Dynamic_Policy_Agent"] += 1

    gt_bin = [0 if "Benign" in str(x) else 1 for x in ground_truth]
    # In 3-action space: BLOCK and FLAG are positive detections of an anomaly
    pred_detected = [1 if a in ("BLOCK", "FLAG") else 0 for a in actions]
    pred_blocked = [1 if a == "BLOCK" else 0 for a in actions]

    tp_det = sum(1 for g, p in zip(gt_bin, pred_detected) if g == 1 and p == 1)
    fp_det = sum(1 for g, p in zip(gt_bin, pred_detected) if g == 0 and p == 1)
    fn_det = sum(1 for g, p in zip(gt_bin, pred_detected) if g == 1 and p == 0)
    tn_det = sum(1 for g, p in zip(gt_bin, pred_detected) if g == 0 and p == 0)

    det_recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
    det_precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
    fpr = fp_det / (fp_det + tn_det) if (fp_det + tn_det) > 0 else 0.0

    print(f"\n[+] Integrated Windows Scored: {len(windows)}")
    print(f"[+] Total Attack Windows:      {sum(gt_bin)} / {len(windows)}")
    print(f"[+] Action Distribution:")
    action_dist = dict(Counter(actions))
    for act, cnt in action_dist.items():
        print(f"    - {act:<7}: {cnt:>5} ({cnt / len(windows) * 100:>5.1f}%)")

    print(f"\n[+] Detection Performance (FLAG + BLOCK):")
    print(f"    - Attack Detection Rate (Recall):    {det_recall * 100:.2f}%")
    print(f"    - Detection Precision:               {det_precision * 100:.2f}%")
    print(f"    - False Positive Rate (FPR):         {fpr * 100:.2f}%")

    print(f"\n[+] Action Breakdown by Ground Truth:")
    for gt_label in sorted(set(ground_truth)):
        acts_for_lbl = [a for g, a in zip(ground_truth, actions) if g == gt_label]
        c = Counter(acts_for_lbl)
        print(f"    - {gt_label:<25} Total: {len(acts_for_lbl):>3} | ALLOW: {c['ALLOW']:>3} | FLAG: {c['FLAG']:>3} | BLOCK: {c['BLOCK']:>3}")

    print(f"\n[+] Decision Engine Activations:")
    for r_name, cnt in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {r_name:<25}: {cnt:>4} activations ({cnt / len(windows) * 100:.1f}%)")

    return {
        "n_windows": len(windows),
        "action_distribution": action_dist,
        "detection_recall": round(det_recall, 4),
        "detection_precision": round(det_precision, 4),
        "fpr": round(fpr, 4),
        "rule_counts": dict(rule_counts),
        "report_file": str(report_path),
    }


def main():
    print("=" * 80)
    print(f"STARTING COMPREHENSIVE VALIDATION ON: {SYNTHETIC_CSV.name}")
    print("=" * 80)

    df = pd.read_csv(SYNTHETIC_CSV, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]

    print(f"Dataset Loaded: {len(df):,} flows")
    print(f"Ground Truth Class Counts:\n{df['Label'].value_counts()}\n")

    results = {}
    results["flow"] = validate_flow_model(df, n_samples=800)
    results["time_window"] = validate_time_window_model(df, window_seconds=30, max_windows=100)
    results["packet"] = validate_packet_model(df, n_samples=800)
    results["session"] = validate_session_model(df, n_samples=1200)
    results["integrated"] = validate_integrated_pipeline(str(SYNTHETIC_CSV), max_windows=100)

    out_file = OUT_DIR / "synthetic_validation_full_summary.json"
    out_file.write_text(json.dumps(results, indent=2, default=_JSON_DEFAULT), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"VALIDATION COMPLETED SUCCESSFULLY. SUMMARY SAVED TO:\n{out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
