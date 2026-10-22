#!/usr/bin/env python3
"""
Independent Model Validation Script
====================================
Phases 2–6: Test all four models independently BEFORE integration.

Usage
-----
  cd "D:\\Multi-Granular IDS\\Multi-Granular_IDS"
  python run_independent_tests.py [--csv 02-14-2018.csv] [--n-samples 500]

Output
------
  nids/validation_results/independent_test_results.json
  nids/validation_results/model_health_report.json
  nids/validation_results/attack_coverage_table.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIME_WINDOW_DIR = HERE / "time_window"
PROCESSED_DIR   = TIME_WINDOW_DIR / "processed"
OUT_DIR         = HERE / "nids" / "validation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(TIME_WINDOW_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_WINDOW_DIR))

import step0_preprocessing as preprocessing
from nids.adapters.flow_adapter        import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.packet_adapter      import PacketModelAdapter
from nids.adapters.session_adapter     import SessionModelAdapter

_JSON_DEFAULT = lambda v: v.item() if isinstance(v, np.generic) else str(v)


# ===========================================================================
def test_flow_model(csv_path: str, n_samples: int = 500) -> dict:
    print("\n" + "="*60)
    print("FLOW MODEL — INDEPENDENT TEST")
    print("="*60)

    adapter = FlowModelAdapter()

    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False, nrows=min(n_samples * 5, 50000))
    df = preprocessing.clean_chunk(df)
    if df.empty:
        return {"status": "error", "reason": "No valid rows in CSV"}

    # Sample benign and attack flows
    benign_df = df[df["Label"] == "Benign"].head(n_samples // 2)
    attack_df = df[df["Label"] != "Benign"].head(n_samples // 2)
    test_df   = pd.concat([benign_df, attack_df], ignore_index=True).sample(
        frac=1, random_state=42
    )
    ground_truth = test_df["Label"].tolist()

    predictions   = []
    confidences   = []
    attack_probs  = []
    latencies     = []

    for _, row in test_df.iterrows():
        ev = adapter.predict(row.to_dict())
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append(ev.inference_latency_ms)

    # Metrics
    metrics = _compute_metrics(ground_truth, predictions, attack_probs)
    metrics["n_tested"]            = len(test_df)
    metrics["mean_confidence"]     = round(float(np.mean(confidences)), 4)
    metrics["mean_attack_prob"]    = round(float(np.mean(attack_probs)), 4)
    metrics["mean_latency_ms"]     = round(float(np.mean(latencies)), 3)
    metrics["p95_latency_ms"]      = round(float(np.percentile(latencies, 95)), 3)

    # Sample outputs
    samples = _build_sample_outputs(
        test_df, ground_truth, predictions, confidences, attack_probs, adapter, n=5
    )

    print_model_output("FLOW MODEL", samples, metrics, adapter)

    return {
        "model":           "Flow XGBoost",
        "granularity":     "flow",
        "model_file":      str(HERE / "xgb_flow_model.pkl"),
        "n_features":      len(adapter.feature_names),
        "trained_classes": adapter.trained_classes,
        "n_classes":       len(adapter.trained_classes),
        "status":          "ok",
        "metrics":         metrics,
        "samples":         samples,
    }


# ===========================================================================
def test_time_window_model(n_samples: int = 200) -> dict:
    print("\n" + "="*60)
    print("TIME-WINDOW MODEL — INDEPENDENT TEST")
    print("="*60)

    adapter = TimeWindowAdapter()

    # Load preprocessed test data
    X_test    = np.load(PROCESSED_DIR / "X_test.npy")
    y_test    = np.load(PROCESSED_DIR / "y_test.npy")
    class_names = np.load(PROCESSED_DIR / "window_class_names.npy",
                          allow_pickle=True).astype(str)
    times_test  = np.load(PROCESSED_DIR / "window_times_test.npy")
    base_feat   = np.load(PROCESSED_DIR / "window_base_feature_names.npy",
                          allow_pickle=True).astype(str)

    # We have X_test already scaled + temporal, but for the adapter test
    # we load the raw base features and let the adapter handle the temporal pipeline.
    # Use the first n_samples from the test set.
    n = min(n_samples, len(y_test))
    idx_sample = np.arange(n)

    # Load raw (unscaled) base window features from window_features.npy
    raw_features = np.load(PROCESSED_DIR / "window_features.npy")
    test_indices = np.load(PROCESSED_DIR / "test_indices.npy")[:n]
    X_base_test  = raw_features[test_indices]
    times_sel    = np.load(PROCESSED_DIR / "window_times.npy")[test_indices]
    y_sel        = y_test[:n]
    ground_truth = [str(class_names[int(yi)]) if int(yi) < len(class_names) else f"cls_{yi}"
                    for yi in y_sel]

    adapter.reset_history()
    predictions  = []
    confidences  = []
    attack_probs = []
    latencies    = []

    for i in range(n):
        base_vec   = X_base_test[i].astype(np.float32)
        window_ts  = times_sel[i]
        input_dict = {name: float(val) for name, val in zip(base_feat, base_vec)}
        input_dict["window_start"] = window_ts

        ev = adapter.predict(input_dict, window_id=f"tw_{i}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append(ev.inference_latency_ms)

    metrics = _compute_metrics(ground_truth, predictions, attack_probs)
    metrics["n_tested"]        = n
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    samples = _build_sample_outputs_tw(
        ground_truth[:5], predictions[:5], confidences[:5], attack_probs[:5]
    )
    print_model_output("TIME-WINDOW XGBoost", samples, metrics, adapter)

    return {
        "model":           "Time-Window XGBoost",
        "granularity":     "time_window",
        "model_file":      str(PROCESSED_DIR / "xgb_time_window_model.joblib"),
        "n_features":      len(adapter.feature_names),
        "trained_classes": adapter.trained_classes,
        "n_classes":       len(adapter.trained_classes),
        "status":          "ok",
        "metrics":         metrics,
        "samples":         samples,
    }


# ===========================================================================
def test_packet_model(csv_path: str, n_samples: int = 500) -> dict:
    print("\n" + "="*60)
    print("PACKET MODEL — INDEPENDENT TEST")
    print("="*60)
    adapter = PacketModelAdapter()
    df = pd.read_csv(csv_path, nrows=n_samples * 2, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"].iloc[:n_samples]

    ground_truth = df["Label"].tolist()
    predictions  = []
    confidences  = []
    attack_probs = []
    latencies    = []

    for i in range(len(df)):
        row_df = df.iloc[[i]]
        t0 = time.perf_counter()
        ev = adapter.predict(row_df, window_id=f"pkt_{i}")
        predictions.append(ev.prediction)
        confidences.append(ev.confidence)
        attack_probs.append(ev.attack_probability)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = _compute_metrics(ground_truth, predictions, attack_probs)
    metrics["n_tested"]        = len(df)
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    samples = _build_sample_outputs(
        df.head(5), ground_truth[:5], predictions[:5], confidences[:5], attack_probs[:5], adapter
    )
    print_model_output("PACKET 1D-CNN", samples, metrics, adapter)

    return {
        "model":           "Packet 1D-CNN",
        "granularity":     "packet",
        "model_file":      str(adapter.model_dir / "cnn_packet_model.keras"),
        "n_features":      len(adapter.feature_names),
        "trained_classes": adapter.trained_classes,
        "n_classes":       len(adapter.trained_classes),
        "status":          "ok",
        "metrics":         metrics,
        "samples":         samples,
    }


# ===========================================================================
def test_session_model(csv_path: str, n_samples: int = 500) -> dict:
    print("\n" + "="*60)
    print("SESSION MODEL — INDEPENDENT TEST")
    print("="*60)
    adapter = SessionModelAdapter()
    df = pd.read_csv(csv_path, nrows=n_samples * 2, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"].iloc[:n_samples]

    ground_truth = []
    predictions  = []
    confidences  = []
    attack_probs = []
    latencies    = []

    chunk_size = 30
    for start_idx in range(0, len(df), chunk_size):
        chunk_df = df.iloc[start_idx:start_idx + chunk_size]
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

    metrics = _compute_metrics(ground_truth, predictions, attack_probs)
    metrics["n_tested"]        = len(ground_truth)
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 3)

    samples = _build_sample_outputs_tw(
        ground_truth[:5], predictions[:5], confidences[:5], attack_probs[:5]
    )
    print_model_output("SESSION CNN-LSTM", samples, metrics, adapter)

    return {
        "model":           "Session CNN-LSTM",
        "granularity":     "session",
        "model_file":      str(adapter.model_dir / "session_cnn_lstm.pth"),
        "n_features":      len(adapter.feature_names),
        "trained_classes": adapter.trained_classes,
        "n_classes":       len(adapter.trained_classes),
        "status":          "ok",
        "metrics":         metrics,
        "samples":         samples,
    }


# ===========================================================================
def _compute_metrics(
    ground_truth: list,
    predictions:  list,
    attack_probs: list,
    attack_threshold: float = 0.35,
) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        f1_score, classification_report
    )

    # Binary (attack vs benign)
    gt_binary = [0 if g == "Benign" else 1 for g in ground_truth]
    pr_binary = [0 if p == "Benign" else 1 for p in predictions]

    # Attack gate binary (using P(attack) threshold)
    gate_binary = [1 if p >= attack_threshold else 0 for p in attack_probs]

    acc = accuracy_score(ground_truth, predictions)

    # Safely compute macro/weighted F1
    try:
        macro_f1 = f1_score(ground_truth, predictions, average="macro", zero_division=0)
        w_f1     = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
    except Exception:
        macro_f1 = w_f1 = 0.0

    # Per-class report (suppress printing)
    try:
        report = classification_report(
            ground_truth, predictions, zero_division=0, output_dict=True
        )
    except Exception:
        report = {}

    # Binary confusion
    tp = sum(1 for g, p in zip(gt_binary, pr_binary) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_binary, pr_binary) if g == 0 and p == 1)
    tn = sum(1 for g, p in zip(gt_binary, pr_binary) if g == 0 and p == 0)
    fn = sum(1 for g, p in zip(gt_binary, pr_binary) if g == 1 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Prediction class distribution
    pred_dist: dict = {}
    for p in predictions:
        pred_dist[p] = pred_dist.get(p, 0) + 1

    gt_dist: dict = {}
    for g in ground_truth:
        gt_dist[g] = gt_dist.get(g, 0) + 1

    return {
        "accuracy":             round(acc, 4),
        "macro_f1":             round(macro_f1, 4),
        "weighted_f1":          round(w_f1, 4),
        "binary_TP":            tp,
        "binary_FP":            fp,
        "binary_TN":            tn,
        "binary_FN":            fn,
        "false_positive_rate":  round(fpr, 4),
        "attack_recall":        round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "attack_precision":     round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4),
        "gt_class_distribution":   gt_dist,
        "pred_class_distribution": pred_dist,
        "per_class_report":        {k: v for k, v in report.items()
                                    if isinstance(v, dict)},
    }


def _build_sample_outputs(df, gts, preds, confs, a_probs, adapter, n=5):
    samples = []
    for i in range(min(n, len(gts))):
        row = df.iloc[i]
        samples.append({
            "input_label":      str(row.get("Label", "?")),
            "ground_truth":     gts[i],
            "prediction":       preds[i],
            "confidence":       round(confs[i], 4),
            "attack_probability": round(a_probs[i], 4),
            "correct":          gts[i] == preds[i],
        })
    return samples


def _build_sample_outputs_tw(gts, preds, confs, a_probs):
    samples = []
    for i in range(len(gts)):
        samples.append({
            "ground_truth":       gts[i],
            "prediction":         preds[i],
            "confidence":         round(confs[i], 4),
            "attack_probability": round(a_probs[i], 4),
            "correct":            gts[i] == preds[i],
        })
    return samples


def print_model_output(model_name: str, samples: list, metrics: dict, adapter):
    hdr = f"\n{'='*50}\n{model_name} OUTPUT\n{'='*50}"
    print(hdr)
    print(f"  Trained classes ({len(adapter.trained_classes)}):")
    for c in adapter.trained_classes:
        print(f"    - {c}")
    print(f"\n  Accuracy:    {metrics.get('accuracy', 'N/A')}")
    print(f"  Macro F1:    {metrics.get('macro_f1', 'N/A')}")
    print(f"  Weighted F1: {metrics.get('weighted_f1', 'N/A')}")
    print(f"  Attack Recall:    {metrics.get('attack_recall', 'N/A')}")
    print(f"  Attack Precision: {metrics.get('attack_precision', 'N/A')}")
    print(f"  FPR:              {metrics.get('false_positive_rate', 'N/A')}")
    print(f"\n  Sample Predictions:")
    for s in samples[:5]:
        mark = "OK" if s["correct"] else "XX"
        print(f"    [{mark}] GT={s['ground_truth']:<30} Pred={s['prediction']:<30} "
              f"conf={s['confidence']:.3f} P(atk)={s['attack_probability']:.3f}")


# ===========================================================================
def build_attack_coverage_table(flow_result, tw_result, pkt_result, sess_result) -> list:
    rows = []

    def row(name, gran, model_res):
        classes = model_res.get("trained_classes", [])
        intended = model_res.get("intended_coverage", classes)
        status   = model_res.get("status", "unknown")
        pred_dist = model_res.get("metrics", {}).get("pred_class_distribution", {})
        tested   = list(pred_dist.keys()) if pred_dist else (["N/A"] if status == "unavailable" else [])
        return {
            "model":              name,
            "granularity":        gran,
            "status":             status,
            "trained_classes":    classes,
            "intended_coverage":  intended,
            "actually_predicted": tested,
            "n_trained_classes":  len(classes),
            "n_predicted_classes": len(tested),
        }

    rows.append(row("Flow XGBoost",        "flow",        flow_result))
    rows.append(row("Time-Window XGBoost", "time_window", tw_result))
    rows.append(row("Packet 1D-CNN",       "packet",      pkt_result))
    rows.append(row("Session CNN-LSTM",     "session",    sess_result))
    return rows


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",       default=str(HERE / "02-14-2018.csv"),
                        help="CICIDS-2018 CSV for flow model testing")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    all_results = {}

    # ---- Test each model independently ------------------------------------
    all_results["flow"]        = test_flow_model(args.csv, args.n_samples)
    all_results["time_window"] = test_time_window_model(args.n_samples // 2)
    all_results["packet"]      = test_packet_model(args.csv, args.n_samples)
    all_results["session"]     = test_session_model(args.csv, args.n_samples)

    # ---- Attack coverage table --------------------------------------------
    coverage = build_attack_coverage_table(
        all_results["flow"],
        all_results["time_window"],
        all_results["packet"],
        all_results["session"],
    )
    all_results["attack_coverage_table"] = coverage

    # ---- Print coverage table ---------------------------------------------
    print("\n" + "="*80)
    print("ATTACK COVERAGE TABLE")
    print("="*80)
    print(f"{'Model':<28} {'Granularity':<14} {'Status':<13} {'Trained':<8} {'Predicted':<10}")
    print("-"*80)
    for r in coverage:
        print(
            f"{r['model']:<28} {r['granularity']:<14} {r['status']:<13} "
            f"{r['n_trained_classes']:<8} {r['n_predicted_classes']:<10}"
        )

    # ---- Save results ------------------------------------------------------
    out_path = OUT_DIR / "independent_test_results.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, default=_JSON_DEFAULT),
        encoding="utf-8",
    )
    print(f"\nResults saved -> {out_path}")

    # Coverage table
    cov_path = OUT_DIR / "attack_coverage_table.json"
    cov_path.write_text(
        json.dumps(coverage, indent=2, default=_JSON_DEFAULT),
        encoding="utf-8",
    )
    print(f"Coverage table -> {cov_path}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
