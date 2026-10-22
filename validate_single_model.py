#!/usr/bin/env python3
"""
=============================================================================
MULTI-GRANULAR NIDS: STANDALONE SINGLE-MODEL EVALUATION CLI
=============================================================================
Allows evaluating each of the 4 models separately on any CSV dataset
(including synthetic datasets like realistic_synthetic_cicids2018_converted.csv).

Usage Examples:
---------------
1. Run Packet 1D-CNN separately:
   python validate_single_model.py --model packet --csv realistic_synthetic_cicids2018_converted.csv

2. Run Flow XGBoost separately:
   python validate_single_model.py --model flow --csv realistic_synthetic_cicids2018_converted.csv

3. Run Time-Window XGBoost separately:
   python validate_single_model.py --model time_window --csv realistic_synthetic_cicids2018_converted.csv

4. Run Session CNN-LSTM separately:
   python validate_single_model.py --model session --csv realistic_synthetic_cicids2018_converted.csv

5. Run all 4 models sequentially:
   python validate_single_model.py --model all --csv realistic_synthetic_cicids2018_converted.csv
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))
sys.path.insert(0, str(HERE / "packet_level"))
sys.path.insert(0, str(HERE / "session_level"))

import step0_preprocessing as preprocessing
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter


def evaluate_packet_model(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("EVALUATING: PACKET 1D-CNN (DIMENSIONLESS HEADER MODEL)")
    print("=" * 80)
    adapter = PacketModelAdapter()
    
    t0 = time.perf_counter()
    y_true = df["Label"].astype(str).str.strip().values
    
    # Extract features using the adapter's trained pipeline
    from train_packet_1d_cnn import extract_dimensionless_packet_features
    X_df, _ = extract_dimensionless_packet_features(df)
    X = X_df[adapter.features].values.astype(np.float32)
    X_sc = adapter.scaler.transform(X)
    X_cnn = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)
    
    # 1D-CNN + Platt Calibrator
    raw_probs = adapter.model.predict(X_cnn, batch_size=256, verbose=0)
    cal_probs = adapter.calibrator.predict_proba(raw_probs)
    pred_idx = np.argmax(cal_probs, axis=1)
    y_pred = np.array([adapter.classes[i] for i in pred_idx])
    elapsed = time.perf_counter() - t0
    
    _print_metrics(y_true, y_pred, elapsed, "Packet 1D-CNN")


def evaluate_flow_model(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("EVALUATING: FLOW XGBOOST (STATISTICAL FLOW-LEVEL MODEL)")
    print("=" * 80)
    adapter = FlowModelAdapter()
    
    t0 = time.perf_counter()
    y_true = df["Label"].astype(str).str.strip().values
    
    X = adapter._prepare_features(df)
    proba_matrix = adapter._model.predict_proba(X)
    pred_indices = np.argmax(proba_matrix, axis=1)
    
    # Map model indices back to class labels
    idx_to_label = {col_idx: adapter._idx_to_label[label_idx] for col_idx, label_idx in enumerate(adapter._class_indices)}
    y_pred = np.array([idx_to_label.get(i, "Unknown") for i in pred_indices])
    elapsed = time.perf_counter() - t0
    
    _print_metrics(y_true, y_pred, elapsed, "Flow XGBoost")


def evaluate_session_model(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("EVALUATING: SESSION CNN-LSTM (SEQUENTIAL SESSION MODEL)")
    print("=" * 80)
    from train_session_cnn_lstm import SessionCNNLSTM, SEQUENCE_LENGTH
    import torch
    
    adapter = SessionModelAdapter()
    t0 = time.perf_counter()
    
    # Ensure features
    for col in adapter.features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        elif col != "Time_Delta":
            df[col] = 0.0

    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
        df["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)
    else:
        df["Time_Delta"] = 0.1

    num_cols = [c for c in adapter.features if c in df.columns]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feat_matrix = df[adapter.features].values.astype(np.float32)
    labels = df["Label"].astype(str).str.strip().values
    seq_len = SEQUENCE_LENGTH  # 30

    session_tensors = []
    session_y = []
    step = max(1, seq_len // 2)

    for start_idx in range(0, len(df) - seq_len + 1, step):
        end_idx = start_idx + seq_len
        sess_window = feat_matrix[start_idx:end_idx]
        sess_lbls = labels[start_idx:end_idx]
        
        non_benign = [lbl for lbl in sess_lbls if "Benign" not in str(lbl)]
        if non_benign:
            chosen_label = pd.Series(non_benign).mode()[0]
        else:
            chosen_label = "Benign"
            
        session_tensors.append(sess_window)
        session_y.append(chosen_label)

    if not session_tensors:
        print("[!] Not enough flows to construct sequences of length 30.")
        return

    X_sess = np.array(session_tensors, dtype=np.float32)
    N_s, T, D = X_sess.shape
    X_sess_sc = adapter.scaler.transform(X_sess.reshape(-1, D)).reshape(N_s, T, D)

    with torch.no_grad():
        tensor_x = torch.tensor(X_sess_sc, dtype=torch.float32).to(adapter.device)
        logits = adapter.model(tensor_x)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()

    y_pred = np.array([adapter.classes[i] for i in pred_idx])
    y_true = np.array(session_y)
    elapsed = time.perf_counter() - t0
    
    _print_metrics(y_true, y_pred, elapsed, "Session CNN-LSTM")


def evaluate_time_window_model(df: pd.DataFrame, window_seconds: int = 30):
    print("\n" + "=" * 80)
    print("EVALUATING: TIME-WINDOW XGBOOST (MACRO 30-SECOND ROLLING MODEL)")
    print("=" * 80)
    adapter = TimeWindowAdapter()
    
    t0 = time.perf_counter()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").fillna(pd.Timestamp("2018-02-15 01:00:00"))
    df["window_start"] = df["Timestamp"].dt.floor(f"{window_seconds}s")
    
    y_true = []
    y_pred = []
    
    adapter.reset_history()
    for w_start, w_df in df.groupby("window_start", sort=True):
        agg = preprocessing.window_aggregate(w_df, w_start)
        ev = adapter.predict(agg, window_id=str(w_start))
        
        # Determine window ground truth
        majority_lbl = w_df["Label"].astype(str).str.strip().mode()[0]
        y_true.append(majority_lbl)
        y_pred.append(ev.prediction)
        
    elapsed = time.perf_counter() - t0
    _print_metrics(np.array(y_true), np.array(y_pred), elapsed, "Time-Window XGBoost")


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed: float, model_name: str):
    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Binary attack vs benign metrics
    is_gt_att = np.array(["Benign" not in str(l) for l in y_true])
    is_pred_att = np.array(["Benign" not in str(l) for l in y_pred])
    
    tp = np.sum(is_gt_att & is_pred_att)
    fp = np.sum(~is_gt_att & is_pred_att)
    fn = np.sum(is_gt_att & ~is_pred_att)
    tn = np.sum(~is_gt_att & ~is_pred_att)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"\n[+] PERFORMANCE SUMMARY — {model_name.upper()}:")
    print(f"    - Total Records Evaluated:   {len(y_true):,}")
    print(f"    - Execution Time:            {elapsed:.2f}s ({elapsed/max(1, len(y_true))*1000:.2f} ms/record)")
    print(f"    - Multi-Class Accuracy:      {acc * 100:.2f}%")
    print(f"    - Attack Detection Recall:   {recall * 100:.2f}% ({tp}/{tp+fn} attacks)")
    print(f"    - Attack Precision:          {precision * 100:.2f}%")
    print(f"    - False Positive Rate (FPR): {fpr * 100:.2f}% ({fp}/{fp+tn} benign)")
    print(f"    - Weighted F1-Score:         {w_f1 * 100:.2f}%")
    print(f"    - Macro F1-Score:            {m_f1 * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("DETAILED PER-CLASS CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation of NIDS Models on Synthetic/Real CSVs.")
    parser.add_argument(
        "--model",
        choices=["packet", "flow", "time_window", "session", "all"],
        default="all",
        help="Which model to run independently (default: all)",
    )
    parser.add_argument(
        "--csv",
        default=str(HERE / "realistic_synthetic_cicids2018_converted.csv"),
        help="Path to CSV dataset (default: realistic_synthetic_cicids2018_converted.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows to load from CSV (optional)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[!] Error: CSV file not found at: {csv_path}")
        sys.exit(1)

    print("=" * 80)
    print("MULTI-GRANULAR NIDS: INDEPENDENT MODEL BENCHMARK")
    print(f"Target Dataset: {csv_path.name}")
    print(f"Model Selection: {args.model.upper()}")
    print("=" * 80)

    print(f"[*] Loading dataset from {csv_path.name}...")
    df = pd.read_csv(csv_path, nrows=args.max_rows, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]
    print(f"[+] Loaded {len(df):,} flows.")

    if args.model in ("packet", "all"):
        evaluate_packet_model(df.copy())

    if args.model in ("flow", "all"):
        evaluate_flow_model(df.copy())

    if args.model in ("time_window", "all"):
        evaluate_time_window_model(df.copy())

    if args.model in ("session", "all"):
        evaluate_session_model(df.copy())


if __name__ == "__main__":
    main()

