"""
Sweep script to find optimal Flow Aggregation parameters on 02-14-2018 validation data.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "time_window"))

import step0_preprocessing as preprocessing
from nids.adapters.flow_adapter import FlowModelAdapter

csv_path = "02-14-2018.csv"
window_seconds = 30
os.environ["WINDOW_SECONDS"] = str(window_seconds)

print("Reading CSV and preprocessing chunks...")
frame = pd.read_csv(csv_path, low_memory=False)
frame = preprocessing.clean_chunk(frame)
frame["window_start"] = frame["Timestamp"].dt.floor(f"{window_seconds}s")

# Load flow model
flow_adapter = FlowModelAdapter()
model = flow_adapter._model
feature_names = flow_adapter._feature_names
class_indices = flow_adapter._class_indices
labels = flow_adapter._labels
idx_to_label = flow_adapter._idx_to_label

benign_idx = None
for col_idx, label_idx in enumerate(class_indices):
    if idx_to_label[label_idx] == "Benign":
        benign_idx = col_idx
        break

print(f"Benign column index in flow model: {benign_idx}")

# Precompute per-flow predictions for each window
window_data = []
print("Evaluating flows for each window...")

for window_start, window_df in frame.groupby("window_start", sort=True):
    gt = window_df["Label"].value_counts().index[0]
    is_attack = (gt != "Benign")
    
    X = flow_adapter._prepare_features(window_df)
    probs = model.predict_proba(X) # (N, 15)
    
    # Per-flow attack probability
    flow_attack_probs = 1.0 - probs[:, benign_idx]
    
    max_p = float(np.max(flow_attack_probs))
    mean_p = float(np.mean(flow_attack_probs))
    med_p = float(np.median(flow_attack_probs))
    p90 = float(np.percentile(flow_attack_probs, 90))
    p95 = float(np.percentile(flow_attack_probs, 95))
    p99 = float(np.percentile(flow_attack_probs, 99))
    
    frac_50 = float(np.mean(flow_attack_probs >= 0.50))
    frac_70 = float(np.mean(flow_attack_probs >= 0.70))
    frac_90 = float(np.mean(flow_attack_probs >= 0.90))
    
    window_data.append({
        "gt": gt,
        "is_attack": is_attack,
        "n_flows": len(window_df),
        "max_p": max_p,
        "mean_p": mean_p,
        "med_p": med_p,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "frac_50": frac_50,
        "frac_70": frac_70,
        "frac_90": frac_90,
    })

print(f"Computed summary stats for {len(window_data)} windows.")

# Now sweep thresholds
single_threshs = [0.80, 0.85, 0.90, 0.95]
density_threshs = [0.005, 0.01, 0.02, 0.05]

y_true = np.array([1 if w["is_attack"] else 0 for w in window_data])
n_total = len(y_true)
n_attack = np.sum(y_true == 1)
n_benign = np.sum(y_true == 0)

print(f"\nTotal: {n_total}, Attack: {n_attack}, Benign: {n_benign}")
print("\n" + "="*80)
print(f"{'Single Thresh':15s} | {'Density Thresh':15s} | {'Attack Rec':12s} | {'FPR':10s} | {'Prec':10s} | {'F1':10s}")
print("="*80)

best_f1 = 0
best_config = None

for st in single_threshs:
    for dt in density_threshs:
        # Decision logic:
        # Window is flagged as attack if max_p >= st OR frac_50 >= dt
        y_pred = []
        for w in window_data:
            flag = (w["max_p"] >= st or w["frac_50"] >= dt)
            y_pred.append(1 if flag else 0)
        
        y_pred = np.array(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = (st, dt, rec, fpr, prec, f1)
            
        print(f"{st:15.2f} | {dt*100:13.1f}% | {rec:12.4f} | {fpr:10.4f} | {prec:10.4f} | {f1:10.4f}")

print("\n" + "="*80)
print(f"BEST CONFIG: Single >= {best_config[0]}, Density >= {best_config[1]*100}% -> Recall: {best_config[2]:.4f}, FPR: {best_config[3]:.4f}, Prec: {best_config[4]:.4f}, F1: {best_config[5]:.4f}")
print("="*80)

