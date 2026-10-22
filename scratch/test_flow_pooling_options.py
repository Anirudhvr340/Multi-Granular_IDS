"""
Test Flow Pooling Options on 02-14-2018
"""
import os, sys, joblib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, './time_window')
import step0_preprocessing as preprocessing
from nids.adapters.flow_adapter import FlowModelAdapter

frame = pd.read_csv('02-14-2018.csv', low_memory=False)
frame = preprocessing.clean_chunk(frame)
frame['window_start'] = frame['Timestamp'].dt.floor('30s')

adapter = FlowModelAdapter()
model = adapter._model
labels = adapter._labels
benign_col = [i for i, c in enumerate(model.classes_) if labels[c] == 'Benign'][0]

windows_data = []

for w_idx, (window_start, w_df) in enumerate(frame.groupby('window_start', sort=True)):
    agg = preprocessing.window_aggregate(w_df, window_start)
    gt = agg.get('majority_label', 'Unknown')
    is_attack = (gt != 'Benign')
    
    X = adapter._prepare_features(w_df)
    probs = model.predict_proba(X)
    flow_p_attack = 1.0 - probs[:, benign_col]
    
    max_p = float(np.max(flow_p_attack))
    mean_p = float(np.mean(flow_p_attack))
    med_p = float(np.median(flow_p_attack))
    p90 = float(np.percentile(flow_p_attack, 90))
    p95 = float(np.percentile(flow_p_attack, 95))
    p99 = float(np.percentile(flow_p_attack, 99))
    
    f50 = float(np.mean(flow_p_attack >= 0.50))
    f70 = float(np.mean(flow_p_attack >= 0.70))
    f90 = float(np.mean(flow_p_attack >= 0.90))
    
    windows_data.append({
        'gt': gt, 'is_attack': is_attack,
        'max_p': max_p, 'mean_p': mean_p, 'med_p': med_p,
        'p90': p90, 'p95': p95, 'p99': p99,
        'f50': f50, 'f70': f70, 'f90': f90,
        'n': len(w_df)
    })

y_true = np.array([1 if w['is_attack'] else 0 for w in windows_data])

def eval_score(scores, thresh=0.50):
    preds = (scores >= thresh).astype(int)
    tp = np.sum((y_true == 1) & (preds == 1))
    fp = np.sum((y_true == 0) & (preds == 1))
    tn = np.sum((y_true == 0) & (preds == 0))
    fn = np.sum((y_true == 1) & (preds == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return rec, prec, fpr, f1

print("="*85)
print(f"{'Method':35s} | {'Thresh':6s} | {'Recall':8s} | {'Prec':8s} | {'FPR':8s} | {'F1':8s}")
print("="*85)

# Baseline: pure max
r, p, fpr, f1 = eval_score(np.array([w['max_p'] for w in windows_data]), 0.35)
print(f"{'Baseline: Pure max':35s} | {0.35:6.2f} | {r:8.4f} | {p:8.4f} | {fpr:8.4f} | {f1:8.4f}")

# P90
for th in [0.80, 0.90, 0.95]:
    r, p, fpr, f1 = eval_score(np.array([w['p90'] for w in windows_data]), th)
    print(f"{'P90 quantile':35s} | {th:6.2f} | {r:8.4f} | {p:8.4f} | {fpr:8.4f} | {f1:8.4f}")

# Density: frac_70
for th in [0.15, 0.20, 0.25]:
    r, p, fpr, f1 = eval_score(np.array([w['f70'] for w in windows_data]), th)
    print(f"{'Density: frac >= 0.70':35s} | {th:6.2f} | {r:8.4f} | {p:8.4f} | {fpr:8.4f} | {f1:8.4f}")

# Density: frac_90
for th in [0.08, 0.10, 0.15]:
    r, p, fpr, f1 = eval_score(np.array([w['f90'] for w in windows_data]), th)
    print(f"{'Density: frac >= 0.90':35s} | {th:6.2f} | {r:8.4f} | {p:8.4f} | {fpr:8.4f} | {f1:8.4f}")

# Hybrid Density-Weighted Score:
# If high density (frac_70 >= 0.15) -> high attack prob (use p95 or max_p)
# Else -> mean_p (low for benign)
hybrid_scores = []
for w in windows_data:
    if w['f70'] >= 0.15 or w['f90'] >= 0.08:
        # High density attack
        score = w['p95']
    elif w['max_p'] >= 0.999 and w['f50'] >= 0.10:
        # Strong concentrated attack
        score = w['p90']
    else:
        # Diluted background flows
        score = w['mean_p']
    hybrid_scores.append(score)

hybrid_scores = np.array(hybrid_scores)
for th in [0.40, 0.50, 0.60]:
    r, p, fpr, f1 = eval_score(hybrid_scores, th)
    print(f"{'Hybrid Density Score':35s} | {th:6.2f} | {r:8.4f} | {p:8.4f} | {fpr:8.4f} | {f1:8.4f}")
print("="*85)

