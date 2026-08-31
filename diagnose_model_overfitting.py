"""Analyze why the model only detects one attack type."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Load models
flow_model = joblib.load(HERE / "xgb_flow_model.pkl")
flow_labels = joblib.load(HERE / "labels.pkl").astype(str)
window_model = joblib.load(HERE / "time_window" / "processed" / "xgb_time_window_model.joblib")
window_labels = np.load(HERE / "time_window" / "processed" / "window_class_names.npy", allow_pickle=True).astype(str)

print("=" * 80)
print("FLOW MODEL ANALYSIS")
print("=" * 80)
print(f"Model type: {type(flow_model)}")
print(f"Classes trained on: {flow_labels}")
print(f"Number of classes: {len(flow_labels)}")
print(f"\nFeature importance (top 15):")

# Get feature importance if available
if hasattr(flow_model, "feature_importances_"):
    importance = flow_model.feature_importances_
    flow_features = joblib.load(HERE / "important_features.pkl")
    
    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {flow_features[idx]:30s} -> {importance[idx]:.6f}")
else:
    print("  Model does not expose feature importance")

print("\n" + "=" * 80)
print("WINDOW MODEL ANALYSIS")
print("=" * 80)
print(f"Model type: {type(window_model)}")
print(f"Classes trained on: {window_labels}")
print(f"Number of classes: {len(window_labels)}")
print(f"\nFeature importance (top 15):")

if hasattr(window_model, "feature_importances_"):
    importance = window_model.feature_importances_
    window_features = np.load(HERE / "time_window" / "processed" / "window_feature_names.npy", allow_pickle=True).astype(str)
    
    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {window_features[idx]:30s} -> {importance[idx]:.6f}")
else:
    print("  Model does not expose feature importance")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("""
If the top features are:
- Attack-type specific (e.g., "known Slowloris indicators")
- Dataset metadata (timestamp, IP ranges from training)
- Rare/unique patterns from training data only

THEN: The model learned TRAINING DATA PATTERNS, not generalizable attack features.

SOLUTION: Retrain with:
1. BALANCED attack types (equal samples per class)
2. CROSS-VALIDATION on different attack types
3. REGULARIZATION (prevent overfitting)
4. TEST on held-out attack types (e.g., train on Slowloris+Hulk, test on LOIC)
5. FEATURE SELECTION (keep only generalizable features)
6. DATA AUGMENTATION (synthetic variants of each attack)
""")

print("=" * 80)
print("CLASS DISTRIBUTION IN TRAINING DATA (estimated from saved labels)")
print("=" * 80)

# Try to find training data to analyze distribution
training_csvs = list(HERE.glob("*.csv"))
if training_csvs:
    print(f"\nFound {len(training_csvs)} CSV files. Analyzing first one...")
    df = pd.read_csv(training_csvs[0], low_memory=False)
    if "Label" in df.columns:
        label_dist = df["Label"].value_counts()
        print("\nLabel distribution in training data:")
        print(label_dist)
        print(f"\nBenign: {label_dist.get('Benign', 0) / len(df) * 100:.1f}%")
        print(f"Total attack samples: {(len(df) - label_dist.get('Benign', 0)) / len(df) * 100:.1f}%")
        print(f"\nClass imbalance ratio (max:min): {label_dist.max() / label_dist.min():.1f}x")
        if label_dist.get('Slowloris', 0) > 0:
            print(f"Slowloris in training: {label_dist.get('Slowloris', 0)} ({label_dist.get('Slowloris', 0) / len(df) * 100:.1f}%)")
else:
    print("No CSV files found to analyze training distribution")
