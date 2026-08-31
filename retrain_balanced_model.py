"""Retrain XGBoost flow model with behavioral features only and proper regularization."""

import os
import shutil
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE
OUTPUT_DIR = HERE

MAX_ROWS_PER_FILE = 200_000
CHUNK_SIZE = 100_000

# Features that memorize dataset-specific identifiers instead of attack behavior.
IDENTIFIER_FEATURES = {
    "Dst Port",         # Memorizes port-attack associations
    "Fwd Seg Size Min", # TCP MSS negotiation, varies by OS/NIC
    "Init Fwd Win Byts",# TCP window size, OS-specific
    "Init Bwd Win Byts",# TCP window size, OS-specific
}

all_features_needed = joblib.load(HERE / "important_features.pkl")
behavioral_features = [f for f in all_features_needed if f not in IDENTIFIER_FEATURES]

print("=" * 80)
print("RETRAINING FLOW MODEL — BEHAVIORAL FEATURES ONLY")
print("=" * 80)
print(f"Original features: {len(all_features_needed)}")
print(f"Behavioral features ({len(behavioral_features)}): {behavioral_features}")

print("\n1. Loading training data from CSV files (in chunks)...")
all_data = []

csv_files = sorted(TRAIN_DIR.glob("0*-*-2018.csv"))
for csv_file in csv_files:
    print(f"   Loading {csv_file.name}...")
    try:
        loaded_rows = 0
        file_chunks = []
        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            nrows=MAX_ROWS_PER_FILE,
            low_memory=False,
            na_values=["", "NA", "NaN", "nan", "?"],
        ):
            if "Label" not in chunk.columns:
                continue
            
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            chunk = chunk[chunk["Label"] != "Label"]
            chunk = chunk[chunk["Label"] != ""]
            
            if "total_packets" not in chunk.columns and "Tot Fwd Pkts" in chunk.columns:
                chunk["total_packets"] = (
                    pd.to_numeric(chunk.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
                    + pd.to_numeric(chunk.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
                )
            if "burstiness" not in chunk.columns:
                chunk["burstiness"] = 0.0

            cols = [c for c in behavioral_features if c in chunk.columns]
            chunk_clean = chunk[cols + ["Label"]].dropna()
            for c in cols:
                chunk_clean[c] = pd.to_numeric(chunk_clean[c], errors="coerce").fillna(0.0)
            
            file_chunks.append(chunk_clean)
            loaded_rows += len(chunk_clean)
            if loaded_rows >= MAX_ROWS_PER_FILE:
                break
        
        if file_chunks:
            file_df = pd.concat(file_chunks, ignore_index=True)
            print(f"      Loaded {len(file_df):,} samples")
            all_data.append(file_df)
    except Exception as e:
        print(f"      Error: {e}")

if not all_data:
    print("ERROR: No training data loaded")
    raise SystemExit(1)

data = pd.concat(all_data, ignore_index=True)
print(f"\n   Total samples: {len(data):,}")

# Filter out very rare single-sample corrupted labels
label_counts = data["Label"].value_counts()
valid_labels = label_counts[label_counts >= 10].index
data = data[data["Label"].isin(valid_labels)]

print("\n2. Class distribution:")
class_dist = data["Label"].value_counts()
for label, count in class_dist.items():
    pct = count / len(data) * 100
    print(f"   {label:30s}: {count:8d} ({pct:5.1f}%)")

print("\n3. Computing class weights...")
class_weights = {}
total_samples = len(data)
n_classes = len(class_dist)
for label, count in class_dist.items():
    weight = total_samples / (n_classes * count)
    class_weights[label] = weight

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Label"])
feature_cols = [c for c in behavioral_features if c in data.columns]
X = data[feature_cols].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n4. Train samples: {len(X_train):,}  Test samples: {len(X_test):,}")

sample_weights = np.array([class_weights[label_encoder.classes_[label]] for label in y_train], dtype=np.float32)

print("\n5. Training XGBoost with regularization + early stopping...")
new_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    min_child_weight=10,
    gamma=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
)

sample_weights_test = np.array([class_weights[label_encoder.classes_[label]] for label in y_test], dtype=np.float32)
new_model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[sample_weights_test],
    verbose=20,
)

print("\n6. Model evaluation:")
train_acc = new_model.score(X_train, y_train)
test_acc = new_model.score(X_test, y_test)
print(f"   Train accuracy: {train_acc:.4f}")
print(f"   Test accuracy:  {test_acc:.4f}")
print(f"   Gap:            {train_acc - test_acc:.4f}")

print("\n7. Saving model and updated feature list...")
model_path = OUTPUT_DIR / "xgb_flow_model_balanced.pkl"
joblib.dump(new_model, model_path)

backup_path = OUTPUT_DIR / "xgb_flow_model_original.pkl"
if not backup_path.exists():
    shutil.copy(OUTPUT_DIR / "xgb_flow_model.pkl", backup_path)

shutil.copy(model_path, OUTPUT_DIR / "xgb_flow_model.pkl")
joblib.dump(feature_cols, OUTPUT_DIR / "important_features.pkl")
joblib.dump(label_encoder.classes_, OUTPUT_DIR / "labels.pkl")
print(f"   ✅ Saved xgb_flow_model.pkl + important_features.pkl ({len(feature_cols)} features)")

print("\n" + "=" * 80)
print("FLOW MODEL RETRAINING COMPLETE")
print("=" * 80)
