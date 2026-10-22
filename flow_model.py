import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

# ============================================================
# 1. LOAD DATA (STREAMING - 500K ROWS PER FILE)
# ============================================================
def load_data(path, rows_per_file=500000, chunk_size=100000):
    files = sorted(glob.glob(path))
    print(f"Total files found: {len(files)}")

    df_list = []
    drop_cols = {'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port'}

    for file in files:
        print(f"Processing: {file}")
        file_chunks = []
        rows_accumulated = 0
        
        try:
            # We use a nested loop to stop reading a file once we hit our target
            for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                # 1. Clean Column Names (Crucial for 2018 dataset)
                chunk.columns = chunk.columns.str.strip()
                
                # 2. Drop unnecessary columns immediately
                current_drop = [c for c in drop_cols if c in chunk.columns]
                chunk = chunk.drop(columns=current_drop)

                # 3. Handle Label strings and numeric conversion
                if 'Label' in chunk.columns:
                    chunk['Label'] = chunk['Label'].astype(str).str.strip()
                
                for col in chunk.columns:
                    if col != 'Label':
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')

                # 4. Determine how many rows to take from this chunk
                remaining_needed = rows_per_file - rows_accumulated
                if len(chunk) <= remaining_needed:
                    file_chunks.append(chunk)
                    rows_accumulated += len(chunk)
                else:
                    file_chunks.append(chunk.iloc[:remaining_needed])
                    rows_accumulated += remaining_needed
                    break # Stop reading this CSV file
            
            if file_chunks:
                file_combined = pd.concat(file_chunks, ignore_index=True)
                df_list.append(file_combined)
                print(f"  Successfully loaded {len(file_combined)} rows.")

        except Exception as e:
            print(f"  Error in {file}: {e}")

    print("\nFinal Concatenation...")
    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0)
    
    print(f"Total dataset size: {len(df)} rows.")
    return df

# ============================================================
# EXECUTION
# ============================================================
# 500k rows from 10 files = 5 million rows. 
# Ensure you have at least 16GB RAM for this volume.



# ============================================================
# 2. CLEAN DATA (SAFE)
# ============================================================
def clean_data(df):
    print("\nCleaning data...")
    
    # Convert Label to string to handle mixed types before encoding
    df['Label'] = df['Label'].astype(str)

    # Clean numeric columns one at a time. Calling replace/fillna on the whole
    # 4M+ row dataframe can force Pandas to allocate a huge temporary block.
    numeric_cols = [col for col in df.columns if col != "Label"]
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').astype('float32')
        values = values.mask(np.isinf(values), 0).fillna(0)
        df[col] = values

    print("Cleaned dataset size:", len(df))
    return df


# ============================================================
# 3. FEATURE ENGINEERING (SAFE CHECKS)
# ============================================================
def feature_engineering(df):

    def safe_col(name):
        return df[name] if name in df.columns else 0

    df['packet_ratio'] = safe_col('Tot Fwd Pkts') / (safe_col('Tot Bwd Pkts') + 1)
    df['byte_ratio'] = safe_col('TotLen Fwd Pkts') / (safe_col('TotLen Bwd Pkts') + 1)
    df['total_packets'] = safe_col('Tot Fwd Pkts') + safe_col('Tot Bwd Pkts')
    df['total_bytes'] = safe_col('TotLen Fwd Pkts') + safe_col('TotLen Bwd Pkts')
    df['avg_pkt_size'] = df['total_bytes'] / (df['total_packets'] + 1)
    df['flow_intensity'] = df['total_packets'] / (safe_col('Flow Duration') + 1)
    df['burstiness'] = safe_col('Pkt Len Std') / (safe_col('Pkt Len Mean') + 1)

    return df


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

df = load_data("D:/capstone/dataset 2018/*.csv", rows_per_file=500000) 
df = clean_data(df)

df = feature_engineering(df)

# LABEL ENCODING
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# SPLIT
X = df.drop('Label', axis=1).astype('float32')
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)

# ============================================================
# 5. FEATURE SELECTION (LIGHTER)
# ============================================================
print("\nSelecting important features...")

temp_model = XGBClassifier(
    tree_method="hist",
    n_estimators=100,   # reduced
    max_depth=6
)

temp_model.fit(X_train, y_train)

importance = temp_model.feature_importances_

top_feature_count = max(1, int(np.ceil(len(X.columns) * 0.25)))
top_feature_indices = np.argsort(importance)[::-1][:top_feature_count]
important_features = X.columns[top_feature_indices]

X_train = X_train[important_features]
X_test = X_test[important_features]

print("Selected features:", len(important_features))
print("\nTop 25% important features selected:")
for feature, score in zip(important_features, importance[top_feature_indices]):
    print(f"{feature}: {score:.6f}")


# ============================================================
# 6. TRAIN MODEL (OPTIMIZED)
# ============================================================
print("\nTraining XGBoost...")

model = XGBClassifier(
    tree_method="hist",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)


# ============================================================
# 7. CALIBRATION (FIXED MEMORY ISSUE)
# ============================================================
print("\nCalibrating...")

# ⚠️ Use SMALL SAMPLE to avoid RAM crash
sample_idx = np.random.choice(len(X_train), size=min(100000, len(X_train)), replace=False)

train_probs = model.predict_proba(X_train.iloc[sample_idx])
y_sample = y_train.iloc[sample_idx]

test_probs = model.predict_proba(X_test)

calibrator = LogisticRegression(max_iter=500)
calibrator.fit(train_probs, y_sample)

final_probs = calibrator.predict_proba(test_probs)
y_pred = np.argmax(final_probs, axis=1)


# ============================================================
# 8. EVALUATION
# ============================================================
print("\n📊 TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Weighted F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\n📌 LABEL MAPPING")
for i, label in enumerate(le.classes_):
    print(f"{i} → {label}")

print("\n📌 PER-CLASS PERFORMANCE")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))


# ============================================================
# 9. SAVE
# ============================================================
joblib.dump(model, "xgb_flow_model.pkl")
joblib.dump(calibrator, "calibrator_lr.pkl")
joblib.dump(important_features.tolist(), "important_features.pkl")
joblib.dump(le.classes_, "labels.pkl")

print("\n✔ Models saved successfully!")