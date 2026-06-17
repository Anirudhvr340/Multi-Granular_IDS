import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ==============================
# CONFIG
# ==============================
DATA_PATH = "."
OUTPUT_DIR = "processed"

WINDOW_SECONDS = 2   # 🔥 IMPORTANT (keep this)
FIXED_WINDOW_SIZE = 200
MAX_ROWS_PER_FILE = 500_000   # limit per file

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
print("🔹 Loading CSV files...")

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
print("Using files:", files)

df_list = []

for file in files:
    print(f"\n🔸 Loading file: {file}")

    temp = pd.read_csv(file, low_memory=False)

    temp.drop_duplicates(inplace=True)
    temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp.dropna(inplace=True)

    print(f"    Original shape: {temp.shape}")

    # 🔥 RANDOM SAMPLE (NOT HEAD)
    if len(temp) > MAX_ROWS_PER_FILE:
        temp = temp.sample(MAX_ROWS_PER_FILE, random_state=42)

    print(f"    Sampled shape: {temp.shape}")

    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print(f"✅ Total rows used: {len(df)}")

# ==============================
# TIMESTAMP
# ==============================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)
df = df.sort_values('Timestamp')

# ==============================
# FEATURES
# ==============================
label_col = "Label"

features = df.drop(columns=[label_col, 'Timestamp'], errors='ignore')

features = features.apply(pd.to_numeric, errors='coerce')
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(axis=1, how='all', inplace=True)
features.fillna(0, inplace=True)

df = pd.concat([features, df[[label_col, 'Timestamp']]], axis=1)

# ==============================
# CLASS DISTRIBUTION
# ==============================
print("\n🔹 Class distribution BEFORE windowing:")
print(df[label_col].value_counts())

# ==============================
# ENCODE LABELS
# ==============================
le = LabelEncoder()
df[label_col] = le.fit_transform(df[label_col])

class_names = le.classes_
np.save(os.path.join(OUTPUT_DIR, "class_names.npy"), class_names)

# ==============================
# SCALE
# ==============================
X = df.drop(columns=[label_col, 'Timestamp']).values
y = df[label_col].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X)
df_scaled['Label'] = y
df_scaled['Timestamp'] = df['Timestamp'].values

# ==============================
# WINDOWING
# ==============================
print("\n🔹 Creating time-based windows...")

df_scaled = df_scaled.sort_values('Timestamp')
df_scaled.set_index('Timestamp', inplace=True)

groups = df_scaled.groupby(pd.Grouper(freq=f'{WINDOW_SECONDS}s'))

X_windows = []
y_windows = []

for i, (_, window_df) in enumerate(groups):
    if len(window_df) == 0:
        continue

    features = window_df.drop(columns=['Label']).values

    if len(features) >= FIXED_WINDOW_SIZE:
        features = features[:FIXED_WINDOW_SIZE]
    else:
        pad = np.zeros((FIXED_WINDOW_SIZE - len(features), features.shape[1]))
        features = np.vstack([features, pad])

    X_windows.append(features)
    y_windows.append(window_df['Label'].values[-1])

    if (i + 1) % 500 == 0:
        print(f"{i+1} windows processed")

X_windows = np.array(X_windows, dtype=np.float32)
y_windows = np.array(y_windows, dtype=np.int64)

print(f"\n✅ Window shape: {X_windows.shape}")

# ==============================
# FINAL DISTRIBUTION
# ==============================
print("\n🔹 Class distribution AFTER windowing:")

unique, counts = np.unique(y_windows, return_counts=True)
for u, c in zip(unique, counts):
    print(f"class {u} ({class_names[u]}): {c}")

# ==============================
# SAVE
# ==============================
np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
np.save(os.path.join(OUTPUT_DIR, "y_windows.npy"), y_windows)

print("\n✅ Step 0 COMPLETED 🚀")