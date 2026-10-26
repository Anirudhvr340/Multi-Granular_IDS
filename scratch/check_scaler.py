import sys
sys.path.insert(0, '.')
import joblib
import numpy as np
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.session_adapter import SESSION_FEATURES

scaler = joblib.load('session_level/session_scaler.pkl')
print("Scaler feature count:", len(scaler.mean_))

df = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=60, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()

for col in SESSION_FEATURES[:-1]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    else:
        df[col] = 0.0
df['Time_Delta'] = 0.1

X = df[SESSION_FEATURES].values
X_sc = scaler.transform(X)
print("X_sc mean:", np.mean(X_sc, axis=0)[:5])
print("X_sc max:", np.max(X_sc, axis=0)[:5])
print("X_sc min:", np.min(X_sc, axis=0)[:5])
