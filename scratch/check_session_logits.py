import sys
sys.path.insert(0, '.')
import torch
import joblib
import numpy as np
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.session_adapter import SessionModelAdapter, SessionCNNLSTM, SESSION_FEATURES

ad = SessionModelAdapter()
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
X_sess = np.array([X[:30], X[15:45], X[30:60]], dtype=np.float32)
X_sc = ad.scaler.transform(X_sess.reshape(-1, 27)).reshape(3, 30, 27)

with torch.no_grad():
    tx = torch.tensor(X_sc, dtype=torch.float32).to(ad.device)
    logits = ad.model(tx)
    probs = torch.softmax(logits / ad.temperature, dim=-1).cpu().numpy()

for i, p in enumerate(probs):
    top_indices = np.argsort(p)[::-1][:5]
    print(f"Session {i}:")
    for idx in top_indices:
        print(f"  {ad.classes[idx]}: {p[idx]:.4f}")

