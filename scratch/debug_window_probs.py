import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
import joblib

from nids_dashboard import COL_RENAME_2017
from session_level.train_session_cnn_lstm import SessionCNNLSTM, SESSION_FEATURES

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=300, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

# Preprocessing
for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
    if w_col in df.columns:
        df[w_col] = pd.to_numeric(df[w_col], errors='coerce').fillna(0).clip(lower=0)

durations = pd.to_numeric(df.get('Flow Duration', 1000), errors='coerce').fillna(1000).clip(lower=50) / 1e6
cum_time = np.cumsum(durations.values)
t_base = pd.Timestamp("2018-02-15 09:00:00")
df["Timestamp"] = [t_base + pd.Timedelta(seconds=float(s)) for s in cum_time]

scaler = joblib.load("session_level/session_scaler.pkl")
le = joblib.load("session_level/session_labels.pkl")
classes = list(le.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SessionCNNLSTM(in_features=27, n_classes=len(classes)).to(device)
model.load_state_dict(torch.load("session_level/session_cnn_lstm.pth", map_location=device))
model.eval()

# Process window 1 (flows 60 to 120)
w = df.iloc[60:120].copy()
ts = pd.to_datetime(w["Timestamp"], errors="coerce")
w["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)

for f in SESSION_FEATURES:
    w[f] = pd.to_numeric(w[f], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

feat_matrix = w[SESSION_FEATURES].values.astype(np.float32)
seq_len = 30
step = 15
session_tensors = []
for start_idx in range(0, len(feat_matrix) - seq_len + 1, step):
    session_tensors.append(feat_matrix[start_idx : start_idx + seq_len])

X_sess = np.array(session_tensors, dtype=np.float32)
N_s, T, D = X_sess.shape
X_sess_sc = scaler.transform(X_sess.reshape(-1, D)).reshape(N_s, T, D)

with torch.no_grad():
    logits = model(torch.tensor(X_sess_sc, dtype=torch.float32).to(device))
    t_val = joblib.load("session_level/session_temperature.pkl")["temperature"]
    probs = torch.softmax(logits / t_val, dim=-1).cpu().numpy()

print(f"Extracted {N_s} sessions from Window 1:")
for s_i in range(N_s):
    top_c = classes[np.argmax(probs[s_i])]
    benign_p = probs[s_i, classes.index("Benign")]
    print(f"  Session {s_i}: Top={top_c} ({probs[s_i].max():.3f}), Benign P={benign_p:.3f}")

