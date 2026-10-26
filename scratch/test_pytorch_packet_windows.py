import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import joblib
import numpy as np
import torch
from nids_dashboard import COL_RENAME_2017
from train_pytorch_packet_1d_cnn import Packet1DCNN, extract_dimensionless_packet_features

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1000, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

# Preprocessing
for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
    if w_col in df.columns:
        df[w_col] = pd.to_numeric(df[w_col], errors='coerce').fillna(0).clip(lower=0)
if 'TotLen Fwd Pkts' in df.columns and 'Fwd Header Len' in df.columns:
    df['TotLen Fwd Pkts'] = df['TotLen Fwd Pkts'] + df['Fwd Header Len']
if 'TotLen Bwd Pkts' in df.columns and 'Bwd Header Len' in df.columns:
    df['TotLen Bwd Pkts'] = df['TotLen Bwd Pkts'] + df['Bwd Header Len']

scaler = joblib.load("packet_level/cnn_scaler.pkl")
le = joblib.load("packet_level/cnn_labels.pkl")
t_dict = joblib.load("packet_level/cnn_temperature.pkl")
T = t_dict["temperature"]
classes = list(le.classes_)

model = Packet1DCNN(in_features=27, n_classes=15)
model.load_state_dict(torch.load("packet_level/cnn_packet_model.pth"))
model.eval()

feats_df, _ = extract_dimensionless_packet_features(df)
X = scaler.transform(feats_df.values)

with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits / T, dim=-1).numpy()

benign_idx = classes.index("Benign")
for i in range(0, 300, 60):
    w_probs = probs[i : i + 60]
    mean_benign = w_probs[:, benign_idx].mean()
    p_att = 1.0 - mean_benign
    top_class_idx = np.argmax(w_probs.mean(axis=0))
    print(f"Window {i//60}: Pred={classes[top_class_idx]}, Mean Benign Prob={mean_benign:.3f}, P(att)={p_att:.3f}")

