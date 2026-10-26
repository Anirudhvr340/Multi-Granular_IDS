import sys
from pathlib import Path
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

print("=" * 80)
print("OVERFITTING & GENERALIZATION AUDIT: TRAIN VS TEST HOLDOUT EVALUATION")
print("=" * 80)

# 1. Packet Model
print("\n--- 1. Packet 1D-CNN Evaluation ---")
from train_pytorch_packet_1d_cnn import Packet1DCNN, extract_dimensionless_packet_features
import joblib

pkt_scaler = joblib.load("packet_level/cnn_scaler.pkl")
pkt_le = joblib.load("packet_level/cnn_labels.pkl")
pkt_feats = joblib.load("packet_level/cnn_packet_features.pkl")
pkt_temp = joblib.load("packet_level/cnn_temperature.pkl")["temperature"]

df_pkt = pd.read_csv("data_cache/packet_train_balanced.csv")
X_p_raw, _ = extract_dimensionless_packet_features(df_pkt)
X_p = X_p_raw[pkt_feats].values.astype(np.float32)
X_p = np.nan_to_num(X_p, nan=0.0, posinf=0.0, neginf=0.0)
y_p = pkt_le.transform(df_pkt["Label"])

X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(X_p, y_p, test_size=0.30, random_state=42, stratify=y_p)
X_p_tr_sc = pkt_scaler.transform(X_p_tr)
X_p_te_sc = pkt_scaler.transform(X_p_te)

pkt_model = Packet1DCNN(in_features=len(pkt_feats), n_classes=len(pkt_le.classes_))
pkt_model.load_state_dict(torch.load("packet_level/cnn_packet_model.pth", map_location="cuda"))
pkt_model.to("cuda")
pkt_model.eval()

with torch.no_grad():
    p_tr_logits = pkt_model(torch.tensor(X_p_tr_sc, dtype=torch.float32).cuda())
    p_tr_pred = torch.argmax(p_tr_logits, dim=-1).cpu().numpy()
    
    p_te_logits = pkt_model(torch.tensor(X_p_te_sc, dtype=torch.float32).cuda())
    p_te_pred = torch.argmax(p_te_logits, dim=-1).cpu().numpy()

p_tr_acc = accuracy_score(y_p_tr, p_tr_pred)
p_te_acc = accuracy_score(y_p_te, p_te_pred)
p_tr_f1 = f1_score(y_p_tr, p_tr_pred, average="weighted", zero_division=0)
p_te_f1 = f1_score(y_p_te, p_te_pred, average="weighted", zero_division=0)

print(f"Packet Train Accuracy: {p_tr_acc*100:.2f}% | Train Weighted F1: {p_tr_f1*100:.2f}%")
print(f"Packet Test Accuracy:  {p_te_acc*100:.2f}% | Test Weighted F1:  {p_te_f1*100:.2f}%")
print(f"Generalization Gap:    {abs(p_tr_acc - p_te_acc)*100:.2f}% (Industry standard target: < 5%)")

# 2. Session Model
print("\n--- 2. Session CNN-LSTM Evaluation ---")
from nids.adapters.session_adapter import SessionCNNLSTM

sess_scaler = joblib.load("session_level/session_scaler.pkl")
sess_le = joblib.load("session_level/session_labels.pkl")
sess_cfg = joblib.load("session_level/session_config.pkl")
sess_temp = joblib.load("session_level/session_temperature.pkl")["temperature"]

s_data = np.load("data_cache/session_train_augmented.npz", allow_pickle=True)
X_s_orig, y_s_orig = s_data["X"].astype(np.float32), s_data["y"]

# Reconstruct stratified split
counts = Counter(y_s_orig)
X_list, y_list = list(X_s_orig), list(y_s_orig)
rng = np.random.default_rng(42)
for cls, cnt in counts.items():
    if cnt < 30:
        idxs = [i for i, label in enumerate(y_s_orig) if label == cls]
        needed = 30 - cnt
        extras = rng.choice(idxs, needed, replace=True)
        for ei in extras:
            X_list.append(X_s_orig[ei])
            y_list.append(cls)

X_s = np.array(X_list, dtype=np.float32)
y_s = sess_le.transform(y_list)

X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(X_s, y_s, test_size=0.30, random_state=42, stratify=y_s)
N_tr, T, D = X_s_tr.shape
N_te = X_s_te.shape[0]
X_s_tr_sc = sess_scaler.transform(X_s_tr.reshape(-1, D)).reshape(N_tr, T, D)
X_s_te_sc = sess_scaler.transform(X_s_te.reshape(-1, D)).reshape(N_te, T, D)

sess_model = SessionCNNLSTM(
    in_features=D, n_classes=len(sess_le.classes_),
    conv1_out=64, conv2_out=128, lstm_hidden=128, fc_out=256,
    bidirectional=True
)
sess_model.load_state_dict(torch.load("session_level/session_cnn_lstm.pth", map_location="cuda"))
sess_model.to("cuda")
sess_model.eval()

with torch.no_grad():
    s_tr_logits = sess_model(torch.tensor(X_s_tr_sc, dtype=torch.float32).cuda())
    s_tr_pred = torch.argmax(s_tr_logits, dim=-1).cpu().numpy()
    
    s_te_logits = sess_model(torch.tensor(X_s_te_sc, dtype=torch.float32).cuda())
    s_te_pred = torch.argmax(s_te_logits, dim=-1).cpu().numpy()

s_tr_acc = accuracy_score(y_s_tr, s_tr_pred)
s_te_acc = accuracy_score(y_s_te, s_te_pred)
s_tr_f1 = f1_score(y_s_tr, s_tr_pred, average="weighted", zero_division=0)
s_te_f1 = f1_score(y_s_te, s_te_pred, average="weighted", zero_division=0)

print(f"Session Train Accuracy: {s_tr_acc*100:.2f}% | Train Weighted F1: {s_tr_f1*100:.2f}%")
print(f"Session Test Accuracy:  {s_te_acc*100:.2f}% | Test Weighted F1:  {s_te_f1*100:.2f}%")
print(f"Generalization Gap:     {abs(s_tr_acc - s_te_acc)*100:.2f}% (Industry standard target: < 5%)")

