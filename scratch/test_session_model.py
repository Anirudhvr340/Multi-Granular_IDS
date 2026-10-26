import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np
import torch
from session_level.train_session_cnn_lstm import SessionCNNLSTM

# Load balanced data
data = np.load("data_cache/session_train_balanced.npz")
X = data["X"]
y = data["y"]
print(f"Loaded {len(X)} session sequences of shape {X.shape[1:]}")
print("Class counts in session data:")
import pandas as pd
print(pd.Series(y).value_counts())

le = joblib.load("session_level/session_labels.pkl")
classes = list(le.classes_)
print(f"Session model classes ({len(classes)}): {classes}")

scaler = joblib.load("session_level/session_scaler.pkl")
model = SessionCNNLSTM(in_features=27, n_classes=len(classes))
model.load_state_dict(torch.load("session_level/session_cnn_lstm.pth"))
model.eval()

# Test benign sequences
benign_mask = (y == "Benign")
X_benign = X[benign_mask][:500]
N, T, D = X_benign.shape
X_benign_sc = scaler.transform(X_benign.reshape(-1, D)).reshape(N, T, D)

with torch.no_grad():
    logits = model(torch.tensor(X_benign_sc, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = le.inverse_transform(np.argmax(probs, axis=1))

print("\nPredictions on 500 Benign sequences from 2018 data cache:")
print(pd.Series(preds).value_counts())

