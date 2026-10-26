import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import joblib
import numpy as np
import torch

from train_pytorch_packet_1d_cnn import Packet1DCNN, extract_dimensionless_packet_features

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1000)
df.columns = df.columns.str.strip()

feats, _ = extract_dimensionless_packet_features(df)
scaler = joblib.load("packet_level/cnn_scaler.pkl")
le = joblib.load("packet_level/cnn_labels.pkl")
t_dict = joblib.load("packet_level/cnn_temperature.pkl")
T = t_dict["temperature"]

model = Packet1DCNN(in_features=27, n_classes=15)
model.load_state_dict(torch.load("packet_level/cnn_packet_model.pth"))
model.eval()

X = scaler.transform(feats.values)
with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits / T, dim=-1).numpy()
    preds = le.inverse_transform(np.argmax(probs, axis=1))

print("New PyTorch Packet 1D-CNN Evaluation on 1000 Monday Benign flows:")
print("Benign probability mean:", probs[:, 0].mean())
print("Prediction counts:")
print(pd.Series(preds).value_counts())
