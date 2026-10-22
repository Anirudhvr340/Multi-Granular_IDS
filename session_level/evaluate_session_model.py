"""
=============================================================================
STANDALONE PYTORCH SESSION-LEVEL CNN-LSTM EVALUATOR (SESSION LEVEL)
=============================================================================
Academic Citations:
  - Hwang et al., "Packet-Header and Session-Sequence Deep Learning for Network
    Intrusion Detection", IEEE Access, Vol. 8, pp. 195744-195758, 2020.
  - Du et al., "DeepLog: Anomaly Detection and System Log Analysis Using Deep
    Sequence Models", ACM CCS, 2017.
  - Radford et al., "Network Traffic Anomaly Detection Using Recurrent Neural
    Networks", IEEE CNS, 2018.

Loads the trained PyTorch SessionCNNLSTM model, feature Scaler, and LabelEncoder,
constructs chronological multi-flow session sequences (T = 30), and outputs
Accuracy, Weighted F1, Macro F1, and complete Per-Class Classification Report.
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_session_cnn_lstm import SessionCNNLSTM, SESSION_FEATURES, SEQUENCE_LENGTH


class SessionEvaluator:
    def __init__(self, model_dir=HERE):
        print(f"[*] Loading PyTorch Session Artifacts from: {model_dir}")
        self.le = joblib.load(os.path.join(model_dir, "session_labels.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "session_scaler.pkl"))
        self.features = joblib.load(os.path.join(model_dir, "session_features.pkl"))
        self.classes = list(self.le.classes_)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SessionCNNLSTM(
            in_features=len(self.features),
            n_classes=len(self.classes),
            conv1_out=64,
            conv2_out=128,
            lstm_hidden=128,
            fc_out=128
        ).to(self.device)
        
        weights_path = os.path.join(model_dir, "session_cnn_lstm.pth")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        print(f"[+] Session CNN-LSTM Model Loaded Successfully ({len(self.classes)} classes).")

    def evaluate_file(self, csv_path: str, nrows: int = 30000):
        print("\n" + "=" * 80)
        print(f"[*] EVALUATING SESSION CNN-LSTM ON: {os.path.basename(csv_path)}")
        print("=" * 80)
        
        df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[df["Label"] != "Label"]
        
        for col in self.features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif col != "Time_Delta":
                df[col] = 0.0
                
        if "Timestamp" in df.columns:
            ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(df["Timestamp"], errors="coerce")
            ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
            df["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)
        else:
            df["Time_Delta"] = 0.1
            
        num_cols = [c for c in self.features if c in df.columns]
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Build session sequences
        feat_matrix = df[self.features].values.astype(np.float32)
        labels = df["Label"].values
        seq_len = SEQUENCE_LENGTH
        
        session_tensors = []
        session_y = []
        step = max(1, seq_len // 2)
        for start_idx in range(0, len(df) - seq_len + 1, step):
            end_idx = start_idx + seq_len
            sess_window = feat_matrix[start_idx:end_idx]
            sess_lbls = labels[start_idx:end_idx]
            
            non_benign = [lbl for lbl in sess_lbls if "Benign" not in str(lbl)]
            if non_benign:
                chosen_label = pd.Series(non_benign).mode()[0]
            else:
                chosen_label = "Benign"
                
            session_tensors.append(sess_window)
            session_y.append(chosen_label)
            
        if not session_tensors:
            print("[!] Not enough flows in file to construct sessions.")
            return
            
        X_sess = np.array(session_tensors, dtype=np.float32)
        N_s, T, D = X_sess.shape
        X_sess_sc = self.scaler.transform(X_sess.reshape(-1, D)).reshape(N_s, T, D)
        
        with torch.no_grad():
            tensor_x = torch.tensor(X_sess_sc, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_x)
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
            
        preds_labels = self.le.inverse_transform(preds_idx)
        
        acc = accuracy_score(session_y, preds_labels)
        w_f1 = f1_score(session_y, preds_labels, average="weighted", zero_division=0)
        m_f1 = f1_score(session_y, preds_labels, average="macro", zero_division=0)
        
        print(f"\n[+] EVALUATION RESULTS:")
        print(f"    - Total Sessions Evaluated:  {len(session_y):,}")
        print(f"    - Overall Session Accuracy:  {acc * 100:.2f}%")
        print(f"    - Weighted F1-Score:         {w_f1 * 100:.2f}%")
        print(f"    - Macro F1-Score:            {m_f1 * 100:.2f}%")
        
        print("\n" + "=" * 80)
        print("DETAILED PER-CLASS CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(session_y, preds_labels, digits=4, zero_division=0))


if __name__ == "__main__":
    evaluator = SessionEvaluator()
    sample_file = r"F:\Day 2 - Feb 15\csv\02-15-2018.csv"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    if os.path.exists(sample_file):
        evaluator.evaluate_file(sample_file, nrows=30000)
    else:
        print(f"[!] Please provide a valid CSV path to evaluate.")

