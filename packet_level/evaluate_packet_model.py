"""
=============================================================================
STANDALONE PACKET-LEVEL 1D-CNN EVALUATOR (PACKET LEVEL)
=============================================================================
Academic Citations:
  - Lotfollahi et al., "Deep Packet: A Novel Approach for Encrypted Traffic
    Classification Using Deep Learning", Soft Computing (Springer), 2020.
  - Wang et al., "End-to-End Encrypted Traffic Classification with One-Dimensional
    Convolutional Neural Networks", IEEE BigData, 2017.
  - Lin et al., "Payload-Byte-Level 1D-CNN for Low-Latency Intrusion Detection",
    IEEE Transactions on Network and Service Management, 2022.

Loads the trained 1D-CNN Keras model, Scaler, and Platt Calibrator, transforms
raw network traffic into scale-free dimensionless packet features, and outputs
Accuracy, Weighted F1, Macro F1, and complete Per-Class Classification Report.
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
os.environ.setdefault("KERAS_BACKEND", "torch")
try:
    import keras
except ImportError:
    import tensorflow.keras as keras

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_packet_1d_cnn import extract_dimensionless_packet_features


class Packet1DCNNEvaluator:
    def __init__(self, model_dir=HERE):
        print(f"[*] Loading Packet 1D-CNN Artifacts from: {model_dir}")
        self.model = keras.models.load_model(os.path.join(model_dir, "cnn_packet_model.keras"))
        self.scaler = joblib.load(os.path.join(model_dir, "cnn_scaler.pkl"))
        self.le = joblib.load(os.path.join(model_dir, "cnn_labels.pkl"))
        self.features = joblib.load(os.path.join(model_dir, "cnn_packet_features.pkl"))
        self.calibrator = joblib.load(os.path.join(model_dir, "cnn_calibrator_lr.pkl"))
        if not hasattr(self.calibrator, "multi_class"):
            self.calibrator.multi_class = "deprecated"
        self.classes = list(self.le.classes_)
        print(f"[+] Packet 1D-CNN Model Loaded Successfully ({len(self.classes)} classes).")

    def evaluate_file(self, csv_path: str, nrows: int = 30000):
        print("\n" + "=" * 80)
        print(f"[*] EVALUATING PACKET 1D-CNN ON: {os.path.basename(csv_path)}")
        print("=" * 80)
        
        df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[df["Label"] != "Label"]
        
        # 1. Feature Extraction
        X_df, _ = extract_dimensionless_packet_features(df)
        X = X_df[self.features].values.astype(np.float32)
        X_sc = self.scaler.transform(X)
        X_cnn = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)
        
        # 2. 1D-CNN + Platt Scaling Calibration
        raw_probs = self.model.predict(X_cnn, batch_size=256, verbose=0)
        cal_probs = self.calibrator.predict_proba(raw_probs)
        y_pred_idx = np.argmax(cal_probs, axis=1)
        y_pred = self.le.inverse_transform(y_pred_idx)
        y_true = df["Label"].values
        
        # 3. Metrics
        acc = accuracy_score(y_true, y_pred)
        w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        m_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        print(f"\n[+] EVALUATION RESULTS:")
        print(f"    - Total Packets/Flows Evaluated: {len(y_true):,}")
        print(f"    - Overall Accuracy:              {acc * 100:.2f}%")
        print(f"    - Weighted F1-Score:             {w_f1 * 100:.2f}%")
        print(f"    - Macro F1-Score:                {m_f1 * 100:.2f}%")
        
        print("\n" + "=" * 80)
        print("DETAILED PER-CLASS CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    evaluator = Packet1DCNNEvaluator()
    sample_file = r"F:\Day 2 - Feb 15\csv\02-15-2018.csv"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    if os.path.exists(sample_file):
        evaluator.evaluate_file(sample_file, nrows=30000)
    else:
        print(f"[!] Please provide a valid CSV path to evaluate.")

