"""
Session-Level Model Adapter (PyTorch Bi-LSTM + Self-Attention CNN)
==================================================================
Ingests multi-flow sequences within a time window, constructs chronological
session tensors (T=30), and runs the trained PyTorch Bi-LSTM model with
Temperature Scaling probability calibration.

Research Citations:
  - Hwang et al. (2020) — Session-Sequence Deep Learning for NIDS (IEEE Access)
  - Yin et al. (2017) — RNN-IDS (IEEE TIFS)
  - Guo et al. (2017) — On Calibration of Modern Neural Networks (ICML)
  - Lin et al. (2017) — Focal Loss for Dense Object Detection (ICCV)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nids.adapters.base_adapter import BaseAdapter
from nids.evidence_schema import (
    EvidenceRecord, STATUS_OK, STATUS_ERROR, GRANULARITY_SESSION, normalise_family
)

HERE = Path(__file__).resolve().parent.parent.parent
SESSION_DIR = HERE / "session_level"

SEQUENCE_LENGTH = 30

SESSION_FEATURES = [
    'Flow Duration', 'Flow Pkts/s', 'Flow Byts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std',
    'Down/Up Ratio', 'Fwd Header Len', 'Bwd Header Len', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Time_Delta'
]


class SessionCNNLSTM(nn.Module):
    """
    1D-CNN + Bidirectional LSTM + Self-Attention hybrid.
    Tries to load the new BiLSTM architecture first; falls back to simple
    architecture if checkpoint was trained with the older model.
    """
    def __init__(self, in_features=27, n_classes=15,
                 conv1_out=64, conv2_out=128,
                 lstm_hidden=128, fc_out=256,
                 dropout_conv=0.2, dropout_fc=0.3,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional

        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=conv1_out,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_conv)

        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_conv)

        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=bidirectional
        )

        lstm_output_size = lstm_hidden * (2 if bidirectional else 1)

        self.attn_fc = nn.Linear(lstm_output_size, 1)

        self.fc1 = nn.Linear(lstm_output_size, fc_out)
        self.bn_fc = nn.BatchNorm1d(fc_out)
        self.relu3 = nn.ReLU()
        self.drop_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(fc_out, 128)
        self.relu4 = nn.ReLU()
        self.drop_fc2 = nn.Dropout(dropout_fc)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        c1 = self.drop1(self.relu1(self.bn1(self.conv1(x_t))))
        c2 = self.drop2(self.relu2(self.bn2(self.conv2(c1))))
        c2_trans = c2.transpose(1, 2)
        lstm_out, _ = self.lstm(c2_trans)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        h = self.drop_fc(self.relu3(self.bn_fc(self.fc1(context))))
        h2 = self.drop_fc2(self.relu4(self.fc2(h)))
        return self.out(h2)


class SessionModelAdapter(BaseAdapter):
    """
    Adapter for the Session-Level PyTorch Bi-LSTM model.
    Automatically detects model architecture from config.
    """

    MODEL_NAME   = "Session CNN-LSTM (Sequential)"
    GRANULARITY  = GRANULARITY_SESSION
    BENIGN_LABEL = "Benign"

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        attack_threshold: float = 0.35,
    ):
        super().__init__(attack_threshold)
        self.model_dir = Path(model_dir) if model_dir else SESSION_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()

    def _load_artifacts(self):
        model_path   = self.model_dir / "session_cnn_lstm.pth"
        scaler_path  = self.model_dir / "session_scaler.pkl"
        labels_path  = self.model_dir / "session_labels.pkl"
        features_path= self.model_dir / "session_features.pkl"
        config_path  = self.model_dir / "session_config.pkl"
        temp_path    = self.model_dir / "session_temperature.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Session model not found at {model_path}")

        self.le = joblib.load(labels_path)
        self.scaler = joblib.load(scaler_path)

        # Load features; fall back to SESSION_FEATURES constant
        try:
            self.features = joblib.load(features_path)
        except Exception:
            self.features = SESSION_FEATURES

        self.classes = [str(c) for c in self.le.classes_]

        if temp_path.exists():
            t_data = joblib.load(temp_path)
            self.temperature = float(t_data.get("temperature", 1.0))
        else:
            self.temperature = 1.0

        # Detect architecture from config
        bidirectional = True
        fc_out = 256
        if config_path.exists():
            try:
                cfg = joblib.load(config_path)
                bidirectional = cfg.get("bidirectional", True)
                fc_out = cfg.get("fc_out", 256)
            except Exception:
                pass

        # Try BiLSTM architecture first, fall back to unidirectional if mismatch
        for bi in ([bidirectional, not bidirectional] if bidirectional else [False, True]):
            try:
                model = SessionCNNLSTM(
                    in_features=len(self.features),
                    n_classes=len(self.classes),
                    conv1_out=64, conv2_out=128,
                    lstm_hidden=128, fc_out=fc_out,
                    bidirectional=bi
                ).to(self.device)
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                self.model = model
                break
            except RuntimeError:
                continue

        self._benign_idx = (
            self.classes.index(self.BENIGN_LABEL)
            if self.BENIGN_LABEL in self.classes else 0
        )

    def _run_inference(self, input_data: Any) -> EvidenceRecord:
        if input_data is None or (isinstance(input_data, pd.DataFrame) and input_data.empty):
            return self._benign_record()

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"SessionModelAdapter expects DataFrame or dict, got {type(input_data)}")

        df.columns = df.columns.str.strip()

        # Clean numeric features
        for col in self.features:
            if col in df.columns:
                df[col] = (pd.to_numeric(df[col], errors="coerce")
                           .replace([np.inf, -np.inf], np.nan)
                           .fillna(0.0))
            elif col != "Time_Delta":
                df[col] = 0.0

        # Clip window sizes and negative values
        for c in ['Init Fwd Win Byts', 'Init Bwd Win Byts', 'Flow IAT Min',
                  'Fwd IAT Min', 'Bwd IAT Min']:
            if c in df.columns:
                df[c] = df[c].clip(lower=0.0)

        # Compute Time_Delta if missing
        if "Timestamp" in df.columns and "Time_Delta" not in df.columns:
            ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(df["Timestamp"], errors="coerce")
            ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
            df["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(0.0, 300.0)
        elif "Time_Delta" not in df.columns:
            # Synthetic time delta from flow durations
            if "Flow Duration" in df.columns:
                durations = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(1000).clip(50) / 1e6
                deltas = np.concatenate([[0.0], np.diff(np.cumsum(durations.values))])
                df["Time_Delta"] = np.clip(deltas, 0.0, 300.0)
            else:
                df["Time_Delta"] = 0.1

        feat_matrix = df[self.features].values.astype(np.float32)
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        n_flows = len(feat_matrix)
        seq_len = SEQUENCE_LENGTH

        if n_flows < 5:
            return self._sparse_record(n_flows)

        # Build sliding-window session tensors
        session_tensors = []
        if n_flows < seq_len:
            repeats = int(np.ceil(seq_len / n_flows))
            padded = np.tile(feat_matrix, (repeats, 1))[:seq_len]
            session_tensors.append(padded)
        else:
            step = max(1, seq_len // 2)
            for s_i in range(0, n_flows - seq_len + 1, step):
                session_tensors.append(feat_matrix[s_i : s_i + seq_len])
                if len(session_tensors) >= 40:
                    break

        X_sess = np.array(session_tensors, dtype=np.float32)
        N_s, T, D = X_sess.shape
        X_sc = self.scaler.transform(X_sess.reshape(-1, D)).reshape(N_s, T, D)

        with torch.no_grad():
            tensor_x = torch.tensor(X_sc, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_x)
            cal_logits = logits / self.temperature
            probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()
            mean_class_probs = probs.mean(axis=0)

        per_sess_attack = 1.0 - probs[:, self._benign_idx]
        mean_benign = float(probs[:, self._benign_idx].mean())
        mean_attack = float(np.mean(per_sess_attack))
        p90_attack  = float(np.percentile(per_sess_attack, 90))
        p95_attack  = float(np.percentile(per_sess_attack, 95))
        frac_50     = float(np.mean(per_sess_attack >= 0.50))
        frac_70     = float(np.mean(per_sess_attack >= 0.70))
        frac_attack = float(np.mean(per_sess_attack >= self.ATTACK_THRESHOLD))

        # Density-aware pooling with conservative benign bias
        # Requires ≥40% of sessions to indicate attack before overriding benign
        if mean_benign >= 0.45 and frac_70 < 0.40:
            window_p_attack = min(mean_attack, 0.22)
            is_attack = False
            pred_idx = self._benign_idx
        elif frac_70 >= 0.40 or (p95_attack >= 0.88 and frac_50 >= 0.30):
            window_p_attack = p95_attack
            is_attack = True
            atk_probs = probs.mean(axis=0).copy()
            atk_probs[self._benign_idx] = 0.0
            pred_idx = int(np.argmax(atk_probs))
        else:
            window_p_attack = float(np.clip(0.5 * p95_attack + 0.5 * mean_attack, 0.0, 1.0))
            is_attack = window_p_attack >= max(self.ATTACK_THRESHOLD, 0.50)
            if is_attack:
                atk_probs = probs.mean(axis=0).copy()
                atk_probs[self._benign_idx] = 0.0
                pred_idx = int(np.argmax(atk_probs))
            else:
                pred_idx = self._benign_idx

        pred_label = self.classes[pred_idx]
        conf = float(mean_class_probs[pred_idx])
        class_prob_dict = {self.classes[i]: float(mean_class_probs[i])
                           for i in range(len(self.classes))}
        sorted_preds = sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        return EvidenceRecord(
            model_name=self.MODEL_NAME,
            granularity=self.GRANULARITY,
            status=STATUS_OK,
            prediction=pred_label,
            confidence=round(conf, 4),
            attack_probability=round(window_p_attack, 4),
            is_attack=is_attack,
            class_probabilities=class_prob_dict,
            top_predictions=sorted_preds,
            original_prediction=pred_label,
            normalised_family=normalise_family(pred_label),
            supporting_evidence={
                "sessions_evaluated": N_s,
                "flows_in_window": n_flows,
                "p95_attack_prob": round(p95_attack, 4),
                "mean_attack_prob": round(mean_attack, 4),
                "fraction_attack_sessions": round(frac_attack, 4),
            },
        )

    def _benign_record(self):
        return EvidenceRecord(
            model_name=self.MODEL_NAME, granularity=self.GRANULARITY, status=STATUS_OK,
            prediction=self.BENIGN_LABEL, confidence=1.0, attack_probability=0.0,
            is_attack=False,
            class_probabilities={c: (1.0 if c == self.BENIGN_LABEL else 0.0) for c in self.classes},
            top_predictions=[(self.BENIGN_LABEL, 1.0)],
            original_prediction=self.BENIGN_LABEL, normalised_family=None,
            supporting_evidence={"reason": "empty_window"},
        )

    def _sparse_record(self, n_flows):
        return EvidenceRecord(
            model_name=self.MODEL_NAME, granularity=self.GRANULARITY, status=STATUS_OK,
            prediction=self.BENIGN_LABEL, confidence=0.95, attack_probability=0.05,
            is_attack=False,
            class_probabilities={c: (0.95 if c == self.BENIGN_LABEL else 0.05 / max(1, len(self.classes) - 1))
                                  for c in self.classes},
            top_predictions=[(self.BENIGN_LABEL, 0.95)],
            original_prediction=self.BENIGN_LABEL, normalised_family=None,
            supporting_evidence={"sessions_evaluated": 0, "flows_in_window": n_flows,
                                 "reason": "sparse_flows_below_session_threshold"},
        )

    @property
    def trained_classes(self) -> list:
        return self.classes

    @property
    def feature_names(self) -> list:
        return self.features
