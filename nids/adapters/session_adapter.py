"""
Session-Level Model Adapter (PyTorch CNN-LSTM)
==============================================
Ingests multi-flow sequences within a time window, constructs chronological session
tensors (T = 30), and runs the trained PyTorch 1D-CNN + LSTM sequence model.

Academic Citations:
  - Hwang et al. (2020) — "Packet-Header and Session-Sequence Deep Learning for Network Intrusion Detection"
  - Du et al. (2017) — "DeepLog: Anomaly Detection and System Log Analysis Using Deep Sequence Models"
  - Radford et al. (2018) — "Network Traffic Anomaly Detection Using Recurrent Neural Networks"
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
if str(SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(SESSION_DIR))

from train_session_cnn_lstm import SessionCNNLSTM, SESSION_FEATURES, SEQUENCE_LENGTH


class SessionModelAdapter(BaseAdapter):
    """
    Adapter for the Session-Level PyTorch CNN-LSTM model.
    """

    MODEL_NAME   = "Session CNN-LSTM"
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
        model_path = self.model_dir / "session_cnn_lstm.pth"
        scaler_path = self.model_dir / "session_scaler.pkl"
        labels_path = self.model_dir / "session_labels.pkl"
        features_path = self.model_dir / "session_features.pkl"
        config_path = self.model_dir / "session_config.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Session model not found at {model_path}")

        self.le = joblib.load(labels_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        self.classes = [str(c) for c in self.le.classes_]

        self.model = SessionCNNLSTM(
            in_features=len(self.features),
            n_classes=len(self.classes),
            conv1_out=64,
            conv2_out=128,
            lstm_hidden=128,
            fc_out=128,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._benign_idx = (
            self.classes.index(self.BENIGN_LABEL)
            if self.BENIGN_LABEL in self.classes
            else 0
        )

    def _run_inference(self, input_data: Any) -> EvidenceRecord:
        if input_data is None or (isinstance(input_data, pd.DataFrame) and input_data.empty):
            return EvidenceRecord(
                model_name=self.MODEL_NAME,
                granularity=self.GRANULARITY,
                status=STATUS_OK,
                prediction=self.BENIGN_LABEL,
                confidence=1.0,
                attack_probability=0.0,
                is_attack=False,
                class_probabilities={c: (1.0 if c == self.BENIGN_LABEL else 0.0) for c in self.classes},
                top_predictions=[(self.BENIGN_LABEL, 1.0)],
                original_prediction=self.BENIGN_LABEL,
                normalised_family=None,
                supporting_evidence={"reason": "empty_window"},
            )

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"SessionModelAdapter accepts DataFrame or dict, got {type(input_data)}")

        # Ensure numeric features
        for col in self.features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif col != "Time_Delta":
                df[col] = 0.0

        if "Timestamp" in df.columns and "Time_Delta" not in df.columns:
            ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(df["Timestamp"], errors="coerce")
            ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
            df["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)
        elif "Time_Delta" not in df.columns:
            df["Time_Delta"] = 0.1

        num_cols = [c for c in self.features if c in df.columns]
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        feat_matrix = df[self.features].values.astype(np.float32)
        n_flows = len(feat_matrix)
        seq_len = SEQUENCE_LENGTH  # 30

        # Sparse window handling (< 5 flows cannot form a multi-flow temporal session)
        if n_flows < 5:
            return EvidenceRecord(
                model_name=self.MODEL_NAME,
                granularity=self.GRANULARITY,
                status=STATUS_OK,
                prediction=self.BENIGN_LABEL,
                confidence=0.95,
                attack_probability=0.05,
                is_attack=False,
                class_probabilities={c: (0.95 if c == self.BENIGN_LABEL else 0.05 / max(1, len(self.classes) - 1)) for c in self.classes},
                top_predictions=[(self.BENIGN_LABEL, 0.95)],
                original_prediction=self.BENIGN_LABEL,
                normalised_family=None,
                supporting_evidence={
                    "sessions_evaluated": 0,
                    "flows_in_window": n_flows,
                    "p95_attack_prob": 0.05,
                    "mean_attack_prob": 0.05,
                    "fraction_attack_sessions": 0.0,
                    "reason": "sparse_flows_below_session_threshold",
                },
            )

        session_tensors = []
        if n_flows < seq_len:
            # Replicate flows rather than padding with zero vectors to avoid out-of-distribution scaling spikes
            repeats = int(np.ceil(seq_len / n_flows))
            padded = np.tile(feat_matrix, (repeats, 1))[:seq_len]
            session_tensors.append(padded)
        else:
            # Stride 15, capped at max 40 sequences per window for performance
            step = max(1, seq_len // 2)
            for start_idx in range(0, n_flows - seq_len + 1, step):
                session_tensors.append(feat_matrix[start_idx:start_idx + seq_len])
                if len(session_tensors) >= 40:
                    break

        X_sess = np.array(session_tensors, dtype=np.float32)
        N_s, T, D = X_sess.shape
        X_sess_sc = self.scaler.transform(X_sess.reshape(-1, D)).reshape(N_s, T, D)

        with torch.no_grad():
            tensor_x = torch.tensor(X_sess_sc, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            mean_class_probs = probs.mean(axis=0)

        per_sess_attack = 1.0 - probs[:, self._benign_idx]
        mean_benign = float(probs[:, self._benign_idx].mean())
        mean_attack = float(np.mean(per_sess_attack))
        p90_attack = float(np.percentile(per_sess_attack, 90))
        p95_attack = float(np.percentile(per_sess_attack, 95))
        frac_50 = float(np.mean(per_sess_attack >= 0.50))
        frac_70 = float(np.mean(per_sess_attack >= 0.70))
        frac_attack = float(np.mean(per_sess_attack >= self.ATTACK_THRESHOLD))

        # Density-aware session pooling: require multiple attack sessions before overriding benign majority
        if mean_benign >= 0.50 and frac_70 < 0.25:
            window_p_attack = min(mean_attack, 0.25)
            is_attack = False
            pred_idx = self._benign_idx
        elif frac_70 >= 0.25 or (p95_attack >= 0.90 and frac_50 >= 0.20):
            window_p_attack = p95_attack
            is_attack = True
            attack_class_probs = probs.mean(axis=0).copy()
            attack_class_probs[self._benign_idx] = 0.0
            pred_idx = int(np.argmax(attack_class_probs))
        else:
            window_p_attack = float(np.clip(0.5 * p95_attack + 0.5 * mean_attack, 0.0, 1.0))
            is_attack = window_p_attack >= max(self.ATTACK_THRESHOLD, 0.50)
            if is_attack:
                attack_class_probs = probs.mean(axis=0).copy()
                attack_class_probs[self._benign_idx] = 0.0
                pred_idx = int(np.argmax(attack_class_probs))
            else:
                pred_idx = self._benign_idx

        pred_label = self.classes[pred_idx]
        conf = float(mean_class_probs[pred_idx])

        class_prob_dict = {
            self.classes[i]: float(mean_class_probs[i])
            for i in range(len(self.classes))
        }

        sorted_preds = sorted(
            class_prob_dict.items(), key=lambda x: x[1], reverse=True
        )[:5]

        norm_family = normalise_family(pred_label)

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
            normalised_family=norm_family,
            supporting_evidence={
                "sessions_evaluated": N_s,
                "flows_in_window": n_flows,
                "p95_attack_prob": round(p95_attack, 4),
                "mean_attack_prob": round(mean_attack, 4),
                "fraction_attack_sessions": round(frac_attack, 4),
            },
        )

    @property
    def trained_classes(self) -> list:
        return self.classes

    @property
    def feature_names(self) -> list:
        return self.features
