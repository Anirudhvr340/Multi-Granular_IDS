"""
Packet-Level Model Adapter (Keras 3 1D-CNN)
===========================================
Ingests per-flow traffic DataFrames within a time window, derives 27 dimensionless
packet-header and TCP flag symmetry ratios, and applies the trained regularized 1D-CNN
plus Platt scaling calibrator.

Academic Citations:
  - Lotfollahi et al. (2020) — "Deep Packet: A Novel Deep Learning Approach for Encrypted Traffic Classification"
  - Wang et al. (2017) — "End-to-end Traffic Classification with 1D-CNN"
  - Platt (1999) — "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"
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

os.environ.setdefault("KERAS_BACKEND", "torch")
try:
    import keras
except ImportError:
    import tensorflow.keras as keras

from nids.adapters.base_adapter import BaseAdapter
from nids.evidence_schema import (
    EvidenceRecord, STATUS_OK, STATUS_ERROR, GRANULARITY_PACKET, normalise_family
)

HERE = Path(__file__).resolve().parent.parent.parent
PACKET_DIR = HERE / "packet_level"
if str(PACKET_DIR) not in sys.path:
    sys.path.insert(0, str(PACKET_DIR))

from train_packet_1d_cnn import extract_dimensionless_packet_features


class PacketModelAdapter(BaseAdapter):
    """
    Adapter for the Packet-Level 1D-CNN model.
    """

    MODEL_NAME   = "Packet 1D-CNN"
    GRANULARITY  = GRANULARITY_PACKET
    BENIGN_LABEL = "Benign"

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        attack_threshold: float = 0.35,
    ):
        super().__init__(attack_threshold)
        self.model_dir = Path(model_dir) if model_dir else PACKET_DIR
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.model_dir / "cnn_packet_model.keras"
        scaler_path = self.model_dir / "cnn_scaler.pkl"
        labels_path = self.model_dir / "cnn_labels.pkl"
        features_path = self.model_dir / "cnn_packet_features.pkl"
        calibrator_path = self.model_dir / "cnn_calibrator_lr.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Packet model not found at {model_path}")

        self.model = keras.models.load_model(str(model_path))
        self.scaler = joblib.load(scaler_path)
        self.le = joblib.load(labels_path)
        self.features = joblib.load(features_path)
        self.calibrator = joblib.load(calibrator_path)
        if not hasattr(self.calibrator, "multi_class"):
            self.calibrator.multi_class = "deprecated"

        self.classes = [str(c) for c in self.le.classes_]
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
            raise TypeError(f"PacketModelAdapter accepts DataFrame or dict, got {type(input_data)}")

        feats_df, _ = extract_dimensionless_packet_features(df)
        for col in self.features:
            if col not in feats_df.columns:
                feats_df[col] = 0.0

        X = feats_df[self.features].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale and reshape for 1D-CNN
        X_sc = self.scaler.transform(X)
        X_cnn = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)

        raw_probs = self.model.predict(X_cnn, batch_size=512, verbose=0)
        cal_probs = self.calibrator.predict_proba(raw_probs)
        mean_class_probs = cal_probs.mean(axis=0)

        # Extract attack probability per packet/flow: 1 - P(benign)
        per_pkt_attack = 1.0 - cal_probs[:, self._benign_idx]
        mean_benign = float(cal_probs[:, self._benign_idx].mean())
        mean_attack = float(np.mean(per_pkt_attack))
        p90_attack = float(np.percentile(per_pkt_attack, 90))
        p95_attack = float(np.percentile(per_pkt_attack, 95))
        frac_50 = float(np.mean(per_pkt_attack >= 0.50))
        frac_70 = float(np.mean(per_pkt_attack >= 0.70))
        frac_attack = float(np.mean(per_pkt_attack >= self.ATTACK_THRESHOLD))

        # Density-aware window pooling: require genuine attack density to override benign majority
        if mean_benign >= 0.50 and frac_70 < 0.20:
            window_p_attack = min(mean_attack, 0.25)
            is_attack = False
            pred_idx = self._benign_idx
        elif frac_70 >= 0.20 or (p95_attack >= 0.90 and frac_50 >= 0.15):
            window_p_attack = max(p90_attack, p95_attack)
            is_attack = True
            attack_class_probs = cal_probs.mean(axis=0).copy()
            attack_class_probs[self._benign_idx] = -1.0
            pred_idx = int(np.argmax(attack_class_probs))
        else:
            window_p_attack = float(np.clip(0.5 * p90_attack + 0.5 * mean_attack, 0.0, 1.0))
            is_attack = window_p_attack >= self.ATTACK_THRESHOLD
            if is_attack:
                attack_class_probs = cal_probs.mean(axis=0).copy()
                attack_class_probs[self._benign_idx] = -1.0
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
                "packets_evaluated": len(df),
                "p95_attack_prob": round(p95_attack, 4),
                "p90_attack_prob": round(p90_attack, 4),
                "mean_attack_prob": round(mean_attack, 4),
                "fraction_attack_packets": round(frac_attack, 4),
            },
        )

    @property
    def trained_classes(self) -> list:
        return self.classes

    @property
    def feature_names(self) -> list:
        return self.features
