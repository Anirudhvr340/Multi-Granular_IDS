"""
Flow-Level XGBoost Adapter
==========================
Wraps xgb_flow_model.pkl (17-feature, 15-class flow-level classifier).

Input:   pandas DataFrame with CICIDS2018 per-flow records (one or many rows).
Output:  EvidenceRecord with flow-granularity evidence.

Trained classes (15):
  Benign, Bot, Brute Force -Web, Brute Force -XSS,
  DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP,
  DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest,
  DoS attacks-Slowloris, FTP-BruteForce, Infilteration,
  SQL Injection, SSH-Bruteforce

Dataset: CICIDS 2018 (02-14-2018.csv … 03-02-2018.csv)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nids.adapters.base_adapter import BaseAdapter
from nids.evidence_schema import (
    EvidenceRecord, STATUS_OK, GRANULARITY_FLOW, normalise_family
)

# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent.parent  # repo root


class FlowModelAdapter(BaseAdapter):
    """
    Adapter for the flow-level XGBoost model.

    Accepts per-flow DataFrames or single-row dicts.
    Uses max-pool over attack probabilities when multiple flows are provided,
    so a single attack flow in a batch is not diluted by benign ones.
    """

    MODEL_NAME    = "Flow XGBoost"
    GRANULARITY   = GRANULARITY_FLOW
    BENIGN_LABEL  = "Benign"

    # Features expected by the model (in order)
    REQUIRED_FEATURES = [
        "Tot Bwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
        "Bwd Pkt Len Max", "Bwd Pkt Len Std", "Flow IAT Min",
        "Fwd IAT Tot", "Fwd IAT Std", "Fwd IAT Min", "Fwd URG Flags",
        "Fwd Header Len", "Fwd Pkts/s", "RST Flag Cnt", "PSH Flag Cnt",
        "URG Flag Cnt", "total_packets", "burstiness",
    ]

    def __init__(
        self,
        model_path: str | None = None,
        features_path: str | None = None,
        labels_path: str | None = None,
        attack_threshold: float = 0.35,
    ):
        super().__init__(attack_threshold)
        model_path    = Path(model_path)    if model_path    else HERE / "xgb_flow_model.pkl"
        features_path = Path(features_path) if features_path else HERE / "important_features.pkl"
        labels_path   = Path(labels_path)   if labels_path   else HERE / "labels.pkl"

        self._model         = joblib.load(model_path)
        self._feature_names = list(joblib.load(features_path))
        labels_raw          = joblib.load(labels_path)
        self._labels        = labels_raw.astype(str)  # index → class name
        self._class_indices = self._model.classes_.astype(int)

        # Map model output index → label name
        self._idx_to_label = {
            int(idx): str(self._labels[idx]) for idx in self._class_indices
        }

    # -----------------------------------------------------------------------
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Align dataframe columns to the expected feature order.
        Adds total_packets and burstiness if not already present.
        """
        df = data.copy()

        # Derive total_packets if missing
        if "total_packets" not in df.columns:
            fwd = pd.to_numeric(df.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
            bwd = pd.to_numeric(df.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
            df["total_packets"] = fwd + bwd

        # Derive burstiness if missing
        if "burstiness" not in df.columns:
            df["burstiness"] = 0.0

        X = (
            df.reindex(columns=self._feature_names)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        return X

    # -----------------------------------------------------------------------
    def _run_inference(self, input_data: Any) -> EvidenceRecord:
        """
        input_data : pd.DataFrame | dict (single flow) | list[dict]
        """
        t0 = time.perf_counter()

        # Normalise input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"FlowModelAdapter expects dict/list/DataFrame, got {type(input_data)}")

        if df.empty:
            raise ValueError("Empty input DataFrame — no flows to predict.")

        X = self._prepare_features(df)
        proba_matrix = self._model.predict_proba(X)  # shape: (n_flows, n_classes)

        # ---- Fix 1 & Fix 2: Density-Aware Flow Aggregation -----------------
        n_classes = proba_matrix.shape[1]
        n_flows = len(df)

        benign_col = None
        for col_idx, label_idx in enumerate(self._class_indices):
            label = self._idx_to_label[label_idx]
            if label == self.BENIGN_LABEL:
                benign_col = col_idx
                break

        if benign_col is None:
            benign_col = 0

        # Per-flow attack probability
        flow_attack_prob = 1.0 - proba_matrix[:, benign_col]

        max_p = float(np.max(flow_attack_prob))
        mean_p = float(np.mean(flow_attack_prob))
        med_p = float(np.median(flow_attack_prob))
        p90 = float(np.percentile(flow_attack_prob, 90))
        p95 = float(np.percentile(flow_attack_prob, 95))
        p99 = float(np.percentile(flow_attack_prob, 99))
        frac_50 = float(np.mean(flow_attack_prob >= 0.50))
        frac_70 = float(np.mean(flow_attack_prob >= 0.70))
        frac_90 = float(np.mean(flow_attack_prob >= 0.90))

        # Density-aware window attack probability
        if n_flows <= 3:
            window_attack_prob = max_p if max_p >= 0.90 else mean_p
        elif p90 >= 0.90 or frac_90 >= 0.10:
            window_attack_prob = max(p95, p90)
        elif p95 >= 0.95 and frac_70 >= 0.15:
            window_attack_prob = p90
        else:
            window_attack_prob = min(mean_p, 0.20)

        window_attack_prob = float(np.clip(window_attack_prob, 0.0, 1.0))

        # Multiclass probability distribution
        class_probs: dict = {}
        if window_attack_prob >= self.ATTACK_THRESHOLD:
            # Filter to malicious flows (or top flow if none >= 0.50)
            attack_flow_mask = flow_attack_prob >= 0.50
            if not np.any(attack_flow_mask):
                attack_flow_mask = np.zeros(n_flows, dtype=bool)
                attack_flow_mask[np.argmax(flow_attack_prob)] = True

            mean_attack_flow_probs = proba_matrix[attack_flow_mask].mean(axis=0)
            # Remove benign component and re-normalize attack probabilities
            mean_attack_flow_probs[benign_col] = 0.0
            sum_att = mean_attack_flow_probs.sum()
            if sum_att > 0:
                mean_attack_flow_probs /= sum_att

            for col_idx, label_idx in enumerate(self._class_indices):
                label = self._idx_to_label[label_idx]
                if col_idx == benign_col:
                    class_probs[label] = 1.0 - window_attack_prob
                else:
                    class_probs[label] = float(window_attack_prob * mean_attack_flow_probs[col_idx])

            # Top label among attack classes
            attack_labels = {k: v for k, v in class_probs.items() if k != self.BENIGN_LABEL}
            top_label = max(attack_labels, key=attack_labels.get) if attack_labels else self.BENIGN_LABEL
            confidence = class_probs[top_label]
        else:
            # Benign dominant window
            mean_all_probs = proba_matrix.mean(axis=0)
            mean_all_probs[benign_col] = 0.0
            sum_att = mean_all_probs.sum()
            if sum_att > 0:
                mean_all_probs /= sum_att

            for col_idx, label_idx in enumerate(self._class_indices):
                label = self._idx_to_label[label_idx]
                if col_idx == benign_col:
                    class_probs[label] = 1.0 - window_attack_prob
                else:
                    class_probs[label] = float(window_attack_prob * mean_all_probs[col_idx])

            top_label = self.BENIGN_LABEL
            confidence = class_probs[self.BENIGN_LABEL]

        attack_prob = window_attack_prob
        is_attack = attack_prob >= self.ATTACK_THRESHOLD
        top_preds = self._build_top_predictions(class_probs)
        norm_family = normalise_family(top_label)

        return EvidenceRecord(
            model_name=self.MODEL_NAME,
            granularity=self.GRANULARITY,
            status=STATUS_OK,
            prediction=top_label,
            confidence=round(confidence, 6),
            attack_probability=round(attack_prob, 6),
            is_attack=is_attack,
            class_probabilities={k: round(v, 6) for k, v in class_probs.items()},
            top_predictions=[(lbl, round(p, 6)) for lbl, p in top_preds],
            original_prediction=top_label,
            normalised_family=norm_family,
            supporting_evidence={
                "n_flows_scored": n_flows,
                "feature_count": len(self._feature_names),
                "aggregation": "density-aware-flow-pooling",
                "max_attack_probability": round(max_p, 6),
                "mean_attack_probability": round(mean_p, 6),
                "median_attack_probability": round(med_p, 6),
                "p90_attack_probability": round(p90, 6),
                "p95_attack_probability": round(p95, 6),
                "p99_attack_probability": round(p99, 6),
                "fraction_attack_gt_050": round(frac_50, 6),
                "fraction_attack_gt_070": round(frac_70, 6),
                "fraction_attack_gt_090": round(frac_90, 6),
                "num_flows": n_flows,
            },
            timestamp=t0,
        )

    # -----------------------------------------------------------------------
    @property
    def trained_classes(self) -> list:
        return [self._idx_to_label[i] for i in sorted(self._idx_to_label)]

    @property
    def feature_names(self) -> list:
        return list(self._feature_names)

