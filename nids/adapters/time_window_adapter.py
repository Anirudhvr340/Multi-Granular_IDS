"""
Time-Window XGBoost Adapter
============================
Wraps xgb_time_window_model.joblib (268-feature, 11-class window classifier).

Input:   A dict produced by step0_preprocessing.window_aggregate(), or a
         pre-built base-feature vector (48 values matching window_base_feature_names).
         The adapter applies temporal feature engineering + scaling internally.

Trained classes (11 trained, 12 in class_names array):
  Benign, Bot, DDOS attack-HOIC, DDOS attack-LOIC-UDP,
  DDoS attacks-LOIC-HTTP, DoS attacks-GoldenEye, DoS attacks-Hulk,
  DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce,
  Infilteration, SSH-Bruteforce

NOTE — class-index mismatch:
  window_class_names has 12 entries (indices 0-11).
  xgb_model_classes.npy maps model output cols to class_names indices:
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  (class index 1 = "Bot" was not in training data on the test split.)
  The adapter resolves this mapping explicitly.

Dataset: CICIDS 2018 (aggregated into time windows)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent.parent          # repo root
TIME_WINDOW_DIR = HERE / "time_window"
PROCESSED_DIR   = TIME_WINDOW_DIR / "processed"

# Add time_window to path so we can import temporal_features
if str(TIME_WINDOW_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_WINDOW_DIR))

from nids.adapters.base_adapter import BaseAdapter
from nids.evidence_schema import (
    EvidenceRecord, STATUS_OK, GRANULARITY_WINDOW, normalise_family
)


class TimeWindowAdapter(BaseAdapter):
    """
    Adapter for the time-window XGBoost model.

    Maintains a rolling history buffer of base feature vectors so temporal
    features (lag, delta, rolling stats, trend, persistence) can be computed
    correctly without leaking future data.
    """

    MODEL_NAME   = "Time-Window XGBoost"
    GRANULARITY  = GRANULARITY_WINDOW
    BENIGN_LABEL = "Benign"

    def __init__(
        self,
        model_path: str | None = None,
        class_names_path: str | None = None,
        base_feat_path: str | None = None,
        all_feat_path: str | None = None,
        model_feat_path: str | None = None,
        scaler_path: str | None = None,
        temporal_config_path: str | None = None,
        attack_threshold: float = 0.35,
        history_size: int = 5,
    ):
        super().__init__(attack_threshold)
        self._history_size = history_size

        # ---- Load artifacts -----------------------------------------------
        model_path           = Path(model_path)           if model_path           else PROCESSED_DIR / "xgb_time_window_model.joblib"
        class_names_path     = Path(class_names_path)     if class_names_path     else PROCESSED_DIR / "window_class_names.npy"
        base_feat_path       = Path(base_feat_path)       if base_feat_path       else PROCESSED_DIR / "window_base_feature_names.npy"
        all_feat_path        = Path(all_feat_path)        if all_feat_path        else PROCESSED_DIR / "window_feature_names.npy"
        model_feat_path      = Path(model_feat_path)      if model_feat_path      else PROCESSED_DIR / "window_model_feature_names.npy"
        scaler_path          = Path(scaler_path)          if scaler_path          else PROCESSED_DIR / "time_window_scaler.joblib"
        temporal_config_path = Path(temporal_config_path) if temporal_config_path else PROCESSED_DIR / "temporal_feature_config.joblib"

        self._model       = joblib.load(model_path)
        self._class_names = np.load(class_names_path, allow_pickle=True).astype(str)
        self._base_feat   = np.load(base_feat_path,   allow_pickle=True).astype(str)
        self._all_feat    = np.load(all_feat_path,    allow_pickle=True).astype(str)
        self._model_feat  = np.load(model_feat_path,  allow_pickle=True).astype(str)
        self._scaler      = joblib.load(scaler_path)
        self._temporal_config = joblib.load(temporal_config_path) if temporal_config_path.exists() else None

        # xgb_model_classes: maps model.classes_ output column → class_names index
        xgb_cls_path = PROCESSED_DIR / "xgb_model_classes.npy"
        if xgb_cls_path.exists():
            self._xgb_classes = joblib.load(xgb_cls_path).astype(int)
        else:
            # Fallback: assume model.classes_ directly index class_names
            self._xgb_classes = self._model.classes_.astype(int)

        # Build index-to-label map using xgb_classes
        # model output column i → self._class_names[self._xgb_classes[i]]
        self._col_to_label = {}
        n_model_cols = len(self._model.classes_)
        for col_i in range(n_model_cols):
            if col_i < len(self._xgb_classes):
                name_idx = int(self._xgb_classes[col_i])
                if name_idx < len(self._class_names):
                    self._col_to_label[col_i] = str(self._class_names[name_idx])
                else:
                    self._col_to_label[col_i] = f"class_{name_idx}"
            else:
                self._col_to_label[col_i] = f"col_{col_i}"

        # All known class labels for this model
        self._all_class_labels = list(self._class_names.astype(str))

        # Precompute selected feature column indices in the full feature array
        all_feat_list   = list(self._all_feat.astype(str))
        model_feat_list = list(self._model_feat.astype(str))
        self._selected_indices = [
            all_feat_list.index(f) for f in model_feat_list if f in all_feat_list
        ]

        # Rolling history buffer: list of raw base-feature vectors (np.ndarray)
        self._history_buffer: list[np.ndarray] = []
        self._history_times:  list[np.datetime64] = []

    # -----------------------------------------------------------------------
    def reset_history(self):
        """Clear the rolling temporal history (call between independent sessions)."""
        self._history_buffer = []
        self._history_times  = []

    # -----------------------------------------------------------------------
    def _base_vector_from_aggregate(self, agg: dict) -> np.ndarray:
        """Convert a window_aggregate dict to the base feature numpy vector."""
        row = pd.DataFrame([agg]).reindex(columns=self._base_feat).fillna(0.0)
        return row.to_numpy(dtype=np.float32)[0]

    # -----------------------------------------------------------------------
    def _run_inference(self, input_data: Any) -> EvidenceRecord:
        """
        input_data : dict (window_aggregate output) OR np.ndarray (base feature vector)

        The adapter maintains internal history for temporal features.
        """
        from temporal_features import build_temporal_features, fit_temporal_config

        t0 = time.perf_counter()

        # ---- Resolve base vector -------------------------------------------
        if isinstance(input_data, dict):
            base_vec = self._base_vector_from_aggregate(input_data)
            window_ts = input_data.get("window_start", np.datetime64("now"))
        elif isinstance(input_data, np.ndarray):
            base_vec = input_data.astype(np.float32)
            if base_vec.ndim == 2:
                base_vec = base_vec[0]
            window_ts = np.datetime64("now")
        else:
            raise TypeError(f"TimeWindowAdapter expects dict or ndarray, got {type(input_data)}")

        # ---- Maintain history buffer ----------------------------------------
        self._history_buffer.append(base_vec)
        self._history_times.append(window_ts)
        # Keep only the last `history_size` windows
        self._history_buffer = self._history_buffer[-self._history_size:]
        self._history_times  = self._history_times[-self._history_size:]

        history_matrix = np.array(self._history_buffer, dtype=np.float32)
        time_array     = np.array(self._history_times)

        # ---- Compute temporal features -------------------------------------
        active_config = self._temporal_config or fit_temporal_config(
            history_matrix, self._base_feat
        )
        temporal_matrix, generated_names = build_temporal_features(
            history_matrix, time_array, self._base_feat, active_config
        )
        # Use the LAST row (current window's temporal features)
        current_temporal = temporal_matrix[-1]

        # ---- Align to full feature space -----------------------------------
        generated_list = list(generated_names.astype(str))
        all_feat_list  = list(self._all_feat.astype(str))
        aligned = np.zeros(len(all_feat_list), dtype=np.float32)
        for i, name in enumerate(all_feat_list):
            if name in generated_list:
                aligned[i] = current_temporal[generated_list.index(name)]

        # ---- Scale + select features ---------------------------------------
        scaled = self._scaler.transform(aligned.reshape(1, -1))
        selected = scaled[:, self._selected_indices]

        # ---- Predict -------------------------------------------------------
        proba = self._model.predict_proba(selected)[0]

        # ---- Build class probability dict ----------------------------------
        class_probs: dict = {lbl: 0.0 for lbl in self._all_class_labels}
        for col_i, prob in enumerate(proba):
            lbl = self._col_to_label.get(col_i, f"col_{col_i}")
            class_probs[lbl] = float(prob)

        # ---- Normalise (only mapped classes contribute to sum) -------------
        total = sum(class_probs.values())
        if total > 0:
            class_probs = {k: v / total for k, v in class_probs.items()}

        top_label   = max(class_probs, key=class_probs.get)
        confidence  = class_probs[top_label]
        attack_prob = self._compute_attack_prob(class_probs, self.BENIGN_LABEL)
        is_attack   = attack_prob >= self.ATTACK_THRESHOLD
        top_preds   = self._build_top_predictions(class_probs)
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
                "history_depth": len(self._history_buffer),
                "n_base_features": len(self._base_feat),
                "n_temporal_features": len(all_feat_list),
                "n_selected_features": len(self._selected_indices),
                "window_ts": str(window_ts),
            },
            timestamp=t0,
        )

    # -----------------------------------------------------------------------
    @property
    def trained_classes(self) -> list:
        return self._all_class_labels

    @property
    def feature_names(self) -> list:
        return list(self._model_feat.astype(str))

