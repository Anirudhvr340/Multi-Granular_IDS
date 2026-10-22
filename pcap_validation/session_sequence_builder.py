"""
Session Sequence Builder & Model Branch
Constructs chronological (T=30, F=27) tensors and invokes Session CNN-LSTM.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from nids.adapters.session_adapter import SessionModelAdapter
from nids.evidence_schema import EvidenceRecord

logger = logging.getLogger(__name__)


class SessionSequenceBuilder:
    """
    Branch extractor and runner for Session CNN-LSTM.
    """

    def __init__(self, attack_threshold: float = 0.35):
        self.adapter = SessionModelAdapter(attack_threshold=attack_threshold)
        self.feature_names = self.adapter.features
        self.scaler = self.adapter.scaler
        self.seq_len = 30

    def prepare_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts features, scales them, and constructs sequence chunks (N, 30, 27).
        """
        feats = self.adapter._extract_features(df)
        X_scaled = self.scaler.transform(feats.values)
        return X_scaled

    def predict_window(self, window_df: pd.DataFrame, window_id: str) -> EvidenceRecord:
        """
        Runs the Session model on the window DataFrame and returns an EvidenceRecord.
        """
        return self.adapter.predict(window_df, window_id=window_id)
