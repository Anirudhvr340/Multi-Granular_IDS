"""
Packet Feature Extractor & Model Branch
Extracts 27 dimensionless packet-header ratios & indicators and invokes Packet 1D-CNN.
"""

import os
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from nids.adapters.packet_adapter import PacketModelAdapter
from nids.evidence_schema import EvidenceRecord

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent.parent


class PacketFeatureExtractor:
    """
    Branch extractor for the Packet 1D-CNN model.
    """

    def __init__(self, attack_threshold: float = 0.35):
        self.adapter = PacketModelAdapter(attack_threshold=attack_threshold)
        self.feature_names = self.adapter.features
        self.scaler = self.adapter.scaler
        self.labels = self.adapter.classes

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts the 27 dimensionless packet-header features and scales them.
        """
        feats, feat_names = self.adapter._extract_dimensionless_features(df)
        X_scaled = self.scaler.transform(feats.values)
        return X_scaled, feat_names

    def predict_window(self, window_df: pd.DataFrame, window_id: str) -> EvidenceRecord:
        """
        Runs the Packet model on the window DataFrame and returns an EvidenceRecord.
        """
        return self.adapter.predict(window_df, window_id=window_id)
