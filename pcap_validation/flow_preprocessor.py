"""
Flow Preprocessor & Model Branch
Extracts 17 selected flow features, computes burstiness, and invokes Flow XGBoost.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from nids.adapters.flow_adapter import FlowModelAdapter
from nids.evidence_schema import EvidenceRecord

logger = logging.getLogger(__name__)


class FlowPreprocessor:
    """
    Branch extractor and runner for Flow XGBoost.
    """

    def __init__(self, attack_threshold: float = 0.35):
        self.adapter = FlowModelAdapter(attack_threshold=attack_threshold)
        self.feature_names = self.adapter.REQUIRED_FEATURES

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        return self.adapter._prepare_features(df)

    def predict_window(self, window_df: pd.DataFrame, window_id: str) -> EvidenceRecord:
        return self.adapter.predict(window_df, window_id=window_id)

