"""
Window Aggregator & Model Branch
Aggregates flows into 30-second windows, computes 268 temporal features, and invokes Time-Window XGBoost.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "time_window"))

import step0_preprocessing as preprocessing
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.evidence_schema import EvidenceRecord

logger = logging.getLogger(__name__)


class WindowAggregator:
    """
    Branch extractor and runner for Time-Window XGBoost.
    """

    def __init__(self, attack_threshold: float = 0.35):
        self.adapter = TimeWindowAdapter(attack_threshold=attack_threshold)

    def aggregate_window(self, window_df: pd.DataFrame, window_start: Any) -> Dict[str, Any]:
        """
        Computes 268 aggregate temporal features for the 30-second window.
        """
        return preprocessing.window_aggregate(window_df, window_start)

    def predict_window(self, aggregate: Dict[str, Any], window_id: str) -> EvidenceRecord:
        """
        Runs Time-Window XGBoost on the aggregated window dictionary.
        """
        return self.adapter.predict(aggregate, window_id=window_id)

