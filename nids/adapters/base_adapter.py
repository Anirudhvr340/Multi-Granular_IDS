"""
Base Adapter
============
Abstract base class that every model adapter must implement.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from nids.evidence_schema import EvidenceRecord, normalise_family, STATUS_OK


class BaseAdapter(ABC):
    """
    Abstract base for all model adapters.

    Subclasses implement _run_inference() and expose .predict().
    The base class handles timing, error wrapping, and common fields.
    """

    MODEL_NAME: str = "BaseModel"
    GRANULARITY: str = "unknown"
    BENIGN_LABEL: str = "Benign"

    # Attack-probability threshold: P(attack) >= threshold → is_attack = True
    # Can be overridden per-adapter or set globally.
    ATTACK_THRESHOLD: float = 0.35

    def __init__(self, attack_threshold: float | None = None):
        if attack_threshold is not None:
            self.ATTACK_THRESHOLD = attack_threshold

    # -----------------------------------------------------------------------
    @abstractmethod
    def _run_inference(self, input_data: Any) -> EvidenceRecord:
        """
        Perform model inference on input_data.
        Must return a populated EvidenceRecord with status=STATUS_OK.
        Raise any exception on failure — the base predict() will catch it.
        """

    # -----------------------------------------------------------------------
    def predict(self, input_data: Any, window_id: str | None = None) -> EvidenceRecord:
        """
        Public entry point.  Wraps _run_inference() with timing + error handling.
        """
        t0 = time.perf_counter()
        try:
            record = self._run_inference(input_data)
        except Exception as exc:
            record = EvidenceRecord.error(
                model_name=self.MODEL_NAME,
                granularity=self.GRANULARITY,
                reason=f"{type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record.inference_latency_ms = round(elapsed_ms, 3)
        record.window_id = window_id
        return record

    # -----------------------------------------------------------------------
    @staticmethod
    def _build_top_predictions(class_probabilities: dict, n: int = 5) -> list:
        return sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)[:n]

    @staticmethod
    def _compute_attack_prob(class_probabilities: dict, benign_label: str = "Benign") -> float:
        """P(attack) = 1 - P(benign).  Safe against missing benign key."""
        return float(1.0 - class_probabilities.get(benign_label, 0.0))

