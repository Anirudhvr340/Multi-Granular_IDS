"""
Common Evidence Schema
======================
Standardized output format produced by every model adapter.

All four models (Packet, Flow, Time-Window, Session) must return a dict
conforming to this schema.  The integration pipeline never inspects
model internals — it only consumes evidence dicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Granularity constants
# ---------------------------------------------------------------------------
GRANULARITY_PACKET = "packet"
GRANULARITY_FLOW   = "flow"
GRANULARITY_WINDOW = "time_window"
GRANULARITY_SESSION = "session"

# Model status constants
STATUS_OK          = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR       = "error"


@dataclass
class EvidenceRecord:
    """
    Canonical output produced by every model adapter.

    Fields
    ------
    model_name      : Human-readable model identifier.
    granularity     : One of 'packet' | 'flow' | 'time_window' | 'session'.
    status          : 'ok' | 'unavailable' | 'error'.
    prediction      : Top-1 predicted class label (str).
    confidence      : Probability of the top-1 prediction (0-1).
    attack_probability : P(attack) = 1 - P(benign).  Critical for Attack Gate.
    is_attack       : True when attack_probability exceeds the model threshold.
    class_probabilities : {class_name: probability} for all known classes.
    top_predictions : Ordered list of (label, probability) tuples, highest first.
    original_prediction : Raw label before any family normalisation.
    normalised_family   : Normalised attack family (DDoS, BruteForce, etc.) or None.
    supporting_evidence : Model-specific auxiliary info (feature values, window stats …).
    timestamp       : Unix timestamp when this record was produced.
    window_id       : Optional window/session identifier for temporal tracking.
    reason          : Short human-readable explanation (populated on error/unavailable).
    inference_latency_ms : Time spent running inference (ms).
    """
    model_name: str
    granularity: str
    status: str = STATUS_OK

    prediction: str = "Unknown"
    confidence: float = 0.0
    attack_probability: float = 0.0
    is_attack: bool = False

    class_probabilities: Dict[str, float] = field(default_factory=dict)
    top_predictions: List[tuple] = field(default_factory=list)

    original_prediction: str = "Unknown"
    normalised_family: Optional[str] = None

    supporting_evidence: Dict[str, Any] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)
    window_id: Optional[str] = None
    reason: str = ""
    inference_latency_ms: float = 0.0

    # -----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialisation
        d["top_predictions"] = [[label, prob] for label, prob in self.top_predictions]
        return d

    # -----------------------------------------------------------------------
    @classmethod
    def unavailable(cls, model_name: str, granularity: str, reason: str) -> "EvidenceRecord":
        """Factory: produce an 'unavailable' evidence record (model missing / failed)."""
        return cls(
            model_name=model_name,
            granularity=granularity,
            status=STATUS_UNAVAILABLE,
            prediction="UNAVAILABLE",
            confidence=0.0,
            attack_probability=0.0,
            is_attack=False,
            reason=reason,
        )

    @classmethod
    def error(cls, model_name: str, granularity: str, reason: str) -> "EvidenceRecord":
        """Factory: produce an 'error' evidence record (inference crashed)."""
        return cls(
            model_name=model_name,
            granularity=granularity,
            status=STATUS_ERROR,
            prediction="ERROR",
            confidence=0.0,
            attack_probability=0.0,
            is_attack=False,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Attack family normalisation map
# ---------------------------------------------------------------------------
# Maps original model class names → normalised attack family.
# Only semantically justified mappings are included.
# DO NOT add mappings that are speculative.
ATTACK_FAMILY_MAP: Dict[str, str] = {
    # DDoS variants
    "DDOS attack-HOIC":       "DDoS",
    "DDOS attack-LOIC-UDP":   "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDoS":                   "DDoS",

    # DoS variants
    "DoS attacks-GoldenEye":  "DoS",
    "DoS attacks-Hulk":       "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris":  "DoS",

    # Brute Force variants
    "Brute Force -Web":       "BruteForce",
    "Brute Force -XSS":       "BruteForce",
    "FTP-BruteForce":         "BruteForce",
    "SSH-Bruteforce":         "BruteForce",

    # Other
    "Bot":                    "Bot",
    "Infilteration":          "Infiltration",
    "SQL Injection":          "WebAttack",

    # Benign
    "Benign":                 None,
}


def normalise_family(label: str) -> Optional[str]:
    """Return the normalised attack family for a label, or None for Benign."""
    return ATTACK_FAMILY_MAP.get(label, label)  # unknown → keep as-is


# ---------------------------------------------------------------------------
# Multi-model integrated evidence container
# ---------------------------------------------------------------------------
@dataclass
class IntegratedEvidence:
    """
    Holds all four model evidence records + computed fusion metrics.
    Passed to the Rule Engine, Context Engine, and Policy Agent.
    """
    packet:      Optional[EvidenceRecord] = None
    flow:        Optional[EvidenceRecord] = None
    time_window: Optional[EvidenceRecord] = None
    session:     Optional[EvidenceRecord] = None

    # Fusion-level metrics (computed by attack gate)
    fused_attack_probability: float = 0.0
    fused_attack_family: Optional[str] = None
    model_agreement_score: float = 0.0   # 0-1
    models_indicating_attack: int = 0
    models_available: int = 0

    timestamp: float = field(default_factory=time.time)
    window_id: Optional[str] = None

    def available_models(self) -> List[EvidenceRecord]:
        """Return only the records that are actually available (status=ok)."""
        return [
            r for r in [self.packet, self.flow, self.time_window, self.session]
            if r is not None and r.status == STATUS_OK
        ]

    def all_records(self) -> Dict[str, Optional[EvidenceRecord]]:
        return {
            "packet":      self.packet,
            "flow":        self.flow,
            "time_window": self.time_window,
            "session":     self.session,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet":      self.packet.to_dict() if self.packet else None,
            "flow":        self.flow.to_dict() if self.flow else None,
            "time_window": self.time_window.to_dict() if self.time_window else None,
            "session":     self.session.to_dict() if self.session else None,
            "fused_attack_probability": round(self.fused_attack_probability, 6),
            "fused_attack_family":      self.fused_attack_family,
            "model_agreement_score":    round(self.model_agreement_score, 4),
            "models_indicating_attack": self.models_indicating_attack,
            "models_available":         self.models_available,
            "timestamp":               self.timestamp,
            "window_id":               self.window_id,
        }

