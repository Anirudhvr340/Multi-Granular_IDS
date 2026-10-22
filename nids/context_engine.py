"""
Context Engine  —  Stage 3
===========================
Stateful memory that tracks attack behaviour across time windows.

The context engine transforms a stream of independent evidence records
into a temporal narrative:

  Window 1: attack_prob=0.41  → ALERT_1
  Window 2: attack_prob=0.57  → escalating
  Window 3: attack_prob=0.72  → escalating
  Window 4: attack_prob=0.88  → PERSISTENT HIGH
  Window 5: attack_prob=0.93  → escalating → BLOCK trigger

Rather than treating each window as an isolated event, the context
engine feeds escalation state, trend, and persistence to the Rule Engine
and Policy Agent.

Design constraints
------------------
- Bounded memory: state for at most `max_windows` recent events.
- No label leakage: never uses ground-truth labels as features.
- No future leakage: only reads past windows.
- Thread-safe: per-instance state, no global mutable state.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, List, Optional

from nids.evidence_schema import IntegratedEvidence


# ---------------------------------------------------------------------------
@dataclass
class WindowSnapshot:
    """Immutable snapshot of a single window's attack evidence."""
    window_id:        Optional[str]
    timestamp:        float
    attack_prob:      float
    is_attack:        bool
    attack_family:    Optional[str]
    model_agreement:  float
    models_available: int
    tw_prediction:    Optional[str] = None
    tw_confidence:    float = 0.0
    tw_status:        str = "unavailable"
    benign_signal:    str = "neutral"   # "strong_benign" | "moderate_benign" | "neutral" | "attack"
    action_taken:     Optional[str] = None   # set after decision


# ---------------------------------------------------------------------------
@dataclass
class ContextState:
    """
    Current context state exported to Rule Engine and Policy Agent.
    All fields computed purely from past evidence — no ground truth.
    """
    # Counters
    total_windows:             int   = 0
    attack_windows:            int   = 0
    consecutive_attack_windows: int  = 0
    max_consecutive_attacks:   int   = 0

    # Probability tracking
    current_attack_prob:       float = 0.0
    max_attack_prob:           float = 0.0
    mean_attack_prob_recent:   float = 0.0   # over last N windows
    trend_slope:               float = 0.0   # positive = escalating

    # Temporal
    first_alert_time:          Optional[float] = None
    last_alert_time:           Optional[float] = None
    attack_duration_seconds:   float = 0.0

    # Family
    dominant_attack_family:    Optional[str] = None
    attack_family_counts:      dict = field(default_factory=dict)

    # Escalation
    escalation_level:          str  = "NONE"   # NONE | LOW | MEDIUM | HIGH | CRITICAL
    is_escalating:             bool = False
    is_persistent:             bool = False

    # Actions taken
    last_action:               Optional[str] = None
    actions_taken:             list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
class ContextEngine:
    """
    Stage 3: Stateful context engine.

    Usage
    -----
    ctx = ContextEngine(max_windows=20, trend_window=5)
    state = ctx.update(integrated_evidence, action_taken="FLAG")
    # state.escalation_level, state.consecutive_attack_windows, …
    """

    def __init__(
        self,
        max_windows:   int   = 20,
        trend_window:  int   = 5,
        persist_thresh: int  = 3,    # consecutive attacks → is_persistent
        escalate_slope: float = 0.05, # trend slope threshold → is_escalating
    ):
        self.max_windows    = max_windows
        self.trend_window   = trend_window
        self.persist_thresh = persist_thresh
        self.escalate_slope = escalate_slope

        self._history: Deque[WindowSnapshot] = deque(maxlen=max_windows)
        self._state   = ContextState()

    # -----------------------------------------------------------------------
    def update(
        self,
        evidence: IntegratedEvidence,
        action_taken: Optional[str] = None,
    ) -> ContextState:
        """
        Ingest a new window's evidence and return the updated context state.
        Call AFTER the action has been decided (or pass None and update later).
        """
        # Determine benign decay vs attack signal
        tw_rec = evidence.time_window
        tw_pred = tw_rec.prediction if (tw_rec and tw_rec.status == "ok") else None
        tw_conf = tw_rec.confidence if (tw_rec and tw_rec.status == "ok") else 0.0
        tw_st   = tw_rec.status if tw_rec else "unavailable"

        fused_p = evidence.fused_attack_probability
        if (tw_pred == "Benign" and tw_conf >= 0.70) or fused_p <= 0.15:
            b_signal = "strong_benign"
        elif (tw_pred == "Benign" and tw_conf >= 0.50) or fused_p <= 0.30:
            b_signal = "moderate_benign"
        elif fused_p >= 0.35:
            b_signal = "attack"
        else:
            b_signal = "neutral"

        snap = WindowSnapshot(
            window_id=evidence.window_id,
            timestamp=evidence.timestamp,
            attack_prob=evidence.fused_attack_probability,
            is_attack=(b_signal == "attack"),
            attack_family=evidence.fused_attack_family,
            model_agreement=evidence.model_agreement_score,
            models_available=evidence.models_available,
            tw_prediction=tw_pred,
            tw_confidence=tw_conf,
            tw_status=tw_st,
            benign_signal=b_signal,
            action_taken=action_taken,
        )
        self._history.append(snap)
        self._recompute_state()
        return self._state

    def record_action(self, action: str):
        """Record the action taken for the most recent window."""
        if self._history:
            last = self._history[-1]
            # Dataclass is not frozen — update directly
            last.action_taken = action
        self._state.last_action = action
        if action not in self._state.actions_taken:
            self._state.actions_taken.append(action)

    # -----------------------------------------------------------------------
    def _recompute_state(self):
        """Recompute all derived state fields from history."""
        h    = list(self._history)
        n    = len(h)
        s    = self._state

        s.total_windows = n

        # ---- Attack window counts ------------------------------------------
        attack_snaps = [w for w in h if w.is_attack]
        s.attack_windows = len(attack_snaps)

        # ---- Consecutive attack tracking with explicit decay/reset ---------
        # Strong benign: reset to 0 immediately
        # Moderate benign: decrement persistence by 1
        # Attack: increment persistence
        # Neutral/missing: keep current persistence
        running_persist = 0
        for snap in h:
            if snap.benign_signal == "strong_benign":
                running_persist = 0
            elif snap.benign_signal == "moderate_benign":
                running_persist = max(0, running_persist - 1)
            elif snap.benign_signal == "attack":
                running_persist += 1

        s.consecutive_attack_windows = running_persist
        s.max_consecutive_attacks    = max(s.max_consecutive_attacks, running_persist)

        # ---- Probability tracking ------------------------------------------
        s.current_attack_prob = h[-1].attack_prob if h else 0.0
        s.max_attack_prob     = max((w.attack_prob for w in h), default=0.0)

        recent = h[-self.trend_window:] if len(h) >= 2 else h
        s.mean_attack_prob_recent = (
            sum(w.attack_prob for w in recent) / len(recent) if recent else 0.0
        )

        # ---- Trend slope (linear regression on attack_prob) ----------------
        if len(recent) >= 2:
            xs = list(range(len(recent)))
            ys = [w.attack_prob for w in recent]
            xm = sum(xs) / len(xs)
            ym = sum(ys) / len(ys)
            num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
            den = sum((x - xm) ** 2 for x in xs)
            s.trend_slope = num / den if den > 0 else 0.0
        else:
            s.trend_slope = 0.0

        # ---- Temporal --------------------------------------------------
        if attack_snaps:
            s.first_alert_time = attack_snaps[0].timestamp
            s.last_alert_time  = attack_snaps[-1].timestamp
            s.attack_duration_seconds = (
                s.last_alert_time - s.first_alert_time
                if s.first_alert_time and s.last_alert_time
                else 0.0
            )
        else:
            s.first_alert_time = None
            s.last_alert_time  = None
            s.attack_duration_seconds = 0.0

        # ---- Attack family -------------------------------------------------
        family_counts: dict = {}
        for snap in h:
            if snap.is_attack and snap.attack_family:
                family_counts[snap.attack_family] = (
                    family_counts.get(snap.attack_family, 0) + 1
                )
        s.attack_family_counts = family_counts
        s.dominant_attack_family = (
            max(family_counts, key=family_counts.get) if family_counts else None
        )

        # ---- Escalation assessment -----------------------------------------
        s.is_persistent  = s.consecutive_attack_windows >= self.persist_thresh
        s.is_escalating  = s.trend_slope >= self.escalate_slope
        s.escalation_level = self._compute_escalation_level(s)

    def _compute_escalation_level(self, s: ContextState) -> str:
        prob  = s.current_attack_prob
        cons  = s.consecutive_attack_windows
        trend = s.trend_slope

        if prob >= 0.90 or cons >= 6:
            return "CRITICAL"
        if prob >= 0.70 or (cons >= 3 and trend > 0):
            return "HIGH"
        if prob >= 0.50 or cons >= 2:
            return "MEDIUM"
        if prob >= 0.35:
            return "LOW"
        return "NONE"

    # -----------------------------------------------------------------------
    def get_state(self) -> ContextState:
        return self._state

    def reset(self):
        """Clear history (call between independent traffic sessions)."""
        self._history.clear()
        self._state = ContextState()

    def summary(self) -> dict:
        """Return a JSON-serialisable summary for the Policy Agent."""
        s = self._state
        return {
            "total_windows":              s.total_windows,
            "attack_windows":             s.attack_windows,
            "consecutive_attack_windows": s.consecutive_attack_windows,
            "max_consecutive_attacks":    s.max_consecutive_attacks,
            "current_attack_prob":        round(s.current_attack_prob, 4),
            "max_attack_prob":            round(s.max_attack_prob, 4),
            "mean_attack_prob_recent":    round(s.mean_attack_prob_recent, 4),
            "trend_slope":                round(s.trend_slope, 4),
            "is_escalating":              s.is_escalating,
            "is_persistent":              s.is_persistent,
            "escalation_level":           s.escalation_level,
            "dominant_attack_family":     s.dominant_attack_family,
            "attack_family_counts":       s.attack_family_counts,
            "attack_duration_seconds":    round(s.attack_duration_seconds, 1),
            "last_action":                s.last_action,
        }

