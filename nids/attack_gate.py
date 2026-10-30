"""
Attack Gate  —  Stage 1
========================
Binary IS_ATTACK decision from multi-model evidence.

Core principle
--------------
Do NOT use multiclass argmax alone.  A model that predicts:

  Benign = 0.30
  DDoS-A = 0.05
  DDoS-B = 0.05
  DoS    = 0.05
  ...  (14 attack classes summing to 0.70)

would incorrectly choose Benign by argmax.

Instead:
  P(attack) = 1 - P(benign)   → 0.70  ← IS_ATTACK

Then, only after the binary decision is made, route to ATTACK_FAMILY.

Fusion strategy
---------------
Weighted average of per-model attack probabilities.
Weights are based on:
  - Model granularity reliability (session > window > flow > packet for slow attacks)
  - Model availability (unavailable models are excluded)
  - Model confidence (higher confidence → higher weight)

Thresholds
----------
Default threshold: 0.35  (configurable)
Evaluated thresholds: 0.25, 0.30, 0.35, 0.40, 0.50
Threshold is applied to the FUSED attack probability, not per-model.
"""

from __future__ import annotations

import math
from typing import Optional

from nids.evidence_schema import (
    EvidenceRecord, IntegratedEvidence, STATUS_OK,
    GRANULARITY_PACKET, GRANULARITY_FLOW, GRANULARITY_WINDOW, GRANULARITY_SESSION,
    normalise_family,
)

# ---------------------------------------------------------------------------
# Default granularity reliability weights (prior, not hardcoded ratios)
# These are STARTING weights; the actual fusion adjusts for confidence.
_GRANULARITY_BASE_WEIGHT = {
    GRANULARITY_PACKET:  0.15,
    GRANULARITY_FLOW:    0.35,
    GRANULARITY_WINDOW:  0.30,
    GRANULARITY_SESSION: 0.20,
}


class AttackGate:
    """
    Stage 1: Binary attack gate + fused attack probability.

    Usage
    -----
    gate = AttackGate(threshold=0.35)
    integrated = gate.evaluate(packet_ev, flow_ev, time_ev, session_ev)
    if integrated.fused_attack_probability >= gate.threshold:
        # Route to attack handling
    """

    CONFIGURABLE_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.50]

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    # -----------------------------------------------------------------------
    def evaluate(
        self,
        packet_ev:  Optional[EvidenceRecord],
        flow_ev:    Optional[EvidenceRecord],
        time_ev:    Optional[EvidenceRecord],
        session_ev: Optional[EvidenceRecord],
        window_id:  Optional[str] = None,
    ) -> IntegratedEvidence:
        """
        Fuse four evidence records into an IntegratedEvidence.

        Only STATUS_OK records contribute to the fused probability.
        UNAVAILABLE / ERROR models are excluded but NOT treated as Benign.
        """

        records_by_granularity = {
            GRANULARITY_PACKET:  packet_ev,
            GRANULARITY_FLOW:    flow_ev,
            GRANULARITY_WINDOW:  time_ev,
            GRANULARITY_SESSION: session_ev,
        }

        available_ok = {
            g: r for g, r in records_by_granularity.items()
            if r is not None and r.status == STATUS_OK
        }

        # Multi-granular Byzantine consensus guardrail (Sommer & Paxson 2010):
        # If at least 3 models confirm Benign traffic (attack_prob <= self.threshold, not is_attack),
        # attenuate single-model out-of-distribution noise (e.g. Flow or Session false alarms)
        benign_models = [
            r for r in available_ok.values()
            if not r.is_attack and r.attack_probability <= self.threshold
        ]
        if len(benign_models) >= 3:
            for r in available_ok.values():
                if r.is_attack and r.attack_probability > self.threshold:
                    r.attack_probability = min(r.attack_probability, 0.25)
                    r.prediction = 'Benign'
                    r.is_attack = False

        # ---- Weighted fusion of attack probabilities with Situational Adaptation ----
        fused_attack_prob = 0.0
        models_attacking  = 0
        raw_weights = {}
        normalized_weights: dict[str, float] = {}

        if available_ok:
            total_weight = 0.0

            for granularity, record in available_ok.items():
                # 1. Base prior weight from granularity
                base_w = _GRANULARITY_BASE_WEIGHT.get(granularity, 0.25)

                # 2. Dynamic session sparsity attenuation:
                # A 30-flow sequential BiLSTM requires sufficient sequence continuity
                if granularity == GRANULARITY_SESSION:
                    n_f = record.supporting_evidence.get("flows_in_window", 30) if record.supporting_evidence else 30
                    if n_f < 5:
                        base_w *= 0.10  # Heavily de-weight isolated 1-4 flows
                    elif n_f < 15:
                        base_w *= 0.50  # De-weight sparse partial windows

                # 3. Dynamic Time-Window temporal buffer maturity:
                # Macro temporal lag/trend features need rolling history to avoid cold-start noise
                elif granularity == GRANULARITY_WINDOW:
                    h_depth = record.supporting_evidence.get("history_depth", 5) if record.supporting_evidence else 5
                    if h_depth < 3:
                        base_w *= 0.40  # De-weight cold-start window models
                    elif h_depth >= 5:
                        base_w *= 1.10  # Mature temporal rolling stats

                # 4. Dynamic Packet 1D-CNN volumetric specialization:
                # Dimensionless packet ratios are primary sensors for high-rate flood anomalies
                elif granularity == GRANULARITY_PACKET:
                    if record.is_attack and record.attack_probability >= 0.80:
                        base_w *= 1.25  # Boost packet sensor on confirmed packet header anomalies

                # 5. Dynamic Confidence modulation:
                # High confidence (close to 0.0 or 1.0) gets higher weight; ambiguous (0.50) is penalized
                binary_conf = max(record.attack_probability, 1.0 - record.attack_probability)
                conf_factor = 0.5 + 0.5 * math.tanh((binary_conf - 0.5) * 4)

                w = base_w * conf_factor
                raw_weights[granularity] = w
                total_weight += w

                if record.attack_probability >= self.threshold:
                    models_attacking += 1

            if total_weight > 0:
                weighted_sum = sum(w * available_ok[g].attack_probability for g, w in raw_weights.items())
                fused_attack_prob = weighted_sum / total_weight
                normalized_weights = {g: round(w / total_weight, 4) for g, w in raw_weights.items()}
            else:
                fused_attack_prob = 0.0
                normalized_weights = {g: 0.25 for g in available_ok}

        # ---- Agreement score -----------------------------------------------
        n_avail = len(available_ok)
        model_agreement = models_attacking / n_avail if n_avail > 0 else 0.0

        # ---- Attack family routing ----------------------------------------
        fused_family = self._route_attack_family(available_ok, fused_attack_prob)

        return IntegratedEvidence(
            packet=packet_ev,
            flow=flow_ev,
            time_window=time_ev,
            session=session_ev,
            fused_attack_probability=round(fused_attack_prob, 6),
            fused_attack_family=fused_family,
            model_agreement_score=round(model_agreement, 4),
            models_indicating_attack=models_attacking,
            models_available=n_avail,
            model_weights=normalized_weights,
            window_id=window_id,
        )

    # -----------------------------------------------------------------------
    def _route_attack_family(
        self,
        available: dict,
        fused_prob: float,
    ) -> Optional[str]:
        """
        Determine the most likely attack family from available models.
        Uses confidence-weighted voting over normalised families.
        """
        if fused_prob < self.threshold:
            return None

        family_scores: dict[str, float] = {}
        for granularity, record in available.items():
            if record.attack_probability < self.threshold:
                continue
            fam = record.normalised_family
            if fam is None:
                continue  # Benign prediction — skip
            w = record.attack_probability * record.confidence
            family_scores[fam] = family_scores.get(fam, 0.0) + w

        if not family_scores:
            # Fall back: look at top prediction from most confident model
            best = max(available.values(), key=lambda r: r.confidence)
            return normalise_family(best.prediction)

        return max(family_scores, key=family_scores.get)

    # -----------------------------------------------------------------------
    def is_attack(self, integrated: IntegratedEvidence) -> bool:
        return integrated.fused_attack_probability >= self.threshold

    # -----------------------------------------------------------------------
    def threshold_report(self, integrated: IntegratedEvidence) -> dict:
        """
        Report attack/benign decision at all configurable thresholds.
        Useful for validation without tuning on test data.
        """
        p = integrated.fused_attack_probability
        return {
            t: {"is_attack": p >= t, "fused_prob": round(p, 6)}
            for t in self.CONFIGURABLE_THRESHOLDS
        }

