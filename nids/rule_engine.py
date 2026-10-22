"""
Rule Engine  —  Stage 2
========================
Deterministic fast-path rules applied BEFORE Gemini.

Purpose
-------
1. Handle obvious high-confidence cases immediately (no LLM needed).
2. Reduce unnecessary Gemini API calls.
3. Provide explainable, auditable security decisions.
4. Act as a guardrail — does NOT replace the ML models.

Rules are:
  - Configurable (enable/disable per rule)
  - Explainable (each rule produces a reason string)
  - Logged (each triggered rule is recorded)
  - Conservative (prefer FLAG over BLOCK when uncertain)

Rule design
-----------
Rules are evaluated in priority order.  The first triggered rule returns.
Each rule returns a RuleResult(action, severity, rule_name, reason) or None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from nids.evidence_schema import IntegratedEvidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    action:    str          # "ALLOW" | "FLAG" | "BLOCK"
    severity:  str          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    rule_name: str
    reason:    str
    triggered: bool = True


# ---------------------------------------------------------------------------
@dataclass
class RuleConfig:
    """
    Configurable thresholds and toggles for each rule.
    Change values here to tune rule sensitivity.
    """
    # Rule 1: All models agree — high confidence attack
    r1_enabled: bool  = True
    r1_min_agreement: float = 0.9    # ≥90% agreement
    r1_min_prob:      float = 0.85   # fused P(attack) ≥ 0.85
    r1_min_available: int   = 2      # at least 2 models must be OK

    # Rule 2: Extreme attack probability — single-model override
    r2_enabled: bool  = True
    r2_min_prob: float = 0.95        # P(attack) ≥ 0.95 from fused signal

    # Rule 3: Persistent escalating attack (multi-window context)
    r3_enabled: bool  = True
    r3_min_consecutive: int = 3      # ≥3 consecutive attack windows
    r3_min_prob:        float = 0.60

    # Rule 4: DDoS flood indicator (calibrated for high precision)
    r4_enabled: bool  = True
    r4_ddos_families: tuple = ("DDoS", "DoS")
    r4_min_prob:      float = 0.85   # raised from 0.70 for high precision
    r4_min_agreement: float = 0.75   # raised from 0.50 for high precision

    # Rule 5: Very low evidence — benign path
    r5_enabled: bool  = True
    r5_max_prob: float = 0.15        # P(attack) ≤ 0.15 → ALLOW

    # Rule 6: Moderate evidence — FLAG (calibrated)
    r6_enabled: bool  = True
    r6_min_prob: float = 0.45        # shifted from 0.35 to 0.45
    r6_max_prob: float = 0.65        # shifted from 0.60 to 0.65


# ---------------------------------------------------------------------------
class RuleEngine:
    """
    Stage 2: Deterministic rule engine.

    Usage
    -----
    engine = RuleEngine(config=RuleConfig())
    result = engine.evaluate(integrated, context)
    if result:
        # Use result.action — skip Gemini
    else:
        # Route to Gemini for complex reasoning
    """

    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()
        self._triggered_rules: list[RuleResult] = []

    # -----------------------------------------------------------------------
    def evaluate(
        self,
        integrated: IntegratedEvidence,
        context: Optional[dict] = None,
    ) -> Optional[RuleResult]:
        """
        Evaluate all enabled rules in priority order.
        Returns the first triggered RuleResult, or None (→ route to agent).
        """
        self._triggered_rules = []
        context = context or {}
        cfg = self.config

        # Rule 2 first — extreme probability overrides everything
        if cfg.r2_enabled:
            r = self._rule2_extreme_prob(integrated)
            if r:
                self._log_rule(r)
                return r

        # Rule 1: consensus block
        if cfg.r1_enabled:
            r = self._rule1_consensus_block(integrated)
            if r:
                self._log_rule(r)
                return r

        # Rule 4: DDoS flood
        if cfg.r4_enabled:
            r = self._rule4_ddos_flood(integrated)
            if r:
                self._log_rule(r)
                return r

        # Rule 3: persistent attack (needs context)
        if cfg.r3_enabled and context:
            r = self._rule3_persistent(integrated, context)
            if r:
                self._log_rule(r)
                return r

        # Rule 5: clear benign → ALLOW fast path
        if cfg.r5_enabled:
            r = self._rule5_allow_benign(integrated)
            if r:
                self._log_rule(r)
                return r

        # Rule 6: moderate evidence → FLAG (context-aware)
        if cfg.r6_enabled:
            r = self._rule6_moderate_flag(integrated, context)
            if r:
                self._log_rule(r)
                return r

        # No rule triggered → route to dynamic agent
        return None

    # -----------------------------------------------------------------------
    def _rule1_consensus_block(self, ev: IntegratedEvidence) -> Optional[RuleResult]:
        """BLOCK when ≥90% of available models agree it's an attack at high prob."""
        cfg = self.config
        if (
            ev.models_available >= cfg.r1_min_available
            and ev.model_agreement_score >= cfg.r1_min_agreement
            and ev.fused_attack_probability >= cfg.r1_min_prob
        ):
            return RuleResult(
                action="BLOCK",
                severity="CRITICAL",
                rule_name="R1_CONSENSUS_BLOCK",
                reason=(
                    f"≥{cfg.r1_min_agreement*100:.0f}% model agreement "
                    f"({ev.models_indicating_attack}/{ev.models_available} models) "
                    f"with P(attack)={ev.fused_attack_probability:.3f} "
                    f"≥ threshold {cfg.r1_min_prob}."
                ),
            )
        return None

    def _rule2_extreme_prob(self, ev: IntegratedEvidence) -> Optional[RuleResult]:
        """BLOCK immediately when fused P(attack) is extreme."""
        cfg = self.config
        if ev.fused_attack_probability >= cfg.r2_min_prob:
            return RuleResult(
                action="BLOCK",
                severity="CRITICAL",
                rule_name="R2_EXTREME_PROB",
                reason=(
                    f"Extreme fused P(attack)={ev.fused_attack_probability:.3f} "
                    f"≥ {cfg.r2_min_prob}. Immediate BLOCK triggered."
                ),
            )
        return None

    def _rule3_persistent(self, ev: IntegratedEvidence, context: dict) -> Optional[RuleResult]:
        """BLOCK persistent escalating attacks across multiple windows."""
        cfg = self.config
        consecutive = context.get("consecutive_attack_windows", 0)
        if (
            consecutive >= cfg.r3_min_consecutive
            and ev.fused_attack_probability >= cfg.r3_min_prob
        ):
            family = ev.fused_attack_family or "Unknown"
            return RuleResult(
                action="BLOCK",
                severity="HIGH",
                rule_name="R3_PERSISTENT_ATTACK",
                reason=(
                    f"Persistent attack detected: {consecutive} consecutive "
                    f"attack windows, family={family}, "
                    f"P(attack)={ev.fused_attack_probability:.3f}."
                ),
            )
        return None

    def _rule4_ddos_flood(self, ev: IntegratedEvidence) -> Optional[RuleResult]:
        """BLOCK high-confidence DDoS/DoS floods; downgrade uncertain floods to FLAG."""
        cfg = self.config
        family = ev.fused_attack_family or ""
        if family in cfg.r4_ddos_families:
            if (
                ev.fused_attack_probability >= cfg.r4_min_prob
                and ev.model_agreement_score >= cfg.r4_min_agreement
            ):
                return RuleResult(
                    action="BLOCK",
                    severity="HIGH",
                    rule_name="R4_DDOS_FLOOD",
                    reason=(
                        f"High-confidence {family} detected: "
                        f"P(attack)={ev.fused_attack_probability:.3f}, "
                        f"agreement={ev.model_agreement_score:.2f}."
                    ),
                )
            elif ev.fused_attack_probability >= 0.70:
                return RuleResult(
                    action="FLAG",
                    severity="MEDIUM",
                    rule_name="R4_DDOS_FLOOD",
                    reason=(
                        f"Potential {family} detected with moderate consensus: "
                        f"P(attack)={ev.fused_attack_probability:.3f}. Flagging for review."
                    ),
                )
        return None

    def _rule5_allow_benign(self, ev: IntegratedEvidence) -> Optional[RuleResult]:
        """Fast-path ALLOW for very low attack probability."""
        cfg = self.config
        if ev.fused_attack_probability <= cfg.r5_max_prob:
            return RuleResult(
                action="ALLOW",
                severity="LOW",
                rule_name="R5_ALLOW_BENIGN",
                reason=(
                    f"Fused P(attack)={ev.fused_attack_probability:.3f} "
                    f"≤ {cfg.r5_max_prob}. Traffic classified as benign."
                ),
            )
        return None

    def _rule6_moderate_flag(
        self,
        ev: IntegratedEvidence,
        context: Optional[dict] = None,
    ) -> Optional[RuleResult]:
        """FLAG moderate-evidence windows only if multiple sources or context support suspicion."""
        cfg = self.config
        if cfg.r6_min_prob <= ev.fused_attack_probability <= cfg.r6_max_prob:
            is_multi = ev.models_indicating_attack >= 2
            is_persistent = bool(
                context
                and (
                    context.get("consecutive_attack_windows", 0) >= 1
                    or context.get("is_escalating", False)
                )
            )
            # Only flag if there is corroborating evidence or persistence
            if is_multi or is_persistent:
                return RuleResult(
                    action="FLAG",
                    severity="MEDIUM",
                    rule_name="R6_MODERATE_FLAG",
                    reason=(
                        f"Moderate P(attack)={ev.fused_attack_probability:.3f} "
                        f"with corroborating context (multi-source={is_multi}, persistent={is_persistent}). "
                        f"Flagging for investigation."
                    ),
                )
        return None

    def _log_rule(self, result: RuleResult):
        self._triggered_rules.append(result)
        logger.info(
            "[RuleEngine] %s → %s (%s): %s",
            result.rule_name, result.action, result.severity, result.reason
        )

    @property
    def last_triggered_rules(self) -> list[RuleResult]:
        return list(self._triggered_rules)

