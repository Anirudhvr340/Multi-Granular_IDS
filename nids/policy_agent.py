"""
Policy Agent  —  Stage 4
=========================
Hybrid Deterministic and Agentic Policy Reasoning Agent backed by local Qwen 2.5 (7B Instruct).

The agent is NOT a classifier. The four ML models are the classifiers.
The agent RECEIVES their evidence and DECIDES: ALLOW | FLAG | BLOCK.

Hybrid Architecture:
1. Fast-Path Deterministic Engine: Handles unambiguous cases and rule triggers (R1-R6) with 0ms latency.
2. Agentic Reasoning Layer (Qwen 2.5 7B Instruct): Dispatched on confusing, ambiguous, or contested outputs
   (e.g., cross-model disagreement, borderline probabilities, conflicting attack families).
3. Evaluates context state (escalation, persistence, history).
4. Produces transparent, machine-verifiable security decisions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from nids.evidence_schema import IntegratedEvidence
from nids.rule_engine import RuleResult

logger = logging.getLogger(__name__)

VALID_ACTIONS   = {"ALLOW", "FLAG", "BLOCK"}
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
@dataclass
class AgentDecision:
    """Final decision produced by the Policy Agent."""
    action:             str    = "FLAG"
    severity:           str    = "MEDIUM"
    confidence:         float  = 0.5
    attack_family:      Optional[str] = None
    reason:             str    = ""
    evidence_used:      list   = field(default_factory=list)
    recommended_actions: list  = field(default_factory=list)
    source:             str    = "agent"   # "qwen-2.5-7b" | "deterministic" | "rule" | "heuristic"
    rule_triggered:     Optional[str] = None
    risk_score:         float  = 0.0
    risk_components:    dict   = field(default_factory=dict)
    context_snapshot:   dict   = field(default_factory=dict)
    gemini_raw:         Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
class PolicyAgent:
    """
    Stage 4: Dynamic policy agent.

    Hybrid Deterministic + Agentic Policy Engine.
    Dispatches ambiguous/confusing model outputs to local Qwen 2.5 (7B Instruct).

    Usage
    -----
    agent = PolicyAgent()
    decision = agent.decide(integrated, context_summary, rule_result)
    """

    LLM_MODEL_DEFAULT = "qwen2.5:7b"
    OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434/api/generate"

    def __init__(
        self,
        gemini_model:        Optional[str]  = None,
        gemini_api_key:      Optional[str]  = None,
        allow_threshold:     float = 0.25,
        flag_threshold:      float = 0.35,
        block_threshold:     float = 0.65,
        use_gemini:          bool  = True,
        llm_model:           Optional[str]  = None,
        ollama_url:          Optional[str]  = None,
    ):
        self.llm_model   = llm_model or os.environ.get("LLM_MODEL", self.LLM_MODEL_DEFAULT)
        self.ollama_url  = ollama_url or os.environ.get("OLLAMA_URL", self.OLLAMA_URL_DEFAULT)
        self.allow_t     = allow_threshold
        self.flag_t      = flag_threshold
        self.block_t     = block_threshold
        self.use_gemini  = use_gemini  # Controls whether agentic reasoning is enabled

        self._llm_calls  = 0
        self._llm_errors = 0
        self._confusing_dispatched = 0
        self._deterministic_handled = 0

    # -----------------------------------------------------------------------
    def _is_confusing_evidence(
        self,
        ev: IntegratedEvidence,
        ctx: dict,
        risk_score: float,
    ) -> bool:
        """
        Identifies ambiguous, contested, or confusing outputs across the multi-granular models.
        
        Confusing output criteria:
        1. Model Contradiction: Some models predict attack while others predict benign.
        2. Low Consensus Agreement: Model agreement score < 0.70.
        3. Ambiguous Risk Zone: Dynamic risk score in the critical decision margin (0.30 <= risk <= 0.65).
        4. Attack Family Conflict: Conflicting attack families predicted across available models.
        """
        # Model disagreement on binary verdict (attack vs benign)
        has_split_models = (ev.models_available > 1 and 0 < ev.models_indicating_attack < ev.models_available)
        
        # Low consensus agreement among models
        low_agreement = (ev.model_agreement_score < 0.70)
        
        # Borderline risk score in the ambiguous transition zone where deterministic rules don't fire
        ambiguous_risk = (0.30 <= risk_score <= 0.65)
        
        # Discrepant attack families among attack models
        attack_families = {
            m.normalised_family for m in ev.available_models()
            if m.is_attack and m.normalised_family
        }
        family_conflict = len(attack_families) > 1

        return has_split_models or low_agreement or (ambiguous_risk and family_conflict)

    # -----------------------------------------------------------------------
    def decide(
        self,
        integrated:      IntegratedEvidence,
        context_summary: dict,
        rule_result:     Optional[RuleResult] = None,
    ) -> AgentDecision:
        """
        Main entry point for hybrid deterministic + agentic AI decision making.
        """
        # ---- Compute transparent dynamic risk score ------------------------
        risk_score, risk_components = self._compute_risk_score(integrated, context_summary)

        # ---- Stage A: Deterministic Fast-Path (Rules) ---------------------
        if rule_result is not None:
            self._deterministic_handled += 1
            decision = AgentDecision(
                action=rule_result.action,
                severity=rule_result.severity,
                confidence=integrated.fused_attack_probability,
                attack_family=integrated.fused_attack_family,
                reason=rule_result.reason,
                evidence_used=self._summarise_evidence(integrated),
                source="rule",
                rule_triggered=rule_result.rule_name,
                risk_score=risk_score,
                risk_components=risk_components,
                context_snapshot=context_summary,
            )
            logger.info("[Agent] Rule %s → %s", rule_result.rule_name, rule_result.action)
            return decision

        # ---- Stage B: Hybrid Agentic Reasoning on Confusing Outputs -------
        is_confusing = self._is_confusing_evidence(integrated, context_summary, risk_score)

        if self.use_gemini and is_confusing:
            self._confusing_dispatched += 1
            session_summary = self._build_session_summary(integrated, context_summary)
            session_summary["risk_score"] = risk_score
            session_summary["risk_components"] = risk_components

            llm_result = self._call_qwen(session_summary, integrated)
            if llm_result and llm_result.get("status") == "ok":
                decision = self._parse_llm_response(llm_result["report"], integrated, context_summary)
                self._llm_calls += 1
            else:
                self._llm_errors += 1
                decision = self._heuristic_decision(integrated, context_summary)
                if llm_result and llm_result.get("status") == "error":
                    decision.reason += f" [Qwen fallback: {llm_result.get('reason', '')}]"
        else:
            # Deterministic clear-cut resolution
            self._deterministic_handled += 1
            decision = self._heuristic_decision(integrated, context_summary)
            if not is_confusing and self.use_gemini:
                decision.source = "deterministic"

        decision.risk_score = risk_score
        decision.risk_components = risk_components
        return decision

    # -----------------------------------------------------------------------
    def _build_session_summary(
        self,
        ev: IntegratedEvidence,
        ctx: dict,
    ) -> dict:
        """
        Build the structured prompt payload sent to the LLM agent.
        """
        def _safe(record):
            if record is None:
                return {"status": "unavailable"}
            return {
                "status":           record.status,
                "prediction":       record.prediction,
                "confidence":       round(record.confidence, 4),
                "attack_probability": round(record.attack_probability, 4),
                "is_attack":        record.is_attack,
                "normalised_family": record.normalised_family,
                "top_predictions":  [(l, round(p, 4)) for l, p in record.top_predictions[:3]],
                "inference_latency_ms": record.inference_latency_ms,
            }

        return {
            "packet_model":      _safe(ev.packet),
            "flow_model":        _safe(ev.flow),
            "time_window_model": _safe(ev.time_window),
            "session_model":     _safe(ev.session),
            "fused": {
                "attack_probability": round(ev.fused_attack_probability, 4),
                "attack_family":      ev.fused_attack_family,
                "model_agreement":    round(ev.model_agreement_score, 4),
                "models_available":   ev.models_available,
                "models_attacking":   ev.models_indicating_attack,
            },
            "context": ctx,
        }

    # -----------------------------------------------------------------------
    @staticmethod
    def _clean_and_parse_json(text: str) -> dict:
        """Robustly parse JSON text, removing markdown fences or falling back to regex."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            act_match = re.search(r'"action"\s*:\s*"([A-Za-z]+)"', cleaned)
            sev_match = re.search(r'"severity"\s*:\s*"([A-Za-z]+)"', cleaned)
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
            fam_match = re.search(r'"attack_family"\s*:\s*"([^"]*)"', cleaned)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)

            if act_match:
                return {
                    "action": act_match.group(1).upper(),
                    "severity": sev_match.group(1).upper() if sev_match else "MEDIUM",
                    "attack_family": fam_match.group(1) if fam_match else None,
                    "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                    "reason": reason_match.group(1) if reason_match else "Recovered from partial response",
                    "evidence_used": [],
                    "recommended_actions": [],
                }
            raise

    # -----------------------------------------------------------------------
    def _call_qwen(self, session_summary: dict, ev: IntegratedEvidence) -> dict:
        """Send session summary to local Qwen 2.5 7B Instruct via Ollama."""
        allowed_attacks = set()
        for record in ev.available_models():
            if record.normalised_family:
                allowed_attacks.add(record.normalised_family)
            if record.prediction not in ("Benign", "Unknown", "UNAVAILABLE", "ERROR"):
                allowed_attacks.add(record.prediction)

        system_prompt = (
            "You are strictly the POLICY DECISION LAYER of a hybrid Network Intrusion Detection System.\n"
            "CRITICAL ARCHITECTURAL CONTEXT:\n"
            "1. You are NOT a traffic classifier. The 4 specialized ML models have already classified the traffic.\n"
            "2. NEVER USE MAJORITY VOTING (do NOT count 3 vs 1 or 2 vs 2 models). In a multi-granular IDS, models inspect completely different granularities:\n"
            "   - Packet 1D-CNN: L3/L4 header anomalies (SYN floods, port scans, invalid flags). Blind to multi-flow or session behaviors.\n"
            "   - Flow XGBoost: Per-flow statistics (duration, IAT, packet size metrics). Catches brute force and single-flow DoS.\n"
            "   - Session CNN-LSTM: Multi-flow sequential dynamics over time. Catches slow-rate attacks (Slowloris, SlowHTTPTest).\n"
            "   - Time-Window XGBoost: Macro volumetric aggregate counters over 30s. Catches large distributed floods.\n"
            "   Because granularities are asymmetric, stealthy or application-layer attacks (e.g., Slowloris, SQL injection) are often visible to ONLY ONE OR TWO specialized models, while aggregate models legitimately see benign background traffic. A high attack probability (e.g., P >= 0.70) from a specialized model MUST NOT be outvoted by benign votes from unrelated granularities.\n\n"
            "PROBABILITY-BASED ARBITRATION RULES:\n"
            "- ALLOW: Decide ALLOW only when ALL models report low attack probability (P < 0.35) or when non-zero detections are proven low-confidence background noise without corroboration.\n"
            "- FLAG: Decide FLAG when a specialized model indicates a credible attack (P >= 0.70) while other models are blind to it, OR when models show genuine probability conflicts (0.35 <= P < 0.65) requiring human SOC analyst quarantine and triage.\n"
            "- BLOCK: Decide BLOCK when high attack probability (P >= 0.70) is corroborated by multiple models, OR when a specialized model detects a severe attack with high probability (P >= 0.80) accompanied by fused risk >= 0.65, temporal persistence, or escalation.\n\n"
            "CONSTRAINTS:\n"
            "1. Choose exactly one action: ALLOW, FLAG, or BLOCK.\n"
            "2. attack_family MUST be one of: " + json.dumps(sorted(allowed_attacks)) + " or null if ALLOW/benign.\n"
            "3. Return ONLY strict JSON with these exact keys: "
            "action, severity, attack_family, confidence, reason, evidence_used, recommended_actions.\n"
            "4. severity must be: LOW, MEDIUM, HIGH, or CRITICAL.\n"
            "5. confidence is a float 0.0-1.0 representing your confidence in this policy decision.\n"
            "6. In 'reason', explicitly cite the specific model probabilities and domain specialization that justified your decision, proving no majority voting was used.\n"
        )

        user_prompt = "EVIDENCE SUMMARY:\n" + json.dumps(session_summary, default=str) + "\n\nDecide action and return strict JSON:"

        payload = {
            "model": self.llm_model,
            "prompt": system_prompt + "\n\n" + user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 384,
                "num_ctx": 1536,
            }
        }

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            text = body.get("response", "")
            if not text:
                return {"status": "error", "reason": "Empty response from Qwen local model"}

            report = self._clean_and_parse_json(text)
            return {"status": "ok", "report": report}
        except Exception as e:
            logger.warning("[PolicyAgent] Qwen inference error: %s", e)
            return {"status": "error", "reason": str(e)}

    # -----------------------------------------------------------------------
    def _parse_llm_response(
        self,
        report: dict,
        ev: IntegratedEvidence,
        ctx: dict,
    ) -> AgentDecision:
        """Parse and validate LLM's JSON response."""
        action   = str(report.get("action", "FLAG")).upper()
        severity = str(report.get("severity", "MEDIUM")).upper()

        if action   not in VALID_ACTIONS:    action   = "FLAG"
        if severity not in VALID_SEVERITIES: severity = "MEDIUM"

        allowed_families = {
            r.normalised_family for r in ev.available_models()
            if r.normalised_family
        } | {
            r.prediction for r in ev.available_models()
            if r.prediction not in ("Benign", "Unknown", "UNAVAILABLE", "ERROR")
        }
        raw_family   = report.get("attack_family", "")
        attack_family = raw_family if raw_family in allowed_families else ev.fused_attack_family

        return AgentDecision(
            action=action,
            severity=severity,
            confidence=float(report.get("confidence", ev.fused_attack_probability)),
            attack_family=attack_family,
            reason=str(report.get("reason", "")),
            evidence_used=list(report.get("evidence_used", [])),
            recommended_actions=list(report.get("recommended_actions", [])),
            source="qwen-2.5-7b",
            context_snapshot=ctx,
            gemini_raw=report,
        )

    # -----------------------------------------------------------------------
    def _compute_risk_score(
        self,
        ev: IntegratedEvidence,
        ctx: dict,
    ) -> tuple[float, dict]:
        """
        Compute transparent, normalized dynamic risk score in [0, 1].
        Evaluates model evidence, agreement, temporal persistence, escalation,
        flow density, and availability coverage.
        """
        attack_prob = ev.fused_attack_probability
        agreement = ev.model_agreement_score
        persistence = min(ctx.get("consecutive_attack_windows", 0) / 4.0, 1.0)
        esc_level = ctx.get("escalation_level", "NONE")
        esc_weight = {
            "NONE": 0.0, "LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6, "CRITICAL": 1.0
        }.get(esc_level, 0.0)

        # Contextual amplification: persistence & escalation amplify active suspicion,
        # but must not turn low-probability or uncorroborated traffic into a false alarm
        if attack_prob < 0.45 or agreement <= 0.25 or ev.models_indicating_attack <= 1:
            persistence = 0.0
            esc_weight = 0.0

        # Flow density from supporting evidence if available
        flow_density = 0.0
        if ev.flow and ev.flow.supporting_evidence:
            flow_density = float(ev.flow.supporting_evidence.get("fraction_attack_gt_070", 0.0))

        # Model availability coverage
        model_coverage = float(ev.models_available) / 4.0

        risk = (
            0.40 * attack_prob +
            0.20 * agreement +
            0.15 * persistence +
            0.15 * esc_weight +
            0.10 * min(flow_density * 2.0, 1.0)
        )
        risk = float(np.clip(risk, 0.0, 1.0))

        components = {
            "attack_probability": round(attack_prob, 4),
            "model_agreement":    round(agreement, 4),
            "persistence":        round(persistence, 4),
            "escalation":         round(esc_weight, 4),
            "flow_density":       round(flow_density, 4),
            "model_coverage":     round(model_coverage, 4),
        }
        return round(risk, 4), components

    # -----------------------------------------------------------------------
    def _heuristic_decision(
        self,
        ev: IntegratedEvidence,
        ctx: dict,
    ) -> AgentDecision:
        """
        Deterministic policy reasoning based on dynamic risk score.
        """
        risk_score, risk_components = self._compute_risk_score(ev, ctx)

        if risk_score >= self.block_t:
            families = [
                r.normalised_family for r in [ev.packet, ev.flow, ev.time_window, ev.session]
                if r and r.is_attack and r.normalised_family
            ]
            if len(set(families)) > 1:
                action = "FLAG"
                sev = "HIGH"
            else:
                action = "BLOCK"
                sev = "CRITICAL" if risk_score >= 0.85 else "HIGH"
        elif risk_score >= self.flag_t:
            action = "FLAG"
            sev = "MEDIUM"
        else:
            action = "ALLOW"
            sev = "LOW"

        reason = (
            f"Dynamic risk score R={risk_score:.3f} "
            f"(P_att={risk_components['attack_probability']}, "
            f"agr={risk_components['model_agreement']}, "
            f"pers={risk_components['persistence']}, "
            f"esc={risk_components['escalation']}). Action={action}."
        )

        return AgentDecision(
            action=action,
            severity=sev,
            confidence=round(ev.fused_attack_probability, 4),
            attack_family=ev.fused_attack_family,
            reason=reason,
            evidence_used=self._summarise_evidence(ev),
            source="heuristic",
            risk_score=risk_score,
            risk_components=risk_components,
            context_snapshot=ctx,
        )

    # -----------------------------------------------------------------------
    def _summarise_evidence(self, ev: IntegratedEvidence) -> list:
        parts = []
        for rec in [ev.packet, ev.flow, ev.time_window, ev.session]:
            if rec and rec.status == "ok":
                parts.append(
                    f"{rec.model_name}: {rec.prediction} "
                    f"(P(attack)={rec.attack_probability:.3f})"
                )
        return parts

    # -----------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "gemini_calls": self._llm_calls,
            "gemini_errors": self._llm_errors,
            "llm_calls": self._llm_calls,
            "llm_errors": self._llm_errors,
            "llm_model": self.llm_model,
            "confusing_cases_dispatched": self._confusing_dispatched,
            "deterministic_handled": self._deterministic_handled,
        }
