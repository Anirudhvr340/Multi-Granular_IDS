"""
Policy Agent  —  Stage 4
=========================
Dynamic reasoning agent backed by Gemini.

The agent is NOT a classifier.  The four ML models are the classifiers.
The agent RECEIVES their evidence and DECIDES: ALLOW | FLAG | BLOCK.

The agent:
1. Receives structured evidence from all four models.
2. Receives context state (escalation, persistence, history).
3. Receives rule-engine result (if any).
4. Constructs a structured SESSION SUMMARY for Gemini.
5. Sends the summary to Gemini with a constrained prompt.
6. Validates and parses Gemini's JSON response.
7. Falls back to a deterministic heuristic if Gemini fails.

Safety constraints
------------------
- Gemini cannot invent attack classes (restricted to model predictions).
- Gemini cannot produce actions outside ALLOW | FLAG | BLOCK.
- All responses are validated before use.
- Gemini is never given raw traffic — only structured model evidence.

Fallback policy
---------------
If Gemini is unavailable or produces invalid output, a rule-based heuristic
makes the final decision based solely on fused_attack_probability.
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
    source:             str    = "agent"   # "gemini" | "heuristic" | "rule"
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

    Performs:  OBSERVE → INTERPRET → EVALUATE → CONSIDER HISTORY → DECIDE → ACT

    Usage
    -----
    agent = PolicyAgent()
    decision = agent.decide(integrated, context_summary, rule_result)
    """

    GEMINI_MODEL_DEFAULT = "gemini-1.5-flash"

    def __init__(
        self,
        gemini_model:        Optional[str]  = None,
        gemini_api_key:      Optional[str]  = None,
        allow_threshold:     float = 0.25,
        flag_threshold:      float = 0.35,
        block_threshold:     float = 0.65,
        use_gemini:          bool  = True,
    ):
        raw_model = gemini_model or os.environ.get("GEMINI_MODEL", self.GEMINI_MODEL_DEFAULT)
        if "gemini-2.5" in raw_model:
            raw_model = "gemini-1.5-flash"
        self._model      = raw_model
        self._api_key    = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.allow_t     = allow_threshold
        self.flag_t      = flag_threshold
        self.block_t     = block_threshold
        self.use_gemini  = use_gemini
        self._gemini_calls = 0
        self._gemini_errors = 0

    # -----------------------------------------------------------------------
    def decide(
        self,
        integrated:      IntegratedEvidence,
        context_summary: dict,
        rule_result:     Optional[RuleResult] = None,
    ) -> AgentDecision:
        """
        Main entry point.  Returns an AgentDecision.

        If rule_result is not None, the rule engine already decided — we
        wrap that decision in an AgentDecision without calling Gemini.
        """

        # ---- Compute transparent dynamic risk score ------------------------
        risk_score, risk_components = self._compute_risk_score(integrated, context_summary)

        # ---- If a deterministic rule fired, use it -------------------------
        if rule_result is not None:
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

        # ---- Build session summary for Gemini ------------------------------
        session_summary = self._build_session_summary(integrated, context_summary)
        session_summary["risk_score"] = risk_score
        session_summary["risk_components"] = risk_components

        # ---- Try Gemini ----------------------------------------------------
        gemini_result = None
        if self.use_gemini and self._api_key:
            gemini_result = self._call_gemini(session_summary, integrated)

        # ---- Parse or fall back --------------------------------------------
        if gemini_result and gemini_result.get("status") == "ok":
            decision = self._parse_gemini(gemini_result["report"], integrated, context_summary)
            self._gemini_calls += 1
        else:
            decision = self._heuristic_decision(integrated, context_summary)
            if gemini_result and gemini_result.get("status") == "error":
                self._gemini_errors += 1
                decision.reason += f" [Gemini unavailable: {gemini_result.get('reason', '')}]"

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
        Build the structured prompt payload sent to Gemini.
        Gemini NEVER sees raw traffic — only model evidence.
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
            # Fallback regex extraction if text was cut off or has minor syntax errors
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
    def _call_gemini(self, session_summary: dict, ev: IntegratedEvidence) -> dict:
        """Send session summary to Gemini; return parsed result dict with retry and schema validation."""

        # Build allowed attack types from model predictions only
        allowed_attacks = set()
        for record in ev.available_models():
            if record.normalised_family:
                allowed_attacks.add(record.normalised_family)
            if record.prediction not in ("Benign", "Unknown", "UNAVAILABLE", "ERROR"):
                allowed_attacks.add(record.prediction)

        system_prompt = (
            "You are the policy-reasoning layer of a network intrusion detection system. "
            "The ML models have already made all attack predictions. "
            "Your role is to reason over their combined evidence and decide an action.\n\n"
            "RULES:\n"
            "1. Choose exactly one action: ALLOW, FLAG, or BLOCK.\n"
            "2. attack_family MUST come from: " + json.dumps(sorted(allowed_attacks)) + " or null if benign.\n"
            "3. Do NOT invent attack types or change predicted labels.\n"
            "4. Return ONLY strict JSON with these exact keys: "
            "action, severity, attack_family, confidence, reason, evidence_used, recommended_actions.\n"
            "5. severity must be: LOW, MEDIUM, HIGH, or CRITICAL.\n"
            "6. confidence is a float 0.0-1.0.\n"
        )

        payload = {
            "contents": [{
                "parts": [{
                    "text": system_prompt + "\n\nEVIDENCE:\n" + json.dumps(session_summary, default=str)
                }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 1024,
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "action": {
                            "type": "STRING",
                            "enum": ["ALLOW", "FLAG", "BLOCK"]
                        },
                        "severity": {
                            "type": "STRING",
                            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                        },
                        "attack_family": {
                            "type": "STRING"
                        },
                        "confidence": {
                            "type": "NUMBER"
                        },
                        "reason": {
                            "type": "STRING"
                        },
                        "evidence_used": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "recommended_actions": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["action", "severity", "confidence", "reason"]
                }
            },
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())

                candidates = body.get("candidates", [])
                if not candidates or "content" not in candidates[0]:
                    finish_reason = candidates[0].get("finishReason", "UNKNOWN") if candidates else "NO_CANDIDATES"
                    return {"status": "error", "reason": f"No content returned (finishReason: {finish_reason})"}

                parts = candidates[0]["content"].get("parts", [])
                if not parts or "text" not in parts[0]:
                    return {"status": "error", "reason": "No text in Gemini response candidate parts"}

                text = parts[0]["text"]
                report = self._clean_and_parse_json(text)
                return {"status": "ok", "report": report}

            except urllib.error.HTTPError as e:
                details = e.read().decode("utf-8", errors="replace")[:500]
                if e.code == 429 and attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) * 2.5
                    logger.warning("[PolicyAgent] HTTP 429 rate limit hit. Retrying in %.1fs (attempt %d/%d)...", sleep_time, attempt + 1, max_retries)
                    time.sleep(sleep_time)
                    continue
                return {"status": "error", "reason": f"HTTP {e.code}: {details}"}
            except Exception as e:
                return {"status": "error", "reason": str(e)}

        return {"status": "error", "reason": "Max retries exceeded"}

    # -----------------------------------------------------------------------
    def _parse_gemini(
        self,
        report: dict,
        ev: IntegratedEvidence,
        ctx: dict,
    ) -> AgentDecision:
        """Parse and validate Gemini's JSON response."""

        action   = str(report.get("action", "FLAG")).upper()
        severity = str(report.get("severity", "MEDIUM")).upper()

        if action   not in VALID_ACTIONS:    action   = "FLAG"
        if severity not in VALID_SEVERITIES: severity = "MEDIUM"

        # Validate attack_family — must come from model predictions
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
            source="gemini",
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

        # Flow density from supporting evidence if available
        flow_density = 0.0
        if ev.flow and ev.flow.supporting_evidence:
            flow_density = float(ev.flow.supporting_evidence.get("fraction_attack_gt_070", 0.0))

        # Model availability coverage (e.g. 2/4 = 0.5 currently)
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
        Fallback deterministic policy reasoning based on dynamic risk score.
        """
        risk_score, risk_components = self._compute_risk_score(ev, ctx)

        if risk_score >= self.block_t:
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
            "gemini_calls": self._gemini_calls,
            "gemini_errors": self._gemini_errors,
        }

