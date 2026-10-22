"""
Action Executor  —  Stage 5
============================
Receives ALLOW | FLAG | BLOCK from the Policy Agent and creates
structured decision artifacts on disk.

The executor does NOT run system commands or touch firewall rules.
It writes JSON decision artifacts and optionally maintains a decision log.

Artifact format
---------------
decision_{window_id}_{timestamp}.json:
{
    "timestamp":      "ISO-8601",
    "window_id":      "...",
    "action":         "BLOCK",
    "severity":       "HIGH",
    "attack_family":  "DDoS",
    "confidence":     0.92,
    "reason":         "...",
    "source":         "gemini | heuristic | rule",
    "rule_triggered": "R2_EXTREME_PROB | null",
    "models": {
        "packet":      {...},
        "flow":        {...},
        "time_window": {...},
        "session":     {...}
    },
    "context":        {...},
    "recommended_actions": [...]
}
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from nids.evidence_schema import IntegratedEvidence
from nids.policy_agent import AgentDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent   # repo root


def _json_default(obj):
    """JSON serialisation helper for numpy types etc."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
class ActionExecutor:
    """
    Stage 5: Action executor.

    Usage
    -----
    executor = ActionExecutor(output_dir="nids/decisions")
    artifact = executor.execute(decision, integrated_evidence)
    print(artifact["path"])   # path to written JSON
    """

    VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK"}

    def __init__(
        self,
        output_dir: str | None = None,
        append_to_log: bool = True,
    ):
        self.output_dir    = Path(output_dir) if output_dir else HERE / "nids" / "decisions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path      = self.output_dir / "decision_log.jsonl"
        self.append_to_log = append_to_log

        # Session-level counters
        self._counts = {"ALLOW": 0, "FLAG": 0, "BLOCK": 0, "UNKNOWN": 0}

    # -----------------------------------------------------------------------
    def execute(
        self,
        decision: AgentDecision,
        evidence: IntegratedEvidence,
    ) -> dict:
        """
        Validate the decision, write a JSON artifact, and return a summary.
        """

        # ---- Validate action -----------------------------------------------
        action = str(decision.action).upper()
        if action not in self.VALID_ACTIONS:
            logger.warning("ActionExecutor: invalid action '%s' → defaulting to FLAG", action)
            action = "FLAG"

        # ---- Build artifact ------------------------------------------------
        ts_str = datetime.now(tz=timezone.utc).isoformat()
        wid    = evidence.window_id or f"w{int(evidence.timestamp)}"

        artifact = {
            "timestamp":      ts_str,
            "window_id":      wid,
            "action":         action,
            "severity":       decision.severity,
            "attack_family":  decision.attack_family,
            "confidence":     round(decision.confidence, 6),
            "reason":         decision.reason,
            "source":         decision.source,
            "rule_triggered": decision.rule_triggered,
            "risk_score":     round(getattr(decision, "risk_score", 0.0), 4),
            "risk_components": getattr(decision, "risk_components", {}),
            "models": {
                "packet":      evidence.packet.to_dict()      if evidence.packet      else None,
                "flow":        evidence.flow.to_dict()        if evidence.flow        else None,
                "time_window": evidence.time_window.to_dict() if evidence.time_window else None,
                "session":     evidence.session.to_dict()     if evidence.session     else None,
            },
            "fused_evidence": {
                "attack_probability": evidence.fused_attack_probability,
                "attack_family":      evidence.fused_attack_family,
                "model_agreement":    evidence.model_agreement_score,
                "models_available":   evidence.models_available,
                "models_attacking":   evidence.models_indicating_attack,
            },
            "context":        decision.context_snapshot,
            "recommended_actions": decision.recommended_actions,
        }

        # ---- Write individual artifact -------------------------------------
        safe_wid = re.sub(r"[^\w\-.]", "_", str(wid))
        safe_ts  = ts_str.replace(":", "-").replace("+", "Z").split(".")[0]
        fname    = f"decision_{safe_wid}_{safe_ts}.json"
        fpath    = self.output_dir / fname
        fpath.write_text(
            json.dumps(artifact, indent=2, default=_json_default),
            encoding="utf-8",
        )

        # ---- Append to rolling decision log --------------------------------
        if self.append_to_log:
            log_line = json.dumps(
                {k: artifact[k] for k in
                 ("timestamp", "window_id", "action", "severity",
                  "attack_family", "confidence", "source", "rule_triggered")},
                default=_json_default,
            )
            with open(self.log_path, "a", encoding="utf-8") as lf:
                lf.write(log_line + "\n")

        # ---- Update counters -----------------------------------------------
        self._counts[action] = self._counts.get(action, 0) + 1

        # ---- Log to console / logger ----------------------------------------
        level = logging.WARNING if action == "BLOCK" else logging.INFO
        logger.log(level, "[ActionExecutor] %s → %s (%s) | %s", wid, action, decision.severity, decision.reason[:80])

        return {"path": str(fpath), "action": action, "artifact": artifact}

    # -----------------------------------------------------------------------
    @property
    def action_counts(self) -> dict:
        return dict(self._counts)

    def action_distribution(self) -> dict:
        total = sum(self._counts.values()) or 1
        return {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in self._counts.items()
        }

