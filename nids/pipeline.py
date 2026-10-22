"""
Multi-Granular IDS Pipeline
============================
Orchestrates all five stages for a CICIDS-2018 input CSV.

Architecture
------------

   CICIDS-2018 CSV
         |
   ┌─────┴──────────────────────────────────┐
   │  per-flow preprocessing (step0)        │
   └─────┬──────────────────────────────────┘
         │  window grouping
   ┌─────┴──────────────────────────────────┐
   │ FLOW MODEL   TIME-WINDOW MODEL         │
   │ (per flow)   (per window aggregate)    │
   │                                        │
   │ PACKET MODEL  SESSION MODEL            │
   │ (STUB)        (STUB)                   │
   └─────┬──────────────────────────────────┘
         │  EvidenceRecord × 4
   ┌─────▼──────────────────────────────────┐
   │  Stage 1: Attack Gate                  │
   │  P(attack) fusion + family routing     │
   └─────┬──────────────────────────────────┘
         │  IntegratedEvidence
   ┌─────▼──────────────────────────────────┐
   │  Stage 2: Rule Engine                  │
   │  deterministic fast-path rules         │
   └─────┬──────────────────────────────────┘
         │  rule_result (or None)
   ┌─────▼──────────────────────────────────┐
   │  Stage 3: Context Engine               │
   │  stateful memory / escalation          │
   └─────┬──────────────────────────────────┘
         │  context_summary
   ┌─────▼──────────────────────────────────┐
   │  Stage 4: Policy Agent (Gemini)        │
   │  OBSERVE → INTERPRET → DECIDE          │
   └─────┬──────────────────────────────────┘
         │  AgentDecision
   ┌─────▼──────────────────────────────────┐
   │  Stage 5: Action Executor              │
   │  ALLOW | FLAG | BLOCK  → artifact      │
   └────────────────────────────────────────┘

Usage
-----
from nids.pipeline import NIDSPipeline

pipe = NIDSPipeline()
results = pipe.run("02-14-2018.csv", window_seconds=30, report="report.json")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure time_window helpers are importable
HERE = Path(__file__).resolve().parent.parent
TIME_WINDOW_DIR = HERE / "time_window"
if str(TIME_WINDOW_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_WINDOW_DIR))

from nids.adapters.flow_adapter       import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.packet_adapter     import PacketModelAdapter
from nids.adapters.session_adapter    import SessionModelAdapter
from nids.attack_gate                 import AttackGate
from nids.rule_engine                 import RuleEngine, RuleConfig
from nids.context_engine              import ContextEngine
from nids.policy_agent                import PolicyAgent
from nids.action_executor             import ActionExecutor
from nids.evidence_schema             import IntegratedEvidence

logger = logging.getLogger(__name__)

_JSON_DEFAULT = lambda v: v.item() if isinstance(v, np.generic) else str(v)


# ---------------------------------------------------------------------------
class NIDSPipeline:
    """
    Full four-model hierarchical evidence-fusion pipeline.

    Parameters
    ----------
    attack_threshold : float
        Fused P(attack) threshold (default 0.35).
    use_gemini : bool
        Whether to call Gemini for policy reasoning.
    output_dir : str | None
        Where to write decision artifacts (default: nids/decisions/).
    """

    def __init__(
        self,
        attack_threshold: float = 0.35,
        use_gemini: bool = True,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.attack_threshold = attack_threshold

        # ---- Initialise model adapters ------------------------------------
        logger.info("Loading model adapters …")
        self.flow_adapter       = FlowModelAdapter(attack_threshold=attack_threshold)
        self.time_window_adapter = TimeWindowAdapter(attack_threshold=attack_threshold)
        self.packet_adapter     = PacketModelAdapter(attack_threshold=attack_threshold)
        self.session_adapter    = SessionModelAdapter(attack_threshold=attack_threshold)

        # ---- Initialise pipeline stages -----------------------------------
        self.attack_gate  = AttackGate(threshold=attack_threshold)
        self.rule_engine  = RuleEngine(config=RuleConfig())
        self.context_engine = ContextEngine(max_windows=20)
        self.policy_agent  = PolicyAgent(
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
        self.executor      = ActionExecutor(output_dir=output_dir)

        logger.info("Pipeline initialised.")

    # -----------------------------------------------------------------------
    def run(
        self,
        input_csv: str,
        window_seconds: int = 30,
        report_path: str = "nids_pipeline_report.json",
        no_llm: bool = False,
        max_windows: Optional[int] = None,
    ) -> dict:
        """
        Process a CICIDS-2018 CSV end-to-end.

        Returns the complete report dict.
        """
        t_start = time.perf_counter()

        os.environ["WINDOW_SECONDS"] = str(window_seconds)

        import step0_preprocessing as preprocessing

        logger.info("Reading %s …", input_csv)
        frame = pd.read_csv(input_csv, low_memory=False)

        # Validate schema
        required = ["Timestamp", "Label", "Dst Port", "Protocol",
                    "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Duration"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        frame = preprocessing.clean_chunk(frame)
        if frame.empty:
            raise ValueError("No valid 2018 flow records found in input.")

        frame["window_start"] = frame["Timestamp"].dt.floor(f"{window_seconds}s")

        # ---- Process windows in temporal order ----------------------------
        window_results = []
        self.time_window_adapter.reset_history()
        self.context_engine.reset()

        for w_idx, (window_start, window_df) in enumerate(
            frame.groupby("window_start", sort=True)
        ):
            if max_windows is not None and w_idx >= max_windows:
                break
            window_id = f"W{w_idx:05d}_{window_start}"
            result = self._process_window(
                window_df, window_start, window_id, window_seconds,
                preprocessing, no_llm,
            )
            window_results.append(result)

            if w_idx % 50 == 0:
                logger.info("Processed %d windows …", w_idx + 1)

        # ---- Compile report -----------------------------------------------
        elapsed = round(time.perf_counter() - t_start, 2)
        report  = self._compile_report(
            input_csv, window_seconds, window_results, elapsed
        )
        _write_report(report, report_path)

        actions = self.executor.action_distribution()
        print(json.dumps({
            "windows_processed": len(window_results),
            "elapsed_seconds": elapsed,
            "actions": actions,
            "gemini_stats": self.policy_agent.stats,
        }, indent=2))
        print(f"Full report -> {Path(report_path).resolve()}")

        return report

    # -----------------------------------------------------------------------
    def _process_window(
        self,
        window_df:      pd.DataFrame,
        window_start,
        window_id:      str,
        window_seconds: int,
        preprocessing,
        no_llm:         bool,
    ) -> dict:
        """Run the full pipeline for a single time window."""

        # ---- Level 1: Individual model outputs ----------------------------
        # Flow model — per-flow inference, max-pooled over the window
        flow_ev = self.flow_adapter.predict(window_df, window_id=window_id)

        # Time-window model — aggregated window features
        aggregate = preprocessing.window_aggregate(window_df, window_start)
        tw_ev = self.time_window_adapter.predict(aggregate, window_id=window_id)

        # Packet model — dimensionless packet header 1D-CNN
        pkt_ev = self.packet_adapter.predict(window_df, window_id=window_id)

        # Session model — chronological sequence CNN-LSTM
        sess_ev = self.session_adapter.predict(window_df, window_id=window_id)

        # ---- Level 2: Attack gate (fused evidence) ------------------------
        integrated: IntegratedEvidence = self.attack_gate.evaluate(
            packet_ev=pkt_ev,
            flow_ev=flow_ev,
            time_ev=tw_ev,
            session_ev=sess_ev,
            window_id=window_id,
        )

        # ---- Stage 2: Rule engine -----------------------------------------
        ctx_state    = self.context_engine.get_state()
        ctx_summary  = self.context_engine.summary()
        rule_result  = self.rule_engine.evaluate(integrated, ctx_summary)

        # ---- Stage 3: Update context (with evidence, before decision) -----
        self.context_engine.update(integrated)
        ctx_summary = self.context_engine.summary()

        # ---- Stage 4: Policy agent ----------------------------------------
        if no_llm:
            self.policy_agent.use_gemini = False

        decision = self.policy_agent.decide(integrated, ctx_summary, rule_result)

        # ---- Stage 5: Action executor -------------------------------------
        artifact  = self.executor.execute(decision, integrated)
        self.context_engine.record_action(decision.action)

        # ---- Ground truth from window label -------------------------------
        gt_label = aggregate.get("majority_label", "Unknown")

        return {
            "window_id":    window_id,
            "window_start": str(window_start),
            "ground_truth": gt_label,
            "n_flows":      len(window_df),

            # Level 1 — individual model outputs
            "level1": {
                "packet":      pkt_ev.to_dict(),
                "flow":        flow_ev.to_dict(),
                "time_window": tw_ev.to_dict(),
                "session":     sess_ev.to_dict(),
            },
            # Level 2 — fused evidence
            "level2": integrated.to_dict(),
            # Level 3 — agent decision
            "level3": decision.to_dict(),

            "action":    decision.action,
            "severity":  decision.severity,
            "source":    decision.source,
            "rule":      rule_result.rule_name if rule_result else None,
            "artifact":  artifact["path"],
        }

    # -----------------------------------------------------------------------
    def _compile_report(
        self,
        input_csv:      str,
        window_seconds: int,
        results:        list,
        elapsed:        float,
    ) -> dict:
        actions = [r["action"] for r in results]
        sources = [r["source"] for r in results]

        from collections import Counter
        action_dist = dict(Counter(actions))
        source_dist = dict(Counter(sources))

        # Agreement stats
        agreements = [
            r["level2"]["model_agreement_score"]
            for r in results if r["level2"]
        ]
        avg_agreement = float(np.mean(agreements)) if agreements else 0.0

        # Attack windows
        attack_windows = [r for r in results if r["action"] in ("FLAG", "BLOCK")]

        return {
            "input":            str(Path(input_csv).resolve()),
            "window_seconds":   window_seconds,
            "architecture":     "packet+flow+time_window+session → evidence_fusion → rule_engine → context → agent → ALLOW/FLAG/BLOCK",
            "elapsed_seconds":  elapsed,
            "windows_total":    len(results),
            "action_distribution": action_dist,
            "source_distribution": source_dist,
            "average_model_agreement": round(avg_agreement, 4),
            "windows_flagged_or_blocked": len(attack_windows),
            "gemini_stats":     self.policy_agent.stats,
            "windows":          results,
        }


# ---------------------------------------------------------------------------
def _write_report(report: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, indent=2, default=_JSON_DEFAULT),
        encoding="utf-8",
    )

