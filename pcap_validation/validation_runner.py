#!/usr/bin/env python3
"""
Unified PCAP Validation Orchestrator & CLI Runner
Processes raw PCAP files, coordinates all four model branches, and executes agentic decisions.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Path setup
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from pcap_validation.pcap_reader import PCAPReader
from pcap_validation.cicflowmeter_runner import CICFlowMeterRunner
from pcap_validation.packet_feature_extractor import PacketFeatureExtractor
from pcap_validation.session_sequence_builder import SessionSequenceBuilder
from pcap_validation.flow_preprocessor import FlowPreprocessor
from pcap_validation.window_aggregator import WindowAggregator
from pcap_validation.evidence_aligner import EvidenceAligner

from nids.rule_engine import RuleEngine, RuleConfig
from nids.context_engine import ContextEngine
from nids.policy_agent import PolicyAgent
from nids.action_executor import ActionExecutor
import step0_preprocessing as preprocessing


class PCAPValidationOrchestrator:
    """
    Coordinates the four-model validation pipeline from a single PCAP source.
    """

    def __init__(
        self,
        attack_threshold: float = 0.35,
        window_seconds: int = 30,
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-flash",
        output_dir: Optional[str] = None,
    ):
        self.attack_threshold = attack_threshold
        self.window_seconds = window_seconds
        self.output_dir = Path(output_dir or HERE / "reports").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("[*] Initializing 4-Model PCAP Validation Orchestrator...")
        self.flow_runner = FlowPreprocessor(attack_threshold=attack_threshold)
        self.window_runner = WindowAggregator(attack_threshold=attack_threshold)
        self.packet_runner = PacketFeatureExtractor(attack_threshold=attack_threshold)
        self.session_runner = SessionSequenceBuilder(attack_threshold=attack_threshold)

        self.aligner = EvidenceAligner(attack_threshold=attack_threshold)
        self.rule_engine = RuleEngine(config=RuleConfig())
        self.context_engine = ContextEngine(max_windows=20)
        self.policy_agent = PolicyAgent(
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
        self.executor = ActionExecutor(output_dir=str(self.output_dir))
        print("[+] All model branches and agentic layers initialized.")

    def run_on_pcap(
        self,
        pcap_path: str | Path,
        max_packets: Optional[int] = None,
        max_windows: Optional[int] = None,
        active_branch: str = "all",  # "all", "packet", "flow", "time_window", "session"
        ground_truth_label: str = "Benign",
        label_csv: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        pcap_path = Path(pcap_path).resolve()
        t_start = time.perf_counter()

        print("=" * 80)
        print(f"RUNNING PCAP VALIDATION ON: {pcap_path.name}")
        print(f"Active Branch: {active_branch.upper()} | Window Duration: {self.window_seconds}s")
        print("=" * 80)

        # Step 1: Reassemble PCAP into CICIDS flow DataFrame
        cfm = CICFlowMeterRunner()
        intermediate_csv = self.output_dir / f"intermediate_flows_{pcap_path.stem}.csv"
        flow_df = cfm.convert_pcap_to_flow_df(pcap_path, max_packets=max_packets)

        if flow_df.empty:
            raise ValueError(f"No valid IP packets or flows found in {pcap_path}")

        print(f"[+] Reassembled {len(flow_df):,} bidirectional flows from PCAP.")

        # Step 2: Incorporate external ground truth if available
        if label_csv and Path(label_csv).exists():
            print(f"[*] Merging labels from external ground-truth CSV: {label_csv}")
            try:
                lbl_df = pd.read_csv(label_csv, usecols=["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Label"], low_memory=False)
                lbl_df.columns = lbl_df.columns.str.strip()
                flow_df = flow_df.merge(lbl_df, on=["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"], how="left", suffixes=("", "_gt"))
                if "Label_gt" in flow_df.columns:
                    flow_df["Label"] = flow_df["Label_gt"].fillna(ground_truth_label)
            except Exception as e:
                print(f"[!] Ground-truth merge warning: {e}")
        else:
            flow_df["Label"] = ground_truth_label

        flow_df.to_csv(intermediate_csv, index=False)
        print(f"[+] Saved intermediate flow artifact to: {intermediate_csv.name}")

        # Step 3: Segment into chronological 30-second windows
        cleaned_df = preprocessing.clean_chunk(flow_df.copy())
        cleaned_df["window_start"] = cleaned_df["Timestamp"].dt.floor(f"{self.window_seconds}s")

        windows_processed = []
        flow_predictions = []
        self.context_engine.reset()

        for w_idx, (window_start, window_df) in enumerate(cleaned_df.groupby("window_start", sort=True)):
            if max_windows is not None and w_idx >= max_windows:
                break

            window_id = f"W{w_idx:05d}_{window_start}"

            # Branch execution
            pkt_ev = None
            flow_ev = None
            tw_ev = None
            sess_ev = None

            if active_branch in ("all", "packet"):
                pkt_ev = self.packet_runner.predict_window(window_df, window_id=window_id)
            if active_branch in ("all", "flow"):
                flow_ev = self.flow_runner.predict_window(window_df, window_id=window_id)
            if active_branch in ("all", "time_window"):
                agg = self.window_runner.aggregate_window(window_df, window_start)
                tw_ev = self.window_runner.predict_window(agg, window_id=window_id)
            if active_branch in ("all", "session"):
                sess_ev = self.session_runner.predict_window(window_df, window_id=window_id)

            # Evidence fusion
            integrated = self.aligner.align_and_fuse(
                packet_ev=pkt_ev,
                flow_ev=flow_ev,
                time_ev=tw_ev,
                session_ev=sess_ev,
                window_id=window_id,
            )

            # Agentic decision pipeline
            ctx_summary = self.context_engine.summary()
            rule_result = self.rule_engine.evaluate(integrated, ctx_summary)
            self.context_engine.update(integrated)
            ctx_summary = self.context_engine.summary()

            decision = self.policy_agent.decide(integrated, ctx_summary, rule_result)
            self.executor.execute(decision, integrated)
            self.context_engine.record_action(decision.action)

            # Determine ground truth for window (attack presence)
            non_benign = [lbl for lbl in window_df["Label"] if "Benign" not in str(lbl)]
            window_contains_attack = len(non_benign) > 0
            gt_label = non_benign[0] if non_benign else "Benign"

            window_record = {
                "window_id": window_id,
                "window_start": str(window_start),
                "n_flows": len(window_df),
                "ground_truth": gt_label,
                "contains_attack": window_contains_attack,
                "action": decision.action,
                "severity": decision.severity,
                "fused_attack_prob": integrated.fused_attack_probability,
                "model_agreement": integrated.model_agreement_score,
                "rule_triggered": decision.rule_triggered,
                "source": decision.source,
                "packet_prob": pkt_ev.attack_probability if pkt_ev else None,
                "flow_prob": flow_ev.attack_probability if flow_ev else None,
                "window_prob": tw_ev.attack_probability if tw_ev else None,
                "session_prob": sess_ev.attack_probability if sess_ev else None,
            }
            windows_processed.append(window_record)

        total_elapsed = time.perf_counter() - t_start

        # Generate outputs and reports
        results_summary = self._compile_and_save_reports(
            pcap_path=pcap_path,
            windows=windows_processed,
            flow_df=flow_df,
            elapsed_seconds=total_elapsed,
            active_branch=active_branch,
        )

        return results_summary

    def _compile_and_save_reports(
        self,
        pcap_path: Path,
        windows: List[Dict[str, Any]],
        flow_df: pd.DataFrame,
        elapsed_seconds: float,
        active_branch: str,
    ) -> Dict[str, Any]:
        """Compiles CSV/JSON/MD reports and artifacts."""
        # 1. Window-level predictions CSV
        win_df = pd.DataFrame(windows)
        win_csv = self.output_dir / "window_level_predictions.csv"
        win_df.to_csv(win_csv, index=False)

        # 2. Flow-level predictions CSV
        flow_csv = self.output_dir / "flow_level_predictions.csv"
        flow_df[["Timestamp", "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Label"]].to_csv(flow_csv, index=False)

        # 3. Integrated decisions CSV
        dec_csv = self.output_dir / "integrated_decisions.csv"
        win_df[["window_id", "ground_truth", "contains_attack", "action", "severity", "fused_attack_prob", "rule_triggered", "source"]].to_csv(dec_csv, index=False)

        # 4. Model execution audit JSON
        audit_json = self.output_dir / "model_execution_audit.json"
        audit_data = {
            "pcap_file": pcap_path.name,
            "windows_count": len(windows),
            "flows_count": len(flow_df),
            "model_stats": self.aligner.get_audit_summary(),
        }
        audit_json.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")

        # 5. Validation run manifest JSON
        hasher = hashlib.sha256()
        with open(pcap_path, "rb") as f:
            hasher.update(f.read(1024 * 1024))  # sample checksum
        manifest_json = self.output_dir / "validation_run_manifest.json"
        manifest_data = {
            "pcap_name": pcap_path.name,
            "pcap_sha256_head": hasher.hexdigest(),
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "active_branch": active_branch,
            "window_seconds": self.window_seconds,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
        manifest_json.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        # 6. Per-model metrics CSV & Summary calculation
        actions = Counter(w["action"] for w in windows)
        gt_attacks = sum(1 for w in windows if w["contains_attack"])
        detected = sum(1 for w in windows if w["contains_attack"] and w["action"] in ("BLOCK", "FLAG"))
        recall = detected / gt_attacks if gt_attacks > 0 else 1.0

        metrics_csv = self.output_dir / "per_model_metrics.csv"
        metrics_df = pd.DataFrame([
            {"model": "Flow XGBoost", "granularity": "flow", "invoked": self.aligner.stats["flow"]["invoked"], "success": self.aligner.stats["flow"]["success"]},
            {"model": "Time-Window XGBoost", "granularity": "window", "invoked": self.aligner.stats["time_window"]["invoked"], "success": self.aligner.stats["time_window"]["success"]},
            {"model": "Packet 1D-CNN", "granularity": "packet", "invoked": self.aligner.stats["packet"]["invoked"], "success": self.aligner.stats["packet"]["success"]},
            {"model": "Session CNN-LSTM", "granularity": "session", "invoked": self.aligner.stats["session"]["invoked"], "success": self.aligner.stats["session"]["success"]},
        ])
        metrics_df.to_csv(metrics_csv, index=False)

        # 7. Comprehensive Markdown Report
        report_md = self.output_dir / "pcap_validation_report.md"
        report_text = f"""# Unified PCAP Validation Report

**Dataset Source:** `{pcap_path.name}`  
**Evaluation Mode:** `{active_branch.upper()}`  
**Run Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
**Execution Time:** `{elapsed_seconds:.2f} seconds`  

---

## 1. Execution Summary
- **Total Packets Reassembled into Flows:** {len(flow_df):,} flows
- **30-Second Windows Scored:** {len(windows):,} windows
- **Attack Windows Present:** {gt_attacks}
- **Attack Windows Detected (BLOCK + FLAG):** {detected} ({recall*100:.2f}% recall)

## 2. Action Distribution
- **ALLOW:** {actions['ALLOW']} ({actions['ALLOW']/max(1, len(windows))*100:.1f}%)
- **FLAG:** {actions['FLAG']} ({actions['FLAG']/max(1, len(windows))*100:.1f}%)
- **BLOCK:** {actions['BLOCK']} ({actions['BLOCK']/max(1, len(windows))*100:.1f}%)

## 3. Model Branch Participation
- **Flow XGBoost:** {self.aligner.stats['flow']['success']} / {self.aligner.stats['flow']['invoked']} successful invocations
- **Time-Window XGBoost:** {self.aligner.stats['time_window']['success']} / {self.aligner.stats['time_window']['invoked']} successful invocations
- **Packet 1D-CNN:** {self.aligner.stats['packet']['success']} / {self.aligner.stats['packet']['invoked']} successful invocations
- **Session CNN-LSTM:** {self.aligner.stats['session']['success']} / {self.aligner.stats['session']['invoked']} successful invocations

All raw logs and prediction files are saved in `{self.output_dir}`.
"""
        report_md.write_text(report_text, encoding="utf-8")
        print(f"\n[+] Validation Report generated: {report_md}")

        return {
            "pcap_file": pcap_path.name,
            "windows_processed": len(windows),
            "flows_processed": len(flow_df),
            "actions": dict(actions),
            "attack_recall": recall,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "report_path": str(report_md),
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-Granular NIDS PCAP Validation Orchestrator")
    parser.add_argument("pcap_path", help="Path to input .pcap file or directory of PCAPs")
    parser.add_argument("--branch", default="all", choices=["all", "packet", "flow", "time_window", "session"], help="Active model branch")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window duration (default: 30)")
    parser.add_argument("--max-packets", type=int, default=None, help="Max packets to read per PCAP")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to evaluate")
    parser.add_argument("--label", default="Benign", help="Default ground-truth label if unlabelled")
    parser.add_argument("--label-csv", default=None, help="External CSV for ground-truth 5-tuple matching")
    parser.add_argument("--output-dir", default=str(HERE / "reports"), help="Output directory for reports")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic policy fallback")
    parser.add_argument("--api-key", default=None, help="Gemini API Key")
    args = parser.parse_args()

    orchestrator = PCAPValidationOrchestrator(
        attack_threshold=0.35,
        window_seconds=args.window_seconds,
        use_gemini=not args.no_llm,
        gemini_api_key=args.api_key,
        output_dir=args.output_dir,
    )

    pcap_input = Path(args.pcap_path).resolve()
    if pcap_input.is_dir():
        pcap_files = list(pcap_input.glob("*.pcap")) + list(pcap_input.glob("*.pcapng"))
        print(f"[*] Found {len(pcap_files)} PCAP files in directory.")
        for p in pcap_files:
            orchestrator.run_on_pcap(
                pcap_path=p,
                max_packets=args.max_packets,
                max_windows=args.max_windows,
                active_branch=args.branch,
                ground_truth_label=args.label,
                label_csv=args.label_csv,
            )
    else:
        orchestrator.run_on_pcap(
            pcap_path=pcap_input,
            max_packets=args.max_packets,
            max_windows=args.max_windows,
            active_branch=args.branch,
            ground_truth_label=args.label,
            label_csv=args.label_csv,
        )


if __name__ == "__main__":
    main()

