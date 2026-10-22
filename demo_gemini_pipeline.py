#!/usr/bin/env python3
"""
Interactive 4-Model Pipeline Demo with Gemini API Integration
=============================================================
Runs the integrated 4-Model NIDS pipeline (Packet 1D-CNN, Flow XGBoost,
Time-Window XGBoost, Session CNN-LSTM) on the synthetic dataset, routing
ambiguous and high-risk traffic windows to the Gemini Policy Agent.

Usage:
  python demo_gemini_pipeline.py --api-key "YOUR_GEMINI_API_KEY" [--max-windows 50]
"""

import os
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nids.pipeline import NIDSPipeline


def main():
    parser = argparse.ArgumentParser(description="Demo: 4-Model NIDS + Gemini Policy Agent")
    parser.add_argument(
        "--csv",
        default=str(HERE / "realistic_synthetic_cicids2018_converted.csv"),
        help="Path to synthetic CSV dataset (default: realistic_synthetic_cicids2018_converted.csv)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Google Gemini API Key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini model name (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=30,
        help="Time window size in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=30,
        help="Number of windows to evaluate during demo (default: 30; set 0 or omit for all)",
    )
    parser.add_argument(
        "--report",
        default=str(HERE / "nids" / "validation_results" / "demo_gemini_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        print("\n" + "=" * 80)
        print("ATTENTION: GEMINI API KEY NOT DETECTED")
        print("=" * 80)
        print("You can supply your Gemini API key in two ways:")
        print('  1. Pass the --api-key flag:   python demo_gemini_pipeline.py --api-key "AIzaSy..."')
        print('  2. Set environment variable:  $env:GEMINI_API_KEY = "AIzaSy..."')
        print("\nProceeding with deterministic fallback heuristic for demonstration...\n")
        use_gemini = False
    else:
        print("\n" + "=" * 80)
        print(f"[*] GEMINI API INTEGRATION ACTIVE (Model: {args.gemini_model})")
        print("=" * 80)
        use_gemini = True

    max_w = None if args.max_windows <= 0 else args.max_windows
    print(f"[*] Input Dataset:   {Path(args.csv).name}")
    print(f"[*] Window Size:     {args.window_seconds} seconds")
    print(f"[*] Max Windows:     {max_w if max_w else 'Full Dataset (2,880 windows)'}")
    print(f"[*] Output Report:   {args.report}")
    print("=" * 80 + "\n")

    pipeline = NIDSPipeline(
        attack_threshold=0.35,
        use_gemini=use_gemini,
        gemini_api_key=api_key if use_gemini else None,
        gemini_model=args.gemini_model,
    )

    report = pipeline.run(
        input_csv=args.csv,
        window_seconds=args.window_seconds,
        report_path=args.report,
        no_llm=not use_gemini,
        max_windows=max_w,
    )

    print("\n" + "=" * 80)
    print("DEMO EXECUTION COMPLETE")
    print("=" * 80)
    actions = report.get("action_distribution", {})
    total = report.get("windows_total", 0)
    print(f"Total Windows Scored: {total}")
    for act, count in actions.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  - Action {act:<7}: {count:>5} ({pct:>5.1f}%)")
    print(f"Average 4-Model Agreement: {report.get('average_model_agreement', 0.0) * 100:.2f}%")
    if use_gemini:
        stats = pipeline.policy_agent.stats
        print(f"Gemini API Calls: {stats['gemini_calls']} (Errors: {stats['gemini_errors']})")
    print("=" * 80)


if __name__ == "__main__":
    main()

