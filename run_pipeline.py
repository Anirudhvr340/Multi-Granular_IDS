#!/usr/bin/env python3
"""
Four-Model Agentic Pipeline CLI
================================
Run the complete Packet + Flow + Time-Window + Session pipeline.

Usage
-----
  cd "D:\\Multi-Granular IDS\\Multi-Granular_IDS"

  # With Gemini (set GEMINI_API_KEY first):
  python run_pipeline.py 02-14-2018.csv --window-seconds 30

  # Without Gemini (deterministic heuristic only):
  python run_pipeline.py 02-14-2018.csv --window-seconds 30 --no-llm

  # Custom output:
  python run_pipeline.py 02-14-2018.csv --report my_report.json
"""

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv",
                        help="CICIDS-2018 flow CSV (e.g. 02-14-2018.csv)")
    parser.add_argument("--window-seconds", type=int, default=30,
                        help="Time-window size in seconds (default: 30)")
    parser.add_argument("--report", default="nids_pipeline_report.json",
                        help="Output report path")
    parser.add_argument("--attack-threshold", type=float, default=0.35,
                        help="Fused P(attack) threshold (default: 0.35)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable Gemini, use deterministic heuristic")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API Key (or set GEMINI_API_KEY environment variable)")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash",
                        help="Gemini model name (default: gemini-1.5-flash)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Maximum number of windows to process (default: all)")
    parser.add_argument("--output-dir", default=None,
                        help="Decision artifacts directory")
    args = parser.parse_args()

    from nids.pipeline import NIDSPipeline

    pipe = NIDSPipeline(
        attack_threshold=args.attack_threshold,
        use_gemini=not args.no_llm,
        gemini_api_key=args.api_key,
        gemini_model=args.gemini_model,
        output_dir=args.output_dir,
    )

    pipe.run(
        input_csv=args.input_csv,
        window_seconds=args.window_seconds,
        report_path=args.report,
        no_llm=args.no_llm,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()

