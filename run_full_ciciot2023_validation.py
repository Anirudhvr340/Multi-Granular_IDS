#!/usr/bin/env python3
"""
=============================================================================
FULL CICIOT2023 DATASET VALIDATION RUNNER
=============================================================================
Runs validation across the ENTIRE converted CICIOT2023 dataset without window
truncation limits (no 30-window cap).

Supports:
1. Full Integrated 4-Model Pipeline (Time-Window + Flow + Packet + Session)
2. Standalone Model Sweeps on all records (Flow XGBoost, Packet 1D-CNN, Session CNN-LSTM, Time-Window XGBoost)
3. Grounded streaming dataset conversion from raw CICIOT2023 to CICIDS2018 schema.

Usage Examples:
---------------
# 1. Run the Full 4-Model Pipeline across ALL windows on the converted dataset:
python run_full_ciciot2023_validation.py --mode pipeline --csv cicids2023/ciciot2023_converted_benchmark.csv

# 2. Run with Gemini LLM Active on all windows:
python run_full_ciciot2023_validation.py --mode pipeline --csv cicids2023/ciciot2023_converted_benchmark.csv --api-key "YOUR_KEY"

# 3. Run all 4 standalone models independently on ALL flows:
python run_full_ciciot2023_validation.py --mode models --model all --csv cicids2023/ciciot2023_converted_benchmark.csv

# 4. Convert and validate the full 1.17M rows of validation.csv:
python run_full_ciciot2023_validation.py --convert-full --mode pipeline
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))
sys.path.insert(0, str(HERE / "packet_level"))
sys.path.insert(0, str(HERE / "session_level"))

from convert_ciciot23_to_cicids2018 import convert_dataset
from validate_day import validate_day_dataset
import validate_single_model as single_eval


def run_full_pipeline(csv_path: str, window_seconds: int = 30, api_key: str | None = None, report_path: str | None = None):
    print("=" * 80)
    print("STARTING FULL-DATASET 4-MODEL PIPELINE VALIDATION (NO WINDOW LIMIT)")
    print(f"Dataset:        {csv_path}")
    print(f"Window Seconds: {window_seconds}s")
    print(f"Policy Engine:  {'Gemini LLM Active' if api_key else 'Deterministic Rule Engine Fallback'}")
    print("=" * 80)
    
    validate_day_dataset(
        csv_file=csv_path,
        window_seconds=window_seconds,
        max_windows=None,  # None means process ALL windows in the dataset
        attack_threshold=0.35,
        no_llm=(not bool(api_key)),
        api_key=api_key,
        report_file=report_path,
    )


def run_standalone_models(csv_path: str, model_name: str = "all"):
    print("=" * 80)
    print(f"STARTING STANDALONE MODEL VALIDATION ON FULL DATASET: {model_name.upper()}")
    print(f"Dataset: {csv_path}")
    print("=" * 80)
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[+] Loaded {len(df):,} total flows from {Path(csv_path).name}")
    
    if model_name in ("flow", "all"):
        single_eval.evaluate_flow_model(df)
        
    if model_name in ("packet", "all"):
        single_eval.evaluate_packet_model(df)
        
    if model_name in ("session", "all"):
        single_eval.evaluate_session_model(df)
        
    if model_name in ("time_window", "all"):
        single_eval.evaluate_time_window_model(df)


def main():
    parser = argparse.ArgumentParser(description="Validate Multi-Granular NIDS on full CICIOT2023 dataset")
    parser.add_argument(
        "--csv",
        default="cicids2023/ciciot2023_converted_benchmark.csv",
        help="Path to converted CSV dataset (default: cicids2023/ciciot2023_converted_benchmark.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "models"],
        default="pipeline",
        help="Validation mode: 'pipeline' for integrated 4-model fusion, 'models' for standalone classifiers",
    )
    parser.add_argument(
        "--model",
        choices=["all", "flow", "packet", "session", "time_window"],
        default="all",
        help="Which model to evaluate when --mode=models (default: all)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=30,
        help="Time-window aggregation duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", None),
        help="Gemini API Key (optional, defaults to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to save evaluation JSON report",
    )
    parser.add_argument(
        "--convert-full",
        action="store_true",
        help="Convert the complete raw validation.csv (1.17M flows) before validating",
    )
    parser.add_argument(
        "--raw-input",
        default="cicids2023/CICIOT23/validation/validation.csv",
        help="Raw CICIOT2023 CSV file if --convert-full is specified",
    )
    args = parser.parse_args()

    csv_target = Path(args.csv)
    
    if args.convert_full:
        full_converted = Path("cicids2023/ciciot2023_validation_full_converted.csv")
        if not full_converted.exists():
            print(f"[*] Converting complete raw dataset from {args.raw_input}...")
            convert_dataset(
                input_csv=args.raw_input,
                output_csv=str(full_converted),
                chunksize=100000,
            )
        csv_target = full_converted

    if not csv_target.exists():
        print(f"[!] Target CSV file not found: {csv_target}")
        print("[!] Tip: You can generate it by running:")
        print(f"    python convert_ciciot23_to_cicids2018.py --input \"{args.raw_input}\" --output \"{csv_target}\" --sample 50000")
        sys.exit(1)

    if args.mode == "pipeline":
        run_full_pipeline(
            csv_path=str(csv_target),
            window_seconds=args.window_seconds,
            api_key=args.api_key,
            report_path=args.report,
        )
    elif args.mode == "models":
        run_standalone_models(
            csv_path=str(csv_target),
            model_name=args.model,
        )


if __name__ == "__main__":
    main()

