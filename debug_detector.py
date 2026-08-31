"""Debug detector to see actual predictions and confidence scores."""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIME_WINDOW = HERE / "time_window"
PROCESSED = TIME_WINDOW / "processed"


def load_artifacts():
    flow_model = joblib.load(HERE / "xgb_flow_model.pkl")
    flow_features = joblib.load(HERE / "important_features.pkl")
    flow_labels = joblib.load(HERE / "labels.pkl").astype(str)
    window_model = joblib.load(PROCESSED / "xgb_time_window_model.joblib")
    window_labels = np.load(PROCESSED / "window_class_names.npy", allow_pickle=True).astype(str)
    window_features = np.load(PROCESSED / "window_feature_names.npy", allow_pickle=True).astype(str)
    scaler = joblib.load(PROCESSED / "time_window_scaler.joblib")
    return flow_model, flow_features, flow_labels, window_model, window_labels, window_features, scaler


def debug_score_file(input_path, window_seconds, max_windows=10):
    os.environ["WINDOW_SECONDS"] = str(window_seconds)
    from time_window import step0_preprocessing as preprocessing

    artifacts = load_artifacts()
    flow_model, flow_features, flow_labels, window_model, window_labels, window_feature_names, scaler = artifacts
    frame = pd.read_csv(input_path, low_memory=False)

    required_cols = ["Timestamp", "Label", "Dst Port", "Protocol", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Duration"]
    missing = [column for column in required_cols if column not in frame.columns]
    if missing:
        expected_file = str(Path(__file__).resolve().parent / "02-14-2018.csv")
        raise ValueError(
            f"Input file '{input_path}' does not match the CICIDS 2018 flow schema. "
            f"Missing required columns: {missing}. Use a project CSV like '{expected_file}' instead."
        )

    frame = preprocessing.clean_chunk(frame)
    if frame.empty:
        raise ValueError("The input contains no valid 2018 flow records.")
    
    frame["window_start"] = frame["Timestamp"].dt.floor(f"{window_seconds}s")
    
    debug_rows = []
    window_count = 0
    
    for window_start, window in frame.groupby("window_start", sort=True):
        if window_count >= max_windows:
            break
        
        aggregate = preprocessing.window_aggregate(window, window_start)
        
        timestamps = pd.to_datetime(window["Timestamp"]).sort_values()
        inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
        burstiness = float(inter_arrivals.std(ddof=0) / max(inter_arrivals.mean(), 1e-6)) if len(inter_arrivals) else 0.0
        flow_input = window.copy()
        flow_input["total_packets"] = (
            pd.to_numeric(flow_input.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
            + pd.to_numeric(flow_input.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
        )
        flow_input["burstiness"] = burstiness
        
        values = flow_input.reindex(columns=flow_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        flow_probabilities = flow_model.predict_proba(values)
        flow_distribution = dict(zip(flow_labels[flow_model.classes_.astype(int)], flow_probabilities.mean(axis=0)))
        
        window_vector = pd.DataFrame([aggregate]).reindex(columns=window_feature_names).fillna(0.0)
        window_probability = window_model.predict_proba(scaler.transform(window_vector))[0]
        window_distribution = dict(zip(window_labels[window_model.classes_.astype(int)], window_probability))
        
        flow_label = max(flow_distribution, key=flow_distribution.get)
        window_label = max(window_distribution, key=window_distribution.get)
        
        common = sorted(set(flow_distribution) & set(window_distribution))
        flow_weight = 0.5
        window_weight = 0.5
        fused = {
            label: flow_weight * flow_distribution.get(label, 0.0) + window_weight * window_distribution.get(label, 0.0)
            for label in common
        }
        detected = max(fused, key=fused.get)
        
        # Ground truth from data
        ground_truth_labels = window["Label"].unique()
        ground_truth = "Benign" if all(l == "Benign" for l in ground_truth_labels) else ground_truth_labels[0]
        
        debug_rows.append({
            "window": window_count,
            "window_start": str(window_start),
            "flows": len(window),
            "ground_truth": ground_truth,
            "flow_model_prediction": flow_label,
            "flow_model_confidence": round(float(flow_distribution[flow_label]), 6),
            "flow_benign_prob": round(float(flow_distribution.get("Benign", 0.0)), 6),
            "window_model_prediction": window_label,
            "window_model_confidence": round(float(window_distribution[window_label]), 6),
            "window_benign_prob": round(float(window_distribution.get("Benign", 0.0)), 6),
            "fused_prediction": detected,
            "fused_confidence": round(float(fused[detected]), 6),
            "flagged": detected != "Benign",
            "flow_distribution": {k: round(float(v), 6) for k, v in flow_distribution.items()},
            "window_distribution": {k: round(float(v), 6) for k, v in window_distribution.items()},
            "fused_distribution": {k: round(float(v), 6) for k, v in fused.items()},
        })
        
        window_count += 1
    
    return debug_rows


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug detector output")
    parser.add_argument("input_csv", help="CSV file containing CICIDS2018 flow records")
    parser.add_argument("--window-seconds", type=int, default=int(os.environ.get("WINDOW_SECONDS", "30")))
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--report", default="debug_detector_report.json")
    args = parser.parse_args()
    
    debug_rows = debug_score_file(args.input_csv, args.window_seconds, args.max_windows)
    
    report = {
        "input": str(Path(args.input_csv).resolve()),
        "window_seconds": args.window_seconds,
        "windows_analyzed": len(debug_rows),
        "windows": debug_rows,
    }
    
    Path(args.report).write_text(
        json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    
    print(json.dumps({"input": report["input"], "windows_analyzed": report["windows_analyzed"]}, indent=2))
    print(f"Debug report saved to {Path(args.report).resolve()}")
