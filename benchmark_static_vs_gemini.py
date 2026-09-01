"""
Benchmark: Static Rule-Based Fusion vs. Gemini Dynamic Integration
=================================================================
Compares classical fixed-rule decision fusion (4 methods) against
the dynamic agentic AI decision layer on the same validation dataset.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

HERE = Path(__file__).resolve().parent
TIME_WINDOW = HERE / "time_window"
PROCESSED = TIME_WINDOW / "processed"

# Import the static fusion engine
import importlib.util
spec = importlib.util.spec_from_file_location("fusion_module", str(HERE / "static_rule_fusion (1).py"))
fusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_module)
StaticRuleFusionEngine = fusion_module.StaticRuleFusionEngine
ALL_CLASSES = fusion_module.ALL_CLASSES
CLASS_TO_IDX = fusion_module.CLASS_TO_IDX


def load_artifacts():
    """Load all trained models and scalers."""
    flow_model = joblib.load(HERE / "xgb_flow_model.pkl")
    flow_features = joblib.load(HERE / "important_features.pkl")
    flow_labels = joblib.load(HERE / "labels.pkl").astype(str)
    
    window_model = joblib.load(PROCESSED / "xgb_time_window_model.joblib")
    window_labels = np.load(PROCESSED / "window_class_names.npy", allow_pickle=True).astype(str)
    window_features = np.load(PROCESSED / "window_feature_names.npy", allow_pickle=True).astype(str)
    scaler = joblib.load(PROCESSED / "time_window_scaler.joblib")
    
    return flow_model, flow_features, flow_labels, window_model, window_labels, window_features, scaler


def score_validation_dataset(input_csv: str, window_seconds: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score the entire validation dataset and return:
    - ground_truth: array of true labels (per window or per-flow aggregate)
    - p_flow: flow model probabilities (n_samples, n_classes_flow)
    - p_window: window model probabilities (n_windows, n_classes_window)
    """
    os.environ["WINDOW_SECONDS"] = str(window_seconds)
    from time_window import step0_preprocessing as preprocessing

    artifacts = load_artifacts()
    flow_model, flow_features, flow_labels, window_model, window_labels, window_features, scaler = artifacts
    
    frame = pd.read_csv(input_csv, low_memory=False)
    frame = preprocessing.clean_chunk(frame)
    
    if frame.empty:
        raise ValueError("No valid flow records found.")
    
    frame["window_start"] = frame["Timestamp"].dt.floor(f"{window_seconds}s")
    
    # Collect ground truth and predictions per window
    window_truths = []
    window_flow_probs = []
    window_window_probs = []
    
    print(f"Processing {len(frame)} flows across {frame['window_start'].nunique()} windows...")
    
    for window_start, window_df in frame.groupby("window_start", sort=True):
        aggregate = preprocessing.window_aggregate(window_df, window_start)
        
        # Ground truth: majority class in window
        label_counts = window_df["Label"].value_counts()
        ground_truth_label = label_counts.idxmax()
        window_truths.append(ground_truth_label)
        
        # Flow-level predictions
        flow_input = window_df.copy()
        flow_input["total_packets"] = (
            pd.to_numeric(flow_input.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
            + pd.to_numeric(flow_input.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
        )
        timestamps = pd.to_datetime(window_df["Timestamp"]).sort_values()
        inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
        burstiness = float(inter_arrivals.std(ddof=0) / max(inter_arrivals.mean(), 1e-6)) if len(inter_arrivals) else 0.0
        flow_input["burstiness"] = burstiness
        
        values = flow_input.reindex(columns=flow_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        flow_probs = flow_model.predict_proba(values)
        # Average across flows in window
        avg_flow_probs = flow_probs.mean(axis=0)
        window_flow_probs.append(avg_flow_probs)
        
        # Window-level predictions
        window_vector = pd.DataFrame([aggregate]).reindex(columns=window_features).fillna(0.0)
        window_probs = window_model.predict_proba(scaler.transform(window_vector))[0]
        window_window_probs.append(window_probs)
    
    # Convert to numpy arrays
    y_true = np.array(window_truths)
    p_flow = np.array(window_flow_probs)  # Shape: (n_windows, n_classes_flow)
    p_window = np.array(window_window_probs)  # Shape: (n_windows, n_classes_window)
    
    print(f"Dataset shape: {len(y_true)} windows")
    print(f"Flow probabilities shape: {p_flow.shape}")
    print(f"Window probabilities shape: {p_window.shape}")
    print(f"Ground truth distribution:\n{pd.Series(y_true).value_counts()}")
    
    return y_true, p_flow, p_window, flow_labels, window_labels


def evaluate_static_rules(y_true: np.ndarray, p_flow: np.ndarray, p_window: np.ndarray, 
                         flow_labels: np.ndarray, window_labels: np.ndarray) -> Dict:
    """Apply all 4 static fusion rules and evaluate against ground truth."""
    
    engine = StaticRuleFusionEngine(
        flow_classes=list(flow_labels),
        packet_classes=list(flow_labels),  # Use flow as "packet" proxy
        session_classes=list(window_labels),
        confidence_threshold=0.50
    )
    
    results = {}
    
    # Rule 1: Weighted Sum (Kittler 1998)
    print("\n[1/4] Evaluating Kittler Weighted Sum Rule (1998)...")
    preds_ws, confs_ws = engine.rule_weighted_sum(p_flow, p_flow, p_window)
    acc_ws = accuracy_score(y_true, preds_ws)
    prec_ws, rec_ws, f1_ws, _ = precision_recall_fscore_support(y_true, preds_ws, average="weighted", zero_division=0)
    results["Kittler Weighted Sum (1998)"] = {
        "predictions": preds_ws,
        "confidences": confs_ws,
        "accuracy": acc_ws,
        "precision": prec_ws,
        "recall": rec_ws,
        "f1_weighted": f1_ws,
    }
    print(f"  ✓ Accuracy: {acc_ws*100:.2f}% | F1: {f1_ws*100:.2f}%")
    
    # Rule 2: Max Confidence (Dasarathy 1997)
    print("[2/4] Evaluating Dasarathy Max-Confidence Rule (1997)...")
    preds_mc, confs_mc = engine.rule_max_confidence(p_flow, p_flow, p_window)
    acc_mc = accuracy_score(y_true, preds_mc)
    prec_mc, rec_mc, f1_mc, _ = precision_recall_fscore_support(y_true, preds_mc, average="weighted", zero_division=0)
    results["Dasarathy Max-Confidence (1997)"] = {
        "predictions": preds_mc,
        "confidences": confs_mc,
        "accuracy": acc_mc,
        "precision": prec_mc,
        "recall": rec_mc,
        "f1_weighted": f1_mc,
    }
    print(f"  ✓ Accuracy: {acc_mc*100:.2f}% | F1: {f1_mc*100:.2f}%")
    
    # Rule 3: Priority Cascade (Peddabachigari 2007)
    print("[3/4] Evaluating Peddabachigari Priority Cascade (2007)...")
    preds_pc, confs_pc = engine.rule_priority_cascade(p_flow, p_flow, p_window)
    acc_pc = accuracy_score(y_true, preds_pc)
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(y_true, preds_pc, average="weighted", zero_division=0)
    results["Peddabachigari Priority Cascade (2007)"] = {
        "predictions": preds_pc,
        "confidences": confs_pc,
        "accuracy": acc_pc,
        "precision": prec_pc,
        "recall": rec_pc,
        "f1_weighted": f1_pc,
    }
    print(f"  ✓ Accuracy: {acc_pc*100:.2f}% | F1: {f1_pc*100:.2f}%")
    
    # Rule 4: Majority Vote (Kittler 1998)
    print("[4/4] Evaluating Kittler Majority Vote (1998)...")
    preds_mv, confs_mv = engine.rule_majority_vote(p_flow, p_flow, p_window)
    acc_mv = accuracy_score(y_true, preds_mv)
    prec_mv, rec_mv, f1_mv, _ = precision_recall_fscore_support(y_true, preds_mv, average="weighted", zero_division=0)
    results["Kittler Majority Vote (1998)"] = {
        "predictions": preds_mv,
        "confidences": confs_mv,
        "accuracy": acc_mv,
        "precision": prec_mv,
        "recall": rec_mv,
        "f1_weighted": f1_mv,
    }
    print(f"  ✓ Accuracy: {acc_mv*100:.2f}% | F1: {f1_mv*100:.2f}%")
    
    return results


def load_gemini_baseline(report_path: str = None) -> Dict:
    """Load the Gemini dynamic integration report for comparison."""
    if report_path is None:
        report_path = HERE / "time_window" / "reports" / "final_validation_report.json"
    
    with open(report_path) as f:
        gemini_report = json.load(f)
    
    return gemini_report


def generate_comparison_report(y_true: np.ndarray, static_results: Dict, gemini_report: Dict, output_path: str) -> None:
    """Generate a comprehensive comparison report."""
    
    # Build comparison summary
    comparison = {
        "dataset": gemini_report["input"],
        "windows_scored": gemini_report["windows_scored"],
        "windows_flagged": gemini_report["windows_flagged"],
        "ground_truth_distribution": pd.Series(y_true).value_counts().to_dict(),
        
        "static_fusion_methods": {},
        "gemini_dynamic_integration": {
            "final_action": gemini_report["final_decision"]["action"],
            "priority": gemini_report["final_decision"].get("priority", "Unknown"),
            "attack_types": gemini_report["final_decision"].get("attack_types", []),
            "windows_flagged": gemini_report["windows_flagged"],
            "detection_rate": round(100 * gemini_report["windows_flagged"] / gemini_report["windows_scored"], 2),
        },
    }
    
    # Add static fusion results
    for method_name, result in static_results.items():
        flagged_count = np.sum(result["predictions"] != "Benign")
        comparison["static_fusion_methods"][method_name] = {
            "accuracy": round(result["accuracy"] * 100, 2),
            "precision": round(result["precision"] * 100, 2),
            "recall": round(result["recall"] * 100, 2),
            "f1_weighted": round(result["f1_weighted"] * 100, 2),
            "windows_flagged": int(flagged_count),
            "detection_rate": round(100 * flagged_count / len(y_true), 2),
        }
    
    # Save report
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON: Static Rules vs. Gemini Dynamic")
    print(f"{'='*80}\n")
    
    print("📊 STATIC RULE-BASED FUSION METHODS:\n")
    for method, metrics in comparison["static_fusion_methods"].items():
        print(f"  {method}")
        print(f"    Accuracy: {metrics['accuracy']}% | Precision: {metrics['precision']}%")
        print(f"    Recall: {metrics['recall']}% | F1: {metrics['f1_weighted']}%")
        print(f"    Windows Flagged: {metrics['windows_flagged']} ({metrics['detection_rate']}%)\n")
    
    print(f"{'─'*80}\n")
    print("🤖 GEMINI DYNAMIC INTEGRATION (BASELINE):\n")
    print(f"  Final Action: {comparison['gemini_dynamic_integration']['final_action']} (Priority: {comparison['gemini_dynamic_integration']['priority']})")
    print(f"  Attack Types: {', '.join(comparison['gemini_dynamic_integration']['attack_types'])}")
    print(f"  Windows Flagged: {comparison['gemini_dynamic_integration']['windows_flagged']} ({comparison['gemini_dynamic_integration']['detection_rate']}%)\n")
    
    print(f"{'='*80}\n")
    print(f"Full report saved to: {output_path}\n")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark static fusion vs. Gemini dynamic integration")
    parser.add_argument(
        "input_csv",
        default=str(HERE / "realistic_synthetic_cicids2018_converted.csv"),
        nargs="?",
        help="Validation dataset (default: realistic_synthetic_cicids2018_converted.csv)"
    )
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--report", default="static_vs_gemini_benchmark.json")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STATIC RULE-BASED FUSION vs. GEMINI DYNAMIC INTEGRATION")
    print("="*80 + "\n")
    
    # Score validation dataset
    print(f"📁 Loading validation dataset: {args.input_csv}")
    y_true, p_flow, p_window, flow_labels, window_labels = score_validation_dataset(args.input_csv, args.window_seconds)
    
    # Evaluate static fusion rules
    print(f"\n🔧 Evaluating 4 classical static fusion rules (Kittler, Dasarathy, Peddabachigari)...")
    static_results = evaluate_static_rules(y_true, p_flow, p_window, flow_labels, window_labels)
    
    # Load Gemini baseline
    print(f"\n📋 Loading Gemini dynamic integration baseline...")
    gemini_report = load_gemini_baseline()
    
    # Generate comparison
    output_report = HERE / "time_window" / "reports" / args.report
    comparison = generate_comparison_report(y_true, static_results, gemini_report, str(output_report))
    
    print(f"✅ Benchmark complete!\n")
