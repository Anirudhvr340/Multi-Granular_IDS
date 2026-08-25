"""Ablations for testing whether time-window models use temporal information."""

import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def feature_groups(names):
    static_terms = (
        "mean_flow_duration", "mean_flow_bytes", "mean_flow_pkts", "mean_pkt_len",
        "mean_iat", "mean_active", "mean_idle", "std_flow_", "total_fwd_",
        "total_bwd_", "fwd_bwd_", "total_syn", "total_fin", "total_rst",
        "total_psh", "total_ack", "total_urg", "unique_dst_ports", "unique_protocols",
    )
    temporal_terms = (
        "total_flows", "flow_rate", "total_packets", "packet_rate", "total_bytes",
        "byte_rate", "mean_inter_arrival", "std_inter_arrival", "burstiness",
        "previous_", "delta_",
    )
    static = [i for i, name in enumerate(names) if name.startswith(static_terms)]
    temporal = [i for i, name in enumerate(names) if name.startswith(temporal_terms)]
    return static, temporal


def run_xgb(X_train, y_train, X_test, y_test, class_names, name):
    train_classes = np.unique(y_train)
    mapping = {label: index for index, label in enumerate(train_classes)}
    encoded = np.array([mapping[label] for label in y_train], dtype=np.int64)
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(train_classes), eval_metric="logloss",
        tree_method="hist", n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    started = time.perf_counter()
    model.fit(X_train, encoded, sample_weight=compute_sample_weight("balanced", encoded))
    predictions = train_classes[model.predict(X_test).astype(np.int64)]
    elapsed = time.perf_counter() - started
    report = classification_report(
        y_test, predictions, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "experiment": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "training_seconds": elapsed,
        "per_class": {
            key: {metric: value for metric, value in value.items() if metric in ("recall", "f1-score", "support")}
            for key, value in report.items() if isinstance(value, dict)
        },
    }


def main():
    X = np.load(os.path.join(INPUT_DIR, "window_features.npy"))
    y = np.load(os.path.join(INPUT_DIR, "window_multiclass_labels.npy"))
    names = np.load(os.path.join(INPUT_DIR, "window_feature_names.npy"), allow_pickle=True).astype(str)
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)
    train = np.load(os.path.join(INPUT_DIR, "train_indices.npy"))
    test = np.load(os.path.join(INPUT_DIR, "test_indices.npy"))
    static, temporal = feature_groups(names)
    experiments = {
        "static_only": static,
        "temporal_only": temporal,
        "static_plus_temporal": sorted(set(static + temporal)),
    }
    results = []
    for name, columns in experiments.items():
        results.append(run_xgb(X[train][:, columns], y[train], X[test][:, columns], y[test], class_names, name))

    # A timestamp/order control: preserve each endpoint and randomly permute its lookback.
    # The LSTM runner consumes this flag to train the same architecture with shuffled steps.
    results.append({
        "experiment": "lstm_order_shuffle",
        "status": "run with SHUFFLE_SEQUENCE=1 python step5_temporal_lstm.py",
    })
    results.append({
        "experiment": "timestamp_shuffle_control",
        "status": "rebuild raw rows with shuffled timestamps before step0; do not use saved windows",
    })
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "temporal_ablation_results.json"), "w") as output:
        json.dump(results, output, indent=2)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()