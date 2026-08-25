import json
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = os.path.dirname(__file__)
PROCESSED = os.path.join(HERE, "processed")
REPORTS = os.path.join(HERE, "reports")


def run_size(seconds):
    env = os.environ.copy()
    env["WINDOW_SECONDS"] = str(seconds)
    subprocess.run([sys.executable, "step0_preprocessing.py"], cwd=HERE, env=env, check=True)
    subprocess.run([sys.executable, "step1_split.py"], cwd=HERE, env=env, check=True)
    X = np.load(os.path.join(PROCESSED, "window_features.npy"))
    y = np.load(os.path.join(PROCESSED, "window_multiclass_labels.npy"))
    names = np.load(os.path.join(PROCESSED, "window_class_names.npy"), allow_pickle=True).astype(str)
    train = np.load(os.path.join(PROCESSED, "train_indices.npy"))
    test = np.load(os.path.join(PROCESSED, "test_indices.npy"))
    flow_i = list(np.load(os.path.join(PROCESSED, "window_feature_names.npy"), allow_pickle=True).astype(str)).index("total_flows")
    packet_i = list(np.load(os.path.join(PROCESSED, "window_feature_names.npy"), allow_pickle=True).astype(str)).index("total_packets")
    train_classes = np.unique(y[train])
    mapping = {label: index for index, label in enumerate(train_classes)}
    encoded = np.array([mapping[label] for label in y[train]], dtype=np.int64)
    model = XGBClassifier(objective="multi:softprob", num_class=len(train_classes), eval_metric="logloss", tree_method="hist", n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    start = time.perf_counter()
    model.fit(X[train], encoded, sample_weight=compute_sample_weight("balanced", encoded))
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = train_classes[model.predict(X[test]).astype(np.int64)]
    inference_seconds = time.perf_counter() - start
    report = classification_report(y[test], predictions, labels=np.arange(len(names)), target_names=names, output_dict=True, zero_division=0)
    result = {"window_seconds": seconds, "windows": int(len(y)), "train_windows": int(len(train)), "test_windows": int(len(test)), "avg_flows_per_window": float(X[:, flow_i].mean()), "avg_packets_per_window": float(X[:, packet_i].mean()), "accuracy": float(accuracy_score(y[test], predictions)), "macro_f1": float(f1_score(y[test], predictions, average="macro", zero_division=0)), "weighted_f1": float(f1_score(y[test], predictions, average="weighted", zero_division=0)), "training_seconds": train_seconds, "inference_seconds": inference_seconds, "slowloris_f1": report.get("DoS attacks-Slowloris", {}).get("f1-score", 0.0), "slowhttptest_f1": report.get("DoS attacks-SlowHTTPTest", {}).get("f1-score", 0.0), "bot_f1": report.get("Bot", {}).get("f1-score", 0.0), "infiltration_f1": report.get("Infilteration", {}).get("f1-score", 0.0)}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    results = [run_size(seconds) for seconds in (1, 5, 10, 30, 60)]
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "window_size_sweep.json"), "w") as output:
        json.dump(results, output, indent=2)
