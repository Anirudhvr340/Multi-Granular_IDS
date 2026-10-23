#!/usr/bin/env python3
"""
=============================================================================
DETAILED PERFORMANCE METRICS PER ATTACK CLASS (CLOSEST 14 CICIDS2018 CLASSES)
=============================================================================
Calculates performance metrics for each attack class present in the converted
CICIOT2023 dataset, mapped to the closest 14 classes the models were trained on.
=============================================================================
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

def main():
    full_report_path = "reports/ciciot2023_integrated_validation_report.json"
    qwen_report_path = "reports/comparison_with_qwen2.5_7b.json"
    no_llm_report_path = "reports/comparison_without_llm.json"
    csv_path = "ciciot2023/ciciot2023_converted_benchmark.csv"

    with open(full_report_path, "r") as f:
        full_rep = json.load(f)

    with open(qwen_report_path, "r") as f:
        qwen_rep = json.load(f)

    with open(no_llm_report_path, "r") as f:
        no_llm_rep = json.load(f)

    # Map CICIOT attacks mapped to each closest class
    df = pd.read_csv(csv_path, usecols=["Original_CICIOT23_Label", "Label"])
    ciciot_mapping = defaultdict(set)
    for _, row in df.iterrows():
        ciciot_mapping[row["Label"]].add(row["Original_CICIOT23_Label"])

    # 1. Full Dataset Metrics (3435 windows)
    full_windows = full_rep["windows"]
    class_stats = defaultdict(lambda: {
        "windows": 0,
        "flows": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "actions": Counter(),
        "pkt_hits": 0,
        "flow_hits": 0,
        "tw_hits": 0,
        "sess_hits": 0,
        "fused_probs": [],
        "fused_families": Counter(),
        "top_preds": Counter(),
    })

    for w in full_windows:
        gt = w.get("ground_truth", "Unknown")
        act = w.get("action", "ALLOW")
        n_flows = w.get("n_flows", 1)
        is_attack = ("benign" not in gt.lower())

        cs = class_stats[gt]
        cs["windows"] += 1
        cs["flows"] += n_flows
        cs["actions"][act] += 1

        if is_attack:
            if act in ("FLAG", "BLOCK"):
                cs["tp"] += 1
            else:
                cs["fn"] += 1
        else:
            if act in ("FLAG", "BLOCK"):
                cs["fp"] += 1
            else:
                cs["tn"] += 1

        # Granular hits
        l1 = w.get("level1", {})
        if l1.get("packet", {}).get("is_attack"): cs["pkt_hits"] += 1
        if l1.get("flow", {}).get("is_attack"): cs["flow_hits"] += 1
        if l1.get("time_window", {}).get("is_attack"): cs["tw_hits"] += 1
        if l1.get("session", {}).get("is_attack"): cs["sess_hits"] += 1

        # Fused prob
        l2 = w.get("level2", {})
        cs["fused_probs"].append(l2.get("fused_attack_probability", 0.0))
        fam = str(l2.get("fused_attack_family", "None"))
        cs["fused_families"][fam] += 1

    # 2. Qwen vs Without LLM comparison by class on identical windows
    qwen_by_class = defaultdict(lambda: {"windows": 0, "detected": 0})
    for w in qwen_rep["windows"]:
        gt = w.get("ground_truth", "Unknown")
        act = w.get("action", "ALLOW")
        qwen_by_class[gt]["windows"] += 1
        if act in ("FLAG", "BLOCK"):
            qwen_by_class[gt]["detected"] += 1

    no_llm_by_class = defaultdict(lambda: {"windows": 0, "detected": 0})
    for w in no_llm_rep["windows"]:
        gt = w.get("ground_truth", "Unknown")
        act = w.get("action", "ALLOW")
        no_llm_by_class[gt]["windows"] += 1
        if act in ("FLAG", "BLOCK"):
            no_llm_by_class[gt]["detected"] += 1

    # Compile Table
    results = []
    for cls_name, cs in sorted(class_stats.items(), key=lambda x: x[1]["windows"], reverse=True):
        n = cs["windows"]
        is_b = ("benign" in cls_name.lower())
        
        if is_b:
            recall = cs["tn"] / (cs["tn"] + cs["fp"]) if (cs["tn"] + cs["fp"]) > 0 else 0.0
            precision = cs["tn"] / (cs["tn"] + cs["fn"]) if (cs["tn"] + cs["fn"]) > 0 else 0.0
            det_rate = (cs["tn"] / n * 100) if n > 0 else 0.0
        else:
            recall = cs["tp"] / (cs["tp"] + cs["fn"]) if (cs["tp"] + cs["fn"]) > 0 else 0.0
            precision = cs["tp"] / (cs["tp"] + cs["fp"]) if (cs["tp"] + cs["fp"]) > 0 else 1.0 # global precision computed separately
            det_rate = (cs["tp"] / n * 100) if n > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        pkt_rate = (cs["pkt_hits"] / n * 100) if n > 0 else 0.0
        flw_rate = (cs["flow_hits"] / n * 100) if n > 0 else 0.0
        tw_rate  = (cs["tw_hits"] / n * 100) if n > 0 else 0.0
        ses_rate = (cs["sess_hits"] / n * 100) if n > 0 else 0.0
        mean_p   = np.mean(cs["fused_probs"]) if cs["fused_probs"] else 0.0

        dominant_fam = cs["fused_families"].most_common(1)[0][0] if cs["fused_families"] else "None"
        source_attacks = sorted(list(ciciot_mapping.get(cls_name, [])))

        # Qwen comparison if present in benchmark
        qw_win = qwen_by_class[cls_name]["windows"]
        qw_det = qwen_by_class[cls_name]["detected"]
        qw_rate = (qw_det / qw_win * 100) if qw_win > 0 else None

        nl_win = no_llm_by_class[cls_name]["windows"]
        nl_det = no_llm_by_class[cls_name]["detected"]
        nl_rate = (nl_det / nl_win * 100) if nl_win > 0 else None

        results.append({
            "class_name": cls_name,
            "mapped_from_ciciot": source_attacks,
            "source_attack_count": len(source_attacks),
            "windows": n,
            "flows": cs["flows"],
            "detection_rate_pct": round(det_rate, 2),
            "recall": round(recall, 4),
            "mean_fused_prob": round(float(mean_p), 4),
            "dominant_family": dominant_fam,
            "models_recall_pct": {
                "packet_1dcnn": round(pkt_rate, 2),
                "flow_xgboost": round(flw_rate, 2),
                "time_window_xgboost": round(tw_rate, 2),
                "session_cnnlstm": round(ses_rate, 2),
                "integrated_system": round(det_rate, 2),
            },
            "actions": dict(cs["actions"]),
            "qwen_benchmark": {
                "windows": qw_win,
                "without_llm_rate_pct": round(nl_rate, 2) if nl_rate is not None else None,
                "with_qwen2.5_rate_pct": round(qw_rate, 2) if qw_rate is not None else None,
            } if qw_win > 0 else None
        })

    out_file = "reports/per_attack_closest_classes_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Per-attack metrics compiled successfully to {out_file}")

if __name__ == "__main__":
    main()

