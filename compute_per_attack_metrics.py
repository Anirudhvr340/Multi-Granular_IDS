#!/usr/bin/env python3
"""
=============================================================================
COMPUTE PER-ATTACK PERFORMANCE METRICS ON CICIOT2023 DATASET
=============================================================================
Analyzes the entire validation run across all 3,435 windows (13,992 flows)
and outputs detailed performance metrics for:
1. Each of the 34 original CICIOT2023 attack classes
2. Each mapped attack family / category
3. Individual granular models (Packet, Flow, Session, Time-Window) vs Integrated
=============================================================================
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

def categorize_ciciot(label: str) -> str:
    lbl = label.lower()
    if "benign" in lbl:
        return "Benign"
    elif "ddos" in lbl:
        return "DDoS"
    elif "dos" in lbl:
        return "DoS"
    elif "mirai" in lbl:
        return "Mirai / IoT Botnet"
    elif "recon" in lbl or "scan" in lbl:
        return "Reconnaissance"
    elif "bruteforce" in lbl:
        return "Brute Force"
    elif "injection" in lbl or "xss" in lbl or "uploading" in lbl or "browser" in lbl:
        return "Web & Injection"
    elif "spoofing" in lbl:
        return "Spoofing / MITM"
    elif "backdoor" in lbl:
        return "Malware / Backdoor"
    else:
        return "Other Attack"

def main():
    report_file = "reports/ciciot2023_integrated_validation_report.json"
    csv_file = "ciciot2023/ciciot2023_converted_benchmark.csv"

    print(f"Loading validation report: {report_file}...")
    with open(report_file, "r") as f:
        report = json.load(f)

    print(f"Loading dataset CSV: {csv_file}...")
    df = pd.read_csv(csv_file, low_memory=False)
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp"] = ts
    df["window_start_dt"] = df["Timestamp"].dt.floor("30s")

    # Group dataframe to extract ground truth per window
    window_meta = {}
    for ws, group in df.groupby("window_start_dt", sort=True):
        orig_label = group["Original_CICIOT23_Label"].mode()[0] if not group.empty else "Unknown"
        mapped_label = group["Label"].mode()[0] if not group.empty else "Unknown"
        window_meta[str(ws)] = {
            "orig_label": orig_label,
            "mapped_label": mapped_label,
            "flows": len(group),
            "category": categorize_ciciot(orig_label),
        }

    windows = report["windows"]
    print(f"Total report windows: {len(windows)}")

    # Containers for stats
    # 1. By original CICIOT2023 label
    orig_stats = defaultdict(lambda: {
        "total_windows": 0,
        "total_flows": 0,
        "detected": 0,      # FLAG or BLOCK
        "allowed": 0,       # ALLOW
        "actions": Counter(),
        "sources": Counter(),
        "rules": Counter(),
        "packet_hits": 0,
        "flow_hits": 0,
        "time_window_hits": 0,
        "session_hits": 0,
        "fused_attack_probs": [],
        "agreement_scores": [],
        "category": "",
    })

    # 2. By Mapped CICIDS2018 label
    mapped_stats = defaultdict(lambda: {
        "total_windows": 0,
        "total_flows": 0,
        "detected": 0,
        "allowed": 0,
        "actions": Counter(),
        "packet_hits": 0,
        "flow_hits": 0,
        "time_window_hits": 0,
        "session_hits": 0,
        "fused_attack_probs": [],
    })

    # 3. By High-level Category
    cat_stats = defaultdict(lambda: {
        "total_windows": 0,
        "total_flows": 0,
        "detected": 0,
        "allowed": 0,
        "actions": Counter(),
    })

    for w in windows:
        w_start = w.get("window_start")
        meta = window_meta.get(w_start, {
            "orig_label": w.get("ground_truth", "Unknown"),
            "mapped_label": w.get("ground_truth", "Unknown"),
            "flows": w.get("n_flows", 1),
            "category": categorize_ciciot(w.get("ground_truth", "Unknown")),
        })

        orig_lbl = meta["orig_label"]
        mapped_lbl = meta["mapped_label"]
        cat = meta["category"]
        flows = meta["flows"]

        act = w.get("action", "ALLOW")
        src = w.get("source", "unknown")
        rule = w.get("rule")

        is_attack = ("benign" not in orig_lbl.lower())
        detected = (act in ("FLAG", "BLOCK"))

        # Level 1 flags
        l1 = w.get("level1", {})
        pkt_hit = 1 if l1.get("packet", {}).get("is_attack") else 0
        flow_hit = 1 if l1.get("flow", {}).get("is_attack") else 0
        tw_hit = 1 if l1.get("time_window", {}).get("is_attack") else 0
        sess_hit = 1 if l1.get("session", {}).get("is_attack") else 0

        # Level 2 values
        l2 = w.get("level2", {})
        fused_p = l2.get("fused_attack_probability", 0.0)
        agree_s = l2.get("model_agreement_score", 0.0)

        # Update orig_stats
        s = orig_stats[orig_lbl]
        s["total_windows"] += 1
        s["total_flows"] += flows
        s["category"] = cat
        s["actions"][act] += 1
        s["sources"][src] += 1
        if rule:
            s["rules"][rule] += 1
        if detected:
            s["detected"] += 1
        else:
            s["allowed"] += 1
        s["packet_hits"] += pkt_hit
        s["flow_hits"] += flow_hit
        s["time_window_hits"] += tw_hit
        s["session_hits"] += sess_hit
        s["fused_attack_probs"].append(fused_p)
        s["agreement_scores"].append(agree_s)

        # Update mapped_stats
        ms = mapped_stats[mapped_lbl]
        ms["total_windows"] += 1
        ms["total_flows"] += flows
        ms["actions"][act] += 1
        if detected:
            ms["detected"] += 1
        else:
            ms["allowed"] += 1
        ms["packet_hits"] += pkt_hit
        ms["flow_hits"] += flow_hit
        ms["time_window_hits"] += tw_hit
        ms["session_hits"] += sess_hit
        ms["fused_attack_probs"].append(fused_p)

        # Update cat_stats
        cs = cat_stats[cat]
        cs["total_windows"] += 1
        cs["total_flows"] += flows
        cs["actions"][act] += 1
        if detected:
            cs["detected"] += 1
        else:
            cs["allowed"] += 1

    # Format results
    orig_table = []
    for lbl, s in sorted(orig_stats.items(), key=lambda x: x[1]["total_windows"], reverse=True):
        n = s["total_windows"]
        is_b = ("benign" in lbl.lower())
        det_rate = (s["detected"] / n * 100) if n > 0 else 0.0
        pkt_rate = (s["packet_hits"] / n * 100) if n > 0 else 0.0
        flw_rate = (s["flow_hits"] / n * 100) if n > 0 else 0.0
        tw_rate  = (s["time_window_hits"] / n * 100) if n > 0 else 0.0
        ses_rate = (s["session_hits"] / n * 100) if n > 0 else 0.0
        mean_p   = np.mean(s["fused_attack_probs"]) if s["fused_attack_probs"] else 0.0

        orig_table.append({
            "attack_class": lbl,
            "category": s["category"],
            "windows": n,
            "flows": s["total_flows"],
            "detected_windows": s["detected"],
            "allowed_windows": s["allowed"],
            "detection_rate_pct": round(det_rate, 2),
            "mean_fused_prob": round(float(mean_p), 4),
            "packet_model_rate_pct": round(pkt_rate, 2),
            "flow_model_rate_pct": round(flw_rate, 2),
            "time_window_model_rate_pct": round(tw_rate, 2),
            "session_model_rate_pct": round(ses_rate, 2),
            "actions": dict(s["actions"]),
            "rules_triggered": dict(s["rules"]),
        })

    mapped_table = []
    for lbl, ms in sorted(mapped_stats.items(), key=lambda x: x[1]["total_windows"], reverse=True):
        n = ms["total_windows"]
        is_b = ("benign" in lbl.lower())
        det_rate = (ms["detected"] / n * 100) if n > 0 else 0.0
        pkt_rate = (ms["packet_hits"] / n * 100) if n > 0 else 0.0
        flw_rate = (ms["flow_hits"] / n * 100) if n > 0 else 0.0
        tw_rate  = (ms["time_window_hits"] / n * 100) if n > 0 else 0.0
        ses_rate = (ms["session_hits"] / n * 100) if n > 0 else 0.0
        mean_p   = np.mean(ms["fused_attack_probs"]) if ms["fused_attack_probs"] else 0.0

        mapped_table.append({
            "mapped_class": lbl,
            "windows": n,
            "flows": ms["total_flows"],
            "detected_windows": ms["detected"],
            "allowed_windows": ms["allowed"],
            "detection_rate_pct": round(det_rate, 2),
            "mean_fused_prob": round(float(mean_p), 4),
            "packet_model_rate_pct": round(pkt_rate, 2),
            "flow_model_rate_pct": round(flw_rate, 2),
            "time_window_model_rate_pct": round(tw_rate, 2),
            "session_model_rate_pct": round(ses_rate, 2),
            "actions": dict(ms["actions"]),
        })

    cat_table = []
    for cat, cs in sorted(cat_stats.items(), key=lambda x: x[1]["total_windows"], reverse=True):
        n = cs["total_windows"]
        is_b = (cat == "Benign")
        rate = (cs["allowed"] / n * 100) if is_b else (cs["detected"] / n * 100)
        cat_table.append({
            "category": cat,
            "windows": n,
            "flows": cs["total_flows"],
            "detected": cs["detected"],
            "allowed": cs["allowed"],
            "rate_pct": round(rate, 2),
            "rate_name": "Specificity (Pass %)" if is_b else "Detection Rate (%)",
            "actions": dict(cs["actions"]),
        })

    # Save to JSON
    output_data = {
        "total_windows": len(windows),
        "total_attacks_count": 34,
        "per_original_attack_metrics": orig_table,
        "per_mapped_class_metrics": mapped_table,
        "per_category_metrics": cat_table,
    }

    out_json = "reports/per_attack_performance_metrics.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Per-attack metrics calculated and written to {out_json}")

if __name__ == "__main__":
    main()

