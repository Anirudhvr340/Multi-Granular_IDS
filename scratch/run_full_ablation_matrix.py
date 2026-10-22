"""
Script to compute the exact ablation experiments A through H requested in Fix 18.
Evaluates on 02-14-2018 (1,086 windows).
"""

import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# Load windows from the optimized pipeline report (contains ground truth, flow predictions, tw predictions)
with open("nids/validation_results/pipeline_report_optimized_02_14.json", "r", encoding="utf-8") as f:
    opt_rep = json.load(f)

# Also load windows from the baseline pipeline report
with open("nids/validation_results/pipeline_report.json", "r", encoding="utf-8") as f:
    base_rep = json.load(f)

windows_opt = opt_rep["windows"]
windows_base = base_rep["windows"]
n_windows = len(windows_opt)

y_true = np.array([1 if w["ground_truth"] != "Benign" else 0 for w in windows_opt])
n_attack = np.sum(y_true == 1)
n_benign = np.sum(y_true == 0)

def compute_metrics(actions):
    y_pred_any = np.array([1 if a in ("FLAG", "BLOCK") else 0 for a in actions])
    y_pred_block = np.array([1 if a == "BLOCK" else 0 for a in actions])
    y_pred_flag = np.array([1 if a == "FLAG" else 0 for a in actions])
    y_pred_allow = np.array([1 if a == "ALLOW" else 0 for a in actions])

    tp = np.sum((y_true == 1) & (y_pred_any == 1))
    fp = np.sum((y_true == 0) & (y_pred_any == 1))
    tn = np.sum((y_true == 0) & (y_pred_any == 0))
    fn = np.sum((y_true == 1) & (y_pred_any == 0))

    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / len(y_true)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Action precisions
    n_block = np.sum(y_pred_block)
    block_tp = np.sum((y_true == 1) & (y_pred_block == 1))
    block_prec = block_tp / n_block if n_block else 0.0

    n_flag = np.sum(y_pred_flag)
    flag_tp = np.sum((y_true == 1) & (y_pred_flag == 1))
    flag_prec = flag_tp / n_flag if n_flag else 0.0

    n_allow = np.sum(y_pred_allow)
    allow_tn = np.sum((y_true == 0) & (y_pred_allow == 1))
    allow_prec = allow_tn / n_allow if n_allow else 0.0

    allow_pct = (n_allow / len(actions)) * 100
    flag_pct = (n_flag / len(actions)) * 100
    block_pct = (n_block / len(actions)) * 100

    return {
        "recall": rec, "precision": prec, "fpr": fpr, "f1": f1, "accuracy": acc,
        "block_prec": block_prec, "flag_prec": flag_prec, "allow_prec": allow_prec,
        "allow_pct": allow_pct, "flag_pct": flag_pct, "block_pct": block_pct,
        "counts": {"ALLOW": int(n_allow), "FLAG": int(n_flag), "BLOCK": int(n_block)}
    }

# 1. BASELINE: extracted directly from pipeline_report.json
baseline_actions = [w["action"] for w in windows_base]
m_baseline = compute_metrics(baseline_actions)

# 2. FINAL OPTIMIZED (G / H): extracted directly from pipeline_report_optimized_02_14.json
opt_actions = [w["action"] for w in windows_opt]
m_opt = compute_metrics(opt_actions)

# Now simulate ablations A, B, C, D, E, F, G
# Each window has:
# - raw flow probs (baseline max vs optimized density)
# - tw probs
# - simulated context
def simulate_variant(use_flow_opt=True, use_ctx_reset=True, r4_calibrated=True, r6_calibrated=True, attack_thresh=0.35):
    actions = []
    running_persist = 0
    history = []

    for w_idx, (w_b, w_o) in enumerate(zip(windows_base, windows_opt)):
        flow_rec = w_o["level1"]["flow"] if use_flow_opt else w_b["level1"]["flow"]
        tw_rec = w_o["level1"]["time_window"]

        flow_p = flow_rec["attack_probability"]
        flow_conf = flow_rec["confidence"]
        tw_p = tw_rec["attack_probability"]
        tw_conf = tw_rec["confidence"]
        tw_pred = tw_rec["prediction"]

        # Fused attack probability (weights: flow 0.35, tw 0.30 modulated by conf)
        w_flow = 0.35 * (0.5 + 0.5 * np.tanh((flow_conf - 0.5) * 4))
        w_tw = 0.30 * (0.5 + 0.5 * np.tanh((tw_conf - 0.5) * 4))
        fused_p = (w_flow * flow_p + w_tw * tw_p) / (w_flow + w_tw)

        # Agreement
        models_att = (1 if flow_p >= attack_thresh else 0) + (1 if tw_p >= attack_thresh else 0)
        agreement = models_att / 2.0

        # Family
        family = flow_rec.get("normalised_family") or tw_rec.get("normalised_family") or ""

        # Context update
        if use_ctx_reset:
            if (tw_pred == "Benign" and tw_conf >= 0.70) or fused_p <= 0.15:
                running_persist = 0
            elif (tw_pred == "Benign" and tw_conf >= 0.50) or fused_p <= 0.30:
                running_persist = max(0, running_persist - 1)
            elif fused_p >= attack_thresh:
                running_persist += 1
        else:
            # Baseline: backwards count on fused_p >= 0.35
            if fused_p >= 0.35:
                running_persist += 1
            else:
                running_persist = 0

        # Escalation level
        if fused_p >= 0.90 or running_persist >= 6:
            esc_level = "CRITICAL"
        elif fused_p >= 0.70 or running_persist >= 3:
            esc_level = "HIGH"
        elif fused_p >= 0.50 or running_persist >= 2:
            esc_level = "MEDIUM"
        elif fused_p >= 0.35:
            esc_level = "LOW"
        else:
            esc_level = "NONE"

        ctx_dict = {"consecutive_attack_windows": running_persist, "escalation_level": esc_level}

        # Rule engine
        action = None
        # R2
        if fused_p >= 0.95:
            action = "BLOCK"
        # R1
        elif agreement >= 0.90 and fused_p >= 0.85:
            action = "BLOCK"
        # R4
        elif family in ("DDoS", "DoS"):
            if r4_calibrated:
                if fused_p >= 0.85 and agreement >= 0.75:
                    action = "BLOCK"
                elif fused_p >= 0.70:
                    action = "FLAG"
            else:
                if fused_p >= 0.70 and agreement >= 0.50:
                    action = "BLOCK"
        # R3
        if action is None and running_persist >= 3 and fused_p >= 0.60:
            action = "BLOCK"
        # R5
        if action is None and fused_p <= 0.15:
            action = "ALLOW"
        # R6
        if action is None:
            if r6_calibrated:
                if 0.45 <= fused_p <= 0.65 and (models_att >= 2 or running_persist >= 1 or esc_level in ("HIGH", "CRITICAL")):
                    action = "FLAG"
            else:
                if 0.35 <= fused_p <= 0.60:
                    action = "FLAG"

        # Heuristic / Risk Score fallback
        if action is None:
            esc_w = {"NONE": 0.0, "LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6, "CRITICAL": 1.0}[esc_level]
            f_density = flow_rec.get("supporting_evidence", {}).get("fraction_attack_gt_070", 0.0) if use_flow_opt else 0.0
            risk_score = 0.40 * fused_p + 0.20 * agreement + 0.15 * min(running_persist / 4.0, 1.0) + 0.15 * esc_w + 0.10 * min(f_density * 2.0, 1.0)
            if risk_score >= 0.65:
                action = "BLOCK"
            elif risk_score >= 0.35:
                action = "FLAG"
            else:
                action = "ALLOW"

        actions.append(action)

    return compute_metrics(actions)

experiments = {
    "BASELINE": m_baseline,
    "A (Flow Density only)": simulate_variant(use_flow_opt=True, use_ctx_reset=False, r4_calibrated=False, r6_calibrated=False),
    "B (Context Reset only)": simulate_variant(use_flow_opt=False, use_ctx_reset=True, r4_calibrated=False, r6_calibrated=False),
    "C (R4 Calibration only)": simulate_variant(use_flow_opt=False, use_ctx_reset=False, r4_calibrated=True, r6_calibrated=False),
    "D (R6 Calibration only)": simulate_variant(use_flow_opt=False, use_ctx_reset=False, r4_calibrated=False, r6_calibrated=True),
    "E (Flow + Context)": simulate_variant(use_flow_opt=True, use_ctx_reset=True, r4_calibrated=False, r6_calibrated=False),
    "F (Flow + Context + R4)": simulate_variant(use_flow_opt=True, use_ctx_reset=True, r4_calibrated=True, r6_calibrated=False),
    "G (Flow + Ctx + R4 + R6)": m_opt,
    "H (Threshold = 0.42)": simulate_variant(use_flow_opt=True, use_ctx_reset=True, r4_calibrated=True, r6_calibrated=True, attack_thresh=0.42),
}

print("="*115)
print(f"{'Experiment':25s} | {'Recall':7s} | {'Prec':7s} | {'FPR':7s} | {'F1':7s} | {'BLOCK Prec':10s} | {'ALLOW%':7s} | {'FLAG%':7s} | {'BLOCK%'}")
print("="*115)
for name, m in experiments.items():
    print(f"{name:25s} | {m['recall']:7.4f} | {m['precision']:7.4f} | {m['fpr']:7.4f} | {m['f1']:7.4f} | {m['block_prec']*100:9.1f}% | {m['allow_pct']:6.1f}% | {m['flag_pct']:6.1f}% | {m['block_pct']:6.1f}%")
print("="*115)

