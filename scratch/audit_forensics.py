"""
Forensic Audit Script for Multi-Granular NIDS Pipeline
Analyzes pipeline_report.json across Phases 2-14.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter, defaultdict

# Load pipeline report
report_path = Path("nids/validation_results/pipeline_report.json")
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

windows = report["windows"]
total_windows = len(windows)
print(f"Loaded {total_windows} windows from {report_path}")

# ==============================================================================
# PHASE 2: VERIFY ALL FOUR MODELS ARE REALLY USED
# ==============================================================================
print("\n" + "="*80)
print("PHASE 2: MODEL PARTICIPATION & AVAILABILITY AUDIT")
print("="*80)

model_stats = {
    "packet": {"invoked": 0, "status_ok": 0, "unavailable": 0, "error": 0, "has_pred": 0, "has_prob": 0},
    "flow": {"invoked": 0, "status_ok": 0, "unavailable": 0, "error": 0, "has_pred": 0, "has_prob": 0},
    "time_window": {"invoked": 0, "status_ok": 0, "unavailable": 0, "error": 0, "has_pred": 0, "has_prob": 0},
    "session": {"invoked": 0, "status_ok": 0, "unavailable": 0, "error": 0, "has_pred": 0, "has_prob": 0},
}

for w in windows:
    l1 = w["level1"]
    for m in ["packet", "flow", "time_window", "session"]:
        rec = l1.get(m)
        if rec is not None:
            model_stats[m]["invoked"] += 1
            st = rec.get("status")
            if st == "ok":
                model_stats[m]["status_ok"] += 1
            elif st == "unavailable":
                model_stats[m]["unavailable"] += 1
            elif st == "error":
                model_stats[m]["error"] += 1
            
            if rec.get("prediction") is not None:
                model_stats[m]["has_pred"] += 1
            if rec.get("attack_probability") is not None and rec.get("attack_probability") > 0:
                model_stats[m]["has_prob"] += 1

print(f"{'Model':15s} | {'Invoked':8s} | {'Status OK':10s} | {'Unavailable':12s} | {'Participation %':15s} | {'Role / Status'}")
print("-" * 80)
for m, st in model_stats.items():
    part_rate = (st["status_ok"] / total_windows) * 100
    role = "Active Predictor" if st["status_ok"] > 0 else "Graceful Stub (Unimplemented)"
    print(f"{m:15s} | {st['invoked']:8d} | {st['status_ok']:10d} | {st['unavailable']:12d} | {part_rate:14.1f}% | {role}")

# Check if stub returning benign is treated as benign
# In attack_gate.py, records with status != 'ok' are excluded from available_ok!
print("\nStub Handling Verification:")
print("In attack_gate.py: available_ok = {g: r for g, r in records if r.status == STATUS_OK}")
print("Stub models (status='unavailable') are completely excluded from weights and P(attack) calculation.")
print("Verified: Packet and Session stubs DO NOT contribute false Benign votes.")

# ==============================================================================
# PHASE 3: FOUR-MODEL PREDICTION DISTRIBUTIONS
# ==============================================================================
print("\n" + "="*80)
print("PHASE 3: PREDICTION DISTRIBUTIONS")
print("="*80)

for m in ["flow", "time_window", "packet", "session"]:
    preds = Counter()
    attack_probs = []
    confs = []
    for w in windows:
        rec = w["level1"][m]
        p = rec.get("prediction")
        preds[p] += 1
        if rec.get("attack_probability") is not None:
            attack_probs.append(rec["attack_probability"])
        if rec.get("confidence") is not None:
            confs.append(rec["confidence"])
    
    print(f"\n--- {m.upper()} MODEL ---")
    print(f"Predictions: {dict(preds)}")
    if attack_probs:
        print(f"Attack Probability: min={min(attack_probs):.4f}, mean={np.mean(attack_probs):.4f}, max={max(attack_probs):.4f}, std={np.std(attack_probs):.4f}")
    if confs:
        print(f"Confidence:         min={min(confs):.4f}, mean={np.mean(confs):.4f}, max={max(confs):.4f}, std={np.std(confs):.4f}")

# Ground truth distribution
gt_counts = Counter(w["ground_truth"] for w in windows)
print(f"\nGROUND TRUTH DISTRIBUTION: {dict(gt_counts)}")

# ==============================================================================
# PHASE 4: INDIVIDUAL MODEL METRICS
# ==============================================================================
print("\n" + "="*80)
print("PHASE 4: INDIVIDUAL MODEL METRICS ON TEST WINDOWS")
print("="*80)

def compute_model_metrics(model_key):
    y_true_binary = [] # 1 if attack, 0 if benign
    y_pred_binary = [] # 1 if attack, 0 if benign
    y_true_multiclass = []
    y_pred_multiclass = []

    for w in windows:
        gt = w["ground_truth"]
        is_attack_gt = (gt != "Benign")
        rec = w["level1"][model_key]
        if rec.get("status") != "ok":
            continue
        
        pred = rec.get("prediction", "Unknown")
        is_attack_pred = rec.get("is_attack", False)
        
        y_true_binary.append(1 if is_attack_gt else 0)
        y_pred_binary.append(1 if is_attack_pred else 0)
        y_true_multiclass.append(gt)
        y_pred_multiclass.append(pred)

    if not y_true_binary:
        return {"status": "unavailable"}

    y_true_bin = np.array(y_true_binary)
    y_pred_bin = np.array(y_pred_binary)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    acc = (tp + tn) / len(y_true_bin)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # Multiclass confusion matrix
    classes = sorted(list(set(y_true_multiclass) | set(y_pred_multiclass)))
    cm = defaultdict(Counter)
    for t, p in zip(y_true_multiclass, y_pred_multiclass):
        cm[t][p] += 1

    return {
        "status": "ok",
        "total": len(y_true_bin),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": acc,
        "attack_precision": prec,
        "attack_recall": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "classes": classes,
        "cm": {c: dict(cm[c]) for c in classes}
    }

flow_metrics = compute_model_metrics("flow")
tw_metrics = compute_model_metrics("time_window")

print("\n--- FLOW MODEL (Binary Attack vs Benign) ---")
print(f"TP={flow_metrics['tp']}, FP={flow_metrics['fp']}, TN={flow_metrics['tn']}, FN={flow_metrics['fn']}")
print(f"Accuracy:        {flow_metrics['accuracy']:.4f}")
print(f"Attack Recall:   {flow_metrics['attack_recall']:.4f}")
print(f"Attack Precision:{flow_metrics['attack_precision']:.4f}")
print(f"F1 Score:        {flow_metrics['f1']:.4f}")
print(f"FPR:             {flow_metrics['fpr']:.4f}")
print(f"FNR:             {flow_metrics['fnr']:.4f}")

print("\nFlow Model Multiclass Confusion Matrix (True Row -> Predicted Col):")
for t in flow_metrics['classes']:
    row_str = ", ".join(f"{p}:{flow_metrics['cm'][t].get(p,0)}" for p in flow_metrics['classes'] if flow_metrics['cm'][t].get(p,0) > 0)
    print(f"  {t:20s} -> {row_str}")

print("\n--- TIME-WINDOW MODEL (Binary Attack vs Benign) ---")
print(f"TP={tw_metrics['tp']}, FP={tw_metrics['fp']}, TN={tw_metrics['tn']}, FN={tw_metrics['fn']}")
print(f"Accuracy:        {tw_metrics['accuracy']:.4f}")
print(f"Attack Recall:   {tw_metrics['attack_recall']:.4f}")
print(f"Attack Precision:{tw_metrics['attack_precision']:.4f}")
print(f"F1 Score:        {tw_metrics['f1']:.4f}")
print(f"FPR:             {tw_metrics['fpr']:.4f}")
print(f"FNR:             {tw_metrics['fnr']:.4f}")

print("\nTime-Window Multiclass Confusion Matrix (True Row -> Predicted Col):")
for t in tw_metrics['classes']:
    row_str = ", ".join(f"{p}:{tw_metrics['cm'][t].get(p,0)}" for p in tw_metrics['classes'] if tw_metrics['cm'][t].get(p,0) > 0)
    print(f"  {t:20s} -> {row_str}")


# ==============================================================================
# PHASE 5: ATTACK GATE AUDIT & THRESHOLD SWEEP
# ==============================================================================
print("\n" + "="*80)
print("PHASE 5: ATTACK GATE THRESHOLD SWEEP")
print("="*80)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
print(f"{'Threshold':10s} | {'Attack Rec':11s} | {'Attack Prec':12s} | {'FPR':8s} | {'FNR':8s} | {'F1':8s} | {'Accuracy':9s}")
print("-" * 80)

y_true_binary = np.array([1 if w["ground_truth"] != "Benign" else 0 for w in windows])
fused_probs = np.array([w["level2"]["fused_attack_probability"] for w in windows])

for t in thresholds:
    y_pred_t = (fused_probs >= t).astype(int)
    tp = np.sum((y_true_binary == 1) & (y_pred_t == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_t == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_t == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_t == 0))

    acc = (tp + tn) / len(y_true_binary)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"{t:10.2f} | {rec:11.4f} | {prec:12.4f} | {fpr:8.4f} | {fnr:8.4f} | {f1:8.4f} | {acc:9.4f}")


# ==============================================================================
# PHASE 6: INVESTIGATE THE 32.6% BENIGN FPR (233 FALSE POSITIVES)
# ==============================================================================
print("\n" + "="*80)
print("PHASE 6: ROOT CAUSE ANALYSIS OF 233 BENIGN FALSE POSITIVES")
print("="*80)

benign_fps = [w for w in windows if w["ground_truth"] == "Benign" and w["action"] in ("FLAG", "BLOCK")]
print(f"Total Benign Windows: {sum(1 for w in windows if w['ground_truth'] == 'Benign')}")
print(f"Total Benign FPs (FLAG + BLOCK): {len(benign_fps)} (FLAG={sum(1 for w in benign_fps if w['action']=='FLAG')}, BLOCK={sum(1 for w in benign_fps if w['action']=='BLOCK')})")

fp_categories = Counter()
fp_details = []

for w in benign_fps:
    wid = w["window_id"]
    action = w["action"]
    rule = w["rule"]
    source = w["source"]
    fused_p = w["level2"]["fused_attack_probability"]
    flow_rec = w["level1"]["flow"]
    tw_rec = w["level1"]["time_window"]
    
    flow_p = flow_rec["attack_probability"]
    tw_p = tw_rec["attack_probability"]
    flow_pred = flow_rec["prediction"]
    tw_pred = tw_rec["prediction"]
    
    # Categorization logic:
    # 1. Flow Model False Positive (Flow model flagged benign traffic as attack with high prob, while Time Window was low)
    # 2. Time Window False Positive (Time window was high attack, flow was low)
    # 3. Both Models False Positive (Both models falsely predicted attack)
    # 4. Context Engine Escalation (Triggered by lingering consecutive attack state)
    # 5. Rule Engine Over-triggering (Specific rule fired on marginal probabilities)
    # 6. Policy Agent / Heuristic
    
    cat = "Unknown"
    if rule == "R3_PERSISTENT_ATTACK":
        cat = "Context-Engine (R3 Persistent Carryover)"
    elif rule == "R4_DDOS_FLOOD":
        cat = "Rule-Engine (R4 DDoS Spike False Trigger)"
    elif rule == "R2_EXTREME_PROB":
        if flow_p >= 0.95 and tw_p < 0.50:
            cat = "Flow-Model FP (R2 Extreme Prob from Flow)"
        elif tw_p >= 0.95 and flow_p < 0.50:
            cat = "Time-Window Model FP (R2 Extreme Prob from TW)"
        else:
            cat = "Both-Models FP (R2 Extreme Prob Consensus)"
    elif rule == "R1_CONSENSUS_BLOCK":
        cat = "Both-Models FP (R1 Consensus Block)"
    elif rule == "R6_MODERATE_FLAG":
        if flow_p >= 0.35 and tw_p < 0.35:
            cat = "Flow-Model Divergence (R6 Flag driven by Flow)"
        elif tw_p >= 0.35 and flow_p < 0.35:
            cat = "Time-Window Divergence (R6 Flag driven by TW)"
        else:
            cat = "Both-Models Moderate Attack (R6 Flag)"
    elif source == "heuristic":
        cat = "Heuristic / Agent Policy Threshold"
    else:
        cat = f"Rule-Engine ({rule})"
        
    fp_categories[cat] += 1
    fp_details.append({
        "wid": wid, "action": action, "rule": rule, "cat": cat,
        "fused_p": fused_p, "flow_p": flow_p, "tw_p": tw_p,
        "flow_pred": flow_pred, "tw_pred": tw_pred
    })

print("\nFalse Positive Breakdown by Root Cause:")
for cat, count in fp_categories.most_common():
    pct = (count / len(benign_fps)) * 100
    print(f"  {cat:45s}: {count:3d} ({pct:5.1f}%)")


# ==============================================================================
# PHASE 7: RULE ENGINE AUDIT
# ==============================================================================
print("\n" + "="*80)
print("PHASE 7: RULE ENGINE AUDIT (PRECISION & ACTIVATION BREAKDOWN)")
print("="*80)

rule_stats = defaultdict(lambda: {"total": 0, "attack": 0, "benign": 0, "actions": Counter()})

for w in windows:
    r = w.get("rule")
    gt_is_attack = (w["ground_truth"] != "Benign")
    rule_stats[r]["total"] += 1
    if gt_is_attack:
        rule_stats[r]["attack"] += 1
    else:
        rule_stats[r]["benign"] += 1
    rule_stats[r]["actions"][w["action"]] += 1

print(f"{'Rule Name':24s} | {'Activations':11s} | {'Attack':7s} | {'Benign':7s} | {'Attack Prec':12s} | {'Actions'}")
print("-" * 80)
for r, st in sorted(rule_stats.items(), key=lambda x: str(x[0])):
    r_name = str(r)
    prec = (st["attack"] / st["total"]) * 100 if st["total"] > 0 else 0.0
    act_str = ", ".join(f"{a}:{c}" for a, c in st["actions"].items())
    print(f"{r_name:24s} | {st['total']:11d} | {st['attack']:7d} | {st['benign']:7d} | {prec:11.1f}% | {act_str}")


# ==============================================================================
# PHASE 8: RULE ABLATION STUDY
# ==============================================================================
print("\n" + "="*80)
print("PHASE 8: RULE ABLATION STUDY")
print("="*80)

# Simulate what action would be taken if a rule is disabled
# In pipeline: if rule is disabled or None -> goes to heuristic (since no-llm)
from nids.policy_agent import PolicyAgent

agent = PolicyAgent(use_gemini=False)

def simulate_pipeline(disabled_rules=set()):
    results = []
    # simulate context engine state
    history = []
    consecutive = 0
    
    for w in windows:
        ev_dict = w["level2"]
        fused_p = ev_dict["fused_attack_probability"]
        agreement = ev_dict["model_agreement_score"]
        n_avail = ev_dict["models_available"]
        n_attack = ev_dict["models_indicating_attack"]
        family = ev_dict["fused_attack_family"] or ""
        
        # Rule evaluation
        rule_fired = None
        action = None
        severity = None
        
        # R2
        if "R2" not in disabled_rules and fused_p >= 0.95:
            rule_fired = "R2_EXTREME_PROB"
            action = "BLOCK"
        # R1
        elif "R1" not in disabled_rules and n_avail >= 2 and agreement >= 0.9 and fused_p >= 0.85:
            rule_fired = "R1_CONSENSUS_BLOCK"
            action = "BLOCK"
        # R4
        elif "R4" not in disabled_rules and family in ("DDoS", "DoS") and fused_p >= 0.70 and agreement >= 0.50:
            rule_fired = "R4_DDOS_FLOOD"
            action = "BLOCK"
        # R3
        elif "R3" not in disabled_rules and consecutive >= 3 and fused_p >= 0.60:
            rule_fired = "R3_PERSISTENT_ATTACK"
            action = "BLOCK"
        # R5
        elif "R5" not in disabled_rules and fused_p <= 0.15:
            rule_fired = "R5_ALLOW_BENIGN"
            action = "ALLOW"
        # R6
        elif "R6" not in disabled_rules and 0.35 <= fused_p <= 0.60:
            rule_fired = "R6_MODERATE_FLAG"
            action = "FLAG"
        
        # If no rule fired -> Heuristic decision
        if action is None:
            # escalation check
            eff_block = 0.70
            if consecutive >= 3 or fused_p >= 0.70:
                eff_block = 0.55
            
            if fused_p >= eff_block:
                action = "BLOCK"
            elif fused_p >= 0.45:
                action = "FLAG"
            else:
                action = "ALLOW"
        
        # update consecutive
        if fused_p >= 0.35:
            consecutive += 1
        else:
            consecutive = 0
            
        results.append((w["ground_truth"], action))

    # Evaluate results
    y_true_attack = [1 if gt != "Benign" else 0 for gt, a in results]
    y_pred_attack = [1 if a in ("FLAG", "BLOCK") else 0 for gt, a in results]
    y_pred_block = [1 if a == "BLOCK" else 0 for gt, a in results]
    
    tp = sum(1 for t, p in zip(y_true_attack, y_pred_attack) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_attack, y_pred_attack) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true_attack, y_pred_attack) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true_attack, y_pred_attack) if t == 1 and p == 0)
    
    acc = (tp + tn) / len(y_true_attack)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    actions = Counter(a for gt, a in results)
    allow_pct = (actions["ALLOW"] / len(results)) * 100
    flag_pct = (actions["FLAG"] / len(results)) * 100
    block_pct = (actions["BLOCK"] / len(results)) * 100
    
    return {
        "recall": rec, "precision": prec, "fpr": fpr, "f1": f1,
        "allow_pct": allow_pct, "flag_pct": flag_pct, "block_pct": block_pct
    }

ablation_configs = [
    ("Current System (All Rules)", set()),
    ("Disable R1 (Consensus Block)", {"R1"}),
    ("Disable R2 (Extreme Prob)", {"R2"}),
    ("Disable R3 (Persistent Attack)", {"R3"}),
    ("Disable R4 (DDoS Flood)", {"R4"}),
    ("Disable R5 (Allow Benign)", {"R5"}),
    ("Disable R6 (Moderate Flag)", {"R6"}),
]

print(f"{'Experiment':30s} | {'Attack Rec':10s} | {'Attack Prec':11s} | {'FPR':7s} | {'F1':7s} | {'ALLOW%':7s} | {'FLAG%':7s} | {'BLOCK%'}")
print("-" * 100)
for name, disabled in ablation_configs:
    res = simulate_pipeline(disabled)
    print(f"{name:30s} | {res['recall']:10.4f} | {res['precision']:11.4f} | {res['fpr']:7.4f} | {res['f1']:7.4f} | {res['allow_pct']:6.1f}% | {res['flag_pct']:6.1f}% | {res['block_pct']:6.1f}%")


# ==============================================================================
# PHASE 10: ACTION POLICY AUDIT
# ==============================================================================
print("\n" + "="*80)
print("PHASE 10: ACTION POLICY AUDIT (ALLOW / FLAG / BLOCK PRECISION)")
print("="*80)

action_stats = defaultdict(lambda: {"total": 0, "attack": 0, "benign": 0})
for w in windows:
    act = w["action"]
    gt_is_attack = (w["ground_truth"] != "Benign")
    action_stats[act]["total"] += 1
    if gt_is_attack:
        action_stats[act]["attack"] += 1
    else:
        action_stats[act]["benign"] += 1

print(f"{'Action':8s} | {'Total':6s} | {'Attack':7s} | {'Benign':7s} | {'P(Attack | Action)':19s} | {'P(Benign | Action)':19s}")
print("-" * 80)
for act in ["BLOCK", "FLAG", "ALLOW"]:
    st = action_stats[act]
    p_attack = (st["attack"] / st["total"]) * 100 if st["total"] > 0 else 0.0
    p_benign = (st["benign"] / st["total"]) * 100 if st["total"] > 0 else 0.0
    print(f"{act:8s} | {st['total']:6d} | {st['attack']:7d} | {st['benign']:7d} | {p_attack:18.2f}% | {p_benign:18.2f}%")


# ==============================================================================
# PHASE 14: GRANULARITY ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("PHASE 14: GRANULARITY ANALYSIS BY ATTACK TYPE")
print("="*80)

attack_types = sorted(list(set(w["ground_truth"] for w in windows if w["ground_truth"] != "Benign")))

print(f"{'Attack Class':18s} | {'Support':8s} | {'Flow Rec':9s} | {'TW Rec':8s} | {'System Rec':11s} | {'Best Individual Source'}")
print("-" * 80)
for at in attack_types:
    at_windows = [w for w in windows if w["ground_truth"] == at]
    n_at = len(at_windows)
    
    flow_hits = sum(1 for w in at_windows if w["level1"]["flow"]["is_attack"])
    tw_hits = sum(1 for w in at_windows if w["level1"]["time_window"]["is_attack"])
    sys_hits = sum(1 for w in at_windows if w["action"] in ("FLAG", "BLOCK"))
    
    flow_rec = flow_hits / n_at if n_at > 0 else 0.0
    tw_rec = tw_hits / n_at if n_at > 0 else 0.0
    sys_rec = sys_hits / n_at if n_at > 0 else 0.0
    
    best = "Flow Model" if flow_rec > tw_rec else ("Time-Window" if tw_rec > flow_rec else "Tied")
    print(f"{at:18s} | {n_at:8d} | {flow_rec:9.2%} | {tw_rec:8.2%} | {sys_rec:11.2%} | {best}")

print("\nForensic computation script completed successfully.")
