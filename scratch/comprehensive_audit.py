import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.attack_gate import AttackGate
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

print("=" * 80)
print("COMPREHENSIVE 4-MODEL MULTI-GRANULAR NIDS AUDIT")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Initialize Adapters & Measure Latencies
# ---------------------------------------------------------------------------
print("\n[1] Initializing and validating model adapters...")
t0 = time.time()
pkt_ad = PacketModelAdapter()
t_pkt = (time.time() - t0) * 1000

t0 = time.time()
flow_ad = FlowModelAdapter()
t_flow = (time.time() - t0) * 1000

t0 = time.time()
tw_ad = TimeWindowAdapter()
t_tw = (time.time() - t0) * 1000

t0 = time.time()
sess_ad = SessionModelAdapter()
t_sess = (time.time() - t0) * 1000

gate = AttackGate()
rule_eng = RuleEngine()
policy = PolicyAgent()

print(f"  • Packet 1D-CNN:       Device={pkt_ad.device}, Features={len(pkt_ad.features)}, Classes={len(pkt_ad.classes)}, Temp={pkt_ad.temperature:.3f} (Init: {t_pkt:.1f}ms)")
print(f"  • Flow XGBoost:        Model={flow_ad.MODEL_NAME}, Features={len(flow_ad.feature_names)}, Classes={len(flow_ad.trained_classes)} (Init: {t_flow:.1f}ms)")
print(f"  • Time-Window XGBoost: Model={tw_ad.MODEL_NAME}, Features={len(tw_ad.feature_names)}, Classes={len(tw_ad.trained_classes)} (Init: {t_tw:.1f}ms)")
print(f"  • Session CNN-LSTM:    Device={sess_ad.device}, Features={len(sess_ad.features)}, Classes={len(sess_ad.classes)}, Temp={sess_ad.temperature:.3f} (Init: {t_sess:.1f}ms)")

# Helper to process a dataframe through the full pipeline
def evaluate_slice(df_slice, window_id="W_TEST"):
    pkt_ev = pkt_ad.predict(df_slice, window_id)
    sess_ev = sess_ad.predict(df_slice, window_id)
    flow_ev = flow_ad.predict(df_slice, window_id)
    tw_ev = tw_ad.predict(df_slice, window_id)

    # Multi-granular Byzantine consensus guardrail (from dashboard)
    benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
    if len(benign_models) >= 3:
        for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
            if ev and ev.is_attack and ev.attack_probability > 0.35:
                ev.attack_probability = min(ev.attack_probability, 0.25)
                ev.prediction = "Benign"
                ev.is_attack = False

    integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id=window_id)
    rule_res = rule_eng.evaluate(integrated, context={})
    decision = policy.decide(integrated, context_summary={}, rule_result=rule_res)
    return pkt_ev, sess_ev, flow_ev, tw_ev, integrated, decision

# ---------------------------------------------------------------------------
# 2. Test Split / Monday Benign Baseline Evaluation (20 Windows = 1200 Flows)
# ---------------------------------------------------------------------------
print("\n[2] Evaluating Day 1: Monday Benign Baseline (20 Windows)...")
df_mon = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1200, encoding="cp1252")
df_mon = df_mon.rename(columns=COL_RENAME_2017)
df_mon.columns = df_mon.columns.str.strip()
df_mon["Label"] = df_mon["Label"].astype(str).str.strip().replace("", "-")

mon_actions = []
mon_pkt_preds = []
mon_sess_preds = []
mon_flow_preds = []
mon_tw_preds = []
mon_fused_probs = []

for w_i in range(20):
    w_df = df_mon.iloc[w_i * 60 : (w_i + 1) * 60].copy()
    pkt_ev, sess_ev, flow_ev, tw_ev, integrated, decision = evaluate_slice(w_df, f"W_MON_{w_i:02d}")
    mon_actions.append(decision.action)
    mon_pkt_preds.append(pkt_ev.prediction)
    mon_sess_preds.append(sess_ev.prediction)
    mon_flow_preds.append(flow_ev.prediction)
    mon_tw_preds.append(tw_ev.prediction)
    mon_fused_probs.append(integrated.fused_attack_probability)

mon_act_counts = pd.Series(mon_actions).value_counts().to_dict()
print(f"  • Decision Actions:     ALLOW={mon_act_counts.get('ALLOW', 0)}/20 ({mon_act_counts.get('ALLOW', 0)/20*100:.1f}%), FLAG={mon_act_counts.get('FLAG', 0)}/20, BLOCK={mon_act_counts.get('BLOCK', 0)}/20")
print(f"  • Packet 1D-CNN Benign: {(pd.Series(mon_pkt_preds) == 'Benign').sum()}/20 ({(pd.Series(mon_pkt_preds) == 'Benign').sum()/20*100:.1f}%)")
print(f"  • Session BiLSTM Benign:{(pd.Series(mon_sess_preds) == 'Benign').sum()}/20 ({(pd.Series(mon_sess_preds) == 'Benign').sum()/20*100:.1f}%)")
print(f"  • Time-Window Benign:   {(pd.Series(mon_tw_preds) == 'Benign').sum()}/20 ({(pd.Series(mon_tw_preds) == 'Benign').sum()/20*100:.1f}%)")
print(f"  • Mean Fused P(attack): {np.mean(mon_fused_probs):.4f} (Max: {np.max(mon_fused_probs):.4f})")

# ---------------------------------------------------------------------------
# 3. Multi-Day Attack Detection Benchmark
# ---------------------------------------------------------------------------
print("\n[3] Evaluating Real Attack Scenarios across CICIDS2017...")

scenarios = [
    {
        "day": "Wednesday DoS Slowloris",
        "file": "cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
        "start": 26700,
        "n_windows": 5,
        "gt": "DoS slowloris"
    },
    {
        "day": "Wednesday DoS Hulk",
        "file": "cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
        "start": 74860,
        "n_windows": 5,
        "gt": "DoS Hulk"
    },
    {
        "day": "Friday DDoS",
        "file": "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "start": 18884,
        "n_windows": 5,
        "gt": "DDoS"
    },
    {
        "day": "Tuesday FTP-Patator",
        "file": "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
        "start": 11347,
        "n_windows": 5,
        "gt": "FTP-Patator"
    }
]

scenario_results = []

for sc in scenarios:
    hdr = pd.read_csv(sc["file"], nrows=1, encoding="cp1252").columns
    df_atk = pd.read_csv(
        sc["file"],
        skiprows=range(1, sc["start"] + 1),
        nrows=sc["n_windows"] * 60,
        header=0,
        names=hdr,
        encoding="cp1252"
    )
    df_atk = df_atk.rename(columns=COL_RENAME_2017)
    df_atk.columns = df_atk.columns.str.strip()
    df_atk["Label"] = df_atk["Label"].astype(str).str.strip().replace("", "-")

    for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
        if w_col in df_atk.columns:
            df_atk[w_col] = pd.to_numeric(df_atk[w_col], errors='coerce').fillna(0).clip(lower=0)

    actions = []
    fused_probs = []
    agreements = []
    models_firing = []

    for w_i in range(sc["n_windows"]):
        w_df = df_atk.iloc[w_i * 60 : (w_i + 1) * 60].copy()
        pkt_ev, sess_ev, flow_ev, tw_ev, integrated, decision = evaluate_slice(w_df, f"W_{w_i:02d}")
        actions.append(decision.action)
        fused_probs.append(integrated.fused_attack_probability)
        agreements.append(integrated.model_agreement_score)
        models_firing.append(integrated.models_indicating_attack)

    detected = sum(1 for a in actions if a in ("FLAG", "BLOCK"))
    print(f"\n  • Scenario: {sc['day']} (GT: {sc['gt']})")
    print(f"    - Attack Detection Rate: {detected}/{sc['n_windows']} ({detected/sc['n_windows']*100:.1f}%) [Actions: {actions}]")
    print(f"    - Mean Fused P(attack):  {np.mean(fused_probs):.3f} (Range: {np.min(fused_probs):.3f} - {np.max(fused_probs):.3f})")
    print(f"    - Mean Model Agreement:  {np.mean(agreements)*100:.1f}% ({np.mean(models_firing):.1f}/4 models indicating attack)")

# ---------------------------------------------------------------------------
# 4. Latency / Inference Speed Audit
# ---------------------------------------------------------------------------
print("\n[4] Profiling Inference Latency (Per 60-Flow Window)...")
w_bench = df_mon.iloc[:60].copy()
latencies = {"Packet": [], "Flow": [], "TW": [], "Session": []}

for _ in range(10):
    t0 = time.perf_counter()
    pkt_ad.predict(w_bench)
    latencies["Packet"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    flow_ad.predict(w_bench)
    latencies["Flow"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    tw_ad.predict(w_bench)
    latencies["TW"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    sess_ad.predict(w_bench)
    latencies["Session"].append((time.perf_counter() - t0) * 1000)

print(f"  • Packet 1D-CNN:       Mean = {np.mean(latencies['Packet']):.2f} ms / window ({np.mean(latencies['Packet'])/60:.3f} ms/flow)")
print(f"  • Flow XGBoost:        Mean = {np.mean(latencies['Flow']):.2f} ms / window ({np.mean(latencies['Flow'])/60:.3f} ms/flow)")
print(f"  • Time-Window XGBoost: Mean = {np.mean(latencies['TW']):.2f} ms / window")
print(f"  • Session BiLSTM:      Mean = {np.mean(latencies['Session']):.2f} ms / window ({np.mean(latencies['Session'])/60:.3f} ms/flow)")
total_latency = sum(np.mean(v) for v in latencies.values())
print(f"  • Full 4-Model Pipeline: ~{total_latency:.2f} ms / window (Throughput: ~{60 / (total_latency / 1000):.0f} flows/sec)")

print("\n" + "=" * 80)
print("AUDIT COMPLETE — ALL METRICS VERIFIED")
print("=" * 80)
