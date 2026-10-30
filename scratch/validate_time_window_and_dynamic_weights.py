#!/usr/bin/env python3
"""
=============================================================================
TIME-WINDOW MODEL IN-DEPTH VALIDATION & DYNAMIC WEIGHTING SITUATIONAL AUDIT
=============================================================================
1. Validates Time-Window Model alone on CICIDS2018 test sets and CICIDS2017.
2. Analyzes why it detects volumetric/stateful attacks vs misses low-rate/sparse attacks.
3. Audits situational dynamic weights assigned to all 4 models across distinct real scenarios.
=============================================================================
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))
sys.path.insert(0, str(HERE / "packet_level"))
sys.path.insert(0, str(HERE / "session_level"))

import step0_preprocessing as preprocessing
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.attack_gate import AttackGate
from scratch.run_cross_dataset_benchmark import COL_RENAME_2017

def run_time_window_audit():
    print("=" * 110)
    print("PART 1: TIME-WINDOW XGBOOST MODEL COMPREHENSIVE VALIDATION (CICIDS2018 & CICIDS2017)")
    print("=" * 110)

    tw = TimeWindowAdapter()

    scenarios = [
        # (Dataset, Category, FilePath, RawLabel, MinFlows)
        ("CICIDS2018", "DDOS attack-LOIC-UDP", "cicids2018/02-21-2018.csv", "DDOS attack-LOIC-UDP", 60),
        ("CICIDS2018", "DoS attacks-SlowHTTPTest", "cicids2018/02-16-2018.csv", "DoS attacks-SlowHTTPTest", 60),
        ("CICIDS2018", "FTP-BruteForce", "cicids2018/02-14-2018.csv", "FTP-BruteForce", 60),
        ("CICIDS2018", "DoS attacks-GoldenEye", "cicids2018/02-15-2018.csv", "DoS attacks-GoldenEye", 60),
        ("CICIDS2018", "Infilteration", "cicids2018/03-01-2018.csv", "Infilteration", 60),
        ("CICIDS2018", "DoS attacks-Hulk", "cicids2018/02-16-2018.csv", "DoS attacks-Hulk", 60),
        ("CICIDS2018", "DDOS attack-HOIC", "cicids2018/02-21-2018.csv", "DDOS attack-HOIC", 60),
        ("CICIDS2018", "SSH-Bruteforce", "cicids2018/02-14-2018.csv", "SSH-Bruteforce", 60),
        ("CICIDS2018", "DoS attacks-Slowloris", "cicids2018/02-15-2018.csv", "DoS attacks-Slowloris", 60),
        ("CICIDS2018", "Bot", "cicids2018/03-02-2018.csv", "Bot", 60),
        ("CICIDS2018", "Benign (Daytime Baseline)", "cicids2018/02-15-2018.csv", "Benign", 60),
        
        # 2017 Cross-Domain
        ("CICIDS2017", "DoS attacks-SlowHTTPTest", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", "DoS Slowhttptest", 60),
        ("CICIDS2017", "DoS attacks-GoldenEye", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", "DoS GoldenEye", 60),
        ("CICIDS2017", "DoS attacks-Hulk", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", "DoS Hulk", 60),
        ("CICIDS2017", "DDOS attack-HOIC", "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS", 60),
        ("CICIDS2017", "FTP-BruteForce", "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv", "FTP-Patator", 60),
        ("CICIDS2017", "SSH-Bruteforce", "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv", "SSH-Patator", 60),
        ("CICIDS2017", "PortScan", "cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan", 60),
        ("CICIDS2017", "Benign (Monday Baseline)", "cicids2017/Monday-WorkingHours.pcap_ISCX.csv", "BENIGN", 60),
    ]

    print(f"{'Dataset':<11} | {'Target Scenario':<25} | {'Top-1 Prediction':<22} | {'P(Attack)':<10} | {'Verdict':<8} | {'Detection Status'}")
    print("-" * 110)

    for dset, cat, fpath, raw_lbl, min_f in scenarios:
        if not Path(fpath).exists():
            continue
        try:
            w_df = None
            if "2017" in dset:
                for chunk in pd.read_csv(fpath, chunksize=50000, low_memory=False, encoding="cp1252"):
                    col = [c for c in chunk.columns if "label" in c.lower()][0]
                    chunk[col] = chunk[col].astype(str).str.strip()
                    m = chunk[chunk[col] == raw_lbl].reset_index(drop=True)
                    if len(m) >= min_f:
                        w_df = m.head(60).rename(columns=COL_RENAME_2017)
                        w_df.columns = w_df.columns.str.strip()
                        break
            else:
                for chunk in pd.read_csv(fpath, chunksize=100000, low_memory=False):
                    chunk["Label"] = chunk["Label"].astype(str).str.strip()
                    m = chunk[chunk["Label"] == raw_lbl].reset_index(drop=True)
                    if len(m) >= min_f:
                        w_df = m.head(60)
                        break

            if w_df is None or len(w_df) == 0:
                continue

            ev = tw.predict(w_df)
            p_att = ev.attack_probability
            is_att = ev.is_attack
            top1 = ev.prediction

            if cat.startswith("Benign"):
                status = "CORRECT (Clean Pass)" if not is_att else "FALSE ALARM"
                verd = "ALLOW" if not is_att else "FLAG"
            else:
                status = "DETECTED [PASS]" if is_att else "MISSED (Low/Diffused)"
                verd = "FLAG" if is_att else "ALLOW"

            print(f"{dset:<11} | {cat:<25} | {top1:<22} | {p_att:<10.3f} | {verd:<8} | {status}")
        except Exception as e:
            print(f"{dset:<11} | {cat:<25} | Error: {e}")

    print("=" * 110)


def run_dynamic_weights_situational_audit():
    print("\n" + "=" * 115)
    print("PART 2: SITUATIONAL DYNAMIC WEIGHTING AUDIT ACROSS 4 MODELS (AttackGate Adaptation Engine)")
    print("=" * 115)

    pkt_ad = PacketModelAdapter()
    flow_ad = FlowModelAdapter()
    tw_ad = TimeWindowAdapter()
    sess_ad = SessionModelAdapter()
    gate = AttackGate(threshold=0.35)

    # 1. Load authentic attack window (DDoS HOIC from 02-21-2018)
    w_attack = None
    for chunk in pd.read_csv("cicids2018/02-21-2018.csv", chunksize=50000, low_memory=False):
        m = chunk[chunk["Label"] == "DDOS attack-HOIC"].reset_index(drop=True)
        if len(m) >= 60:
            w_attack = m.head(60).copy()
            break

    # 2. Load authentic benign window from 02-15-2018
    w_benign = None
    for chunk in pd.read_csv("cicids2018/02-15-2018.csv", chunksize=50000, low_memory=False):
        m = chunk[chunk["Label"] == "Benign"].reset_index(drop=True)
        if len(m) >= 60:
            w_benign = m.head(60).copy()
            break

    situations = []

    # Situation 1: High-Volume DDoS / Volumetric Flood (All 60 flows active attack)
    p_ddos = pkt_ad.predict(w_attack)
    f_ddos = flow_ad.predict(w_attack)
    t_ddos = tw_ad.predict(w_attack)
    s_ddos = sess_ad.predict(w_attack)
    integ_ddos = gate.evaluate(p_ddos, f_ddos, t_ddos, s_ddos, window_id="W_VOLUMETRIC")
    situations.append({
        "name": "Situation 1: Volumetric Flood Attack (DDoS HOIC, 60 flows)",
        "traffic_condition": "High flow rate (60 flows), packet asymmetry flags, strong L3/L4 saturation",
        "integrated": integ_ddos,
        "models": {"packet": p_ddos, "flow": f_ddos, "time_window": t_ddos, "session": s_ddos}
    })

    # Situation 2: Sparse / Low-Flow Window (Only 4 flows in 30s)
    w_sparse = w_attack.head(4).copy()
    p_sparse = pkt_ad.predict(w_sparse)
    f_sparse = flow_ad.predict(w_sparse)
    t_sparse = tw_ad.predict(w_sparse)
    s_sparse = sess_ad.predict(w_sparse)  # will register flows_in_window = 4
    integ_sparse = gate.evaluate(p_sparse, f_sparse, t_sparse, s_sparse, window_id="W_SPARSE")
    situations.append({
        "name": "Situation 2: Sparse Traffic Window (Low density: 4 flows)",
        "traffic_condition": "Window has only 4 flows -> Session sequence model cannot form T=30 tensor",
        "integrated": integ_sparse,
        "models": {"packet": p_sparse, "flow": f_sparse, "time_window": t_sparse, "session": s_sparse}
    })

    # Situation 3: Cold-Start Window (Time-Window history depth = 1)
    tw_ad.reset_history()
    t_cold = tw_ad.predict(w_attack)  # history_depth = 1
    integ_cold = gate.evaluate(p_ddos, f_ddos, t_cold, s_ddos, window_id="W_COLD_START")
    situations.append({
        "name": "Situation 3: Cold-Start Window (Time-Window history = 1)",
        "traffic_condition": "NIDS stream just initialized -> Temporal lag/rolling features are cold (depth < 3)",
        "integrated": integ_cold,
        "models": {"packet": p_ddos, "flow": f_ddos, "time_window": t_cold, "session": s_ddos}
    })

    # Situation 4: Clean Daytime Benign Baseline Traffic (Working Hours)
    p_b = pkt_ad.predict(w_benign)
    f_b = flow_ad.predict(w_benign)
    t_b = tw_ad.predict(w_benign)
    s_b = sess_ad.predict(w_benign)
    integ_b = gate.evaluate(p_b, f_b, t_b, s_b, window_id="W_BENIGN")
    situations.append({
        "name": "Situation 4: Clean Benign Baseline Traffic (Working Hours)",
        "traffic_condition": "Unanimous benign traffic -> High confidence on Benign (P < 0.25 across models)",
        "integrated": integ_b,
        "models": {"packet": p_b, "flow": f_b, "time_window": t_b, "session": s_b}
    })

    # Situation 5: Contested / Disagreeing Models (Flow alarms, other models low)
    # Simulate a single-model anomaly
    p_d = pkt_ad.predict(w_benign)   # Benign (0.25)
    f_d = flow_ad.predict(w_attack)   # Attack (1.00)
    t_d = tw_ad.predict(w_benign)     # Benign (0.28)
    s_d = sess_ad.predict(w_benign)   # Benign (0.02)
    integ_d = gate.evaluate(p_d, f_d, t_d, s_d, window_id="W_DISAGREE")
    situations.append({
        "name": "Situation 5: Ambiguous / Split-Consensus Traffic",
        "traffic_condition": "Flow model raises isolated alarm (P=1.0) while Packet, TW, & Session confirm Benign",
        "integrated": integ_d,
        "models": {"packet": p_d, "flow": f_d, "time_window": t_d, "session": s_d}
    })

    for s in situations:
        print(f"\n--- {s['name']} ---")
        print(f"Operational Context: {s['traffic_condition']}")
        integ = s["integrated"]
        weights = integ.model_weights
        print(f"Assigned Dynamic Weights: Packet: {weights.get('packet',0)*100:4.1f}% | Flow: {weights.get('flow',0)*100:4.1f}% | Time-Window: {weights.get('time_window',0)*100:4.1f}% | Session: {weights.get('session',0)*100:4.1f}%")
        print(f"Per-Model P(Attack):     Packet: {s['models']['packet'].attack_probability:.2f}  | Flow: {s['models']['flow'].attack_probability:.2f}  | Time-Window: {s['models']['time_window'].attack_probability:.2f}  | Session: {s['models']['session'].attack_probability:.2f}")
        print(f"Fused P(Attack):         {integ.fused_attack_probability:.3f} (Agreement: {integ.model_agreement_score*100:.0f}%, Attacking: {integ.models_indicating_attack}/{integ.models_available})")
        print(f"Consensus Verdict:       {'ATTACK [FLAG/BLOCK]' if integ.fused_attack_probability >= 0.35 else 'BENIGN [ALLOW]'}")

    print("\n" + "=" * 115)


if __name__ == "__main__":
    run_time_window_audit()
    run_dynamic_weights_situational_audit()
