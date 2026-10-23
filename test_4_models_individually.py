#!/usr/bin/env python3
"""
=============================================================================
AUDIT & BENCHMARK: TEST ALL 4 ML MODELS INDIVIDUALLY ON AUTHENTIC CICIDS2018
=============================================================================
Tests Packet 1D-CNN, Flow XGBoost, Session CNN-LSTM, and Time-Window XGBoost
on pure slices of each attack type from the authentic CICIDS2018 files:
- FTP-BruteForce & SSH-Bruteforce (02-14-2018)
- DoS-GoldenEye & DoS-Slowloris (02-15-2018)
- DoS-SlowHTTPTest (02-16-2018)
- DDoS-HOIC & DDoS-LOIC-UDP (02-21-2018)
- SQL Injection, Brute Force -Web, Brute Force -XSS (02-22-2018)
- Botnet (03-02-2018)
- Benign Traffic
=============================================================================
"""

import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

import step0_preprocessing as preprocessing
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter


def load_pure_slices():
    """Load genuine, pure slices of attack and benign flows from CICIDS2018 daily CSVs."""
    test_specs = [
        ("cicids2018/02-14-2018.csv", "FTP-BruteForce", 200),
        ("cicids2018/02-14-2018.csv", "SSH-Bruteforce", 200),
        ("cicids2018/02-15-2018.csv", "DoS attacks-GoldenEye", 200),
        ("cicids2018/02-15-2018.csv", "DoS attacks-Slowloris", 200),
        ("cicids2018/02-16-2018.csv", "DoS attacks-SlowHTTPTest", 200),
        ("cicids2018/02-21-2018.csv", "DDOS attack-HOIC", 200),
        ("cicids2018/02-21-2018.csv", "DDOS attack-LOIC-UDP", 200),
        ("cicids2018/02-22-2018.csv", "SQL Injection", 50),
        ("cicids2018/02-22-2018.csv", "Brute Force -Web", 100),
        ("cicids2018/02-22-2018.csv", "Brute Force -XSS", 100),
        ("cicids2018/03-02-2018.csv", "Bot", 200),
        ("cicids2018/02-14-2018.csv", "Benign", 200),
    ]

    slices = {}
    for csv_file, target_label, n_samples in test_specs:
        if not Path(csv_file).exists():
            continue
        try:
            # Read chunk to find target label
            for chunk in pd.read_csv(csv_file, chunksize=100_000, low_memory=False):
                chunk["Label"] = chunk["Label"].astype(str).str.strip()
                match = chunk[chunk["Label"].str.lower() == target_label.lower()]
                if len(match) > 0:
                    sample = match.head(n_samples)
                    slices[target_label] = sample
                    break
        except Exception as e:
            print(f"Error loading {target_label} from {csv_file}: {e}")

    return slices


def main():
    print("=" * 90)
    print("INDIVIDUAL AUDIT OF ALL 4 ML MODELS ON AUTHENTIC CICIDS2018 TRAFFIC")
    print("=" * 90)

    print("\n[1/5] Loading pure ground-truth slices from CICIDS2018 daily CSVs...")
    slices = load_pure_slices()
    print(f"Loaded {len(slices)} ground-truth test slices: {list(slices.keys())}")

    print("\n[2/5] Initializing all 4 model adapters...")
    pkt_adapter = PacketModelAdapter()
    flow_adapter = FlowModelAdapter()
    tw_adapter = TimeWindowAdapter()
    sess_adapter = SessionModelAdapter()

    # Results table
    model_results = defaultdict(lambda: defaultdict(dict))

    print("\n[3/5] Testing each model on pure attack and benign slices...")

    for label, df_slice in slices.items():
        is_benign = (label == "Benign")
        n_flows = len(df_slice)

        # Standard clean chunk for pipeline
        cleaned = preprocessing.clean_chunk(df_slice)
        if cleaned.empty:
            cleaned = df_slice

        # Dummy window timestamp
        window_start = pd.Timestamp("2018-03-01 08:00:00")

        # 1. Packet 1D-CNN
        try:
            ev_pkt = pkt_adapter.predict(cleaned, window_id=f"W_{label}")
            pkt_top1 = ev_pkt.prediction
            pkt_p = ev_pkt.attack_probability
            pkt_conf = ev_pkt.confidence
            pkt_is_att = ev_pkt.is_attack
        except Exception as e:
            pkt_top1, pkt_p, pkt_conf, pkt_is_att = f"Error: {e}", 0.0, 0.0, False

        # 2. Flow XGBoost
        try:
            ev_flw = flow_adapter.predict(cleaned, window_id=f"W_{label}")
            flw_top1 = ev_flw.prediction
            flw_p = ev_flw.attack_probability
            flw_conf = ev_flw.confidence
            flw_is_att = ev_flw.is_attack
        except Exception as e:
            flw_top1, flw_p, flw_conf, flw_is_att = f"Error: {e}", 0.0, 0.0, False

        # 3. Time-Window XGBoost
        try:
            agg = preprocessing.window_aggregate(cleaned, window_start)
            ev_tw = tw_adapter.predict(agg, window_id=f"W_{label}")
            tw_top1 = ev_tw.prediction
            tw_p = ev_tw.attack_probability
            tw_conf = ev_tw.confidence
            tw_is_att = ev_tw.is_attack
        except Exception as e:
            tw_top1, tw_p, tw_conf, tw_is_att = f"Error: {e}", 0.0, 0.0, False

        # 4. Session CNN-LSTM
        try:
            ev_sess = sess_adapter.predict(cleaned, window_id=f"W_{label}")
            sess_top1 = ev_sess.prediction
            sess_p = ev_sess.attack_probability
            sess_conf = ev_sess.confidence
            sess_is_att = ev_sess.is_attack
        except Exception as e:
            sess_top1, sess_p, sess_conf, sess_is_att = f"Error: {e}", 0.0, 0.0, False

        model_results[label] = {
            "n_flows": n_flows,
            "packet": {"top1": pkt_top1, "P_att": pkt_p, "conf": pkt_conf, "is_att": pkt_is_att},
            "flow": {"top1": flw_top1, "P_att": flw_p, "conf": flw_conf, "is_att": flw_is_att},
            "time_window": {"top1": tw_top1, "P_att": tw_p, "conf": tw_conf, "is_att": tw_is_att},
            "session": {"top1": sess_top1, "P_att": sess_p, "conf": sess_conf, "is_att": sess_is_att},
        }

    # Print Comprehensive Table
    print("\n" + "=" * 125)
    print("DETAILED PREDICTION & CONFIDENCE BREAKDOWN FOR EACH MODEL ON GROUND-TRUTH SLICES")
    print("=" * 125)
    print(f"{'Target Attack Slice':<22} | {'Packet 1D-CNN':<24} | {'Flow XGBoost':<24} | {'Time-Window XGBoost':<24} | {'Session CNN-LSTM':<22}")
    print("-" * 125)

    for label, res in model_results.items():
        p_str = f"{res['packet']['top1'][:14]} (P={res['packet']['P_att']:.2f})"
        f_str = f"{res['flow']['top1'][:14]} (P={res['flow']['P_att']:.2f})"
        t_str = f"{res['time_window']['top1'][:14]} (P={res['time_window']['P_att']:.2f})"
        s_str = f"{res['session']['top1'][:14]} (P={res['session']['P_att']:.2f})"

        print(f"{label:<22} | {p_str:<24} | {f_str:<24} | {t_str:<24} | {s_str:<22}")

    print("=" * 125)


if __name__ == "__main__":
    main()

