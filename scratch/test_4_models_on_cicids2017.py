#!/usr/bin/env python3
"""
=============================================================================
EVALUATE ALL 4 MODELS & HYBRID DECISION LAYER ON FULL CICIDS2017 DATASET
=============================================================================
"""

import os
import sys
import glob
import time
from pathlib import Path
from collections import Counter, defaultdict
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
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

# Mapping from 2017 CSV columns (with leading spaces) to 2018 schema
COL_RENAME_2017 = {
    ' Destination Port': 'Dst Port',
    ' Flow Duration': 'Flow Duration',
    ' Total Fwd Packets': 'Tot Fwd Pkts',
    ' Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    ' Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    ' Fwd Packet Length Max': 'Fwd Pkt Len Max',
    ' Fwd Packet Length Min': 'Fwd Pkt Len Min',
    ' Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
    ' Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    ' Bwd Packet Length Min': 'Bwd Pkt Len Min',
    ' Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    ' Bwd Packet Length Std': 'Bwd Pkt Len Std',
    'Flow Bytes/s': 'Flow Byts/s',
    ' Flow Packets/s': 'Flow Pkts/s',
    ' Flow IAT Mean': 'Flow IAT Mean',
    ' Flow IAT Std': 'Flow IAT Std',
    ' Flow IAT Max': 'Flow IAT Max',
    ' Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Tot',
    ' Fwd IAT Mean': 'Fwd IAT Mean',
    ' Fwd IAT Std': 'Fwd IAT Std',
    ' Fwd IAT Max': 'Fwd IAT Max',
    ' Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Tot',
    ' Bwd IAT Mean': 'Bwd IAT Mean',
    ' Bwd IAT Std': 'Bwd IAT Std',
    ' Bwd IAT Max': 'Bwd IAT Max',
    ' Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    ' Bwd PSH Flags': 'Bwd PSH Flags',
    ' Fwd URG Flags': 'Fwd URG Flags',
    ' Bwd URG Flags': 'Bwd URG Flags',
    ' Fwd Header Length': 'Fwd Header Len',
    ' Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s',
    ' Bwd Packets/s': 'Bwd Pkts/s',
    ' Min Packet Length': 'Pkt Len Min',
    ' Max Packet Length': 'Pkt Len Max',
    ' Packet Length Mean': 'Pkt Len Mean',
    ' Packet Length Std': 'Pkt Len Std',
    ' Packet Length Variance': 'Pkt Len Var',
    'FIN Flag Count': 'FIN Flag Cnt',
    ' SYN Flag Count': 'SYN Flag Cnt',
    ' RST Flag Count': 'RST Flag Cnt',
    ' PSH Flag Count': 'PSH Flag Cnt',
    ' ACK Flag Count': 'ACK Flag Cnt',
    ' URG Flag Count': 'URG Flag Cnt',
    ' CWE Flag Count': 'CWE Flag Count',
    ' ECE Flag Count': 'ECE Flag Cnt',
    ' Down/Up Ratio': 'Down/Up Ratio',
    ' Average Packet Size': 'Pkt Size Avg',
    ' Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    ' Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    ' Fwd Header Length.1': 'Fwd Header Len.1',
    'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg',
    ' Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
    ' Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg',
    ' Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
    ' Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg',
    'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts',
    ' Subflow Fwd Bytes': 'Subflow Fwd Byts',
    ' Subflow Bwd Packets': 'Subflow Bwd Pkts',
    ' Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    ' Init_Win_bytes_backward': 'Init Bwd Win Byts',
    ' act_data_pkt_fwd': 'Fwd Act Data Pkts',
    ' min_seg_size_forward': 'Fwd Seg Size Min',
    'Active Mean': 'Active Mean',
    ' Active Std': 'Active Std',
    ' Active Max': 'Active Max',
    ' Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    ' Idle Std': 'Idle Std',
    ' Idle Max': 'Idle Max',
    ' Idle Min': 'Idle Min',
    ' Label': 'Label',
}

LABEL_NORM_2017 = {
    'BENIGN': 'Benign',
    'DDoS': 'DDOS attack-HOIC',
    'PortScan': 'PortScan',
    'Bot': 'Bot',
    'Infiltration': 'Infilteration',
    'FTP-Patator': 'FTP-BruteForce',
    'SSH-Patator': 'SSH-Bruteforce',
    'DoS GoldenEye': 'DoS attacks-GoldenEye',
    'DoS Hulk': 'DoS attacks-Hulk',
    'DoS Slowhttptest': 'DoS attacks-SlowHTTPTest',
    'DoS slowloris': 'DoS attacks-Slowloris',
    'Heartbleed': 'Heartbleed',
}

def load_2017_slices():
    print("[*] Scanning and extracting pure slices from CICIDS2017 datasets...")
    target_files = {
        'DDOS attack-HOIC': ('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'DDoS', 200),
        'PortScan': ('cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'PortScan', 200),
        'Bot': ('cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Bot', 200),
        'Infilteration': ('cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', 'Infiltration', 36),
        'FTP-BruteForce': ('cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv', 'FTP-Patator', 200),
        'SSH-Bruteforce': ('cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv', 'SSH-Patator', 200),
        'DoS attacks-GoldenEye': ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'DoS GoldenEye', 200),
        'DoS attacks-Hulk': ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'DoS Hulk', 200),
        'DoS attacks-SlowHTTPTest': ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'DoS Slowhttptest', 200),
        'DoS attacks-Slowloris': ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'DoS slowloris', 200),
        'Benign': ('cicids2017/Monday-WorkingHours.pcap_ISCX.csv', 'BENIGN', 200),
    }

    # Also load Web Attacks from Thursday Morning
    web_file = 'cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    slices = {}

    for norm_lbl, (fpath, raw_lbl, n_samples) in target_files.items():
        if not os.path.exists(fpath):
            continue
        try:
            for chunk in pd.read_csv(fpath, chunksize=50000, low_memory=False, encoding='cp1252'):
                lbl_col = [c for c in chunk.columns if 'label' in c.lower()][0]
                chunk[lbl_col] = chunk[lbl_col].astype(str).str.strip()
                match = chunk[chunk[lbl_col] == raw_lbl]
                if len(match) > 0:
                    sample = match.head(n_samples).copy()
                    # Rename columns to 2018 schema
                    sample = sample.rename(columns=COL_RENAME_2017)
                    sample['Label'] = norm_lbl
                    # Ensure Timestamp and Protocol
                    if 'Timestamp' not in sample.columns:
                        sample['Timestamp'] = '15/02/2018 01:00:00'
                    if 'Protocol' not in sample.columns:
                        sample['Protocol'] = 6
                    slices[norm_lbl] = sample
                    break
        except Exception as e:
            print(f"Error loading {norm_lbl}: {e}")

    # Load web attacks
    if os.path.exists(web_file):
        try:
            for chunk in pd.read_csv(web_file, chunksize=50000, low_memory=False, encoding='cp1252'):
                lbl_col = [c for c in chunk.columns if 'label' in c.lower()][0]
                for raw_lbl, norm_lbl in [('Web Attack', 'Web Attack')]:
                    match_sqli = chunk[chunk[lbl_col].str.contains('Sql Injection', case=False, na=False)]
                    if len(match_sqli) > 0 and 'SQL Injection' not in slices:
                        s = match_sqli.copy().rename(columns=COL_RENAME_2017)
                        s['Label'] = 'SQL Injection'
                        s['Timestamp'] = '15/02/2018 01:00:00'
                        s['Protocol'] = 6
                        slices['SQL Injection'] = s

                    match_brute = chunk[chunk[lbl_col].str.contains('Brute Force', case=False, na=False)]
                    if len(match_brute) > 0 and 'Brute Force -Web' not in slices:
                        s = match_brute.head(100).copy().rename(columns=COL_RENAME_2017)
                        s['Label'] = 'Brute Force -Web'
                        s['Timestamp'] = '15/02/2018 01:00:00'
                        s['Protocol'] = 6
                        slices['Brute Force -Web'] = s

                    match_xss = chunk[chunk[lbl_col].str.contains('XSS', case=False, na=False)]
                    if len(match_xss) > 0 and 'Brute Force -XSS' not in slices:
                        s = match_xss.head(100).copy().rename(columns=COL_RENAME_2017)
                        s['Label'] = 'Brute Force -XSS'
                        s['Timestamp'] = '15/02/2018 01:00:00'
                        s['Protocol'] = 6
                        slices['Brute Force -XSS'] = s
        except Exception as e:
            print(f"Error loading web attacks: {e}")

    return slices

def main():
    print("=" * 90)
    print("TESTING ALL 4 ML MODELS & HYBRID DECISION LAYER ON CICIDS2017 TRAFFIC")
    print("=" * 90)

    slices = load_2017_slices()
    print(f"[+] Loaded {len(slices)} attack slices from CICIDS2017: {list(slices.keys())}\n")

    pkt_adapter = PacketModelAdapter()
    flow_adapter = FlowModelAdapter()
    tw_adapter = TimeWindowAdapter()
    sess_adapter = SessionModelAdapter()
    gate = AttackGate(threshold=0.35)
    rule_engine = RuleEngine()
    policy_agent = PolicyAgent(use_gemini=True)

    results = []

    for label, df_slice in slices.items():
        n = len(df_slice)
        
        # 1. Packet Model
        try:
            pkt_ev = pkt_adapter.predict(df_slice)
            pkt_str = f"{pkt_ev.prediction} (P={pkt_ev.attack_probability:.2f})"
        except Exception as e:
            pkt_ev = None
            pkt_str = f"Error: {e}"

        # 2. Flow Model
        try:
            flow_ev = flow_adapter.predict(df_slice)
            flow_str = f"{flow_ev.prediction} (P={flow_ev.attack_probability:.2f})"
        except Exception as e:
            flow_ev = None
            flow_str = f"Error: {e}"

        # 3. Time Window Model
        try:
            tw_adapter.reset_history()
            agg = preprocessing.window_aggregate(df_slice, window_start=pd.Timestamp('2018-02-15 01:00:00'))
            tw_ev = tw_adapter.predict(agg)
            tw_str = f"{tw_ev.prediction} (P={tw_ev.attack_probability:.2f})"
        except Exception as e:
            tw_ev = None
            tw_str = f"Error: {e}"

        # 4. Session Model
        try:
            sess_ev = sess_adapter.predict(df_slice)
            sess_str = f"{sess_ev.prediction} (P={sess_ev.attack_probability:.2f})"
        except Exception as e:
            sess_ev = None
            sess_str = f"Error: {e}"

        # 5. Integrated Pipeline with Hybrid Decision Layer
        try:
            integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id="W_2017")
            rule_res = rule_engine.evaluate(integrated, context={'consecutive_attack_windows': 1, 'escalation_level': 'LOW'})
            decision = policy_agent.decide(integrated, context_summary={'consecutive_attack_windows': 1, 'escalation_level': 'LOW'}, rule_result=rule_res)
            pipeline_str = f"{decision.action} ({decision.severity}, src={decision.source}, fam={decision.attack_family})"
        except Exception as e:
            pipeline_str = f"Error: {e}"

        results.append({
            'Target Attack (2017)': label,
            'Packet 1D-CNN': pkt_str,
            'Flow XGBoost': flow_str,
            'Time-Window': tw_str,
            'Session CNN-LSTM': sess_str,
            'Hybrid Decision Layer': pipeline_str,
        })

    print("=" * 160)
    print(f"{'Target Attack (2017)':<24} | {'Packet 1D-CNN':<22} | {'Flow XGBoost':<22} | {'Time-Window':<22} | {'Session CNN-LSTM':<22} | {'Hybrid Decision Layer'}")
    print("-" * 160)
    for r in results:
        print(f"{r['Target Attack (2017)']:<24} | {r['Packet 1D-CNN']:<22} | {r['Flow XGBoost']:<22} | {r['Time-Window']:<22} | {r['Session CNN-LSTM']:<22} | {r['Hybrid Decision Layer']}")
    print("=" * 160)

if __name__ == '__main__':
    main()

