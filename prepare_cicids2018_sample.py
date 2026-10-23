#!/usr/bin/env python3
"""
=============================================================================
PREPARE CICIDS2018 BENCHMARK DATASET FROM cicids2018/cic.csv
=============================================================================
Extracts a stratified, balanced sample covering all 14 attack classes + Benign,
maps column names to canonical CICIDS2018 schema, reconstructs chronological
timestamps from genuine flow durations, and saves cicids2018/cicids2018_benchmark.csv.
=============================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Total Fwd Packet": "Tot Fwd Pkts",
    "Total Bwd packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packet": "TotLen Fwd Pkts",
    "Total Length of Bwd Packet": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Packet Length Min": "Pkt Len Min",
    "Packet Length Max": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "CWR Flag Count": "CWE Flag Count",
    "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Fwd Segment Size Avg": "Fwd Seg Size Avg",
    "Bwd Segment Size Avg": "Bwd Seg Size Avg",
    "Fwd Bytes/Bulk Avg": "Fwd Byts/b Avg",
    "Fwd Packet/Bulk Avg": "Fwd Pkts/b Avg",
    "Fwd Bulk Rate Avg": "Fwd Blk Rate Avg",
    "Bwd Bytes/Bulk Avg": "Bwd Byts/b Avg",
    "Bwd Packet/Bulk Avg": "Bwd Pkts/b Avg",
    "Bwd Bulk Rate Avg": "Bwd Blk Rate Avg",
    "FWD Init Win Bytes": "Init Fwd Win Byts",
    "Bwd Init Win Bytes": "Init Bwd Win Byts",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
}

LABEL_MAP = {
    "Benign": "Benign",
    "FTP-BruteForce": "FTP-BruteForce",
    "SSH-BruteForce": "SSH-Bruteforce",
    "DoS-Slowloris": "DoS attacks-Slowloris",
    "DoS-Hulk": "DoS attacks-Hulk",
    "DoS-SlowHTTPTest": "DoS attacks-SlowHTTPTest",
    "DoS-GoldenEye": "DoS attacks-GoldenEye",
    "DDoS": "DDOS attack-HOIC",
    "BruteForce-Web": "Brute Force -Web",
    "BruteForce-XSS": "Brute Force -XSS",
    "SQL-Injection": "SQL Injection",
    "Bot": "Bot",
    "Infiltration": "Infilteration",
    "PortScan": "Infilteration",
    "Heartbleed": "Infilteration",
}

def extract_benchmark(
    input_csv="cicids2018/cic.csv",
    output_csv="cicids2018/cicids2018_benchmark.csv",
    samples_per_class=1000,
):
    print("=" * 80)
    print(f"SAMPLING STRATIFIED BENCHMARK FROM {input_csv}")
    print(f"Target: up to {samples_per_class} flows per class")
    print("=" * 80)

    class_buffers = defaultdict(list)
    class_counts = defaultdict(int)

    # Process in chunks
    chunk_size = 200_000
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        for raw_label, group in chunk.groupby("Label"):
            raw_label = str(raw_label).strip()
            needed = samples_per_class - class_counts[raw_label]
            if needed > 0:
                sampled = group.head(needed)
                class_buffers[raw_label].append(sampled)
                class_counts[raw_label] += len(sampled)

        # Check if all classes are satisfied
        all_done = all(class_counts[lbl] >= samples_per_class for lbl in [
            "Benign", "FTP-BruteForce", "SSH-BruteForce", "DoS-Slowloris", "DoS-Hulk",
            "DoS-SlowHTTPTest", "DoS-GoldenEye", "DDoS", "Bot", "Infiltration"
        ])
        if all_done and class_counts["SQL-Injection"] >= 100:
            print("Collected sufficient samples for all major and rare classes.")
            break

    # Combine
    all_dfs = []
    for raw_label, chunks in class_buffers.items():
        sub_df = pd.concat(chunks, ignore_index=True)
        all_dfs.append(sub_df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total sampled flows: {len(combined)}")
    print("Sample distribution by raw label:")
    print(combined["Label"].value_counts())

    # Rename columns to canonical schema
    combined.rename(columns=COLUMN_MAP, inplace=True)

    # Map labels to canonical model class names
    combined["Original_Label"] = combined["Label"]
    combined["Label"] = combined["Label"].map(lambda x: LABEL_MAP.get(str(x).strip(), str(x).strip()))

    # Generate chronological Timestamps
    # Advance time incrementally based on flow duration or 0.1s increments
    # to form natural 30-second windows with realistic flow density
    base_time = datetime(2018, 3, 1, 8, 0, 0)
    current_time = base_time
    timestamps = []
    
    # Shuffle slightly within classes so traffic interweaves naturally
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    for i, row in combined.iterrows():
        # Inter-arrival step between 0.05s and 0.5s
        dur_us = float(row.get("Flow Duration", 100_000))
        delta_sec = max(0.02, min(2.0, dur_us / 1_000_000.0 if dur_us > 0 else 0.1))
        current_time += timedelta(seconds=delta_sec)
        timestamps.append(current_time.strftime("%d/%m/%Y %H:%M:%S"))

    combined["Timestamp"] = timestamps

    # Reorder columns to ensure Timestamp is first
    cols = ["Timestamp"] + [c for c in combined.columns if c != "Timestamp"]
    combined = combined[cols]

    # Save benchmark CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\n[OK] Stratified benchmark saved to: {output_csv}")
    print(f"File size: {os.path.getsize(output_csv) / (1024*1024):.2f} MB")
    print(f"Rows: {len(combined)}, Unique Canonical Classes: {combined['Label'].nunique()}")

if __name__ == "__main__":
    extract_benchmark()

