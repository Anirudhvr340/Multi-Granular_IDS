"""Convert all CICIDS2017 no-metadata Parquet files to the project flow schema."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Fwd Packets Length Total": "TotLen Fwd Pkts",
    "Bwd Packets Length Total": "TotLen Bwd Pkts",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Fwd URG Flags",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
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
    "Down/Up Ratio": "Down/Up Ratio",
    "Avg Packet Size": "Pkt Size Avg",
    "Fwd Seg Size Min": "Fwd Seg Size Min",
    "Active Mean": "Active Mean",
    "Idle Mean": "Idle Mean",
    "Init Fwd Win Bytes": "Init Fwd Win Byts",
    "Init Bwd Win Bytes": "Init Bwd Win Byts",
}

REQUIRED_COLUMNS = [
    "Timestamp", "Label", "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Pkt Len Mean",
    "Active Mean", "Idle Mean", "SYN Flag Cnt", "FIN Flag Cnt",
    "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
]

LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS attacks-Hulk",
    "DoS GoldenEye": "DoS attacks-GoldenEye",
    "DoS slowloris": "DoS attacks-Slowloris",
    "DoS Slowhttptest": "DoS attacks-SlowHTTPTest",
    "DDoS": "DDOS attack-HOIC",
    "FTP-Patator": "FTP-BruteForce",
    "SSH-Patator": "SSH-Bruteforce",
    "Web Attack - Brute Force": "Brute Force -Web",
    "Web Attack - XSS": "Brute Force -XSS",
    "Web Attack - Sql Injection": "SQL Injection",
    "Infiltration": "Infilteration",
    "Bot": "Bot",
    "PortScan": "Portscan",
}


def convert_folder(input_dir, output_csv):
    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    total = 0
    start_time = datetime(2018, 1, 1)
    for file_index, path in enumerate(files):
        frame = pd.read_parquet(path)
        frame = frame.rename(columns=COLUMN_MAP)
        for column in REQUIRED_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0 if column not in ("Timestamp", "Label", "Protocol") else ("Benign" if column == "Label" else "0")
        def normalize_label(value):
            value = str(value).strip().replace("�", "-")
            lowered = value.lower()
            if "web attack" in lowered and "brute" in lowered:
                return "Brute Force -Web"
            if "web attack" in lowered and "xss" in lowered:
                return "Brute Force -XSS"
            if "web attack" in lowered and "sql" in lowered:
                return "SQL Injection"
            return LABEL_MAP.get(value, value)

        frame["Label"] = frame["Label"].map(normalize_label)
        frame["Timestamp"] = [
            start_time + timedelta(days=file_index, seconds=index * 86400 / max(len(frame), 1))
            for index in range(len(frame))
        ]
        frame["Dst Port"] = 0
        frame["Protocol"] = pd.to_numeric(frame["Protocol"], errors="coerce").fillna(0).astype(np.float32)
        for column in REQUIRED_COLUMNS:
            if column not in ("Timestamp", "Label", "Protocol"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        frame = frame[REQUIRED_COLUMNS]
        frame.to_csv(output, mode="a", header=not output.exists(), index=False)
        total += len(frame)
        print(f"{path.name}: {len(frame):,} rows")
    print(f"Converted {total:,} rows from {len(files)} files to {output}")
    print(pd.read_csv(output, usecols=["Label"])["Label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output", default="cicids2017_combined_cicids2018_schema.csv")
    args = parser.parse_args()
    convert_folder(args.input_dir, args.output)
