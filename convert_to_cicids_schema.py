"""Convert flow-only data to CICIDS 2018 format with synthetic timestamps and labels."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def convert_to_cicids_format(input_csv, output_csv=None):
    """
    Convert a flow-feature CSV to CICIDS 2018 format by adding:
    - Timestamp (synthetic, sequential)
    - Label (synthetic, based on flow characteristics)
    """
    
    df = pd.read_csv(input_csv)
    df = df.copy()
    
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()[:30]}")
    
    # Add synthetic timestamps - spread evenly over 24 hours starting from 2018-03-01
    base_time = datetime(2018, 3, 1, 8, 0, 0)
    n_rows = len(df)
    time_deltas = [timedelta(seconds=float(i) * 86400 / n_rows) for i in range(n_rows)]
    df['Timestamp'] = [base_time + td for td in time_deltas]
    
    # Add synthetic labels based on flow characteristics
    # Simple heuristic: if Tot Fwd Pkts >> Tot Bwd Pkts, likely attack
    def assign_label(row):
        try:
            fwd_pkts = float(row.get('Tot Fwd Pkts', 0))
            bwd_pkts = float(row.get('Tot Bwd Pkts', 0))
            duration = float(row.get('Flow Duration', 0))
            
            # Attack heuristics
            if fwd_pkts > 100 and bwd_pkts < 5:
                return "DoS attacks-Hulk"
            elif fwd_pkts > 50 and duration > 5000000:
                return "DoS attacks-Slowloris"
            elif bwd_pkts > 50 and fwd_pkts < 10:
                return "DDOS attack-LOIC-UDP"
            elif fwd_pkts > 200:
                return "Infilteration"
            else:
                return "Benign"
        except:
            return "Benign"
    
    df['Label'] = df.apply(assign_label, axis=1)
    
    # Ensure all expected columns exist (fill with 0 if missing)
    expected_cols = [
        'Timestamp', 'Label', 'Dst Port', 'Protocol', 'Flow Duration',
        'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
        'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
        'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
        'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
        'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
        'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
        'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
        'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
        'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
        'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
        'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
        'Idle Max', 'Idle Min'
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Ensure Timestamp is first, Label is last
    cols = ['Timestamp'] + [c for c in expected_cols if c not in ['Timestamp', 'Label']] + ['Label']
    df = df[cols]
    
    # Save converted file
    if output_csv is None:
        output_csv = str(Path(input_csv).stem) + "_cicids_format.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"\nConverted output: {output_csv}")
    print(f"Output shape: {df.shape}")
    print(f"Label distribution:")
    print(df['Label'].value_counts())
    
    return output_csv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert flow-only data to CICIDS 2018 format")
    parser.add_argument("input_csv", help="Input CSV file (flow features without Timestamp/Label)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file (default: input_stem_cicids_format.csv)")
    args = parser.parse_args()
    
    output_path = convert_to_cicids_format(args.input_csv, args.output)
    print(f"\n✅ Conversion complete: {output_path}")
