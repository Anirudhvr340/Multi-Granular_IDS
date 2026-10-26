import os
import glob
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

SESSION_FEATURES = [
    'Flow Duration', 'Flow Pkts/s', 'Flow Byts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std',
    'Down/Up Ratio', 'Fwd Header Len', 'Bwd Header Len', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Time_Delta'
]

PACKET_RAW_COLS = [
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Header Len', 'Bwd Header Len', 'FIN Flag Cnt', 'SYN Flag Cnt',
    'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt', 'Init Fwd Win Byts', 'Protocol',
    'Dst Port', 'Timestamp', 'Label',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Size Avg',
    'Flow Duration', 'Flow Pkts/s', 'Flow Byts/s', 'Down/Up Ratio'
]

# Known attack classes per file
FILE_TARGETS = {
    "02-14-2018.csv": ["FTP-BruteForce", "SSH-Bruteforce"],
    "02-15-2018.csv": ["DoS attacks-GoldenEye", "DoS attacks-Slowloris"],
    "02-16-2018.csv": ["DoS attacks-SlowHTTPTest", "DoS attacks-Hulk"],
    "02-20-2018.csv": ["DDoS attacks-LOIC-HTTP"],
    "02-21-2018.csv": ["DDOS attack-LOIC-UDP", "DDOS attack-HOIC"],
    "02-22-2018.csv": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
    "02-23-2018.csv": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
    "02-28-2018.csv": ["Infilteration"],
    "03-01-2018.csv": ["Infilteration"],
    "03-02-2018.csv": ["Bot"],
}

ALL_COLS_SET = set(SESSION_FEATURES) | set(PACKET_RAW_COLS)

def main():
    print("=" * 80, flush=True)
    print("HIGH-SPEED STRATIFIED MULTI-GRANULAR DATA EXTRACTION (D: DRIVE)", flush=True)
    print("=" * 80, flush=True)
    
    csv_paths = sorted(glob.glob(r"D:\Day *\csv\*.csv"))
    print(f"Found {len(csv_paths)} CSV files.", flush=True)
    
    target_samples_per_attack = 8000
    target_benign_per_day = 12000
    target_benign_sess_per_day = 1200
    
    packet_data_by_class = defaultdict(list)
    session_sequences_by_class = defaultdict(list)
    
    t0 = time.time()
    
    for path in csv_paths:
        fname = os.path.basename(path)
        expected_attacks = FILE_TARGETS.get(fname, [])
        print(f"\n[*] Processing {fname} (Target attacks: {expected_attacks})...", flush=True)
        
        benign_collected_today = 0
        benign_sess_collected_today = 0
        
        try:
            # Determine which columns exist in file
            sample_df = pd.read_csv(path, nrows=2, low_memory=False)
            sample_cols = [c.strip() for c in sample_df.columns]
            use_cols = [c for c in sample_df.columns if c.strip() in ALL_COLS_SET or c.strip() == "Label"]
            
            for chunk_idx, raw_chunk in enumerate(pd.read_csv(path, usecols=use_cols, chunksize=50000, low_memory=False)):
                raw_chunk.columns = raw_chunk.columns.str.strip()
                chunk = raw_chunk[raw_chunk["Label"] != "Label"].copy()
                chunk["Label"] = chunk["Label"].astype(str).str.strip()
                if chunk.empty:
                    continue
                    
                # Clean numerical columns
                num_cols = [c for c in chunk.columns if c not in ["Label", "Timestamp"]]
                chunk[num_cols] = chunk[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
                # Ingest Packet Data
                for label, group in chunk.groupby("Label"):
                    if label == "Benign":
                        needed = target_benign_per_day - benign_collected_today
                        if needed > 0:
                            take = group.iloc[:needed]
                            packet_data_by_class["Benign"].append(take)
                            benign_collected_today += len(take)
                    else:
                        current_count = sum(len(x) for x in packet_data_by_class[label])
                        needed = target_samples_per_attack - current_count
                        if needed > 0:
                            take = group.iloc[:needed]
                            packet_data_by_class[label].append(take)
                
                # Session Sequence Extraction (sliding window of 30 flows)
                if "Timestamp" in chunk.columns:
                    ts = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                    if ts.isna().all():
                        ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
                    ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
                    chunk["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)
                else:
                    chunk["Time_Delta"] = 0.1
                
                for f_col in SESSION_FEATURES:
                    if f_col not in chunk.columns:
                        chunk[f_col] = 0.0
                
                feat_matrix = chunk[SESSION_FEATURES].values.astype(np.float32)
                labels = chunk["Label"].values
                seq_len = 30
                step = 15
                
                for s_i in range(0, len(chunk) - seq_len + 1, step):
                    sess_window = feat_matrix[s_i : s_i + seq_len]
                    sess_lbls = labels[s_i : s_i + seq_len]
                    
                    counts = pd.Series(sess_lbls).value_counts()
                    top_label = counts.index[0]
                    top_frac = counts.iloc[0] / seq_len
                    
                    # Sequence majority threshold: require >= 40% of flows
                    if top_label == "Benign":
                        chosen_label = "Benign"
                    elif top_frac >= 0.40:
                        chosen_label = top_label
                    else:
                        chosen_label = "Benign"
                        
                    curr_sess_cnt = len(session_sequences_by_class[chosen_label])
                    if chosen_label == "Benign":
                        if benign_sess_collected_today < target_benign_sess_per_day:
                            session_sequences_by_class["Benign"].append(sess_window)
                            benign_sess_collected_today += 1
                    else:
                        if curr_sess_cnt < 1500:
                            session_sequences_by_class[chosen_label].append(sess_window)
                
                # Check Early Exit for this file:
                all_attacks_satisfied = True
                for att in expected_attacks:
                    att_cnt = sum(len(x) for x in packet_data_by_class[att])
                    if att_cnt < target_samples_per_attack:
                        all_attacks_satisfied = False
                        break
                        
                if benign_collected_today >= target_benign_per_day and benign_sess_collected_today >= target_benign_sess_per_day and all_attacks_satisfied:
                    print(f"  -> Early exit at chunk {chunk_idx+1} (satisfied {expected_attacks} & {benign_collected_today} Benign flows, {benign_sess_collected_today} Benign sessions)", flush=True)
                    break
                    
        except Exception as e:
            print(f"[!] Error in {fname}: {e}", flush=True)
            
    print("\n" + "=" * 80, flush=True)
    print("EXTRACTION SUMMARY:", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Packet Dataset
    all_packet_dfs = []
    for cls, df_list in sorted(packet_data_by_class.items()):
        cls_df = pd.concat(df_list, ignore_index=True)
        print(f"  Packet Class: {cls:30s} -> {len(cls_df):,} flows", flush=True)
        all_packet_dfs.append(cls_df)
        
    final_packet_df = pd.concat(all_packet_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    packet_out = CACHE_DIR / "packet_train_balanced.csv"
    final_packet_df.to_csv(packet_out, index=False)
    print(f"\n[+] Saved balanced packet dataset: {packet_out} ({len(final_packet_df):,} flows, {os.path.getsize(packet_out)/(1024*1024):.1f} MB)", flush=True)
    
    # 2. Session Dataset
    all_X_sess = []
    all_y_sess = []
    print("\nSession Sequence Class Counts:", flush=True)
    for cls, seq_list in sorted(session_sequences_by_class.items()):
        if seq_list:
            seq_arr = np.array(seq_list, dtype=np.float32)
            all_X_sess.append(seq_arr)
            all_y_sess.extend([cls] * len(seq_list))
            print(f"  Session Class: {cls:30s} -> {len(seq_list):,} sessions", flush=True)
            
    X_sess_all = np.concatenate(all_X_sess, axis=0)
    y_sess_all = np.array(all_y_sess)
    
    shuff_idx = np.random.RandomState(42).permutation(len(X_sess_all))
    X_sess_all = X_sess_all[shuff_idx]
    y_sess_all = y_sess_all[shuff_idx]
    
    session_out = CACHE_DIR / "session_train_balanced.npz"
    np.savez_compressed(session_out, X=X_sess_all, y=y_sess_all)
    print(f"\n[+] Saved balanced session dataset: {session_out} ({len(X_sess_all):,} sessions, {os.path.getsize(session_out)/(1024*1024):.1f} MB)", flush=True)
    print(f"[+] Total extraction elapsed time: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
