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
    'Dst Port', 'Timestamp', 'Label'
]

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk[chunk["Label"] != "Label"]
    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    
    # Clean numerical columns
    for col in chunk.columns:
        if col not in ["Label", "Timestamp"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk = chunk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return chunk

def main():
    print("=" * 80)
    print("BUILDING BALANCED MULTI-GRANULAR DATASET FROM D: DRIVE (ALL 10 DAYS)")
    print("=" * 80)
    
    csv_paths = sorted(glob.glob(r"D:\Day *\csv\*.csv"))
    print(f"Found {len(csv_paths)} CSV files.")
    
    target_samples_per_attack = 10000
    target_benign_per_day = 3000
    
    packet_data_by_class = defaultdict(list)
    session_sequences_by_class = defaultdict(list)
    
    t0 = time.time()
    
    for path in csv_paths:
        fname = os.path.basename(path)
        print(f"\n[*] Streaming from {fname}...")
        benign_collected_today = 0
        
        # Read file in 50k chunks
        try:
            for chunk_idx, raw_chunk in enumerate(pd.read_csv(path, chunksize=50000, low_memory=False)):
                chunk = clean_chunk(raw_chunk)
                if chunk.empty:
                    continue
                    
                # 1. Packet Data Collection (Per Flow/Packet Record)
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
                
                # 2. Session Sequence Extraction (Sliding Windows of 30 flows)
                # Compute Time_Delta
                if "Timestamp" in chunk.columns:
                    ts = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                    if ts.isna().all():
                        ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
                    ts = ts.fillna(pd.Timestamp("2018-02-15 01:00:00"))
                    chunk["Time_Delta"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=300.0)
                else:
                    chunk["Time_Delta"] = 0.1
                
                # Check required session features
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
                    
                    # Robust Majority Labeling for Session Sequences:
                    # Require >= 40% of flows in the sequence to belong to the attack
                    counts = pd.Series(sess_lbls).value_counts()
                    top_label = counts.index[0]
                    top_frac = counts.iloc[0] / seq_len
                    
                    if top_label == "Benign":
                        chosen_label = "Benign"
                    elif top_frac >= 0.40:
                        chosen_label = top_label
                    else:
                        chosen_label = "Benign"  # If attack is <40% of sequence, it is mostly background
                        
                    curr_sess_cnt = len(session_sequences_by_class[chosen_label])
                    max_sess = 2500 if chosen_label != "Benign" else 4000
                    if curr_sess_cnt < max_sess:
                        session_sequences_by_class[chosen_label].append(sess_window)
                        
        except Exception as e:
            print(f"[!] Error in {fname}: {e}")
            
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE! Summary of Ingested Dataset:")
    print("=" * 80)
    
    # Combine Packet Data
    all_packet_dfs = []
    for cls, df_list in packet_data_by_class.items():
        cls_df = pd.concat(df_list, ignore_index=True)
        print(f"  Packet Class: {cls:30s} -> {len(cls_df):,} flows")
        all_packet_dfs.append(cls_df)
        
    final_packet_df = pd.concat(all_packet_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    packet_out = CACHE_DIR / "packet_train_balanced.parquet"
    final_packet_df.to_parquet(packet_out, index=False)
    print(f"\n[+] Saved balanced packet dataset: {packet_out} ({len(final_packet_df):,} flows, {os.path.getsize(packet_out)/(1024*1024):.1f} MB)")
    
    # Combine Session Sequences
    all_X_sess = []
    all_y_sess = []
    print("\nSession Sequence Class Counts:")
    for cls, seq_list in session_sequences_by_class.items():
        if seq_list:
            seq_arr = np.array(seq_list, dtype=np.float32)
            all_X_sess.append(seq_arr)
            all_y_sess.extend([cls] * len(seq_list))
            print(f"  Session Class: {cls:30s} -> {len(seq_list):,} sessions (Shape: {seq_arr.shape})")
            
    X_sess_all = np.concatenate(all_X_sess, axis=0)
    y_sess_all = np.array(all_y_sess)
    
    # Shuffle sessions
    shuff_idx = np.random.RandomState(42).permutation(len(X_sess_all))
    X_sess_all = X_sess_all[shuff_idx]
    y_sess_all = y_sess_all[shuff_idx]
    
    session_out = CACHE_DIR / "session_train_balanced.npz"
    np.savez_compressed(session_out, X=X_sess_all, y=y_sess_all)
    print(f"\n[+] Saved balanced session dataset: {session_out} ({len(X_sess_all):,} sessions, {os.path.getsize(session_out)/(1024*1024):.1f} MB)")
    print(f"[+] Total extraction elapsed time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

