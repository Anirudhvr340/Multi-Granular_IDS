"""
=============================================================================
AUGMENT SESSION TRAINING WITH CICIDS2017 BENIGN SESSIONS
=============================================================================
Adds cross-dataset benign sessions from CICIDS2017 Monday + Tuesday + Wednesday
to the existing CICIDS2018 session_train_balanced.npz to improve domain
generalization.

Research basis:
  - Sommer & Paxson (2010) "Outside the Closed World": IDS models must be
    validated on diverse benign environments.
  - Engelen et al. (2021) "Troubleshooting an IDS Dataset": CICIDS2017/2018
    benign traffic has different statistical characteristics.
  - Li et al. (2024) "Cross-Dataset Transfer Learning for NIDS": domain
    adaptation via benign data augmentation improves generalisation.
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from nids_dashboard import COL_RENAME_2017

SESSION_FEATURES = [
    'Flow Duration', 'Flow Pkts/s', 'Flow Byts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std',
    'Down/Up Ratio', 'Fwd Header Len', 'Bwd Header Len', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Time_Delta'
]

CICIDS2017_BENIGN_FILES = [
    "cicids2017/Monday-WorkingHours.pcap_ISCX.csv",
    "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
    "cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
]

TARGET_PER_FILE = 2000   # 6000 total new benign sessions from 3 2017 files
SEQ_LEN = 30
STEP = 8    # denser stride for more session diversity

print("=" * 70)
print("CICIDS2017 BENIGN SESSION EXTRACTION FOR DOMAIN ADAPTATION")
print("=" * 70)

new_benign_sessions = []

for csv_path in CICIDS2017_BENIGN_FILES:
    p = Path(csv_path)
    if not p.exists():
        print(f"  [!] Skipping missing: {csv_path}")
        continue

    print(f"\n[*] Processing: {p.name}...")
    collected = 0

    for chunk in pd.read_csv(csv_path, chunksize=50000, encoding="cp1252", low_memory=False):
        if collected >= TARGET_PER_FILE:
            break

        chunk = chunk.rename(columns=COL_RENAME_2017)
        chunk.columns = chunk.columns.str.strip()
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # Keep only benign rows
        benign = chunk[chunk["Label"].str.upper() == "BENIGN"].copy()
        if benign.empty:
            continue

        # Clean numeric
        for col in SESSION_FEATURES[:-1]:
            if col in benign.columns:
                benign[col] = pd.to_numeric(benign[col], errors="coerce").replace(
                    [float("inf"), float("-inf")], np.nan
                ).fillna(0.0).clip(lower=0.0)
            else:
                benign[col] = 0.0

        # Clip Init Win values (negative means -1 unset)
        for w in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
            if w in benign.columns:
                benign[w] = benign[w].clip(lower=0.0)

        # Compute Time_Delta from flow durations (synthetic timestamp)
        durations = pd.to_numeric(
            benign.get("Flow Duration", 1000), errors="coerce"
        ).fillna(1000).clip(lower=50) / 1e6
        benign = benign.reset_index(drop=True)
        delta = np.concatenate([[0.0], np.diff(np.cumsum(durations.values))])
        benign["Time_Delta"] = np.clip(delta, 0.0, 300.0)

        feat_matrix = benign[SESSION_FEATURES].values.astype(np.float32)
        n_rows = len(feat_matrix)

        for s_i in range(0, n_rows - SEQ_LEN + 1, STEP):
            window = feat_matrix[s_i : s_i + SEQ_LEN]
            # Reject degenerate all-zero windows
            if np.all(window == 0.0):
                continue
            new_benign_sessions.append(window)
            collected += 1
            if collected >= TARGET_PER_FILE:
                break

    print(f"  [+] Extracted {collected} benign sessions from {p.name}")

print(f"\n[+] Total new CICIDS2017 benign sessions: {len(new_benign_sessions)}")

# ── Load original 2018 dataset ─────────────────────────────────────────────
npz_path = Path("data_cache/session_train_balanced.npz")
data = np.load(npz_path, allow_pickle=True)
X_orig = data["X"].astype(np.float32)
y_orig = data["y"]

print(f"\n[+] Original 2018 dataset: {len(X_orig)} sessions")
print("Original class counts:")
print(pd.Series(y_orig).value_counts().to_string())

# ── Merge: augment benign, keep all attacks ────────────────────────────────
new_X = np.array(new_benign_sessions, dtype=np.float32)
new_y = np.array(["Benign"] * len(new_X))

attack_mask = np.array([label != "Benign" for label in y_orig])
X_attacks = X_orig[attack_mask]
y_attacks = y_orig[attack_mask]

X_benign_2018 = X_orig[~attack_mask]
y_benign_2018 = y_orig[~attack_mask]

print(f"\n2018 benign sessions: {len(X_benign_2018)}")
print(f"2017 benign sessions added: {len(new_X)}")
print(f"Attack sessions: {len(X_attacks)}")

X_combined = np.concatenate([X_benign_2018, new_X, X_attacks], axis=0)
y_combined = np.concatenate([y_benign_2018, new_y, y_attacks], axis=0)

# Shuffle
rng = np.random.default_rng(42)
idx = rng.permutation(len(X_combined))
X_combined = X_combined[idx]
y_combined = y_combined[idx]

print(f"\n[+] Combined dataset: {len(X_combined)} sessions")
print("Combined class counts:")
print(pd.Series(y_combined).value_counts().to_string())

out_path = Path("data_cache/session_train_augmented.npz")
np.savez_compressed(out_path, X=X_combined, y=y_combined)
sz = out_path.stat().st_size / (1024 * 1024)
print(f"\n[+] Saved augmented session dataset → {out_path} ({sz:.1f} MB)")

