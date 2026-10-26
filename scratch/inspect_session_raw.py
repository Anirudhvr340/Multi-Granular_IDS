import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
from nids_dashboard import COL_RENAME_2017
from nids.adapters.session_adapter import SessionModelAdapter, SESSION_FEATURES

sess_ad = SessionModelAdapter()

df = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=60, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()

print("Flow Duration mean:", df['Flow Duration'].mean())
print("Flow Pkts/s mean:", df['Flow Pkts/s'].mean())
print("Flow Byts/s mean:", df['Flow Byts/s'].mean())
print("Tot Fwd Pkts mean:", df['Tot Fwd Pkts'].mean())
print("Tot Bwd Pkts mean:", df['Tot Bwd Pkts'].mean())

# Check session tensors built by adapter
feats = df[sess_ad.features[:-1]].copy()
print("Clean features shape:", feats.shape)
print("NaN count in features:", feats.isna().sum().sum())

