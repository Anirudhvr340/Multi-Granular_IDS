import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from session_level.train_session_cnn_lstm import SESSION_FEATURES

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=300, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

w1 = df.iloc[60:120]
print("Session features in Window 1 (Monday):")
for f in SESSION_FEATURES:
    if f in w1.columns:
        print(f"{f}: mean={w1[f].mean():.2f}, min={w1[f].min():.2f}, max={w1[f].max():.2f}")
    else:
        print(f"{f}: NOT FOUND!")

