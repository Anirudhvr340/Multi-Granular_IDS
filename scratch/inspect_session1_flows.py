import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=150, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

s1_flows = df.iloc[75:105]
print("Session 1 flows summary:")
print(s1_flows[['Dst Port', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Header Len', 'Bwd Header Len', 'Init Fwd Win Byts']].describe())

