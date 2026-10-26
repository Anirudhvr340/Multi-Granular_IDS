import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.session_adapter import SESSION_FEATURES

df = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=60, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()

missing = [col for col in SESSION_FEATURES if col not in df.columns]
print("Missing features:", missing)
present = [col for col in SESSION_FEATURES if col in df.columns]
print("Present features:", len(present), "out of", len(SESSION_FEATURES))
print(df[present].head(2))

