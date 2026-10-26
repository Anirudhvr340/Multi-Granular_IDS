import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.time_window_adapter import TimeWindowAdapter

tw = TimeWindowAdapter()
df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=300, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

for i in range(5):
    w = df.iloc[i*60 : (i+1)*60]
    ev = tw.predict(w)
    print(f"\nWindow {i}: top={ev.prediction}, p_att={ev.attack_probability:.3f}, conf={ev.confidence:.3f}")
    top3 = sorted(ev.class_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    print("  Top 3:", top3)

