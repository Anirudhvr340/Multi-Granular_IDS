import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import SOCDashboardApp, COL_RENAME_2017
import tkinter as tk

root = tk.Tk()
root.withdraw()
app = SOCDashboardApp(root)

print("=" * 80)
print("TESTING LIVE DASHBOARD ON WEDNESDAY ATTACKS (CICIDS2017 - DoS)")
print("=" * 80)

# Load Wednesday rows around where DoS attacks occur (starts at index 6558)
df_wed = pd.read_csv("cicids2017/Wednesday-workingHours.pcap_ISCX.csv", skiprows=6550, nrows=200, header=None, encoding="cp1252")
headers = pd.read_csv("cicids2017/Wednesday-workingHours.pcap_ISCX.csv", nrows=1, encoding="cp1252").columns
df_wed.columns = headers
df_wed = df_wed.rename(columns=COL_RENAME_2017)
df_wed["Label"] = df_wed["Label"].astype(str).str.strip().replace("", "-")

print(f"Loaded {len(df_wed)} rows around first attack:")
print(df_wed["Label"].value_counts())

for w_i in range(0, 180, 60):
    w_slice = df_wed.iloc[w_i : w_i + 60].copy()
    rec = app._process_window(w_slice, "Day 4: Wednesday (DoS Attacks)", "Wednesday-workingHours.pcap_ISCX.csv")
    print(f"\nWindow {w_i//60}:")
    print(f"  Ground Truth:    {rec['ground_truth']}")
    print(f"  Action:          {rec['action']} [{rec['severity']}]")
    print(f"  Detected Family: {rec['attack_family']}")
    print(f"  Evidence: Packet={rec['pkt_ev']['pred']} (P={rec['pkt_ev']['prob']:.3f}), "
          f"Flow={rec['flow_ev']['pred']} (P={rec['flow_ev']['prob']:.3f}), "
          f"TW={rec['tw_ev']['pred']} (P={rec['tw_ev']['prob']:.3f}), "
          f"Session={rec['sess_ev']['pred']} (P={rec['sess_ev']['prob']:.3f})")

root.destroy()
