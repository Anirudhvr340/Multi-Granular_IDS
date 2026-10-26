import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import SOCDashboardApp, COL_RENAME_2017
import tkinter as tk

# Initialize hidden Tk root to create the dashboard app instance
root = tk.Tk()
root.withdraw()

app = SOCDashboardApp(root)

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1200, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)
df["Label"] = df["Label"].astype(str).str.strip().replace("", "-")

print("\n" + "=" * 80)
print("TESTING LIVE DASHBOARD WINDOW PROCESSING ON 20 MONDAY BENIGN WINDOWS:")
print("=" * 80)

actions = []
for w_idx in range(0, 1200, 60):
    w_slice = df.iloc[w_idx : w_idx + 60].copy()
    rec = app._process_window(w_slice, "Day 1: Monday (Benign Baseline)", "Monday-WorkingHours.pcap_ISCX.csv")
    actions.append(rec['action'])
    print(f"Window {w_idx//60:02d} ({rec['time_str']}): Action: {rec['action']:<6} [{rec['severity']:<8}] | Detected: {rec['attack_family']} | "
          f"Pkt={rec['pkt_ev']['pred']} ({rec['pkt_ev']['prob']:.2f}), "
          f"Sess={rec['sess_ev']['pred']} ({rec['sess_ev']['prob']:.2f}), "
          f"Flow={rec['flow_ev']['pred']} ({rec['flow_ev']['prob']:.2f}), "
          f"TW={rec['tw_ev']['pred']} ({rec['tw_ev']['prob']:.2f})")

print("\nAction counts across 20 windows:")
print(pd.Series(actions).value_counts())

root.destroy()
