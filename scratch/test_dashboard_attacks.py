import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import SOCDashboardApp, COL_RENAME_2017
import tkinter as tk

root = tk.Tk()
root.withdraw()
app = SOCDashboardApp(root)

scenarios = [
    ("Wednesday DoS Slowloris", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", 26700),
    ("Wednesday DoS Hulk", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", 74860),
    ("Friday DDoS", "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 18884),
]

print("=" * 80)
print("TESTING LIVE DASHBOARD WINDOW PROCESSING ON ATTACK DATASETS:")
print("=" * 80)

for name, filepath, start_row in scenarios:
    print(f"\n--- {name} ---")
    header_df = pd.read_csv(filepath, nrows=1, encoding="cp1252")
    cols = header_df.columns
    df = pd.read_csv(
        filepath,
        skiprows=range(1, start_row + 1),
        nrows=180,
        header=0,
        names=cols,
        encoding="cp1252"
    )
    df = df.rename(columns=COL_RENAME_2017)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip().replace("", "-")

    for w_idx in range(0, 180, 60):
        w_slice = df.iloc[w_idx : w_idx + 60].copy()
        rec = app._process_window(w_slice, name, Path(filepath).name)
        print(f"Window {w_idx//60:02d} | GT: {rec['ground_truth']:<18} | Action: {rec['action']:<6} [{rec['severity']:<8}] | Detected: {rec['attack_family']} | "
              f"Pkt={rec['pkt_ev']['pred']} ({rec['pkt_ev']['prob']:.2f}), "
              f"Sess={rec['sess_ev']['pred']} ({rec['sess_ev']['prob']:.2f}), "
              f"Flow={rec['flow_ev']['pred']} ({rec['flow_ev']['prob']:.2f}), "
              f"TW={rec['tw_ev']['pred']} ({rec['tw_ev']['prob']:.2f})")

root.destroy()

