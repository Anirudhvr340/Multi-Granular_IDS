import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter

print("=" * 80)
print("END-TO-END 4-MODEL VALIDATION ON MONDAY BENIGN BASELINE (CICIDS2017)")
print("=" * 80)

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1200, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)

# Preprocessing matching dashboard
for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
    if w_col in df.columns:
        df[w_col] = pd.to_numeric(df[w_col], errors='coerce').fillna(0).clip(lower=0)

durations = pd.to_numeric(df.get('Flow Duration', 1000), errors='coerce').fillna(1000).clip(lower=50) / 1e6
cum_time = [0.0]
for d in durations.values:
    cum_time.append(cum_time[-1] + float(d))
t_base = pd.Timestamp("2018-02-15 09:00:00")
df["Timestamp"] = [t_base + pd.Timedelta(seconds=float(s)) for s in cum_time[:len(df)]]

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()
flow_ad = FlowModelAdapter()
tw_ad = TimeWindowAdapter()

window_size = 60
num_windows = len(df) // window_size
print(f"Evaluating {num_windows} windows of {window_size} flows each...")

results = []
for w_i in range(num_windows):
    w_df = df.iloc[w_i * window_size : (w_i + 1) * window_size].copy()
    
    pkt_ev = pkt_ad.predict(w_df)
    sess_ev = sess_ad.predict(w_df)
    flow_ev = flow_ad.predict(w_df)
    tw_ev = tw_ad.predict(w_df)
    
    results.append({
        "Window": f"W_{w_i:02d}",
        "Packet_Pred": pkt_ev.prediction,
        "Packet_Patt": pkt_ev.attack_probability,
        "Session_Pred": sess_ev.prediction,
        "Session_Patt": sess_ev.attack_probability,
        "Flow_Pred": flow_ev.prediction,
        "Flow_Patt": flow_ev.attack_probability,
        "TW_Pred": tw_ev.prediction,
        "TW_Patt": tw_ev.attack_probability,
    })

res_df = pd.DataFrame(results)
print("\nResults per Window:")
print(res_df.to_string(index=False))

print("\nSummary of Predictions on Monday (Ground Truth: 100% Benign):")
print("Packet Model:")
print(res_df["Packet_Pred"].value_counts())
print("\nSession Model:")
print(res_df["Session_Pred"].value_counts())
print("\nFlow Model:")
print(res_df["Flow_Pred"].value_counts())
print("\nTime Window Model:")
print(res_df["TW_Pred"].value_counts())
