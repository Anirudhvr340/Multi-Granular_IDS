import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1000, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)
df["Label"] = df["Label"].astype(str).str.strip()

# Apply the same pre-processing as dashboard
for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
    if w_col in df.columns:
        df[w_col] = pd.to_numeric(df[w_col], errors='coerce').fillna(0).clip(lower=0)

if 'TotLen Fwd Pkts' in df.columns and 'Fwd Header Len' in df.columns:
    df['TotLen Fwd Pkts'] = df['TotLen Fwd Pkts'] + df['Fwd Header Len']
if 'TotLen Bwd Pkts' in df.columns and 'Bwd Header Len' in df.columns:
    df['TotLen Bwd Pkts'] = df['TotLen Bwd Pkts'] + df['Bwd Header Len']

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()

# Evaluate windows of 60 flows
print("Evaluating windows of 60 flows from Monday (Benign):")
for i in range(0, 300, 60):
    w = df.iloc[i : i + 60]
    p_res = pkt_ad.predict(w)
    s_res = sess_ad.predict(w)
    print(f"Window {i//60}:")
    print(f"  Packet: Pred={p_res.prediction}, P(att)={p_res.attack_probability:.3f}, Conf={p_res.confidence:.3f}")
    print(f"  Session: Pred={s_res.prediction}, P(att)={s_res.attack_probability:.3f}, Conf={s_res.confidence:.3f}")

