import sys
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
import step0_preprocessing as preprocessing

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()
flow_ad = FlowModelAdapter()
tw_ad = TimeWindowAdapter()

hdr = pd.read_csv("cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", nrows=1, encoding="cp1252").columns
df_atk = pd.read_csv(
    "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    skiprows=range(1, 18885),
    nrows=300,
    header=0,
    names=hdr,
    encoding="cp1252"
)
df_atk = df_atk.rename(columns=COL_RENAME_2017)
df_atk.columns = df_atk.columns.str.strip()

for i in range(5):
    w = df_atk.iloc[i*60 : (i+1)*60].copy()
    p = pkt_ad.predict(w)
    s = sess_ad.predict(w)
    f = flow_ad.predict(w)
    tw = tw_ad.predict(w)
    print(f"Window {i}:")
    print(f"  Packet:  {p.prediction:<20} P={p.attack_probability:.3f} is_att={p.is_attack}")
    print(f"  Session: {s.prediction:<20} P={s.attack_probability:.3f} is_att={s.is_attack}")
    print(f"  Flow:    {f.prediction:<20} P={f.attack_probability:.3f} is_att={f.is_attack}")
    print(f"  TW:      {tw.prediction:<20} P={tw.attack_probability:.3f} is_att={tw.is_attack}")

