import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter

pkt_ad = PacketModelAdapter()

# 1. Monday Benign
df_mon = pd.read_csv('cicids2017/Monday-WorkingHours.pcap_ISCX.csv', nrows=1200, encoding='cp1252')
df_mon = df_mon.rename(columns=COL_RENAME_2017)
print("=== Monday Benign Stats ===")
for i in range(5):
    w = df_mon.iloc[i*60 : (i+1)*60]
    ev = pkt_ad.predict(w)
    s = ev.supporting_evidence
    print(f"Mon W{i}: mean_att={s['mean_attack_prob']:.3f}, frac_50={s.get('frac_50', 0):.3f}, frac_att={s['fraction_attack_packets']:.3f}, p95={s['p95_attack_prob']:.3f}")

# 2. Friday DDoS
df_ddos = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=300, encoding='cp1252')
df_ddos = df_ddos.rename(columns=COL_RENAME_2017)
print("\n=== Friday DDoS Stats ===")
for i in range(5):
    w = df_ddos.iloc[i*60 : (i+1)*60]
    ev = pkt_ad.predict(w)
    s = ev.supporting_evidence
    print(f"DDoS W{i}: mean_att={s['mean_attack_prob']:.3f}, frac_att={s['fraction_attack_packets']:.3f}, p95={s['p95_attack_prob']:.3f}")

