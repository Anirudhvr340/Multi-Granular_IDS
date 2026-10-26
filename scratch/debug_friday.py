import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()

df = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=60, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)

pkt_ev = pkt_ad.predict(df)
print('Packet ev pred:', pkt_ev.prediction, 'p_att:', pkt_ev.attack_probability, 'is_att:', pkt_ev.is_attack)
print('Packet top predictions:', pkt_ev.top_predictions)
print('Packet supporting:', pkt_ev.supporting_evidence)

sess_ev = sess_ad.predict(df)
print('\nSession ev pred:', sess_ev.prediction, 'p_att:', sess_ev.attack_probability, 'is_att:', sess_ev.is_attack)
print('Session top predictions:', sess_ev.top_predictions)
print('Session supporting:', sess_ev.supporting_evidence)

