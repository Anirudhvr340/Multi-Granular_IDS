import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from train_pytorch_packet_1d_cnn import extract_dimensionless_packet_features
import joblib
import torch

df = pd.read_csv('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skiprows=range(1, 18884), nrows=60, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()
print("Labels:", df['Label'].value_counts())
print("\nSample Ports:", df['Dst Port'].value_counts().head())
print("\nSample Tot Fwd Pkts:", df['Tot Fwd Pkts'].describe())
print("\nSample Flow Duration:", df['Flow Duration'].describe())

feats_df, _ = extract_dimensionless_packet_features(df)
print("\nExtracted features sample:")
print(feats_df[['pkt_len_mean', 'pkt_len_max', 'flow_pkts_s', 'down_up_ratio']].head())

le = joblib.load('packet_level/cnn_labels.pkl')
print("\nPacket classes:", list(le.classes_))
