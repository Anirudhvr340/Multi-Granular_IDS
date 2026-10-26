import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017

df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=10, encoding="cp1252")
print("Original 2017 columns with leading space count:", sum(1 for c in df.columns if c.startswith(" ")))

# Apply rename
df_renamed = df.rename(columns=COL_RENAME_2017)
print("After COL_RENAME_2017:")
print("Dst Port in df_renamed:", "Dst Port" in df_renamed.columns)
print("Tot Fwd Pkts in df_renamed:", "Tot Fwd Pkts" in df_renamed.columns)
print("TotLen Fwd Pkts in df_renamed:", "TotLen Fwd Pkts" in df_renamed.columns)
print("Fwd Header Len in df_renamed:", "Fwd Header Len" in df_renamed.columns)
print("Init Fwd Win Byts in df_renamed:", "Init Fwd Win Byts" in df_renamed.columns)

# Now check if extract_dimensionless_packet_features produces non-zeros!
from train_pytorch_packet_1d_cnn import extract_dimensionless_packet_features
feats, cols = extract_dimensionless_packet_features(df_renamed)
print("\nExtracted features mean on renamed df:")
print(feats.mean().round(3))
