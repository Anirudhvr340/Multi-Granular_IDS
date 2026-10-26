import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np

from train_pytorch_packet_1d_cnn import extract_dimensionless_packet_features

# 1. 2017 Monday Benign
df_2017 = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1000)
df_2017.columns = df_2017.columns.str.strip()

# 2. 2018 Training Balanced
df_2018 = pd.read_csv("data_cache/packet_train_balanced.csv")
b_2018 = df_2018[df_2018["Label"] == "Benign"]
inf_2018 = df_2018[df_2018["Label"] == "Infilteration"]

f_2017, cols = extract_dimensionless_packet_features(df_2017)
f_b18, _ = extract_dimensionless_packet_features(b_2018)
f_inf, _ = extract_dimensionless_packet_features(inf_2018)

comp = pd.DataFrame({
    "2017_Monday_Benign": f_2017.mean().round(3),
    "2018_Benign": f_b18.mean().round(3),
    "2018_Infilteration": f_inf.mean().round(3),
})
comp["Diff_2017_vs_2018Benign"] = (comp["2017_Monday_Benign"] - comp["2018_Benign"]).round(3)
comp["Diff_2017_vs_Infil"] = (comp["2017_Monday_Benign"] - comp["2018_Infilteration"]).round(3)

print("Feature Comparison:")
print(comp.to_string())

