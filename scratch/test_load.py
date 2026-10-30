import sys
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

# Quick test of 2018 and 2017 data loading
print("Testing 2018 sample load...")
df18 = pd.read_csv("cicids2018/02-14-2018.csv", nrows=100)
print(f"2018 columns ({len(df18.columns)}):", df18.columns[:5].tolist())

print("Testing 2017 sample load...")
df17 = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=100, encoding="cp1252")
print(f"2017 columns ({len(df17.columns)}):", df17.columns[:5].tolist())

