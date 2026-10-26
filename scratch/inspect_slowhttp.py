import pandas as pd

ports = {}
for c in pd.read_csv(r"D:\Day 3 - Feb 16\csv\02-16-2018.csv", chunksize=100000, usecols=["Dst Port", "Label"], low_memory=False):
    m = c[c["Label"] == "DoS attacks-SlowHTTPTest"]
    if not m.empty:
        for p, count in m["Dst Port"].value_counts().items():
            ports[p] = ports.get(p, 0) + count
print("All SlowHTTPTest ports:", ports)

