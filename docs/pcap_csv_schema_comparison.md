# Schema Comparison: Raw PCAP vs. Generated Flow CSV vs. Model Features

## 1. Schema Mapping Overview

This document compares the raw packet capture properties, the intermediate flow CSV schema produced from PCAPs, and the exact feature subsets expected by each of the four detection models.

```
+--------------------------------------------------------------------------------------------------+
| 1. RAW PCAP PACKET HEADERS (Ethernet / IP / TCP / UDP)                                            |
|    - Timestamp (epoch float)                                                                     |
|    - Src IP, Dst IP, Src Port, Dst Port, Protocol                                                |
|    - Packet Length (bytes)                                                                       |
|    - IP TTL, IP ID, IP Flags                                                                     |
|    - TCP Flags (SYN, ACK, FIN, RST, PSH, URG, ECE, CWR)                                          |
|    - TCP Window Size, Urgent Pointer                                                             |
+--------------------------------------------------------------------------------------------------+
                                               |
                                               v
+--------------------------------------------------------------------------------------------------+
| 2. INTERMEDIATE FLOW CSV SCHEMA (CICFlowMeter Standard - 79 Columns)                              |
|    - Identification: Timestamp, Dst Port, Protocol                                               |
|    - Volume: Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts                        |
|    - Length Moments: Fwd Pkt Len Max/Min/Mean/Std, Bwd Pkt Len Max/Min/Mean/Std                  |
|    - Inter-Arrival Times: Flow IAT Mean/Std/Max/Min, Fwd IAT Mean/Std/Max/Min, Bwd IAT ...       |
|    - Rates: Flow Byts/s, Flow Pkts/s, Fwd Pkts/s, Bwd Pkts/s                                     |
|    - TCP Flags: FIN, SYN, RST, PSH, ACK, URG, CWE, ECE Flag Counts                              |
|    - Subflows & Buffers: Subflow Fwd/Bwd Pkts/Bytes, Init Fwd/Bwd Win Byts                        |
|    - Activity & Idle: Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min                             |
+--------------------------------------------------------------------------------------------------+
             |                            |                           |                          |
             v                            v                           v                          v
+--------------------------+ +--------------------------+ +--------------------------+ +---------------------+
| 3. PACKET 1D-CNN (27)    | | 4. FLOW XGBOOST (17)     | | 5. WINDOW XGBOOST (268)  | | 6. SESSION CNN (27) |
| - pkt_fwd_ratio          | | - Tot Bwd Pkts           | | - 30s aggregate rates    | | - Flow Duration     |
| - pkt_bwd_ratio          | | - TotLen Bwd Pkts        | | - Packet length moments  | | - Flow Pkts/s       |
| - byte_fwd_ratio         | | - Fwd Pkt Len Max        | | - Port entropy           | | - Flow Byts/s       |
| - byte_bwd_ratio         | | - Bwd Pkt Len Max        | | - Protocol distributions | | - Flow IAT Mean..Min|
| - byte_per_pkt_avg       | | - Bwd Pkt Len Std        | | - Active / Idle sums     | | - Fwd/Bwd IAT Mean..|
| - fwd_hdr_ratio          | | - Flow IAT Min           | | - Window correlation     | | - Active / Idle Mean|
| - bwd_hdr_ratio          | | - Fwd IAT Tot/Std/Min    | |   (See temporal_features)| | - Down/Up Ratio     |
| - Flag ratios (fin..ece) | | - Fwd URG Flags          | |                          | | - Fwd/Bwd Header Len|
| - Window tiers (zero..lg)| | - Fwd Header Len         | |                          | | - Init Win Byts     |
| - Protocol indicators    | | - Fwd Pkts/s             | |                          | | - Flag Cnts (6)     |
| - Port categories        | | - RST/PSH/URG Flag Cnts  | |                          | | - Time_Delta        |
|                          | | - total_packets          | |                          | | (Sequence T = 30)   |
|                          | | - burstiness             | |                          | |                     |
+--------------------------+ +--------------------------+ +--------------------------+ +---------------------+
```

---

## 2. Detailed Feature Cross-Reference Table

| Model | Required Features | Source in Flow CSV | Preprocessing & Transformation |
| :--- | :--- | :--- | :--- |
| **Flow XGBoost** | 17 features | Direct columns + 2 engineered | `total_packets = Tot Fwd Pkts + Tot Bwd Pkts`; `burstiness = Flow IAT Std / (Flow IAT Mean + 1e-5)` |
| **Packet 1D-CNN** | 27 features | Derived from flow packet totals, lengths, flags, ports, protocols | Ratios normalized by total packets and total bytes; scaled with `cnn_scaler.pkl`; reshaped to `(N, 27, 1)` |
| **Session CNN-LSTM** | 27 features $\times$ 30 steps | 26 flow columns + `Time_Delta` | Contiguous sequences of 30 flows; `Time_Delta = Timestamp.diff().total_seconds()`; scaled with `session_scaler.pkl` |
| **Time-Window XGBoost**| 268 features | 30-second window aggregations | Computed via `step0_preprocessing.window_aggregate()`; min, max, mean, std, quantiles, and entropy |

---

## 3. Discrepancy Mitigations

1. **Header Name Normalization:** Raw PCAP generators sometimes output `Flow Duration` in microseconds or milliseconds. The pipeline enforces microsecond integers matching the CICIDS-2018 specification.
2. **Missing Flag Columns:** UDP packets do not contain TCP flags. The converter strictly zeroes TCP flags (`SYN`, `ACK`, `RST`, `FIN`, etc.) for UDP traffic rather than leaving `NaN` or uninitialized memory.
3. **Infinite Rates (`Flow Byts/s`, `Flow Pkts/s`):** When `Flow Duration == 0`, rate columns can become `inf`. The pipeline clips duration to $\ge 1.0\ \mu\text{s}$ to guarantee finite rate values.

