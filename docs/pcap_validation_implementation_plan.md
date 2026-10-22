# PCAP-Based Validation Implementation Plan: Multi-Granular Agentic NIDS

## 1. Executive Summary & Verification Findings

This document outlines the forensic audit and technical implementation plan for unifying all four detection models (**Packet 1D-CNN**, **Flow XGBoost**, **Time-Window XGBoost**, and **Session CNN-LSTM**) under a **single, common PCAP data source**.

### Critical Discovery from Repository Audit
Prior documentation suggested that the Packet and Session models operated on raw PCAP packet bytes. **Forensic inspection of the actual training scripts (`train_packet_1d_cnn.py` and `train_session_cnn_lstm.py`) disproves this assumption:**
1. **Packet Model (`Packet 1D-CNN`):** Trained on **27 dimensionless packet-header ratios and flag densities** derived from flow records (`extract_dimensionless_packet_features`). It does *not* consume raw Ethernet/IP byte payloads directly.
2. **Session Model (`Session CNN-LSTM`):** Trained on **sliding sequences of $T=30$ consecutive bidirectional flows** (each represented by 27 statistical flow features and inter-arrival `Time_Delta`). It consumes sequences of flows, not raw byte streams.
3. **Flow Model (`Flow XGBoost`):** Trained on **17 selected discriminative flow features** from CICFlowMeter output.
4. **Time-Window Model (`Time-Window XGBoost`):** Trained on **268 temporal features** aggregated over 30-second windows of flow records.

**Architectural Consequence:**
Because all four models ultimately derive their features from the **bidirectional flow abstraction (the CICFlowMeter schema)**, converting raw PCAP files into standardized bidirectional flow records via CICFlowMeter provides a mathematically consistent, common input stream for all four granularities.

---

## 2. Actual Training vs. Inference Feature Contracts

### 2.1 Flow-Level Model (`Flow XGBoost`)
- **Model Artifact:** `xgb_flow_model.pkl` (XGBoost Classifier)
- **Features Artifact:** `important_features.pkl` (17 features)
- **Label Artifact:** `labels.pkl` (15 classes)
- **Calibration Artifact:** `calibrator_lr.pkl` (LogisticRegression calibrator)
- **Required Feature Names (Order Enforced):**
  1. `Tot Bwd Pkts`
  2. `TotLen Bwd Pkts`
  3. `Fwd Pkt Len Max`
  4. `Bwd Pkt Len Max`
  5. `Bwd Pkt Len Std`
  6. `Flow IAT Min`
  7. `Fwd IAT Tot`
  8. `Fwd IAT Std`
  9. `Fwd IAT Min`
  10. `Fwd URG Flags`
  11. `Fwd Header Len`
  12. `Fwd Pkts/s`
  13. `RST Flag Cnt`
  14. `PSH Flag Cnt`
  15. `URG Flag Cnt`
  16. `total_packets` (`Tot Fwd Pkts + Tot Bwd Pkts`)
  17. `burstiness` (`Flow IAT Std / (Flow IAT Mean + 1e-5)`)
- **Feature Scaling:** No scaler artifact required; tree-based model trained on raw values with engineered burstiness and total packet columns.
- **Output Classes (15):** `Benign`, `Bot`, `Brute Force -Web`, `Brute Force -XSS`, `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP`, `DDoS attacks-LOIC-HTTP`, `DoS attacks-GoldenEye`, `DoS attacks-Hulk`, `DoS attacks-SlowHTTPTest`, `DoS attacks-Slowloris`, `FTP-BruteForce`, `Infilteration`, `SQL Injection`, `SSH-Bruteforce`.

---

### 2.2 Time-Window Model (`Time-Window XGBoost`)
- **Model Artifact:** `time_window/processed/xgb_time_window_model.joblib` (XGBoost Classifier)
- **Feature Count:** 268 temporal features (`n_features_in_ = 268`)
- **Temporal Grouping:** Fixed 30-second chronological time slices (`floor('30s')`).
- **Feature Engineering:** Calculated by `temporal_features.py` and `step0_preprocessing.py`:
  - Volumetric sums, packet and byte transfer rates.
  - Mean, std, min, max, median, skewness, kurtosis of flow metrics across the window.
  - Rolling window trends and entropy of destination ports and protocols.
- **Output Classes (11):** 11 active attack and benign classes mapped to integers $[0..10]$.

---

### 2.3 Packet-Level Model (`Packet 1D-CNN`)
- **Model Artifact:** `packet_level/cnn_packet_model.keras` (Keras 3 / PyTorch backend 1D-CNN)
- **Feature Artifact:** `packet_level/cnn_packet_features.pkl` (27 features)
- **Scaler Artifact:** `packet_level/cnn_scaler.pkl` (`StandardScaler`, `n_features_in_ = 27`)
- **Label Artifact:** `packet_level/cnn_labels.pkl` (`LabelEncoder`, 15 classes)
- **Calibrator Artifact:** `packet_level/cnn_calibrator_lr.pkl`
- **Input Tensor Shape:** `(Batch_Size, 27, 1)`
- **Feature Set (27 Dimensionless Ratios & Indicators):**
  1. `pkt_fwd_ratio`: $\text{Tot Fwd Pkts} / \text{Total Pkts}$
  2. `pkt_bwd_ratio`: $\text{Tot Bwd Pkts} / \text{Total Pkts}$
  3. `byte_fwd_ratio`: $\text{TotLen Fwd Pkts} / \text{Total Bytes}$
  4. `byte_bwd_ratio`: $\text{TotLen Bwd Pkts} / \text{Total Bytes}$
  5. `byte_per_pkt_avg`: $\text{Total Bytes} / \text{Total Pkts}$
  6. `fwd_hdr_ratio`: $\text{Fwd Header Len} / (\text{TotLen Fwd Pkts} + 1)$
  7. `bwd_hdr_ratio`: $\text{Bwd Header Len} / (\text{TotLen Bwd Pkts} + 1)$
  8. `fin_ratio`: $\text{FIN Flag Cnt} / \text{Total Pkts}$
  9. `syn_ratio`: $\text{SYN Flag Cnt} / \text{Total Pkts}$
  10. `rst_ratio`: $\text{RST Flag Cnt} / \text{Total Pkts}$
  11. `psh_ratio`: $\text{PSH Flag Cnt} / \text{Total Pkts}$
  12. `ack_ratio`: $\text{ACK Flag Cnt} / \text{Total Pkts}$
  13. `urg_ratio`: $\text{URG Flag Cnt} / \text{Total Pkts}$
  14. `cwe_ratio`: $\text{CWE Flag Count} / \text{Total Pkts}$
  15. `ece_ratio`: $\text{ECE Flag Cnt} / \text{Total Pkts}$
  16. `fwd_win_zero`: Indicator ($\text{Init Fwd Win Byts} \le 0$)
  17. `fwd_win_small`: Indicator ($0 < \text{Init Fwd Win Byts} \le 1024$)
  18. `fwd_win_std`: Indicator ($1024 < \text{Init Fwd Win Byts} \le 65535$)
  19. `fwd_win_large`: Indicator ($\text{Init Fwd Win Byts} > 65535$)
  20. `is_tcp`: Indicator ($\text{Protocol} == 6$)
  21. `is_udp`: Indicator ($\text{Protocol} == 17$)
  22. `is_other_proto`: Indicator ($\text{Protocol} \notin \{6, 17\}$)
  23. `port_http`: Indicator ($\text{Dst Port} == 80$)
  24. `port_https`: Indicator ($\text{Dst Port} == 443$)
  25. `port_ssh`: Indicator ($\text{Dst Port} == 22$)
  26. `port_ftp`: Indicator ($\text{Dst Port} \in \{20, 21\}$)
  27. `port_ephemeral`: Indicator ($\text{Dst Port} > 1024$)

---

### 2.4 Session-Level Model (`Session CNN-LSTM`)
- **Model Artifact:** `session_level/session_cnn_lstm.pth` (PyTorch `SessionCNNLSTM`)
- **Config Artifact:** `session_level/session_config.pkl`
- **Features Artifact:** `session_level/session_features.pkl` (27 features)
- **Scaler Artifact:** `session_level/session_scaler.pkl` (`StandardScaler`, `n_features_in_ = 27`)
- **Labels Artifact:** `session_level/session_labels.pkl` (15 classes)
- **Input Tensor Shape:** `(Batch_Size, Sequence_Length = 30, Features = 27)`
- **Sequence Features (27):**
  `Flow Duration`, `Flow Pkts/s`, `Flow Byts/s`, `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`, `Flow IAT Min`, `Fwd IAT Mean`, `Fwd IAT Std`, `Bwd IAT Mean`, `Bwd IAT Std`, `Active Mean`, `Active Std`, `Idle Mean`, `Idle Std`, `Down/Up Ratio`, `Fwd Header Len`, `Bwd Header Len`, `Init Fwd Win Byts`, `Init Bwd Win Byts`, `FIN Flag Cnt`, `SYN Flag Cnt`, `RST Flag Cnt`, `PSH Flag Cnt`, `ACK Flag Cnt`, `URG Flag Cnt`, `Time_Delta`.
- **Temporal Grouping:** Flows are ordered by timestamp; sequential chunks of 30 flows are assembled, scaled with `session_scaler.pkl`, and processed through the Conv1D-LSTM architecture.

---

## 3. Discrepancies Between Training & Inference

| Dimension | Training Script Behavior | Previous Inference Adapter | Unified Solution |
| :--- | :--- | :--- | :--- |
| **Packet Model Input** | Computed 27 dimensionless features from flow columns. | Attempted to re-map or slice 77 features. | Strict feature pipeline computing the verified 27 dimensionless ratios, applying `cnn_scaler.pkl`, reshaping to `(N, 27, 1)`. |
| **Session Model Grouping** | Sliding sequence of 30 contiguous flows with `Time_Delta`. | Grouped by 30 flows, padded with zeros if sparse. | Preserved chronological session chunking with forward padding/masking for short windows. |
| **Flow Model Features** | Computed `total_packets` and `burstiness`. | Adapter computed `total_packets` and `burstiness` dynamically. | Maintained in `FlowPreprocessor`. |
| **Time-Window Alignment** | 30-second windows using `window_aggregate`. | 30-second windows using `step0_preprocessing`. | Reused deterministic windowing from `step0_preprocessing`. |

---

## 4. Changes Safe to Implement Without Retraining

1. **PCAP Ingestion & Flow Conversion:** Using `dpkt` / `scapy` / `cicflowmeter` to reconstruct bidirectional flows and calculate standard CICFlowMeter features directly from PCAP.
2. **Feature Extraction Pipeline:** Enforcing exact 17, 268, 27, and 27 feature orders for Flow, Window, Packet, and Session models respectively.
3. **Evidence Alignment & Aggregation:** Aggregating per-flow and per-sequence predictions into 30-second temporal windows for fusion.
4. **Agentic Layer Integration:** Forwarding aligned `IntegratedEvidence` into the unchanged `RuleEngine`, `ContextEngine`, and `PolicyAgent`.

---

## 5. Unresolved Assumptions & Limitations

1. **PCAP Ground-Truth Annotations:** Raw PCAP files do not store attack labels in packet headers. Ground truth must be mapped from the attack timing schedules or joined with the corresponding official daily CSVs via 5-tuple matching `(Src IP, Dst IP, Src Port, Dst Port, Protocol, Timestamp)`.
2. **Infiltration Probability Bias:** The Packet 1D-CNN and Session CNN-LSTM exhibit known elevated baseline probabilities for `Infilteration` due to class imbalance in their training holdouts. The agentic fusion engine safely counterbalances this via `R6_MODERATE_FLAG`.
3. **Execution Time on Large PCAPs:** Multi-gigabyte PCAPs require streaming chunk processing to prevent memory exhaustion on Windows development environments.

