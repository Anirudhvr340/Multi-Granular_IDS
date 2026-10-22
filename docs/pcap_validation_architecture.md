# PCAP Validation Architecture: Multi-Granular Agentic NIDS

## 1. System Architecture Overview

The PCAP-based validation pipeline establishes a unified ingestion and inference flow where a single raw packet capture (`.pcap`) serves as the root data source for all four detection models and the hierarchical agentic decision layer.

```
                           +--------------------------------------+
                           |   Raw Network Capture (.pcap)        |
                           |   (Local or External HDD Storage)    |
                           +------------------+-------------------+
                                              |
                                              v
                           +--------------------------------------+
                           |   PCAP Reader & Flow Assembler       |
                           |   (dpkt / scapy / cicflowmeter)      |
                           +------------------+-------------------+
                                              |
                                              v
                           +--------------------------------------+
                           | Standardized Bidirectional Flow CSV  |
                           | (79 Statistical Features + 5-Tuples) |
                           +------------------+-------------------+
                                              |
      +------------------------+--------------+--------------+------------------------+
      |                        |                             |                        |
      v                        v                             v                        v
+---------------+      +----------------+            +---------------+        +---------------+
| PACKET BRANCH |      |  FLOW BRANCH   |            | WINDOW BRANCH |        |SESSION BRANCH |
| Dimensionless |      |  17 Selected   |            |  30s Temporal |        | Sequence (30) |
| Ratios (27)   |      |  Features (17) |            |Aggregate (268)|        |  Chunkn (27)  |
| 1D-CNN Model  |      | XGBoost Model  |            | XGBoost Model |        |CNN-LSTM Model |
+-------+-------+      +-------+--------+            +-------+-------+        +-------+-------+
        |                      |                             |                        |
        +----------------------+--------------+--------------+------------------------+
                                              |
                                              v
                           +--------------------------------------+
                           |   Evidence Aligner & Aggregator      |
                           |   (30s Window Identifier Wxxxxx)     |
                           +------------------+-------------------+
                                              |
                                              v
                           +--------------------------------------+
                           |      Attack Gate & Fusion Engine     |
                           |      Fused P(attack) + Agreement     |
                           +------------------+-------------------+
                                              |
                                              v
                           +--------------------------------------+
                           |   Temporal Context Engine (State)    |
                           |   Trend, Streaks, Persistence        |
                           +------------------+-------------------+
                                              |
                                              v
                           +--------------------------------------+
                           | Deterministic Rule Engine (R1 - R6)  |
                           +------------------+-------------------+
                                              |
                      +-----------------------+-----------------------+
                      | High Certainty                                | Ambiguous
                      v                                               v
             +-----------------+                             +-----------------+
             |   Fast Action   |                             |  Policy Agent   |
             |   (Rule Path)   |                             | (LLM / Heuristic|
             +--------+--------+                             +--------+--------+
                      |                                               |
                      +-----------------------+-----------------------+
                                              |
                                              v
                           +--------------------------------------+
                           | Final Security Action & Log Artifact |
                           |       ALLOW  |  FLAG  |  BLOCK       |
                           +--------------------------------------+
```

---

## 2. Component Specifications

### 2.1 `pcap_reader.py`
- Ingests raw `.pcap` files using streaming memory buffers.
- Parses Ethernet, IPv4, TCP, and UDP packet headers.
- Reassembles bidirectional packets into 5-tuple conversations:
  `Key = (min(src_ip, dst_ip), max(src_ip, dst_ip), min(src_port, dst_port), max(src_port, dst_port), protocol)`
- Tracks inter-arrival packet timestamps, byte volumes, and TCP flags.

### 2.2 `cicflowmeter_runner.py`
- Orchestrates conversion of PCAP packets into standard 79-feature bidirectional flow records.
- Computes flow durations, forward/backward packet statistics, inter-arrival times (IAT), and TCP window parameters.
- Validates data types, handles infinities/NaNs, and persists the flow DataFrame as an intermediate artifact.

### 2.3 `packet_feature_extractor.py`
- Computes the 27 dimensionless packet-header ratios and flag densities expected by `Packet 1D-CNN`.
- Applies `cnn_scaler.pkl` and shapes tensors to `(N, 27, 1)`.
- Evaluates individual flow records and aggregates packet predictions into window-level evidence via 95th-percentile quantile pooling.

### 2.4 `flow_preprocessor.py`
- Extracts the 17 top features (`Tot Bwd Pkts`, `Flow Duration`, `IAT`, `burstiness`, `total_packets`, etc.) required by `Flow XGBoost`.
- Invocates calibrated multi-class prediction and calculates $P(\text{attack}) = 1 - P(\text{benign})$.

### 2.5 `window_aggregator.py`
- Segments flows into discrete 30-second temporal windows based on timestamps.
- Aggregates flows into the 268 temporal statistical features required by `Time-Window XGBoost`.

### 2.6 `session_sequence_builder.py`
- Assembles chronological flow sequences of length $T = 30$ with 27 features each.
- Invocates PyTorch `SessionCNNLSTM` to detect multi-flow attack patterns.

### 2.7 `evidence_aligner.py`
- Maps predictions from all four models to their respective 30-second window identifier (`W00000_YYYY-MM-DD HH:MM:SS`).
- Emits structured `IntegratedEvidence` records containing individual and fused probabilities.

### 2.8 `validation_runner.py`
- End-to-end command-line orchestrator supporting single PCAP, directory of PCAPs, independent model runs, and integrated agentic decisions.
- Generates reproducible performance reports, decision audit logs, and prediction matrices.

