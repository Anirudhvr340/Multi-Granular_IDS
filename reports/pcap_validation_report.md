# Unified PCAP Validation Report

**Dataset Source:** `UCAP172.31.69.25`  
**Evaluation Mode:** `ALL`  
**Run Timestamp:** `2026-10-13 01:30:28`  
**Execution Time:** `5.17 seconds`  

---

## 1. Execution Summary
- **Total Packets Reassembled into Flows:** 2,012 flows
- **30-Second Windows Scored:** 10 windows
- **Attack Windows Present:** 0
- **Attack Windows Detected (BLOCK + FLAG):** 0 (100.00% recall)

## 2. Action Distribution
- **ALLOW:** 0 (0.0%)
- **FLAG:** 9 (90.0%)
- **BLOCK:** 1 (10.0%)

## 3. Model Branch Participation
- **Flow XGBoost:** 10 / 10 successful invocations
- **Time-Window XGBoost:** 10 / 10 successful invocations
- **Packet 1D-CNN:** 10 / 10 successful invocations
- **Session CNN-LSTM:** 10 / 10 successful invocations

All raw logs and prediction files are saved in `D:\Multi-Granular IDS\Multi-Granular_IDS\reports`.
