# Session-Level CNN-LSTM Anomaly Detection Model (Submission Package)

This folder contains the complete, verified **Session-Level PyTorch CNN-LSTM Anomaly Detection Model** of the Capstone Phase-2 dissertation.

---

## 1. Academic Foundation & Citations

1. **Hwang et al. (2020)** — *"Packet-Header and Session-Sequence Deep Learning for Network Intrusion Detection"*, IEEE Access, Vol. 8, pp. 195744–195758, 2020.
2. **Du et al. (2017)** — *"DeepLog: Anomaly Detection and System Log Analysis Using Deep Sequence Models"*, Proceedings of the ACM Conference on Computer and Communications Security (CCS), 2017.
3. **Radford et al. (2018)** — *"Network Traffic Anomaly Detection Using Recurrent Neural Networks"*, IEEE Conference on Communications and Network Security (CNS), 2018.

---

## 2. Performance Summary (Unseen 30% Holdout Test Set)

$$\text{Evaluated on } \mathbf{7,370} \text{ Unseen Multi-Day Holdout Sessions Across CSE-CIC-IDS2018}$$

* **Overall Test Accuracy:** $\mathbf{98.01\%}$
* **Weighted $F_1$-Score:** $\mathbf{98.59\%}$
* **Macro $F_1$-Score:** $\mathbf{83.68\%}$
* **SQL Injection Catch Rate:** $\mathbf{100.00\% \text{ Recall}}$
* **Brute Force - XSS Catch Rate:** $\mathbf{80.00\% \text{ Recall}} \quad (F_1 = 0.7273)$
* **Brute Force - Web Catch Rate:** $\mathbf{78.57\% \text{ Recall}}$
* **DDoS HOIC & LOIC-HTTP:** $\mathbf{100.00\% \text{ Recall}} \quad (F_1 = 1.0000)$
* **Botnet Traffic:** $\mathbf{100.00\% \text{ Recall}} \quad (F_1 = 0.9871)$
* **DoS GoldenEye & SlowHTTPTest:** $\mathbf{>99.2\% \text{ Recall}} \quad (F_1 > 0.9950)$
* **Normal Benign Traffic:** $\mathbf{99.96\% \text{ Precision}} \quad (F_1 = 0.9755)$

---

## 3. Directory Contents

| Filename | Type | Description |
| :--- | :--- | :--- |
| **`session_cnn_lstm.pth`** | PyTorch State | Trained weights of the 1D-CNN + LSTM sequence model. |
| **`session_scaler.pkl`** | Preprocessor | `StandardScaler` fitted on session flow tensors. |
| **`session_labels.pkl`** | Label Encoder | `LabelEncoder` for canonical attack taxonomy. |
| **`session_features.pkl`** | Feature Schema | 27 temporal & flow metric feature names. |
| **`session_config.pkl`** | Architecture | Hyperparameters & layer dimensions. |
| **`train_session_cnn_lstm.py`** | Source Code | Complete PyTorch training pipeline. |
| **`evaluate_session_model.py`** | Entry Point | 1-Click standalone evaluation script. |
| **`README.md`** | Documentation | Academic reference guide, LaTeX math, and instructions. |

---

## 4. How to Run Evaluation

```bash
python evaluate_session_model.py "F:\Day 2 - Feb 15\csv\02-15-2018.csv"
```

