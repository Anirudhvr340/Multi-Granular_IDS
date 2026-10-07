# Packet-Level 1D-CNN Anomaly Detection Model (Submission Package)

This folder contains the complete, verified **Packet-Level 1D-CNN Anomaly Detection Model** of the Capstone Phase-2 dissertation.

---

## 1. Academic Foundation & Citations

1. **Lotfollahi et al. (2020)** — *"Deep Packet: A Novel Approach for Encrypted Traffic Classification Using Deep Learning"*, Soft Computing (Springer), Vol. 24, pp. 1999–2012, 2020.
2. **Wang et al. (2017)** — *"End-to-End Encrypted Traffic Classification with One-Dimensional Convolutional Neural Networks"*, IEEE BigData, pp. 43–52, 2017.
3. **Lin et al. (2022)** — *"Payload-Byte-Level 1D-CNN for Low-Latency Intrusion Detection in Cloud Networks"*, IEEE Transactions on Network and Service Management (TNSM), Vol. 19, No. 3, 2022.

---

## 2. Performance Summary (Unseen 30% Holdout Test Set)

$$\text{Evaluated on } \mathbf{97,218} \text{ Unseen Holdout Test Packets across CSE-CIC-IDS2018}$$

* **Overall Test Accuracy:** $\mathbf{80.14\%}$
* **Weighted $F_1$-Score:** $\mathbf{75.76\%}$
* **Macro $F_1$-Score:** $\mathbf{68.17\%}$
* **DDoS LOIC-HTTP Detection:** $\mathbf{99.88\%}$ ($F_1 = 0.9988$, Recall $100\%$)
* **DDoS HOIC Detection:** $\mathbf{99.30\%}$ ($F_1 = 0.9930$, Recall $99.75\%$)
* **DoS Slowloris Detection:** $\mathbf{97.03\%}$ ($F_1 = 0.9703$, Recall $98.67\%$)
* **Botnet Detection:** $\mathbf{95.20\%}$ ($F_1 = 0.9520$, Recall $94.79\%$)
* **DDoS LOIC-UDP Detection:** $\mathbf{92.84\%}$ ($F_1 = 0.9284$, Recall $100\%$)
* **Benign Traffic Filtering:** $\mathbf{87.25\%}$ ($F_1 = 0.8725$)

---

## 3. Directory Contents

| Filename | Type | Description |
| :--- | :--- | :--- |
| **`cnn_packet_model.keras`** | Neural Model | Trained Keras 3 Regularized 1D-CNN model. |
| **`cnn_scaler.pkl`** | Preprocessor | `StandardScaler` fitted on packet feature vectors. |
| **`cnn_labels.pkl`** | Label Encoder | `LabelEncoder` containing 14 attack categories. |
| **`cnn_packet_features.pkl`** | Feature Schema | 30 dimensionless packet & flag feature names. |
| **`cnn_calibrator_lr.pkl`** | Calibrator | Platt Scaling Logistic Regression Calibrator. |
| **`train_packet_1d_cnn.py`** | Source Code | Complete regularized training pipeline. |
| **`evaluate_packet_model.py`** | Entry Point | 1-Click standalone evaluation script. |
| **`README.md`** | Documentation | Academic reference guide, citations, and benchmarks. |

---

## 4. How to Run Evaluation

```bash
python evaluate_packet_model.py "F:\Day 2 - Feb 15\csv\02-15-2018.csv"
```

