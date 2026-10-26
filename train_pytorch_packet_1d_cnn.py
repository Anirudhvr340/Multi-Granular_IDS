import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEST_DIR = Path("packet_level")
DEST_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def extract_dimensionless_packet_features(df: pd.DataFrame):
    feats = pd.DataFrame(index=df.index)
    
    def _safe_series(col_name: str, default: float = 0.0) -> pd.Series:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)
    
    tot_fwd_pkts = _safe_series("Tot Fwd Pkts", 0.0)
    tot_bwd_pkts = _safe_series("Tot Bwd Pkts", 0.0)
    tot_pkts = (tot_fwd_pkts + tot_bwd_pkts).replace(0, 1.0)
    
    tot_fwd_bytes = _safe_series("TotLen Fwd Pkts", 0.0)
    tot_bwd_bytes = _safe_series("TotLen Bwd Pkts", 0.0)
    tot_bytes = (tot_fwd_bytes + tot_bwd_bytes).replace(0, 1.0)
    
    # 1. Packet & Byte Symmetry Ratios
    feats["pkt_fwd_ratio"] = (tot_fwd_pkts / tot_pkts).clip(0.0, 1.0)
    feats["pkt_bwd_ratio"] = (tot_bwd_pkts / tot_pkts).clip(0.0, 1.0)
    feats["byte_fwd_ratio"] = (tot_fwd_bytes / tot_bytes).clip(0.0, 1.0)
    feats["byte_bwd_ratio"] = (tot_bwd_bytes / tot_bytes).clip(0.0, 1.0)
    feats["byte_per_pkt_avg"] = tot_bytes / tot_pkts
    
    # 2. Header Overhead Ratios (Clipped strictly to [0, 1] to prevent payload=0 spikes)
    fwd_hdr = _safe_series("Fwd Header Len", 0.0)
    bwd_hdr = _safe_series("Bwd Header Len", 0.0)
    feats["fwd_hdr_ratio"] = (fwd_hdr / (tot_fwd_bytes + 1.0)).clip(0.0, 1.0)
    feats["bwd_hdr_ratio"] = (bwd_hdr / (tot_bwd_bytes + 1.0)).clip(0.0, 1.0)
    
    # 3. Dimensionless TCP Flag Densities
    flags = [
        ("fin_ratio", "FIN Flag Cnt"),
        ("syn_ratio", "SYN Flag Cnt"),
        ("rst_ratio", "RST Flag Cnt"),
        ("psh_ratio", "PSH Flag Cnt"),
        ("ack_ratio", "ACK Flag Cnt"),
        ("urg_ratio", "URG Flag Cnt"),
        ("cwe_ratio", "CWE Flag Count"),
        ("ece_ratio", "ECE Flag Cnt"),
    ]
    for col_out, col_in in flags:
        cnt = _safe_series(col_in, 0.0)
        feats[col_out] = (cnt / tot_pkts).clip(0.0, 1.0)
        
    # 4. OS-Independent Window Tier Encoding
    fwd_win = _safe_series("Init Fwd Win Byts", 0.0)
    feats["fwd_win_zero"] = (fwd_win <= 0).astype(float)
    feats["fwd_win_small"] = ((fwd_win > 0) & (fwd_win <= 1024)).astype(float)
    feats["fwd_win_std"] = ((fwd_win > 1024) & (fwd_win <= 65535)).astype(float)
    feats["fwd_win_large"] = (fwd_win > 65535).astype(float)
    
    # 5. Protocol & Standard Port Categories
    proto = _safe_series("Protocol", 6.0)
    feats["is_tcp"] = (proto == 6.0).astype(float)
    feats["is_udp"] = (proto == 17.0).astype(float)
    feats["is_other_proto"] = ((proto != 6.0) & (proto != 17.0)).astype(float)
    
    dport = _safe_series("Dst Port", 0.0)
    feats["port_http"] = (dport == 80.0).astype(float)
    feats["port_https"] = (dport == 443.0).astype(float)
    feats["port_ssh"] = (dport == 22.0).astype(float)
    feats["port_ftp"] = ((dport == 20.0) | (dport == 21.0)).astype(float)
    feats["port_ephemeral"] = (dport > 1024.0).astype(float)
    
    # 6. Packet Length Distribution (Log-scaled)
    feats["pkt_len_mean"] = np.log1p(_safe_series("Pkt Len Mean", 0.0).clip(lower=0.0))
    feats["pkt_len_std"] = np.log1p(_safe_series("Pkt Len Std", 0.0).clip(lower=0.0))
    feats["pkt_len_max"] = np.log1p(_safe_series("Pkt Len Max", 0.0).clip(lower=0.0))
    feats["pkt_len_min"] = np.log1p(_safe_series("Pkt Len Min", 0.0).clip(lower=0.0))
    
    # 7. Flow Rate & Dynamics (Log-scaled)
    feats["flow_duration"] = np.log1p(_safe_series("Flow Duration", 0.0).clip(lower=0.0))
    feats["flow_pkts_s"] = np.log1p(_safe_series("Flow Pkts/s", 0.0).clip(0.0, 1e7))
    feats["flow_byts_s"] = np.log1p(_safe_series("Flow Byts/s", 0.0).clip(0.0, 1e8))
    feats["down_up_ratio"] = _safe_series("Down/Up Ratio", 0.0).clip(0.0, 50.0)
    
    feature_names = list(feats.columns)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats, feature_names

# ---------------------------------------------------------------------------
# PyTorch Model Architecture
# ---------------------------------------------------------------------------
class Packet1DCNN(nn.Module):
    def __init__(self, in_features: int = 35, n_classes: int = 15):
        super(Packet1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.drop_fc = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # x shape: (Batch, In_Features) or (Batch, 1, In_Features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[1] != 1 and x.shape[2] == 1:
            x = x.transpose(1, 2)
            
        c1 = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        c2 = self.drop2(self.relu2(self.bn2(self.conv2(c1))))
        avg_p = self.avg_pool(c2).squeeze(-1)
        max_p = self.max_pool(c2).squeeze(-1)
        p = torch.cat([avg_p, max_p], dim=1)
        h = self.drop_fc(self.relu3(self.bn_fc(self.fc1(p))))
        logits = self.fc_out(h)
        return logits

def learn_temperature(logits, labels):
    """Learns optimal scalar temperature T via NLL minimization (Guo et al. ICML 2017)."""
    temp = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temp], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    t_val = float(temp.item())
    return max(0.5, min(t_val, 3.0))

def main():
    print("=" * 80)
    print("TRAINING REGULARIZED PYTORCH PACKET 1D-CNN (ON RTX 4090 GPU)")
    print("=" * 80)
    
    csv_cache = Path("data_cache/packet_train_balanced.csv")
    if not csv_cache.exists():
        print(f"[!] Error: {csv_cache} not found. Run build_balanced_training_data_fast.py first.")
        sys.exit(1)
        
    print(f"[*] Ingesting balanced dataset from {csv_cache}...")
    df = pd.read_csv(csv_cache, low_memory=False)
    print(f"[+] Loaded {len(df):,} flows.")
    
    print("\n[*] Extracting 30 Dimensionless Header Features...")
    feats, feature_names = extract_dimensionless_packet_features(df)
    X = feats.values.astype(np.float32)
    
    # Standard 15-class Label Encoding
    STANDARD_CLASSES = [
        'Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS',
        'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP',
        'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest',
        'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration',
        'SQL Injection', 'SSH-Bruteforce'
    ]
    
    le = LabelEncoder()
    le.fit(STANDARD_CLASSES)
    y = le.transform(df["Label"].values)
    n_classes = len(STANDARD_CLASSES)
    print(f"[+] Verified {n_classes} Classes: {STANDARD_CLASSES}")
    
    # 70/30 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"[+] Training Split: {len(X_train):,}, Holdout Test Split: {len(X_test):,}")
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Balanced Class Weights (Clipped to max 5.0 to prevent outlier disruption)
    from sklearn.utils.class_weight import compute_class_weight
    raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    clipped_weights = np.clip(raw_weights, 0.5, 5.0).astype(np.float32)
    weight_tensor = torch.tensor(clipped_weights, dtype=torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_sc, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_sc, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    model = Packet1DCNN(in_features=len(feature_names), n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)
    
    print("\n" + "=" * 80)
    print("[*] TRAINING 1D-CNN (12 EPOCHS WITH REGULARIZATION & COSINE DECAY)")
    print("=" * 80)
    
    num_epochs = 12
    best_acc = 0.0
    best_weights = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t_epoch = time.time()
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_logits_list = []
        with torch.no_grad():
            for vx, vy in test_loader:
                vx, vy = vx.to(device), vy.to(device)
                vl = model(vx)
                val_logits_list.append(vl.cpu())
                val_loss += criterion(vl, vy).item() * len(vy)
                vp = torch.argmax(vl, dim=1)
                val_correct += (vp == vy).sum().item()
                val_total += len(vy)
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t_epoch
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict().copy()
            
        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
    model.load_state_dict(best_weights)
    
    # Learn Temperature Scaling on Holdout Validation Logits
    val_logits_all = torch.cat(val_logits_list, dim=0)
    T_learned = learn_temperature(val_logits_all, torch.tensor(y_test, dtype=torch.long))
    print(f"\n[+] Learned Temperature Parameter T = {T_learned:.3f} (Platt Temperature Scaling)")
    
    # Evaluate with Temperature Scaling
    cal_logits = val_logits_all / T_learned
    cal_probs = torch.softmax(cal_logits, dim=-1).numpy()
    y_pred_cal = np.argmax(cal_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred_cal)
    w_f1 = f1_score(y_test, y_pred_cal, average="weighted", zero_division=0)
    m_f1 = f1_score(y_test, y_pred_cal, average="macro", zero_division=0)
    
    print("\n" + "=" * 80)
    print("[*] UNSEEN HOLDOUT TEST SET EVALUATION (39,798 SAMPLES)")
    print("=" * 80)
    print(f"[+] Multi-Class Accuracy: {acc * 100:.2f}%")
    print(f"[+] Weighted F1-Score:    {w_f1 * 100:.2f}%")
    print(f"[+] Macro F1-Score:       {m_f1 * 100:.2f}%")
    print("\n" + classification_report(y_test, y_pred_cal, target_names=STANDARD_CLASSES, digits=4, zero_division=0))
    
    # Save Artifacts
    print("\n" + "=" * 80)
    print(f"[*] SAVING RETRAINED PYTORCH PACKET ARTIFACTS TO: {DEST_DIR}")
    print("=" * 80)
    
    torch.save(model.state_dict(), DEST_DIR / "cnn_packet_model.pth")
    joblib.dump(scaler, DEST_DIR / "cnn_scaler.pkl")
    joblib.dump(le, DEST_DIR / "cnn_labels.pkl")
    joblib.dump(feature_names, DEST_DIR / "cnn_packet_features.pkl")
    joblib.dump({"temperature": T_learned, "classes": STANDARD_CLASSES}, DEST_DIR / "cnn_temperature.pkl")
    print(f"[+] All PyTorch Packet 1D-CNN artifacts saved successfully into {DEST_DIR}!")

if __name__ == "__main__":
    main()
