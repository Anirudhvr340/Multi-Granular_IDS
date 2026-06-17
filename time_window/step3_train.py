import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# =============================
# MODEL
# =============================
class TCN_BiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Dropout(0.3)
        )

        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.fc(x)


# =============================
# MAIN
# =============================
def main():
    print("🔹 Loading data...")

    X = np.load('processed/ssae_features_train.npy')
    y = np.load('processed/y_train.npy')

    # =============================
    # ALIGN LENGTH
    # =============================
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # =============================
    # RESHAPE (for TCN)
    # =============================
    X = X[:, np.newaxis, :]

    # =============================
    # RELABEL
    # =============================
    unique_classes = np.unique(y)
    label_map = {old: i for i, old in enumerate(unique_classes)}

    y = np.array([label_map[v] for v in y])
    num_classes = len(unique_classes)

    print("Classes:", np.unique(y, return_counts=True))

    # =============================
    # TRAIN / VAL SPLIT
    # =============================
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================
    # DATALOADERS (NO SAMPLER)
    # =============================
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=256
    )

    # =============================
    # MODEL
    # =============================
    model = TCN_BiLSTM(
        input_dim=X.shape[2],
        num_classes=num_classes
    ).to(device)

    # =============================
    # LOSS + OPTIMIZER
    # =============================
    criterion = nn.CrossEntropyLoss()   # ✅ SIMPLE (VERY IMPORTANT)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("🔹 Training started...")

    best_acc = 0

    for epoch in range(30):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # =============================
        # VALIDATION
        # =============================
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                out = model(xb)
                pred = out.argmax(1)

                correct += (pred == yb).sum().item()
                total += yb.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | Val Acc {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'processed/tcn_bilstm.pth')

    print("\n✅ Training complete!")
    print("Best accuracy:", best_acc)


if __name__ == "__main__":
    main()