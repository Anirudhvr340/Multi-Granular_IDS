import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================
# MODEL (same as training)
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
    print("🔹 Loading test data...")

    X_test = np.load('processed/ssae_features_test.npy')
    y_test = np.load('processed/y_test.npy')
    y_train = np.load('processed/y_train.npy')   # 🔥 IMPORTANT

    # Add time dimension
    X_test = X_test[:, np.newaxis, :]

    # =============================
    # USE TRAIN CLASSES (FIX)
    # =============================
    unique_classes = np.unique(y_train)
    label_map = {old: i for i, old in enumerate(unique_classes)}

    # Map test labels
    y_test = np.array([label_map.get(v, -1) for v in y_test])

    # Remove unknown labels
    mask = y_test != -1
    X_test = X_test[mask]
    y_test = y_test[mask]

    num_classes = len(unique_classes)

    print("🔹 Total classes (from training):", num_classes)
    print("🔹 Test samples:", len(y_test))

    # =============================
    # LOAD MODEL
    # =============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCN_BiLSTM(
        input_dim=X_test.shape[2],
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(torch.load('processed/tcn_bilstm.pth', map_location=device))
    model.eval()

    # =============================
    # PREDICTION
    # =============================
    print("🔹 Running inference...")

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    # =============================
    # REPORT
    # =============================
    print("\n🔹 Classification Report:\n")

    report = classification_report(y_test, preds, digits=4, zero_division=0)
    print(report)

    # Save report
    os.makedirs("reports", exist_ok=True)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # =============================
    # CONFUSION MATRIX
    # =============================
    cm = confusion_matrix(y_test, preds)

    np.save("reports/confusion_matrix.npy", cm)
    np.savetxt("reports/confusion_matrix.csv", cm, delimiter=",")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    print("\n✅ Evaluation complete! Results saved in 'reports/'")


if __name__ == "__main__":
    main()