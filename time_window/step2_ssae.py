import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os

# =============================
# SSAE MODEL
# =============================
class SSAE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


# =============================
# 🔥 FIXED FLATTEN (CRITICAL)
# =============================
def flatten_windows(X, fixed_size=200):
    fixed = []

    for window in X:
        if len(window) >= fixed_size:
            fixed.append(window[:fixed_size])
        else:
            pad = np.zeros((fixed_size - len(window), window.shape[1]))
            fixed.append(np.vstack([window, pad]))

    fixed = np.array(fixed, dtype=np.float32)
    return fixed.reshape((fixed.shape[0], -1))


# =============================
# TRAIN SSAE
# =============================
def train_ssae(X_train_flat, hidden_dims=[1024, 256], epochs=30, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_train_flat.shape[1]
    model = SSAE(input_dim, hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train_flat, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🔹 Training SSAE...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model


# =============================
# MAIN
# =============================
def main():
    print("🔹 Loading window data...")

    # 🔥 FIX: allow_pickle for variable-length windows
    X_train = np.load('processed/X_train.npy', allow_pickle=True)
    X_test = np.load('processed/X_test.npy', allow_pickle=True)

    print(f"Train samples: {len(X_train)}")

    # =============================
    # 🔥 FIX: CONVERT VARIABLE → FIXED
    # =============================
    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)

    print("After flatten:", X_train_flat.shape)

    # =============================
    # SCALING
    # =============================
    print("🔹 Scaling features...")

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    os.makedirs('processed', exist_ok=True)
    joblib.dump(scaler, 'processed/ssae_scaler.pkl')

    # =============================
    # TRAIN SSAE
    # =============================
    ssae = train_ssae(X_train_flat)

    # =============================
    # FEATURE EXTRACTION
    # =============================
    print("🔹 Extracting SSAE features...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssae.eval()

    with torch.no_grad():
        X_train_ssae = ssae.encode(
            torch.tensor(X_train_flat, dtype=torch.float32).to(device)
        ).cpu().numpy()

        X_test_ssae = ssae.encode(
            torch.tensor(X_test_flat, dtype=torch.float32).to(device)
        ).cpu().numpy()

    print("SSAE feature shape:", X_train_ssae.shape)

    # =============================
    # SAVE
    # =============================
    np.save('processed/ssae_features_train.npy', X_train_ssae)
    np.save('processed/ssae_features_test.npy', X_test_ssae)

    torch.save(ssae.state_dict(), 'processed/ssae_model.pth')

    print("✅ SSAE completed successfully!")


if __name__ == '__main__':
    main()