import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
MODEL_PATH = os.path.join(INPUT_DIR, "temporal_lstm.pth")
SEQUENCE_LENGTH = 12
BATCH_SIZE = 512
EPOCHS = 30
PATIENCE = 3


class WindowSequenceDataset(Dataset):
    def __init__(self, features, labels, end_indices, sequence_length):
        self.features = features
        self.labels = labels
        self.end_indices = end_indices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, index):
        end = self.end_indices[index]
        start = end - self.sequence_length + 1
        sequence = self.features[start : end + 1]
        if os.environ.get("SHUFFLE_SEQUENCE") == "1":
            sequence = sequence[np.random.default_rng(end).permutation(len(sequence))]
        return (
            torch.from_numpy(sequence),
            torch.tensor(self.labels[end], dtype=torch.long),
        )


class TemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.classifier(sequence_output[:, -1, :])


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    features = np.load(os.path.join(INPUT_DIR, "window_features.npy")).astype(np.float32)
    labels = np.load(os.path.join(INPUT_DIR, "window_multiclass_labels.npy"))
    times = np.load(os.path.join(INPUT_DIR, "window_times.npy"))
    class_names = np.load(
        os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True
    ).astype(str)

    order = np.argsort(times)
    features = np.nan_to_num(features[order], nan=0.0, posinf=0.0, neginf=0.0)
    labels = labels[order]
    times = times[order]

    # Add lag features matching step1_split
    from step1_split import add_lag_features
    feature_names = np.load(os.path.join(INPUT_DIR, "window_feature_names.npy"), allow_pickle=True).astype(str)
    # Exclude lag feature names if already in file
    base_names = [n for n in feature_names if not n.startswith(("previous_", "delta_"))]
    features_lagged, _ = add_lag_features(features, times, np.array(base_names, dtype=object))

    train_end_indices = np.load(os.path.join(INPUT_DIR, "train_indices.npy"))
    test_end_indices = np.load(os.path.join(INPUT_DIR, "test_indices.npy"))
    scaler = joblib.load(os.path.join(INPUT_DIR, "time_window_scaler.joblib"))
    features = scaler.transform(features_lagged).astype(np.float32)
    days = times.astype("datetime64[D]")
    valid_sequence_endpoints = np.array([
        end >= SEQUENCE_LENGTH - 1 and len(set(days[end - SEQUENCE_LENGTH + 1 : end + 1])) == 1
        for end in range(len(labels))
    ])
    train_end_indices = train_end_indices[valid_sequence_endpoints[train_end_indices]]
    test_end_indices = test_end_indices[valid_sequence_endpoints[test_end_indices]]
    train_dataset = WindowSequenceDataset(
        features, labels, train_end_indices, SEQUENCE_LENGTH
    )
    test_dataset = WindowSequenceDataset(
        features, labels, test_end_indices, SEQUENCE_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts = np.bincount(labels[train_end_indices], minlength=len(class_names)).astype(np.float32)
    weights = np.ones(len(class_names), dtype=np.float32)
    present = counts > 0
    weights[present] = np.clip(len(train_end_indices) / (len(class_names) * counts[present]), 0.1, 10.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    val_criterion = nn.CrossEntropyLoss()
    model = TemporalLSTM(features.shape[1], 96, len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    print(f"Temporal order: {times[0]} to {times[-1]}")
    print("Split: day-based temporal split from step1")
    print(f"Lookback: {SEQUENCE_LENGTH} windows")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    print(f"Device: {device}")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
        train_loss = total_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                loss = val_criterion(model(batch_features), batch_labels)
                val_loss += loss.item() * len(batch_labels)
        val_loss /= max(len(test_dataset), 1)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            output = model(batch_features.to(device))
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            actual.extend(batch_labels.numpy())

    predictions = np.asarray(predictions)
    actual = np.asarray(actual)
    report = classification_report(
        actual,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"Accuracy: {accuracy_score(actual, predictions):.4f}")
    print(f"Macro F1: {f1_score(actual, predictions, average='macro', zero_division=0):.4f}")
    print(report)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "temporal_lstm_classification_report.txt"), "w") as report_file:
        report_file.write(
            f"Accuracy: {accuracy_score(actual, predictions):.4f}\n"
            f"Macro F1: {f1_score(actual, predictions, average='macro', zero_division=0):.4f}\n\n"
            + report
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": features.shape[1],
            "num_classes": len(class_names),
            "sequence_length": SEQUENCE_LENGTH,
        },
        MODEL_PATH,
    )
    print(f"Temporal model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
