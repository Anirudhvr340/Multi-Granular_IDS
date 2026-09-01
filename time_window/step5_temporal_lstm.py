import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
MODEL_PATH = os.path.join(INPUT_DIR, "temporal_tcn.pth")
SEQUENCE_LENGTH = 60
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu2(out + res)


class TemporalTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_levels=4, kernel_size=3, dropout=0.3):
        super().__init__()
        # input shape: (batch, seq_len, features) -> conv1d expects (batch, channels, seq_len)
        layers = []
        in_channels = input_size
        for i in range(num_levels):
            out_channels = hidden_size
            dilation = 2 ** i
            layers.append(TemporalResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # -> (batch, features, seq_len)
        out = self.network(x)  # -> (batch, hidden, seq_len)
        out = self.pool(out).squeeze(-1)  # -> (batch, hidden)
        # LayerNorm expects (batch, features)
        return self.classifier(out)


class TemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.classifier(sequence_output[:, -1, :])


# --- Focal loss implementation ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        # input: (batch, C) raw logits
        # target: (batch,) long
        logpt = -nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_receptive_field(kernel_size, num_levels):
    # For this TCN design with two convs per level and dilation doubling each level,
    # receptive field = 1 + 2 * sum_{i=0..L-1} ( (kernel_size-1) * 2^i )
    k = kernel_size - 1
    total = 0
    for i in range(num_levels):
        total += k * (2 ** i)
    return 1 + 2 * total


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

    # Load train/test split indices and compute lag features separately to avoid leakage
    train_end_indices = np.load(os.path.join(INPUT_DIR, "train_indices.npy"))
    test_end_indices = np.load(os.path.join(INPUT_DIR, "test_indices.npy"))

    # Compute lag features independently on each split (prevents future/test leakage)
    features_train_raw = features[train_end_indices]
    features_test_raw = features[test_end_indices]
    times_train = times[train_end_indices]
    times_test = times[test_end_indices]

    features_train_lagged, lag_feature_names = add_lag_features(
        features_train_raw, times_train, np.array(base_names, dtype=object)
    )
    features_test_lagged, _ = add_lag_features(
        features_test_raw, times_test, np.array(base_names, dtype=object)
    )

    # Reassemble full feature matrix with lagged columns (train/test positions filled)
    n_cols = features_train_lagged.shape[1]
    full_features_lagged = np.zeros((len(features), n_cols), dtype=features.dtype)
    full_features_lagged[train_end_indices] = features_train_lagged
    full_features_lagged[test_end_indices] = features_test_lagged

    # Use scaler fitted on training split (saved by step1_split)
    scaler = joblib.load(os.path.join(INPUT_DIR, "time_window_scaler.joblib"))
    features = scaler.transform(full_features_lagged).astype(np.float32)

    # Validate sequence endpoints by ensuring strictly increasing timestamps across the window
    # (allows multi-day sequences but preserves temporal order). Optionally, change to check for
    # contiguous windows by adding a max-gap constraint if needed.
    valid_sequence_endpoints = np.array([
        end >= SEQUENCE_LENGTH - 1
        and np.all(np.diff(times[end - SEQUENCE_LENGTH + 1 : end + 1]) > np.timedelta64(0, 's'))
        for end in range(len(labels))
    ])
    train_end_indices = train_end_indices[valid_sequence_endpoints[train_end_indices]]
    test_end_indices = test_end_indices[valid_sequence_endpoints[test_end_indices]]

    # --- Time-based features (hour_of_day, day_of_week cyclic encodings) ---
    def compute_time_feats(times_arr):
        # times_arr is a numpy array of datetime64
        py = times_arr.astype('datetime64[s]').astype(object)
        hours = np.array([dt.hour for dt in py], dtype=np.float32)
        weekdays = np.array([dt.weekday() for dt in py], dtype=np.float32)
        hour_rad = 2 * np.pi * hours / 24.0
        weekday_rad = 2 * np.pi * weekdays / 7.0
        return np.vstack([
            np.sin(hour_rad),
            np.cos(hour_rad),
            np.sin(weekday_rad),
            np.cos(weekday_rad),
        ]).T

    time_feats_full = compute_time_feats(times)
    # Fit scaler on training time features only to avoid leakage
    time_scaler = StandardScaler()
    if len(train_end_indices) > 0:
        time_scaler.fit(time_feats_full[train_end_indices])
    time_feats_scaled = time_scaler.transform(time_feats_full).astype(np.float32)

    # Concatenate time features to the existing (scaled) features
    features = np.concatenate([features, time_feats_scaled], axis=1)

    train_dataset = WindowSequenceDataset(
        features, labels, train_end_indices, SEQUENCE_LENGTH
    )
    test_dataset = WindowSequenceDataset(
        features, labels, test_end_indices, SEQUENCE_LENGTH
    )

    # Optionally use a WeightedRandomSampler to balance training batches
    use_sampler = os.environ.get('USE_SAMPLER', '1') == '1'
    if use_sampler:
        # per-class sample weight = 1 / class_freq
        class_counts = np.bincount(labels[train_end_indices], minlength=len(class_names)).astype(float)
        class_weights = np.zeros_like(class_counts)
        mask = class_counts > 0
        class_weights[mask] = 1.0 / class_counts[mask]
        sample_weights = class_weights[labels[train_end_indices]]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts = np.bincount(labels[train_end_indices], minlength=len(class_names)).astype(np.float32)
    weights = np.ones(len(class_names), dtype=np.float32)
    present = counts > 0
    weights[present] = np.clip(len(train_end_indices) / (len(class_names) * counts[present]), 0.1, 10.0)

    # Optionally use focal loss
    use_focal = os.environ.get('USE_FOCAL', '1') == '1'

    # Configure optimizer and dropout from env
    lr = float(os.environ.get('LR', '1e-4'))
    weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-5'))
    dropout_val = float(os.environ.get('DROPOUT', '0.3'))

    # Check receptive field for the TCN settings
    tcn_levels = int(os.environ.get('TCN_LEVELS', '5'))
    tcn_kernel = int(os.environ.get('TCN_KERNEL', '3'))
    receptive_field = compute_receptive_field(tcn_kernel, tcn_levels)
    print(f"TCN receptive field (kernel={tcn_kernel}, levels={tcn_levels}): {receptive_field} timesteps")
    if receptive_field < SEQUENCE_LENGTH:
        print(f"[!] Warning: receptive field {receptive_field} < SEQUENCE_LENGTH {SEQUENCE_LENGTH}. Consider increasing num_levels or kernel_size.")

    if use_focal:
        criterion = FocalLoss(gamma=2.0, weight=torch.tensor(weights, device=device))
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    val_criterion = nn.CrossEntropyLoss()

    print(f"Temporal order: {times[0]} to {times[-1]}")
    print("Split: day-based temporal split from step1")
    print(f"Lookback: {SEQUENCE_LENGTH} windows")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    print(f"Device: {device}")

    def train_and_evaluate(model_type: str):
        # Build model
        if model_type == 'lstm':
            model_local = TemporalLSTM(features.shape[1], 96, len(class_names), dropout=dropout_val).to(device)
            report_name = os.path.join(REPORT_DIR, "temporal_lstm_classification_report.txt")
            model_path_local = os.path.join(INPUT_DIR, "temporal_lstm.pth")
        else:
            model_local = TemporalTCN(features.shape[1], 96, len(class_names), num_levels=tcn_levels, kernel_size=tcn_kernel, dropout=dropout_val).to(device)
            report_name = os.path.join(REPORT_DIR, "temporal_tcn_classification_report.txt")
            model_path_local = os.path.join(INPUT_DIR, "temporal_tcn.pth")

        optimizer_local = torch.optim.AdamW(model_local.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_local = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_local, mode="min", patience=2, factor=0.5)

        best_val_score_local = -float("inf")  # maximize macro F1
        best_state_local = None
        epochs_without_improvement_local = 0

        for epoch in range(EPOCHS):
            model_local.train()
            total_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer_local.zero_grad()
                loss = criterion(model_local(batch_features), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_local.parameters(), max_norm=1.0)
                optimizer_local.step()
                total_loss += loss.item() * len(batch_labels)
            train_loss = total_loss / len(train_dataset)

            model_local.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = model_local(batch_features)
                    loss = val_criterion(logits, batch_labels)
                    val_loss += loss.item() * len(batch_labels)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            val_loss /= max(len(test_dataset), 1)
            scheduler_local.step(val_loss)

            val_preds = np.asarray(val_preds)
            val_labels = np.asarray(val_labels)
            val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

            # Print per-epoch summary incl. per-class recall for monitoring
            from sklearn.metrics import recall_score
            per_class_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
            print(f"{model_type.upper()} Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_macro_f1:.4f}  lr={optimizer_local.param_groups[0]['lr']:.6f}")
            # print first few class recalls
            print("Per-class recall (first 6):", per_class_recall[:6])

            # early stopping based on macro F1
            if val_macro_f1 > best_val_score_local:
                best_val_score_local = val_macro_f1
                best_state_local = {k: v.cpu().clone() for k, v in model_local.state_dict().items()}
                epochs_without_improvement_local = 0
            else:
                epochs_without_improvement_local += 1
                if epochs_without_improvement_local >= PATIENCE:
                    print(f"{model_type.upper()} Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
                    break

        if best_state_local is not None:
            model_local.load_state_dict(best_state_local)
            model_local.to(device)

        model_local.eval()
        predictions = []
        actual = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                output = model_local(batch_features.to(device))
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                actual.extend(batch_labels.numpy())

        predictions = np.asarray(predictions)
        actual = np.asarray(actual)
        report = classification_report(actual, predictions, labels=np.arange(len(class_names)), target_names=class_names, digits=4, zero_division=0)
        acc = accuracy_score(actual, predictions)
        macro_f1 = f1_score(actual, predictions, average='macro', zero_division=0)

        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(report_name, 'w') as rf:
            rf.write(f"Accuracy: {acc:.4f}\n")
            rf.write(f"Macro F1: {macro_f1:.4f}\n\n")
            rf.write(report)

        torch.save({
            'model_state_dict': model_local.state_dict(),
            'input_size': features.shape[1],
            'num_classes': len(class_names),
            'sequence_length': SEQUENCE_LENGTH,
        }, model_path_local)
        print(f"Saved {model_type.upper()} model to {model_path_local}")

        return { 'model': model_type, 'accuracy': acc, 'macro_f1': macro_f1, 'report': report }

    # Train and evaluate both models
    results_lstm = train_and_evaluate('lstm')
    results_tcn = train_and_evaluate('tcn')

    # Print comparison
    print('\n=== Model comparison ===')
    print(f"LSTM   - Accuracy: {results_lstm['accuracy']:.4f}, Macro F1: {results_lstm['macro_f1']:.4f}")
    print(f"TCN    - Accuracy: {results_tcn['accuracy']:.4f}, Macro F1: {results_tcn['macro_f1']:.4f}")

    # Write comparison report
    with open(os.path.join(REPORT_DIR, 'temporal_model_comparison.txt'), 'w') as compf:
        compf.write('Model comparison (SEQUENCE_LENGTH=' + str(SEQUENCE_LENGTH) + ')\n')
        compf.write(f"LSTM   - Accuracy: {results_lstm['accuracy']:.4f}, Macro F1: {results_lstm['macro_f1']:.4f}\n")
        compf.write(f"TCN    - Accuracy: {results_tcn['accuracy']:.4f}, Macro F1: {results_tcn['macro_f1']:.4f}\n")



if __name__ == "__main__":
    main()
