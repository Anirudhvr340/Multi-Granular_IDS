import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

os.makedirs(INPUT_DIR, exist_ok=True)


def main():
    print("🔹 Loading window data...")
    X = np.load(os.path.join(INPUT_DIR, "window_features.npy"))
    y = np.load(os.path.join(INPUT_DIR, "window_multiclass_labels.npy"))
    times = np.load(os.path.join(INPUT_DIR, "window_times.npy"))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    all_indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        all_indices,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    total = len(y)
    split_index = len(train_indices)
    print(f"🔹 Total windows: {total}")
    print(f"🔹 Training windows: {split_index}")
    print(f"🔹 Test windows: {total - split_index}")

    print(f"Random state: {RANDOM_STATE}")
    print("Split strategy: stratified random windows")
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    times_train, times_test = times[train_indices], times[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n🔹 Train class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    print("\n🔹 Test class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    np.save(os.path.join(INPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(INPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(INPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(INPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(INPUT_DIR, "window_times_train.npy"), times_train)
    np.save(os.path.join(INPUT_DIR, "window_times_test.npy"), times_test)
    np.save(os.path.join(INPUT_DIR, "train_indices.npy"), train_indices)
    np.save(os.path.join(INPUT_DIR, "test_indices.npy"), test_indices)

    from joblib import dump
    dump(scaler, os.path.join(INPUT_DIR, "time_window_scaler.joblib"))

    print("\n✅ Stratified split completed successfully!")


if __name__ == '__main__':
    main()
