import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

os.makedirs(INPUT_DIR, exist_ok=True)


def add_lag_features(X, times, feature_names):
    """Compute previous_* and delta_* lag features within each day.

    Lag features are computed AFTER splitting to prevent information from
    future/test windows leaking into training lag values.
    """
    lag_columns = ["total_flows", "total_packets", "total_bytes",
                   "flow_rate", "packet_rate", "byte_rate"]

    # Build a name->index map for the base features
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    lag_arrays = []
    lag_names = []
    days = times.astype("datetime64[D]")

    for col in lag_columns:
        if col not in name_to_idx:
            continue
        idx = name_to_idx[col]
        values = X[:, idx].copy()
        previous = np.empty_like(values)
        delta = np.empty_like(values)

        for day in np.unique(days):
            mask = days == day
            day_values = values[mask]
            prev = np.empty_like(day_values)
            prev[0] = day_values[0]  # No previous window -> use own value
            prev[1:] = day_values[:-1]
            previous[mask] = prev
            delta[mask] = day_values - prev

        lag_arrays.append(previous)
        lag_names.append(f"previous_{col}")
        lag_arrays.append(delta)
        lag_names.append(f"delta_{col}")

    if lag_arrays:
        X_with_lags = np.column_stack([X] + [a.reshape(-1, 1) for a in lag_arrays])
        all_names = np.concatenate([feature_names, np.array(lag_names, dtype=object)])
    else:
        X_with_lags = X
        all_names = feature_names

    return X_with_lags, all_names


def main():
    print("[+] Loading window data...")
    X = np.load(os.path.join(INPUT_DIR, "window_features.npy"))
    y = np.load(os.path.join(INPUT_DIR, "window_multiclass_labels.npy"))
    times = np.load(os.path.join(INPUT_DIR, "window_times.npy"))
    feature_names = np.load(
        os.path.join(INPUT_DIR, "window_feature_names.npy"), allow_pickle=True
    ).astype(str)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ----------------------------------------------------------------
    # DAY-BASED TEMPORAL SPLIT
    # Consecutive time windows share temporal autocorrelation. A random
    # split lets the model memorize neighboring-window patterns instead
    # of learning generalizable attack features. We split by calendar
    # day so the model has never seen ANY window from the test period.
    # ----------------------------------------------------------------
    days = times.astype("datetime64[D]")
    unique_days = np.unique(days)
    unique_days.sort()

    split_point = max(1, int(len(unique_days) * TRAIN_RATIO))
    train_days = set(unique_days[:split_point].astype(str))
    test_days = set(unique_days[split_point:].astype(str))

    day_strings = days.astype(str)
    train_mask = np.isin(day_strings, list(train_days))
    test_mask = np.isin(day_strings, list(test_days))

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # If test days have missing classes or no windows, fallback gracefully
    if len(test_indices) == 0:
        print("[!] Only one day available. Splitting 80/20 chronologically...")
        split_idx = int(len(y) * TRAIN_RATIO)
        train_indices = np.arange(split_idx)
        test_indices = np.arange(split_idx, len(y))

    total = len(y)
    print(f"[+] Total windows: {total}")
    print(f"[+] Training windows: {len(train_indices)} ({len(train_days)} days)")
    print(f"[+] Test windows: {len(test_indices)} ({len(test_days)} days)")
    print(f"[+] Train days: {sorted(train_days)}")
    print(f"[+] Test days:  {sorted(test_days)}")
    print("Split strategy: day-based temporal split (no temporal leakage)")

    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    times_train, times_test = times[train_indices], times[test_indices]

    # ----------------------------------------------------------------
    # COMPUTE LAG FEATURES WITHIN EACH SPLIT INDEPENDENTLY
    # This prevents test-set lag values from leaking into training data.
    # ----------------------------------------------------------------
    X_train_lagged, feature_names_lagged = add_lag_features(
        X_train_raw, times_train, feature_names
    )
    X_test_lagged, _ = add_lag_features(X_test_raw, times_test, feature_names)

    # ----------------------------------------------------------------
    # SCALE FEATURES (fit on training data only)
    # ----------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_lagged)
    X_test = scaler.transform(X_test_lagged)

    print("\n[+] Train class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    print("\n[+] Test class distribution:")
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
    np.save(os.path.join(INPUT_DIR, "window_feature_names.npy"), feature_names_lagged)

    dump(scaler, os.path.join(INPUT_DIR, "time_window_scaler.joblib"))

    print("\n[OK] Day-based temporal split completed successfully!")


if __name__ == '__main__':
    main()
