import numpy as np
import os
from sklearn.model_selection import train_test_split

def main():
    print("🔹 Loading window data...")

    # ✅ FIX: allow_pickle=True for variable-length windows
    X_windows = np.load(os.path.join('processed', 'X_windows.npy'), allow_pickle=True)
    y_windows = np.load(os.path.join('processed', 'y_windows.npy'))

    print(f"Total samples: {len(y_windows)}")

    # ==============================
    # CHECK CLASS DISTRIBUTION
    # ==============================
    unique, counts = np.unique(y_windows, return_counts=True)

    print("\n🔹 Class distribution before split:")
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    # ==============================
    # HANDLE RARE CLASSES
    # ==============================
    min_samples = counts.min()

    if min_samples < 2:
        print("\n⚠️ Warning: Some classes have <2 samples")
        print("👉 Using NON-stratified split")
        stratify_option = None
    else:
        stratify_option = y_windows

    # ==============================
    # SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows,
        y_windows,
        test_size=0.2,
        stratify=stratify_option,
        random_state=42
    )

    # ==============================
    # VERIFY DISTRIBUTION
    # ==============================
    print("\n🔹 Train distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    print("\n🔹 Test distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"class {u}: {c}")

    # ==============================
    # SAVE
    # ==============================
    os.makedirs('processed', exist_ok=True)

    # ✅ Keep pickle format for variable-length data
    np.save('processed/X_train.npy', X_train, allow_pickle=True)
    np.save('processed/X_test.npy', X_test, allow_pickle=True)

    np.save('processed/y_train.npy', y_train)
    np.save('processed/y_test.npy', y_test)

    print("\n✅ Split completed successfully!")

if __name__ == '__main__':
    main()