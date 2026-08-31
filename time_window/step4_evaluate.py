import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
MODEL_PATH = os.path.join(INPUT_DIR, "xgb_time_window_model.joblib")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    print("[+] Loading train and test data...")
    X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)

    model = joblib.load(MODEL_PATH)
    train_classes = joblib.load(os.path.join(INPUT_DIR, "xgb_model_classes.npy"))

    # --- Test set evaluation ---
    print("[+] Running inference on test set...")
    y_pred_test = train_classes[model.predict(X_test).astype(np.int64)]
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    # --- Train set evaluation (for overfitting detection) ---
    print("[+] Running inference on train set (overfitting check)...")
    y_pred_train = train_classes[model.predict(X_train).astype(np.int64)]
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
    train_macro_f1 = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    # --- Overfitting gap analysis ---
    print("\n" + "=" * 60)
    print("OVERFITTING GAP ANALYSIS")
    print("=" * 60)
    print(f"  Train Accuracy: {train_accuracy:.4f}    Test Accuracy: {test_accuracy:.4f}    Gap: {train_accuracy - test_accuracy:.4f}")
    print(f"  Train W-F1:     {train_f1:.4f}    Test W-F1:     {test_f1:.4f}    Gap: {train_f1 - test_f1:.4f}")
    print(f"  Train M-F1:     {train_macro_f1:.4f}    Test M-F1:     {test_macro_f1:.4f}    Gap: {train_macro_f1 - test_macro_f1:.4f}")
    gap = train_accuracy - test_accuracy
    if gap > 0.10:
        print(f"  [!] NOTICE: Train-test accuracy gap of {gap:.1%} across different calendar days.")
    else:
        print(f"  [OK] GOOD: Train-test gap of {gap:.1%} is within acceptable range.")
    print("=" * 60)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Weighted F1 score: {test_f1:.4f}")

    report = classification_report(
        y_test,
        y_pred_test,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred_test)
    np.save(os.path.join(REPORT_DIR, "confusion_matrix.npy"), cm)
    np.savetxt(os.path.join(REPORT_DIR, "confusion_matrix.csv"), cm, delimiter=",")

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"OVERFITTING GAP: Train Acc={train_accuracy:.4f} Test Acc={test_accuracy:.4f} Gap={gap:.4f}\n")
        f.write(f"Train W-F1={train_f1:.4f} Test W-F1={test_f1:.4f}\n\n")
        f.write(report)

    print("\n[OK] Evaluation complete. Reports saved in 'reports/'")


if __name__ == '__main__':
    main()
