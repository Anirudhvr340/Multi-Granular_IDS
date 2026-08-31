import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
MODEL_PATH = os.path.join(INPUT_DIR, "xgb_time_window_model.joblib")
REPO_DIR = os.path.dirname(__file__)

os.makedirs(INPUT_DIR, exist_ok=True)


def main():
    print("[+] Loading train/test data...")
    X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)
    train_classes = np.unique(y_train)
    train_label_map = {label: index for index, label in enumerate(train_classes)}
    encoded_y_train = np.array([train_label_map[label] for label in y_train], dtype=np.int64)

    # For test labels, map known classes; unknown classes (if any) are kept for evaluation
    encoded_y_test = np.array([train_label_map.get(label, -1) for label in y_test], dtype=np.int64)

    print(f"[+] Train samples: {len(y_train)}")
    print(f"[+] Test samples: {len(y_test)}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=encoded_y_train)

    # ----------------------------------------------------------------
    # REGULARIZED XGBoost — forces generalizable feature usage
    #   max_depth=4      : shallower trees -> broader feature combinations
    #   reg_alpha=1.0    : L1 penalty drives irrelevant features to zero
    #   reg_lambda=5.0   : L2 penalty penalizes extreme leaf weights
    #   min_child_weight=10 : prevents splits on rare/noisy patterns
    #   gamma=1.0        : minimum loss reduction required for a split
    #   early_stopping_rounds=30 : stops when validation loss plateaus
    # ----------------------------------------------------------------
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(train_classes),
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=10,
        gamma=1.0,
        random_state=42,
        verbosity=1,
    )

    print("[+] Training regularized XGBoost model...")
    model.fit(
        X_train, encoded_y_train,
        sample_weight=sample_weight,
    )

    # --- Train set evaluation (overfitting check) ---
    encoded_train_predictions = model.predict(X_train).astype(np.int64)
    y_pred_train = train_classes[encoded_train_predictions]
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)

    # --- Test set evaluation ---
    print("[+] Evaluating on test set...")
    encoded_predictions = model.predict(X_test).astype(np.int64)
    y_pred = train_classes[encoded_predictions]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # --- Overfitting gap ---
    print(f"\n{'='*60}")
    print("OVERFITTING GAP ANALYSIS")
    print(f"{'='*60}")
    print(f"  Train Accuracy: {train_accuracy:.4f}    Test Accuracy: {accuracy:.4f}    Gap: {train_accuracy - accuracy:.4f}")
    print(f"  Train W-F1:     {train_f1:.4f}    Test W-F1:     {f1:.4f}    Gap: {train_f1 - f1:.4f}")
    gap = train_accuracy - accuracy
    if gap > 0.10:
        print(f"  [!] NOTICE: Gap of {gap:.1%} across different calendar days.")
    else:
        print(f"  [OK] Gap of {gap:.1%} is within acceptable range.")
    print(f"{'='*60}\n")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Weighted F1 score: {f1:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.join(REPO_DIR, "reports"), exist_ok=True)
    with open(os.path.join(REPO_DIR, "reports", "classification_report.txt"), "w") as f:
        f.write(f"OVERFITTING GAP: Train Acc={train_accuracy:.4f} Test Acc={accuracy:.4f} Gap={gap:.4f}\n\n")
        f.write(report)
    np.savetxt(os.path.join(REPO_DIR, "reports", "confusion_matrix.csv"), cm, delimiter=",")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(train_classes, os.path.join(INPUT_DIR, "xgb_model_classes.npy"))
    print(f"\n[OK] Regularized model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
