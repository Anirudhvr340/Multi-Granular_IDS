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
    print("🔹 Loading train/test data...")
    X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)
    train_classes = np.unique(y_train)
    train_label_map = {label: index for index, label in enumerate(train_classes)}
    encoded_y_train = np.array([train_label_map[label] for label in y_train], dtype=np.int64)

    print(f"🔹 Train samples: {len(y_train)}")
    print(f"🔹 Test samples: {len(y_test)}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=encoded_y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(train_classes),
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    print("🔹 Training XGBoost model...")
    model.fit(X_train, encoded_y_train, sample_weight=sample_weight)

    print("🔹 Evaluating on test set...")
    encoded_predictions = model.predict(X_test).astype(np.int64)
    y_pred = train_classes[encoded_predictions]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 score: {f1:.4f}")

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
        f.write(report)
    np.savetxt(os.path.join(REPO_DIR, "reports", "confusion_matrix.csv"), cm, delimiter=",")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(train_classes, os.path.join(INPUT_DIR, "xgb_model_classes.npy"))
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
