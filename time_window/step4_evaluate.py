import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
MODEL_PATH = os.path.join(INPUT_DIR, "xgb_time_window_model.joblib")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    print("🔹 Loading test data...")
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)

    model = joblib.load(MODEL_PATH)
    train_classes = joblib.load(os.path.join(INPUT_DIR, "xgb_model_classes.npy"))

    print("🔹 Running inference...")
    y_pred = train_classes[model.predict(X_test).astype(np.int64)]

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
    np.save(os.path.join(REPORT_DIR, "confusion_matrix.npy"), cm)
    np.savetxt(os.path.join(REPORT_DIR, "confusion_matrix.csv"), cm, delimiter=",")

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\n✅ Evaluation complete. Reports saved in 'reports/'")


if __name__ == '__main__':
    main()
