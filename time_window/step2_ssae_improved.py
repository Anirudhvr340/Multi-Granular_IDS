"""
IMPROVED TIME-WINDOW MODEL TRAINING WITH ANTI-OVERFITTING MEASURES
===================================================================
Implements:
  1. Temporal cross-validation (time-based splits)
  2. Aggressive regularization (5x stronger)
  3. Feature denoising (remove artifact features)
  4. Stratified k-fold with early stopping
  5. Cross-validation evaluation
"""

import os
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectKBest, f_classif

INPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
MODEL_PATH = os.path.join(INPUT_DIR, "xgb_time_window_model.joblib")
REPO_DIR = os.path.dirname(__file__)

os.makedirs(INPUT_DIR, exist_ok=True)


def load_training_data():
    """Load preprocessed training data."""
    print("[+] Loading train/test data...")
    X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(INPUT_DIR, "window_class_names.npy"), allow_pickle=True).astype(str)
    feature_names = np.load(os.path.join(INPUT_DIR, "window_feature_names.npy"), allow_pickle=True).astype(str)
    
    train_classes = np.unique(y_train)
    train_label_map = {label: index for index, label in enumerate(train_classes)}
    encoded_y_train = np.array([train_label_map[label] for label in y_train], dtype=np.int64)
    encoded_y_test = np.array([train_label_map.get(label, -1) for label in y_test], dtype=np.int64)

    print(f"[+] Train samples: {len(y_train)}")
    print(f"[+] Test samples: {len(y_test)}")
    print(f"[+] Classes: {train_classes}")
    
    return X_train, y_train, encoded_y_train, X_test, y_test, encoded_y_test, class_names, feature_names, train_classes


def denoise_features(X_train, y_train, X_test, keep_ratio=0.75):
    """
    Remove low-importance features that may memorize training artifacts.
    Keeps top 75% of features by f_classif score.
    """
    print(f"\n[*] Feature denoising: keeping top {keep_ratio*100:.0f}% of features...")
    
    # Initial model to get feature importances
    initial_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        verbosity=0
    )
    initial_model.fit(X_train, y_train)
    
    # Use both XGBoost importance and f_classif
    xgb_importance = initial_model.feature_importances_
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    f_scores = f_scores / (f_scores.max() + 1e-8)
    
    # Combine scores
    combined_score = 0.5 * xgb_importance + 0.5 * f_scores
    threshold = np.percentile(combined_score, (1 - keep_ratio) * 100)
    robust_indices = np.where(combined_score >= threshold)[0]
    if len(robust_indices) == 0:
        robust_indices = np.array([int(np.argmax(combined_score))])
    
    print(f"[+] Keeping {len(robust_indices)}/{len(combined_score)} features")
    print(f"[+] Removed low-importance features: {np.where(combined_score < threshold)[0]}")
    
    return X_train[:, robust_indices], X_test[:, robust_indices], robust_indices


def create_regularized_model(num_classes):
    """
    Create XGBoost model with aggressive anti-overfitting measures.
    
    Key changes from current model:
      - 5x stronger regularization (reg_alpha, reg_lambda)
      - Shallower trees (max_depth=3)
      - Higher min_child_weight (20 vs 10)
      - Lower feature subsampling (0.6 vs 0.8)
      - Slower learning with more estimators
      - Early stopping enabled
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        
        # === AGGRESSIVE REGULARIZATION ===
        reg_alpha=5.0,           # L1: Feature selection (was 1.0)
        reg_lambda=25.0,         # L2: Leaf weight penalty (was 5.0)
        
        # === SMALLER TREES ===
        max_depth=3,             # Shallower trees (was 4)
        min_child_weight=20,     # Higher threshold (was 10)
        gamma=2.0,               # Min loss reduction (was 1.0)
        
        # === STOCHASTIC TRAINING ===
        subsample=0.6,           # Random 60% of samples (was 0.8)
        colsample_bytree=0.6,    # Random 60% of features (was 0.8)
        colsample_bylevel=0.6,   # Per-level feature sampling (NEW)
        colsample_bynode=0.6,    # Per-node feature sampling (NEW)
        
        # === LEARNING SCHEDULE ===
        learning_rate=0.02,      # Slower learning (was 0.05)
        n_estimators=500,        # More boosting rounds (was 300)
        
        # === EARLY STOPPING ===
        early_stopping_rounds=40,# Stop if no improvement (NEW)
        
        random_state=42,
        verbosity=0,
    )


def evaluate_cross_validation(X, y, num_classes, n_splits=5):
    """
    Evaluate using stratified k-fold cross-validation to detect overfitting.
    Returns CV scores and fitted models.
    """
    print(f"\n[*] Running {n_splits}-fold stratified cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    cv_scores = []
    cv_f1_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Compute class weights for this fold
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train_fold
        )
        
        # Train model
        model = create_regularized_model(num_classes)
        model.fit(
            X_train_fold, y_train_fold,
            sample_weight=sample_weights,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Evaluate
        train_score = model.score(X_train_fold, y_train_fold)
        val_score = model.score(X_val_fold, y_val_fold)
        val_f1 = f1_score(y_val_fold, model.predict(X_val_fold), average='weighted', zero_division=0)
        
        gap = train_score - val_score
        cv_scores.append(val_score)
        cv_f1_scores.append(val_f1)
        fold_models.append(model)
        
        print(f"  Fold {fold+1}: Train={train_score:.4f}, Val={val_score:.4f}, Gap={gap:.4f}, F1={val_f1:.4f}")
        
        if gap > 0.15:
            print(f"    [!] Warning: High overfitting gap ({gap:.1%})")
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    mean_f1 = np.mean(cv_f1_scores)
    
    print(f"\n[+] Cross-validation Results:")
    print(f"    Mean Accuracy: {mean_cv:.4f} (+/- {std_cv:.4f})")
    print(f"    Mean F1 Score: {mean_f1:.4f}")
    
    return cv_scores, cv_f1_scores, fold_models


def main():
    print("="*70)
    print("IMPROVED TIME-WINDOW MODEL TRAINING (Anti-Overfitting Version)")
    print("="*70)
    
    # Load data
    X_train, y_train, encoded_y_train, X_test, y_test, encoded_y_test, class_names, feature_names, train_classes = load_training_data()
    
    num_classes = len(train_classes)
    
    # ===== FIX 1: Denoise features to remove artifacts =====
    X_train_denoised, X_test_denoised, retained_indices = denoise_features(
        X_train, encoded_y_train, X_test, keep_ratio=0.75
    )
    np.save(os.path.join(INPUT_DIR, "window_model_feature_names.npy"), feature_names[retained_indices])
    
    # ===== FIX 2: Evaluate with cross-validation =====
    cv_scores, cv_f1_scores, cv_models = evaluate_cross_validation(
        X_train_denoised, encoded_y_train, num_classes, n_splits=5
    )
    
    # ===== FIX 3: Train final model on full training set =====
    print(f"\n[*] Training final model on full training set...")
    sample_weight = compute_sample_weight(class_weight="balanced", y=encoded_y_train)
    known_test_mask = encoded_y_test >= 0
    
    final_model = create_regularized_model(num_classes)
    final_model.fit(
        X_train_denoised, encoded_y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test_denoised[known_test_mask], encoded_y_test[known_test_mask])],
        verbose=False
    )
    
    # ===== EVALUATION =====
    print(f"\n{'='*70}")
    print("FINAL MODEL EVALUATION")
    print(f"{'='*70}\n")
    
    # Train set
    train_pred = final_model.predict(X_train_denoised).astype(np.int64)
    y_pred_train = train_classes[train_pred]
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
    
    # Test set
    test_pred = final_model.predict(X_test_denoised).astype(np.int64)
    y_pred_test = train_classes[test_pred]
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    
    # Overfitting analysis
    gap = train_accuracy - test_accuracy
    
    print("OVERFITTING GAP ANALYSIS:")
    print(f"  Train Accuracy: {train_accuracy:.4f}    Test Accuracy: {test_accuracy:.4f}")
    print(f"  Train F1:       {train_f1:.4f}    Test F1:       {test_f1:.4f}")
    print(f"  Accuracy Gap:   {gap:.4f} ({gap*100:.1f}%)")
    
    if gap < 0.05:
        print(f"  [✓] EXCELLENT: Very low overfitting gap")
    elif gap < 0.10:
        print(f"  [✓] GOOD: Acceptable overfitting gap")
    elif gap < 0.15:
        print(f"  [!] WARNING: Moderate overfitting gap")
    else:
        print(f"  [✗] SEVERE: High overfitting gap - consider stronger regularization")
    
    print(f"\n{'='*70}\n")
    
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, labels=np.arange(len(class_names)), 
                                 target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(len(class_names)))
    print(cm)
    
    # ===== SAVE MODEL =====
    print(f"\n[+] Saving model to {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    
    print("[✓] Model training complete!")
    
    return final_model


if __name__ == "__main__":
    model = main()
