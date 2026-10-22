"""
OVERFITTING DIAGNOSIS & FIX STRATEGIES FOR TIME-WINDOW MODEL
=============================================================
Current Problem:
  - Train: High accuracy on 2018 CICIDS data
  - Validation: Fails on synthetic data (different distribution)
  - Root Cause: Model memorizes 2018 artifacts instead of learning attack patterns

Recommended Fixes (in priority order):
"""

# ========================================================================
# FIX 1: TEMPORAL CROSS-VALIDATION (NOT RANDOM SPLIT)
# ========================================================================
# Current: Random split by calendar day → temporal leakage
# Fix: Use time-based split → ensure model sees data temporally

TEMPORAL_CV_FIX = """
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import cross_val_score

# Replace random split with temporal split
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_test_fold, y_test_fold)
    print(f"Fold {fold}: {score:.4f}")  # Should show stable, realistic scores

# This prevents data leakage from future -> past
"""

# ========================================================================
# FIX 2: AGGRESSIVE REGULARIZATION LEVELS
# ========================================================================
# Current: reg_alpha=1.0, reg_lambda=5.0 → still allows overfitting
# Fix: Increase by 5-10x for severe overfitting

AGGRESSIVE_REGULARIZATION = """
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(train_classes),
    
    # STRONGER L1/L2 PENALTIES
    reg_alpha=5.0,        # ← Increase from 1.0 (feature selection pressure)
    reg_lambda=25.0,      # ← Increase from 5.0 (leaf weight penalty)
    
    # SMALLER TREES
    max_depth=3,          # ← Reduce from 4 (prevents deep memorization)
    min_child_weight=20,  # ← Increase from 10 (prevent small splits)
    
    # FEATURE SUBSAMPLING
    subsample=0.6,        # ← Reduce from 0.8 (stochastic training)
    colsample_bytree=0.6, # ← Reduce from 0.8 (random feature selection)
    colsample_bylevel=0.6,# ← NEW: per-tree-level feature sampling
    
    # REDUCED LEARNING
    learning_rate=0.02,   # ← Reduce from 0.05 (smaller steps, more updates)
    n_estimators=500,     # ← Increase from 300 (more weak learners)
    
    # LOSS REDUCTION THRESHOLD
    gamma=2.0,            # ← Increase from 1.0 (higher split threshold)
    
    # EARLY STOPPING
    early_stopping_rounds=40,  # ← NEW: Stop when validation doesn't improve
    eval_metric="mlogloss",
    
    tree_method="hist",
    random_state=42,
)

# Train with eval_set for early stopping
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, 
          eval_set=eval_set,
          verbose=False)
"""

# ========================================================================
# FIX 3: FEATURE DENOISING (REMOVE ARTIFACT FEATURES)
# ========================================================================
# Current: All 45+ features used → some memorize 2018 patterns
# Fix: Remove features that don't generalize

FEATURE_DENOISING = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Identify features that overfit
feature_importances = model.feature_importances_
artifact_features = feature_importances < np.percentile(feature_importances, 20)

print(f"Removing {artifact_features.sum()} low-importance features")

# Keep only robust features (top 80%)
robust_feature_indices = np.where(~artifact_features)[0]
X_train_denoised = X_train[:, robust_feature_indices]
X_test_denoised = X_test[:, robust_feature_indices]

# Retrain on denoised features
model.fit(X_train_denoised, y_train)
score = model.score(X_test_denoised, y_test)
print(f"Score after denoising: {score:.4f}")

# Also try SelectKBest to keep only K most discriminative features
selector = SelectKBest(f_classif, k=25)  # Keep top 25 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
"""

# ========================================================================
# FIX 4: STRATIFIED K-FOLD WITH CLASS WEIGHTS
# ========================================================================
# Current: Class weights only on train set
# Fix: Use stratified k-fold + class weights on each fold

STRATIFIED_KFOLD = """
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Compute class weights per fold
    sample_weights = compute_sample_weight(
        class_weight='balanced', 
        y=y_train_fold
    )
    
    model = XGBClassifier(
        reg_alpha=5.0, reg_lambda=25.0,
        max_depth=3, min_child_weight=20,
        subsample=0.6, colsample_bytree=0.6,
        learning_rate=0.02, n_estimators=500,
    )
    
    model.fit(X_train_fold, y_train_fold, 
              sample_weight=sample_weights,
              eval_set=[(X_val_fold, y_val_fold)],
              early_stopping_rounds=40,
              verbose=False)
    
    score = model.score(X_val_fold, y_val_fold)
    scores.append(score)
    print(f"Fold {fold}: {score:.4f}")

print(f"Average CV score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
"""

# ========================================================================
# FIX 5: ENSEMBLE WITH DIVERSE MODELS
# ========================================================================
# Current: Single XGBoost → can memorize
# Fix: Combine XGBoost + LightGBM + CatBoost for robustness

ENSEMBLE_APPROACH = """
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# Diverse ensemble with different architectures
xgb_model = XGBClassifier(
    reg_alpha=5.0, reg_lambda=25.0,
    max_depth=3, learning_rate=0.02,
    n_estimators=300, random_state=42
)

lgb_model = LGBMClassifier(
    num_leaves=15,  # Shallow trees
    lambda_l1=5.0, lambda_l2=25.0,  # L1/L2 penalties
    learning_rate=0.02,
    n_estimators=300,
    random_state=42,
    verbose=-1
)

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train, 
             sample_weight=sample_weight)

score = ensemble.score(X_test, y_test)
print(f"Ensemble score: {score:.4f}")
"""

# ========================================================================
# FIX 6: WINDOW SIZE OPTIMIZATION
# ========================================================================
# Current: Fixed 30-second window
# Fix: Tune window size to capture attack patterns better

WINDOW_SIZE_TUNING = """
import os
os.environ["WINDOW_SECONDS"] = "15"  # Try smaller window first

# Re-run preprocessing with different window sizes
for window_size in [10, 15, 20, 30, 45, 60]:
    os.environ["WINDOW_SECONDS"] = str(window_size)
    # Re-run step0_preprocessing.py → regenerates X_train, X_test
    from time_window import step0_preprocessing as preprocessing
    
    # Score model on new features
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Window {window_size}s: {score:.4f}")
    
# Choose window size with best cross-validation score
"""

# ========================================================================
# FIX 7: DROPOUT & STOCHASTIC REGULARIZATION
# ========================================================================
# Current: Deterministic training
# Fix: Add randomness to prevent overfitting

STOCHASTIC_REGULARIZATION = """
model = XGBClassifier(
    objective="multi:softprob",
    
    # Stochastic sampling (already in current config, but increase)
    subsample=0.5,           # Random 50% of samples per tree
    colsample_bytree=0.5,    # Random 50% of features per tree
    colsample_bylevel=0.5,   # Random 50% of features per split
    colsample_bynode=0.5,    # Random 50% of features per node ← NEW
    
    # Regularization
    reg_alpha=5.0,
    reg_lambda=25.0,
    
    # Learning
    learning_rate=0.01,      # Smaller steps
    n_estimators=800,        # More boosting rounds
    
    tree_method='hist',
    random_state=42,
)
"""

# ========================================================================
# FIX 8: VALIDATION AGAINST MULTIPLE DATASETS
# ========================================================================
# Current: Only validating on synthetic data
# Fix: Validate on multiple heterogeneous datasets

MULTI_DATASET_VALIDATION = """
validation_files = [
    "realistic_synthetic_cicids2018_converted.csv",  # Synthetic
    "02-14-2018.csv",                                 # Original (unseen fold)
    "02-15-2018.csv",                                 # Original (unseen fold)
]

for val_file in validation_files:
    frame = pd.read_csv(val_file)
    # ... preprocess ...
    
    # Score model
    score = model.score(X_val, y_val)
    print(f"{val_file}: {score:.4f}")
    
# Model should generalize to ALL datasets, not just one
"""

# ========================================================================
# IMPLEMENTATION PRIORITY
# ========================================================================
PRIORITY = """
IMMEDIATE (Apply today):
  1. Fix 1: Temporal cross-validation (prevents data leakage)
  2. Fix 2: Aggressive regularization (5x stronger penalties)
  3. Fix 3: Feature denoising (remove artifact features)

SHORT-TERM (Next iteration):
  4. Fix 4: Stratified k-fold (better validation strategy)
  5. Fix 5: Ensemble models (robustness through diversity)

EXPERIMENTAL (If still overfitting):
  6. Fix 6: Window size optimization
  7. Fix 7: Stochastic regularization
  8. Fix 8: Multi-dataset validation
"""

# ========================================================================
# EXPECTED RESULTS
# ========================================================================
EXPECTED_IMPROVEMENTS = """
With all fixes applied:
  
  Before (Current):
    Train Accuracy: 95%+  (overfitting)
    Synthetic Val:  35-40% (fails)
    Gap: 55%+ ← SEVERE overfitting
  
  After (With fixes):
    Train Accuracy: 80-85%  (more realistic)
    Synthetic Val:  75-80%  (generalizes)
    Gap: 5-10% ← Healthy (expected variance)
  
  Key Metric: Gap should be < 10% for good generalization
"""

if __name__ == "__main__":
    print(__doc__)
    print(TEMPORAL_CV_FIX)
    print(AGGRESSIVE_REGULARIZATION)
    print(FEATURE_DENOISING)
    print(STRATIFIED_KFOLD)
    print(ENSEMBLE_APPROACH)
    print(WINDOW_SIZE_TUNING)
    print(STOCHASTIC_REGULARIZATION)
    print(MULTI_DATASET_VALIDATION)
    print("\n" + PRIORITY)
    print("\n" + EXPECTED_IMPROVEMENTS)
