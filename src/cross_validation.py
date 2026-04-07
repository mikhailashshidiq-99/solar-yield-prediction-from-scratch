"""Cross-validation utilities for time-series models."""

import numpy as np

from src.preprocessing import manual_time_series_split, add_bias_column
from src.metrics import compute_r2_score
from src.model import CustomLinearRegression


def run_kfold_cv(features, targets, n_splits=5, learning_rate=0.001, epochs=1000, label="Model"):
    """
    Run k-fold time series cross-validation and print results.
    Returns list of R² scores per fold.
    """
    splits = manual_time_series_split(features, n_splits=n_splits)
    fold_scores = []

    print("=" * 60)
    print(f"  MANUAL TIME SERIES CROSS-VALIDATION ({label})")
    print("=" * 60)

    for fold, (train_index, test_index) in enumerate(splits):
        # slice data chronologically
        X_train_fold = features[train_index]
        X_test_fold = features[test_index]
        y_train_fold = targets[train_index]
        y_test_fold = targets[test_index]

        # fresh model for each fold
        model_cv = CustomLinearRegression(learning_rate=learning_rate, epochs=epochs)
        model_cv.fit(X_train_fold, y_train_fold)

        # predict and clamp negatives
        y_pred_fold = model_cv.predict(X_test_fold)
        y_pred_fold = np.maximum(0, y_pred_fold)

        # score
        score = compute_r2_score(y_test_fold, y_pred_fold)
        fold_scores.append(score)

        print(f"Fold {fold + 1}: R² = {score * 100:.2f}%  |  "
              f"(Train: {len(train_index)} hrs, Test: {len(test_index)} hrs)")

    print("-" * 60)
    print(f"AVERAGE {label.upper()} CV R² SCORE: {np.mean(fold_scores) * 100:.2f}%")
    print("=" * 60)

    return fold_scores
