import numpy as np

def compute_r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
        
    return 1 - (ss_res / ss_tot)

def manual_time_series_split(X, n_splits=5):
    n_samples = len(X)

    fold_size = n_samples // (n_splits + 1)

    splits = []

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = train_end + fold_size

        if i == n_splits:
            test_end = n_samples

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(train_end, test_end)

    splits.append((train_indices, test_indices))
    return splits