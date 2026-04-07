import numpy as np


FEATURE_COLUMNS = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']
TARGET_COLUMN = 'GHI'


def extract_features_and_target(dataframe):
    """Pull feature matrix and target array from a DataFrame."""
    target_values = dataframe[TARGET_COLUMN].values
    feature_matrix = dataframe[FEATURE_COLUMNS].values

    return feature_matrix, target_values


def standardize_features(feature_matrix):
    """
    Z-standardize: Z = (X - mean) / std.
    Returns the scaled matrix, means, and stds.
    """
    feature_means = np.mean(feature_matrix, axis=0)
    feature_stds = np.std(feature_matrix, axis=0)

    # avoid division by zero
    feature_stds = np.where(feature_stds == 0, 1e-8, feature_stds)

    standardized_features = (feature_matrix - feature_means) / feature_stds

    return standardized_features, feature_means, feature_stds


def z_standardization(dataframe):
    """
    Convenience wrapper: extract features from df, then standardize.
    Used by the linear model pipeline.
    """
    feature_matrix, target_values = extract_features_and_target(dataframe)
    standardized_features, feature_means, feature_stds = standardize_features(feature_matrix)

    return standardized_features, target_values, feature_means, feature_stds


def add_bias_column(feature_matrix):
    """Prepend a column of ones for the bias term (+b)."""
    ones_column = np.ones((feature_matrix.shape[0], 1))
    features_with_bias = np.hstack((ones_column, feature_matrix))

    return features_with_bias


def initialize_weights(n_features):
    """Initialize weight vector with zeros."""
    initial_weights = np.zeros(n_features)

    return initial_weights


def time_series_split(feature_matrix, target_values, test_ratio=0.2):
    """Split data chronologically — training before test in time."""
    split_index = int(feature_matrix.shape[0] * (1 - test_ratio))

    train_features = feature_matrix[:split_index]
    test_features = feature_matrix[split_index:]

    train_targets = target_values[:split_index]
    test_targets = target_values[split_index:]

    return train_features, test_features, train_targets, test_targets


def manual_time_series_split(feature_matrix, n_splits=5):
    """Expanding-window time-series cross-validation splits."""
    n_samples = len(feature_matrix)
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
