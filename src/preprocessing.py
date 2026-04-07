import numpy as np


# ===== CONSTANTS =====
FEATURE_COLUMNS = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']
TARGET_COLUMN = 'GHI'


def extract_features_and_target(dataframe):
    """
    Extract feature matrix and target values from a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing weather and solar data.

    Returns
    -------
    feature_matrix : np.ndarray
        Matrix of input features (DNI, DHI, Temperature, Wind Speed, SZA).
    target_values : np.ndarray
        Array of target values (GHI).
    """
    target_values = dataframe[TARGET_COLUMN].values
    feature_matrix = dataframe[FEATURE_COLUMNS].values

    return feature_matrix, target_values


def standardize_features(feature_matrix):
    """
    Apply Z-standardization: Z = (X - mean) / std.

    Standardizes each feature column to have zero mean and unit variance.
    Replaces zero standard deviation with 1e-8 to avoid division by zero.

    Parameters
    ----------
    feature_matrix : np.ndarray
        The raw feature matrix to standardize.

    Returns
    -------
    standardized_features : np.ndarray
        The standardized feature matrix.
    feature_means : np.ndarray
        Mean of each feature column (used for inverse transform).
    feature_stds : np.ndarray
        Standard deviation of each feature column (used for inverse transform).
    """
    feature_means = np.mean(feature_matrix, axis=0)
    feature_stds = np.std(feature_matrix, axis=0)

    # Avoid division by zero
    # If any std value = 0, replace it with 1e-8
    feature_stds = np.where(feature_stds == 0, 1e-8, feature_stds)

    standardized_features = (feature_matrix - feature_means) / feature_stds

    return standardized_features, feature_means, feature_stds


def z_standardization(dataframe):
    """
    Extract features and target from a DataFrame, then Z-standardize the features.

    This is a convenience function that combines extract_features_and_target()
    and standardize_features() into a single call for the linear model pipeline.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing weather and solar data.

    Returns
    -------
    standardized_features : np.ndarray
        The standardized feature matrix.
    target_values : np.ndarray
        Array of target values (GHI).
    feature_means : np.ndarray
        Mean of each feature column.
    feature_stds : np.ndarray
        Standard deviation of each feature column.
    """
    feature_matrix, target_values = extract_features_and_target(dataframe)
    standardized_features, feature_means, feature_stds = standardize_features(feature_matrix)

    return standardized_features, target_values, feature_means, feature_stds


def add_bias_column(feature_matrix):
    """
    Prepend a column of ones to the feature matrix for the bias term (+b).

    Parameters
    ----------
    feature_matrix : np.ndarray
        The feature matrix to augment.

    Returns
    -------
    features_with_bias : np.ndarray
        The feature matrix with a leading column of ones.
    """
    ones_column = np.ones((feature_matrix.shape[0], 1))
    features_with_bias = np.hstack((ones_column, feature_matrix))

    return features_with_bias


def initialize_weights(n_features):
    """
    Initialize a weight vector with zeros.

    Parameters
    ----------
    n_features : int
        Number of features (including bias if applicable).

    Returns
    -------
    initial_weights : np.ndarray
        A zero-initialized weight vector.
    """
    initial_weights = np.zeros(n_features)

    return initial_weights


def time_series_split(feature_matrix, target_values, test_ratio=0.2):
    """
    Split data chronologically for time-series validation.

    Unlike random splitting, this preserves temporal order — the training set
    always comes before the test set in time.

    Parameters
    ----------
    feature_matrix : np.ndarray
        The feature matrix.
    target_values : np.ndarray
        The target values.
    test_ratio : float
        Fraction of data to use for testing (default 0.2).

    Returns
    -------
    train_features : np.ndarray
    test_features : np.ndarray
    train_targets : np.ndarray
    test_targets : np.ndarray
    """
    split_index = int(feature_matrix.shape[0] * (1 - test_ratio))

    train_features = feature_matrix[:split_index]
    test_features = feature_matrix[split_index:]

    train_targets = target_values[:split_index]
    test_targets = target_values[split_index:]

    return train_features, test_features, train_targets, test_targets


def manual_time_series_split(feature_matrix, n_splits=5):
    """
    Create expanding-window time-series cross-validation splits.

    Each fold uses a progressively larger training set while keeping
    the test set as the next chronological block.

    Parameters
    ----------
    feature_matrix : np.ndarray
        The feature matrix to split.
    n_splits : int
        Number of cross-validation folds (default 5).

    Returns
    -------
    splits : list of tuples
        Each tuple contains (train_indices, test_indices).
    """
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
