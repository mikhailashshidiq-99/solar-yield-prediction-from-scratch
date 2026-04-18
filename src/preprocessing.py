import numpy as np

# Z-Standardization
# Z = (X - mu)/sigma

def Z_Standardization(df):
    y = df['GHI'].values

    features = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']
    X = df[features].values

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # Avoid division by zero
    # If any value = 0, this replaces it with 1e-8
    sigma = np.where(sigma == 0, 1e-8, sigma)

    X_scaled = (X-mu) / sigma

    return X_scaled, y, mu, sigma


# For adding +b in the equation
def bias_column(X):
    ones_column = np.ones((X.shape[0], 1))

    X__with_bias = np.hstack((ones_column, X))

    return X__with_bias

def initialize_weights(n_features):
    w = np.zeros(n_features)

    return w

def time_series_split(X, y, test_ratio = 0.2):
    split_index = int(X.shape[0] * (1 - test_ratio))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# ===== NON-LINEAR PREPROCESSING PIPELINE =====
# Adds cos/sin of Solar Zenith Angle + Polynomial Features
# to capture the non-linear physics of solar irradiance.

from src.feature_engineering import SolarZenithTransformer, PolynomialFeatures

def preprocess_nonlinear(df, poly_degree=2):
    y = df['GHI'].values

    features = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']
    X = df[features].values

    sza_transformer = SolarZenithTransformer(sza_column_index=4)
    X_with_trig = sza_transformer.transform(X)

    mu = np.mean(X_with_trig, axis=0)
    sigma = np.std(X_with_trig, axis=0)
    sigma = np.where(sigma == 0, 1e-8, sigma)
    X_scaled = (X_with_trig - mu) / sigma

    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.transform(X_scaled)

    return X_poly, y, mu, sigma

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
