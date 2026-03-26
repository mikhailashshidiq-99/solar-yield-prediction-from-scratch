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
    """
    Full preprocessing pipeline for the non-linear model:
    1. Extract features & target
    2. Add cos(SZA) and sin(SZA) — physics-informed features
    3. Z-standardize all features
    4. Generate polynomial + interaction terms
    
    Returns: X_poly, y, mu, sigma
    """
    y = df['GHI'].values

    features = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']
    X = df[features].values

    # Step 1: Add cos(SZA) and sin(SZA)
    # SZA is column index 4 in our feature list
    sza_transformer = SolarZenithTransformer(sza_column_index=4)
    X_with_trig = sza_transformer.transform(X)
    # Now X has 7 columns: DNI, DHI, Temp, Wind, SZA, cos(SZA), sin(SZA)

    # Step 2: Z-Standardize all features (including the new trig ones)
    mu = np.mean(X_with_trig, axis=0)
    sigma = np.std(X_with_trig, axis=0)
    sigma = np.where(sigma == 0, 1e-8, sigma)
    X_scaled = (X_with_trig - mu) / sigma

    # Step 3: Generate polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.transform(X_scaled)

    return X_poly, y, mu, sigma