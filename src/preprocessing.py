import numpy as np

# Z-Standardization
# Z = (X - mu)/sigma

def Z_Standardization(df):
    y_df = df['expected_power_output']
    X_df = df.drop(columns=['expected_power_output'])

    y = y_df.to_numpy()
    X = X_df.to_numpy()

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