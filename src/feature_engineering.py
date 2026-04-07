import numpy as np


class SolarZenithTransformer:
    """
    Adds cos(SZA) and sin(SZA) columns to the feature matrix.

    Surface irradiance is proportional to cos(zenith_angle),
    so this encodes the physics directly into the features.
    """

    def __init__(self, sza_column_index=4):
        self.sza_column_index = sza_column_index

    def transform(self, feature_matrix):
        sza_radians = np.deg2rad(feature_matrix[:, self.sza_column_index])

        cos_sza = np.cos(sza_radians).reshape(-1, 1)
        sin_sza = np.sin(sza_radians).reshape(-1, 1)

        transformed_matrix = np.hstack((feature_matrix, cos_sza, sin_sza))

        return transformed_matrix


class PolynomialFeatures:
    """
    Generates polynomial and interaction features. Pure NumPy.

    For degree=2 with features [a, b]:
        Output = [a, b, a², ab, b²]
    """

    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, feature_matrix):
        n_samples, n_features = feature_matrix.shape

        features_list = [feature_matrix]

        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    product = (feature_matrix[:, i] * feature_matrix[:, j]).reshape(-1, 1)
                    features_list.append(product)

        if self.degree >= 3:
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        product = (feature_matrix[:, i] * feature_matrix[:, j] * feature_matrix[:, k]).reshape(-1, 1)
                        features_list.append(product)

        polynomial_matrix = np.hstack(features_list)

        return polynomial_matrix
