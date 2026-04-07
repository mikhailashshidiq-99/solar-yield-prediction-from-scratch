import numpy as np


class SolarZenithTransformer:
    """
    Transforms the raw Solar Zenith Angle (in degrees) into its
    cosine and sine components.

    Physics: Surface irradiance is proportional to cos(zenith_angle),
    so encoding this relationship directly gives the linear model
    the "shape" it needs without requiring a non-linear model.
    """

    def __init__(self, sza_column_index=4):
        """
        Parameters
        ----------
        sza_column_index : int
            Index of the 'Solar Zenith Angle' column in the feature matrix.
            Default is 4 (based on feature order: DNI, DHI, Temp, Wind, SZA).
        """
        self.sza_column_index = sza_column_index

    def transform(self, feature_matrix):
        """
        Adds cos(SZA) and sin(SZA) columns to the feature matrix.

        Parameters
        ----------
        feature_matrix : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        transformed_matrix : np.ndarray of shape (n_samples, n_features + 2)
        """
        sza_radians = np.deg2rad(feature_matrix[:, self.sza_column_index])

        cos_sza = np.cos(sza_radians).reshape(-1, 1)
        sin_sza = np.sin(sza_radians).reshape(-1, 1)

        transformed_matrix = np.hstack((feature_matrix, cos_sza, sin_sza))

        return transformed_matrix


class PolynomialFeatures:
    """
    Generates polynomial and interaction features from a feature matrix.
    Pure NumPy implementation — no Scikit-Learn.

    For degree=2 with features [a, b]:
        Output = [a, b, a², ab, b²]

    This allows a linear model (y = Xw) to fit non-linear curves
    because the features themselves are non-linear transforms.
    """

    def __init__(self, degree=2):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree. Default is 2.
        """
        self.degree = degree

    def transform(self, feature_matrix):
        """
        Generate polynomial features up to the specified degree.

        Parameters
        ----------
        feature_matrix : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        polynomial_matrix : np.ndarray
            Matrix with original + polynomial + interaction features.
        """
        n_samples, n_features = feature_matrix.shape

        # Start with the original features
        features_list = [feature_matrix]

        if self.degree >= 2:
            # Add squared terms and interaction terms (degree 2)
            for i in range(n_features):
                for j in range(i, n_features):
                    product = (feature_matrix[:, i] * feature_matrix[:, j]).reshape(-1, 1)
                    features_list.append(product)

        if self.degree >= 3:
            # Add cubic terms (degree 3)
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        product = (feature_matrix[:, i] * feature_matrix[:, j] * feature_matrix[:, k]).reshape(-1, 1)
                        features_list.append(product)

        polynomial_matrix = np.hstack(features_list)

        return polynomial_matrix
