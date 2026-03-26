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

    def transform(self, X):
        """
        Adds cos(SZA) and sin(SZA) columns to the feature matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features + 2)
        """
        sza_radians = np.deg2rad(X[:, self.sza_column_index])

        cos_sza = np.cos(sza_radians).reshape(-1, 1)
        sin_sza = np.sin(sza_radians).reshape(-1, 1)

        X_transformed = np.hstack((X, cos_sza, sin_sza))

        return X_transformed


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

    def transform(self, X):
        """
        Generate polynomial features up to the specified degree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_poly : np.ndarray
            Matrix with original + polynomial + interaction features.
        """
        n_samples, n_features = X.shape

        # Start with the original features
        features_list = [X]

        if self.degree >= 2:
            # Add squared terms and interaction terms (degree 2)
            for i in range(n_features):
                for j in range(i, n_features):
                    product = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features_list.append(product)

        if self.degree >= 3:
            # Add cubic terms (degree 3)
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        product = (X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1)
                        features_list.append(product)

        X_poly = np.hstack(features_list)

        return X_poly
