"""
Pipeline module for building complete preprocessing pipelines.

This module contains orchestrator functions that combine multiple
atomic preprocessing steps into end-to-end pipelines. Each pipeline
function calls smaller, reusable functions from src.preprocessing
and src.feature_engineering.
"""

import numpy as np

from src.preprocessing import extract_features_and_target, standardize_features
from src.feature_engineering import SolarZenithTransformer, PolynomialFeatures


def build_nonlinear_pipeline(dataframe, poly_degree=2):
    """
    Full preprocessing pipeline for the non-linear model.

    Steps:
        1. Extract features & target from the DataFrame
        2. Add cos(SZA) and sin(SZA) — physics-informed features
        3. Z-standardize all features (including the new trig ones)
        4. Generate polynomial + interaction terms

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing weather and solar data.
    poly_degree : int
        Maximum polynomial degree for feature expansion (default 2).

    Returns
    -------
    polynomial_features : np.ndarray
        The final feature matrix after all transformations.
    target_values : np.ndarray
        Array of target values (GHI).
    feature_means : np.ndarray
        Mean of each feature column (after trig transform, before poly).
    feature_stds : np.ndarray
        Std of each feature column (after trig transform, before poly).
    """
    # Step 1: Extract features & target
    feature_matrix, target_values = extract_features_and_target(dataframe)

    # Step 2: Add cos(SZA) and sin(SZA)
    # SZA is column index 4 in our feature list
    sza_transformer = SolarZenithTransformer(sza_column_index=4)
    features_with_trigonometry = sza_transformer.transform(feature_matrix)
    # Now we have 7 columns: DNI, DHI, Temp, Wind, SZA, cos(SZA), sin(SZA)

    # Step 3: Z-Standardize all features (including the new trig ones)
    standardized_features, feature_means, feature_stds = standardize_features(
        features_with_trigonometry
    )

    # Step 4: Generate polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    polynomial_features = poly.transform(standardized_features)

    return polynomial_features, target_values, feature_means, feature_stds
