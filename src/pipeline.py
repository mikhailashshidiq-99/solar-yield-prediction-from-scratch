"""
Orchestrator for building complete preprocessing pipelines.
Combines atomic steps from preprocessing and feature_engineering.
"""

import numpy as np

from src.preprocessing import extract_features_and_target, standardize_features
from src.feature_engineering import SolarZenithTransformer, PolynomialFeatures


def build_nonlinear_pipeline(dataframe, poly_degree=2):
    """
    Full pipeline for the non-linear model:
    1. Extract features & target
    2. Add cos/sin of SZA (physics-informed)
    3. Z-standardize everything
    4. Generate polynomial + interaction terms
    """
    # extract raw features and target
    feature_matrix, target_values = extract_features_and_target(dataframe)

    # add trig features for Solar Zenith Angle (column 4)
    sza_transformer = SolarZenithTransformer(sza_column_index=4)
    features_with_trigonometry = sza_transformer.transform(feature_matrix)

    # standardize all features
    standardized_features, feature_means, feature_stds = standardize_features(
        features_with_trigonometry
    )

    # generate polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    polynomial_features = poly.transform(standardized_features)

    return polynomial_features, target_values, feature_means, feature_stds
