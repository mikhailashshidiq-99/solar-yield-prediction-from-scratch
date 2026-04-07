import numpy as np


def compute_r2_score(actual_values, predicted_values):
    """Compute R-squared score to evaluate model accuracy."""
    sum_squared_residuals = np.sum((actual_values - predicted_values) ** 2)
    sum_squared_total = np.sum((actual_values - np.mean(actual_values)) ** 2)

    return 1 - (sum_squared_residuals / sum_squared_total)
