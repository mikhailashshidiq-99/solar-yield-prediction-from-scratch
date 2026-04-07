import numpy as np


def compute_r2_score(actual_values, predicted_values):
    """
    Compute the R-squared (coefficient of determination) score.

    R² measures how well the model's predictions match the actual values.
    A score of 1.0 means perfect prediction; 0.0 means the model is
    no better than predicting the mean; negative values mean worse
    than the mean.

    Parameters
    ----------
    actual_values : np.ndarray
        The ground-truth target values.
    predicted_values : np.ndarray
        The values predicted by the model.

    Returns
    -------
    r2 : float
        The R-squared score.
    """
    sum_squared_residuals = np.sum((actual_values - predicted_values) ** 2)
    sum_squared_total = np.sum((actual_values - np.mean(actual_values)) ** 2)

    return 1 - (sum_squared_residuals / sum_squared_total)
