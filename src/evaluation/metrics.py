"""Evaluation metrics for the movie recommendation system."""

from typing import Union

import numpy as np


def rmse(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute Root Mean Square Error (RMSE) between true and predicted ratings.

    RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Args:
        y_true: Array-like of ground-truth rating values.
        y_pred: Array-like of predicted rating values, same length as y_true.

    Returns:
        RMSE as a non-negative float. A value of 0.0 indicates a perfect fit.

    Raises:
        ValueError: If either input is empty or the lengths differ.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape} "
            f"but y_pred has shape {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute Mean Absolute Error (MAE) between true and predicted ratings.

    MAE = mean( |y_true - y_pred| )

    Args:
        y_true: Array-like of ground-truth rating values.
        y_pred: Array-like of predicted rating values, same length as y_true.

    Returns:
        MAE as a non-negative float. A value of 0.0 indicates a perfect fit.

    Raises:
        ValueError: If either input is empty or the lengths differ.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape} "
            f"but y_pred has shape {y_pred.shape}."
        )

    return float(np.mean(np.abs(y_true - y_pred)))


if __name__ == "__main__":
    y_true = [4.0, 3.0, 5.0, 2.0, 1.0]
    y_pred = [3.5, 3.0, 4.5, 2.5, 1.5]

    print("Smoke test — metrics.py")
    print(f"  y_true : {y_true}")
    print(f"  y_pred : {y_pred}")
    print(f"  RMSE   : {rmse(y_true, y_pred):.4f}")
    print(f"  MAE    : {mae(y_true, y_pred):.4f}")
