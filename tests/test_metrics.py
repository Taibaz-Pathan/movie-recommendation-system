"""Unit tests for src/evaluation/metrics.py."""

import math

import numpy as np
import pytest

from src.evaluation.metrics import mae, rmse


def test_rmse_perfect_prediction():
    """RMSE of identical arrays should be exactly 0.0."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    """RMSE should match the manually computed expected value."""
    y_true = [3.0, 4.0, 5.0]
    y_pred = [2.0, 4.0, 5.0]
    # Errors: [1, 0, 0] -> MSE = 1/3 -> RMSE = sqrt(1/3)
    expected = math.sqrt(1.0 / 3.0)
    result = rmse(y_true, y_pred)
    assert math.isclose(
        result, expected, rel_tol=1e-9
    ), f"Expected RMSE {expected:.6f}, got {result:.6f}"


def test_mae_perfect_prediction():
    """MAE of identical arrays should be exactly 0.0."""
    y = [1.0, 2.5, 3.5, 4.0]
    assert mae(y, y) == 0.0


def test_mae_known_value():
    """MAE should match the manually computed expected value."""
    y_true = [3.0, 4.0, 5.0]
    y_pred = [2.0, 3.0, 4.0]
    # Absolute errors: [1, 1, 1] -> MAE = 1.0
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_raises_on_empty_input():
    """rmse() should raise ValueError when passed empty arrays."""
    with pytest.raises(ValueError):
        rmse([], [])


def test_mae_raises_on_empty_input():
    """mae() should raise ValueError when passed empty arrays."""
    with pytest.raises(ValueError):
        mae([], [])


def test_rmse_raises_on_shape_mismatch():
    """rmse() should raise ValueError when array lengths differ."""
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])


def test_mae_raises_on_shape_mismatch():
    """mae() should raise ValueError when array lengths differ."""
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_rmse_accepts_numpy_arrays():
    """rmse() should work correctly when passed numpy arrays."""
    y_true = np.array([4.0, 3.0, 5.0])
    y_pred = np.array([4.0, 3.0, 5.0])
    assert rmse(y_true, y_pred) == 0.0


def test_mae_accepts_numpy_arrays():
    """mae() should work correctly when passed numpy arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert mae(y_true, y_pred) == pytest.approx(0.5)
