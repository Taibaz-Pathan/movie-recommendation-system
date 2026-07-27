"""Unit tests for src/models/ibcf.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.ibcf import ItemBasedCF


# --- shared fixtures ---

@pytest.fixture()
def sample_ratings():
    """Small synthetic ratings DataFrame: 3 users, 4 movies."""
    return pd.DataFrame({
        "userId":  [1, 1, 2, 2, 2, 2, 3, 3, 3],
        "movieId": [1, 2, 1, 2, 3, 4, 1, 3, 4],
        "rating":  [5.0, 4.0, 4.0, 5.0, 2.0, 1.0, 1.0, 5.0, 4.0],
    })


@pytest.fixture()
def fitted_model(sample_ratings):
    """ItemBasedCF fitted on sample_ratings.

    min_support=1 because the 3-user fixture can never reach the
    default min_support=5 co-rating threshold.
    """
    model = ItemBasedCF(k=2, min_support=1)
    return model.fit(sample_ratings)


# --- fit ---

def test_fit_returns_self(sample_ratings):
    model = ItemBasedCF(k=2, min_support=1)
    result = model.fit(sample_ratings)
    assert result is model


def test_fit_creates_matrix(fitted_model, sample_ratings):
    n_users = sample_ratings["userId"].nunique()
    n_movies = sample_ratings["movieId"].nunique()
    assert isinstance(fitted_model._matrix, pd.DataFrame)
    assert fitted_model._matrix.shape == (n_users, n_movies)


def test_fit_creates_item_sim_matrix(fitted_model, sample_ratings):
    n_movies = sample_ratings["movieId"].nunique()
    assert isinstance(fitted_model._sim_matrix, np.ndarray)
    assert fitted_model._sim_matrix.shape == (n_movies, n_movies)


# --- predict ---

def test_predict_returns_float(fitted_model):
    pred = fitted_model.predict(1, 3)
    assert isinstance(pred, float)


def test_predict_clipped_to_valid_range(fitted_model):
    for user_id, movie_id in [(1, 3), (1, 4), (3, 2)]:
        pred = fitted_model.predict(user_id, movie_id)
        assert 0.5 <= pred <= 5.0


def test_predict_unknown_user_raises(fitted_model):
    with pytest.raises(ValueError):
        fitted_model.predict(999, 1)


def test_predict_unknown_movie_raises(fitted_model):
    with pytest.raises(ValueError):
        fitted_model.predict(1, 999)


# --- recommend ---

def test_recommend_only_unrated_movies(fitted_model):
    # user 1 rated movies 1 and 2, so only 3 and 4 are eligible
    recs = fitted_model.recommend(1, n=10)
    recommended_ids = {movie_id for movie_id, _ in recs}
    assert recommended_ids == {3, 4}


def test_recommend_sorted_descending(fitted_model):
    recs = fitted_model.recommend(1, n=10)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


# --- evaluate ---

def test_evaluate_returns_rmse_and_mae(fitted_model):
    test_ratings = pd.DataFrame({
        "userId":  [1, 3],
        "movieId": [3, 2],
        "rating":  [3.0, 3.5],
    })
    metrics = fitted_model.evaluate(test_ratings)
    assert set(metrics.keys()) == {"rmse", "mae", "n_predictions"}
    assert metrics["n_predictions"] == 2
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
