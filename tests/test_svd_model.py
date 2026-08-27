"""Unit tests for src/models/svd_model.py."""

import pandas as pd
import pytest

from src.models.svd_model import SVDModel


# --- shared fixture ---


@pytest.fixture()
def sample_ratings():
    """Small synthetic ratings DataFrame: 3 users, 4 movies."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2, 2, 3, 3, 3],
            "movieId": [1, 2, 1, 2, 3, 4, 1, 3, 4],
            "rating": [5.0, 4.0, 4.0, 5.0, 2.0, 1.0, 1.0, 5.0, 4.0],
        }
    )


@pytest.fixture()
def fitted_model(sample_ratings):
    model = SVDModel(n_factors=5, n_epochs=5, random_state=42)
    return model.fit(sample_ratings)


# --- fit ---


def test_fit_returns_self(sample_ratings):
    model = SVDModel(n_factors=5, n_epochs=5, random_state=42)
    result = model.fit(sample_ratings)
    assert result is model


# --- predict ---


def test_predict_returns_float(fitted_model):
    pred = fitted_model.predict(1, 3)
    assert isinstance(pred, float)


def test_predict_clipped_to_valid_range(fitted_model):
    for user_id, movie_id in [(1, 3), (1, 4), (3, 2)]:
        pred = fitted_model.predict(user_id, movie_id)
        assert 0.5 <= pred <= 5.0


# --- recommend ---


def test_recommend_only_unrated_movies(fitted_model):
    # user 1 rated movies 1 and 2, so only 3 and 4 are eligible
    recs = fitted_model.recommend(1, n=10)
    recommended_ids = {movie_id for movie_id, _ in recs}
    assert recommended_ids == {3, 4}


def test_recommend_unknown_user_raises(fitted_model):
    with pytest.raises(ValueError):
        fitted_model.recommend(999, n=10)


# --- evaluate ---


def test_evaluate_returns_rmse_and_mae(fitted_model, sample_ratings):
    metrics = fitted_model.evaluate(sample_ratings)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] >= 0
