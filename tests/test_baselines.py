"""Unit tests for src/models/baselines.py."""

import pandas as pd
import pytest

from src.models.baselines import GlobalMeanBaseline, ItemMeanBaseline, UserMeanBaseline


# --- shared fixture ---

@pytest.fixture()
def sample_ratings():
    """Small synthetic ratings DataFrame: 3 users, 4 movies."""
    return pd.DataFrame({
        "userId":  [1, 1, 2, 2, 2, 2, 3, 3, 3],
        "movieId": [1, 2, 1, 2, 3, 4, 1, 3, 4],
        "rating":  [5.0, 4.0, 4.0, 5.0, 2.0, 1.0, 1.0, 5.0, 4.0],
    })


# --- GlobalMeanBaseline ---

def test_global_fit_returns_self(sample_ratings):
    model = GlobalMeanBaseline()
    result = model.fit(sample_ratings)
    assert result is model


def test_global_predict_returns_float(sample_ratings):
    model = GlobalMeanBaseline().fit(sample_ratings)
    assert isinstance(model.predict(1, 3), float)


def test_global_predict_same_regardless_of_user_or_movie(sample_ratings):
    model = GlobalMeanBaseline().fit(sample_ratings)
    p1 = model.predict(1, 3)
    p2 = model.predict(2, 4)
    p3 = model.predict(3, 1)
    assert p1 == p2 == p3 == pytest.approx(sample_ratings["rating"].mean())


def test_global_recommend_only_unrated_movies(sample_ratings):
    model = GlobalMeanBaseline().fit(sample_ratings)
    recs = model.recommend(1, n=10)
    recommended_ids = {movie_id for movie_id, _ in recs}
    assert recommended_ids == {3, 4}


def test_global_recommend_unknown_user_raises(sample_ratings):
    model = GlobalMeanBaseline().fit(sample_ratings)
    with pytest.raises(ValueError):
        model.recommend(999, n=10)


def test_global_evaluate_returns_rmse_and_mae(sample_ratings):
    model = GlobalMeanBaseline().fit(sample_ratings)
    metrics = model.evaluate(sample_ratings)
    assert "rmse" in metrics
    assert "mae" in metrics


# --- UserMeanBaseline ---

def test_user_mean_fit_returns_self(sample_ratings):
    model = UserMeanBaseline()
    result = model.fit(sample_ratings)
    assert result is model


def test_user_mean_predict_returns_float(sample_ratings):
    model = UserMeanBaseline().fit(sample_ratings)
    assert isinstance(model.predict(1, 3), float)


def test_user_mean_predict_varies_by_user_constant_across_movies(sample_ratings):
    model = UserMeanBaseline().fit(sample_ratings)

    # user 1's mean is constant regardless of which movie is asked about
    assert model.predict(1, 1) == model.predict(1, 3) == model.predict(1, 4)

    # but differs from user 2's mean
    assert model.predict(1, 1) != model.predict(2, 1)


def test_user_mean_recommend_only_unrated_movies(sample_ratings):
    model = UserMeanBaseline().fit(sample_ratings)
    recs = model.recommend(1, n=10)
    recommended_ids = {movie_id for movie_id, _ in recs}
    assert recommended_ids == {3, 4}


def test_user_mean_recommend_unknown_user_raises(sample_ratings):
    model = UserMeanBaseline().fit(sample_ratings)
    with pytest.raises(ValueError):
        model.recommend(999, n=10)


def test_user_mean_evaluate_returns_rmse_and_mae(sample_ratings):
    model = UserMeanBaseline().fit(sample_ratings)
    metrics = model.evaluate(sample_ratings)
    assert "rmse" in metrics
    assert "mae" in metrics


# --- ItemMeanBaseline ---

def test_item_mean_fit_returns_self(sample_ratings):
    model = ItemMeanBaseline()
    result = model.fit(sample_ratings)
    assert result is model


def test_item_mean_predict_returns_float(sample_ratings):
    model = ItemMeanBaseline().fit(sample_ratings)
    assert isinstance(model.predict(1, 1), float)


def test_item_mean_predict_varies_by_movie_constant_across_users(sample_ratings):
    model = ItemMeanBaseline().fit(sample_ratings)

    # movie 1's mean is constant regardless of which user is asked about
    assert model.predict(1, 1) == model.predict(2, 1) == model.predict(3, 1)

    # but differs from movie 3's mean
    assert model.predict(1, 1) != model.predict(1, 3)


def test_item_mean_recommend_only_unrated_movies(sample_ratings):
    model = ItemMeanBaseline().fit(sample_ratings)
    recs = model.recommend(1, n=10)
    recommended_ids = {movie_id for movie_id, _ in recs}
    assert recommended_ids == {3, 4}


def test_item_mean_recommend_unknown_user_raises(sample_ratings):
    model = ItemMeanBaseline().fit(sample_ratings)
    with pytest.raises(ValueError):
        model.recommend(999, n=10)


def test_item_mean_evaluate_returns_rmse_and_mae(sample_ratings):
    model = ItemMeanBaseline().fit(sample_ratings)
    metrics = model.evaluate(sample_ratings)
    assert "rmse" in metrics
    assert "mae" in metrics
