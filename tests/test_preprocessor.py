"""Unit tests for src/data/preprocessor.py."""

import pandas as pd
import numpy as np
import pytest

from src.data.preprocessor import (
    filter_ratings,
    build_user_item_matrix,
    train_test_split_per_user,
    save_splits,
)


# --- shared fixture ---

@pytest.fixture()
def sample_ratings():
    """Small synthetic ratings DataFrame for testing."""
    return pd.DataFrame({
        "userId":  [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "movieId": [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4],
        "rating":  [4.0, 3.0, 5.0, 2.0, 4.0, 3.5, 5.0, 4.0, 2.5, 3.0, 4.5, 1.0],
        "timestamp": pd.to_datetime(["2020-01-01"] * 12),
    })


# --- filter_ratings ---

def test_filter_returns_dataframe(sample_ratings):
    result = filter_ratings(sample_ratings, min_user_ratings=1, min_movie_ratings=1)
    assert isinstance(result, pd.DataFrame)


def test_filter_removes_sparse_users(sample_ratings):
    # require at least 4 ratings per user — none qualify in sample (all have 3)
    result = filter_ratings(sample_ratings, min_user_ratings=4, min_movie_ratings=1)
    assert len(result) == 0


def test_filter_removes_sparse_movies(sample_ratings):
    # require at least 4 ratings per movie — none qualify in sample (all have 3)
    result = filter_ratings(sample_ratings, min_user_ratings=1, min_movie_ratings=4)
    assert len(result) == 0


def test_filter_keeps_all_when_threshold_low(sample_ratings):
    result = filter_ratings(sample_ratings, min_user_ratings=1, min_movie_ratings=1)
    assert len(result) == len(sample_ratings)


def test_filter_resets_index(sample_ratings):
    result = filter_ratings(sample_ratings, min_user_ratings=1, min_movie_ratings=1)
    assert list(result.index) == list(range(len(result)))


# --- build_user_item_matrix ---

def test_matrix_shape(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    n_users  = sample_ratings["userId"].nunique()
    n_movies = sample_ratings["movieId"].nunique()
    assert matrix.shape == (n_users, n_movies)


def test_matrix_index_is_user_id(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    assert list(matrix.index) == sorted(sample_ratings["userId"].unique())


def test_matrix_columns_are_movie_ids(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    assert list(matrix.columns) == sorted(sample_ratings["movieId"].unique())


def test_matrix_values_are_ratings(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    # user 1 rated movie 1 with 4.0
    assert matrix.loc[1, 1] == 4.0


def test_matrix_unrated_is_nan(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    # user 1 did not rate movie 4
    assert pd.isna(matrix.loc[1, 4])


# --- train_test_split_per_user ---

def test_split_returns_two_dataframes(sample_ratings):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_no_overlap(sample_ratings):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    train_pairs = set(zip(train["userId"], train["movieId"]))
    test_pairs  = set(zip(test["userId"],  test["movieId"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_split_covers_all_ratings(sample_ratings):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    assert len(train) + len(test) == len(sample_ratings)


def test_split_all_users_in_train(sample_ratings):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    assert set(train["userId"].unique()) == set(sample_ratings["userId"].unique())


def test_split_all_users_in_test(sample_ratings):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    assert set(test["userId"].unique()) == set(sample_ratings["userId"].unique())


def test_split_reproducible(sample_ratings):
    train1, test1 = train_test_split_per_user(sample_ratings, random_state=42)
    train2, test2 = train_test_split_per_user(sample_ratings, random_state=42)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_raises_on_invalid_test_size(sample_ratings):
    with pytest.raises(ValueError):
        train_test_split_per_user(sample_ratings, test_size=1.5)


# --- save_splits ---

def test_save_creates_files(sample_ratings, tmp_path):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    save_splits(train, test, output_dir=str(tmp_path))
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_save_roundtrip(sample_ratings, tmp_path):
    train, test = train_test_split_per_user(sample_ratings, test_size=0.2)
    save_splits(train, test, output_dir=str(tmp_path))
    loaded_train = pd.read_csv(tmp_path / "train.csv")
    assert len(loaded_train) == len(train)
