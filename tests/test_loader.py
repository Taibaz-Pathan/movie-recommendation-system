"""Unit tests for src/data/loader.py."""

import io
import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.loader import load_movies, load_ratings


SAMPLE_RATINGS_CSV = """\
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
2,1,3.0,964982224
2,5,5.0,964983815
3,2,2.5,964981179
"""

SAMPLE_MOVIES_CSV = """\
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
5,Father of the Bride Part II (1995),Comedy
"""


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a temporary directory with sample ratings.csv and movies.csv."""
    (tmp_path / "ratings.csv").write_text(SAMPLE_RATINGS_CSV)
    (tmp_path / "movies.csv").write_text(SAMPLE_MOVIES_CSV)
    return str(tmp_path)


def test_load_ratings_returns_dataframe(tmp_data_dir):
    """load_ratings() should return a pandas DataFrame."""
    df = load_ratings(tmp_data_dir)
    assert isinstance(df, pd.DataFrame)


def test_load_movies_returns_dataframe(tmp_data_dir):
    """load_movies() should return a pandas DataFrame."""
    df = load_movies(tmp_data_dir)
    assert isinstance(df, pd.DataFrame)


def test_ratings_columns_exist(tmp_data_dir):
    """Loaded ratings DataFrame must contain the expected columns."""
    df = load_ratings(tmp_data_dir)
    for col in ("userId", "movieId", "rating", "timestamp"):
        assert col in df.columns, f"Expected column '{col}' not found in ratings DataFrame."


def test_rating_range(tmp_data_dir):
    """All rating values must fall within the valid MovieLens range [0.5, 5.0]."""
    df = load_ratings(tmp_data_dir)
    assert df["rating"].min() >= 0.5, "Rating below minimum allowed value of 0.5."
    assert df["rating"].max() <= 5.0, "Rating above maximum allowed value of 5.0."


def test_timestamp_converted_to_datetime(tmp_data_dir):
    """The timestamp column should be converted from unix int to datetime."""
    df = load_ratings(tmp_data_dir)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), (
        "Expected 'timestamp' column to be datetime dtype after loading."
    )


def test_load_ratings_raises_on_missing_file(tmp_path):
    """load_ratings() should raise FileNotFoundError when the file is absent."""
    with pytest.raises(FileNotFoundError):
        load_ratings(str(tmp_path))


def test_load_movies_raises_on_missing_file(tmp_path):
    """load_movies() should raise FileNotFoundError when the file is absent."""
    with pytest.raises(FileNotFoundError):
        load_movies(str(tmp_path))
