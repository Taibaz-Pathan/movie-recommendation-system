"""Data loading utilities for the MovieLens dataset."""

import os

import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw", "ml-latest-small")


def load_ratings(data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and return the ratings DataFrame from ratings.csv.

    Converts the unix timestamp column to a human-readable datetime column.

    Args:
        data_dir: Path to the directory containing ratings.csv.

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp (datetime).

    Raises:
        FileNotFoundError: If ratings.csv is not found in data_dir.
    """
    filepath = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"ratings.csv not found at '{filepath}'. "
            "Please download the MovieLens dataset and place it in data/raw/ml-latest-small/."
        )
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies(data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and return the movies DataFrame from movies.csv.

    Args:
        data_dir: Path to the directory containing movies.csv.

    Returns:
        DataFrame with columns: movieId, title, genres.

    Raises:
        FileNotFoundError: If movies.csv is not found in data_dir.
    """
    filepath = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"movies.csv not found at '{filepath}'. "
            "Please download the MovieLens dataset and place it in data/raw/ml-latest-small/."
        )
    return pd.read_csv(filepath)


def load_all(data_dir: str = RAW_DATA_DIR) -> dict:
    """Load all core dataset files and return them as a dictionary.

    Args:
        data_dir: Path to the directory containing the dataset CSV files.

    Returns:
        Dictionary with keys:
            - 'ratings': pd.DataFrame of user ratings
            - 'movies':  pd.DataFrame of movie metadata
    """
    return {
        "ratings": load_ratings(data_dir),
        "movies": load_movies(data_dir),
    }


def dataset_summary(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """Print a descriptive summary of the loaded dataset.

    Displays shape, unique counts, rating statistics, date range,
    and matrix sparsity.

    Args:
        ratings: DataFrame returned by load_ratings().
        movies:  DataFrame returned by load_movies().
    """
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    max_possible = n_users * n_movies
    sparsity = (1 - n_ratings / max_possible) * 100

    print("=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"Ratings shape    : {ratings.shape}")
    print(f"Movies shape     : {movies.shape}")
    print(f"Unique users     : {n_users}")
    print(f"Unique movies    : {n_movies}")
    print(f"Rating range     : {ratings['rating'].min()} – {ratings['rating'].max()}")
    print(f"Rating mean      : {ratings['rating'].mean():.4f}")
    print(
        f"Date range       : {ratings['timestamp'].min().date()} "
        f"to {ratings['timestamp'].max().date()}"
    )
    print(f"Matrix sparsity  : {sparsity:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    data = load_all()
    ratings = data["ratings"]
    movies = data["movies"]

    dataset_summary(ratings, movies)

    print("\nRatings (head):")
    print(ratings.head())

    print("\nMovies (head):")
    print(movies.head())
