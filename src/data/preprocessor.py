"""Data preprocessing utilities: filtering, matrix construction, and train/test splitting."""

import os

import numpy as np
import pandas as pd


def filter_ratings(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 20,
) -> pd.DataFrame:
    """Remove users and movies that have too few ratings.

    Filtering is applied iteratively until convergence because removing
    sparse movies can make some users fall below the threshold and vice versa.

    Args:
        ratings: Raw ratings DataFrame with columns userId, movieId, rating.
        min_user_ratings: Minimum number of ratings a user must have to be kept.
        min_movie_ratings: Minimum number of ratings a movie must have to be kept.

    Returns:
        Filtered DataFrame with the same columns as the input.
    """
    df = ratings.copy()
    prev_shape = None

    while prev_shape != df.shape:
        prev_shape = df.shape

        user_counts = df.groupby("userId")["rating"].transform("count")
        df = df[user_counts >= min_user_ratings]

        movie_counts = df.groupby("movieId")["rating"].transform("count")
        df = df[movie_counts >= min_movie_ratings]

    df = df.reset_index(drop=True)
    print(
        f"After filtering: {df.shape[0]:,} ratings | "
        f"{df['userId'].nunique()} users | "
        f"{df['movieId'].nunique()} movies"
    )
    return df


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a user-item ratings matrix from a ratings DataFrame.

    Rows are users, columns are movies, values are ratings.
    Missing entries (unrated movies) are NaN.

    Args:
        ratings: DataFrame with columns userId, movieId, rating.

    Returns:
        DataFrame of shape (n_users, n_movies) with userId as index
        and movieId as columns.
    """
    matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
    n_users, n_movies = matrix.shape
    sparsity = 1 - matrix.count().sum() / (n_users * n_movies)
    print(
        f"User-item matrix: {n_users} users × {n_movies} movies | "
        f"sparsity: {sparsity * 100:.2f}%"
    )
    return matrix


def train_test_split_per_user(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split ratings into train and test sets on a per-user basis.

    For each user, a random fraction of their ratings is held out as the
    test set. This ensures every user is represented in both splits, which
    is essential for evaluating collaborative filtering models.

    Args:
        ratings: DataFrame with columns userId, movieId, rating, timestamp.
        test_size: Fraction of each user's ratings to hold out for testing.
            Must be between 0 and 1.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df), both with the same columns as the input.

    Raises:
        ValueError: If test_size is not in (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for _, user_df in ratings.groupby("userId"):
        idx = user_df.index.tolist()
        n_test = max(1, round(len(idx) * test_size))
        test_idx = rng.choice(idx, size=n_test, replace=False).tolist()
        train_idx = [i for i in idx if i not in test_idx]
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    train = ratings.loc[train_indices].reset_index(drop=True)
    test = ratings.loc[test_indices].reset_index(drop=True)

    print(
        f"Train: {len(train):,} ratings | Test: {len(test):,} ratings | "
        f"Split: {len(train)/len(ratings)*100:.1f}% / {len(test)/len(ratings)*100:.1f}%"
    )
    return train, test


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = os.path.join("data", "processed"),
) -> None:
    """Save train and test DataFrames as CSV files.

    Args:
        train: Training ratings DataFrame.
        test: Test ratings DataFrame.
        output_dir: Directory where train.csv and test.csv will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    from src.data.loader import load_all

    data = load_all()
    ratings = data["ratings"]

    print("Step 1: Filter ratings")
    filtered = filter_ratings(ratings, min_user_ratings=20, min_movie_ratings=20)

    print("\nStep 2: Train/test split")
    train, test = train_test_split_per_user(filtered, test_size=0.2, random_state=42)

    print("\nStep 3: Build user-item matrix from training data only")
    matrix = build_user_item_matrix(train)

    print("\nStep 4: Save splits")
    save_splits(train, test)

    print("\nDone. Train sample:")
    print(train.head())
    print("\nTest sample:")
    print(test.head())
