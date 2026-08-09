"""Non-personalised baseline recommenders: global mean, user mean, item mean.

These exist to prove that the collaborative-filtering models (UserBasedCF,
ItemBasedCF) add real predictive/ranking value over trivial baselines, not
just to provide plausible-looking numbers.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _rank_candidates(
    score_fn: Callable[[int], float],
    candidate_movie_ids: List[int],
    n: int,
) -> List[Tuple[int, float]]:
    """Score and rank candidate movies, breaking ties by ascending movieId.

    Shared by all three baseline classes' recommend() methods, since their
    ranking logic (score every unrated candidate, sort, truncate) is
    identical and only the scoring function differs.

    Args:
        score_fn: Callable mapping a movie_id to its predicted score.
        candidate_movie_ids: Candidate movieIds to score.
        n: Number of top results to return.

    Returns:
        List of (movie_id, score) tuples sorted by score descending, ties
        broken by ascending movieId for reproducibility (mirrors the
        movieId tie-break tail used in UserBasedCF.recommend()), length
        at most n.
    """
    scored = [(movie_id, score_fn(movie_id)) for movie_id in candidate_movie_ids]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:n]


class GlobalMeanBaseline:
    """Non-personalised baseline: always predicts the global mean rating.

    The simplest possible sanity-check baseline. Any model that can't beat
    this has learned nothing useful.
    """

    def __init__(self) -> None:
        self._global_mean: Optional[float] = None
        self._users: Optional[np.ndarray] = None
        self._movies: Optional[np.ndarray] = None
        self._rated_movies_by_user: Optional[Dict[int, set]] = None

    def fit(self, ratings: pd.DataFrame) -> "GlobalMeanBaseline":
        """Fit by computing the mean of all ratings in the training set.

        Args:
            ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            self, to allow method chaining.
        """
        self._global_mean = float(ratings["rating"].mean())
        self._users = ratings["userId"].unique()
        self._movies = ratings["movieId"].unique()
        self._rated_movies_by_user = (
            ratings.groupby("userId")["movieId"].apply(set).to_dict()
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the global mean rating, regardless of user or movie.

        Args:
            user_id: The target user's identifier (unused).
            movie_id: The target movie's identifier (unused).

        Returns:
            The global mean rating from the training set.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._global_mean is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._global_mean

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommend top-N unrated movies for a user.

        Every candidate ties at the global mean, so all ordering comes
        from the ascending-movieId tie-break.

        Args:
            user_id: The target user's identifier.
            n: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_rating) tuples, length at most n.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id is not in the training data.
        """
        if self._users is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._users:
            raise ValueError(f"user_id {user_id} not found in training data.")

        rated = self._rated_movies_by_user.get(user_id, set())
        candidates = [m for m in self._movies if m not in rated]
        return _rank_candidates(lambda m: self.predict(user_id, m), candidates, n)

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate the model on a test set using RMSE and MAE.

        Args:
            test_ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            dict with keys 'rmse', 'mae', 'n_predictions'.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._global_mean is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        from src.evaluation.metrics import mae, rmse

        y_true = test_ratings["rating"].tolist()
        y_pred = [self._global_mean] * len(y_true)

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(y_true),
        }


class UserMeanBaseline:
    """Non-personalised baseline: predicts each user's own mean rating.

    Ignores movie identity entirely. Falls back to the global mean for
    users not seen during training.
    """

    def __init__(self) -> None:
        self._global_mean: Optional[float] = None
        self._user_means: Optional[pd.Series] = None
        self._users: Optional[np.ndarray] = None
        self._movies: Optional[np.ndarray] = None
        self._rated_movies_by_user: Optional[Dict[int, set]] = None

    def fit(self, ratings: pd.DataFrame) -> "UserMeanBaseline":
        """Fit by computing each user's mean rating, plus a global fallback.

        Args:
            ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            self, to allow method chaining.
        """
        self._global_mean = float(ratings["rating"].mean())
        self._user_means = ratings.groupby("userId")["rating"].mean()
        self._users = ratings["userId"].unique()
        self._movies = ratings["movieId"].unique()
        self._rated_movies_by_user = (
            ratings.groupby("userId")["movieId"].apply(set).to_dict()
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the target user's mean rating, ignoring movie_id.

        Args:
            user_id: The target user's identifier.
            movie_id: The target movie's identifier (unused).

        Returns:
            The user's mean rating, or the global mean if the user was
            not seen during training.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._user_means is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._user_means.index:
            return self._global_mean
        return float(self._user_means[user_id])

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommend top-N unrated movies for a user.

        Every candidate ties at the user's mean rating, so all ordering
        comes from the ascending-movieId tie-break.

        Args:
            user_id: The target user's identifier.
            n: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_rating) tuples, length at most n.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id is not in the training data.
        """
        if self._users is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._users:
            raise ValueError(f"user_id {user_id} not found in training data.")

        rated = self._rated_movies_by_user.get(user_id, set())
        candidates = [m for m in self._movies if m not in rated]
        return _rank_candidates(lambda m: self.predict(user_id, m), candidates, n)

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate the model on a test set using RMSE and MAE.

        Args:
            test_ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            dict with keys 'rmse', 'mae', 'n_predictions'.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._user_means is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        from src.evaluation.metrics import mae, rmse

        y_true, y_pred = [], []
        for _, row in test_ratings.iterrows():
            y_true.append(row["rating"])
            y_pred.append(self.predict(int(row["userId"]), int(row["movieId"])))

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(y_true),
        }


class ItemMeanBaseline:
    """Non-personalised baseline: predicts each movie's own mean rating.

    Ignores user identity entirely. Falls back to the global mean for
    movies not seen during training.
    """

    def __init__(self) -> None:
        self._global_mean: Optional[float] = None
        self._item_means: Optional[pd.Series] = None
        self._users: Optional[np.ndarray] = None
        self._movies: Optional[np.ndarray] = None
        self._rated_movies_by_user: Optional[Dict[int, set]] = None

    def fit(self, ratings: pd.DataFrame) -> "ItemMeanBaseline":
        """Fit by computing each movie's mean rating, plus a global fallback.

        Args:
            ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            self, to allow method chaining.
        """
        self._global_mean = float(ratings["rating"].mean())
        self._item_means = ratings.groupby("movieId")["rating"].mean()
        self._users = ratings["userId"].unique()
        self._movies = ratings["movieId"].unique()
        self._rated_movies_by_user = (
            ratings.groupby("userId")["movieId"].apply(set).to_dict()
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the target movie's mean rating, ignoring user_id.

        Args:
            user_id: The target user's identifier (unused).
            movie_id: The target movie's identifier.

        Returns:
            The movie's mean rating, or the global mean if the movie was
            not seen during training.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._item_means is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if movie_id not in self._item_means.index:
            return self._global_mean
        return float(self._item_means[movie_id])

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommend top-N unrated movies for a user.

        Candidates are sorted by their item mean descending, ties broken
        by ascending movieId.

        Args:
            user_id: The target user's identifier.
            n: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_rating) tuples, length at most n.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id is not in the training data.
        """
        if self._users is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._users:
            raise ValueError(f"user_id {user_id} not found in training data.")

        rated = self._rated_movies_by_user.get(user_id, set())
        candidates = [m for m in self._movies if m not in rated]
        return _rank_candidates(lambda m: self.predict(user_id, m), candidates, n)

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate the model on a test set using RMSE and MAE.

        Args:
            test_ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            dict with keys 'rmse', 'mae', 'n_predictions'.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._item_means is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        from src.evaluation.metrics import mae, rmse

        y_true, y_pred = [], []
        for _, row in test_ratings.iterrows():
            y_true.append(row["rating"])
            y_pred.append(self.predict(int(row["userId"]), int(row["movieId"])))

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(y_true),
        }
