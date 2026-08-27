"""SVD-based collaborative filtering model, wrapping scikit-surprise's SVD."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader


class SVDModel:
    """Matrix-factorization recommender wrapping surprise.SVD.

    Provides the same fit/predict/recommend/evaluate interface as
    UserBasedCF, ItemBasedCF, and the baseline models, so it is a
    drop-in replacement anywhere those are used (e.g. evaluate_ranking()).

    Args:
        n_factors: Number of latent factors.
        n_epochs: Number of SGD training epochs.
        random_state: Random seed for reproducibility.
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, random_state: int = 42) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state

        self._algo: Optional[SVD] = None
        self._users: Optional[np.ndarray] = None
        self._movies: Optional[np.ndarray] = None
        self._rated_movies_by_user: Optional[Dict[int, set]] = None

    def fit(self, ratings: pd.DataFrame) -> "SVDModel":
        """Fit the SVD model on training ratings.

        Args:
            ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            self, to allow method chaining.
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        trainset = dataset.build_full_trainset()

        self._algo = SVD(
            n_factors=self.n_factors, n_epochs=self.n_epochs, random_state=self.random_state
        )
        self._algo.fit(trainset)

        self._users = ratings["userId"].unique()
        self._movies = ratings["movieId"].unique()
        self._rated_movies_by_user = ratings.groupby("userId")["movieId"].apply(set).to_dict()

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie.

        Falls back to surprise's own default estimate (the training set's
        global mean) for unseen users/movies, rather than raising, matching
        surprise's native behaviour.

        Args:
            user_id: The target user's identifier.
            movie_id: The target movie's identifier.

        Returns:
            Predicted rating clipped to [0.5, 5.0].

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._algo is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        prediction = self._algo.predict(user_id, movie_id)
        return float(np.clip(prediction.est, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Generate top-N movie recommendations for a user.

        Only movies the user has not yet rated are considered. Ties are
        broken deterministically by ascending movieId.

        Args:
            user_id: The target user's identifier.
            n: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_rating) tuples sorted by predicted
            rating descending (ties broken by ascending movieId), length
            at most n.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id is not in the training data.
        """
        if self._algo is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._users:
            raise ValueError(f"user_id {user_id} not found in training data.")

        rated = self._rated_movies_by_user.get(user_id, set())
        candidates = [movie_id for movie_id in self._movies if movie_id not in rated]

        predictions = [(movie_id, self.predict(user_id, movie_id)) for movie_id in candidates]
        predictions.sort(key=lambda x: (-x[1], x[0]))
        return predictions[:n]

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate the model on a test set using RMSE and MAE.

        Args:
            test_ratings: DataFrame with columns userId, movieId, rating.

        Returns:
            dict with keys 'rmse', 'mae', 'n_predictions'.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._algo is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        from src.evaluation.metrics import mae, rmse

        y_true = test_ratings["rating"].tolist()
        y_pred = [
            self.predict(int(row["userId"]), int(row["movieId"]))
            for _, row in test_ratings.iterrows()
        ]

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(y_true),
        }
