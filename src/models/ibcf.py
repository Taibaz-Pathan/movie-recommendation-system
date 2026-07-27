"""
src/models/ibcf.py
------------------
Item-Based Collaborative Filtering model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from src.utils.similarity import cosine_similarity_matrix


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering recommender.

    Uses adjusted cosine similarity (ratings mean-centred per user
    before computing item-item similarity) as recommended by
    Sarwar et al. (2001).

    Parameters
    ----------
    k : int
        Number of nearest neighbour items to use for prediction.
    min_support : int
        Minimum number of users who must have co-rated both items
        for their similarity to be considered valid.
    """

    def __init__(self, k: int = 20, min_support: int = 5) -> None:
        self.k = k
        self.min_support = min_support

        self._matrix: Optional[pd.DataFrame] = None
        self._sim_matrix: Optional[np.ndarray] = None
        self._user_means: Optional[pd.Series] = None
        self._item_means: Optional[pd.Series] = None
        self._users: Optional[np.ndarray] = None
        self._movies: Optional[np.ndarray] = None

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        """
        Fit the model on training ratings.

        Parameters
        ----------
        ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating.

        Returns
        -------
        self
        """
        self._matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self._users = self._matrix.index.to_numpy()
        self._movies = self._matrix.columns.to_numpy()
        self._user_means = self._matrix.mean(axis=1)
        self._item_means = self._matrix.mean(axis=0)

        # Adjusted cosine: mean-centre each USER's row before
        # computing similarity between ITEM columns.
        centred = self._matrix.sub(self._user_means, axis=0)
        centred_np = centred.to_numpy()
        centred_np = np.nan_to_num(centred_np, nan=0.0)

        # Co-rating counts per item pair, to apply min_support
        rated_mask = (~self._matrix.isna()).to_numpy().astype(float)
        co_rating_counts = rated_mask.T @ rated_mask

        # Item-item similarity via cosine on the centred, item-column matrix
        self._sim_matrix = cosine_similarity_matrix(centred_np.T)

        # Zero out pairs below min_support
        self._sim_matrix[co_rating_counts < self.min_support] = 0.0

        return self

    def _get_user_index(self, user_id: int) -> int:
        idx = np.where(self._users == user_id)[0]
        if len(idx) == 0:
            raise ValueError(f"User {user_id} not found in training data.")
        return idx[0]

    def _get_movie_index(self, movie_id: int) -> int:
        idx = np.where(self._movies == movie_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Movie {movie_id} not found in training data.")
        return idx[0]

    def _predict_with_support(self, user_id: int, movie_id: int) -> Tuple[float, int]:
        """
        Predict a rating and report how many valid neighbour items backed it.

        Shared internal implementation for predict() and recommend(), so
        the neighbour-selection logic isn't duplicated and recommend() can
        break ties using the neighbour count without recomputing it.

        Parameters
        ----------
        user_id : int
        movie_id : int

        Returns
        -------
        Tuple of (predicted_rating clipped to [0.5, 5.0], n_neighbours used).
        n_neighbours is 0 when the fallback path (no valid neighbour items) is taken.
        """
        u_idx = self._get_user_index(user_id)
        m_idx = self._get_movie_index(movie_id)

        user_mean = self._user_means.iloc[u_idx]
        user_ratings = self._matrix.iloc[u_idx].to_numpy()

        # Similarities of target item to all other items
        similarities = self._sim_matrix[m_idx].copy()
        similarities[m_idx] = 0.0  # exclude self

        # Only keep items the user has actually rated
        rated_mask = ~np.isnan(user_ratings)
        similarities[~rated_mask] = 0.0

        # Top-K most similar rated items
        top_k_idx = np.argsort(np.abs(similarities))[::-1][: self.k]
        top_k_sims = similarities[top_k_idx]
        top_k_ratings = user_ratings[top_k_idx]

        valid = top_k_sims != 0.0
        if not valid.any():
            return float(np.clip(user_mean, 0.5, 5.0)), 0

        top_k_sims = top_k_sims[valid]
        top_k_ratings = top_k_ratings[valid]
        top_k_item_means = self._item_means.iloc[top_k_idx].to_numpy()[valid]

        numerator = np.sum(top_k_sims * (top_k_ratings - top_k_item_means))
        denominator = np.sum(np.abs(top_k_sims))

        target_item_mean = self._item_means.iloc[m_idx]
        prediction = target_item_mean + numerator / denominator
        return float(np.clip(prediction, 0.5, 5.0)), int(valid.sum())

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating for a (user, movie) pair.

        Parameters
        ----------
        user_id : int
        movie_id : int

        Returns
        -------
        float
            Predicted rating, clipped to [0.5, 5.0].
            Falls back to the user's mean rating if no valid
            neighbour items are found.
        """
        prediction, _ = self._predict_with_support(user_id, movie_id)
        return prediction

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend top-N unseen movies for a user.

        Candidates are sorted by predicted rating descending. Ties (e.g.
        multiple predictions clipped to the same ceiling/floor value) are
        broken deterministically by preferring the prediction backed by
        more valid neighbour items, then by ascending movieId for
        reproducibility.

        Parameters
        ----------
        user_id : int
        n : int

        Returns
        -------
        List of (movieId, predicted_rating) tuples, sorted descending
        (ties broken as above).
        """
        u_idx = self._get_user_index(user_id)
        user_row = self._matrix.iloc[u_idx]
        unrated_movies = user_row[user_row.isna()].index.tolist()

        predictions = []
        for movie_id in unrated_movies:
            try:
                pred, n_neighbours = self._predict_with_support(user_id, movie_id)
                predictions.append((movie_id, pred, n_neighbours))
            except ValueError:
                continue

        predictions.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(movie_id, pred) for movie_id, pred, _ in predictions[:n]]

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """
        Evaluate the model on a test set using RMSE and MAE.

        Parameters
        ----------
        test_ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating.

        Returns
        -------
        dict with keys 'rmse', 'mae', 'n_predictions'.
        """
        from src.evaluation.metrics import rmse, mae

        y_true, y_pred = [], []

        for _, row in test_ratings.iterrows():
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])

            if (user_id not in self._users) or (movie_id not in self._movies):
                continue

            try:
                pred = self.predict(user_id, movie_id)
                y_true.append(row["rating"])
                y_pred.append(pred)
            except ValueError:
                continue

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(y_true),
        }
