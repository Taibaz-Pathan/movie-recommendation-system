"""Item-Based Collaborative Filtering model."""

import numpy as np
import pandas as pd


class ItemBasedCF:
    """Item-Based Collaborative Filtering recommender.

    Predicts ratings and generates top-N recommendations by finding
    the k most similar items (neighbours) to a target movie and
    computing a similarity-weighted average of the user's own ratings
    on those neighbour items.

    IBCF is generally faster at prediction time than UBCF because the
    item-item similarity matrix is pre-computed once during fit().

    Args:
        k: Number of nearest item neighbours to consider.
        similarity: Similarity measure to use ('cosine' or 'pearson').
        min_support: Minimum number of users who must have co-rated two
            items for their similarity to be considered valid (Pearson only).
        min_k: Minimum number of valid item neighbours required to make a
            prediction. Falls back to the user's mean rating otherwise.
    """

    def __init__(
        self,
        k: int = 20,
        similarity: str = "cosine",
        min_support: int = 5,
        min_k: int = 2,
    ) -> None:
        self.k = k
        self.similarity = similarity
        self.min_support = min_support
        self.min_k = min_k
        self._ratings_matrix = None
        self._item_similarity = None
        self._user_means = None

    def fit(self, ratings_matrix: pd.DataFrame) -> "ItemBasedCF":
        """Fit the model by pre-computing the item-item similarity matrix.

        For cosine similarity, item vectors are L2-normalised before the
        dot product so that rating scale differences don't dominate.
        For Pearson, pandas corr() is used with min_periods to enforce
        the min_support threshold.

        Args:
            ratings_matrix: DataFrame with users as rows, items as columns,
                and ratings as values. Missing ratings must be NaN.

        Returns:
            self, to allow method chaining.

        Raises:
            ValueError: If an unsupported similarity metric is specified.
        """
        self._ratings_matrix = ratings_matrix
        self._user_means = ratings_matrix.mean(axis=1)

        if self.similarity == "cosine":
            # Each column is an item vector over users; fill NaN with 0
            filled = ratings_matrix.fillna(0).values.astype(float)
            # Normalise each item vector (column) by its L2 norm
            norms = np.linalg.norm(filled, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            normed = filled / norms
            # Item-item cosine similarity: (n_items x n_items)
            sim = normed.T @ normed
            self._item_similarity = pd.DataFrame(
                sim,
                index=ratings_matrix.columns,
                columns=ratings_matrix.columns,
            )

        elif self.similarity == "pearson":
            # corr() on columns gives item-item Pearson correlation.
            # min_periods enforces min_support: NaN when fewer users co-rated.
            self._item_similarity = ratings_matrix.corr(
                method="pearson", min_periods=self.min_support
            )

        else:
            raise ValueError(
                f"Unknown similarity metric: '{self.similarity}'. "
                "Choose 'cosine' or 'pearson'."
            )

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie.

        Finds the k items most similar to the target movie that the user
        has already rated, then computes a similarity-weighted average:

            pred(u, i) = sum(sim(i, j) * r_uj) / sum(|sim(i, j)|)

        Falls back to the user's mean rating when fewer than min_k valid
        item neighbours are found.

        Args:
            user_id: The target user's identifier.
            movie_id: The target movie's identifier.

        Returns:
            Predicted rating clipped to [0.5, 5.0].

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id or movie_id is not in the training data.
        """
        if self._ratings_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._ratings_matrix.index:
            raise ValueError(f"user_id {user_id} not found in training data.")
        if movie_id not in self._ratings_matrix.columns:
            raise ValueError(f"movie_id {movie_id} not found in training data.")

        user_mean = self._user_means[user_id]

        # Similarities from target movie to all other items (drop self, drop NaN)
        sim_scores = self._item_similarity[movie_id].drop(index=movie_id).dropna()

        # Keep only items that this user has actually rated
        user_rated = self._ratings_matrix.loc[user_id].dropna().index
        common = sim_scores.index.intersection(user_rated)

        if len(common) < self.min_k:
            return float(np.clip(user_mean, 0.5, 5.0))

        sim_scores = sim_scores[common]

        # Top-k by absolute similarity, then require positive similarity
        top_k_idx = sim_scores.abs().nlargest(self.k).index
        sim_scores = sim_scores[top_k_idx]
        sim_scores = sim_scores[sim_scores > 0]

        if len(sim_scores) < self.min_k:
            return float(np.clip(user_mean, 0.5, 5.0))

        user_ratings = self._ratings_matrix.loc[user_id, sim_scores.index]

        numerator = (sim_scores * user_ratings).sum()
        denominator = sim_scores.abs().sum()

        prediction = numerator / denominator
        return float(np.clip(prediction, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10, min_movie_ratings: int = 20) -> list:
        """Generate top-N movie recommendations for a user.

        Only movies the user has not yet rated are considered. Movies with
        fewer than min_movie_ratings total ratings are excluded to avoid
        recommending obscure titles with little neighbour support.

        Args:
            user_id: The target user's identifier.
            n: Number of recommendations to return.
            min_movie_ratings: Minimum number of users who must have rated
                a movie for it to be eligible for recommendation.

        Returns:
            List of (movie_id, predicted_rating) tuples sorted by
            predicted rating descending, length at most n.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If user_id is not in the training data.
        """
        if self._ratings_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if user_id not in self._ratings_matrix.index:
            raise ValueError(f"user_id {user_id} not found in training data.")

        user_row = self._ratings_matrix.loc[user_id]

        # Popularity filter: only consider movies with enough total ratings
        rating_counts = self._ratings_matrix.count()
        popular_movies = rating_counts[rating_counts >= min_movie_ratings].index
        unrated = user_row[user_row.isna()].index.intersection(popular_movies).tolist()

        predictions = [
            (movie_id, self.predict(user_id, movie_id))
            for movie_id in unrated
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
