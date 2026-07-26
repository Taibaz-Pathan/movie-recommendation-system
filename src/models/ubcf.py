"""User-Based Collaborative Filtering model."""

import numpy as np
import pandas as pd


class UserBasedCF:
    """User-Based Collaborative Filtering recommender.

    Predicts ratings and generates top-N recommendations by finding
    the k most similar users (neighbours) to a target user and
    computing a mean-centred weighted average of their ratings.

    Args:
        k: Number of nearest neighbours to consider.
        similarity: Similarity measure to use ('pearson' or 'cosine').
        min_support: Minimum number of co-rated items required for a
            similarity score to be considered valid. Pairs with fewer
            co-ratings are treated as dissimilar (NaN → 0). This prevents
            extreme predictions from sparse user pairs.
        min_k: Minimum number of valid neighbours required to make a
            prediction. Falls back to the user's mean rating otherwise.
    """

    def __init__(
        self,
        k: int = 20,
        similarity: str = "pearson",
        min_support: int = 5,
        min_k: int = 2,
    ) -> None:
        self.k = k
        self.similarity = similarity
        self.min_support = min_support
        self.min_k = min_k
        self._ratings_matrix = None
        self._sim_matrix = None
        self._user_means = None

    def fit(self, ratings_matrix: pd.DataFrame) -> "UserBasedCF":
        """Fit the model by pre-computing the user-user similarity matrix.

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

        if self.similarity == "pearson":
            # min_periods enforces the min_support threshold:
            # similarity is NaN when fewer than min_support items are co-rated.
            self._sim_matrix = ratings_matrix.T.corr(
                method="pearson", min_periods=self.min_support
            )

        elif self.similarity == "cosine":
            filled = ratings_matrix.fillna(0).values.astype(float)
            norms = np.linalg.norm(filled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = filled / norms
            sim = normed @ normed.T
            self._sim_matrix = pd.DataFrame(
                sim,
                index=ratings_matrix.index,
                columns=ratings_matrix.index,
            )

        else:
            raise ValueError(
                f"Unknown similarity metric: '{self.similarity}'. "
                "Choose 'pearson' or 'cosine'."
            )

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie.

        Uses mean-centred weighted average over the k nearest neighbours
        who have rated the target movie:

            pred(u, i) = mean_u
                         + sum(sim(u,v) * (r_vi - mean_v))
                           / sum(|sim(u,v)|)

        Falls back to the user's mean rating when fewer than min_k
        valid neighbours are found.

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

        # Similarities to all other users (drop self, drop NaN from min_support)
        sim_scores = self._sim_matrix[user_id].drop(index=user_id).dropna()

        # Keep only neighbours who have rated the target movie
        rated_by = self._ratings_matrix[movie_id].dropna().index
        common = sim_scores.index.intersection(rated_by)
        if len(common) < self.min_k:
            return float(np.clip(user_mean, 0.5, 5.0))

        sim_scores = sim_scores[common]

        # Top-k by absolute similarity, then require positive correlation
        top_k_idx = sim_scores.abs().nlargest(self.k).index
        sim_scores = sim_scores[top_k_idx]
        sim_scores = sim_scores[sim_scores > 0]

        if len(sim_scores) < self.min_k:
            return float(np.clip(user_mean, 0.5, 5.0))

        neighbour_ratings = self._ratings_matrix.loc[sim_scores.index, movie_id]
        neighbour_means = self._user_means[sim_scores.index]

        numerator = (sim_scores * (neighbour_ratings - neighbour_means)).sum()
        denominator = sim_scores.abs().sum()

        prediction = user_mean + numerator / denominator
        return float(np.clip(prediction, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10, min_movie_ratings: int = 20) -> list:
        """Generate top-N movie recommendations for a user.

        Only movies the user has not yet rated are considered. Movies with
        fewer than min_movie_ratings total ratings are excluded to avoid
        recommending obscure titles predicted from a single neighbour.

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

        # Only consider movies with enough total ratings (popularity filter)
        rating_counts = self._ratings_matrix.count()
        popular_movies = rating_counts[rating_counts >= min_movie_ratings].index
        unrated = user_row[user_row.isna()].index.intersection(popular_movies).tolist()

        predictions = [
            (movie_id, self.predict(user_id, movie_id))
            for movie_id in unrated
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
