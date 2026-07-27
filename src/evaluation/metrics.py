"""Evaluation metrics for the movie recommendation system."""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def rmse(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute Root Mean Square Error (RMSE) between true and predicted ratings.

    RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Args:
        y_true: Array-like of ground-truth rating values.
        y_pred: Array-like of predicted rating values, same length as y_true.

    Returns:
        RMSE as a non-negative float. A value of 0.0 indicates a perfect fit.

    Raises:
        ValueError: If either input is empty or the lengths differ.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape} "
            f"but y_pred has shape {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute Mean Absolute Error (MAE) between true and predicted ratings.

    MAE = mean( |y_true - y_pred| )

    Args:
        y_true: Array-like of ground-truth rating values.
        y_pred: Array-like of predicted rating values, same length as y_true.

    Returns:
        MAE as a non-negative float. A value of 0.0 indicates a perfect fit.

    Raises:
        ValueError: If either input is empty or the lengths differ.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape} "
            f"but y_pred has shape {y_pred.shape}."
        )

    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Compute Precision@K for a single user's ranked recommendation list.

    Precision@K = (number of relevant items in the top-k recommendations)
                  / min(k, number of recommendations)

    Args:
        recommended: Ordered list of recommended movieIds, most confident first.
        relevant: List of movieIds the user actually rated at or above the
            relevance threshold.
        k: Cutoff rank to evaluate.

    Returns:
        Precision@K as a float in [0.0, 1.0]. Returns 0.0 if k is 0 or
        recommended is empty.
    """
    if k == 0 or not recommended:
        return 0.0

    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)

    denominator = k if len(top_k) >= k else len(top_k)
    return hits / denominator


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Compute Recall@K for a single user's ranked recommendation list.

    Recall@K = (number of relevant items in the top-k recommendations)
               / (total number of relevant items)

    Args:
        recommended: Ordered list of recommended movieIds, most confident first.
        relevant: List of movieIds the user actually rated at or above the
            relevance threshold.
        k: Cutoff rank to evaluate.

    Returns:
        Recall@K as a float in [0.0, 1.0]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)

    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """Compute the F1 score (harmonic mean) from a precision and recall value.

    Args:
        precision: Precision value, typically Precision@K.
        recall: Recall value, typically Recall@K.

    Returns:
        The harmonic mean of precision and recall. Returns 0.0 if both
        precision and recall are 0.0.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_ranking(
    model: Any,
    test_ratings: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    relevance_threshold: float = 4.0,
    n_candidates: int = 50,
) -> Dict[int, Dict[str, float]]:
    """Evaluate a fitted recommender's ranking quality using Precision/Recall/F1@K.

    For each user in the test set, the items rated at or above
    relevance_threshold are treated as "relevant". The model's top
    n_candidates recommendations for that user are then compared against
    the relevant set at each cutoff in k_values.

    Args:
        model: A fitted recommender exposing `_users` (array-like of known
            userIds) and `recommend(user_id, n) -> List[Tuple[int, float]]`.
        test_ratings: DataFrame with columns userId, movieId, rating.
        k_values: List of cutoff ranks to evaluate.
        relevance_threshold: Minimum rating for an item to count as relevant.
        n_candidates: Number of recommendations to request per user. Should
            be >= max(k_values) so every cutoff has enough candidates.

    Returns:
        Dict mapping each k in k_values to a dict with keys 'precision',
        'recall', and 'f1', averaged across all evaluated users.
    """
    relevant_by_user: Dict[int, List[int]] = (
        test_ratings[test_ratings["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .apply(list)
        .to_dict()
    )

    precisions: Dict[int, List[float]] = {k: [] for k in k_values}
    recalls: Dict[int, List[float]] = {k: [] for k in k_values}

    for user_id, relevant in relevant_by_user.items():
        if not relevant or user_id not in model._users:
            continue

        try:
            recommendations = model.recommend(user_id, n=n_candidates)
        except ValueError:
            continue

        recommended_ids = [movie_id for movie_id, _ in recommendations]

        for k in k_values:
            precisions[k].append(precision_at_k(recommended_ids, relevant, k))
            recalls[k].append(recall_at_k(recommended_ids, relevant, k))

    results: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        avg_precision = float(np.mean(precisions[k])) if precisions[k] else 0.0
        avg_recall = float(np.mean(recalls[k])) if recalls[k] else 0.0
        results[k] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1_at_k(avg_precision, avg_recall),
        }

    return results


if __name__ == "__main__":
    y_true = [4.0, 3.0, 5.0, 2.0, 1.0]
    y_pred = [3.5, 3.0, 4.5, 2.5, 1.5]

    print("Smoke test — metrics.py")
    print(f"  y_true : {y_true}")
    print(f"  y_pred : {y_pred}")
    print(f"  RMSE   : {rmse(y_true, y_pred):.4f}")
    print(f"  MAE    : {mae(y_true, y_pred):.4f}")
