"""Helpers that connect the Streamlit demo to the offline evaluation.

The demo's "hits" view uses the same definitions as evaluate_ranking() in
src/evaluation/metrics.py: a held-out test movie is relevant to a user if
their true rating for it is at least the relevance threshold, and a
recommendation counts as a hit if it is one of those movies. Keeping this
logic here (pandas only, no Streamlit) means it can be unit-tested.
"""

from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from src.evaluation.metrics import precision_at_k

RELEVANCE_THRESHOLD = 4.0

# Column names in reports/full_model_comparison_v2.csv -> display names.
COMPARISON_COLUMNS = {
    "model": "Model",
    "rmse": "RMSE",
    "mae": "MAE",
    "precision_10": "Precision@10",
    "recall_10": "Recall@10",
}

MODEL_PREFIXES = ("UBCF", "IBCF", "SVD")


def relevant_test_items(
    test_ratings: pd.DataFrame,
    user_id: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> Set[int]:
    """Return the movieIds a user rated at or above threshold in the test set.

    Args:
        test_ratings: DataFrame with columns userId, movieId, rating.
        user_id: The user to look up.
        threshold: Minimum true rating for a movie to count as relevant.

    Returns:
        Set of movieIds (empty if the user has no relevant test ratings).
    """
    mask = (test_ratings["userId"] == user_id) & (test_ratings["rating"] >= threshold)
    return set(test_ratings.loc[mask, "movieId"].astype(int))


def hit_summary(recommended_ids: Iterable[int], relevant: Set[int]) -> Dict:
    """Compare one recommendation list against a user's relevant test items.

    Args:
        recommended_ids: Recommended movieIds, most confident first.
        relevant: MovieIds the user rated at or above the relevance threshold.

    Returns:
        Dict with keys 'hits' (recommended movieIds that are relevant, in
        list order), 'n_hits', 'n_recommended', and 'precision' (Precision@K
        for this user with K = number of recommendations).
    """
    ids = [int(m) for m in recommended_ids]
    hits = [m for m in ids if m in relevant]
    return {
        "hits": hits,
        "n_hits": len(hits),
        "n_recommended": len(ids),
        "precision": precision_at_k(ids, list(relevant), k=len(ids)),
    }


def expected_hits_by_model(comparison: pd.DataFrame, n_recommendations: int) -> Dict[str, float]:
    """Average number of hits a list of n recommendations would contain.

    Uses each model's test-set Precision@10 as the per-slot hit rate, so this
    is an approximation for list lengths other than 10.

    Args:
        comparison: DataFrame with columns 'model' and 'precision_10', as in
            reports/full_model_comparison_v2.csv.
        n_recommendations: Length of the recommendation list shown.

    Returns:
        Dict mapping 'UBCF', 'IBCF', 'SVD' to expected hits. Models whose
        row is missing are omitted.
    """
    expected = {}
    for prefix in MODEL_PREFIXES:
        rows = comparison[comparison["model"].str.startswith(prefix)]
        if not rows.empty:
            expected[prefix] = float(rows["precision_10"].iloc[0]) * n_recommendations
    return expected


def held_out_liked(
    test_ratings: pd.DataFrame,
    user_id: int,
    movies: pd.DataFrame,
    threshold: float = RELEVANCE_THRESHOLD,
) -> List[Tuple[int, str, str, float]]:
    """Return a user's relevant test movies as card tuples, best-rated first.

    Args:
        test_ratings: DataFrame with columns userId, movieId, rating.
        user_id: The user to look up.
        movies: DataFrame with columns movieId, title, genres.
        threshold: Minimum true rating for a movie to be included.

    Returns:
        List of (movieId, title, genres, true_rating), sorted by rating
        descending, then movieId ascending.
    """
    mask = (test_ratings["userId"] == user_id) & (test_ratings["rating"] >= threshold)
    liked = (
        test_ratings.loc[mask, ["movieId", "rating"]]
        .merge(movies[["movieId", "title", "genres"]], on="movieId")
        .sort_values(["rating", "movieId"], ascending=[False, True])
    )
    return [
        (int(row.movieId), row.title, row.genres, float(row.rating))
        for row in liked.itertuples(index=False)
    ]


def format_comparison_table(comparison: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round metric columns and use readable headers, matching the report's Table I.

    Args:
        comparison: Raw model comparison DataFrame.
        decimals: Number of decimal places to keep.

    Returns:
        A new DataFrame; the input is not modified.
    """
    table = comparison.copy()
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].round(decimals)
    return table.rename(columns=COMPARISON_COLUMNS)
