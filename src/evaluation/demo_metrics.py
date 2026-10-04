"""Helpers that connect the Streamlit demo to the offline evaluation.

The demo's "hits" view uses the same definitions as evaluate_ranking() in
src/evaluation/metrics.py: a held-out test movie is relevant to a user if
their true rating for it is at least the relevance threshold, and a
recommendation counts as a hit if it is one of those movies. Keeping this
logic here (pandas only, no Streamlit) means it can be unit-tested.
"""

import math
import re
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

# Internal model names in the results file -> names fit for a table.
MODEL_NAME_PATTERNS = (
    (r"^UBCF\b", "User-Based CF"),
    (r"^IBCF\b", "Item-Based CF"),
    (r"n_factors=(\d+), n_epochs=(\d+)", r"\1 factors, \2 epochs"),
    (r"^(Global|User|Item)MeanBaseline$", r"\1 Mean (baseline)"),
    (r"(\w)=(\w)", r"\1 = \2"),
)


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


def star_histogram(ratings: Iterable[float]) -> List[int]:
    """Count ratings per whole-star bin for a compact 1-5 star histogram.

    Half-star ratings are grouped with the next whole star (0.5 and 1.0 ->
    1 star, 1.5 and 2.0 -> 2 stars, ..., 4.5 and 5.0 -> 5 stars).

    Args:
        ratings: Ratings on the 0.5-5.0 scale.

    Returns:
        List of five counts, for 1 to 5 stars.
    """
    counts = [0, 0, 0, 0, 0]
    for rating in ratings:
        star = min(5, max(1, math.ceil(rating)))
        counts[star - 1] += 1
    return counts


def top_contributors(neighbours: List[Dict], n: int = 2) -> List[Dict]:
    """Pick the neighbour items that pushed an IBCF prediction up the most.

    Args:
        neighbours: The 'neighbours' list from ItemBasedCF.explain(), sorted
            by contribution descending.
        n: Maximum number of items to return.

    Returns:
        Up to n neighbours with a positive contribution, largest first. Empty
        if no neighbour raised the prediction, so a caller never presents an
        item that lowered the score as the reason for a recommendation.
    """
    return [item for item in neighbours if item["contribution"] > 0][:n]


def experienced_rating_range(
    train_ratings: pd.DataFrame,
    light_max: int,
    lower_quantile: float = 0.75,
    upper_quantile: float = 0.95,
) -> Tuple[int, int]:
    """Training-rating-count range used to pick "experienced" demo users.

    Derived from the data rather than hard-coded, so the group is never empty.
    The upper quantile leaves out the most extreme raters, who have already
    rated most of the catalog and so leave few candidates to recommend.

    Args:
        train_ratings: DataFrame with a userId column, one row per rating.
        light_max: Upper bound of the light-user group; the experienced range
            always starts above it so the two groups never overlap.
        lower_quantile: Quantile of per-user rating counts for the lower bound.
        upper_quantile: Quantile of per-user rating counts for the upper bound.

    Returns:
        (min_ratings, max_ratings), inclusive, with min_ratings <= max_ratings.
    """
    counts = train_ratings.groupby("userId").size()
    low = max(math.ceil(counts.quantile(lower_quantile)), light_max + 1)
    high = int(counts.quantile(upper_quantile))
    return low, max(low, high)


def display_model_name(name: str) -> str:
    """Turn an internal model name into a readable table label.

    Examples: 'UBCF (k=20, min_support=10)' -> 'User-Based CF (k = 20,
    min_support = 10)'; 'UserMeanBaseline' -> 'User Mean (baseline)'.
    Names that match no pattern are returned unchanged.
    """
    for pattern, replacement in MODEL_NAME_PATTERNS:
        name = re.sub(pattern, replacement, name)
    return name


def format_comparison_table(comparison: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round metrics and use readable headers and model names, matching Table I.

    Args:
        comparison: Raw model comparison DataFrame.
        decimals: Number of decimal places to keep.

    Returns:
        A new DataFrame; the input is not modified.
    """
    table = comparison.copy()
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].round(decimals)
    if "model" in table.columns:
        table["model"] = table["model"].map(display_model_name)
    return table.rename(columns=COMPARISON_COLUMNS)
