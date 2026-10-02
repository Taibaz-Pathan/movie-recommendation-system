"""Unit tests for the explain() methods of UBCF and IBCF and their display helpers.

The key property: an explanation must describe exactly the neighbours the
prediction was computed from, so each test checks explain() against predict().
"""

import math

import numpy as np
import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.evaluation.demo_metrics import star_histogram, top_contributors
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF


def _ratings() -> pd.DataFrame:
    """Six users, six movies. Users 1-3 share one taste, users 4-5 the opposite.

    User 6 has rated only two movies, so no user pair reaches min_support
    with them (the fallback case).
    """
    rows = [
        # userId, movieId, rating
        (1, 10, 5.0), (1, 20, 4.5), (1, 30, 1.0), (1, 40, 2.0), (1, 50, 4.0),
        (2, 10, 4.5), (2, 20, 5.0), (2, 30, 1.5), (2, 40, 1.0), (2, 50, 4.5), (2, 60, 5.0),
        (3, 10, 4.0), (3, 20, 4.5), (3, 30, 2.0), (3, 40, 1.5), (3, 50, 3.5), (3, 60, 4.5),
        (4, 10, 1.0), (4, 20, 1.5), (4, 30, 5.0), (4, 40, 4.5), (4, 50, 2.0), (4, 60, 1.0),
        (5, 10, 2.0), (5, 20, 1.0), (5, 30, 4.5), (5, 40, 5.0), (5, 50, 1.5), (5, 60, 1.5),
        (6, 10, 3.0), (6, 20, 4.0),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def _ubcf() -> UserBasedCF:
    return UserBasedCF(k=20, similarity="pearson", min_support=3).fit(
        build_user_item_matrix(_ratings())
    )


def _ibcf() -> ItemBasedCF:
    return ItemBasedCF(k=30, min_support=1).fit(_ratings())


# --- UserBasedCF.explain ---


def test_ubcf_explain_uses_only_positively_correlated_neighbours():
    # user 1 has not rated movie 60; users 2 and 3 share user 1's taste,
    # users 4 and 5 have the opposite taste and must be excluded
    explanation = _ubcf().explain(user_id=1, movie_id=60)
    assert explanation["fallback"] is False
    assert explanation["n_neighbours"] == 2
    assert sorted(explanation["neighbour_ratings"]) == [4.5, 5.0]
    assert math.isclose(explanation["mean_neighbour_rating"], 4.75)


def test_ubcf_explain_matches_prediction_support():
    model = _ubcf()
    _, n_neighbours = model._predict_with_support(1, 60)
    assert model.explain(1, 60)["n_neighbours"] == n_neighbours


def test_ubcf_prediction_follows_resnick_formula_for_explained_neighbours():
    # recompute the prediction by hand from the neighbours explain() reports
    model = _ubcf()
    sims = model._select_neighbours(1, 60)
    deviations = model._ratings_matrix.loc[sims.index, 60] - model._user_means[sims.index]
    expected = model._user_means[1] + (sims * deviations).sum() / sims.abs().sum()
    assert math.isclose(model.predict(1, 60), float(np.clip(expected, 0.5, 5.0)))


def test_ubcf_explain_reports_fallback_when_too_few_neighbours():
    # user 6 shares at most 2 movies with anyone, below min_support = 3
    model = _ubcf()
    explanation = model.explain(user_id=6, movie_id=30)
    assert explanation == {
        "fallback": True,
        "n_neighbours": 0,
        "neighbour_ratings": [],
        "mean_neighbour_rating": None,
    }
    assert math.isclose(model.predict(6, 30), 3.5)  # user 6's mean of 3.0 and 4.0


def test_ubcf_explain_rejects_unknown_user_or_movie():
    model = _ubcf()
    for user_id, movie_id in [(99, 10), (1, 999)]:
        try:
            model.explain(user_id, movie_id)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# --- ItemBasedCF.explain ---


def test_ibcf_explain_neighbours_are_items_the_user_rated():
    explanation = _ibcf().explain(user_id=1, movie_id=60)
    rated_by_user_1 = {10, 20, 30, 40, 50}
    assert explanation["fallback"] is False
    assert {item["movieId"] for item in explanation["neighbours"]} <= rated_by_user_1
    assert 60 not in {item["movieId"] for item in explanation["neighbours"]}


def test_ibcf_explain_matches_prediction_support():
    model = _ibcf()
    _, n_neighbours = model._predict_with_support(1, 60)
    explanation = model.explain(1, 60)
    assert explanation["n_neighbours"] == n_neighbours == len(explanation["neighbours"])


def test_ibcf_explain_sorted_by_contribution_and_reconstructs_prediction():
    model = _ibcf()
    neighbours = model.explain(1, 60)["neighbours"]
    contributions = [item["contribution"] for item in neighbours]
    assert contributions == sorted(contributions, reverse=True)

    numerator = sum(contributions)
    denominator = sum(abs(item["similarity"]) for item in neighbours)
    expected = model._item_means[60] + numerator / denominator
    assert math.isclose(model.predict(1, 60), float(np.clip(expected, 0.5, 5.0)))


def test_ibcf_explain_reports_fallback_when_no_similar_rated_item():
    # with min_support above the number of users, every similarity is zeroed
    model = ItemBasedCF(k=30, min_support=100).fit(_ratings())
    assert model.explain(1, 60) == {"fallback": True, "n_neighbours": 0, "neighbours": []}


# --- display helpers ---


def test_star_histogram_groups_half_stars_with_next_whole_star():
    assert star_histogram([0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 5.0]) == [2, 2, 0, 2, 3]


def test_star_histogram_empty():
    assert star_histogram([]) == [0, 0, 0, 0, 0]


def test_top_contributors_keeps_only_items_that_raised_the_prediction():
    neighbours = [
        {"movieId": 1, "contribution": 0.9},
        {"movieId": 2, "contribution": 0.4},
        {"movieId": 3, "contribution": 0.1},
        {"movieId": 4, "contribution": -0.5},
    ]
    assert [item["movieId"] for item in top_contributors(neighbours, n=2)] == [1, 2]


def test_top_contributors_empty_when_nothing_raised_the_prediction():
    neighbours = [{"movieId": 4, "contribution": -0.5}, {"movieId": 5, "contribution": 0.0}]
    assert top_contributors(neighbours) == []
