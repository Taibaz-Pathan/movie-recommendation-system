"""Unit tests for src/evaluation/demo_metrics.py (the demo's hit logic)."""

import math

import pandas as pd

from src.evaluation.demo_metrics import (
    expected_hits_by_model,
    format_comparison_table,
    held_out_liked,
    hit_summary,
    relevant_test_items,
)


def _test_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2],
            "movieId": [10, 20, 30, 40, 10, 50],
            "rating": [5.0, 4.0, 3.5, 4.5, 2.0, 3.0],
        }
    )


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Drama", "Comedy", "Action", "Drama|Romance", "Horror"],
        }
    )


# --- relevant_test_items ---


def test_relevant_items_uses_threshold_inclusively():
    # 4.0 counts as relevant; 3.5 does not
    assert relevant_test_items(_test_ratings(), user_id=1) == {10, 20, 40}


def test_relevant_items_only_for_requested_user():
    # movie 10 is relevant for user 1 but rated 2.0 by user 2
    assert relevant_test_items(_test_ratings(), user_id=2) == set()


def test_relevant_items_unknown_user_is_empty():
    assert relevant_test_items(_test_ratings(), user_id=99) == set()


def test_relevant_items_custom_threshold():
    assert relevant_test_items(_test_ratings(), user_id=1, threshold=4.5) == {10, 40}


# --- hit_summary ---


def test_hit_summary_counts_and_keeps_list_order():
    summary = hit_summary([40, 99, 10, 98, 97], relevant={10, 20, 40})
    assert summary["hits"] == [40, 10]
    assert summary["n_hits"] == 2
    assert summary["n_recommended"] == 5
    assert math.isclose(summary["precision"], 0.4)


def test_hit_summary_no_hits():
    summary = hit_summary([1, 2, 3], relevant={10})
    assert summary["n_hits"] == 0
    assert summary["precision"] == 0.0


def test_hit_summary_empty_recommendations():
    summary = hit_summary([], relevant={10})
    assert summary["n_hits"] == 0
    assert summary["n_recommended"] == 0
    assert summary["precision"] == 0.0


def test_hit_summary_accepts_numpy_like_ids():
    # recommend() can return numpy integers; they must match plain ints
    summary = hit_summary(pd.Series([10, 30]).tolist(), relevant={10})
    assert summary["hits"] == [10]


# --- expected_hits_by_model ---


def test_expected_hits_scales_precision_by_list_length():
    comparison = pd.DataFrame(
        {
            "model": ["SVD (n_factors=50)", "UBCF (k=20)", "IBCF (k=30)", "GlobalMeanBaseline"],
            "precision_10": [0.06, 0.05, 0.04, 0.02],
        }
    )
    expected = expected_hits_by_model(comparison, n_recommendations=5)
    assert set(expected) == {"UBCF", "IBCF", "SVD"}
    assert math.isclose(expected["UBCF"], 0.25)
    assert math.isclose(expected["IBCF"], 0.20)
    assert math.isclose(expected["SVD"], 0.30)


def test_expected_hits_skips_missing_models():
    comparison = pd.DataFrame({"model": ["UBCF (k=20)"], "precision_10": [0.05]})
    assert set(expected_hits_by_model(comparison, 10)) == {"UBCF"}


# --- held_out_liked ---


def test_held_out_liked_sorted_best_first_with_titles():
    liked = held_out_liked(_test_ratings(), user_id=1, movies=_movies())
    assert liked == [
        (10, "A", "Drama", 5.0),
        (40, "D", "Drama|Romance", 4.5),
        (20, "B", "Comedy", 4.0),
    ]


def test_held_out_liked_empty_when_nothing_relevant():
    assert held_out_liked(_test_ratings(), user_id=2, movies=_movies()) == []


# --- format_comparison_table ---


def test_format_comparison_rounds_and_renames():
    raw = pd.DataFrame(
        {
            "model": ["SVD"],
            "rmse": [0.837870390790161],
            "mae": [0.6434829017997988],
            "precision_10": [0.06010733452593918],
            "recall_10": [0.04952308350122667],
        }
    )
    table = format_comparison_table(raw)
    assert list(table.columns) == ["Model", "RMSE", "MAE", "Precision@10", "Recall@10"]
    assert table.loc[0, "RMSE"] == 0.8379
    assert table.loc[0, "Precision@10"] == 0.0601
    assert table.loc[0, "Model"] == "SVD"


def test_format_comparison_does_not_modify_input():
    raw = pd.DataFrame({"model": ["SVD"], "rmse": [0.837870390790161]})
    format_comparison_table(raw)
    assert raw.loc[0, "rmse"] == 0.837870390790161
    assert list(raw.columns) == ["model", "rmse"]
