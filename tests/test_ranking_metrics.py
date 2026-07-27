"""Unit tests for the ranking metrics in src/evaluation/metrics.py."""

import pytest

from src.evaluation.metrics import f1_at_k, precision_at_k, recall_at_k


# --- precision_at_k ---

def test_precision_perfect_match():
    # top-3 == relevant exactly -> 3/3 hits = 1.0
    recommended = [1, 2, 3]
    relevant = [1, 2, 3]
    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_precision_no_overlap():
    # none of the top-3 are relevant -> 0/3 = 0.0
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(0.0)


def test_precision_partial_overlap():
    # top-4 = [1,2,3,4], relevant hits = {2,4} -> 2/4 = 0.5
    recommended = [1, 2, 3, 4]
    relevant = [2, 4, 9]
    assert precision_at_k(recommended, relevant, k=4) == pytest.approx(0.5)


def test_precision_cutoff_respected():
    # top-2 = [1,2]; relevant items 4 and 5 exist but fall outside the
    # cutoff, so they must not count -> 0/2 = 0.0
    recommended = [1, 2, 3, 4, 5]
    relevant = [4, 5]
    assert precision_at_k(recommended, relevant, k=2) == pytest.approx(0.0)


def test_precision_empty_recommended():
    assert precision_at_k([], [1, 2], k=5) == pytest.approx(0.0)


# --- recall_at_k ---

def test_recall_perfect_match():
    # all 3 relevant items appear in top-3 -> 3/3 = 1.0
    recommended = [1, 2, 3]
    relevant = [1, 2, 3]
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_recall_partial_match():
    # top-4 = [1,2,3,4], relevant = {2,4,9}; hits = {2,4} -> 2/3 = 0.6667
    recommended = [1, 2, 3, 4]
    relevant = [2, 4, 9]
    assert recall_at_k(recommended, relevant, k=4) == pytest.approx(2 / 3)


def test_recall_no_relevant_items():
    assert recall_at_k([1, 2, 3], [], k=3) == pytest.approx(0.0)


def test_recall_cutoff_respected():
    # top-2 = [1,2]; relevant = {1,5}; only 1 is within the cutoff -> 1/2 = 0.5
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 5]
    assert recall_at_k(recommended, relevant, k=2) == pytest.approx(0.5)


# --- f1_at_k ---

def test_f1_balanced_precision_recall():
    # precision == recall == 0.5 -> harmonic mean = 0.5
    assert f1_at_k(0.5, 0.5) == pytest.approx(0.5)


def test_f1_both_zero():
    assert f1_at_k(0.0, 0.0) == pytest.approx(0.0)


def test_f1_asymmetric_values():
    # precision=1.0, recall=0.5 -> 2*1*0.5 / (1+0.5) = 1/1.5 = 0.6667
    assert f1_at_k(1.0, 0.5) == pytest.approx(2 / 3)
