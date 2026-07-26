"""Unit tests for src/utils/similarity.py."""

import math
import numpy as np
import pytest

from src.utils.similarity import (
    cosine_similarity,
    pearson_similarity,
    cosine_similarity_matrix,
    pearson_similarity_matrix,
)


# ---------------------------------------------------------------------------
# cosine_similarity — vector
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    """Cosine similarity of a vector with itself should be 1.0."""
    u = [4.0, 3.0, 5.0, 2.0]
    assert cosine_similarity(u, u) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    """Orthogonal vectors should have cosine similarity of 0.0."""
    u = [1.0, 0.0]
    v = [0.0, 1.0]
    assert cosine_similarity(u, v) == pytest.approx(0.0)


def test_cosine_known_value():
    """Verify cosine similarity against a manually computed value."""
    u = [3.0, 4.0]
    v = [4.0, 3.0]
    # dot=24, ||u||=5, ||v||=5 → 24/25 = 0.96
    assert cosine_similarity(u, v) == pytest.approx(0.96, rel=1e-6)


def test_cosine_ignores_nan_positions():
    """NaN positions should be excluded from the computation."""
    u = [4.0, np.nan, 3.0]
    v = [4.0, 5.0,    3.0]
    # Only positions 0 and 2 are co-rated: u=[4,3], v=[4,3] → sim=1.0
    assert cosine_similarity(u, v) == pytest.approx(1.0)


def test_cosine_no_common_ratings():
    """Should return 0.0 when there are no co-rated items."""
    u = [4.0, np.nan]
    v = [np.nan, 3.0]
    assert cosine_similarity(u, v) == 0.0


def test_cosine_result_in_range():
    """Cosine similarity must be in [-1, 1]."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.uniform(0.5, 5.0, 20)
        v = rng.uniform(0.5, 5.0, 20)
        sim = cosine_similarity(u, v)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# pearson_similarity — vector
# ---------------------------------------------------------------------------

def test_pearson_identical_vectors():
    """Pearson similarity of a vector with itself should be 1.0."""
    u = [3.0, 4.0, 2.0, 5.0, 1.0]
    assert pearson_similarity(u, u) == pytest.approx(1.0)


def test_pearson_opposite_vectors():
    """Perfectly inversely correlated vectors should give -1.0."""
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    v = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert pearson_similarity(u, v) == pytest.approx(-1.0)


def test_pearson_known_value():
    """Verify Pearson against a manually computed result."""
    # u = [1, 2, 3], v = [1, 2, 3] → r = 1.0
    u = [1.0, 2.0, 3.0]
    v = [1.0, 2.0, 3.0]
    assert pearson_similarity(u, v, min_support=2) == pytest.approx(1.0)


def test_pearson_constant_vector_returns_zero():
    """A constant vector has zero variance — denominator is 0, should return 0."""
    u = [3.0, 3.0, 3.0]
    v = [1.0, 2.0, 3.0]
    assert pearson_similarity(u, v, min_support=2) == 0.0


def test_pearson_below_min_support():
    """Should return 0.0 when co-rated items are fewer than min_support."""
    u = [4.0, np.nan, np.nan]
    v = [3.0, np.nan, np.nan]
    # Only 1 co-rated item, min_support=2
    assert pearson_similarity(u, v, min_support=2) == 0.0


def test_pearson_ignores_nan_positions():
    """NaN positions should not affect the computation."""
    u = [5.0, np.nan, 3.0, 1.0]
    v = [5.0, 4.0,    3.0, 1.0]
    # Co-rated positions: 0, 2, 3 → u=[5,3,1], v=[5,3,1] → r=1.0
    assert pearson_similarity(u, v, min_support=2) == pytest.approx(1.0)


def test_pearson_result_in_range():
    """Pearson similarity must be in [-1, 1]."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = rng.uniform(0.5, 5.0, 20)
        v = rng.uniform(0.5, 5.0, 20)
        sim = pearson_similarity(u, v, min_support=2)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------

def test_cosine_matrix_shape():
    """Output should be (n, n) for an (n, m) input."""
    matrix = np.array([[4.0, 3.0, np.nan],
                       [np.nan, 2.0, 5.0],
                       [3.0, np.nan, 4.0]])
    sim = cosine_similarity_matrix(matrix)
    assert sim.shape == (3, 3)


def test_cosine_matrix_diagonal_is_one():
    """Diagonal entries should all be 1.0."""
    matrix = np.array([[4.0, 3.0, 2.0],
                       [2.0, 5.0, 1.0]])
    sim = cosine_similarity_matrix(matrix)
    np.testing.assert_array_almost_equal(np.diag(sim), [1.0, 1.0])


def test_cosine_matrix_is_symmetric():
    """Similarity matrix must be symmetric."""
    matrix = np.array([[4.0, 3.0, np.nan],
                       [np.nan, 2.0, 5.0],
                       [3.0, np.nan, 4.0]])
    sim = cosine_similarity_matrix(matrix)
    np.testing.assert_array_almost_equal(sim, sim.T)


def test_cosine_matrix_values_in_range():
    """All values must be in [-1, 1]."""
    rng = np.random.default_rng(2)
    matrix = rng.uniform(0.5, 5.0, (10, 20))
    sim = cosine_similarity_matrix(matrix)
    assert sim.min() >= -1.0
    assert sim.max() <= 1.0


# ---------------------------------------------------------------------------
# pearson_similarity_matrix
# ---------------------------------------------------------------------------

def test_pearson_matrix_shape():
    """Output should be (n, n) for an (n, m) input."""
    matrix = np.array([[4.0, 3.0, np.nan],
                       [np.nan, 2.0, 5.0],
                       [3.0, 4.0, 4.0]])
    sim = pearson_similarity_matrix(matrix, min_support=1)
    assert sim.shape == (3, 3)


def test_pearson_matrix_diagonal_is_one():
    """Diagonal entries should all be 1.0."""
    matrix = np.array([[4.0, 3.0, 2.0],
                       [2.0, 5.0, 1.0]])
    sim = pearson_similarity_matrix(matrix, min_support=1)
    np.testing.assert_array_almost_equal(np.diag(sim), [1.0, 1.0])


def test_pearson_matrix_is_symmetric():
    """Similarity matrix must be symmetric."""
    matrix = np.array([[4.0, 3.0, np.nan],
                       [np.nan, 2.0, 5.0],
                       [3.0, 4.0, 4.0]])
    sim = pearson_similarity_matrix(matrix, min_support=1)
    np.testing.assert_array_almost_equal(sim, sim.T)


def test_pearson_matrix_min_support_zeroes_sparse_pairs():
    """Pairs with fewer co-rated items than min_support should be 0.0."""
    # Users 0 and 1 share only 1 co-rated item (position 0)
    matrix = np.array([[4.0, np.nan, np.nan],
                       [3.0, np.nan, np.nan],
                       [4.0, 3.0,    5.0]])
    sim = pearson_similarity_matrix(matrix, min_support=2)
    assert sim[0, 1] == 0.0
    assert sim[1, 0] == 0.0


def test_pearson_matrix_values_in_range():
    """All values must be in [-1, 1]."""
    rng = np.random.default_rng(3)
    matrix = rng.uniform(0.5, 5.0, (15, 30))
    sim = pearson_similarity_matrix(matrix, min_support=2)
    assert sim.min() >= -1.0
    assert sim.max() <= 1.0
