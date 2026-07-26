"""Similarity metrics implemented from scratch using NumPy only.

Provides both vector-level functions (for teaching and single-pair use)
and matrix-level functions (for computing full similarity matrices efficiently).

All vector functions handle NaN values — missing ratings are excluded from
the computation rather than treated as zeros.
"""

from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Vector-level similarity (exact, NaN-aware)
# ---------------------------------------------------------------------------

def cosine_similarity(
    u: Union[np.ndarray, list],
    v: Union[np.ndarray, list],
) -> float:
    """Compute cosine similarity between two rating vectors.

    Only positions where both vectors are non-NaN (i.e. both users rated
    the same item) are used in the computation.

    Formula:
        cos(u, v) = dot(u, v) / (||u|| * ||v||)

    Args:
        u: Rating vector for user/item A. May contain NaN for unrated items.
        v: Rating vector for user/item B. May contain NaN for unrated items.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if there are no
        co-rated items or if either vector is all-zero on co-rated items.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    mask = ~np.isnan(u) & ~np.isnan(v)
    if mask.sum() == 0:
        return 0.0

    u_c = u[mask]
    v_c = v[mask]

    norm_u = np.sqrt(np.dot(u_c, u_c))
    norm_v = np.sqrt(np.dot(v_c, v_c))

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return float(np.clip(np.dot(u_c, v_c) / (norm_u * norm_v), -1.0, 1.0))


def pearson_similarity(
    u: Union[np.ndarray, list],
    v: Union[np.ndarray, list],
    min_support: int = 2,
) -> float:
    """Compute Pearson correlation between two rating vectors.

    Mean-centres each vector using only the co-rated items. This is the
    exact formulation — the mean is computed over shared items only, not
    over each user's full rating history.

    Formula:
        r(u, v) = sum((u_i - mean_u) * (v_i - mean_v))
                  / (||u - mean_u|| * ||v - mean_v||)

        where the sum and means are over co-rated items i only.

    Args:
        u: Rating vector for user/item A. May contain NaN for unrated items.
        v: Rating vector for user/item B. May contain NaN for unrated items.
        min_support: Minimum number of co-rated items required for the
            similarity to be meaningful. Returns 0.0 if below this threshold.

    Returns:
        Pearson correlation in [-1.0, 1.0]. Returns 0.0 when co-rated items
        are fewer than min_support or the denominator is zero.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    mask = ~np.isnan(u) & ~np.isnan(v)
    n_common = int(mask.sum())

    if n_common < min_support:
        return 0.0

    u_c = u[mask]
    v_c = v[mask]

    # Mean-centre on co-rated items only
    u_c = u_c - u_c.mean()
    v_c = v_c - v_c.mean()

    numerator   = np.dot(u_c, v_c)
    denominator = np.sqrt(np.dot(u_c, u_c)) * np.sqrt(np.dot(v_c, v_c))

    if denominator == 0.0:
        return 0.0

    return float(np.clip(numerator / denominator, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Matrix-level similarity (vectorised, efficient)
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute the full cosine similarity matrix for rows of a 2D array.

    NaN entries are treated as 0 before normalisation. This is the standard
    approach for computing item-item or user-user cosine similarity over
    a sparse ratings matrix.

    Args:
        matrix: 2D array of shape (n, m). Rows are entities (users or items),
            columns are features (items or users). May contain NaN.

    Returns:
        Symmetric similarity matrix of shape (n, n) with values in [-1, 1]
        and 1.0 on the diagonal.
    """
    filled = np.where(np.isnan(matrix), 0.0, matrix)

    norms = np.linalg.norm(filled, axis=1, keepdims=True)
    # Avoid division by zero for all-zero rows
    norms = np.where(norms == 0.0, 1.0, norms)

    normed = filled / norms
    sim = normed @ normed.T

    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


def pearson_similarity_matrix(
    matrix: np.ndarray,
    min_support: int = 5,
) -> np.ndarray:
    """Compute the full Pearson similarity matrix for rows of a 2D array.

    Each row is mean-centred using its own overall mean (not per-pair mean).
    Pairs with fewer than min_support co-rated items are set to 0.0.

    This is an efficient vectorised approximation — it uses each row's
    global mean rather than computing a separate mean per pair. The result
    closely matches the exact per-pair formulation and is standard in
    large-scale CF implementations.

    Args:
        matrix: 2D array of shape (n, m). Rows are entities (users or items),
            columns are features. May contain NaN.
        min_support: Minimum number of co-rated items required for a
            similarity score to be kept. Pairs below this are set to 0.0.

    Returns:
        Symmetric similarity matrix of shape (n, n) with values in [-1, 1]
        and 1.0 on the diagonal.
    """
    mask = (~np.isnan(matrix)).astype(float)  # 1 where rated, 0 where NaN

    # Mean-centre each row using that row's overall mean; NaN → 0
    row_means = np.nanmean(matrix, axis=1, keepdims=True)
    centered  = np.where(np.isnan(matrix), 0.0, matrix - row_means)

    # Numerator: pairwise dot products of centred vectors
    # Positions that are NaN in either vector contribute 0 (already zeroed out)
    numerator = centered @ centered.T  # (n, n)

    # Denominator: product of L2 norms of centred vectors
    norms = np.sqrt((centered ** 2).sum(axis=1))  # (n,)
    denom = np.outer(norms, norms)                 # (n, n)

    # Pearson correlation; guard against zero denominator
    sim = np.where(denom > 0.0, numerator / denom, 0.0)

    # Apply min_support: zero out pairs with insufficient co-rated items
    co_rated = mask @ mask.T  # (n, n) — count of shared rated items
    sim = np.where(co_rated >= min_support, sim, 0.0)

    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)
