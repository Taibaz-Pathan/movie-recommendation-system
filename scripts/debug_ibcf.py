"""Diagnostic script: investigate why ItemBasedCF underperforms UserBasedCF."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.models.ibcf import ItemBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
N_ROWS = 20


def count_valid_neighbours(model: ItemBasedCF, user_id: int, movie_id: int) -> int:
    """Replicate predict()'s neighbour-selection logic to count nonzero-similarity neighbours used."""
    u_idx = model._get_user_index(user_id)
    m_idx = model._get_movie_index(movie_id)

    user_ratings = model._matrix.iloc[u_idx].to_numpy()

    similarities = model._sim_matrix[m_idx].copy()
    similarities[m_idx] = 0.0

    rated_mask = ~np.isnan(user_ratings)
    similarities[~rated_mask] = 0.0

    top_k_idx = np.argsort(np.abs(similarities))[::-1][: model.k]
    top_k_sims = similarities[top_k_idx]

    return int(np.sum(top_k_sims != 0.0))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    model = ItemBasedCF(k=20, min_support=5)
    model.fit(train)

    rows = test.head(N_ROWS)

    print(f"{'userId':>8} {'movieId':>8} {'actual':>8} {'predicted':>10} {'n_valid':>8}")
    print("-" * 46)

    n_clipped_low = 0
    n_clipped_high = 0
    n_predicted = 0

    for _, row in rows.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        actual = row["rating"]

        try:
            n_valid = count_valid_neighbours(model, user_id, movie_id)
            predicted = model.predict(user_id, movie_id)
        except ValueError as exc:
            print(f"{user_id:>8} {movie_id:>8} {actual:>8.2f} {'N/A':>10} {'N/A':>8}  ({exc})")
            continue

        n_predicted += 1
        if predicted == 0.5:
            n_clipped_low += 1
        elif predicted == 5.0:
            n_clipped_high += 1

        print(f"{user_id:>8} {movie_id:>8} {actual:>8.2f} {predicted:>10.4f} {n_valid:>8}")

    print()
    print(f"Predictions clipped to exactly 0.5: {n_clipped_low} / {n_predicted}")
    print(f"Predictions clipped to exactly 5.0: {n_clipped_high} / {n_predicted}")

    # --- similarity matrix summary (excluding the trivial diagonal) ---
    sim = model._sim_matrix.copy()
    np.fill_diagonal(sim, 0.0)
    nonzero = sim[sim != 0.0]

    print()
    print("Similarity matrix nonzero values (off-diagonal):")
    if nonzero.size == 0:
        print("  No nonzero off-diagonal similarities survive the min_support filter.")
    else:
        print(f"  count : {nonzero.size}")
        print(f"  min   : {nonzero.min():.4f}")
        print(f"  max   : {nonzero.max():.4f}")
        print(f"  mean  : {nonzero.mean():.4f}")

    # --- min_support sparsity ---
    rated_mask = (~model._matrix.isna()).to_numpy().astype(float)
    co_rating_counts = rated_mask.T @ rated_mask
    n_movies = co_rating_counts.shape[0]

    iu = np.triu_indices(n_movies, k=1)
    total_pairs = iu[0].size
    surviving_pairs = int(np.sum(co_rating_counts[iu] >= model.min_support))
    pct_surviving = 100 * surviving_pairs / total_pairs

    print()
    print(f"Item-item pairs total       : {total_pairs}")
    print(f"Pairs surviving min_support={model.min_support}: {surviving_pairs} ({pct_surviving:.2f}%)")


if __name__ == "__main__":
    main()
