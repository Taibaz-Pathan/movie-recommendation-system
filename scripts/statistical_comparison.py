"""Paired bootstrap significance test comparing tuned UBCF vs tuned IBCF on RMSE."""

import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
N_BOOTSTRAP = 1000
SEED = 42


def get_squared_errors(model, test_ratings: pd.DataFrame) -> np.ndarray:
    """Compute per-prediction squared errors for a fitted model on a test set.

    Skips (user, movie) pairs the model can't predict for (unseen in
    training) rather than raising, so callers can safely bootstrap over
    the returned array.

    Args:
        model: A fitted model exposing predict(user_id, movie_id) -> float
            and raising ValueError for unseen users/movies.
        test_ratings: DataFrame with columns userId, movieId, rating.

    Returns:
        1D numpy array of squared errors, one per predictable test row.
    """
    squared_errors = []
    for _, row in test_ratings.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        try:
            prediction = model.predict(user_id, movie_id)
        except ValueError:
            continue
        squared_errors.append((row["rating"] - prediction) ** 2)

    return np.array(squared_errors)


def bootstrap_rmse_comparison(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap comparison of RMSE between two models' error arrays.

    The same resampled indices are applied to both error arrays on each
    iteration (paired bootstrap), so the comparison accounts for the fact
    that both models were evaluated on the same underlying test rows.

    Args:
        errors_a: Per-prediction squared errors for model A (e.g. UBCF).
        errors_b: Per-prediction squared errors for model B (e.g. IBCF).
            Must be the same length as errors_a.
        n_bootstrap: Number of bootstrap resamples to draw.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            'a_wins_pct': Percentage of resamples where model A's RMSE
                was lower than model B's.
            'mean_rmse_diff': Mean of (RMSE_b - RMSE_a) across resamples.
            'ci_lower': 2.5th percentile of (RMSE_b - RMSE_a).
            'ci_upper': 97.5th percentile of (RMSE_b - RMSE_a).

    Raises:
        ValueError: If errors_a and errors_b have different lengths.
    """
    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"errors_a and errors_b must be the same length, got "
            f"{len(errors_a)} and {len(errors_b)}."
        )

    rng = np.random.default_rng(seed)
    n = len(errors_a)

    a_wins = 0
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rmse_a = np.sqrt(errors_a[idx].mean())
        rmse_b = np.sqrt(errors_b[idx].mean())

        if rmse_a < rmse_b:
            a_wins += 1
        diffs[i] = rmse_b - rmse_a

    return {
        "a_wins_pct": 100 * a_wins / n_bootstrap,
        "mean_rmse_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    print(f"Fitting UBCF (k={ubcf_cfg['k']}, min_support={ubcf_cfg['min_support']})...")
    ubcf = UserBasedCF(
        k=ubcf_cfg["k"],
        similarity=ubcf_cfg["similarity"],
        min_support=ubcf_cfg["min_support"],
    )
    ubcf.fit(train_matrix)

    print(f"Fitting IBCF (k={ibcf_cfg['k']}, min_support={ibcf_cfg['min_support']})...")
    ibcf = ItemBasedCF(k=ibcf_cfg["k"], min_support=ibcf_cfg["min_support"])
    ibcf.fit(train)

    print("Computing per-prediction squared errors...")
    errors_ubcf = get_squared_errors(ubcf, test)
    errors_ibcf = get_squared_errors(ibcf, test)

    print(f"  UBCF predictions : {len(errors_ubcf)}")
    print(f"  IBCF predictions : {len(errors_ibcf)}")

    if len(errors_ubcf) != len(errors_ibcf):
        raise RuntimeError(
            "UBCF and IBCF produced predictions for a different number of "
            "test rows -- paired bootstrap requires them to match. "
            f"UBCF={len(errors_ubcf)}, IBCF={len(errors_ibcf)}."
        )

    print(f"\nRunning paired bootstrap ({N_BOOTSTRAP} resamples, seed={SEED})...")
    result = bootstrap_rmse_comparison(
        errors_ubcf, errors_ibcf, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )

    rmse_ubcf = np.sqrt(errors_ubcf.mean())
    rmse_ibcf = np.sqrt(errors_ibcf.mean())

    print("\n===== Bootstrap RMSE comparison: UBCF (a) vs IBCF (b) =====")
    print(f"Point-estimate RMSE -- UBCF: {rmse_ubcf:.4f}  IBCF: {rmse_ibcf:.4f}")
    print(
        f"UBCF has lower RMSE in {result['a_wins_pct']:.1f}% of {N_BOOTSTRAP} resamples"
    )
    print(f"Mean RMSE difference (IBCF - UBCF): {result['mean_rmse_diff']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    print("\n===== Conclusion =====")
    if result["ci_lower"] > 0:
        print(
            "The 95% CI for (IBCF RMSE - UBCF RMSE) excludes zero and is entirely "
            "positive: UBCF's RMSE advantage over IBCF is statistically significant."
        )
    else:
        print(
            "The 95% CI for (IBCF RMSE - UBCF RMSE) includes zero: UBCF's RMSE "
            "advantage over IBCF is NOT statistically significant at the 95% level."
        )


if __name__ == "__main__":
    main()
