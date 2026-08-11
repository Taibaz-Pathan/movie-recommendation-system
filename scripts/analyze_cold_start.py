"""Cold-start simulation: truncate each user's training history and measure RMSE degradation."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
RESULTS_PATH = os.path.join("reports", "cold_start_results.csv")

TRUNCATION_LEVELS = [1, 3, 5, 10, 20]
SEED = 42


def truncate_user_ratings(train: pd.DataFrame, max_ratings: int, seed: int) -> pd.DataFrame:
    """Randomly keep at most max_ratings rows per user, simulating a sparser history.

    Users with fewer than max_ratings rows keep all of them.

    Args:
        train: Training ratings DataFrame with columns userId, movieId, rating.
        max_ratings: Maximum number of rows to keep per user.
        seed: Random seed for reproducible sampling.

    Returns:
        Truncated DataFrame with the same columns as train.
    """
    rng = np.random.default_rng(seed)
    kept_indices = []

    for _, group in train.groupby("userId"):
        idx = group.index.to_numpy()
        if len(idx) > max_ratings:
            idx = rng.choice(idx, size=max_ratings, replace=False)
        kept_indices.extend(idx.tolist())

    return train.loc[kept_indices].reset_index(drop=True)


def evaluate_at_truncation(
    model_class,
    model_kwargs: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_ratings,
    seed: int,
) -> dict:
    """Truncate training data, fit a model, and evaluate RMSE on the full test set.

    Args:
        model_class: UserBasedCF or ItemBasedCF.
        model_kwargs: Keyword arguments to construct model_class with.
        train: Full training ratings DataFrame.
        test: Full test ratings DataFrame.
        max_ratings: Max ratings to keep per user, or None for no truncation
            (the full/no-truncation baseline).
        seed: Random seed passed to truncate_user_ratings.

    Returns:
        dict with keys 'rmse', 'mae', 'n_predictions' from model.evaluate().
    """
    truncated = train if max_ratings is None else truncate_user_ratings(train, max_ratings, seed)

    fit_data = build_user_item_matrix(truncated) if model_class is UserBasedCF else truncated

    model = model_class(**model_kwargs)
    model.fit(fit_data)
    return model.evaluate(test)


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    model_configs = [
        (
            "UBCF",
            UserBasedCF,
            {
                "k": ubcf_cfg["k"],
                "similarity": ubcf_cfg["similarity"],
                "min_support": ubcf_cfg["min_support"],
            },
        ),
        ("IBCF", ItemBasedCF, {"k": ibcf_cfg["k"], "min_support": ibcf_cfg["min_support"]}),
    ]

    results = []
    for name, model_class, kwargs in model_configs:
        for max_ratings in TRUNCATION_LEVELS:
            print(f"[{name}] Testing max_ratings={max_ratings}...", flush=True)
            start = time.time()
            metrics = evaluate_at_truncation(model_class, kwargs, train, test, max_ratings, SEED)
            elapsed = time.time() - start
            print(
                f"  done in {elapsed:.1f}s  "
                f"(RMSE={metrics['rmse']:.4f}, n_predictions={metrics['n_predictions']})",
                flush=True,
            )
            results.append({
                "model": name,
                "max_ratings": max_ratings,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "n_predictions": metrics["n_predictions"],
            })

        print(f"[{name}] Testing max_ratings=full (no truncation)...", flush=True)
        start = time.time()
        metrics = evaluate_at_truncation(model_class, kwargs, train, test, None, SEED)
        elapsed = time.time() - start
        print(
            f"  done in {elapsed:.1f}s  "
            f"(RMSE={metrics['rmse']:.4f}, n_predictions={metrics['n_predictions']})",
            flush=True,
        )
        results.append({
            "model": name,
            "max_ratings": "full",
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "n_predictions": metrics["n_predictions"],
        })

    results_df = pd.DataFrame(results)[["model", "max_ratings", "rmse", "mae", "n_predictions"]]
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}\n")

    print("===== Cold-start simulation results =====")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
