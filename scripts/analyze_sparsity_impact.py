"""Bucket test-set predictions by neighbour support and measure RMSE per bucket."""

import os
import sys
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
RESULTS_PATH = os.path.join("reports", "sparsity_impact_results.csv")

BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 20)]


def bucket_label(n_neighbours: int, buckets) -> str:
    for low, high in buckets:
        if low <= n_neighbours <= high:
            return f"{low}-{high}"
    return f">{buckets[-1][1]}"


def bucket_by_support(model, test: pd.DataFrame, buckets) -> pd.DataFrame:
    """Bucket per-prediction errors by how many valid neighbours backed each prediction.

    Args:
        model: A fitted model exposing _predict_with_support(user_id, movie_id)
            -> (prediction, n_neighbours), raising ValueError for unseen pairs.
        test: DataFrame with columns userId, movieId, rating.
        buckets: List of (low, high) inclusive n_neighbours ranges.

    Returns:
        DataFrame with columns bucket, rmse, n_predictions, one row per bucket
        (buckets with zero predictions are omitted).
    """
    squared_errors_by_bucket = {bucket_label(low, buckets): [] for low, high in buckets}
    # also collect anything above the highest bucket, in case it occurs
    overflow_label = f">{buckets[-1][1]}"
    squared_errors_by_bucket[overflow_label] = []

    for _, row in test.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        try:
            prediction, n_neighbours = model._predict_with_support(user_id, movie_id)
        except ValueError:
            continue

        label = bucket_label(n_neighbours, buckets)
        squared_errors_by_bucket[label].append((row["rating"] - prediction) ** 2)

    rows = []
    for label, errors in squared_errors_by_bucket.items():
        if not errors:
            continue
        errors = np.array(errors)
        rows.append(
            {
                "bucket": label,
                "rmse": float(np.sqrt(errors.mean())),
                "n_predictions": len(errors),
            }
        )

    return pd.DataFrame(rows)


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

    print("Bucketing UBCF predictions by neighbour count...")
    ubcf_results = bucket_by_support(ubcf, test, BUCKETS)
    ubcf_results.insert(0, "model", "UBCF")

    print("Bucketing IBCF predictions by neighbour count...")
    ibcf_results = bucket_by_support(ibcf, test, BUCKETS)
    ibcf_results.insert(0, "model", "IBCF")

    combined = pd.concat([ubcf_results, ibcf_results], ignore_index=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    combined.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}\n")

    print("===== Sparsity impact results (RMSE by neighbour-count bucket) =====")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
