"""Grid search over k and min_support for UserBasedCF and ItemBasedCF."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.evaluation.metrics import evaluate_ranking
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
RESULTS_PATH = os.path.join("reports", "hyperparameter_tuning_results.csv")

K_GRID = [5, 10, 20, 30, 40]
MIN_SUPPORT_GRID = [1, 3, 5, 10]


def evaluate_config(model_name: str, model, test: pd.DataFrame) -> dict:
    metrics = model.evaluate(test)
    ranking = evaluate_ranking(model, test, k_values=[10])

    return {
        "model": model_name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "precision_10": ranking[10]["precision"],
        "recall_10": ranking[10]["recall"],
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    results = []
    overall_start = time.time()
    n_total = len(K_GRID) * len(MIN_SUPPORT_GRID) * 2
    n_done = 0

    for k in K_GRID:
        for min_support in MIN_SUPPORT_GRID:
            print(f"[UBCF] Testing k={k}, min_support={min_support}...", flush=True)
            start = time.time()
            model = UserBasedCF(k=k, similarity="pearson", min_support=min_support)
            model.fit(train_matrix)
            row = evaluate_config("UBCF", model, test)
            row.update({"k": k, "min_support": min_support})
            elapsed = time.time() - start
            n_done += 1
            print(
                f"  done in {elapsed:.1f}s  "
                f"(RMSE={row['rmse']:.4f}, P@10={row['precision_10']:.4f})  "
                f"[{n_done}/{n_total}]",
                flush=True,
            )
            results.append(row)

            print(f"[IBCF] Testing k={k}, min_support={min_support}...", flush=True)
            start = time.time()
            model = ItemBasedCF(k=k, min_support=min_support)
            model.fit(train)
            row = evaluate_config("IBCF", model, test)
            row.update({"k": k, "min_support": min_support})
            elapsed = time.time() - start
            n_done += 1
            print(
                f"  done in {elapsed:.1f}s  "
                f"(RMSE={row['rmse']:.4f}, P@10={row['precision_10']:.4f})  "
                f"[{n_done}/{n_total}]",
                flush=True,
            )
            results.append(row)

    total_elapsed = time.time() - overall_start
    print(f"\nGrid search complete in {total_elapsed / 60:.1f} minutes.")

    results_df = pd.DataFrame(results)[
        ["model", "k", "min_support", "rmse", "mae", "precision_10", "recall_10"]
    ]
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")

    print("\n===== Best config per model by lowest RMSE =====")
    best_rmse = results_df.loc[results_df.groupby("model")["rmse"].idxmin()]
    print(best_rmse.to_string(index=False))

    print("\n===== Best config per model by highest Precision@10 =====")
    best_precision = results_df.loc[results_df.groupby("model")["precision_10"].idxmax()]
    print(best_precision.to_string(index=False))


if __name__ == "__main__":
    main()
