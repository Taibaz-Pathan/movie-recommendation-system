"""Compare all 6 models: 3 non-personalised baselines vs tuned UBCF vs tuned IBCF vs SVD."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.evaluation.metrics import evaluate_ranking
from src.models.baselines import GlobalMeanBaseline, ItemMeanBaseline, UserMeanBaseline
from src.models.ibcf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
RESULTS_PATH = os.path.join("reports", "full_model_comparison_v2.csv")


def evaluate_model(name: str, model, test: pd.DataFrame) -> dict:
    print(f"Evaluating {name}...", flush=True)
    metrics = model.evaluate(test)
    ranking = evaluate_ranking(model, test, k_values=[10])
    return {
        "model": name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "precision_10": ranking[10]["precision"],
        "recall_10": ranking[10]["recall"],
    }


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    results = []

    results.append(evaluate_model("GlobalMeanBaseline", GlobalMeanBaseline().fit(train), test))
    results.append(evaluate_model("UserMeanBaseline", UserMeanBaseline().fit(train), test))
    results.append(evaluate_model("ItemMeanBaseline", ItemMeanBaseline().fit(train), test))

    ubcf = UserBasedCF(
        k=ubcf_cfg["k"], similarity=ubcf_cfg["similarity"], min_support=ubcf_cfg["min_support"]
    )
    ubcf.fit(train_matrix)
    results.append(evaluate_model(
        f"UBCF (k={ubcf_cfg['k']}, min_support={ubcf_cfg['min_support']})", ubcf, test
    ))

    ibcf = ItemBasedCF(k=ibcf_cfg["k"], min_support=ibcf_cfg["min_support"])
    ibcf.fit(train)
    results.append(evaluate_model(
        f"IBCF (k={ibcf_cfg['k']}, min_support={ibcf_cfg['min_support']})", ibcf, test
    ))

    svd = SVDModel(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(train)
    results.append(evaluate_model("SVD (n_factors=50, n_epochs=20)", svd, test))

    results_df = pd.DataFrame(results)[
        ["model", "rmse", "mae", "precision_10", "recall_10"]
    ].sort_values("rmse", ascending=True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}\n")

    print("===== 6-model comparison, sorted by RMSE ascending =====")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
