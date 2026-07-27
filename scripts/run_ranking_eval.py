"""Compare UBCF and IBCF ranking quality using Precision/Recall/F1@K."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.evaluation.metrics import evaluate_ranking
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
K_VALUES = [5, 10, 20]
RELEVANCE_THRESHOLD = 4.0


def print_ranking_table(model_name: str, results: dict) -> None:
    print(f"\n{model_name}")
    print(f"{'K':>5}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    print("-" * 41)
    for k in K_VALUES:
        metrics = results[k]
        print(
            f"{k:>5}{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}{metrics['f1']:>12.4f}"
        )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    ubcf = UserBasedCF(k=20, similarity="pearson", min_support=5)
    ubcf.fit(train_matrix)
    ubcf_results = evaluate_ranking(
        ubcf, test, k_values=K_VALUES, relevance_threshold=RELEVANCE_THRESHOLD
    )

    ibcf = ItemBasedCF(k=20, min_support=5)
    ibcf.fit(train)
    ibcf_results = evaluate_ranking(
        ibcf, test, k_values=K_VALUES, relevance_threshold=RELEVANCE_THRESHOLD
    )

    print_ranking_table("UBCF (User-Based Collaborative Filtering)", ubcf_results)
    print_ranking_table("IBCF (Item-Based Collaborative Filtering)", ibcf_results)


if __name__ == "__main__":
    main()
