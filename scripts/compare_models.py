"""Fit UBCF and IBCF on the same train/test split and compare RMSE/MAE."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    ubcf = UserBasedCF(k=20, similarity="pearson", min_support=5)
    ubcf.fit(train_matrix)
    ubcf_metrics = ubcf.evaluate(test)

    ibcf = ItemBasedCF(k=20, min_support=5)
    ibcf.fit(train)
    ibcf_metrics = ibcf.evaluate(test)

    print(f"\n{'Model':<10}{'RMSE':>10}{'MAE':>10}{'n_predictions':>16}")
    print("-" * 46)
    for name, metrics in [("UBCF", ubcf_metrics), ("IBCF", ibcf_metrics)]:
        print(
            f"{name:<10}{metrics['rmse']:>10.4f}{metrics['mae']:>10.4f}"
            f"{metrics['n_predictions']:>16}"
        )


if __name__ == "__main__":
    main()
