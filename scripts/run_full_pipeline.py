"""End-to-end pipeline: raw data -> preprocessing -> train all 6 models -> evaluate
-> save results."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.loader import load_all
from src.data.preprocessor import (
    build_user_item_matrix,
    filter_ratings,
    train_test_split_per_user,
)
from src.evaluation.metrics import evaluate_ranking
from src.models.baselines import GlobalMeanBaseline, ItemMeanBaseline, UserMeanBaseline
from src.models.ibcf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

RESULTS_PATH = os.path.join("reports", "pipeline_run_results.csv")


def evaluate_model(name: str, model, test: pd.DataFrame) -> dict:
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
    filter_cfg = config.get("filter", {})
    split_cfg = config["split"]
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    # ===== Stage 1: Load raw ratings + movies =====
    print("===== Stage 1: Load raw data =====", flush=True)
    start = time.perf_counter()
    data = load_all()
    ratings, movies = data["ratings"], data["movies"]
    print(f"  Loaded {len(ratings):,} raw ratings, {len(movies):,} movies", flush=True)
    print(f"  Stage 1 took {time.perf_counter() - start:.2f}s\n", flush=True)

    # ===== Stage 2: Preprocess (filter + split) =====
    print("===== Stage 2: Preprocess (filter + train/test split) =====", flush=True)
    start = time.perf_counter()
    filtered = filter_ratings(
        ratings,
        min_user_ratings=filter_cfg.get("min_user_ratings", 20),
        min_movie_ratings=filter_cfg.get("min_movie_ratings", 20),
    )
    train, test = train_test_split_per_user(
        filtered,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    train_matrix = build_user_item_matrix(train)
    print(f"  Stage 2 took {time.perf_counter() - start:.2f}s\n", flush=True)

    # ===== Stage 3: Train all 6 models =====
    print("===== Stage 3: Train all 6 models =====", flush=True)
    start = time.perf_counter()

    models = {}

    models["GlobalMeanBaseline"] = GlobalMeanBaseline().fit(train)
    models["UserMeanBaseline"] = UserMeanBaseline().fit(train)
    models["ItemMeanBaseline"] = ItemMeanBaseline().fit(train)

    ubcf_name = f"UBCF (k={ubcf_cfg['k']}, min_support={ubcf_cfg['min_support']})"
    models[ubcf_name] = UserBasedCF(
        k=ubcf_cfg["k"],
        similarity=ubcf_cfg["similarity"],
        min_support=ubcf_cfg["min_support"],
    ).fit(train_matrix)

    ibcf_name = f"IBCF (k={ibcf_cfg['k']}, min_support={ibcf_cfg['min_support']})"
    models[ibcf_name] = ItemBasedCF(
        k=ibcf_cfg["k"], min_support=ibcf_cfg["min_support"]
    ).fit(train)

    models["SVD (n_factors=50, n_epochs=20)"] = SVDModel(
        n_factors=50, n_epochs=20, random_state=42
    ).fit(train)

    print(f"  Trained: {', '.join(models.keys())}", flush=True)
    print(f"  Stage 3 took {time.perf_counter() - start:.2f}s\n", flush=True)

    # ===== Stage 4: Evaluate all 6 models =====
    print("===== Stage 4: Evaluate all 6 models =====", flush=True)
    start = time.perf_counter()

    results = []
    for name, model in models.items():
        print(f"  Evaluating {name}...", flush=True)
        results.append(evaluate_model(name, model, test))

    print(f"  Stage 4 took {time.perf_counter() - start:.2f}s\n", flush=True)

    # ===== Stage 5: Save results =====
    print("===== Stage 5: Save results =====", flush=True)
    start = time.perf_counter()

    results_df = pd.DataFrame(results)[
        ["model", "rmse", "mae", "precision_10", "recall_10"]
    ].sort_values("rmse", ascending=True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"  Saved results to {RESULTS_PATH}", flush=True)
    print(f"  Stage 5 took {time.perf_counter() - start:.2f}s\n", flush=True)

    print("===== Full pipeline results, sorted by RMSE ascending =====")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
