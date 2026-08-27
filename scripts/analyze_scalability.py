"""Scalability analysis: how UBCF/IBCF fit time and prediction latency scale with data size."""

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
RESULTS_PATH = os.path.join("reports", "scalability_results.csv")
LATENCY_PATH = os.path.join("reports", "scalability_prediction_latency.txt")

SAMPLE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEED = 42
N_LATENCY_CALLS = 20


def sample_train(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Randomly sample a fraction of rows from train, without replacement.

    Args:
        train: Full training ratings DataFrame.
        fraction: Fraction of rows to keep, in (0, 1]. 1.0 returns train unchanged.
        seed: Random seed for reproducible sampling.

    Returns:
        Sampled DataFrame with the same columns as train.
    """
    if fraction >= 1.0:
        return train
    rng = np.random.default_rng(seed)
    n = int(len(train) * fraction)
    idx = rng.choice(train.index.to_numpy(), size=n, replace=False)
    return train.loc[idx].reset_index(drop=True)


def time_fit(model_class, model_kwargs: dict, fit_data):
    """Fit a model and return (fitted_model, wall_clock_seconds)."""
    model = model_class(**model_kwargs)
    start = time.perf_counter()
    model.fit(fit_data)
    elapsed = time.perf_counter() - start
    return model, elapsed


def measure_predict_latency(model, user_id: int, movie_id: int, n_calls: int) -> float:
    """Average predict() wall-clock latency in milliseconds over n_calls repeated calls."""
    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        model.predict(user_id, movie_id)
        times.append(time.perf_counter() - start)
    return float(np.mean(times) * 1000)


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)

    ubcf_kwargs = {
        "k": ubcf_cfg["k"], "similarity": ubcf_cfg["similarity"], "min_support": ubcf_cfg["min_support"]
    }
    ibcf_kwargs = {"k": ibcf_cfg["k"], "min_support": ibcf_cfg["min_support"]}

    results = []
    full_models = {}

    for fraction in SAMPLE_FRACTIONS:
        sampled = sample_train(train, fraction, SEED)
        n_ratings = len(sampled)

        print(f"[UBCF] Fitting at fraction={fraction} (n_ratings={n_ratings})...", flush=True)
        matrix = build_user_item_matrix(sampled)
        model, elapsed = time_fit(UserBasedCF, ubcf_kwargs, matrix)
        print(f"  done in {elapsed:.2f}s", flush=True)
        results.append({"model": "UBCF", "fraction": fraction, "n_ratings": n_ratings, "fit_time_s": elapsed})
        if fraction == 1.0:
            full_models["UBCF"] = model

        print(f"[IBCF] Fitting at fraction={fraction} (n_ratings={n_ratings})...", flush=True)
        model, elapsed = time_fit(ItemBasedCF, ibcf_kwargs, sampled)
        print(f"  done in {elapsed:.2f}s", flush=True)
        results.append({"model": "IBCF", "fraction": fraction, "n_ratings": n_ratings, "fit_time_s": elapsed})
        if fraction == 1.0:
            full_models["IBCF"] = model

    results_df = pd.DataFrame(results)[["model", "fraction", "n_ratings", "fit_time_s"]]
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}\n")
    print("===== Scalability fit-time results =====")
    print(results_df.to_string(index=False))

    print(f"\nMeasuring single predict() latency ({N_LATENCY_CALLS} calls, full data)...")

    ubcf_model = full_models["UBCF"]
    ubcf_user_id = int(ubcf_model._ratings_matrix.index[0])
    ubcf_movie_id = int(ubcf_model._ratings_matrix.columns[0])
    ubcf_avg_ms = measure_predict_latency(ubcf_model, ubcf_user_id, ubcf_movie_id, N_LATENCY_CALLS)
    print(f"UBCF avg latency: {ubcf_avg_ms:.4f} ms")

    ibcf_model = full_models["IBCF"]
    ibcf_user_id = int(ibcf_model._users[0])
    ibcf_movie_id = int(ibcf_model._movies[0])
    ibcf_avg_ms = measure_predict_latency(ibcf_model, ibcf_user_id, ibcf_movie_id, N_LATENCY_CALLS)
    print(f"IBCF avg latency: {ibcf_avg_ms:.4f} ms")

    lines = [
        "Prediction latency (full training data, 20 repeated predict() calls, same user/movie pair)\n",
        "\n",
        f"UBCF: avg {ubcf_avg_ms:.4f} ms/call "
        f"(user_id={ubcf_user_id}, movie_id={ubcf_movie_id}, n={N_LATENCY_CALLS})\n",
        f"IBCF: avg {ibcf_avg_ms:.4f} ms/call "
        f"(user_id={ibcf_user_id}, movie_id={ibcf_movie_id}, n={N_LATENCY_CALLS})\n",
    ]
    os.makedirs(os.path.dirname(LATENCY_PATH), exist_ok=True)
    with open(LATENCY_PATH, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"\nSaved latency results to {LATENCY_PATH}")


if __name__ == "__main__":
    main()
