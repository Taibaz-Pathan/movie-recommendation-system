"""Diagnostic: investigate why UserBasedCF.predict() produces raw values outside [0.5, 5.0].

Used to confirm that prediction saturation above 5.0 is expected behaviour
of the unbounded mean-centred (Resnick-style) formula -- roughly 1.07% of
predictions, concentrated in high-mean users -- not a defect. Kept for
reproducibility/reference rather than deleted.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.models.ubcf import UserBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
SAMPLE_USER_ID = 1


def raw_predict(model: UserBasedCF, user_id: int, movie_id: int):
    """Replicates UserBasedCF.predict()'s logic but returns the raw
    (pre-clip) prediction plus the intermediate arithmetic, instead of
    the clipped float that predict() returns.

    Returns None if the fallback path (< min_k neighbours) is taken.
    """
    user_mean = model._user_means[user_id]

    sim_scores = model._sim_matrix[user_id].drop(index=user_id).dropna()
    rated_by = model._ratings_matrix[movie_id].dropna().index
    common = sim_scores.index.intersection(rated_by)
    if len(common) < model.min_k:
        return None

    sim_scores = sim_scores[common]
    top_k_idx = sim_scores.abs().nlargest(model.k).index
    sim_scores = sim_scores[top_k_idx]
    sim_scores = sim_scores[sim_scores > 0]

    if len(sim_scores) < model.min_k:
        return None

    neighbour_ratings = model._ratings_matrix.loc[sim_scores.index, movie_id]
    neighbour_means = model._user_means[sim_scores.index]
    deviations = neighbour_ratings - neighbour_means

    numerator = (sim_scores * deviations).sum()
    denominator = sim_scores.abs().sum()
    raw_prediction = user_mean + numerator / denominator

    return {
        "user_mean": user_mean,
        "n_neighbours": len(sim_scores),
        "sim_scores": sim_scores,
        "neighbour_ratings": neighbour_ratings,
        "neighbour_means": neighbour_means,
        "deviations": deviations,
        "numerator": numerator,
        "denominator": denominator,
        "raw_prediction": raw_prediction,
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_matrix = build_user_item_matrix(train)

    model = UserBasedCF(k=20, similarity="pearson", min_support=5)
    model.fit(train_matrix)

    # --- Step 1: per-prediction detail for SAMPLE_USER_ID ---
    user_row = model._ratings_matrix.loc[SAMPLE_USER_ID]
    unrated = user_row[user_row.isna()].index.tolist()

    print(
        f"===== User {SAMPLE_USER_ID}: user_mean = {model._user_means[SAMPLE_USER_ID]:.4f} =====\n"
    )
    print(
        f"{'movieId':>8}{'n_neigh':>9}{'raw_pred':>11}{'numerator':>12}{'denominator':>13}"
    )
    print("-" * 53)

    saturated_rows = []
    for movie_id in unrated:
        result = raw_predict(model, SAMPLE_USER_ID, movie_id)
        if result is None:
            continue
        raw_pred = result["raw_prediction"]
        if raw_pred > 5.0 or raw_pred < 0.5:
            saturated_rows.append((movie_id, result))
            print(
                f"{movie_id:>8}{result['n_neighbours']:>9}{raw_pred:>11.4f}"
                f"{result['numerator']:>12.4f}{result['denominator']:>13.4f}"
            )

    n_checked = sum(
        1 for m in unrated if raw_predict(model, SAMPLE_USER_ID, m) is not None
    )
    print(
        f"\n{len(saturated_rows)} / {n_checked} predictions for user {SAMPLE_USER_ID} "
        f"fall outside [0.5, 5.0] before clipping."
    )

    # --- Step 2: full arithmetic breakdown for 3 saturated examples ---
    print("\n\n===== Full arithmetic for 3 saturated examples =====")
    for movie_id, result in saturated_rows[:3]:
        print(f"\n--- user {SAMPLE_USER_ID}, movie {movie_id} ---")
        print(f"user_mean            : {result['user_mean']:.4f}")
        print(f"n_neighbours used    : {result['n_neighbours']}")
        detail = pd.DataFrame(
            {
                "sim": result["sim_scores"],
                "neighbour_rating": result["neighbour_ratings"],
                "neighbour_mean": result["neighbour_means"],
                "deviation": result["deviations"],
                "sim_x_deviation": result["sim_scores"] * result["deviations"],
            }
        )
        print(detail.to_string())
        print(f"numerator (sum sim*deviation) : {result['numerator']:.4f}")
        print(f"denominator (sum |sim|)       : {result['denominator']:.4f}")
        print(
            f"numerator / denominator       : {result['numerator'] / result['denominator']:.4f}"
        )
        print(
            f"raw_prediction = user_mean + (numerator/denominator) "
            f"= {result['user_mean']:.4f} + {result['numerator'] / result['denominator']:.4f} "
            f"= {result['raw_prediction']:.4f}"
        )

    # --- Step 3: full test-set sweep ---
    print("\n\n===== Full test-set sweep: how common is clip-saturation? =====")
    n_total = 0
    n_saturated_high = 0
    n_saturated_low = 0
    n_few_neighbours_saturated = 0  # saturated AND used <= 3 neighbours
    sparse_user_saturated = 0  # saturated AND user has <= 20 ratings in train
    sparse_movie_saturated = 0  # saturated AND movie has <= 20 ratings in train

    user_rating_counts = model._ratings_matrix.count(axis=1)
    movie_rating_counts = model._ratings_matrix.count(axis=0)

    for _, row in test.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        if user_id not in model._ratings_matrix.index:
            continue
        if movie_id not in model._ratings_matrix.columns:
            continue

        result = raw_predict(model, user_id, movie_id)
        if result is None:
            continue

        n_total += 1
        raw_pred = result["raw_prediction"]
        if raw_pred > 5.0:
            n_saturated_high += 1
        elif raw_pred < 0.5:
            n_saturated_low += 1
        else:
            continue

        if result["n_neighbours"] <= 3:
            n_few_neighbours_saturated += 1
        if user_rating_counts[user_id] <= 20:
            sparse_user_saturated += 1
        if movie_rating_counts[movie_id] <= 20:
            sparse_movie_saturated += 1

    n_saturated_total = n_saturated_high + n_saturated_low
    pct = 100 * n_saturated_total / n_total if n_total else 0.0

    print(f"Total genuine (non-fallback) predictions checked : {n_total}")
    print(f"Raw prediction > 5.0                              : {n_saturated_high}")
    print(f"Raw prediction < 0.5                               : {n_saturated_low}")
    print(
        f"Total saturated                                    : {n_saturated_total} ({pct:.2f}%)"
    )
    print(
        f"  ...of which used <= 3 neighbours                 : {n_few_neighbours_saturated}"
    )
    print(
        f"  ...of which user has <= 20 ratings in train       : {sparse_user_saturated}"
    )
    print(
        f"  ...of which movie has <= 20 ratings in train      : {sparse_movie_saturated}"
    )


if __name__ == "__main__":
    main()
