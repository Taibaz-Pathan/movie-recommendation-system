"""Generate worked recommendation examples (real movie titles) for a handful of sample users."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.loader import load_movies
from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
OUTPUT_PATH = os.path.join("reports", "recommendation_examples.md")
SEED = 42
N_SAMPLE_USERS = 3
MIN_TRAIN_RATINGS = 15
MAX_TRAIN_RATINGS = 30
N_TOP_RATED = 5
N_RECOMMENDATIONS = 10


def pick_sample_users(train: pd.DataFrame, n: int, seed: int) -> list:
    """Pick n userIds with a moderate rating count (a meaningful, non-sparse taste profile)."""
    rating_counts = train.groupby("userId").size()
    eligible = rating_counts[
        (rating_counts >= MIN_TRAIN_RATINGS) & (rating_counts <= MAX_TRAIN_RATINGS)
    ].index.to_numpy()

    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def top_rated_movies(
    train: pd.DataFrame, user_id: int, movies: pd.DataFrame, n: int
) -> pd.DataFrame:
    user_ratings = (
        train[train["userId"] == user_id]
        .sort_values(["rating", "movieId"], ascending=[False, True])
        .head(n)
    )
    return user_ratings.merge(movies, on="movieId")[
        ["movieId", "title", "genres", "rating"]
    ]


def recommendations_table(recs: list, movies: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(recs, columns=["movieId", "predicted_score"])
    return df.merge(movies, on="movieId")[
        ["movieId", "title", "genres", "predicted_score"]
    ]


def format_table(df: pd.DataFrame, score_col: str = None) -> str:
    lines = []
    if score_col:
        header = f"| movieId | Title | Genres | {score_col} |"
        sep = "|---|---|---|---|"
    else:
        header = "| movieId | Title | Genres | Rating |"
        sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        value_col = row["predicted_score"] if score_col else row["rating"]
        lines.append(
            f"| {row['movieId']} | {row['title']} | {row['genres']} | {value_col:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    movies = load_movies()
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

    sample_users = pick_sample_users(train, N_SAMPLE_USERS, SEED)
    print(f"Sample users (15-30 train ratings): {sample_users}")

    sections = ["# Worked Recommendation Examples\n"]
    sections.append(
        f"Three sample users, each with {MIN_TRAIN_RATINGS}-{MAX_TRAIN_RATINGS} ratings in "
        f"the training set (a meaningful but not overly dense taste profile), selected via "
        f"`numpy.random.default_rng({SEED})`. UBCF: k={ubcf_cfg['k']}, "
        f"similarity={ubcf_cfg['similarity']}, min_support={ubcf_cfg['min_support']}. "
        f"IBCF: k={ibcf_cfg['k']}, min_support={ibcf_cfg['min_support']}.\n"
    )

    for user_id in sample_users:
        print(f"\n===== User {user_id} =====")
        top_rated = top_rated_movies(train, user_id, movies, N_TOP_RATED)
        ubcf_recs = recommendations_table(
            ubcf.recommend(user_id, n=N_RECOMMENDATIONS), movies
        )
        ibcf_recs = recommendations_table(
            ibcf.recommend(user_id, n=N_RECOMMENDATIONS), movies
        )

        overlap_ids = set(ubcf_recs["movieId"]) & set(ibcf_recs["movieId"])
        overlap_titles = ubcf_recs[ubcf_recs["movieId"].isin(overlap_ids)][
            "title"
        ].tolist()

        print(f"Top {N_TOP_RATED} rated movies:")
        print(top_rated.to_string(index=False))
        print(f"\nUBCF top {N_RECOMMENDATIONS} recommendations:")
        print(ubcf_recs.to_string(index=False))
        print(f"\nIBCF top {N_RECOMMENDATIONS} recommendations:")
        print(ibcf_recs.to_string(index=False))
        print(
            f"\nOverlap between UBCF and IBCF: {len(overlap_ids)} movie(s) -- {overlap_titles}"
        )

        section = [f"## User {user_id}\n"]
        section.append(
            f"### Taste profile: top {N_TOP_RATED} highest-rated movies in training\n"
        )
        section.append(format_table(top_rated))
        section.append(f"\n### UBCF top {N_RECOMMENDATIONS} recommendations\n")
        section.append(format_table(ubcf_recs, score_col="Predicted score"))
        section.append(f"\n### IBCF top {N_RECOMMENDATIONS} recommendations\n")
        section.append(format_table(ibcf_recs, score_col="Predicted score"))
        section.append("\n### Overlap\n")
        if overlap_ids:
            section.append(
                f"{len(overlap_ids)} movie(s) appear in both UBCF's and IBCF's top-10: "
                + ", ".join(overlap_titles)
            )
        else:
            section.append(
                "No overlap -- UBCF and IBCF recommended entirely disjoint movie sets."
            )
        section.append("")

        sections.append("\n".join(section))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
