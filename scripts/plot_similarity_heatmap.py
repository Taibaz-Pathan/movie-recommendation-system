"""Heatmaps of a random sample of the UBCF user-user and IBCF item-item similarity matrices."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
FIGURES_DIR = os.path.join("reports", "figures")
SAMPLE_SIZE = 30
SEED = 42


def plot_heatmap(
    matrix: np.ndarray, title: str, axis_label: str, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_title(title)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)

    fig.colorbar(im, ax=ax, label="Similarity")
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    train_matrix = build_user_item_matrix(train)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- UBCF: user-user similarity ---
    print(f"Fitting UBCF (k={ubcf_cfg['k']}, min_support={ubcf_cfg['min_support']})...")
    ubcf = UserBasedCF(
        k=ubcf_cfg["k"],
        similarity=ubcf_cfg["similarity"],
        min_support=ubcf_cfg["min_support"],
    )
    ubcf.fit(train_matrix)

    rng = np.random.default_rng(SEED)
    sampled_users = rng.choice(ubcf._users, size=SAMPLE_SIZE, replace=False)
    ubcf_submatrix = (
        ubcf._sim_matrix.loc[sampled_users, sampled_users].fillna(0.0).to_numpy()
    )

    plot_heatmap(
        ubcf_submatrix,
        "Sample User-User Similarity Matrix (UBCF, n=30)",
        "User index",
        os.path.join(FIGURES_DIR, "ubcf_similarity_heatmap.png"),
    )

    # --- IBCF: item-item similarity ---
    print(f"Fitting IBCF (k={ibcf_cfg['k']}, min_support={ibcf_cfg['min_support']})...")
    ibcf = ItemBasedCF(k=ibcf_cfg["k"], min_support=ibcf_cfg["min_support"])
    ibcf.fit(train)

    rng = np.random.default_rng(SEED)
    sampled_movies = rng.choice(ibcf._movies, size=SAMPLE_SIZE, replace=False)
    movie_indices = [ibcf._get_movie_index(int(m)) for m in sampled_movies]
    ibcf_submatrix = ibcf._sim_matrix[np.ix_(movie_indices, movie_indices)]

    plot_heatmap(
        ibcf_submatrix,
        "Sample Item-Item Similarity Matrix (IBCF, n=30)",
        "Movie index",
        os.path.join(FIGURES_DIR, "ibcf_similarity_heatmap.png"),
    )


if __name__ == "__main__":
    main()
