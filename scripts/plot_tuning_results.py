"""Plot RMSE and Precision@10 vs k, one line per min_support, for UBCF and IBCF."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = os.path.join("reports", "hyperparameter_tuning_results.csv")
FIGURES_DIR = os.path.join("reports", "figures")


def plot_metric_vs_k(df: pd.DataFrame, model: str, metric: str, ylabel: str, out_path: str) -> None:
    subset = df[df["model"] == model]

    fig, ax = plt.subplots(figsize=(7, 5))
    for min_support, group in subset.groupby("min_support"):
        group = group.sort_values("k")
        ax.plot(group["k"], group[metric], marker="o", label=f"min_support={min_support}")

    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model}: {ylabel} vs k")
    ax.legend(title="min_support")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_metric_vs_k(
        df, "UBCF", "rmse", "RMSE",
        os.path.join(FIGURES_DIR, "ubcf_rmse_vs_k.png"),
    )
    plot_metric_vs_k(
        df, "UBCF", "precision_10", "Precision@10",
        os.path.join(FIGURES_DIR, "ubcf_precision_vs_k.png"),
    )
    plot_metric_vs_k(
        df, "IBCF", "rmse", "RMSE",
        os.path.join(FIGURES_DIR, "ibcf_rmse_vs_k.png"),
    )
    plot_metric_vs_k(
        df, "IBCF", "precision_10", "Precision@10",
        os.path.join(FIGURES_DIR, "ibcf_precision_vs_k.png"),
    )


if __name__ == "__main__":
    main()
