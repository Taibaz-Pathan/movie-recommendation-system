"""Bar-chart comparison of RMSE and Precision@10 across all 5 models for the final report."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = os.path.join("reports", "full_model_comparison.csv")
OUTPUT_PATH = os.path.join("reports", "figures", "final_model_comparison.png")

BASELINE_COLOR = "gray"
UBCF_COLOR = "steelblue"
IBCF_COLOR = "darkorange"


def bar_color(model_name: str) -> str:
    if model_name.startswith("UBCF"):
        return UBCF_COLOR
    if model_name.startswith("IBCF"):
        return IBCF_COLOR
    return BASELINE_COLOR


def add_value_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    fig, (ax_rmse, ax_precision) = plt.subplots(1, 2, figsize=(14, 6))

    # --- RMSE: ascending, lower is better ---
    rmse_df = df.sort_values("rmse", ascending=True)
    colors = [bar_color(m) for m in rmse_df["model"]]
    bars = ax_rmse.bar(rmse_df["model"], rmse_df["rmse"], color=colors)
    add_value_labels(ax_rmse, bars)
    ax_rmse.set_title("RMSE by model (lower is better)")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xticks(range(len(rmse_df)))
    ax_rmse.set_xticklabels(rmse_df["model"], rotation=30, ha="right")
    ax_rmse.grid(True, axis="y", alpha=0.3)

    # --- Precision@10: descending, higher is better ---
    precision_df = df.sort_values("precision_10", ascending=False)
    colors = [bar_color(m) for m in precision_df["model"]]
    bars = ax_precision.bar(
        precision_df["model"], precision_df["precision_10"], color=colors
    )
    add_value_labels(ax_precision, bars)
    ax_precision.set_title("Precision@10 by model (higher is better)")
    ax_precision.set_ylabel("Precision@10")
    ax_precision.set_xticks(range(len(precision_df)))
    ax_precision.set_xticklabels(precision_df["model"], rotation=30, ha="right")
    ax_precision.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final Model Comparison: RMSE vs Precision@10", fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
