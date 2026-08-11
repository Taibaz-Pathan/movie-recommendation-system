"""Plot RMSE vs training-data availability (cold-start simulation) for UBCF and IBCF."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = os.path.join("reports", "cold_start_results.csv")
OUTPUT_PATH = os.path.join("reports", "figures", "cold_start_curve.png")

# "full" is plotted past the largest truncation level, at a fixed sentinel x-value,
# since it has no single max_ratings count (every user keeps all their ratings).
FULL_BASELINE_X = 40


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, color in [("UBCF", "steelblue"), ("IBCF", "darkorange")]:
        subset = df[df["model"] == model_name].copy()
        subset["x"] = subset["max_ratings"].apply(
            lambda v: FULL_BASELINE_X if v == "full" else int(v)
        )
        subset = subset.sort_values("x")
        ax.plot(subset["x"], subset["rmse"], marker="o", label=model_name, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Max ratings kept per user (truncation level, log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title("Cold-Start Simulation: RMSE vs Training Data Availability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Label the "full" point explicitly since it's not a real truncation count.
    ax.axvline(FULL_BASELINE_X, color="gray", linestyle=":", alpha=0.5)
    ax.text(FULL_BASELINE_X, ax.get_ylim()[1], "full", ha="center", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
