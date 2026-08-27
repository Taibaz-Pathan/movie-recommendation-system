"""Plot fit time vs training set size for UBCF and IBCF."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = os.path.join("reports", "scalability_results.csv")
OUTPUT_PATH = os.path.join("reports", "figures", "scalability_curve.png")


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, color in [("UBCF", "steelblue"), ("IBCF", "darkorange")]:
        subset = df[df["model"] == model_name].sort_values("n_ratings")
        ax.plot(
            subset["n_ratings"],
            subset["fit_time_s"],
            marker="o",
            label=model_name,
            color=color,
        )

    ax.set_xlabel("Training set size (n_ratings)")
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Scalability: Model Fit Time vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
