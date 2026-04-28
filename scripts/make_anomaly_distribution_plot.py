"""Generate the modified Z-score distribution figure for the report.

Loads validation residuals, computes per-(building, meter) modified Z-scores,
and saves a histogram with the 3.5 Iglewicz-Hoaglin threshold marked.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "results" / "predictions.csv"
OUT = REPO / "report-latex" / "figures" / "anomaly_distribution.png"

THRESHOLD = 3.5
CONST = 0.6745


def modified_z_per_group(residuals: np.ndarray) -> np.ndarray:
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad == 0:
        return np.zeros_like(residuals)
    return CONST * (residuals - med) / mad


def main() -> None:
    df = pd.read_csv(PRED, usecols=["building_id", "meter", "residual"])
    print(f"Loaded {len(df):,} validation residuals")

    scores = np.empty(len(df), dtype=np.float32)
    pos = 0
    for _, group in df.groupby(["building_id", "meter"], sort=False):
        n = len(group)
        scores[pos : pos + n] = modified_z_per_group(group["residual"].to_numpy())
        pos += n

    abs_scores = np.abs(scores)
    flagged = (abs_scores > THRESHOLD).sum()
    print(f"Anomalies (|M| > {THRESHOLD}): {flagged:,} ({flagged / len(df):.2%})")

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=200)

    bins = np.linspace(0, 12, 80)
    ax.hist(
        abs_scores,
        bins=bins,
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.92,
    )
    n_beyond = (abs_scores > 12).sum()

    ax.axvline(THRESHOLD, color="#C73E1D", linestyle="--", linewidth=1.6, label=f"Threshold = {THRESHOLD}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|M_t|$  (modified Z-score)", fontsize=10)
    ax.set_ylabel("Frequency (log scale)", fontsize=10)
    ax.set_xlim(0, 12)
    ax.tick_params(axis="both", labelsize=9)

    pct = flagged / len(df) * 100
    annotation = (
        f"{flagged:,} hours flagged\n({pct:.2f}% of validation)\n"
        f"{n_beyond:,} hours with $|M_t|>12$"
    )
    ax.text(
        0.97,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#888888", linewidth=0.5),
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.78), fontsize=9, frameon=False)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
