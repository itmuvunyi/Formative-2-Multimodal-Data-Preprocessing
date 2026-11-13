"""Labelled EDA and evaluation charts, shared by the notebook and the pipeline."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "figure.dpi": 110,
})


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")


def plot_distributions(tx: pd.DataFrame, social: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    panels = [(tx["purchase_amount"], "Purchase amount", "Amount"),
              (tx["customer_rating"].dropna(), "Customer rating", "Rating (1 to 5)"),
              (social["engagement_score"], "Engagement score", "Score"),
              (social["purchase_interest_score"], "Purchase interest", "Score (1 to 5)")]
    for ax, (values, title, xlabel) in zip(axes, panels):
        ax.hist(values, bins=12, color=BLUE, edgecolor="white", linewidth=2)
        ax.set(title=title, xlabel=xlabel, ylabel="Records")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "eda_distributions.png")
    return fig


def plot_outliers(tx: pd.DataFrame, social: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))
    panels = [(tx["purchase_amount"], "Purchase amount"), (tx["customer_rating"].dropna(), "Customer rating"),
              (social["engagement_score"], "Engagement score"), (social["purchase_interest_score"], "Purchase interest")]
    for ax, (values, title) in zip(axes, panels):
        ax.boxplot(values, vert=False, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#cfe0f6", color=BLUE), medianprops=dict(color=INK, linewidth=2),
                   whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                   flierprops=dict(marker="o", markerfacecolor=ORANGE, markeredgecolor="white", markersize=8))
        ax.set(title=title, yticks=[], xlabel="Value")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "eda_outliers.png")
    return fig


def iqr_outlier_counts(tx: pd.DataFrame, social: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in [("purchase_amount", tx["purchase_amount"]), ("customer_rating", tx["customer_rating"].dropna()),
                    ("engagement_score", social["engagement_score"]),
                    ("purchase_interest_score", social["purchase_interest_score"])]:
        q1, q3 = s.quantile([0.25, 0.75])
        fence = 1.5 * (q3 - q1)
        rows.append({"variable": name, "min": s.min(), "max": s.max(), "lower_fence": q1 - fence,
                     "upper_fence": q3 + fence, "outliers": int(((s < q1 - fence) | (s > q3 + fence)).sum())})
    return pd.DataFrame(rows)


def plot_correlations(merged: pd.DataFrame):
    cols = ["purchase_amount", "customer_rating", "engagement_mean", "interest_mean", "sentiment_mean",
            "social_records", "prior_txn_count", "prior_avg_spend"]
    corr = merged[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Pearson correlation, merged dataset")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    _save(fig, "eda_correlations.png")
    return fig


def plot_target(merged: pd.DataFrame):
    counts = merged["product_category"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].barh(counts.index[::-1], counts.values[::-1], color=BLUE, height=0.65)
    axes[0].set(title="Target classes after the merge", xlabel="Transactions")
    axes[0].grid(axis="y", visible=False)
    order = counts.index.tolist()
    axes[1].boxplot([merged.loc[merged["product_category"] == c, "interest_mean"] for c in order],
                    tick_labels=order, patch_artist=True, boxprops=dict(facecolor="#cfe0f6", color=BLUE),
                    medianprops=dict(color=INK, linewidth=2), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    axes[1].set(title="Purchase interest by product", ylabel="Mean interest score")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "eda_target.png")
    return fig


def plot_confusion(matrix, labels, title: str, filename: str):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 2.2, 0.8 * len(labels) + 1.8))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else INK)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    _save(fig, filename)
    return fig
