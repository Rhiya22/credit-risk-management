# eda.py
#
# Exploratory analysis. The point of this isn't to make pretty charts —
# it's to answer specific questions before touching the model:
#
# 1. How bad is the class imbalance?
#    (Spoiler: 14:1. Need to handle it explicitly.)
#
# 2. Do the features actually separate defaulters from non-defaulters?
#    (If not, no model will help.)
#
# 3. Is the relationship between features and default monotone and sensible?
#    (If utilisation is negatively correlated with default, something is wrong.)
#
# 4. Are features too correlated with each other?
#    (Multicollinearity kills logistic regression coefficients.)
#
# These charts are what you'd present to the Chief Credit Officer before
# starting to build the model. They validate that the data matches business
# intuition and flags anything weird before you've invested weeks of work.

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import TARGET, FIGURES_DIR, FIGURE_DPI, FIGURE_EXT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

PALETTE = {0: "#4C72B0", 1: "#C44E52"}
LABELS  = {0: "Non-Default", 1: "Default"}


def plot_target_distribution(df, save=True):
    counts = df[TARGET].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [LABELS[k] for k in counts.index],
        counts.values,
        color=[PALETTE[k] for k in counts.index],
        width=0.5, alpha=0.85,
    )
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", fontsize=11)

    ax.set_title("Class Distribution — 14:1 imbalance\nMust handle before training")
    ax.set_ylabel("Customers")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    if save: _save(fig, "target_distribution")
    return fig


def plot_feature_distributions(df, save=True):
    # Overlay KDE (or histogram for low-cardinality features) for
    # defaulters vs non-defaulters on each feature.
    # Features where the distributions are well-separated = useful signal.
    # Features where they overlap completely = probably not useful.

    features = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "DebtRatio", "MonthlyIncome",
        "NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for ax, feat in zip(axes, features):
        for label, color in PALETTE.items():
            subset = df[df[TARGET] == label][feat].dropna()
            subset = subset[subset < subset.quantile(0.99)]
            if subset.nunique() > 5:
                try:
                    subset.plot.kde(ax=ax, color=color, lw=2,
                                    label=LABELS[label], bw_method=0.3)
                    continue
                except Exception:
                    pass
            ax.hist(subset, bins=20, color=color, alpha=0.5,
                    density=True, label=LABELS[label])
        ax.set_title(feat.replace("NumberOf", "N "), fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Feature Distributions by Default Status",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save: _save(fig, "feature_distributions")
    return fig


def plot_default_rate_by_feature(df, save=True):
    # Decile analysis. This is how credit analysts actually think:
    # "customers in the top utilisation quintile default at X times the rate
    # of the bottom quintile". If that relationship is monotone and strong,
    # the feature belongs in the model.

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tmp = df.copy()

    tmp["util_q"] = pd.qcut(
        tmp["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1), q=5,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    )
    tmp.groupby("util_q", observed=True)[TARGET].mean().mul(100).plot.bar(
        ax=axes[0], color="#C44E52", alpha=0.8
    )
    axes[0].set_title("Default Rate by Credit Utilisation")
    axes[0].set_ylabel("Default rate (%)"); axes[0].tick_params(axis="x", rotation=30)

    tmp["age_q"] = pd.qcut(tmp["age"], q=5, duplicates="drop")
    tmp.groupby("age_q", observed=True)[TARGET].mean().mul(100).plot.bar(
        ax=axes[1], color="#4C72B0", alpha=0.8
    )
    axes[1].set_title("Default Rate by Age Band")
    axes[1].set_ylabel("Default rate (%)"); axes[1].tick_params(axis="x", rotation=30)

    tmp["NumberOfTimes90DaysLate"].clip(0, 5).pipe(
        lambda s: tmp.groupby(s)[TARGET].mean().mul(100)
    ).plot.bar(ax=axes[2], color="#55A868", alpha=0.8)
    axes[2].set_title("Default Rate by 90-Day Late Count")
    axes[2].set_ylabel("Default rate (%)")

    for ax in axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.suptitle("Default Rate by Feature — validates monotone risk relationships",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save: _save(fig, "default_rate_by_feature")
    return fig


def plot_correlation_heatmap(df, save=True):
    # Two things to look for:
    # 1. High correlation between features (multicollinearity) — hurts LR
    # 2. Suspiciously high correlation with the target — possible data leakage
    #
    # NumberOfTimes90DaysLate and the other past-due columns will be
    # correlated with each other. That's expected and why we engineer
    # a single DelinquencyScore instead of using all three raw columns.

    corr = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
        ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, square=True,
    )
    ax.set_title("Correlation Heatmap\nLook for multicollinearity and leakage")
    plt.tight_layout()
    if save: _save(fig, "correlation_heatmap")
    return fig


def run_eda(df):
    log.info("Running EDA...")
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_default_rate_by_feature(df)
    plot_correlation_heatmap(df)
    log.info(f"EDA done. Charts saved to {FIGURES_DIR}")


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIGURE_EXT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info(f"Saved → {path}")
    plt.close(fig)
