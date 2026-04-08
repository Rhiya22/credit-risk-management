# evaluate.py
#
# Everything related to measuring how good (or bad) the models actually are.
#
# The most important thing to understand about evaluation here is that we're
# not trying to maximise a single number. We're trying to make a decision
# about where to draw the line between "approve" and "reject", and that
# decision has asymmetric costs.
#
# Missing a defaulter costs the bank the loan principal.
# Wrongly rejecting a good customer costs the foregone interest margin.
# Those are not the same number — which means the optimal threshold
# is almost never 0.5.

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import COST_FN, COST_FP, FIGURES_DIR, FIGURE_DPI, FIGURE_EXT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# Consistent colours across all plots
COLORS = {
    "logistic_regression": "#4C72B0",
    "random_forest":       "#55A868",
    "gradient_boosting":   "#C44E52",
}


def compute_metrics(y_true, y_prob, threshold=0.5, model_name="Model") -> dict:
    """
    Full metric suite for one model at one threshold.

    We report both ROC-AUC and PR-AUC. Here's why both matter:

    ROC-AUC is the standard banking metric (Basel II uses it).
    But on a 14:1 imbalanced dataset, ROC-AUC can be optimistic because
    it rewards true negatives — which are easy to get right when 93% of
    customers are non-defaulters. You can have a mediocre model that still
    scores well on ROC-AUC just by being good at the majority class.

    PR-AUC only looks at the positive (default) class. It asks: of all the
    customers you flagged as risky, how many actually defaulted? And of all
    the customers who actually defaulted, how many did you catch? This is
    the honest measure of whether your model is doing anything useful.

    The random baseline for PR-AUC is the prevalence rate (~6.7%).
    A model must beat that significantly to be worth deploying.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "model":          model_name,
        "threshold":      threshold,
        "roc_auc":        roc_auc_score(y_true, y_prob),
        "pr_auc":         average_precision_score(y_true, y_prob),
        "recall":         tp / (tp + fn) if (tp + fn) > 0 else 0,
        "precision":      tp / (tp + fp) if (tp + fp) > 0 else 0,
        "specificity":    tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1":             2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "business_cost":  fn * COST_FN + fp * COST_FP,
    }

    log.info(
        f"{model_name} | AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f} | Recall={metrics['recall']:.3f} | "
        f"Precision={metrics['precision']:.3f} | Cost={metrics['business_cost']:,}"
    )
    return metrics


def find_optimal_threshold(y_true, y_prob, criterion="business_cost"):
    """
    Scan every threshold from 0.05 to 0.95 and find the best one.

    The criterion determines what "best" means:
      - business_cost: minimise FN*10 + FP*1 (recommended — reflects real economics)
      - f1: maximise F1 score on the default class
      - recall: maximise recall, subject to precision >= 0.10

    In practice, banks use the business cost criterion and document the
    cost assumptions in the model risk report so regulators can challenge them.
    """
    records = []
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        records.append({
            "threshold":     t,
            "business_cost": fn * COST_FN + fp * COST_FP,
            "f1":            2*tp / (2*tp + fp + fn + 1e-9),
            "recall":        tp / (tp + fn + 1e-9),
            "precision":     tp / (tp + fp + 1e-9),
            "fn": int(fn), "fp": int(fp),
        })

    df = pd.DataFrame(records)

    if criterion == "business_cost":
        best_t = float(df.loc[df["business_cost"].idxmin(), "threshold"])
    elif criterion == "f1":
        best_t = float(df.loc[df["f1"].idxmax(), "threshold"])
    elif criterion == "recall":
        filtered = df[df["precision"] >= 0.10]
        best_t = float(
            filtered.loc[filtered["recall"].idxmax(), "threshold"]
            if len(filtered) else df.loc[df["recall"].idxmax(), "threshold"]
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    log.info(f"Optimal threshold ({criterion}): {best_t:.2f}")
    return best_t, df


def plot_roc_curves(models_dict, X_test, y_test, save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for key, res in models_dict.items():
        y_prob = res["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(key, "steelblue"),
                label=f"{res['name']} (AUC={auc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save: _save(fig, "roc_curves")
    return fig


def plot_pr_curves(models_dict, X_test, y_test, save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1,
               label=f"Random (AP={y_test.mean():.3f})")

    for key, res in models_dict.items():
        y_prob = res["model"].predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, color=COLORS.get(key, "steelblue"),
                label=f"{res['name']} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save: _save(fig, "pr_curves")
    return fig


def plot_confusion_matrix(y_true, y_prob, threshold, model_name, save=True):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")

    labels_cm = ["Non-default\n(Good customer)", "Default\n(Bad customer)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_cm)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_cm)

    # Annotate each cell with count + business interpretation
    cell_text = [
        [f"{tn:,}\n✓ Approved correctly",
         f"{fn:,}\n✗ Missed default\nCost={fn*COST_FN:,}"],
        [f"{fp:,}\n✗ Wrongly rejected\nCost={fp*COST_FP:,}",
         f"{tp:,}\n✓ Caught defaulter"],
    ]
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_text[i][j], ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.2f})")
    plt.tight_layout()
    if save: _save(fig, f"confusion_{model_name.lower().replace(' ','_')}")
    return fig


def plot_threshold_analysis(threshold_df, best_t, model_name, save=True):
    # This chart is what you show to the model risk committee.
    # Left panel: where does the business cost bottom out?
    # Right panel: how do precision and recall trade off across thresholds?
    # The vertical line shows where we chose to draw the line, and why.

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(threshold_df["threshold"], threshold_df["business_cost"],
                 color="#C44E52", lw=2)
    axes[0].axvline(best_t, color="navy", linestyle="--", lw=1.5,
                    label=f"Optimal = {best_t:.2f}")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Business cost")
    axes[0].set_title("Business Cost vs Threshold\n"
                      f"(FN costs {COST_FN}x, FP costs {COST_FP}x)")
    axes[0].legend()

    axes[1].plot(threshold_df["threshold"], threshold_df["recall"],
                 color="#55A868", lw=2, label="Recall")
    axes[1].plot(threshold_df["threshold"], threshold_df["precision"],
                 color="#4C72B0", lw=2, label="Precision")
    axes[1].plot(threshold_df["threshold"], threshold_df["f1"],
                 color="orange", lw=2, linestyle="--", label="F1")
    axes[1].axvline(best_t, color="navy", linestyle="--", lw=1.5,
                    label=f"Optimal = {best_t:.2f}")
    axes[1].set_xlabel("Threshold"); axes[1].set_title("Recall / Precision / F1 vs Threshold")
    axes[1].legend(fontsize=9)

    plt.suptitle(f"Threshold Optimisation — {model_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    if save: _save(fig, f"threshold_{model_name.lower().replace(' ','_')}")
    return fig


def plot_calibration(models_dict, X_test, y_test, save=True):
    # Calibration matters for Basel III.
    # If the model says P(default)=10%, roughly 10% of those customers
    # should actually default. If not, the bank's capital reserves are wrong.

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for key, res in models_dict.items():
        y_prob = res["model"].predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", lw=2,
                color=COLORS.get(key, "steelblue"), label=res["name"])

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual defaults")
    ax.set_title("Calibration — Predicted vs Actual Default Rate")
    ax.legend()
    ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
    plt.tight_layout()
    if save: _save(fig, "calibration")
    return fig


def full_evaluation_report(models_dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for key, res in models_dict.items():
        y_prob = res["model"].predict_proba(X_test)[:, 1]
        opt_t, t_df = find_optimal_threshold(y_test, y_prob)
        m = compute_metrics(y_test, y_prob, threshold=opt_t, model_name=res["name"])
        m["optimal_threshold"] = opt_t
        rows.append(m)
        plot_threshold_analysis(t_df, opt_t, res["name"])
        plot_confusion_matrix(y_test, y_prob, opt_t, res["name"])

    plot_roc_curves(models_dict, X_test, y_test)
    plot_pr_curves(models_dict, X_test, y_test)
    plot_calibration(models_dict, X_test, y_test)

    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    log.info("\nFinal Summary:")
    log.info(summary[["model", "roc_auc", "pr_auc", "recall",
                       "precision", "business_cost", "optimal_threshold"]].to_string(index=False))
    return summary


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIGURE_EXT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info(f"Saved → {path}")
    plt.close(fig)
