# explain.py
#
# Model explainability — global and local.
#
# "Global" means: across the whole portfolio, which features matter most?
# "Local" means: for this specific customer, why did the model score them this way?
#
# The local explanation is the one that actually matters in practice.
# When a customer calls to ask why their loan was rejected, you need to
# tell them something specific. "The model gave you a low score" is not
# a compliant response under GDPR Article 22.
#
# For production, use SHAP (pip install shap). It gives exact Shapley
# value attribution — the mathematically correct way to distribute a
# model's prediction across features.
#
# What we implement here is an ablation-based approximation:
# set each feature to zero, measure how much the probability changes,
# attribute that change to the feature. It's not exact but it's fast,
# dependency-free, and gives directionally correct results.

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import FIGURES_DIR, FIGURE_DPI, FIGURE_EXT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def get_tree_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Impurity-based feature importance from the tree itself.

    Fast, but has a known bias: it overstates high-cardinality features
    because they get more opportunities to split. DebtRatio (continuous)
    will tend to look more important than HasRealEstate (binary) even if
    they contribute equally to predictions.

    Use permutation_feature_importance() for a more reliable ranking,
    especially when comparing engineered features of different types.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model doesn't have feature_importances_. Use permutation importance instead.")

    imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp["pct"] = imp["importance"] / imp["importance"].sum() * 100
    return imp


def get_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """
    Permutation importance on the held-out test set.

    For each feature: randomly shuffle its values and measure how much
    the model's AUC drops. A feature that's genuinely useful will cause
    a large drop when shuffled. A feature the model ignores will cause
    almost no change.

    This is more reliable than impurity-based importance because:
    - It uses the test set (no overfitting bias)
    - It reflects actual prediction impact, not tree structure
    - It works consistently across feature types

    n_repeats=10 means we shuffle each feature 10 times and average the
    drops — reduces noise from a single unlucky shuffle.
    """
    log.info(f"Computing permutation importance (n_repeats={n_repeats})...")
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp_df, model_name, top_n=15, save=True):
    df = imp_df.head(top_n).sort_values("importance", ascending=True)
    has_std = "std" in df.columns

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        df["feature"], df["importance"],
        xerr=df["std"].values if has_std else None,
        color="#4C72B0", alpha=0.85, height=0.65,
        error_kw={"elinewidth": 1, "capsize": 3},
    )
    ax.set_xlabel("Importance (mean AUC decrease when shuffled)")
    ax.set_title(f"Top {top_n} Features — {model_name}")
    for bar, val in zip(bars, df["importance"].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    if save: _save(fig, f"feature_importance_{model_name.lower().replace(' ','_')}")
    return fig


def explain_single_prediction(model, X_instance, feature_names, raw_values, threshold):
    """
    Explain a single prediction using feature ablation.

    For each feature, we zero it out and see how much the predicted
    probability changes. Features that cause a large positive change
    when zeroed are ones that were pushing the probability up (increasing
    risk). Features that cause a large negative change when zeroed were
    keeping the probability down (reducing risk).

    This isn't as precise as SHAP, but the direction and rough magnitude
    are correct, and it requires no additional dependencies.

    To upgrade to exact SHAP explanations:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance.reshape(1, -1))
        # shap_values[1] gives the per-feature contribution to P(default)
    """
    prob = model.predict_proba(X_instance.reshape(1, -1))[0, 1]
    decision = "REJECT" if prob >= threshold else "APPROVE"
    risk_cat = _risk_category(prob)

    contribs = {}
    for i, feat in enumerate(feature_names):
        ablated = X_instance.copy()
        ablated[i] = 0
        ablated_prob = model.predict_proba(ablated.reshape(1, -1))[0, 1]
        contribs[feat] = prob - ablated_prob

    top5 = dict(sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:5])
    plain = _build_explanation(raw_values, top5, prob, risk_cat, decision, threshold)

    return {
        "probability":      round(prob, 4),
        "risk_category":    risk_cat,
        "decision":         decision,
        "top_risk_factors": top5,
        "plain_english":    plain,
        "all_contributions": contribs,
    }


def _risk_category(prob: float) -> str:
    # These bands are illustrative. A real bank would calibrate them to
    # actual loss rates in each band, then name them (Prime, Near-Prime, Subprime).
    if prob < 0.10:  return "Low Risk"
    if prob < 0.20:  return "Medium Risk"
    if prob < 0.40:  return "High Risk"
    return "Very High Risk"


# Plain-English templates for the most important features.
# In a real system these would be reviewed by a compliance officer
# before going into customer-facing communications.
_TEMPLATES = {
    "DelinquencyScore": (
        "history of {val:.0f} weighted delinquency events — "
        "past payment failures are the strongest predictor of future default"
    ),
    "RevolvingUtilizationOfUnsecuredLines": (
        "revolving credit utilisation at {val:.0%} — "
        "high utilisation signals the applicant is relying heavily on credit"
    ),
    "NumberOfTimes90DaysLate": (
        "{val:.0f} occurrence(s) of 90+ day delinquency — "
        "this is a severe delinquency flag and significantly increases assessed risk"
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "{val:.0f} occurrence(s) of 30-59 day late payments in the past 2 years"
    ),
    "TotalDaysPastDue": (
        "{val:.0f} total delinquency events across all severity levels"
    ),
    "MonthlyIncome": (
        "monthly income of ${val:,.0f} — lower income reduces capacity to absorb shocks"
    ),
    "DebtRatio": (
        "debt-to-income ratio of {val:.2f} — monthly debt obligations relative to income"
    ),
    "age": (
        "applicant age of {val:.0f} — associated with elevated risk in this segment"
    ),
}


def _build_explanation(raw_values, top_factors, prob, risk_cat, decision, threshold):
    lines = [
        f"Decision:        {decision}",
        f"P(default):      {prob:.1%}",
        f"Risk category:   {risk_cat}",
        f"Threshold used:  {threshold:.2f}",
        "",
        "Key factors in this assessment:",
    ]
    for i, (feat, contrib) in enumerate(top_factors.items(), 1):
        raw_val = raw_values.get(feat, 0)
        direction = "increases" if contrib > 0 else "reduces"
        if feat in _TEMPLATES:
            try:
                desc = _TEMPLATES[feat].format(val=raw_val)
            except (KeyError, ValueError):
                desc = f"{feat} contributed to the score"
        else:
            desc = f"{feat} (value: {raw_val:.2f})"
        lines.append(f"  {i}. {desc.capitalize()} [{direction} risk by {abs(contrib):.4f}]")
    return "\n".join(lines)


def plot_local_explanation(contribs, model_name, probability, save=True):
    s = pd.Series(contribs).sort_values()
    display = pd.concat([s.head(5), s.tail(5)]).sort_values()

    colors = ["#C44E52" if v > 0 else "#55A868" for v in display.values]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(display.index, display.values, color=colors, alpha=0.85, height=0.65)
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, display.values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8
        )
    ax.set_xlabel("Contribution to P(default)\nred = increases risk, green = reduces risk")
    ax.set_title(f"Prediction Explanation — {model_name}\nP(default) = {probability:.1%}")
    plt.tight_layout()
    if save: _save(fig, f"local_explanation_{model_name.lower().replace(' ','_')}")
    return fig


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIGURE_EXT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info(f"Saved → {path}")
    plt.close(fig)
