# features.py
#
# Feature engineering. This is where most of the modelling insight lives.
#
# The raw features are fine but they don't capture how credit analysts
# actually think about risk. A DebtRatio of 0.8 means nothing without
# knowing whether that's $800 on $1000 income or $8000 on $10000 income.
# The ratio is the same; the risk isn't.
#
# We engineer features around the "5 Cs of credit" framework that
# underwriters have used for decades:
#   - Character  → how have you handled debt in the past?
#   - Capacity   → can you afford the repayments?
#   - Capital    → what assets do you have?
#   - Collateral → what can the bank claim if you default?
#   - Conditions → external factors (we proxy this with age/utilisation bands)

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import TARGET, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EPS = 1e-6  # small constant to prevent log(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Character: delinquency history ────────────────────────────────────────
    #
    # The three separate past-due columns are related but carry different severity.
    # Rather than feeding three correlated columns to the model, we compress them
    # into a single severity score.
    #
    # The weights (1x, 2x, 3x) mirror how FICO scores penalise delinquency:
    # missing 3 payments in a row is far worse than missing 1.

    df["TotalDaysPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    df["DelinquencyScore"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + df["NumberOfTimes90DaysLate"] * 3
    )

    # ── Capacity: income features ─────────────────────────────────────────────
    #
    # Raw MonthlyIncome is right-skewed — a handful of high earners pull the
    # distribution. Log transform brings it closer to normal, which helps
    # logistic regression and makes tree splits more meaningful across
    # the bulk of the distribution (not just the tail).
    #
    # IncomePerDependent is a proxy for disposable income. Someone earning
    # $5000/month with 4 kids has less financial buffer than someone earning
    # $5000/month with no dependents. The raw income column misses this.

    df["LogMonthlyIncome"]   = np.log(df["MonthlyIncome"] + EPS)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # ── Capital: debt burden ──────────────────────────────────────────────────
    #
    # Same log-transform rationale as income — DebtRatio has a long right tail.
    # IsHighDebtRatio is a simple flag because the relationship isn't linear:
    # going from 0.4 to 0.5 matters a lot; going from 1.5 to 2.0 less so.

    df["LogDebtRatio"]    = np.log(df["DebtRatio"] + EPS)
    df["IsHighDebtRatio"] = (df["DebtRatio"] > 0.5).astype(int)

    # ── Collateral: real estate ───────────────────────────────────────────────
    #
    # Owning property signals stability and gives the bank something to claim
    # against. It's one of the strongest negative predictors of default.

    df["HasRealEstate"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(int)

    # ── Conditions: utilisation buckets ──────────────────────────────────────
    #
    # The relationship between utilisation and default isn't linear.
    # Below 30%: fine. 30-70%: watch. Above 70%: strong warning signal.
    # Above 100%: over the credit limit — very high risk.
    #
    # Bucketing lets tree models learn these thresholds without having to
    # discover them from scratch in every split.

    df["CreditLineUtilBucket"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-np.inf, 0.30, 0.70, 1.00, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # ── Age bands ─────────────────────────────────────────────────────────────
    #
    # Raw age has a U-shaped relationship with default: young borrowers with
    # thin credit history and older borrowers on fixed/reduced income both
    # show elevated risk. Age bands let the model learn this shape without
    # assuming age has a linear effect.

    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    log.info(f"Feature engineering complete — {df.shape[1]} total columns")
    return df


# Columns that don't need scaling (binary flags, ordinal buckets)
_NO_SCALE = {
    TARGET,
    "CreditLineUtilBucket",
    "HasRealEstate",
    "IsHighDebtRatio",
    "AgeGroup",
    "DelinquencyScore",
    "TotalDaysPastDue",
}


def build_Xy(train_df: pd.DataFrame, test_df: pd.DataFrame, scale: bool = True):
    """
    Split into X/y arrays and scale continuous features.

    The scaler is fit on training data only. Fitting on the full dataset
    leaks test-set statistics (mean, std) into the training process.
    It's a subtle form of data leakage that inflates reported performance.

    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    feature_cols = [c for c in train_df.columns if c != TARGET]
    scale_cols   = [c for c in feature_cols if c not in _NO_SCALE]

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
        log.info(f"Scaled {len(scale_cols)} continuous features")

    return X_train.values, X_test.values, y_train, y_test, scaler, feature_cols
