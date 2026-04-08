# pipeline.py
#
# Loads the raw CSV, cleans it, and produces train/test splits.
#
# A few things worth explaining about the cleaning decisions:

# 1. We cap outliers instead of deleting rows.
#    Deleting outliers is tempting but wrong here. A customer with
#    revolving utilisation of 300% is a real person making real decisions.
#    Dropping them biases the model away from exactly the risky customers
#    we care most about. Capping at a sensible ceiling keeps them in
#    without letting one extreme value dominate tree splits.
#
# 2. The value 98 in past-due columns is a data quality sentinel.
#    It means "unknown" or "not applicable", not 98 actual late payments.
#    264 rows have this. Replace with 0 (conservative — treat as unknown = clean).
#
# 3. MonthlyIncome is missing for 19.8% of rows.
#    We impute using median income within age quintiles, not the global median.
#    A 25-year-old and a 55-year-old have very different income distributions.
#    Flat median imputation understates older earners and overstates younger ones.

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_FILE, TRAIN_FILE, TEST_FILE, TARGET,
    RAW_FEATURES, TEST_SIZE, RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    log.info(f"Loading {path}")
    df = pd.read_csv(path, index_col=0)
    log.info(f"{len(df):,} rows, {df.shape[1]} columns")

    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Revolving utilisation: should be [0, 1]. Values above 1 can happen
    # (over-limit fees), but 3321 rows exceed 1.0. Cap at 1.
    df["RevolvingUtilizationOfUnsecuredLines"] = (
        df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    )

    # Age: one record has age=0. That's a data entry error.
    # Set to NaN so the imputation step handles it consistently.
    df.loc[df["age"] < 18, "age"] = np.nan

    # DebtRatio: values above 1000 are almost certainly unit errors
    # (monthly debt entered as annual, or similar). Cap at 1000.
    df["DebtRatio"] = df["DebtRatio"].clip(0, 1000)

    # MonthlyIncome: cap at 99th percentile (~$35k/month).
    # 70 people show incomes above $100k/month — likely data errors.
    p99 = df["MonthlyIncome"].quantile(0.99)
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(0, p99)

    # Past-due counters: sentinel value 98 means "unknown", not 98 late payments.
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]:
        df[col] = df[col].replace(98, 0).clip(0, 20)

    log.info("Outlier capping done")
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age: one row, global median is fine
    df["age"] = df["age"].fillna(df["age"].median())

    # MonthlyIncome: age-quintile median
    df["_age_q"] = pd.qcut(df["age"], q=5, labels=False, duplicates="drop")
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
        df.groupby("_age_q")["MonthlyIncome"].transform("median")
    ).fillna(df["MonthlyIncome"].median())
    df.drop(columns=["_age_q"], inplace=True)

    # NumberOfDependents: global median (which is 0.0)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    remaining = df[RAW_FEATURES].isnull().sum().sum()
    log.info(f"Imputation done. Remaining missing in features: {remaining}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _cap_outliers(df)
    df = _impute_missing(df)

    # Fix dtypes — imputation can cast ints to float
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "age",
    ]:
        df[col] = df[col].round().astype(int)

    log.info(f"Cleaning complete. Shape: {df.shape}")
    return df


def split(df: pd.DataFrame):
    # Stratify on the target so both splits have the same ~6.7% default rate.
    # Without this, random chance could give you a test set with 5% or 9% defaults,
    # making evaluation metrics not comparable across runs.
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    log.info(
        f"Train: {len(train):,} rows ({train[TARGET].mean()*100:.1f}% default) | "
        f"Test: {len(test):,} rows ({test[TARGET].mean()*100:.1f}% default)"
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_pipeline(save: bool = True):
    df = load_raw()
    df = clean(df)
    train, test = split(df)

    if save:
        TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(TRAIN_FILE, index=False)
        test.to_csv(TEST_FILE, index=False)
        log.info(f"Splits saved to {TRAIN_FILE.parent}")

    return train, test


if __name__ == "__main__":
    train, test = run_pipeline()
    print(train.head(3))
