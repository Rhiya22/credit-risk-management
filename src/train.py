# train.py
#
# Three models, increasing complexity. There's a reason we don't just
# jump straight to gradient boosting:
#
# 1. Logistic Regression is the baseline. If your fancy model can't
#    meaningfully beat it, the complexity isn't justified — and regulators
#    will ask exactly that question. LR also gives you coefficients, which
#    are directly interpretable as log-odds. Useful for explaining decisions.
#
# 2. Random Forest shows whether the performance gain from ensembling is
#    worth the interpretability cost. It also gives stable feature importance
#    estimates that are less sensitive to individual tree structure than GB.
#
# 3. Gradient Boosting (GradientBoostingClassifier here, XGBoost in production)
#    is what most banks actually use for PD modelling. Sequential error
#    correction gives better calibration than RF, and it's SHAP-compatible
#    so you can explain individual predictions.
#
# We use fixed best-known hyperparameters here. In production you'd run
# Optuna or a grid search during training, cache the best params, and
# only retrain with those params going forward.

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def get_cv():
    # StratifiedKFold keeps the 6.7% default rate in every fold.
    # Plain KFold on this dataset could produce folds where some have
    # almost no defaults — making AUC estimates noisy and inconsistent.
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def train_logistic_regression(X_train, y_train):
    log.info("Training Logistic Regression...")
    t0 = time.time()

    # C=0.1: moderate regularisation. Without it, LR overfits on the
    # high-cardinality count features (past-due counts, open credit lines).
    # class_weight='balanced': multiplies each defaulter's loss by ~14
    # (the inverse class frequency), so the model doesn't just learn
    # to always predict "no default".
    model = LogisticRegression(
        C=0.1,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    cv_auc = cross_val_score(
        model, X_train, y_train,
        cv=get_cv(), scoring="roc_auc", n_jobs=-1
    ).mean()

    log.info(f"LR done — CV AUC: {cv_auc:.4f} | {time.time()-t0:.1f}s")
    return model, cv_auc


def train_random_forest(X_train, y_train):
    log.info("Training Random Forest...")
    t0 = time.time()

    # max_depth=10 and min_samples_leaf=20 prevent the forest from
    # memorising individual customers. We want it to learn patterns,
    # not noise. Without these constraints RF easily overfits on this
    # dataset because a few customers have very extreme feature combinations.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cv_auc = cross_val_score(
        model, X_train, y_train,
        cv=get_cv(), scoring="roc_auc", n_jobs=-1
    ).mean()

    log.info(f"RF done — CV AUC: {cv_auc:.4f} | {time.time()-t0:.1f}s")
    return model, cv_auc


def train_gradient_boosting(X_train, y_train):
    log.info("Training Gradient Boosting...")
    t0 = time.time()

    # subsample=0.8: use 80% of rows per tree (stochastic GB).
    # Reduces variance and speeds up training without hurting performance much.
    # learning_rate=0.1 with 150 trees: standard starting point.
    # Going lower LR + more trees improves performance but takes longer.
    #
    # If you have XGBoost installed, swap this for:
    #   from xgboost import XGBClassifier
    #   model = XGBClassifier(
    #       n_estimators=300, max_depth=4, learning_rate=0.05,
    #       subsample=0.8, colsample_bytree=0.8,
    #       scale_pos_weight=14,  # replaces class_weight='balanced'
    #       eval_metric='auc', random_state=RANDOM_STATE
    #   )

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 3-fold CV here (vs 5-fold for others) because GB is slow.
    # Still gives a reliable estimate — variance across folds is low
    # once you have enough data (120k rows here).
    cv_auc = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1
    ).mean()

    log.info(f"GB done — CV AUC: {cv_auc:.4f} | {time.time()-t0:.1f}s")
    return model, cv_auc


def save_model(model, name: str, metadata: dict = None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "metadata": metadata or {}}, path)
    log.info(f"Saved → {path}")
    return path


def load_model(name: str):
    payload = joblib.load(MODELS_DIR / f"{name}.joblib")
    return payload["model"], payload["metadata"]


def train_all(X_train, y_train, feature_names: list) -> dict:
    results = {}

    lr, lr_auc = train_logistic_regression(X_train, y_train)
    results["logistic_regression"] = {"model": lr, "name": "Logistic Regression", "cv_auc": lr_auc}
    save_model(lr, "logistic_regression", {"cv_auc": lr_auc, "features": feature_names})

    rf, rf_auc = train_random_forest(X_train, y_train)
    results["random_forest"] = {"model": rf, "name": "Random Forest", "cv_auc": rf_auc}
    save_model(rf, "random_forest", {"cv_auc": rf_auc, "features": feature_names})

    gb, gb_auc = train_gradient_boosting(X_train, y_train)
    results["gradient_boosting"] = {"model": gb, "name": "Gradient Boosting", "cv_auc": gb_auc}
    save_model(gb, "gradient_boosting", {"cv_auc": gb_auc, "features": feature_names})

    log.info("\nCV AUC Summary:")
    for res in results.values():
        log.info(f"  {res['name']:<25} {res['cv_auc']:.4f}")

    return results
