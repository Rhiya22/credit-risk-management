# main.py
#
# Runs the whole pipeline from raw data to saved models and evaluation charts.
#
# Usage:
#   python src/main.py              # run everything
#   python src/main.py --stage eda
#   python src/main.py --stage train
#   python src/main.py --stage evaluate
#   python src/main.py --stage explain

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, DATA_PROC, TARGET
from pipeline import run_pipeline
from features import engineer_features, build_Xy
from eda import run_eda
from train import train_all
from evaluate import full_evaluation_report, find_optimal_threshold
from explain import (
    get_tree_feature_importance,
    get_permutation_importance,
    plot_feature_importance,
    explain_single_prediction,
    plot_local_explanation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_full():
    log.info("Step 1/6 — Data pipeline")
    train_df, test_df = run_pipeline(save=True)

    log.info("Step 2/6 — EDA")
    run_eda(train_df)

    log.info("Step 3/6 — Feature engineering")
    train_fe = engineer_features(train_df)
    test_fe  = engineer_features(test_df)
    X_train, X_test, y_train, y_test, scaler, feat_names = build_Xy(train_fe, test_fe)
    log.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    log.info("Step 4/6 — Model training")
    models = train_all(X_train, y_train, feat_names)

    # Save scaler for the app
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feat_names},
                MODELS_DIR / "scaler.joblib")

    log.info("Step 5/6 — Evaluation")
    summary = full_evaluation_report(models, X_test, y_test)

    log.info("Step 6/6 — Explainability")
    best_model = models["gradient_boosting"]["model"]

    imp = get_tree_feature_importance(best_model, feat_names)
    log.info("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))
    plot_feature_importance(imp, "Gradient Boosting")

    gb_prob = best_model.predict_proba(X_test)[:, 1]
    opt_t, _ = find_optimal_threshold(y_test, gb_prob)

    high_risk_idx = np.argmax(gb_prob)
    sample_X   = X_test[high_risk_idx]
    sample_raw = {feat: float(sample_X[i]) for i, feat in enumerate(feat_names)}

    expl = explain_single_prediction(
        best_model, sample_X, feat_names, sample_raw, opt_t
    )
    print("\n" + expl["plain_english"])
    plot_local_explanation(expl["all_contributions"], "Gradient Boosting", expl["probability"])

    # Business summary
    best = summary.iloc[0]
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              RESULTS SUMMARY                             ║
╠══════════════════════════════════════════════════════════╣
║  Best model:    {best['model']:<42}║
║  ROC-AUC:       {best['roc_auc']:.4f}                               ║
║  PR-AUC:        {best['pr_auc']:.4f}                               ║
║  Recall:        {best['recall']:.4f}  (defaulters caught)            ║
║  Threshold:     {best['optimal_threshold']:.2f}    (business-cost optimised)       ║
║  Business cost: {int(best['business_cost']):,}                              ║
╚══════════════════════════════════════════════════════════╝
""")

    log.info("Done. Models in models/, charts in reports/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "pipeline", "eda", "train", "evaluate", "explain"])
    args = parser.parse_args()

    if args.stage == "all":
        run_full()
        sys.exit(0)

    # Partial runs
    if args.stage == "pipeline":
        run_pipeline(save=True)
        sys.exit(0)

    # Everything else needs processed data
    train_df = pd.read_csv(DATA_PROC / "train.csv")
    test_df  = pd.read_csv(DATA_PROC / "test.csv")

    if args.stage == "eda":
        run_eda(train_df)

    elif args.stage in ("train", "evaluate", "explain"):
        train_fe = engineer_features(train_df)
        test_fe  = engineer_features(test_df)
        X_train, X_test, y_train, y_test, scaler, feat_names = build_Xy(train_fe, test_fe)

        if args.stage == "train":
            train_all(X_train, y_train, feat_names)

        else:
            # Load saved models
            models = {}
            for key, label in [
                ("logistic_regression", "Logistic Regression"),
                ("random_forest", "Random Forest"),
                ("gradient_boosting", "Gradient Boosting"),
            ]:
                p = joblib.load(MODELS_DIR / f"{key}.joblib")
                models[key] = {"model": p["model"], "name": label}

            if args.stage == "evaluate":
                full_evaluation_report(models, X_test, y_test)

            elif args.stage == "explain":
                best = models["gradient_boosting"]["model"]
                imp  = get_tree_feature_importance(best, feat_names)
                plot_feature_importance(imp, "Gradient Boosting")

                y_prob = best.predict_proba(X_test)[:, 1]
                opt_t, _ = find_optimal_threshold(y_test, y_prob)
                idx   = np.argmax(y_prob)
                expl  = explain_single_prediction(
                    best, X_test[idx], feat_names,
                    {f: float(X_test[idx][i]) for i, f in enumerate(feat_names)},
                    opt_t
                )
                print(expl["plain_english"])
