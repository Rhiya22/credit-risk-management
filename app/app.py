# app/app.py
#
# Streamlit scoring interface.
#
# The point of this app isn't to look good in a demo (though it does).
# The point is to show that the model can be put in front of a credit
# analyst and they can actually use it — enter a customer's details,
# get a probability, understand why.
#
# Run with: streamlit run app/app.py
 
import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
 
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
 
from features import engineer_features, build_Xy
from explain import explain_single_prediction, _risk_category
from config import MODELS_DIR, TARGET, RAW_FILE, RANDOM_STATE
 
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
)
 
st.markdown("""
<style>
.decision-approve { background:#d4edda; color:#155724; padding:14px 20px;
                    border-radius:8px; font-size:1.2rem; font-weight:600; }
.decision-reject  { background:#f8d7da; color:#721c24; padding:14px 20px;
                    border-radius:8px; font-size:1.2rem; font-weight:600; }
.risk-low         { color:#155724; font-weight:600; }
.risk-medium      { color:#856404; font-weight:600; }
.risk-high        { color:#721c24; font-weight:600; }
.risk-veryhigh    { color:#ffffff; background:#842029; padding:2px 8px; border-radius:4px; font-weight:600; }
</style>
""", unsafe_allow_html=True)
 
# Decision threshold from business cost optimisation.
# Gradient Boosting with 10:1 FN:FP cost ratio → 0.09 minimises total cost.
THRESHOLD = 0.09
 
 
def _train_model_now():
    """Train just the Gradient Boosting model + scaler this app actually needs.
 
    Skips Logistic Regression, Random Forest, and cross-validation scoring
    (those exist for the README's model-comparison table, not for scoring
    live customers) so a fresh deployment is ready in well under a minute
    instead of the ~3 minutes the full `src/main.py` pipeline takes.
 
    Requires `data/raw/cs-training.csv` to already be present in the repo
    (it's committed on purpose - see README - since this dataset needs a
    Kaggle login to fetch and can't be auto-downloaded anonymously).
    """
    from pipeline import run_pipeline
    from train import save_model
    from sklearn.ensemble import GradientBoostingClassifier
 
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} not found. This app trains itself on first run, but "
            "needs the raw dataset committed at data/raw/cs-training.csv "
            "(see README for why this one file is an exception to the "
            ".gitignore data rule)."
        )
 
    train_df, test_df = run_pipeline(save=False)
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    X_train, X_test, y_train, y_test, scaler, feat_names = build_Xy(train_fe, test_fe)
 
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
 
    save_model(model, "gradient_boosting", {"features": feat_names})
    joblib.dump({"scaler": scaler, "feature_names": feat_names}, MODELS_DIR / "scaler.joblib")
 
 
@st.cache_resource
def load_model():
    if not (MODELS_DIR / "gradient_boosting.joblib").exists() or not (MODELS_DIR / "scaler.joblib").exists():
        with st.spinner("First-time setup: training the model (only happens once per deployment, ~1 minute)..."):
            try:
                _train_model_now()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Automatic model training failed: {exc}")
                st.stop()
 
    try:
        model_p  = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
        scaler_p = joblib.load(MODELS_DIR / "scaler.joblib")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading model artifacts: {exc}")
        st.stop()
 
    return model_p["model"], scaler_p["scaler"], scaler_p["feature_names"]
 
 
model, scaler, feature_names = load_model()
 
NO_SCALE = {
    TARGET, "CreditLineUtilBucket", "HasRealEstate",
    "IsHighDebtRatio", "AgeGroup", "DelinquencyScore", "TotalDaysPastDue"
}
 
# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Customer Details")
 
with st.sidebar:
    st.subheader("Credit History")
    util = st.slider("Revolving utilisation", 0.0, 1.0, 0.35, 0.01,
                     help="0.70 = 70% of credit limit used. Above 1.0 = over limit.")
    late_30 = st.number_input("30–59 day late payments (last 2 years)", 0, 20, 0)
    late_60 = st.number_input("60–89 day late payments (last 2 years)", 0, 20, 0)
    late_90 = st.number_input("90+ day late payments (last 2 years)", 0, 20, 0,
                               help="Strongest single predictor of default in this model.")
 
    st.subheader("Financial Profile")
    age            = st.slider("Age", 18, 100, 40)
    monthly_income = st.number_input("Monthly income (£)", 0, 500_000, 5_000, 100)
    debt_ratio     = st.slider("Debt ratio (monthly debt / income)", 0.0, 5.0, 0.35, 0.01,
                                help="Above 1.0 = total monthly debt exceeds income.")
    dependents     = st.number_input("Dependents", 0, 20, 1)
 
    st.subheader("Credit Portfolio")
    open_lines  = st.number_input("Open credit lines and loans", 0, 50, 8)
    real_estate = st.number_input("Real estate loans / lines", 0, 20, 1)
 
    score_btn = st.button("Score this customer", use_container_width=True, type="primary")
 
# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Scorer")
st.markdown(
    "Gradient Boosting model trained on 150,000 historical loans. "
    "ROC-AUC 0.87. Threshold set at 0.09 to minimise business cost "
    "(missed defaults cost 10× more than wrongly rejected good customers)."
)
 
if not score_btn:
    st.info("Fill in the customer details in the sidebar and click **Score this customer**.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Gradient Boosting")
    col2.metric("ROC-AUC", "0.8695")
    col3.metric("PR-AUC", "0.4047")
    col4.metric("Default rate caught", "71.2%")
    st.stop()
 
# Build input row
raw_input = {
    "SeriousDlqin2yrs":                    0,
    "RevolvingUtilizationOfUnsecuredLines": util,
    "age":                                  age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30,
    "DebtRatio":                            debt_ratio,
    "MonthlyIncome":                        monthly_income,
    "NumberOfOpenCreditLinesAndLoans":      open_lines,
    "NumberOfTimes90DaysLate":              late_90,
    "NumberRealEstateLoansOrLines":         real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60,
    "NumberOfDependents":                   dependents,
}
 
# Apply exact same pipeline as training
input_fe = engineer_features(pd.DataFrame([raw_input]))
feat_cols = [c for c in input_fe.columns if c != TARGET]
X_in = input_fe[feat_cols].copy()
scale_cols = [c for c in feat_cols if c not in NO_SCALE]
X_in[scale_cols] = scaler.transform(X_in[scale_cols])
X_arr = X_in.values[0]
 
# Score
# Use the engineered (unscaled) values for the plain-English explanation, not
# just the raw sidebar inputs - the explanation templates need engineered
# columns like DelinquencyScore and TotalDaysPastDue, which don't exist in
# raw_input, so passing raw_input alone silently rendered them as "0".
explanation_values = input_fe.iloc[0].to_dict()
expl     = explain_single_prediction(model, X_arr, feat_cols, explanation_values, THRESHOLD)
prob     = expl["probability"]
decision = expl["decision"]
risk_cat = expl["risk_category"]
 
# ── Display ───────────────────────────────────────────────────────────────────
st.markdown("---")
col_l, col_r = st.columns(2)
 
with col_l:
    st.subheader("Result")
 
    css = "decision-approve" if decision == "APPROVE" else "decision-reject"
    icon = "✅" if decision == "APPROVE" else "❌"
    st.markdown(
        f'<div class="{css}">{icon} {decision} — P(default) = {prob:.1%}</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"**Risk category:** {risk_cat}")
    st.markdown(f"**Threshold:** {THRESHOLD:.0%} (business-cost optimised)")
    st.markdown("")
 
    # Simple gauge
    fig, ax = plt.subplots(figsize=(5, 1.0))
    bar_color = "#198754" if prob < 0.10 else "#ffc107" if prob < 0.20 else "#fd7e14" if prob < 0.40 else "#dc3545"
    ax.barh([""], [prob],          color=bar_color, height=0.5)
    ax.barh([""], [1 - prob], left=[prob], color="#e9ecef", height=0.5)
    ax.axvline(THRESHOLD, color="navy", lw=2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(default)")
    ax.text(THRESHOLD + 0.01, 0, f"Cutoff {THRESHOLD:.0%}", fontsize=8, color="navy", va="center")
    ax.text(prob / 2, 0, f"{prob:.1%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white" if prob > 0.15 else "black")
    # Explicit white background (not transparent) so the chart reads
    # correctly regardless of whether the viewer's Streamlit theme is
    # light or dark - a transparent figure inherits the page's dark
    # background, and default matplotlib text/ticks are black, making
    # them invisible in dark mode.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
 
with col_r:
    st.subheader("What drove this score")
 
    contribs = expl["all_contributions"]
    s = pd.Series(contribs).sort_values()
    display = pd.concat([s.head(4), s.tail(5)]).sort_values()
    colors = ["#C44E52" if v > 0 else "#55A868" for v in display.values]
 
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.barh(display.index, display.values, color=colors, alpha=0.85, height=0.6)
    ax2.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, display.values):
        ax2.text(val + (0.002 if val >= 0 else -0.002),
                 bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8)
    ax2.set_xlabel("Contribution to P(default)\nred = increases risk, green = reduces it")
    ax2.set_title("Feature contributions")
    # Same fix as the gauge chart above - explicit white background so
    # labels stay legible in Streamlit's dark theme.
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
 
# Explanation text
st.markdown("---")
st.subheader("Explanation (plain English)")
st.markdown("*This is what goes into the adverse action notice under GDPR Art. 22.*")
for line in expl["plain_english"].split("\n"):
    if line.strip():
        st.markdown(line)
 
# Raw inputs for audit trail
with st.expander("Full input details (for audit)"):
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Raw inputs**")
        st.dataframe(pd.DataFrame(raw_input, index=["value"]).T)
    with cb:
        st.markdown("**After feature engineering**")
        st.dataframe(input_fe[[c for c in input_fe.columns if c != TARGET]].T.rename(columns={0: "value"}))
 
st.markdown("---")
st.caption("Model: Gradient Boosting | Data: Give Me Some Credit (Kaggle) | AUC: 0.8695")
 
