# Credit Risk Modeling — Predicting Loan Default

I built this project because I wanted to understand how banks actually use machine learning — not the textbook version, but what really goes into a credit decision. As someone coming from a data science background and looking to move into banking/fintech ML roles, I kept seeing "credit risk" mentioned in job descriptions without really understanding what it meant in practice. This was my way of figuring that out.

The dataset is Give Me Some Credit from Kaggle — 150,000 real borrowers, where the goal is to predict who will default on a loan within 2 years.

---

## What I built

A full end-to-end pipeline:
- Data cleaning and feature engineering
- Three models (Logistic Regression, Random Forest, Gradient Boosting)
- Proper evaluation using ROC-AUC and Precision-Recall curves
- A Streamlit app where you can enter a customer's details and get a live default probability with an explanation of why

The Streamlit app was the part I didn't expect to enjoy as much as I did. Seeing the model actually make a decision in a UI — approve or reject, with reasons — made it feel like a real thing rather than just numbers in a notebook.

---

## Results

| Model | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|
| Gradient Boosting | 0.8695 | 0.4047 | 71.2% |
| Random Forest | 0.8664 | 0.3961 | 66.1% |
| Logistic Regression | 0.8614 | 0.3891 | 69.3% |

The model catches 71% of actual defaulters. That sounds low until you realise the alternative — no model — catches 0%.

One thing that took me a while to understand: accuracy is a useless metric here. 93% of customers don't default, so predicting "no default" for everyone gives you 93% accuracy and is completely worthless. ROC-AUC and Precision-Recall are the right metrics for this problem.

---

## The threshold problem

This was probably the most interesting thing I learned. The model outputs a probability — say 0.23. Whether that becomes "approve" or "reject" depends on where you draw the line, and the right place to draw it isn't 0.5.

A missed default (false negative) costs the bank the full loan amount. A wrongly rejected customer (false positive) costs only the lost interest on that loan. Those aren't the same cost, so the optimal threshold isn't in the middle. I built a cost matrix with a 10:1 ratio (missing a default is 10x more costly than wrongly rejecting someone) and let the math find the threshold that minimises total cost. It landed at 0.09 for Gradient Boosting — much more aggressive than 0.5, but economically correct.

---

## What drives default risk

From feature importance on the best model:

1. **Delinquency severity score** — an engineered feature I created by weighting past-due events (90-day lates count 3x more than 30-day lates, same way FICO does it). This single feature accounts for 64.5% of the model's signal. If someone has missed payments before, they're very likely to miss them again.

2. **Revolving credit utilisation** — people using more than 70% of their available credit are under financial stress. This was the second biggest signal.

3. **Age** — follows a U-shape. Young borrowers with thin credit history and older borrowers on fixed income both show higher risk than the middle age groups.

4. **Income per dependent** — I engineered this from monthly income and number of dependents. Someone earning £5,000/month with 4 kids has less financial buffer than someone earning the same with no dependents.

---

## The hardest part

Honestly, the class imbalance. 93.3% of customers don't default, which means if you're not careful the model just learns to predict "no default" always and looks great on paper. Getting the model to actually pay attention to the 6.7% minority class — the defaulters, the ones that actually matter — required handling the imbalance explicitly through class weighting and then rethinking every metric I was using to evaluate it.

---

## Project structure

```
├── notebooks/
│   └── credit_risk_complete.ipynb   # full walkthrough
├── src/
│   ├── config.py       # all paths and constants
│   ├── pipeline.py     # data cleaning
│   ├── features.py     # feature engineering
│   ├── eda.py          # exploratory analysis
│   ├── train.py        # model training
│   ├── evaluate.py     # metrics and plots
│   ├── explain.py      # feature importance and explanations
│   └── main.py         # runs everything
├── app/
│   └── app.py          # streamlit scoring app
└── requirements.txt
```

---

## How to run it

```bash
git clone https://github.com/Rhiya22/credit-risk-modeling.git
cd credit-risk-modeling
pip install -r requirements.txt
```

Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and put it in `data/raw/`.

Run the notebook: open `notebooks/credit_risk_complete.ipynb` and run all cells.

Then launch the app:
```bash
streamlit run app/app.py
```

---

## What I'd do next

I'd add population stability monitoring — a way to detect when the incoming applicant population starts shifting away from what the model was trained on, which is when models go stale and need retraining. I'd also want to do a proper fairness analysis, checking whether the model performs consistently across different age groups. In a real bank these aren't optional extras, they're regulatory requirements — so understanding them felt like the natural next step.

---

## Stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit, joblib
