# config.py
#
# Central place for every path, constant, and model parameter.
#
# The reason this file exists: if you hardcode paths inside pipeline.py
# or train.py, you end up hunting through 8 files when you want to change
# the test split size. One file, one change.

from pathlib import Path

# Project root — wherever this file lives, go one level up
ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RAW_FILE  = DATA_RAW / "cs-training.csv"
TRAIN_FILE = DATA_PROC / "train.csv"
TEST_FILE  = DATA_PROC / "test.csv"

# Output directories
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Target column
TARGET = "SeriousDlqin2yrs"  # 1 = defaulted within 2 years, 0 = didn't

# The raw feature columns as they arrive in the CSV
RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Split params
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Business cost matrix
# A missed default (false negative) costs the bank the full loan principal.
# A wrongly rejected customer (false positive) costs only lost interest margin.
# The 10:1 ratio is a reasonable approximation — adjust based on actual loan sizes.
COST_FN = 10   # false negative cost
COST_FP = 1    # false positive cost

# Cross-validation
CV_FOLDS = 5

# Plotting
FIGURE_DPI = 150
FIGURE_EXT = "png"
