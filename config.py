"""
===============================================================================
Explainable AI for Loan Approval using Interpretable Machine Learning Models
===============================================================================
------------
Description:
------------
This file contains all project configurations including dataset download,
directory creation, random seed initialization, and reusable constants.

Research References
-------------------
[1] Lessmann et al. (2015)
    Benchmarking State-of-the-Art Classification Algorithms
    for Credit Scoring.

[2] Breiman (2001)
    Random Forests.

[3] Chen & Guestrin (2016)
    XGBoost: A Scalable Tree Boosting System.

Purpose
-------
Keeping all configurable parameters in one place improves maintainability
and reproducibility of research experiments.

===============================================================================
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import kagglehub

warnings.filterwarnings("ignore")

# =============================================================================
# Random Seed
# =============================================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# Project Root Directory
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

# =============================================================================
# Folder Structure
# =============================================================================

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
REPORT_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"

directories = [
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    LOG_DIR,
]

for directory in directories:
    directory.mkdir(exist_ok=True)

# =============================================================================
# Kaggle Dataset
# =============================================================================
"""
Dataset: nikhil1e9/loan-default
Source : https://www.kaggle.com/datasets/nikhil1e9/loan-default
Rows   : ~255,000

Features

LoanID
Age
Income
LoanAmount
CreditScore
MonthsEmployed
NumCreditLines
InterestRate
LoanTerm
DTIRatio
Education
EmploymentType
MaritalStatus
HasMortgage
HasDependents
LoanPurpose
HasCoSigner
Default  ->  remapped to Loan_Status (0=Y approved, 1=N defaulted)
"""

KAGGLE_DATASET = "nikhil1e9/loan-default"
import os


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# Folder Paths
# ============================================================

DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data"
)


OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models"
)


REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports"
)


# ============================================================
# Dataset Paths
# ============================================================

RAW_DATA_PATH = os.path.join(
    DATA_FOLDER,
    "loan_train.csv"
)


PROCESSED_DATA_PATH = os.path.join(
    DATA_FOLDER,
    "processed_loan_data.csv"
)

def download_dataset():
    """
    Downloads latest dataset from Kaggle.

    Reference:
    ----------
    Dissertation Section 6.2
    Data Collection

    Returns
    -------
    dataset_path : Path
    """

    print("=" * 70)
    print("Downloading latest dataset from Kaggle...")
    print("=" * 70)

    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)

    print("\nDataset downloaded successfully.")
    print(f"Location : {dataset_path}\n")

    return Path(dataset_path)


DATASET_PATH = download_dataset()

# =============================================================================
# CSV File Detection
# =============================================================================

csv_files = list(DATASET_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("Dataset CSV not found")

CSV_PATH = csv_files[0]

print("=" * 70)
print("Dataset Found")
print("=" * 70)
print(CSV_PATH)

# =============================================================================
# Train Test Split
# =============================================================================

TEST_SIZE = 0.20

# =============================================================================
# Cross Validation
# =============================================================================

CV = 5

# =============================================================================
# Evaluation Metrics
# =============================================================================

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
]

# =============================================================================
# Models Used
# =============================================================================

MODELS = {

    "Logistic Regression":

        {
            "code": "LR",
            "interpretability": "High"
        },

    "Decision Tree":

        {
            "code": "DT",
            "interpretability": "High"
        },

    "Random Forest":

        {
            "code": "RF",
            "interpretability": "Medium"
        },

    "XGBoost":

        {
            "code": "XGB",
            "interpretability": "Low"
        }

}

# =============================================================================
# Output Files
# =============================================================================

MODEL_COMPARISON_FILE = OUTPUT_DIR / "model_comparison.csv"

FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"

SHAP_OUTPUT = OUTPUT_DIR / "shap"

LIME_OUTPUT = OUTPUT_DIR / "lime"

SHAP_OUTPUT.mkdir(exist_ok=True)

LIME_OUTPUT.mkdir(exist_ok=True)

# =============================================================================
# Logging
# =============================================================================

LOG_FILE = LOG_DIR / "loan_xai.log"

# =============================================================================
# Banner
# =============================================================================

print("=" * 70)
print("Explainable AI for Loan Approval")
print("Dissertation")
print("=" * 70)

print(f"Project Root : {ROOT_DIR}")
print(f"Data Folder  : {DATA_DIR}")
print(f"Output Folder: {OUTPUT_DIR}")
print(f"Model Folder : {MODEL_DIR}")
print("=" * 70)
