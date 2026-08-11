"""
===============================================================================
File: data_loader.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description:
------------
This module is responsible for loading the dataset downloaded from Kaggle,
validating it, performing an initial inspection, and generating a basic
dataset profiling report.

Research References
-------------------
[1] Lessmann et al. (2015)
    Benchmarking State-of-the-Art Classification Algorithms
    for Credit Scoring.

[2] Hand & Henley (1997)
    Statistical Classification Methods in Consumer Credit Scoring.

Dissertation Mapping
--------------------
Section 6.2 : Data Collection
Section 7.3 : Dataset Identification & Analysis

===============================================================================
"""

import logging
import pandas as pd

from config import (
    CSV_PATH,
    LOG_FILE,
    REPORT_DIR
)

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Data Loader Class
# =============================================================================

class LoanDatasetLoader:

    """
    Loads and profiles the loan approval dataset.
    """

    def __init__(self):

        self.file_path = CSV_PATH
        self.df = None

    # -------------------------------------------------------------------------
    # Remap Columns
    # -------------------------------------------------------------------------

    def remap_columns(self, df):
        """
        Renames nikhil1e9/loan-default columns to match pipeline expectations.

        Default column meaning:
            0 = loan not defaulted = Approved  -> Loan_Status = Y
            1 = loan defaulted     = Rejected  -> Loan_Status = N
        """

        df = df.rename(
            columns={
                "LoanID"         : "Loan_ID",
                "Income"         : "ApplicantIncome",
                "LoanTerm"       : "Loan_Amount_Term",
                "CreditScore"    : "Credit_History",
                "MaritalStatus"  : "Married",
                "HasDependents"  : "Dependents",
                "EmploymentType" : "Self_Employed",
            }
        )

        df["Loan_Status"] = df["Default"].map(
            {0: "Y", 1: "N"}
        )

        df.drop(
            columns=["Default"],
            inplace=True
        )

        return df

    # -------------------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------------------

    def load_dataset(self):

        print("=" * 70)
        print("Loading Dataset...")
        print("=" * 70)

        self.df = pd.read_csv(self.file_path)

        self.df = self.remap_columns(self.df)

        logging.info("Dataset loaded successfully.")

        print("Dataset Loaded Successfully.\n")

        return self.df

    # -------------------------------------------------------------------------
    # Dataset Overview
    # -------------------------------------------------------------------------

    def dataset_summary(self):

        print("=" * 70)
        print("Dataset Summary")
        print("=" * 70)

        print(f"Rows    : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")

        print("\nColumn Names\n")
        print(list(self.df.columns))

        print("\nData Types\n")
        print(self.df.dtypes)

        logging.info("Dataset summary generated.")

    # -------------------------------------------------------------------------
    # Missing Values
    # -------------------------------------------------------------------------

    def missing_values(self):

        print("=" * 70)
        print("Missing Values")
        print("=" * 70)

        missing = self.df.isnull().sum()

        print(missing)

        return missing

    # -------------------------------------------------------------------------
    # Duplicate Records
    # -------------------------------------------------------------------------

    def duplicate_records(self):

        duplicates = self.df.duplicated().sum()

        print("=" * 70)
        print("Duplicate Records")
        print("=" * 70)

        print(duplicates)

        return duplicates

    # -------------------------------------------------------------------------
    # Numerical Columns
    # -------------------------------------------------------------------------

    def numerical_columns(self):

        numerical = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        print("=" * 70)
        print("Numerical Features")
        print("=" * 70)

        print(numerical)

        return numerical

    # -------------------------------------------------------------------------
    # Categorical Columns
    # -------------------------------------------------------------------------

    def categorical_columns(self):

        categorical = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        print("=" * 70)
        print("Categorical Features")
        print("=" * 70)

        print(categorical)

        return categorical

    # -------------------------------------------------------------------------
    # Statistical Summary
    # -------------------------------------------------------------------------

    def statistical_summary(self):

        print("=" * 70)
        print("Statistical Summary")
        print("=" * 70)

        print(self.df.describe(include="all"))

        logging.info("Statistical summary generated.")

    # -------------------------------------------------------------------------
    # Save Dataset Profile
    # -------------------------------------------------------------------------

    def save_profile(self):

        profile = pd.DataFrame({

            "Column":

                self.df.columns,

            "DataType":

                self.df.dtypes.astype(str),

            "MissingValues":

                self.df.isnull().sum().values,

            "UniqueValues":

                self.df.nunique().values

        })

        output_path = REPORT_DIR / "dataset_profile.csv"

        profile.to_csv(output_path, index=False)

        print("\nDataset profile saved.")

        print(output_path)

        logging.info("Dataset profile exported.")

    # -------------------------------------------------------------------------
    # Execute Complete Data Inspection
    # -------------------------------------------------------------------------

    def run(self):

        self.load_dataset()

        self.dataset_summary()

        self.missing_values()

        self.duplicate_records()

        self.numerical_columns()

        self.categorical_columns()

        self.statistical_summary()

        self.save_profile()

        return self.df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    loader = LoanDatasetLoader()

    df = loader.run()
    