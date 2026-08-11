"""
===============================================================================
File: feature_engineering.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Creates additional business-oriented features for loan approval prediction.

Research Motivation
-------------------
Credit scoring literature shows that carefully engineered features often improve
predictive performance and interpretability.

Created Features
----------------
1. DebtToIncomeRatio
2. LoanPerTerm
3. IncomePerAge
4. CreditScoreBand

===============================================================================
"""

import os
import numpy as np
import pandas as pd

from data_loader import LoanDatasetLoader


class FeatureEngineer:

    def __init__(self):

        loader = LoanDatasetLoader()
        self.df = loader.load_dataset()

    # ------------------------------------------------------------------

    def create_features(self):

        print("=" * 70)
        print("Starting Feature Engineering...")
        print("=" * 70)

        df = self.df.copy()

        # --------------------------------------------------------------
        # Remove Loan_ID
        # --------------------------------------------------------------

        if "Loan_ID" in df.columns:
            df.drop(columns=["Loan_ID"], inplace=True)

        # --------------------------------------------------------------
        # Debt to Income Ratio
        # --------------------------------------------------------------

        df["ApplicantIncome"] = df["ApplicantIncome"].fillna(
            df["ApplicantIncome"].median()
        )

        df["LoanAmount"] = df["LoanAmount"].fillna(
            df["LoanAmount"].median()
        )

        df["DebtToIncomeRatio"] = (
            df["LoanAmount"] /
            (df["ApplicantIncome"] + 1)
        )

        # --------------------------------------------------------------
        # Loan Per Term  (monthly repayment burden)
        # --------------------------------------------------------------

        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
            df["Loan_Amount_Term"].median()
        )

        df["LoanPerTerm"] = (
            df["LoanAmount"] /
            (df["Loan_Amount_Term"] + 1)
        )

        # --------------------------------------------------------------
        # Income Per Age
        # --------------------------------------------------------------

        if "Age" in df.columns:

            df["Age"] = df["Age"].fillna(
                df["Age"].median()
            )

            df["IncomePerAge"] = (
                df["ApplicantIncome"] /
                (df["Age"] + 1)
            )

        # --------------------------------------------------------------
        # Credit Score Band
        # --------------------------------------------------------------

        if "Credit_History" in df.columns:

            df["CreditScoreBand"] = pd.cut(

                df["Credit_History"],

                bins=[0, 580, 670, 740, 800, 900],

                labels=[
                    "Poor",
                    "Fair",
                    "Good",
                    "VeryGood",
                    "Exceptional"
                ]

            ).astype(str)

        # --------------------------------------------------------------
        # Save Engineered Dataset
        # --------------------------------------------------------------

        os.makedirs("data", exist_ok=True)

        output_file = "data/feature_engineered.csv"

        df.to_csv(
            output_file,
            index=False
        )

        print("\nFeature Engineering Completed Successfully")

        print(f"\nDataset saved to:\n{output_file}")

        print(f"\nFinal Shape: {df.shape}")

        return df


if __name__ == "__main__":

    engineer = FeatureEngineer()

    engineer.create_features()
    