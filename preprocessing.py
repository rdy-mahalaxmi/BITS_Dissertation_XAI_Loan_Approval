"""
===============================================================================
File: preprocessing.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Performs data preprocessing following industry best practices.

Research References
-------------------
[1] Hand & Henley (1997)
    Statistical Classification Methods in Consumer Credit Scoring.

[2] Lessmann et al. (2015)
    Benchmarking State-of-the-Art Classification Algorithms
    for Credit Scoring.

Methodology
-----------
1. Remove unnecessary identifier
2. Separate Features and Target
3. Train-Test Split
4. Missing Value Imputation
5. One-Hot Encoding
6. Feature Scaling
7. Save Processed Data

===============================================================================
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE, RANDOM_STATE
from data_loader import LoanDatasetLoader


class LoanPreprocessor:

    def __init__(self):

        loader = LoanDatasetLoader()
        self.df = loader.load_dataset()

    # ----------------------------------------------------------------------

    def preprocess(self):

        print("=" * 70)
        print("Starting Data Preprocessing...")
        print("=" * 70)

        # --------------------------------------------------------------
        # Remove Loan_ID
        # --------------------------------------------------------------

        if "Loan_ID" in self.df.columns:
            self.df.drop(columns=["Loan_ID"], inplace=True)

        # --------------------------------------------------------------
        # Save cleaned raw data for EDA
        # --------------------------------------------------------------

        self.df.to_csv(
            "data/processed_loan_data.csv",
            index=False
        )

        # --------------------------------------------------------------
        # Target Variable Check
        # --------------------------------------------------------------

        if "Loan_Status" not in self.df.columns:
            raise ValueError(
                "Loan_Status column not found.\n"
                "Please ensure the TRAIN dataset is loaded."
            )

        # --------------------------------------------------------------
        # Separate Features and Target
        # --------------------------------------------------------------

        X = self.df.drop("Loan_Status", axis=1)
        y = self.df["Loan_Status"]

        # --------------------------------------------------------------
        # Identify Numerical & Categorical Columns
        # --------------------------------------------------------------

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        print("\nNumerical Columns")
        print(numerical_columns)

        print("\nCategorical Columns")
        print(categorical_columns)

        # --------------------------------------------------------------
        # Numerical Pipeline
        # --------------------------------------------------------------

        numeric_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]

        )

        # --------------------------------------------------------------
        # Categorical Pipeline
        # --------------------------------------------------------------

        categorical_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )

            ]

        )

        # --------------------------------------------------------------
        # Combine Pipelines
        # --------------------------------------------------------------

        preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_pipeline,
                    numerical_columns
                ),

                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )

            ]

        )

        # --------------------------------------------------------------
        # Train-Test Split
        # --------------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )

        print("\nTrain Shape :", X_train.shape)
        print("Test Shape  :", X_test.shape)

        # --------------------------------------------------------------
        # Fit on Training Data
        # --------------------------------------------------------------

        X_train_processed = preprocessor.fit_transform(X_train)

        X_test_processed = preprocessor.transform(X_test)

        print("\nPreprocessing Completed Successfully")

        # --------------------------------------------------------------
        # Save Preprocessor
        # --------------------------------------------------------------

        import joblib

        joblib.dump(
            preprocessor,
            "models/preprocessor.pkl"
        )

        print("\nPreprocessor saved to models/preprocessor.pkl")

        # --------------------------------------------------------------
        # Save Processed Data
        # --------------------------------------------------------------

        train_df = pd.DataFrame(
            X_train_processed.toarray()
            if hasattr(X_train_processed, "toarray")
            else X_train_processed
        )

        train_df["Loan_Status"] = y_train.reset_index(drop=True)

        train_df.to_csv(
            "data/train_processed.csv",
            index=False
        )

        test_df = pd.DataFrame(
            X_test_processed.toarray()
            if hasattr(X_test_processed, "toarray")
            else X_test_processed
        )

        test_df["Loan_Status"] = y_test.reset_index(drop=True)

        test_df.to_csv(
            "data/test_processed.csv",
            index=False
        )

        print("\nProcessed datasets saved successfully.")

        return (

            X_train_processed,
            X_test_processed,

            y_train,
            y_test,

            preprocessor

        )


if __name__ == "__main__":

    processor = LoanPreprocessor()

    processor.preprocess()