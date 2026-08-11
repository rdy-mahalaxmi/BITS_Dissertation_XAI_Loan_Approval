"""
===============================================================================
File: model_training.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Trains multiple supervised machine learning models for loan approval prediction.
The trained models are saved for subsequent evaluation and explainability using
SHAP and LIME.

Algorithms
----------
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extreme Gradient Boosting (XGBoost)

Research References
-------------------

[1] Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).
    Applied Logistic Regression (3rd ed.).
    Wiley.

[2] Breiman, L. (2001).
    Random Forests.
    Machine Learning, 45(1), 5–32.

[3] Quinlan, J. R. (1986).
    Induction of Decision Trees.
    Machine Learning, 1(1), 81–106.

[4] Chen, T., & Guestrin, C. (2016).
    XGBoost: A Scalable Tree Boosting System.
    Proceedings of the 22nd ACM SIGKDD International Conference
    on Knowledge Discovery and Data Mining.

[5] Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015).
    Benchmarking State-of-the-Art Classification Algorithms
    for Credit Scoring.
    European Journal of Operational Research, 247(1), 124–136.

Methodology
-----------
1. Load processed training dataset
2. Separate features and target
3. Train multiple ML algorithms
4. Save trained models
5. Forward models for evaluation
===============================================================================
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


class LoanModelTrainer:

    def __init__(self):

        self.train_path = "data/train_processed.csv"

        self.model_dir = "models"

        os.makedirs(self.model_dir, exist_ok=True)

        print("=" * 70)
        print("Loading Processed Training Dataset...")
        print("=" * 70)

        self.df = pd.read_csv(self.train_path)

    # ------------------------------------------------------------------

    def prepare_data(self):

        X = self.df.drop("Loan_Status", axis=1)

        y = self.df["Loan_Status"]

        if y.isin(["Y", "N"]).any():
            y = y.map({"N": 0, "Y": 1})

        y = y.astype(int)

        return X, y



    # ------------------------------------------------------------------

    def train_models(self):

        X, y = self.prepare_data()

        models = {

            "logistic_regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

            "decision_tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "random_forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

            "xgboost":
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="logloss"
                )
        }

        print("\nTraining Models...\n")

        for name, model in models.items():

            print(f"Training {name}...")

            model.fit(X, y)

            model_path = os.path.join(
                self.model_dir,
                f"{name}.pkl"
            )

            joblib.dump(model, model_path)

            print(f"Saved -> {model_path}")

        print("\nAll models trained successfully.")


# =============================================================================

if __name__ == "__main__":

    trainer = LoanModelTrainer()

    trainer.train_models()
