"""
===============================================================================
File: model_evaluation.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Evaluates trained machine learning models using multiple classification metrics
and identifies the best-performing model for Explainable AI analysis.

Methodology
-----------
1. Load processed test dataset
2. Load trained models
3. Generate predictions
4. Compute evaluation metrics
5. Create confusion matrices
6. Create ROC comparison plot
7. Save results
8. Select best-performing model

===============================================================================
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


class LoanModelEvaluator:

    def __init__(self):

        self.test_path = "data/test_processed.csv"

        self.model_dir = "models"

        self.output_dir = "outputs"

        self.report_dir = "reports"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        print("=" * 70)
        print("Loan Approval Model Evaluation")
        print("=" * 70)

        self.df = pd.read_csv(self.test_path)

    # ------------------------------------------------------------------

    def prepare_data(self):

        X = self.df.drop("Loan_Status", axis=1)

        y = self.df["Loan_Status"]

        if y.isin(["Y", "N"]).any():
            y = y.map({"N": 0, "Y": 1})

        y = y.astype(int)

        return X, y


    # ------------------------------------------------------------------

    def load_models(self):

        models = {}

        model_files = [
            "logistic_regression.pkl",
            "decision_tree.pkl",
            "random_forest.pkl",
            "xgboost.pkl"
        ]

        for file in model_files:

            model_path = os.path.join(
                self.model_dir,
                file
            )

            if os.path.exists(model_path):

                model_name = file.replace(".pkl", "")

                models[model_name] = joblib.load(
                    model_path
                )

        return models

    # ------------------------------------------------------------------

    def evaluate_models(self):

        X_test, y_test = self.prepare_data()

        models = self.load_models()

        results = []

        roc_data = {}

        best_model_name = None
        best_auc = 0

        print("\nEvaluating Models...\n")

        for model_name, model in models.items():

            print(f"Evaluating {model_name}")

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):

                y_prob = model.predict_proba(X_test)[:, 1]

            else:

                y_prob = y_pred

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

            results.append({

                "Model": model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4)

            })

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )

            roc_data[model_name] = (
                fpr,
                tpr,
                roc_auc
            )

            self.save_confusion_matrix(
                y_test,
                y_pred,
                model_name
            )

            if roc_auc > best_auc:

                best_auc = roc_auc

                best_model_name = model_name

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="ROC-AUC",
            ascending=False
        )

        results_df.to_csv(
            os.path.join(
                self.output_dir,
                "evaluation_metrics.csv"
            ),
            index=False
        )

        results_df.to_csv(
            os.path.join(
                self.report_dir,
                "model_comparison.csv"
            ),
            index=False
        )

        self.save_roc_curve(
            roc_data
        )

        self.save_best_model(
            best_model_name,
            best_auc
        )

        print("\nEvaluation Completed Successfully")

        print("\nBest Model:")
        print(best_model_name)

        print("\nResults Saved Successfully")

        return results_df

    # ------------------------------------------------------------------

    def save_confusion_matrix(
        self,
        y_true,
        y_pred,
        model_name
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        disp.plot(ax=ax)

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                f"confusion_matrix_{model_name}.png"
            )
        )

        plt.close()

    # ------------------------------------------------------------------

    def save_roc_curve(
        self,
        roc_data
    ):

        plt.figure(
            figsize=(8, 6)
        )

        for model_name, values in roc_data.items():

            fpr, tpr, auc_score = values

            plt.plot(
                fpr,
                tpr,
                label=f"{model_name} (AUC={auc_score:.3f})"
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve Comparison"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "roc_curve_comparison.png"
            )
        )

        plt.close()

    # ------------------------------------------------------------------

    def save_best_model(
        self,
        model_name,
        auc_score
    ):

        file_path = os.path.join(
            self.report_dir,
            "best_model.txt"
        )

        with open(
            file_path,
            "w"
        ) as f:

            f.write(
                f"Best Model: {model_name}\n"
            )

            f.write(
                f"ROC-AUC: {auc_score:.4f}\n"
            )

        print(
            f"\nBest model saved -> {file_path}"
        )


# =============================================================================

if __name__ == "__main__":

    evaluator = LoanModelEvaluator()

    results = evaluator.evaluate_models()

    print("\nModel Performance Summary\n")

    print(results)
