"""
===============================================================================
File: shap_explainer.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Generates Explainable AI interpretations using SHAP (SHapley Additive
exPlanations) for the selected loan approval prediction model.

Methodology
-----------
1. Load best performing model
2. Load test dataset
3. Generate SHAP values
4. Create global feature importance plots
5. Generate local prediction explanations
6. Save explanation outputs

===============================================================================
"""


import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

import shap


class SHAPExplainer:


    def __init__(self):

        self.data_path = (
            "data/test_processed.csv"
        )

        self.model_dir = (
            "models"
        )

        self.output_dir = (
            "outputs/shap"
        )

        self.report_file = (
            "reports/best_model.txt"
        )


        os.makedirs(
            self.output_dir,
            exist_ok=True
        )


        print("=" * 70)
        print("SHAP Explainability Module")
        print("=" * 70)



    # ---------------------------------------------------------------

    def load_best_model(self):


        with open(
            self.report_file,
            "r"
        ) as file:

            line = file.readline()


        model_name = (
            line.split(":")[1]
            .strip()
        )


        model_path = os.path.join(
            self.model_dir,
            model_name + ".pkl"
        )


        print(
            "\nLoading Model:",
            model_name
        )


        model = joblib.load(
            model_path
        )


        return model, model_name



    # ---------------------------------------------------------------

    def load_data(self):


        df = pd.read_csv(
            self.data_path
        )


        X = df.drop(
            "Loan_Status",
            axis=1
        )


        y = df["Loan_Status"]


        if y.dtype == object:

            y = y.map(
                {
                    "N":0,
                    "Y":1
                }
            )


        return X, y



    # ---------------------------------------------------------------

    def generate_explanation(self):


        model, model_name = (
            self.load_best_model()
        )


        X, y = (
            self.load_data()
        )


        print(
            "\nGenerating SHAP values..."
        )


        #
        # Tree based models
        #

        if model_name in [
            "random_forest",
            "decision_tree",
            "xgboost"
        ]:


            explainer = shap.TreeExplainer(
                model
            )


            shap_values = (
                explainer.shap_values(X)
            )


            if isinstance(
                shap_values,
                list
            ):

                shap_values = shap_values[1]



        #
        # Logistic Regression
        #

        else:


            explainer = shap.LinearExplainer(
                model,
                X
            )


            shap_values = (
                explainer.shap_values(X)
            )



        feature_names = (
            X.columns
        )


        # -----------------------------------------------------------
        # Feature Importance CSV
        # -----------------------------------------------------------


        importance = pd.DataFrame(

            {

                "Feature":
                    feature_names,

                "Importance":
                    abs(shap_values).mean(axis=0)

            }

        )


        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )


        importance.to_csv(

            os.path.join(
                self.output_dir,
                "shap_feature_importance.csv"
            ),

            index=False

        )


        # -----------------------------------------------------------
        # Summary Plot
        # -----------------------------------------------------------


        plt.figure(
            figsize=(10,8)
        )


        shap.summary_plot(

            shap_values,

            X,

            show=False

        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(
                self.output_dir,
                "shap_summary_plot.png"
            )

        )


        plt.close()



        # -----------------------------------------------------------
        # Bar Plot
        # -----------------------------------------------------------


        plt.figure(
            figsize=(10,8)
        )


        shap.summary_plot(

            shap_values,

            X,

            plot_type="bar",

            show=False

        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(
                self.output_dir,
                "shap_feature_importance_plot.png"
            )

        )


        plt.close()



        # -----------------------------------------------------------
        # Individual Prediction Explanation
        # -----------------------------------------------------------


        sample = X.iloc[
            [0]
        ]


        sample_shap = shap_values[
            0
        ]


        explanation = pd.DataFrame(

            {

                "Feature":
                    feature_names,

                "SHAP_Value":
                    sample_shap,

                "Feature_Value":
                    sample.iloc[0].values

            }

        )


        explanation.sort_values(

            by="SHAP_Value",

            ascending=False

        ).to_csv(

            os.path.join(

                self.output_dir,

                "local_prediction_explanation.csv"

            ),

            index=False

        )


        print(
            "\nSHAP Explanation Completed Successfully"
        )


        print(
            "Results saved at:",
            self.output_dir
        )



# =============================================================================

if __name__ == "__main__":


    explainer = SHAPExplainer()

    explainer.generate_explanation()
