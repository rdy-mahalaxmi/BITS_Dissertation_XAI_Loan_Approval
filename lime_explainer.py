"""
===============================================================================
File: lime_explainer.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Generates Explainable AI interpretations using LIME (Local Interpretable
Model-agnostic Explanations) for the selected loan approval prediction model.

Methodology
-----------
1. Load best performing model and saved preprocessor
2. Load raw (pre-processed) test dataset with original feature names
3. Initialize LIME TabularExplainer on raw feature space
4. Wrap preprocessor + model into a single predict_fn for LIME
5. Generate local prediction explanations for multiple samples
6. Create feature importance plots per sample
7. Save explanation outputs

===============================================================================
"""


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


class LIMEExplainer:


    def __init__(self):

        # Raw (un-transformed) data with original column names
        self.raw_data_path = "data/processed_loan_data.csv"

        self.model_dir   = "models"
        self.output_dir  = "outputs/lime"
        self.report_file = "reports/best_model.txt"

        self.num_samples    = 6   # 3 Approved + 3 Rejected
        self.num_features   = 15   # top features shown per plot

        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 70)
        print("LIME Explainability Module")
        print("=" * 70)


    # ---------------------------------------------------------------

    def load_best_model(self):

        with open(self.report_file, "r") as f:
            line = f.readline()

        model_name = line.split(":")[1].strip()
        model_path = os.path.join(self.model_dir, model_name + ".pkl")

        print("\nLoading Model:", model_name)

        model = joblib.load(model_path)

        return model, model_name


    # ---------------------------------------------------------------

    def load_data(self):
        """
        Load the raw dataset (original feature names, un-transformed values).
        The preprocessor is loaded separately and used inside predict_fn.
        """

        df = pd.read_csv(self.raw_data_path)

        # Drop Loan_ID if present (was removed during preprocessing)
        if "Loan_ID" in df.columns:
            df.drop(columns=["Loan_ID"], inplace=True)

        X = df.drop("Loan_Status", axis=1)
        y = df["Loan_Status"]

        # Encode target: Y (approved) -> 1, N (rejected) -> 0
        # Confirmed from data_loader.py: Default=0 -> Y (approved), Default=1 -> N (rejected)
        if y.dtype == object:
            y = y.map({"N": 0, "Y": 1})

        return X, y


    # ---------------------------------------------------------------

    def generate_explanation(self):

        model, model_name = self.load_best_model()

        preprocessor = joblib.load("models/preprocessor.pkl")

        X_raw, y = self.load_data()

        feature_names = list(X_raw.columns)

        # Identify categorical feature indices for LIME
        categorical_columns = X_raw.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        categorical_indices = [
            feature_names.index(c) for c in categorical_columns
        ]

        # ---------------------------------------------------------------
        # LIME requires a fully numeric training array.
        # Encode categorical columns as integer codes so LIME can scale them.
        # The predict_fn decodes them back to original strings before
        # passing through the preprocessor, so the model is unaffected.
        # ---------------------------------------------------------------

        # Build per-column category mappings  {col: {code: original_value}}
        cat_decoders = {}      # {col: {int_code: original_string}}
        cat_names    = {}      # {feature_index: [label_for_code_0, label_for_code_1, ...]}
        X_lime = X_raw.copy()

        for col in categorical_columns:
            codes, uniques = pd.factorize(X_lime[col])
            decoder = dict(enumerate(uniques))
            cat_decoders[col] = decoder
            X_lime[col] = codes.astype(float)
            col_idx = feature_names.index(col)
            cat_names[col_idx] = list(uniques)

        def predict_fn(raw_array):
            df_temp = pd.DataFrame(raw_array, columns=feature_names)
            # Decode integer codes back to original category strings
            for col, decoder in cat_decoders.items():
                df_temp[col] = (
                    df_temp[col]
                    .round()
                    .astype(int)
                    .clip(0, len(decoder) - 1)
                    .map(decoder)
                )
            X_transformed = preprocessor.transform(df_temp)
            return model.predict_proba(X_transformed)

        # ---------------------------------------------------------------
        # Class mapping (confirmed from data_loader.py remap_columns):
        #   Loan_Status Y = approved  -> encoded as 1
        #   Loan_Status N = rejected  -> encoded as 0
        # ---------------------------------------------------------------
        class_names = ["Rejected", "Approved"]   # index 0 = Rejected, 1 = Approved

        # ---------------------------------------------------------------
        # Debug / validation output
        # ---------------------------------------------------------------
        print("\n--- Debug Info ---")
        print("Original feature names :", feature_names)
        print("Categorical features   :", categorical_columns)
        print("Categorical indices    :", categorical_indices)
        print("Raw X shape            :", X_raw.shape)

        # Verify predict_fn on first sample
        sample_check = X_raw.iloc[[0]]
        X_check = preprocessor.transform(sample_check)
        prob_check = model.predict_proba(X_check)
        print("Sample 0 predict_proba :", prob_check)
        print("Sample 0 predicted class:", class_names[prob_check.argmax()])
        print("------------------\n")

        # ---------------------------------------------------------------
        # Initialise LIME on the raw feature space
        # ---------------------------------------------------------------

        print("Initializing LIME Explainer...")

        explainer = LimeTabularExplainer(

            training_data=X_lime.values,

            feature_names=feature_names,

            class_names=class_names,

            categorical_features=categorical_indices,

            categorical_names=cat_names,
            mode="classification",

            discretize_continuous=True,

            random_state=42

        )

        # ---------------------------------------------------------------
        # Select a balanced mix: 3 Approved + 3 Rejected predictions
        # ---------------------------------------------------------------
        X_transformed_all = preprocessor.transform(X_raw)
        all_preds = model.predict(X_transformed_all)

        approved_indices = list((all_preds == 1).nonzero()[0][:3])
        rejected_indices = list((all_preds == 0).nonzero()[0][:3])
        selected_indices = approved_indices + rejected_indices

        print(f"\nSelected indices — Approved: {approved_indices}, Rejected: {rejected_indices}")
        print(
            "\nGenerating LIME explanations for",
            len(selected_indices),
            "samples (3 Approved + 3 Rejected)..."
        )

        all_explanations = []

        for loop_i, i in enumerate(selected_indices):

            sample = X_lime.iloc[i].values

            explanation = explainer.explain_instance(

                data_row=sample,

                predict_fn=predict_fn,

                num_features=self.num_features,

                top_labels=2

            )

            # Predicted class and probability
            proba        = explanation.predict_proba          # shape (n_classes,)
            pred_class   = int(proba.argmax())
            pred_label   = class_names[pred_class]
            pred_prob    = proba[pred_class] * 100

            # Approval probability (always class index 1)
            approval_prob = proba[1] * 100

            # ---------------------------------------------------------------
            # Debug per sample
            # ---------------------------------------------------------------
            print(f"\nSample {loop_i} (data index {i}):")
            print(f"  predict_proba  : {proba}")
            print(f"  Predicted class: {pred_label} ({pred_prob:.1f}%)")
            print(f"  LIME features  : {[f[0] for f in explanation.as_list(label=pred_class)]}")

            # ---------------------------------------------------------------
            # Local Explanation CSV
            # ---------------------------------------------------------------

            exp_list = explanation.as_list(label=pred_class)

            exp_df = pd.DataFrame(
                exp_list,
                columns=["Feature_Condition", "LIME_Weight"]
            )

            exp_df["Sample_Index"]    = loop_i
            exp_df["Data_Index"]      = i
            exp_df["Predicted_Class"] = pred_label
            exp_df["Approval_Prob"]   = f"{approval_prob:.1f}%"

            exp_df.to_csv(
                os.path.join(
                    self.output_dir,
                    f"lime_explanation_sample_{loop_i}.csv"
                ),
                index=False
            )

            all_explanations.append(exp_df)

            # ---------------------------------------------------------------
            # Local Explanation Bar Plot
            # ---------------------------------------------------------------

            features = [item[0] for item in exp_list]
            weights  = [item[1] for item in exp_list]

            colors = [
                "steelblue" if w >= 0 else "tomato"
                for w in weights
            ]

            fig, ax = plt.subplots(figsize=(11, 7))

            ax.barh(features, weights, color=colors)

            ax.axvline(x=0, color="black", linewidth=0.8)

            ax.set_xlabel("LIME Weight  (positive = supports prediction, negative = opposes prediction)")

            # Title with prediction result and probability
            if pred_label == "Approved":
                outcome_line = f"Prediction: LOAN APPROVED  |  Approval Probability: {approval_prob:.1f}%"
            else:
                outcome_line = f"Prediction: LOAN REJECTED  |  Approval Probability: {approval_prob:.1f}%"

            ax.set_title(
                f"LIME Local Explanation - Sample {loop_i} (Data Index {i})\n{outcome_line}",
                fontsize=12,
                pad=12
            )

            # Legend for bar colours
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="steelblue", label="Supports prediction"),
                Patch(facecolor="tomato",    label="Opposes prediction"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"lime_explanation_sample_{loop_i}.png"
                ),
                dpi=150
            )

            plt.close()


        # -----------------------------------------------------------
        # Aggregate Feature Importance CSV
        # -----------------------------------------------------------

        combined = pd.concat(all_explanations, ignore_index=True)

        importance = (
            combined
            .groupby("Feature_Condition")["LIME_Weight"]
            .apply(lambda x: abs(x).mean())
            .reset_index()
        )

        importance.columns = ["Feature_Condition", "Mean_Abs_Weight"]

        importance = importance.sort_values(
            by="Mean_Abs_Weight",
            ascending=False
        )

        importance.to_csv(
            os.path.join(self.output_dir, "lime_feature_importance.csv"),
            index=False
        )

        # -----------------------------------------------------------
        # Aggregate Feature Importance Bar Plot
        # -----------------------------------------------------------

        plt.figure(figsize=(11, 7))

        plt.barh(
            importance["Feature_Condition"],
            importance["Mean_Abs_Weight"],
            color="steelblue"
        )

        plt.xlabel("Mean Absolute LIME Weight")

        plt.title("LIME Aggregate Feature Importance\n(averaged over all explained samples)")

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.output_dir, "lime_feature_importance_plot.png"),
            dpi=150
        )

        plt.close()

        print("\nLIME Explanation Completed Successfully")
        print("Results saved at:", self.output_dir)



# =============================================================================

if __name__ == "__main__":

    explainer = LIMEExplainer()

    explainer.generate_explanation()
