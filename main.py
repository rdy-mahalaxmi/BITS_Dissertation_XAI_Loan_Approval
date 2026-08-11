"""
===============================================================================
File: main.py

Project:
Explainable AI for Loan Approval using Interpretable Machine Learning Models

Description
-----------
Master pipeline runner. Executes all stages in sequence:

    Stage 1 : Data Loading & Profiling
    Stage 2 : Feature Engineering
    Stage 3 : Preprocessing
    Stage 4 : Exploratory Data Analysis
    Stage 5 : Model Training
    Stage 6 : Model Evaluation

===============================================================================
"""

import os
import sys
import stat
import shutil
import subprocess
import traceback
from pathlib import Path


# =============================================================================
# Cleanup
# =============================================================================

def _force_remove(action, path, exc_info):
    """Error handler for shutil.rmtree to force-delete read-only files."""
    os.chmod(path, stat.S_IWRITE)
    action(path)


def cleanup_outputs():

    root = Path(__file__).resolve().parent

    targets = [
        root / "outputs",
        root / "models",
        root / "reports",
        root / "data" / "train_processed.csv",
        root / "data" / "test_processed.csv",
        root / "data" / "processed_loan_data.csv",
        root / "data" / "feature_engineered.csv",
    ]

    print()
    print("=" * 70)
    print("  Cleaning Up Previous Outputs")
    print("=" * 70)

    for target in targets:

        if target.is_dir():
            shutil.rmtree(target, onerror=_force_remove)
            print(f"  Deleted folder : {target.relative_to(root)}")

        elif target.is_file():
            target.unlink()
            print(f"  Deleted file   : {target.relative_to(root)}")

    print("  Cleanup complete.")


def run_stage(stage_number, stage_name, func):

    print()
    print("=" * 70)
    print(f"  STAGE {stage_number} : {stage_name}")
    print("=" * 70)

    try:
        func()
        print(f"\n  [DONE] Stage {stage_number} completed successfully.")

    except Exception as e:
        print(f"\n  [FAILED] Stage {stage_number} - {stage_name}")
        print(f"  Error : {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Stage 1 : Data Loading
# =============================================================================

def stage_data_loading():

    from data_loader import LoanDatasetLoader

    loader = LoanDatasetLoader()
    loader.run()


# =============================================================================
# Stage 2 : Feature Engineering
# =============================================================================

def stage_feature_engineering():

    from feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()
    engineer.create_features()


# =============================================================================
# Stage 3 : Preprocessing
# =============================================================================

def stage_preprocessing():

    from preprocessing import LoanPreprocessor

    preprocessor = LoanPreprocessor()
    preprocessor.preprocess()


# =============================================================================
# Stage 4 : Exploratory Data Analysis
# =============================================================================

def stage_eda():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from config import PROCESSED_DATA_PATH, OUTPUT_PATH

    plot_path = os.path.join(OUTPUT_PATH, "plots")
    os.makedirs(plot_path, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("\nDataset Shape :", df.shape)

    # Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Loan_Status")
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.savefig(os.path.join(plot_path, "loan_status_distribution.png"), bbox_inches="tight")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.select_dtypes(include=["number"]).corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(plot_path, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    # Income Distributions
    for column in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        if column in df.columns:
            plt.figure(figsize=(7, 4))
            sns.histplot(df[column], kde=True)
            plt.title(f"{column} Distribution")
            plt.savefig(os.path.join(plot_path, f"{column}_distribution.png"), bbox_inches="tight")
            plt.close()

    # Credit History vs Loan Status
    if "Credit_History" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
        plt.title("Credit History vs Loan Approval")
        plt.savefig(os.path.join(plot_path, "credit_history_vs_loan_status.png"), bbox_inches="tight")
        plt.close()

    # Save summary
    df.describe().to_csv(os.path.join(OUTPUT_PATH, "eda_summary.csv"))
    df.head(20).to_csv(os.path.join(OUTPUT_PATH, "eda_sample.csv"), index=False)

    print(f"\nPlots saved at: {plot_path}")


# =============================================================================
# Stage 5 : Model Training
# =============================================================================

def stage_model_training():

    from model_training import LoanModelTrainer

    trainer = LoanModelTrainer()
    trainer.train_models()


# =============================================================================
# Stage 6 : Model Evaluation
# =============================================================================

def stage_model_evaluation():

    from model_evaluation import LoanModelEvaluator

    evaluator = LoanModelEvaluator()
    results = evaluator.evaluate_models()

    print("\nModel Performance Summary\n")
    print(results)


# =============================================================================
# Stage 7 : SHAP Explainability
# =============================================================================

def stage_shap_explainer():

    from shap_explainer import SHAPExplainer

    explainer = SHAPExplainer()
    explainer.generate_explanation()


# =============================================================================
# Stage 8 : LIME Explainability
# =============================================================================

def stage_lime_explainer():

    from lime_explainer import LIMEExplainer

    explainer = LIMEExplainer()
    explainer.generate_explanation()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("  Explainable AI for Loan Approval")
    print("  Full Pipeline Execution")
    print("=" * 70)

    cleanup_outputs()

    run_stage(1, "Data Loading & Profiling",  stage_data_loading)
    run_stage(2, "Feature Engineering",       stage_feature_engineering)
    run_stage(3, "Preprocessing",             stage_preprocessing)
    run_stage(4, "Exploratory Data Analysis", stage_eda)
    run_stage(5, "Model Training",            stage_model_training)
    run_stage(6, "Model Evaluation",          stage_model_evaluation)
    run_stage(7, "SHAP Explainability",        stage_shap_explainer)
    run_stage(8, "LIME Explainability",        stage_lime_explainer)

    print()
    print("=" * 70)
    print("  Pipeline Completed Successfully")
    print("=" * 70)
    print()

    print("Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
