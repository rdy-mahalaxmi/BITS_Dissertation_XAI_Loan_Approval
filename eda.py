import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_PATH, OUTPUT_PATH


# ============================================================
# Paths
# ============================================================

PLOT_PATH = os.path.join(
    OUTPUT_PATH,
    "plots"
)

os.makedirs(
    PLOT_PATH,
    exist_ok=True
)


print("=" * 70)
print("Explainable AI for Loan Approval")
print("Exploratory Data Analysis")
print("=" * 70)


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(
    PROCESSED_DATA_PATH
)

print("Dataset Loaded Successfully")

print("\nDataset Shape:")
print(df.shape)


print("\nDataset Information:")
print(df.info())


# ============================================================
# Statistical Summary
# ============================================================

print("\nStatistical Summary:")

summary = df.describe()

print(summary)


summary.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "eda_summary.csv"
    )
)


# ============================================================
# Target Distribution
# ============================================================

print("\nGenerating Target Distribution Plot...")


plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Loan_Status"
)

plt.title(
    "Loan Approval Distribution"
)

plt.xlabel(
    "Loan Status"
)

plt.ylabel(
    "Count"
)


plt.savefig(
    os.path.join(
        PLOT_PATH,
        "loan_status_distribution.png"
    ),
    bbox_inches="tight"
)

plt.close()



# ============================================================
# Correlation Heatmap
# ============================================================

print("Generating Correlation Heatmap...")


plt.figure(
    figsize=(12,8)
)


sns.heatmap(
    df.select_dtypes(include=["number"]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)



plt.title(
    "Feature Correlation Heatmap"
)


plt.savefig(
    os.path.join(
        PLOT_PATH,
        "correlation_heatmap.png"
    ),
    bbox_inches="tight"
)


plt.close()



# ============================================================
# Income Distribution
# ============================================================

income_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount"
]


for column in income_columns:

    if column in df.columns:

        print(
            f"Generating {column} distribution..."
        )


        plt.figure(
            figsize=(7,4)
        )


        sns.histplot(
            df[column],
            kde=True
        )


        plt.title(
            f"{column} Distribution"
        )


        plt.savefig(
            os.path.join(
                PLOT_PATH,
                f"{column}_distribution.png"
            ),
            bbox_inches="tight"
        )


        plt.close()



# ============================================================
# Credit History Analysis
# ============================================================

if "Credit_History" in df.columns:


    print(
        "Generating Credit History Analysis..."
    )


    plt.figure(
        figsize=(6,4)
    )


    sns.countplot(
        data=df,
        x="Credit_History",
        hue="Loan_Status"
    )


    plt.title(
        "Credit History vs Loan Approval"
    )


    plt.savefig(
        os.path.join(
            PLOT_PATH,
            "credit_history_vs_loan_status.png"
        ),
        bbox_inches="tight"
    )


    plt.close()



# ============================================================
# Save EDA Dataset Snapshot
# ============================================================

df.head(20).to_csv(
    os.path.join(
        OUTPUT_PATH,
        "eda_sample.csv"
    ),
    index=False
)



print("\nEDA Completed Successfully")

print(
    "Plots saved at:",
    PLOT_PATH
)

print("=" * 70)
