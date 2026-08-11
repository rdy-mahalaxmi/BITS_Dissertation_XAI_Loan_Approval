# Explainable AI for Loan Approval
### BITS Pilani Dissertation — Mahalaxmi (2024AA05508)

A full end-to-end machine learning pipeline that predicts loan approval outcomes and explains every prediction using **SHAP** and **LIME** — making the model's decisions transparent and auditable.

---

## Project Overview

Traditional loan approval models are black boxes. This project builds an interpretable ML system that not only predicts whether a loan will be approved or defaulted, but also explains **why** — both globally (which features matter most overall) and locally (why a specific applicant was approved or rejected).

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — nikhil1e9/loan-default](https://www.kaggle.com/datasets/nikhil1e9/loan-default) |
| Rows | ~255,000 |
| Target | `Loan_Status` — `Y` (Approved) / `N` (Rejected) |

**Original Features:** Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner

**Engineered Features (added by pipeline):**
- `DebtToIncomeRatio` — LoanAmount / (ApplicantIncome + 1)
- `LoanPerTerm` — LoanAmount / (LoanTerm + 1), monthly repayment burden
- `IncomePerAge` — ApplicantIncome / (Age + 1)
- `CreditScoreBand` — Categorical band: Poor / Fair / Good / VeryGood / Exceptional

---

## Project Structure

```
Loan_Approval_XAI_v2/
│
├── main.py                  # Master pipeline runner (runs all 8 stages + launches Streamlit)
├── app.py                   # Streamlit dashboard UI
├── config.py                # All paths, constants, Kaggle download logic
│
├── data_loader.py           # Stage 1 — Load & profile dataset
├── feature_engineering.py  # Stage 2 — Create engineered features
├── preprocessing.py         # Stage 3 — Imputation, encoding, scaling
├── eda.py                   # Stage 4 — Exploratory Data Analysis
├── model_training.py        # Stage 5 — Train ML models
├── model_evaluation.py      # Stage 6 — Evaluate & compare models
├── shap_explainer.py        # Stage 7 — SHAP global & local explanations
├── lime_explainer.py        # Stage 8 — LIME per-instance explanations
│
├── data/                    # Raw and processed datasets (git-ignored)
├── models/                  # Saved .pkl model files (git-ignored)
├── outputs/                 # Plots, metrics, SHAP/LIME outputs
│   ├── plots/               # EDA plots
│   ├── shap/                # SHAP summary, importance, local explanation
│   └── lime/                # LIME per-sample plots and CSVs
├── reports/                 # Model comparison, best model, dataset profile
├── logs/                    # Pipeline execution logs
│
└── requirements.txt
```

---

## End-to-End Pipeline

Running `python main.py` executes all 8 stages sequentially, then automatically launches the Streamlit dashboard.

```
python main.py
```

### Stage 1 — Data Loading & Profiling (`data_loader.py`)
- Downloads the dataset from Kaggle using `kagglehub`
- Renames columns to match pipeline conventions (e.g. `CreditScore` → `Credit_History`, `Default` → `Loan_Status`)
- Remaps target: `Default=0` → `Y` (Approved), `Default=1` → `N` (Rejected)
- Generates a dataset profile report saved to `reports/dataset_profile.csv`

### Stage 2 — Feature Engineering (`feature_engineering.py`)
- Creates 4 new business-oriented features: `DebtToIncomeRatio`, `LoanPerTerm`, `IncomePerAge`, `CreditScoreBand`
- Saves the enriched dataset to `data/feature_engineered.csv`

### Stage 3 — Preprocessing (`preprocessing.py`)
- Separates features (`X`) and target (`y`)
- Applies a `ColumnTransformer` with two sub-pipelines:
  - **Numerical:** Median imputation → Standard Scaling
  - **Categorical:** Mode imputation → One-Hot Encoding
- Performs an 80/20 stratified train-test split
- Saves the fitted preprocessor to `models/preprocessor.pkl`
- Saves processed splits to `data/train_processed.csv` and `data/test_processed.csv`

### Stage 4 — Exploratory Data Analysis (`main.py → stage_eda`)
- Generates and saves plots to `outputs/plots/`:
  - Loan approval distribution
  - Feature correlation heatmap
  - Income and loan amount distributions
  - Credit history vs loan status
- Saves statistical summary to `outputs/eda_summary.csv`

### Stage 5 — Model Training (`model_training.py`)
Trains 4 classifiers on the processed training data:

| Model | Key Config |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=200` |
| XGBoost | `n_estimators=200, learning_rate=0.05` |

All models are saved as `.pkl` files under `models/`.

### Stage 6 — Model Evaluation (`model_evaluation.py`)
- Evaluates all 4 models on the test set
- Metrics computed: Accuracy, Precision, Recall, F1, ROC-AUC
- Generates confusion matrices and a combined ROC curve comparison plot
- Saves results to `outputs/evaluation_metrics.csv` and `reports/model_comparison.csv`
- Identifies and saves the best model (by ROC-AUC) to `reports/best_model.txt`

### Stage 7 — SHAP Explainability (`shap_explainer.py`)
- Loads the best model identified in Stage 6
- Uses `TreeExplainer` for tree-based models, `LinearExplainer` for Logistic Regression
- Generates:
  - **Global summary plot** — feature impact distribution across all predictions
  - **Feature importance bar plot** — mean absolute SHAP values
  - **Local explanation CSV** — SHAP values for a single prediction
- Outputs saved to `outputs/shap/`

### Stage 8 — LIME Explainability (`lime_explainer.py`)
- Loads the best model and the saved preprocessor
- Operates on raw (un-transformed) data to preserve original feature names
- Wraps `preprocessor.transform + model.predict_proba` into a single `predict_fn` for LIME
- Selects 6 balanced samples: 3 Approved + 3 Rejected predictions
- Generates per-sample bar plots and CSVs showing which feature conditions drove each decision
- Generates an aggregate feature importance plot across all explained samples
- Outputs saved to `outputs/lime/`

### Auto-launch — Streamlit Dashboard (`app.py`)
After all 8 stages complete, `main.py` automatically runs:
```python
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
```

---

## Streamlit Dashboard

The dashboard has 5 pages accessible from the sidebar:

| Page | Content |
|---|---|
| 🏠 Overview | Pipeline architecture, KPI cards, model comparison table |
| 📊 EDA | Distribution plots, correlation heatmap, data sample |
| 🤖 Model Performance | Radar chart, metrics table, ROC curves, confusion matrices |
| 🔍 SHAP Explainability | Global summary, feature importance, local explanation waterfall |
| 🟡 LIME Explainability | Aggregate importance, per-sample prediction explanations |

To launch the dashboard independently (after the pipeline has run):
```
streamlit run app.py
```

---

## Setup & Installation

**Prerequisites:** Python 3.9+, a Kaggle account with API credentials configured

```bash
# 1. Clone the repository
git clone https://github.com/rdy-mahalaxmi/BITS_Dissertation_XAI_Loan_Approval.git
cd BITS_Dissertation_XAI_Loan_Approval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Kaggle API
# Place your kaggle.json at ~/.kaggle/kaggle.json
# Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY

# 4. Run the full pipeline
python main.py
```

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `scikit-learn` | Preprocessing, model training, evaluation |
| `xgboost` | XGBoost classifier |
| `shap` | SHAP global and local explanations |
| `lime` | LIME instance-level explanations |
| `streamlit` | Interactive dashboard |
| `plotly` | Interactive charts in dashboard |
| `kagglehub` | Automated dataset download from Kaggle |
| `pandas / numpy` | Data manipulation |
| `matplotlib / seaborn` | EDA and evaluation plots |

---

## Research References

- Lessmann et al. (2015) — Benchmarking Classification Algorithms for Credit Scoring
- Breiman (2001) — Random Forests
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Hosmer, Lemeshow & Sturdivant (2013) — Applied Logistic Regression
- Hand & Henley (1997) — Statistical Classification Methods in Consumer Credit Scoring

---

*BITS Pilani M.Tech Dissertation — Work Integrated Learning Programme*
