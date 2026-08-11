"""
Streamlit UI — Explainable AI for Loan Approval
BITS Pilani Dissertation Project
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Loan Approval | BITS Pilani",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { color: #0c4a6e; font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #0369a1; font-size: 0.95rem; margin: 0.4rem 0 0; }

.metric-card {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-card .label { color: #0369a1; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: #0c4a6e; font-size: 1.8rem; font-weight: 700; margin-top: 0.2rem; }
.metric-card .sub   { color: #64748b; font-size: 0.75rem; margin-top: 0.1rem; }

.section-title {
    color: #0c4a6e;
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 4px solid #0284c7;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

.badge-best {
    background: #dcfce7;
    color: #166534;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background: #e0f2fe;
}
[data-testid="stSidebar"] .css-1d391kg { color: #0c4a6e; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #e0f2fe;
    border-radius: 8px 8px 0 0;
    color: #0369a1;
    padding: 0.5rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #0284c7 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
OUTPUTS     = os.path.join(ROOT, "outputs")
REPORTS     = os.path.join(ROOT, "reports")
PLOTS       = os.path.join(OUTPUTS, "plots")
SHAP_DIR    = os.path.join(OUTPUTS, "shap")
LIME_DIR    = os.path.join(OUTPUTS, "lime")

def p(path): return path if os.path.exists(path) else None

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def show_image(path, caption="", use_container_width=True):
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=use_container_width)
    else:
        st.info("Image not yet generated. Run the pipeline first.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 XAI Loan Approval")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 EDA", "🤖 Model Performance", "🔍 SHAP Explainability", "🟡 LIME Explainability"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    best_model_file = os.path.join(REPORTS, "best_model.txt")
    if os.path.exists(best_model_file):
        with open(best_model_file) as f:
            lines = f.readlines()
        bm  = lines[0].split(":")[1].strip() if lines else "—"
        auc = lines[1].split(":")[1].strip() if len(lines) > 1 else "—"
        st.markdown(f"**Best Model**")
        st.markdown(f"`{bm}`")
        st.markdown(f"ROC-AUC: **{auc}**")

    st.markdown("---")
    st.markdown("<small style='color:#0369a1'>BITS Pilani Dissertation by Mahalaxmi(2024AA05508)<br></small>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Explainable AI for Loan Approval using
Interpretable Machine Learning Models</h1>
  <p> Mahalaxmi (2024AA05508) </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "🏠 Overview":

    metrics_df = load_csv(os.path.join(OUTPUTS, "evaluation_metrics.csv"))
    profile_df = load_csv(os.path.join(REPORTS, "dataset_profile.csv"))

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="label">Dataset Rows</div><div class="value">255K</div><div class="sub">nikhil1e9/loan-default</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="label">Features</div><div class="value">17</div><div class="sub">+ 4 engineered</div></div>', unsafe_allow_html=True)
    with c3:
        best_acc = f"{metrics_df['Accuracy'].max()*100:.2f}%" if metrics_df is not None else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Best Accuracy</div><div class="value">{best_acc}</div><div class="sub">XGBoost</div></div>', unsafe_allow_html=True)
    with c4:
        best_auc = f"{metrics_df['ROC-AUC'].max():.4f}" if metrics_df is not None else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Best ROC-AUC</div><div class="value">{best_auc}</div><div class="sub">XGBoost</div></div>', unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">Pipeline Architecture</div>', unsafe_allow_html=True)
        stages = [
            ("1", "Data Loading & Profiling",   "Kaggle dataset download, validation, profiling"),
            ("2", "Feature Engineering",         "DebtToIncomeRatio, LoanPerTerm, IncomePerAge, CreditScoreBand"),
            ("3", "Preprocessing",               "Imputation, One-Hot Encoding, Standard Scaling"),
            ("4", "Exploratory Data Analysis",   "Distributions, correlations, target analysis"),
            ("5", "Model Training",              "LR · DT · RF · XGBoost"),
            ("6", "Model Evaluation",            "Accuracy, Precision, Recall, F1, ROC-AUC"),
            ("7", "SHAP Explainability",         "Global & local feature attribution"),
            ("8", "LIME Explainability",         "Instance-level model-agnostic explanations"),
        ]
        for num, name, desc in stages:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid #1e293b">
              <div style="background:#3b82f6;color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0">{num}</div>
              <div><div style="color:#0c4a6e;font-weight:600;font-size:0.9rem">{name}</div>
              <div style="color:#64748b;font-size:0.8rem">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        if metrics_df is not None:
            display = metrics_df[["Model", "Accuracy", "F1 Score", "ROC-AUC"]].copy()
            display["Accuracy"] = (display["Accuracy"] * 100).round(2).astype(str) + "%"
            st.dataframe(display, use_container_width=True, hide_index=True)

            fig = px.bar(
                metrics_df.sort_values("ROC-AUC"),
                x="ROC-AUC", y="Model", orientation="h",
                color="ROC-AUC", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=220,
                              coloraxis_showscale=False, title="ROC-AUC Comparison")
            st.plotly_chart(fig, use_container_width=True)

    # Dataset profile
    if profile_df is not None:
        st.markdown('<div class="section-title">Dataset Profile</div>', unsafe_allow_html=True)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 2 — EDA
# =============================================================================
elif page == "📊 EDA":

    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda_summary = load_csv(os.path.join(OUTPUTS, "eda_summary.csv"))
    eda_sample  = load_csv(os.path.join(OUTPUTS, "eda_sample.csv"))

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlation", "📋 Data Sample"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            show_image(p(os.path.join(PLOTS, "loan_status_distribution.png")), "Loan Approval Distribution")
            show_image(p(os.path.join(PLOTS, "ApplicantIncome_distribution.png")), "Applicant Income Distribution")
        with c2:
            show_image(p(os.path.join(PLOTS, "credit_history_vs_loan_status.png")), "Credit History vs Loan Status")
            show_image(p(os.path.join(PLOTS, "LoanAmount_distribution.png")), "Loan Amount Distribution")

    with tab2:
        show_image(p(os.path.join(PLOTS, "correlation_heatmap.png")), "Feature Correlation Heatmap")

    with tab3:
        if eda_sample is not None:
            st.dataframe(eda_sample, use_container_width=True)
        if eda_summary is not None:
            st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
            st.dataframe(eda_summary, use_container_width=True)

# =============================================================================
# PAGE 3 — MODEL PERFORMANCE
# =============================================================================
elif page == "🤖 Model Performance":

    st.markdown('<div class="section-title">Model Evaluation Results</div>', unsafe_allow_html=True)

    metrics_df = load_csv(os.path.join(OUTPUTS, "evaluation_metrics.csv"))

    if metrics_df is not None:
        # Radar chart
        models   = metrics_df["Model"].tolist()
        metrics  = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        fig = go.Figure()
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        for i, row in metrics_df.iterrows():
            vals = [row[m] for m in metrics]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=metrics + [metrics[0]],
                fill="toself", name=row["Model"],
                line_color=colors[i % len(colors)], opacity=0.7
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_dark", height=420,
            title="Model Performance Radar",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        styled = metrics_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            styled[col] = (styled[col] * 100).round(2).astype(str) + "%"
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ROC curve
    st.markdown('<div class="section-title">ROC Curve Comparison</div>', unsafe_allow_html=True)
    show_image(p(os.path.join(OUTPUTS, "roc_curve_comparison.png")))

    # Confusion matrices
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    model_keys = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    cols = st.columns(4)
    for col, key in zip(cols, model_keys):
        with col:
            img = p(os.path.join(OUTPUTS, f"confusion_matrix_{key}.png"))
            show_image(img, key.replace("_", " ").title())

# =============================================================================
# PAGE 4 — SHAP
# =============================================================================
elif page == "🔍 SHAP Explainability":

    st.markdown('<div class="section-title">SHAP — SHapley Additive exPlanations</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#0369a1'>Global model interpretability using Shapley values from cooperative game theory.</small>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌐 Global Summary", "📊 Feature Importance", "🔎 Local Explanation"])

    with tab1:
        show_image(p(os.path.join(SHAP_DIR, "shap_summary_plot.png")), "SHAP Summary Plot")

    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            show_image(p(os.path.join(SHAP_DIR, "shap_feature_importance_plot.png")), "SHAP Feature Importance Bar")
        with c2:
            shap_df = load_csv(os.path.join(SHAP_DIR, "shap_feature_importance.csv"))
            if shap_df is not None:
                shap_df.columns = ["Feature Index", "Mean |SHAP|"]
                shap_df["Mean |SHAP|"] = shap_df["Mean |SHAP|"].round(4)
                st.dataframe(shap_df.head(20), use_container_width=True, hide_index=True)

    with tab3:
        local_df = load_csv(os.path.join(SHAP_DIR, "local_prediction_explanation.csv"))
        if local_df is not None:
            local_df["SHAP_Value"] = local_df["SHAP_Value"].round(4)
            fig = px.bar(
                local_df.head(20), x="SHAP_Value", y="Feature",
                orientation="h",
                color="SHAP_Value",
                color_continuous_scale=["#ef4444", "#f8fafc", "#3b82f6"],
                color_continuous_midpoint=0,
                template="plotly_dark",
                title="Local SHAP Explanation — Sample 0",
            )
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(local_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 5 — LIME
# =============================================================================
elif page == "🟡 LIME Explainability":

    st.markdown('<div class="section-title">LIME — Local Interpretable Model-Agnostic Explanations</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#0369a1'>Instance-level explanations by fitting a local surrogate model around each prediction.</small>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Aggregate Importance", "🔎 Per-Sample Explanations"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            show_image(p(os.path.join(LIME_DIR, "lime_feature_importance_plot.png")), "LIME Aggregate Feature Importance")
        with c2:
            lime_df = load_csv(os.path.join(LIME_DIR, "lime_feature_importance.csv"))
            if lime_df is not None:
                lime_df["Mean_Abs_Weight"] = lime_df["Mean_Abs_Weight"].round(4)
                fig = px.bar(
                    lime_df.head(20).sort_values("Mean_Abs_Weight"),
                    x="Mean_Abs_Weight", y="Feature_Condition",
                    orientation="h", template="plotly_dark",
                    color="Mean_Abs_Weight", color_continuous_scale="Blues",
                    title="Top 20 LIME Feature Conditions",
                )
                fig.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sample_idx = st.selectbox("Select Sample", list(range(5)), format_func=lambda x: f"Sample {x}")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            show_image(p(os.path.join(LIME_DIR, f"lime_explanation_sample_{sample_idx}.png")),
                       f"LIME Explanation — Sample {sample_idx}")
        with c2:
            sample_df = load_csv(os.path.join(LIME_DIR, f"lime_explanation_sample_{sample_idx}.csv"))
            if sample_df is not None:
                pred  = sample_df["Predicted_Class"].iloc[0]
                prob  = sample_df["Approval_Prob"].iloc[0]
                color = "#065f46" if pred == "Approved" else "#7f1d1d"
                label = "✅ APPROVED" if pred == "Approved" else "❌ REJECTED"
                st.markdown(f"""
                <div style="background:{color};border-radius:10px;padding:1rem;text-align:center;margin-bottom:1rem">
                  <div style="color:#f1f5f9;font-size:1.3rem;font-weight:700">{label}</div>
                  <div style="color:#94a3b8;font-size:0.85rem">Approval Probability: {prob}</div>
                </div>""", unsafe_allow_html=True)
                st.dataframe(
                    sample_df[["Feature_Condition", "LIME_Weight"]].round(4),
                    use_container_width=True, hide_index=True
                )
