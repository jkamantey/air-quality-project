"""
Diabetes Risk Predictor — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem 1.5rem;
        border-left: 4px solid #E74C3C; margin-bottom: 0.5rem;
    }
    .risk-high   { background:#FDEDEC; border-left:4px solid #E74C3C; border-radius:12px; padding:1.2rem 1.5rem; }
    .risk-medium { background:#FEF9E7; border-left:4px solid #F39C12; border-radius:12px; padding:1.2rem 1.5rem; }
    .risk-low    { background:#EAFAF1; border-left:4px solid #2ECC71; border-radius:12px; padding:1.2rem 1.5rem; }
    .stSlider > div > div > div > div { background: #E74C3C !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "model/diabetes_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

artifacts = load_model()


# ── Feature Engineering (must match notebook) ─────────────────────────────────
def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["BMI_Age_Interaction"]     = df["BMI"] * df["Age"]
    df["Glucose_BMI_Product"]     = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_Ratio"]   = df["Insulin"] / (df["Glucose"] + 1e-6)
    df["Glucose_Insulin_Product"] = df["Glucose"] * df["Insulin"]
    df["Glucose_Category"] = float(pd.cut([df["Glucose"].iloc[0]], bins=[0, 99, 125, 400], labels=[0, 1, 2])[0])
    df["BMI_Category"]     = float(pd.cut([df["BMI"].iloc[0]],     bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3])[0])
    df["Age_Group"]        = float(pd.cut([df["Age"].iloc[0]],     bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])[0])
    return df


def predict(inputs: dict):
    raw = pd.DataFrame([inputs])
    zero_cols = artifacts["zero_cols"]

    # Impute with stored medians
    for col in zero_cols:
        if raw[col].iloc[0] == 0:
            raw[col] = np.nan
    raw[zero_cols] = artifacts["imputer"].transform(raw[zero_cols])

    featured = engineer_features(raw)
    feature_names = artifacts["feature_names"]
    X = featured[feature_names]
    X_sc = pd.DataFrame(artifacts["scaler"].transform(X), columns=feature_names)

    prob = artifacts["model"].predict_proba(X_sc)[0][1]
    threshold = artifacts["threshold"]
    pred = int(prob >= threshold)

    # SHAP
    try:
        explainer = shap.TreeExplainer(artifacts["model"])
        sv = explainer.shap_values(X_sc)
        shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
    except Exception:
        shap_vals = None

    return prob, pred, X_sc, shap_vals, feature_names


# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Data Input")
    st.markdown("---")

    pregnancies = st.slider("Pregnancies",              0, 17,   3,  help="Number of times pregnant")
    glucose     = st.slider("Glucose (mg/dL)",         40, 250, 120, help="Plasma glucose concentration (2h OGTT)")
    bp          = st.slider("Blood Pressure (mm Hg)",  20, 130,  72, help="Diastolic blood pressure")
    skin        = st.slider("Skin Thickness (mm)",      0,  99,  23, help="Triceps skin fold thickness")
    insulin     = st.slider("Insulin (µU/mL)",          0, 900,  79, help="2-Hour serum insulin")
    bmi         = st.slider("BMI",                     10.0, 70.0, 32.0, step=0.1)
    dpf         = st.slider("Diabetes Pedigree",        0.07, 2.50, 0.47, step=0.01, help="Diabetes pedigree function (genetic risk)")
    age         = st.slider("Age (years)",             21,  90,   33)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")


# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🩺 Diabetes Risk Predictor</p>', unsafe_allow_html=True)
st.markdown("""
**Clinical decision-support tool** powered by machine learning.  
Adjust the patient parameters in the sidebar and click **Predict Risk**.

> ⚠️ *For educational/portfolio purposes only — not a medical device.*
""")

col_left, col_right = st.columns([1.1, 1])


# ── Left Column: Patient Summary ───────────────────────────────────────────────
with col_left:
    st.markdown("### Patient Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Glucose",       f"{glucose} mg/dL",
              delta="High" if glucose > 125 else ("Pre-diabetic" if glucose > 99 else "Normal"))
    c2.metric("BMI",           f"{bmi:.1f}",
              delta="Obese" if bmi >= 30 else ("Overweight" if bmi >= 25 else "Normal"))
    c3.metric("Age",           f"{age} yrs")

    c4, c5, c6 = st.columns(3)
    c4.metric("Blood Pressure", f"{bp} mmHg")
    c5.metric("Insulin",        f"{insulin} µU/mL")
    c6.metric("Pedigree",       f"{dpf:.2f}")

    # BMI Classification
    bmi_label = (
        "Underweight (<18.5)" if bmi < 18.5 else
        "Normal (18.5–24.9)"  if bmi < 25.0 else
        "Overweight (25–29.9)" if bmi < 30.0 else
        "Obese (≥30)"
    )
    glucose_label = (
        "Diabetic range (≥126)"   if glucose >= 126 else
        "Pre-diabetic (100–125)"  if glucose >= 100 else
        "Normal (<100)"
    )
    st.info(f"**BMI Category:** {bmi_label}  |  **Glucose Category:** {glucose_label}")


# ── Right Column: Prediction Result ───────────────────────────────────────────
with col_right:
    st.markdown("### Prediction Result")

    if artifacts is None:
        st.warning("Model not found. Run the notebook first to train and save the model.")
    elif predict_btn or True:
        inputs = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }

        with st.spinner("Computing prediction..."):
            prob, pred, X_sc, shap_vals, feat_names = predict(inputs)
            model_name = artifacts["model_name"]
            metrics    = artifacts["metrics"]

        # Risk gauge
        risk_pct = int(prob * 100)
        if prob >= 0.65:
            risk_class, risk_label, risk_emoji = "risk-high",   "HIGH RISK",   "🔴"
        elif prob >= 0.40:
            risk_class, risk_label, risk_emoji = "risk-medium", "MODERATE RISK", "🟡"
        else:
            risk_class, risk_label, risk_emoji = "risk-low",    "LOW RISK",    "🟢"

        st.markdown(f"""
        <div class="{risk_class}">
            <h2 style="margin:0">{risk_emoji} {risk_label}</h2>
            <h1 style="margin:0.2rem 0; font-size:3rem; font-weight:900">{risk_pct}%</h1>
            <p style="margin:0; color:#555">Probability of Diabetes</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        **Model:** {model_name}  
        **Threshold:** {artifacts['threshold']:.2f}  
        **Training AUC:** {metrics['test_roc_auc']}  |  **F1:** {metrics['test_f1']}
        """)

        # Probability gauge bar
        fig_g, ax_g = plt.subplots(figsize=(5, 0.6))
        ax_g.barh(0, 1.0, height=0.5, color="#E0E0E0", edgecolor="none")
        ax_g.barh(0, prob, height=0.5, color="#E74C3C" if prob >= 0.65 else ("#F39C12" if prob >= 0.4 else "#2ECC71"), edgecolor="none")
        ax_g.axvline(artifacts["threshold"], color="black", linewidth=2, linestyle="--")
        ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.5)
        ax_g.axis("off")
        ax_g.text(artifacts["threshold"], 0.35, f"Threshold\n{artifacts['threshold']:.2f}",
                  ha="center", fontsize=7, color="black")
        fig_g.patch.set_facecolor("none")
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)


# ── SHAP Explanation Row ───────────────────────────────────────────────────────
if artifacts is not None and "shap_vals" in dir():
    st.markdown("---")
    st.markdown("### Why this prediction? — SHAP Feature Contributions")
    st.markdown("*Positive (red) values push toward Diabetes; negative (green) values push toward No Diabetes.*")

    if shap_vals is not None:
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:12]
        sv_sorted  = shap_vals[sorted_idx]
        fn_sorted  = [feat_names[i] for i in sorted_idx]

        fig_shap, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in sv_sorted]
        bars = ax.barh(range(len(sv_sorted)), sv_sorted[::-1], color=bar_colors[::-1], edgecolor="white", height=0.6)
        ax.set_yticks(range(len(sv_sorted)))
        ax.set_yticklabels(fn_sorted[::-1])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("SHAP Value (impact on model output)")
        ax.set_title("Feature Contribution to This Prediction", fontweight="bold")
        red_patch   = mpatches.Patch(color="#E74C3C", label="Increases diabetes risk")
        green_patch = mpatches.Patch(color="#2ECC71", label="Decreases diabetes risk")
        ax.legend(handles=[red_patch, green_patch], loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)


# ── Model Performance Info ─────────────────────────────────────────────────────
if artifacts is not None:
    st.markdown("---")
    st.markdown("### Model Performance (Hold-Out Test Set)")
    m = artifacts["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC",  m["test_roc_auc"], help="Area under ROC curve — 0.85+ is excellent")
    c2.metric("F1 Score", m["test_f1"],       help="Harmonic mean of precision & recall")
    c3.metric("Accuracy", m["test_accuracy"], help="Overall classification accuracy")

    with st.expander("About this project"):
        st.markdown("""
        **Dataset:** Pima Indians Diabetes Dataset (NIDDK, 768 patients, 8 clinical features)  
        **Pipeline:** Zero-replacement → Median imputation → Feature engineering → RobustScaler → RandomizedSearchCV  
        **Models tested:** Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest,  
        Extra Trees, Gradient Boosting, AdaBoost, XGBoost, SVM  
        **Explainability:** SHAP TreeExplainer for global and per-patient feature attribution  
        **Author:** Joshua Kamantey | Built for portfolio / internship showcase
        """)
