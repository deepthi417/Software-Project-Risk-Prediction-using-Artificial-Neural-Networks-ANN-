"""
Streamlit App: Software Project Risk Prediction (ANN)
-------------------------------------------------------
Predicts whether a software requirement carries Low or High risk.

This app has ZERO scikit-learn / TensorFlow / pickle dependency at
inference time -- both preprocessing (scaling + one-hot encoding) and
the ANN forward pass are implemented in plain NumPy, using parameters
extracted from the trained pipeline (see preprocess_config.json and
ann_weights.npz). This keeps deployment lightweight and avoids version
-compatibility failures (pickle / TensorFlow install issues) on
constrained platforms like Streamlit Community Cloud's free tier.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push this repo (app.py, requirements.txt, ann_weights.npz,
    preprocess_config.json) to GitHub, then connect the repo at
    https://share.streamlit.io
"""

import json
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Software Project Risk Predictor",
    page_icon="⚠️",
    layout="centered",
)

# ---------------------------------------------------------------------
# Load ANN weights + preprocessing config (cached so it loads once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    weights = np.load("ann_weights.npz")
    W1, b1, W2, b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]
    with open("preprocess_config.json") as f:
        config = json.load(f)
    return W1, b1, W2, b2, config

W1, b1, W2, b2, config = load_artifacts()

NUM_COLS = config["num_cols"]
CAT_COLS = config["cat_cols"]
NUM_MEDIANS = np.array(config["num_medians"])
NUM_MEANS = np.array(config["num_means"])
NUM_SCALES = np.array(config["num_scales"])
CAT_CATEGORIES = config["cat_categories"]  # dict: col -> list of categories

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ann_predict(X):
    """Forward pass: 8 sigmoid hidden neurons -> 1 sigmoid output."""
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return output.ravel()

def preprocess(input_row: dict) -> np.ndarray:
    """
    Reproduces the training-time ColumnTransformer manually:
      1. Numeric columns: median-impute (n/a here, all fields filled by UI),
         then standard-scale using the training mean/scale.
      2. Categorical columns: one-hot encode using the training category
         order (unseen categories are simply all-zero, matching
         OneHotEncoder(handle_unknown="ignore")).
    Column order matches training: numeric block first, then categorical
    block, in the same column order used when the ColumnTransformer was fit.
    """
    # --- numeric block ---
    num_values = np.array([
        input_row[col] if input_row[col] is not None else median
        for col, median in zip(NUM_COLS, NUM_MEDIANS)
    ], dtype=float)
    num_scaled = (num_values - NUM_MEANS) / NUM_SCALES

    # --- categorical block (one-hot) ---
    cat_encoded = []
    for col in CAT_COLS:
        categories = CAT_CATEGORIES[col]
        value = input_row[col]
        one_hot = [1.0 if value == cat else 0.0 for cat in categories]
        cat_encoded.extend(one_hot)

    full_vector = np.concatenate([num_scaled, np.array(cat_encoded)])
    return full_vector.reshape(1, -1)

# ---------------------------------------------------------------------
# Dropdown option lists (pulled directly from the training config,
# so they always match what the model was actually trained on)
# ---------------------------------------------------------------------
PROJECT_CATEGORY_OPTIONS = CAT_CATEGORIES["project Category"]
REQUIREMENT_CATEGORY_OPTIONS = CAT_CATEGORIES["Requirement Category"]
RISK_TARGET_CATEGORY_OPTIONS = CAT_CATEGORIES["Risk Target Category"]
MAGNITUDE_OPTIONS = CAT_CATEGORIES["Magnitude of Risk"]
IMPACT_OPTIONS = CAT_CATEGORIES["Impact"]
DIMENSION_OPTIONS = CAT_CATEGORIES["Dimension of Risk"]

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("⚠️ Software Project Risk Predictor")
st.write(
    "Predicts whether a software requirement carries **Low** or **High** "
    "risk, using a feed-forward back-propagation ANN "
    "(8 sigmoid hidden neurons, 1 sigmoid output, trained with SGD)."
)

st.divider()
st.subheader("Enter Requirement Details")

col1, col2 = st.columns(2)

with col1:
    project_category = st.selectbox("Project Category", PROJECT_CATEGORY_OPTIONS)
    requirement_category = st.selectbox("Requirement Category", REQUIREMENT_CATEGORY_OPTIONS)
    risk_target_category = st.selectbox("Risk Target Category", RISK_TARGET_CATEGORY_OPTIONS)
    magnitude_of_risk = st.selectbox("Magnitude of Risk", MAGNITUDE_OPTIONS)

with col2:
    impact = st.selectbox("Impact", IMPACT_OPTIONS)
    dimension_of_risk = st.selectbox("Dimension of Risk", DIMENSION_OPTIONS)
    probability = st.slider("Probability (%)", 1, 100, 40)
    affecting_modules = st.slider("Affecting No. of Modules", 1, 15, 3)

col3, col4 = st.columns(2)
with col3:
    fixing_duration = st.slider("Fixing Duration (Days)", 1, 30, 5)
with col4:
    fix_cost = st.slider("Fix Cost (% of Project)", 0, 30, 2)

priority = st.slider("Priority", 1, 100, 42)

st.divider()

if st.button("Predict Risk", type="primary", use_container_width=True):
    input_row = {
        "project Category": project_category,
        "Requirement Category": requirement_category,
        "Risk Target Category": risk_target_category,
        "Probability": probability,
        "Magnitude of Risk": magnitude_of_risk,
        "Impact": impact,
        "Dimension of Risk": dimension_of_risk,
        "Afftecting No of Modules": affecting_modules,
        "Fixing Duration (Days)": fixing_duration,
        "Fix Cost (\\% of Project)": fix_cost,
        "Priority": priority,
    }

    X = preprocess(input_row)
    prob_high_risk = float(ann_predict(X)[0])
    predicted_label = "High Risk" if prob_high_risk >= 0.5 else "Low Risk"
    confidence = prob_high_risk if predicted_label == "High Risk" else 1 - prob_high_risk

    st.subheader("Prediction Result")
    if predicted_label == "High Risk":
        st.error(f"🔴 **{predicted_label}** — confidence: {confidence:.1%}")
    else:
        st.success(f"🟢 **{predicted_label}** — confidence: {confidence:.1%}")

    st.progress(prob_high_risk)
    st.caption(f"Model output (probability of High Risk): {prob_high_risk:.4f}")

st.divider()
st.caption(
    "Model: Feed-forward ANN · 60 input features · 8 sigmoid hidden neurons · "
    "1 sigmoid output · SGD (lr=0.6) · Trained on the Software Requirement "
    "Risk Prediction Dataset (Shaukat, Naseem & Zubair, 2018)."
)
