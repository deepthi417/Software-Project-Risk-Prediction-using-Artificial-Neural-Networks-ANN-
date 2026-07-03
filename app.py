"""
Streamlit App: Software Project Risk Prediction (ANN)
-------------------------------------------------------
Loads the trained ANN (model.keras) and the fitted preprocessing
pipeline (preprocessor.pkl) to predict whether a software requirement
carries Low or High risk.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push this repo (app.py, requirements.txt, model.keras, preprocessor.pkl)
    to GitHub, then connect the repo at https://share.streamlit.io
"""

import pickle
import numpy as np
import pandas as pd
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
# Load preprocessor + trained ANN weights (cached so it only loads once)
#
# NOTE: inference is done with plain NumPy instead of TensorFlow/Keras.
# The network is a simple 2-layer sigmoid feed-forward net, so its
# forward pass is just two matrix multiplications + sigmoid -- no need
# to ship the full TensorFlow runtime just to run predictions. This
# avoids TensorFlow install issues on constrained deployment platforms
# (e.g. Streamlit Community Cloud's free tier).
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    weights = np.load("ann_weights.npz")
    W1, b1, W2, b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]
    return preprocessor, W1, b1, W2, b2

preprocessor, W1, b1, W2, b2 = load_artifacts()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ann_predict(X):
    """Forward pass through the trained ANN: 8 sigmoid hidden neurons -> 1 sigmoid output."""
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return output.ravel()

# ---------------------------------------------------------------------
# Dropdown option lists (from the training dataset's categories)
# ---------------------------------------------------------------------
PROJECT_CATEGORY_OPTIONS = [
    "Enterprise System", "Management Information System",
    "Safety Critical System", "Transaction Processing System",
]

REQUIREMENT_CATEGORY_OPTIONS = [
    "Constraints", "Functional", "Interfaces", "Performance",
    "Reliability & Availability", "Safety", "Security",
    "Standards", "Supportability", "Usability",
]

RISK_TARGET_CATEGORY_OPTIONS = [
    "Budget", "Business", "Cost", "Design", "Functionalvalidity",
    "Organizational Environment", "Overdrawn Budget", "People",
    "Performance", "Personal", "Planning & Control", "Process",
    "Project Complexity", "Quality", "Requirement",
    "Resource Availability", "Schedule", "Software", "Team",
    "Time Dimension", "Unrealistic Requirements", "User",
]

MAGNITUDE_OPTIONS = [
    "Negligible", "Very Low", "Low", "Medium", "High", "Very High", "Extreme",
]

IMPACT_OPTIONS = ["Insignificant", "Low", "Moderate", "High", "Catastrophic"]

DIMENSION_OPTIONS = [
    "Cost", "Estimations", "Organizational Environment",
    "Organizational Requirements", "Planning And Control",
    "Project Complexity", "Requirements", "Schedule",
    "Software Requirement", "Team", "User",
]

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
    # Build a single-row dataframe matching the training feature columns
    input_df = pd.DataFrame([{
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
    }])

    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    prob_high_risk = float(ann_predict(X_transformed)[0])
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
    "Model: Feed-forward ANN · 64 input features · 8 sigmoid hidden neurons · "
    "1 sigmoid output · SGD (lr=0.6) · Trained on the Software Requirement "
    "Risk Prediction Dataset (Shaukat, Naseem & Zubair, 2018)."
)
