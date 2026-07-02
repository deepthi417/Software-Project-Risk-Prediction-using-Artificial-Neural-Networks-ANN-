import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Software Project Risk Prediction",
    page_icon="⚠️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = load_model("best_model.keras")
preprocessor = joblib.load("preprocessor.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# -----------------------------
# Title
# -----------------------------
st.title("⚠️ Software Project Risk Prediction")
st.write("Predict the Risk Level of a Software Requirement using ANN.")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

requirement = st.text_area(
    "Requirement",
    height=150,
    placeholder="Enter software requirement..."
)

project_category = st.selectbox(
    "Project Category",
    [
        "Transaction Processing System",
        "Management Information System",
        "Enterprise System",
        "Safety Critical System"
    ]
)

requirement_category = st.selectbox(
    "Requirement Category",
    [
        "Functional",
        "Non Functional"
    ]
)

risk_target = st.selectbox(
    "Risk Target Category",
    [
        "Quality",
        "Cost",
        "Schedule",
        "Performance",
        "Security",
        "Maintainability"
    ]
)

probability = st.slider(
    "Probability",
    0.0,
    100.0,
    50.0
)

magnitude = st.selectbox(
    "Magnitude of Risk",
    [
        "Low",
        "Medium",
        "High"
    ]
)

impact = st.selectbox(
    "Impact",
    [
        "Low",
        "Medium",
        "High"
    ]
)

dimension = st.selectbox(
    "Dimension of Risk",
    [
        "User",
        "Developer",
        "Organization",
        "Technology"
    ]
)

modules = st.number_input(
    "Affected No. of Modules",
    min_value=1.0,
    value=1.0
)

duration = st.number_input(
    "Fixing Duration (Days)",
    min_value=1.0,
    value=1.0
)

fix_cost = st.number_input(
    "Fix Cost (% of Project)",
    min_value=0.0,
    value=5.0
)

priority = st.slider(
    "Priority",
    1.0,
    10.0,
    5.0
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Risk"):

    input_df = pd.DataFrame({

        "Requirements":[requirement],

        "project Category":[project_category],

        "Requirement Category":[requirement_category],

        "Risk Target Category":[risk_target],

        "Probability":[probability],

        "Magnitude of Risk":[magnitude],

        "Impact":[impact],

        "Dimension of Risk":[dimension],

        "Afftecting No of Modules":[modules],

        "Fixing Duration (Days)":[duration],

        "Fix Cost (\\% of Project)":[fix_cost],

        "Priority":[priority]

    })

    X = preprocessor.transform(input_df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    prediction = model.predict(X)

    predicted_class = np.argmax(prediction)

    risk = label_encoder.inverse_transform([predicted_class])[0]

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Risk Level : {risk}")

    st.info(f"Confidence : {confidence:.2f}%")

    st.bar_chart(prediction.T)
