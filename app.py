import json
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Software Project Risk Predictor",
    page_icon="⚠️",
    layout="wide",
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

if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.subheader("📋 Requirement Details")

    tab1, tab2, tab3 = st.tabs(["🏷️ Category & Type", "📈 Risk Profile", "💰 Cost & Schedule"])

    with tab1:
        project_category = st.selectbox("Project Category", PROJECT_CATEGORY_OPTIONS)
        requirement_category = st.selectbox("Requirement Category", REQUIREMENT_CATEGORY_OPTIONS)
        risk_target_category = st.selectbox("Risk Target Category", RISK_TARGET_CATEGORY_OPTIONS)
        dimension_of_risk = st.selectbox("Dimension of Risk", DIMENSION_OPTIONS)

    with tab2:
        magnitude_of_risk = st.selectbox("Magnitude of Risk", MAGNITUDE_OPTIONS)
        impact = st.selectbox("Impact", IMPACT_OPTIONS)
        probability = st.slider("Probability (%)", 1, 100, 40)
        affecting_modules = st.slider("Affecting No. of Modules", 1, 15, 3)

    with tab3:
        fixing_duration = st.slider("Fixing Duration (Days)", 1, 30, 5)
        fix_cost = st.slider("Fix Cost (% of Project)", 0, 30, 2)
        priority = st.slider("Priority", 1, 100, 42)

    predict_clicked = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

with right:
    st.subheader("🎯 Prediction Result")
    result_placeholder = st.container()

if predict_clicked:
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

    # --- simple driver explanation: compare numeric inputs to training means ---
    num_inputs = {
        "Probability": probability,
        "Affecting Modules": affecting_modules,
        "Fixing Duration": fixing_duration,
        "Fix Cost": fix_cost,
        "Priority": priority,
    }
    means_lookup = dict(zip(
        ["Probability", "Affecting Modules", "Fixing Duration", "Fix Cost", "Priority"],
        NUM_MEANS,
    ))
    above_mean = [
        name for name, val in num_inputs.items()
        if val > means_lookup[name]
    ]

    with result_placeholder:
        if predicted_label == "High Risk":
            st.error(f"🔴 **{predicted_label}**")
        else:
            st.success(f"🟢 **{predicted_label}**")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(prob_high_risk)
        st.caption(f"Raw model output (P of High Risk): {prob_high_risk:.4f}")

        st.markdown("**Contributing factors**")
        driver_bits = []
        if impact in ("Catastrophic", "High"):
            driver_bits.append(f"'{impact}' impact")
        if magnitude_of_risk in ("Very High", "High"):
            driver_bits.append(f"'{magnitude_of_risk}' magnitude")
        if above_mean:
            driver_bits.append(f"above-average {', '.join(above_mean).lower()}")
        if driver_bits:
            st.caption("Likely pushing the prediction: " + "; ".join(driver_bits) + ".")
        else:
            st.caption("Inputs are broadly near or below dataset averages.")

    st.session_state.history.append({
        "Project Category": project_category,
        "Requirement Category": requirement_category,
        "Impact": impact,
        "Probability": probability,
        "Prediction": predicted_label,
        "Confidence": f"{confidence:.1%}",
    })
else:
    with result_placeholder:
        st.info("Fill in the requirement details and click **Predict Risk**.")

if st.session_state.history:
    st.divider()
    st.subheader("🕘 Session History")
    st.dataframe(
        list(reversed(st.session_state.history)),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.caption(
    "Model: Feed-forward ANN · 60 input features · 8 sigmoid hidden neurons · "
    "1 sigmoid output · SGD (lr=0.6) · Trained on the Software Requirement "
    "Risk Prediction Dataset (Shaukat, Naseem & Zubair, 2018)."
)
