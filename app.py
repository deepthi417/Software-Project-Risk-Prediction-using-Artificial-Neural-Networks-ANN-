"""
Streamlit App: Software Project Risk Predictor (ANN)
-------------------------------------------------------
Predicts whether a software requirement carries Low or High risk.

Zero scikit-learn / TensorFlow / pickle dependency at inference time --
preprocessing and the ANN forward pass are implemented in plain NumPy
(see preprocess_config.json and ann_weights.npz).

Run locally:      streamlit run app.py
Deploy:            push to GitHub, connect at https://share.streamlit.io
"""

import json
import numpy as np
import streamlit as st

def html_block(s: str) -> str:
    """Collapse a multi-line HTML template into a single line with no
    embedded newlines or leading whitespace. This sidesteps two separate
    Markdown/HTML-block quirks: (1) 4+ space indentation being read as a
    literal code block, and (2) multi-line raw-HTML blocks that mix
    element types (e.g. <svg> followed by <div>s) sometimes only
    rendering the first element and showing the rest as literal text."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RiskScope — Software Requirement Risk Predictor",
    page_icon="◆",
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
CAT_CATEGORIES = config["cat_categories"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ann_predict(X):
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return output.ravel()

def preprocess(input_row: dict) -> np.ndarray:
    num_values = np.array([
        input_row[col] if input_row[col] is not None else median
        for col, median in zip(NUM_COLS, NUM_MEDIANS)
    ], dtype=float)
    num_scaled = (num_values - NUM_MEANS) / NUM_SCALES

    cat_encoded = []
    for col in CAT_COLS:
        categories = CAT_CATEGORIES[col]
        value = input_row[col]
        one_hot = [1.0 if value == cat else 0.0 for cat in categories]
        cat_encoded.extend(one_hot)

    full_vector = np.concatenate([num_scaled, np.array(cat_encoded)])
    return full_vector.reshape(1, -1)

PROJECT_CATEGORY_OPTIONS = CAT_CATEGORIES["project Category"]
REQUIREMENT_CATEGORY_OPTIONS = CAT_CATEGORIES["Requirement Category"]
RISK_TARGET_CATEGORY_OPTIONS = CAT_CATEGORIES["Risk Target Category"]
MAGNITUDE_OPTIONS = CAT_CATEGORIES["Magnitude of Risk"]
IMPACT_OPTIONS = CAT_CATEGORIES["Impact"]
DIMENSION_OPTIONS = CAT_CATEGORIES["Dimension of Risk"]

# =======================================================================
# DESIGN SYSTEM — mission-control / instrument-panel aesthetic
# =======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep: #0B1220;
    --bg-panel: #131B2E;
    --bg-panel-alt: #0F1729;
    --border: #223049;
    --border-bright: #33465F;
    --amber: #FFB020;
    --amber-dim: #7A5B22;
    --teal: #2DD4BF;
    --teal-dim: #1C5F57;
    --violet: #A78BFA;
    --violet-dim: #4C3B82;
    --rose: #FB7185;
    --rose-dim: #7A2F3D;
    --text-primary: #E6EDF7;
    --text-muted: #7C8AA5;
    --text-faint: #4C5A75;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Ambient background texture — faint grid + soft color glows */
.stApp {
    background-color: var(--bg-deep);
    background-image:
        radial-gradient(ellipse 900px 500px at 85% -5%, rgba(255,176,32,0.10), transparent 60%),
        radial-gradient(ellipse 800px 500px at -5% 40%, rgba(167,139,250,0.08), transparent 60%),
        radial-gradient(ellipse 700px 450px at 100% 95%, rgba(45,212,191,0.07), transparent 60%),
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: auto, auto, auto, 34px 34px, 34px 34px;
}

/* Kill Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* ---------- Header strip ---------- */
.rs-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
    margin-bottom: 6px;
    position: relative;
}
.rs-header::after {
    content: "";
    position: absolute;
    bottom: -1px; left: 0;
    height: 1px;
    width: 140px;
    background: linear-gradient(90deg, var(--amber), transparent);
    animation: scan 3.5s ease-in-out infinite;
}
@keyframes scan {
    0%, 100% { transform: translateX(0); opacity: 0.9; }
    50% { transform: translateX(calc(100% + 200px)); opacity: 0.4; }
}
.rs-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
    text-shadow: 0 0 24px rgba(255,176,32,0.12);
}
.rs-title span {
    background: linear-gradient(100deg, var(--amber), var(--rose) 60%, var(--violet));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.rs-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-faint);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
.rs-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--teal);
    letter-spacing: 0.05em;
    text-align: right;
}
.rs-status .dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    margin-right: 6px;
    box-shadow: 0 0 6px var(--teal);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ---------- Eyebrow labels ---------- */
.rs-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 2px;
    border-left: 2px solid var(--amber);
    padding-left: 8px;
}
.rs-eyebrow.violet { color: var(--violet); border-left-color: var(--violet); }
.rs-eyebrow.teal { color: var(--teal); border-left-color: var(--teal); }
.rs-eyebrow.rose { color: var(--rose); border-left-color: var(--rose); }

/* ---------- Panels ---------- */
.rs-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s ease;
}
.rs-panel:hover { border-color: var(--border-bright); }

/* Tabs restyle */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    background: transparent;
    border-radius: 4px 4px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}
.stTabs [data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"] {
    color: var(--violet) !important;
    border-bottom-color: var(--violet) !important;
}
.stTabs [data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}
.stTabs [data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.78rem;
    background: linear-gradient(95deg, var(--amber), var(--rose));
    color: #1A1206;
    border: none;
    border-radius: 4px;
    font-weight: 600;
    transition: box-shadow 0.15s ease, transform 0.1s ease;
}
.stButton > button:active { transform: scale(0.98); }
.stButton > button:hover {
    box-shadow: 0 0 0 3px var(--amber-dim);
    color: #1A1206;
}

/* Result readout card */
.rs-result-card {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-bright);
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
}
.rs-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    letter-spacing: -0.01em;
    margin: 10px 0 2px 0;
}
.rs-verdict.high { color: var(--amber); }
.rs-verdict.low { color: var(--teal); }
.rs-led-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.rs-led {
    width: 9px; height: 9px;
    border-radius: 50%;
}
.rs-led.high { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 1.4s infinite; }
.rs-led.low { background: var(--teal); box-shadow: 0 0 8px var(--teal); animation: pulse 1.4s infinite; }

.rs-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 14px;
}
.rs-readout b { color: var(--text-primary); }

.rs-factors {
    margin-top: 16px;
    text-align: left;
}
.rs-factor-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    background: var(--bg-panel);
    border: 1px solid var(--border-bright);
    color: var(--text-muted);
    padding: 3px 9px;
    border-radius: 20px;
    margin: 3px 4px 0 0;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.rs-factor-pill:hover {
    border-color: var(--amber);
    color: var(--text-primary);
}
.rs-factor-pill.rose { border-color: var(--rose-dim); color: var(--rose); }
.rs-factor-pill.violet { border-color: var(--violet-dim); color: var(--violet); }
.rs-factor-pill.teal { border-color: var(--teal-dim); color: var(--teal); }
.rs-factor-pill.rose:hover { border-color: var(--rose); }
.rs-factor-pill.violet:hover { border-color: var(--violet); }
.rs-factor-pill.teal:hover { border-color: var(--teal); }

.rs-idle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-faint);
    text-align: center;
    padding: 40px 10px;
}

/* dataframe wrapper spacing */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

/* Custom session log table */
.rs-log-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}
.rs-log-table th {
    text-align: left;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-faint);
    background: var(--bg-panel-alt);
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
}
.rs-log-table td {
    padding: 9px 12px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
}
.rs-log-table tr:last-child td { border-bottom: none; }
.rs-log-table tr:hover td { background: rgba(255,255,255,0.02); color: var(--text-primary); }
.rs-badge {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 4px;
}
.rs-badge.high { background: rgba(255,176,32,0.15); color: var(--amber); }
.rs-badge.low { background: rgba(45,212,191,0.15); color: var(--teal); }

.rs-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.03em;
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-top: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.rs-credit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    white-space: nowrap;
}
.rs-credit b { color: var(--amber); font-weight: 600; }
.rs-credit a { color: var(--teal); text-decoration: none; }
.rs-credit a:hover { text-decoration: underline; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def render_gauge(prob_high_risk: float) -> str:
    """Semicircular instrument-style gauge with a sweeping needle."""
    angle_deg = 180 * prob_high_risk  # 0 = far left (safe), 180 = far right (danger)
    angle_rad = np.radians(180 - angle_deg)
    cx, cy, r = 150, 150, 120
    needle_x = cx + r * 0.82 * np.cos(angle_rad)
    needle_y = cy - r * 0.82 * np.sin(angle_rad)
    needle_color = "#FFB020" if prob_high_risk >= 0.5 else "#2DD4BF"

    return html_block(f"""
        <svg viewBox="0 0 300 175" width="100%" style="max-width:280px;">
            <path d="M 30 150 A 120 120 0 0 1 150 30" fill="none" stroke="#2DD4BF" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
            <path d="M 150 30 A 120 120 0 0 1 270 150" fill="none" stroke="#FFB020" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
            <line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
                  stroke="{needle_color}" stroke-width="3.5" stroke-linecap="round"/>
            <circle cx="{cx}" cy="{cy}" r="6" fill="{needle_color}" />
            <text x="30" y="170" font-family="IBM Plex Mono" font-size="10" fill="#7C8AA5">LOW</text>
            <text x="245" y="170" font-family="IBM Plex Mono" font-size="10" fill="#7C8AA5">HIGH</text>
            <text x="150" y="105" font-family="IBM Plex Mono" font-size="26" font-weight="600"
                  fill="#E6EDF7" text-anchor="middle">{prob_high_risk*100:.1f}%</text>
        </svg>
        """)


# =======================================================================
# HEADER
# =======================================================================
st.markdown(
    html_block(
        """
        <div class="rs-header">
            <div>
                <p class="rs-title">Risk<span>Scope</span></p>
                <p class="rs-subtitle">Software Requirement Risk Instrument · ANN-01</p>
            </div>
            <div class="rs-status"><span class="dot"></span>MODEL ONLINE</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1.3, 1], gap="large")

# =======================================================================
# LEFT: INPUT INSTRUMENTS
# =======================================================================
with left:
    st.markdown('<p class="rs-eyebrow violet">Requirement Telemetry</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["CATEGORY", "RISK PROFILE", "COST / SCHEDULE"])

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

    predict_clicked = st.button("RUN PREDICTION", type="primary", use_container_width=True)

# =======================================================================
# RIGHT: RESULT READOUT
# =======================================================================
with right:
    st.markdown('<p class="rs-eyebrow rose">Risk Verdict</p>', unsafe_allow_html=True)
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
    predicted_label = "HIGH RISK" if prob_high_risk >= 0.5 else "LOW RISK"
    css_class = "high" if prob_high_risk >= 0.5 else "low"
    confidence = prob_high_risk if prob_high_risk >= 0.5 else 1 - prob_high_risk

    num_inputs = {
        "probability": probability,
        "module count": affecting_modules,
        "fixing duration": fixing_duration,
        "fix cost": fix_cost,
        "priority": priority,
    }
    means_lookup = dict(zip(
        ["probability", "module count", "fixing duration", "fix cost", "priority"],
        NUM_MEANS,
    ))
    above_mean = [name for name, val in num_inputs.items() if val > means_lookup[name]]

    factor_pills = []
    if impact in ("Catastrophic", "High"):
        factor_pills.append((f"{impact.upper()} IMPACT", "rose"))
    if magnitude_of_risk in ("Very High", "High"):
        factor_pills.append((f"{magnitude_of_risk.upper()} MAGNITUDE", "violet"))
    for name in above_mean:
        factor_pills.append((f"ABOVE-AVG {name.upper()}", "teal"))
    if not factor_pills:
        factor_pills.append(("NEAR BASELINE", "muted"))

    pills_html = "".join(
        f'<span class="rs-factor-pill {color}">{text}</span>' for text, color in factor_pills
    )

    with result_placeholder:
        st.markdown(
            html_block(
                f"""
                <div class="rs-result-card">
                    {render_gauge(prob_high_risk)}
                    <div class="rs-verdict {css_class}">{predicted_label}</div>
                    <div class="rs-led-row"><span class="rs-led {css_class}"></span>CONFIDENCE {confidence*100:.1f}%</div>
                    <div class="rs-readout">P(high risk) = <b>{prob_high_risk:.4f}</b></div>
                    <div class="rs-factors">{pills_html}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

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
        st.markdown(
            '<div class="rs-result-card"><p class="rs-idle">'
            'AWAITING INPUT ―<br>configure the requirement telemetry, then run the prediction.'
            '</p></div>',
            unsafe_allow_html=True,
        )

# =======================================================================
# HISTORY LOG
# =======================================================================
if st.session_state.history:
    st.markdown('<p class="rs-eyebrow teal" style="margin-top: 18px;">Session Log</p>', unsafe_allow_html=True)

    rows_html = ""
    for row in reversed(st.session_state.history):
        badge_class = "high" if row["Prediction"] == "HIGH RISK" else "low"
        rows_html += (
            "<tr>"
            f"<td>{row['Project Category']}</td>"
            f"<td>{row['Requirement Category']}</td>"
            f"<td>{row['Impact']}</td>"
            f"<td>{row['Probability']}%</td>"
            f"<td><span class='rs-badge {badge_class}'>{row['Prediction']}</span></td>"
            f"<td>{row['Confidence']}</td>"
            "</tr>"
        )

    table_html = html_block(f"""
        <table class="rs-log-table">
            <thead><tr>
                <th>Project Category</th><th>Requirement Category</th><th>Impact</th>
                <th>Probability</th><th>Prediction</th><th>Confidence</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """)
    st.markdown(table_html, unsafe_allow_html=True)
    if st.button("CLEAR LOG"):
        st.session_state.history = []
        st.rerun()

st.markdown(
    html_block(
        """
        <div class="rs-footer">
            <span>MODEL: FEED-FORWARD ANN · 60 INPUT FEATURES · 8 SIGMOID HIDDEN UNITS · 1 SIGMOID OUTPUT ·
            SGD (LR=0.6) · TRAINED ON THE SOFTWARE REQUIREMENT RISK PREDICTION DATASET (SHAUKAT, NASEEM &amp; ZUBAIR, 2018)</span>
            <span class="rs-credit">Created by <b>Deepthi</b> · <a href="https://github.com/deepthi417" target="_blank">github.com/deepthi417</a></span>
        </div>
        """
    ),
    unsafe_allow_html=True,
)
