"""
Software Project Risk Prediction using ANN (no SVM)
-----------------------------------------------------
Dataset : Software Requirement Risk Prediction Dataset
          (Shaukat, Naseem & Zubair, 2018) - Zenodo DOI 10.5281/zenodo.1209601
          299 real-world requirement risk cases, 12 attributes + Risk Level target

Architecture (per project spec):
  Input layer  : 64 nodes  (after cleaning + one-hot encoding)
  Hidden layer : 8 neurons, sigmoid activation
  Output layer : 1 neuron,  sigmoid activation
  Learning rate: 0.6
  Bias         : 0.4 (initial bias on hidden + output layers)
  Optimizer    : SGD (plain backprop)

Outputs (for stage-2 Streamlit deployment):
  - model.keras        (trained ANN)
  - preprocessor.pkl   (fitted ColumnTransformer for inference-time encoding)
"""

import pickle
import numpy as np
import pandas as pd
import arff  # liac-arff

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 21
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# 1. Load ARFF dataset
# ---------------------------------------------------------------------
with open("Final Dataset.arff", encoding="latin1") as f:
    raw = arff.load(f)

attrs = [a[0] for a in raw["attributes"]]
df = pd.DataFrame(raw["data"], columns=attrs)
df.columns = [c.strip() for c in df.columns]

print("Raw shape:", df.shape)
print(df.isnull().sum())

# ---------------------------------------------------------------------
# 2. Clean categorical inconsistencies
#    (e.g. 'Project complexity' vs 'Project Complexity' were being
#     treated as separate categories purely due to casing -- normalizing
#     this is what brings the feature count to exactly 64 after encoding)
# ---------------------------------------------------------------------
cat_cols = [
    "project Category", "Requirement Category", "Risk Target Category",
    "Magnitude of Risk", "Impact", "Dimension of Risk",
]
for c in cat_cols:
    df[c] = df[c].str.strip().str.title()

num_cols = [
    "Probability", "Afftecting No of Modules",
    "Fixing Duration (Days)", "Fix Cost (\\% of Project)", "Priority",
]

# ---------------------------------------------------------------------
# 3. Build binary target from Risk Level (1-5)
#    Low risk  = levels 1-2   ->  0
#    High risk = levels 3-5   ->  1
#    (single sigmoid output node needs a binary target)
# ---------------------------------------------------------------------
df["Risk Level"] = df["Risk Level"].astype(int)
df["RiskBinary"] = (df["Risk Level"] >= 3).astype(int)
print("\nBinary target distribution:")
print(df["RiskBinary"].value_counts())

# Drop free-text 'Requirements' column (out of scope for this ANN;
# would need separate NLP/TF-IDF encoding) and the original 5-level target
X = df.drop(columns=["Requirements", "Risk Level", "RiskBinary"])
y = df["RiskBinary"].values

# ---------------------------------------------------------------------
# 4. Train / validation / test split (mirrors reference notebook)
# ---------------------------------------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_train_full
)

print("\nTrain/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ---------------------------------------------------------------------
# 5. Preprocessing pipeline: impute + scale numerics, one-hot categoricals
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

if hasattr(X_train_t, "toarray"):
    X_train_t = X_train_t.toarray()
    X_val_t = X_val_t.toarray()
    X_test_t = X_test_t.toarray()

n_features = X_train_t.shape[1]
print("\nEncoded input feature count:", n_features)

# ---------------------------------------------------------------------
# 6. Build the ANN exactly per spec
# ---------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(
        8, activation="sigmoid",
        kernel_initializer=initializers.GlorotUniform(seed=RANDOM_STATE),
        bias_initializer=initializers.Constant(0.4),
        name="hidden_layer",
    ),
    layers.Dense(
        1, activation="sigmoid",
        kernel_initializer=initializers.GlorotUniform(seed=RANDOM_STATE),
        bias_initializer=initializers.Constant(0.4),
        name="output_layer",
    ),
])

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.6),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# ---------------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------------
history = model.fit(
    X_train_t, y_train,
    validation_data=(X_val_t, y_val),
    epochs=300,
    batch_size=8,
    callbacks=[early],
    verbose=2,
)

# ---------------------------------------------------------------------
# 8. Evaluate on held-out test set
# ---------------------------------------------------------------------
y_pred_prob = model.predict(X_test_t).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== TEST SET PERFORMANCE ===")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# 9. Save deployment artifacts (stage 2: Streamlit)
# ---------------------------------------------------------------------
model.save("model.keras")
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("\nSaved model.keras and preprocessor.pkl")
