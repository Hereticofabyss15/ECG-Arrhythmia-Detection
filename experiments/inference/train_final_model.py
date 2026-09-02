import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "final_xgboost"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

FEATURE_COLS = [
    c for c in df.columns
    if c not in ["patient_id", "database", "label"]
]

X = df[FEATURE_COLS]
y = df["label"]

classes = ["N", "SVEB", "VEB", "F", "Q"]

print(f"Samples : {len(df):,}")
print(f"Patients: {df['patient_id'].nunique()}")
print(f"Features: {len(FEATURE_COLS)}")

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()
label_encoder.fit(classes)

y_encoded = label_encoder.transform(y)

print("\nClass encoding:")

for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = pd.Series(y_encoded).value_counts()

total = len(y_encoded)

weights = {
    i: total /
    (len(classes) * class_counts[i])
    for i in range(len(classes))
}

sample_weights = np.array([
    weights[label]
    for label in y_encoded
])

print("\nClass weights:")

for encoded_class, weight in weights.items():

    class_name = label_encoder.inverse_transform(
        [encoded_class]
    )[0]

    print(
        f"{class_name}: {weight:.4f}"
    )

# ============================================================
# FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 70)

model_params = {
    "objective": "multi:softprob",
    "num_class": 5,

    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 600,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "min_child_weight": 2,
    "gamma": 0.1,

    "reg_alpha": 0,
    "reg_lambda": 1,

    "eval_metric": "mlogloss",
    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1
}

model = XGBClassifier(**model_params)

model.fit(
    X,
    y_encoded,
    sample_weight=sample_weights
)

print("\nFinal model training complete.")

# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.json"
)

model.save_model(MODEL_FILE)

print("\nModel saved:")
print(MODEL_FILE)

# ============================================================
# SAVE LABEL ENCODER
# ============================================================

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    ENCODER_FILE
)

print("\nLabel encoder saved:")
print(ENCODER_FILE)

# ============================================================
# SAVE FEATURE LIST
# ============================================================

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

with open(
    FEATURE_FILE,
    "w"
) as f:

    json.dump(
        FEATURE_COLS,
        f,
        indent=4
    )

print("\nFeature list saved:")
print(FEATURE_FILE)

# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

config = {
    "model_type": "XGBoost",
    "task": "5-class ECG arrhythmia classification",

    "classes": classes,

    "num_features": len(FEATURE_COLS),

    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 600,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "min_child_weight": 2,
    "gamma": 0.1,

    "reg_alpha": 0,
    "reg_lambda": 1,

    "random_state": 42,

    "training_samples": len(df),
    "training_patients": int(
        df["patient_id"].nunique()
    )
}

with open(
    CONFIG_FILE,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

print("\nModel configuration saved:")
print(CONFIG_FILE)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

IMPORTANCE_FILE = os.path.join(
    MODEL_DIR,
    "final_feature_importance.csv"
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)

print("\nFeature importance saved:")
print(IMPORTANCE_FILE)

print("\nTop 15 features:")

print(
    importance.head(15).to_string(
        index=False
    )
)

# ============================================================
# VERIFY MODEL
# ============================================================

print("\n" + "=" * 70)
print("MODEL VERIFICATION")
print("=" * 70)

sample_prob = model.predict_proba(
    X.iloc[:5]
)

sample_pred = np.argmax(
    sample_prob,
    axis=1
)

sample_labels = label_encoder.inverse_transform(
    sample_pred
)

print("Sample predictions:")

for i, label in enumerate(sample_labels):

    print(
        f"Sample {i + 1}: {label}"
    )

print("\nProbability sums:")

print(
    sample_prob.sum(axis=1)
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL CREATION COMPLETE")
print("=" * 70)

print("\nFiles created:")

for file in sorted(
    os.listdir(MODEL_DIR)
):

    print(
        os.path.join(
            MODEL_DIR,
            file
        )
    )