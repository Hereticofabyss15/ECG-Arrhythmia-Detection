import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = "data/rhythm_multilead_fusion_features.csv"
MODEL_PATH = "models/final_rhythm_rf_model.pkl"
FEATURE_PATH = "models/final_rhythm_feature_names.json"


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(DATA_PATH)

metadata_cols = [
    "record_id",
    "rhythm",
    "split"
]

# Exact duplicate removed during final feature selection
drop_feature = "rr_diff_std"

feature_cols = [
    col
    for col in df.columns
    if col not in metadata_cols + [drop_feature]
]


# ================================================================
# TRAIN + VALIDATION DATA ONLY
# ================================================================

train_val = df[
    df["split"].isin(["train", "val"])
].copy()

X_train_val = train_val[feature_cols]
y_train_val = train_val["rhythm"]


print("=" * 70)
print("TRAINING FINAL RHYTHM RANDOM FOREST")
print("=" * 70)

print(f"Training records: {len(train_val)}")
print(f"Number of features: {len(feature_cols)}")
print(f"Number of classes: {y_train_val.nunique()}")

print("\nClass distribution:")
print(y_train_val.value_counts().sort_index())


# ================================================================
# FINAL RANDOM FOREST
# ================================================================

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ================================================================
# TRAIN
# ================================================================

print("\nTraining final model...")

model.fit(
    X_train_val,
    y_train_val
)


# ================================================================
# SAVE MODEL
# ================================================================

import os
import json

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)

with open(FEATURE_PATH, "w") as f:
    json.dump(
        feature_cols,
        f,
        indent=2
    )


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print(f"Model:")
print(MODEL_PATH)

print("\nFeature list:")
print(FEATURE_PATH)

print(f"\nFeatures saved: {len(feature_cols)}")
print("Training data: Train + Validation")
print("Test data: NOT USED")

print("\nFinal model is ready for inference.")