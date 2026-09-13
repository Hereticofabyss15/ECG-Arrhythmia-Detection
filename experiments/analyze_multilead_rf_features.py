import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "data/rhythm_multilead_fusion_features.csv"
OUTPUT_FILE = "data/multilead_rf_feature_importance.csv"

print("=" * 70)
print("MULTI-LEAD RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

DROP_FEATURES = ["rr_diff_std"]

metadata = [
    "record_id",
    "rhythm",
    "split"
]

features = [
    col for col in df.columns
    if col not in metadata
    and col not in DROP_FEATURES
]

assert len(features) == 99

train = df[df["split"] == "train"].copy()

X_train = train[features]
y_train = train["rhythm"]

print(f"\nTraining records : {len(train)}")
print(f"Features         : {len(features)}")

# ------------------------------------------------------------
# Train Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance["rank"] = np.arange(
    1,
    len(importance) + 1
)

importance = importance[
    ["rank", "feature", "importance"]
]

# ------------------------------------------------------------
# Print top features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance.head(30).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Lead-level importance
# ------------------------------------------------------------

def get_lead(feature):

    leads = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6"
    ]

    for lead in leads:

        if feature.startswith(lead + "_"):
            return lead

    return "Other"


importance["lead"] = importance[
    "feature"
].apply(get_lead)

lead_importance = (
    importance
    .groupby("lead")["importance"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("IMPORTANCE BY ECG LEAD")
print("=" * 70)

print(
    lead_importance.to_string()
)

# ------------------------------------------------------------
# Feature-family importance
# ------------------------------------------------------------

base_28 = [
    "mean_rr",
    "median_rr",
    "min_rr",
    "max_rr",
    "mean_hr",
    "min_hr",
    "max_hr",
    "sdnn",
    "rmssd",
    "sdsd",
    "mean_abs_rr_diff",
    "max_abs_rr_diff",
    "pnn50",
    "pnn100",
    "rr_diff_std",
    "rr_cv",
    "rr_iqr",
    "rr_entropy",
    "beat_prob_F",
    "beat_prob_N",
    "beat_prob_Q",
    "beat_prob_SVEB",
    "beat_prob_VEB",
    "ectopic_burden",
    "num_beats",
    "max_VEB_run",
    "beat_transition_rate",
    "beat_class_entropy"
]

base_28_used = [
    f for f in base_28
    if f in features
]

rr_hrv = [
    f for f in base_28_used
    if f in [
        "mean_rr",
        "median_rr",
        "min_rr",
        "max_rr",
        "mean_hr",
        "min_hr",
        "max_hr",
        "sdnn",
        "rmssd",
        "sdsd",
        "mean_abs_rr_diff",
        "max_abs_rr_diff",
        "pnn50",
        "pnn100",
        "rr_diff_std",
        "rr_cv",
        "rr_iqr",
        "rr_entropy"
    ]
]

beat_derived = [
    f for f in base_28_used
    if f not in rr_hrv
]

multi_lead = [
    f for f in features
    if f not in base_28_used
]

family_importance = {
    "RR/HRV": importance[
        importance["feature"].isin(rr_hrv)
    ]["importance"].sum(),

    "Beat-derived": importance[
        importance["feature"].isin(beat_derived)
    ]["importance"].sum(),

    "Multi-lead": importance[
        importance["feature"].isin(multi_lead)
    ]["importance"].sum()
}

family_importance = pd.Series(
    family_importance
).sort_values(
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE FAMILY IMPORTANCE")
print("=" * 70)

print(
    family_importance.to_string()
)

# ------------------------------------------------------------
# Multi-lead features only
# ------------------------------------------------------------

multi_importance = importance[
    importance["feature"].isin(multi_lead)
].copy()

print("\n" + "=" * 70)
print("TOP 30 MULTI-LEAD FEATURES")
print("=" * 70)

print(
    multi_importance.head(30).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

importance.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Saved to: {OUTPUT_FILE}")