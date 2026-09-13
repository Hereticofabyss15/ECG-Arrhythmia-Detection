import pandas as pd
import numpy as np

MULTILEAD_FILE = "data/multilead_features_3115.csv"
BASE_FUSION_FILE = "data/rhythm_fusion_features.csv"

OUTPUT_FILE = "data/rhythm_multilead_fusion_features.csv"
REDUNDANCY_FILE = "data/multilead_redundancy_report.csv"

print("=" * 70)
print("BUILD MULTI-LEAD FUSION DATASET + REDUNDANCY CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

multilead = pd.read_csv(MULTILEAD_FILE)
base = pd.read_csv(BASE_FUSION_FILE)

print(f"\nMulti-lead dataset : {multilead.shape}")
print(f"Existing fusion   : {base.shape}")

# ------------------------------------------------------------
# Select existing 28-feature representation
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
    "beat_class_entropy",
]

base = base[
    ["record_id", "rhythm", "split"] + base_28
].copy()

# ------------------------------------------------------------
# Verify datasets
# ------------------------------------------------------------

assert len(base) == 3115
assert len(multilead) == 3115

assert base["record_id"].is_unique
assert multilead["record_id"].is_unique

assert set(base["record_id"]) == set(multilead["record_id"])

print("\nRecord ID check : PASS")
print("Records         : 3115")

# ------------------------------------------------------------
# Multi-lead feature columns
# ------------------------------------------------------------

multi_feature_columns = [
    col for col in multilead.columns
    if col not in ["record_id", "rhythm", "split"]
]

assert len(multi_feature_columns) == 72

print(f"Multi-lead ECG features : {len(multi_feature_columns)}")

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

final = base.merge(
    multilead[["record_id"] + multi_feature_columns],
    on="record_id",
    how="inner",
    validate="one_to_one"
)

assert len(final) == 3115
assert final["record_id"].is_unique

# ------------------------------------------------------------
# Final feature list
# ------------------------------------------------------------

metadata_columns = [
    "record_id",
    "rhythm",
    "split"
]

final_features = base_28 + multi_feature_columns

assert len(base_28) == 28
assert len(multi_feature_columns) == 72
assert len(final_features) == 100

final = final[
    metadata_columns + final_features
].copy()

print(f"\nExisting features : {len(base_28)}")
print(f"Multi-lead features: {len(multi_feature_columns)}")
print(f"Total model features: {len(final_features)}")

# ------------------------------------------------------------
# Data quality
# ------------------------------------------------------------

numeric = final[final_features]

missing = numeric.isna().sum().sum()
infinite = np.isinf(numeric.to_numpy()).sum()

print(f"\nMissing cells   : {missing}")
print(f"Infinite values : {infinite}")

assert missing == 0
assert infinite == 0

# ------------------------------------------------------------
# Redundancy analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REDUNDANCY ANALYSIS")
print("=" * 70)

# Exact duplicate columns
duplicate_columns = []

for i, col1 in enumerate(final_features):
    for col2 in final_features[i + 1:]:
        if numeric[col1].equals(numeric[col2]):
            duplicate_columns.append(
                (col1, col2)
            )

print(f"\nExact duplicate feature pairs : {len(duplicate_columns)}")

for col1, col2 in duplicate_columns:
    print(f"  {col1} == {col2}")

# Zero variance
variances = numeric.var()

zero_variance = variances[
    variances == 0
]

print(f"\nZero-variance features : {len(zero_variance)}")

for feature in zero_variance.index:
    print(f"  {feature}")

# High correlation
corr = numeric.corr()

high_corr_pairs = []

for i, col1 in enumerate(final_features):

    for col2 in final_features[i + 1:]:

        value = corr.loc[col1, col2]

        if abs(value) >= 0.98:

            high_corr_pairs.append(
                (col1, col2, value)
            )

high_corr_pairs.sort(
    key=lambda x: abs(x[2]),
    reverse=True
)

print(
    f"\nHighly correlated pairs "
    f"(|r| >= 0.98) : {len(high_corr_pairs)}"
)

for col1, col2, value in high_corr_pairs:

    print(
        f"  {col1:35s} "
        f"{col2:35s} "
        f"r = {value:.4f}"
    )

# ------------------------------------------------------------
# Save redundancy report
# ------------------------------------------------------------

redundancy_rows = []

for col1, col2 in duplicate_columns:

    redundancy_rows.append({
        "type": "exact_duplicate",
        "feature_1": col1,
        "feature_2": col2,
        "correlation": 1.0
    })

for col1, col2, value in high_corr_pairs:

    redundancy_rows.append({
        "type": "high_correlation",
        "feature_1": col1,
        "feature_2": col2,
        "correlation": value
    })

redundancy_report = pd.DataFrame(
    redundancy_rows,
    columns=[
        "type",
        "feature_1",
        "feature_2",
        "correlation"
    ]
)

redundancy_report.to_csv(
    REDUNDANCY_FILE,
    index=False
)

# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

assert final.shape == (3115, 103)

print("\nClass distribution:")
print(
    final["rhythm"]
    .value_counts()
    .sort_index()
)

print("\nSplit distribution:")
print(
    final["split"]
    .value_counts()
)

print("\nClass × split:")
print(
    pd.crosstab(
        final["rhythm"],
        final["split"]
    )
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("MULTI-LEAD FUSION DATASET COMPLETE")
print("=" * 70)

print(f"Records           : {len(final)}")
print(f"Model features    : {len(final_features)}")
print(f"Dataset shape     : {final.shape}")
print(f"Saved dataset     : {OUTPUT_FILE}")
print(f"Redundancy report : {REDUNDANCY_FILE}")