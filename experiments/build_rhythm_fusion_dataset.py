import pandas as pd


# ============================================================
# LOAD RHYTHM FEATURE DATA
# ============================================================

rhythm = pd.read_csv(
    "data/rhythm_features_balanced.csv"
)

beat = pd.read_csv(
    "data/beat_derived_features.csv"
)


# ============================================================
# RECREATE SPLIT INFORMATION
# ============================================================

train_ids = pd.read_csv(
    "data/rhythm_train.csv"
)[["record_id"]]

val_ids = pd.read_csv(
    "data/rhythm_val.csv"
)[["record_id"]]

test_ids = pd.read_csv(
    "data/rhythm_test.csv"
)[["record_id"]]


train_ids["split"] = "train"
val_ids["split"] = "val"
test_ids["split"] = "test"

split_data = pd.concat(
    [
        train_ids,
        val_ids,
        test_ids
    ],
    ignore_index=True
)


# ============================================================
# KEEP ONLY BEAT-DERIVED FEATURES
# ============================================================

beat_columns = [
    "record_id",
    "beat_prob_F",
    "beat_prob_N",
    "beat_prob_Q",
    "beat_prob_SVEB",
    "beat_prob_VEB",
    "SVEB_burden",
    "VEB_burden",
    "F_burden",
    "ectopic_burden",
    "num_beats",
    "max_VEB_run",
    "beat_transition_rate",
    "beat_class_entropy"
]

beat = beat[
    beat_columns
]


# ============================================================
# MERGE RHYTHM + BEAT FEATURES
# ============================================================

merged = rhythm.merge(
    beat,
    on="record_id",
    how="inner",
    validate="one_to_one"
)


# ============================================================
# ADD TRAIN / VAL / TEST SPLIT
# ============================================================

merged = merged.merge(
    split_data,
    on="record_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 60)
print("RHYTHM FUSION DATASET")
print("=" * 60)

print(
    f"RR/HRV records: {len(rhythm)}"
)

print(
    f"Beat-derived records: {len(beat)}"
)

print(
    f"Merged records: {len(merged)}"
)

print(
    f"Unique record IDs: "
    f"{merged['record_id'].nunique()}"
)

print()

print("Missing values:")

print(
    merged.isna().sum().sum()
)

print()

print("Split distribution:")

print(
    merged["split"].value_counts()
)

print()

print("Class distribution:")

print(
    merged["rhythm"].value_counts()
)

print()

print("Split × class:")

print(
    pd.crosstab(
        merged["split"],
        merged["rhythm"]
    )
)


# ============================================================
# FINAL CHECKS
# ============================================================

if len(merged) != 3115:
    raise ValueError(
        f"Expected 3115 records, got {len(merged)}"
    )

if merged["record_id"].nunique() != 3115:
    raise ValueError(
        "Duplicate record IDs detected."
    )

if merged.isna().sum().sum() != 0:
    raise ValueError(
        "Missing values detected."
    )

if set(merged["split"]) != {
    "train",
    "val",
    "test"
}:
    raise ValueError(
        "Invalid split information."
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    "data/rhythm_fusion_features.csv"
)

merged.to_csv(
    output_path,
    index=False
)

print()
print(
    f"Saved to: {output_path}"
)

print(
    f"Dataset shape: {merged.shape}"
)