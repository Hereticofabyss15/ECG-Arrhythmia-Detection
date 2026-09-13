import pandas as pd

MULTILEAD_FILE = "data/multilead_features.csv"
FUSION_FILE = "data/rhythm_fusion_features.csv"
OUTPUT_FILE = "data/multilead_features_3115.csv"

print("=" * 70)
print("FILTER MULTI-LEAD FEATURES TO APPROVED 3115 RECORDS")
print("=" * 70)

# Load extracted multi-lead features
multi = pd.read_csv(MULTILEAD_FILE)

# Load authoritative 3115-record dataset
approved = pd.read_csv(FUSION_FILE)

approved = approved[["record_id", "rhythm", "split"]].copy()

print(f"\nExtracted multi-lead records : {len(multi)}")
print(f"Approved records             : {len(approved)}")

# Check approved dataset
assert approved["record_id"].nunique() == 3115, \
    "Approved dataset does not contain exactly 3115 unique records."

assert len(approved) == 3115, \
    "Approved dataset row count is not 3115."

# Check that multi-lead IDs are unique
assert multi["record_id"].is_unique, \
    "Multi-lead feature file contains duplicate record IDs."

# Keep ONLY the approved 3115 records
filtered = approved.merge(
    multi,
    on="record_id",
    how="left",
    suffixes=("", "_extracted")
)

# Verify rhythm consistency
mismatch = filtered[
    filtered["rhythm_extracted"].notna() &
    (filtered["rhythm"] != filtered["rhythm_extracted"])
]

if len(mismatch) > 0:
    print("\nERROR: Rhythm mismatch detected!")
    print(mismatch[["record_id", "rhythm", "rhythm_extracted"]])
    raise ValueError("Rhythm labels do not match.")

# Remove duplicate rhythm column
filtered.drop(columns=["rhythm_extracted"], inplace=True)

# Check for missing multi-lead features
feature_columns = [
    col for col in multi.columns
    if col != "record_id"
]

missing = filtered[feature_columns].isna().sum().sum()
infinite = (~filtered[feature_columns].applymap(pd.api.types.is_number)
            if False else 0)

print(f"\nMissing feature cells : {missing}")

if missing > 0:
    missing_records = filtered[
        filtered[feature_columns].isna().any(axis=1)
    ]

    print("\nRecords with missing features:")
    print(missing_records["record_id"].tolist())

    raise ValueError("Missing multi-lead features detected.")

# Final checks
assert len(filtered) == 3115
assert filtered["record_id"].nunique() == 3115

# Class distribution
print("\nClass distribution:")
print(filtered["rhythm"].value_counts().sort_index())

print("\nSplit distribution:")
print(filtered["split"].value_counts())

print("\nClass × split:")
print(
    pd.crosstab(
        filtered["rhythm"],
        filtered["split"]
    )
)

# Save
filtered.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("FILTERING COMPLETE")
print("=" * 70)
print(f"Final records : {len(filtered)}")
print(f"Final features: {len(multi.columns) - 1}")
print(f"Dataset shape : {filtered.shape}")
print(f"Saved to      : {OUTPUT_FILE}")