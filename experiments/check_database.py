import os
import glob
import pandas as pd

print("=" * 70)
print("CHECKING DATASET DATABASE / PATIENT STRUCTURE")
print("=" * 70)

# ------------------------------------------------------------
# FIND DATASET
# ------------------------------------------------------------

matches = glob.glob(
    os.path.abspath(
        os.path.join(
            os.getcwd(),
            "..",
            "**",
            "features.csv"
        )
    ),
    recursive=True
)

if not matches:
    raise FileNotFoundError("features.csv not found.")

DATA_PATH = matches[0]

print("\nDataset:")
print(DATA_PATH)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# DATABASE COLUMN
# ------------------------------------------------------------

if "database" in df.columns:

    print("\n" + "=" * 70)
    print("DATABASE COLUMN")
    print("=" * 70)

    print("\nUnique database values:")

    print(
        df["database"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRaw unique values:")

    for value in df["database"].drop_duplicates():
        print(repr(value))

else:

    print("\nNO DATABASE COLUMN FOUND.")


# ------------------------------------------------------------
# PATIENT INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT INFORMATION")
print("=" * 70)

print("\nNumber of unique patients:")

print(
    df["patient_id"].nunique()
)

print("\nPatients per database:")

if "database" in df.columns:

    table = pd.crosstab(
        df["database"],
        df["patient_id"]
    )

    print(
        table.shape
    )


# ------------------------------------------------------------
# PATIENT → DATABASE CONSISTENCY
# ------------------------------------------------------------

if "database" in df.columns:

    patient_database_counts = (
        df.groupby("patient_id")["database"]
        .nunique()
    )

    print("\nPatients appearing in multiple database values:")

    print(
        (patient_database_counts > 1).sum()
    )

    if (patient_database_counts > 1).sum() > 0:

        print("\nWARNING:")
        print(
            "Some patients occur under multiple database labels."
        )

        print(
            patient_database_counts[
                patient_database_counts > 1
            ].head(20)
        )


# ------------------------------------------------------------
# PATIENT DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT BEAT COUNTS")
print("=" * 70)

patient_counts = (
    df.groupby("patient_id")
    .size()
    .sort_values(ascending=False)
)

print(
    patient_counts.head(20)
)

print("\nSmallest patient beat counts:")

print(
    patient_counts.tail(20)
)

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)