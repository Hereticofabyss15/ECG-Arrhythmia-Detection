import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "results",
    "f_class_analysis"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

TARGET = "label"
PATIENT = "patient_id"

DROP_COLUMNS = [
    TARGET,
    PATIENT,
    "database"
]

FEATURES = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

print(f"Samples: {len(df):,}")
print(f"Patients: {df[PATIENT].nunique()}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

class_percent = (
    class_counts /
    len(df) *
    100
)

class_distribution = pd.DataFrame({
    "count": class_counts,
    "percentage": class_percent
})

print(class_distribution)

class_distribution.to_csv(
    os.path.join(
        RESULT_DIR,
        "class_distribution.csv"
    )
)


# ============================================================
# 2. F-CLASS PATIENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("F-CLASS PATIENT DISTRIBUTION")
print("=" * 70)

f_df = df[
    df[TARGET] == "F"
].copy()

f_patient_counts = (
    f_df
    .groupby(PATIENT)
    .size()
    .sort_values(ascending=False)
)

print(
    f"\nPatients containing F beats: "
    f"{len(f_patient_counts)}"
)

print(
    f"Total F beats: "
    f"{len(f_df)}"
)

print("\nTop patients by F-beat count:")

print(
    f_patient_counts.head(20)
)

f_patient_counts.to_csv(
    os.path.join(
        RESULT_DIR,
        "F_patient_distribution.csv"
    ),
    header=["F_beat_count"]
)


# ============================================================
# 3. F PATIENT CONCENTRATION
# ============================================================

print("\n" + "=" * 70)
print("F PATIENT CONCENTRATION")
print("=" * 70)

total_f = len(f_df)

concentration_rows = []

for n in [1, 3, 5, 10, 20]:

    if len(f_patient_counts) >= n:

        top_n = (
            f_patient_counts
            .head(n)
            .sum()
        )

        percentage = (
            top_n /
            total_f *
            100
        )

        concentration_rows.append({
            "top_patients": n,
            "F_beats": top_n,
            "percentage_of_all_F": percentage
        })

concentration_df = pd.DataFrame(
    concentration_rows
)

print(concentration_df)

concentration_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "F_patient_concentration.csv"
    ),
    index=False
)


# ============================================================
# 4. FEATURES: F VS ALL OTHER CLASSES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS: F VS NON-F")
print("=" * 70)

X = df[FEATURES].copy()

# Replace invalid values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)

y_binary = (
    df[TARGET] == "F"
).astype(int)


# ------------------------------------------------------------
# ANOVA F-TEST
# ------------------------------------------------------------

f_scores, p_values = f_classif(
    X,
    y_binary
)

feature_stats = pd.DataFrame({

    "feature": FEATURES,

    "F_score": f_scores,

    "p_value": p_values
})

feature_stats = (
    feature_stats
    .sort_values(
        "F_score",
        ascending=False
    )
)

print("\nTop features distinguishing F from non-F:")

print(
    feature_stats.head(20)
)

feature_stats.to_csv(
    os.path.join(
        RESULT_DIR,
        "F_vs_nonF_feature_statistics.csv"
    ),
    index=False
)


# ============================================================
# 5. F VS EACH INDIVIDUAL CLASS
# ============================================================

print("\n" + "=" * 70)
print("F VS INDIVIDUAL CLASSES")
print("=" * 70)

comparison_results = []

other_classes = [
    "N",
    "SVEB",
    "VEB",
    "Q"
]

for other_class in other_classes:

    print(
        f"\nAnalyzing F vs {other_class}..."
    )

    subset = df[
        df[TARGET].isin(
            ["F", other_class]
        )
    ]

    X_pair = subset[
        FEATURES
    ].copy()

    X_pair = X_pair.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_pair = X_pair.fillna(
        X_pair.median()
    )

    y_pair = (
        subset[TARGET] == "F"
    ).astype(int)

    scores, pvals = f_classif(
        X_pair,
        y_pair
    )

    pair_df = pd.DataFrame({

        "feature": FEATURES,

        "F_score": scores,

        "p_value": pvals

    })

    pair_df = pair_df.sort_values(
        "F_score",
        ascending=False
    )

    pair_df["comparison"] = (
        f"F_vs_{other_class}"
    )

    comparison_results.append(
        pair_df
    )

    print(
        pair_df.head(10)
    )


all_pairwise = pd.concat(
    comparison_results,
    ignore_index=True
)

all_pairwise.to_csv(
    os.path.join(
        RESULT_DIR,
        "F_pairwise_feature_statistics.csv"
    ),
    index=False
)


# ============================================================
# 6. FEATURE MEANS FOR EACH CLASS
# ============================================================

print("\n" + "=" * 70)
print("CLASS FEATURE MEANS")
print("=" * 70)

class_means = (
    df
    .groupby(TARGET)[FEATURES]
    .mean()
)

class_means.to_csv(
    os.path.join(
        RESULT_DIR,
        "class_feature_means.csv"
    )
)


# ============================================================
# 7. F CLASS VS OTHER CLASSES - TOP FEATURES
# ============================================================

top_features = (
    feature_stats
    .head(10)["feature"]
    .tolist()
)

print("\nTop 10 F-discriminative features:")

for i, feature in enumerate(
    top_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 8. PLOT CLASS DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

class_counts.plot(
    kind="bar"
)

plt.title(
    "ECG Beat Class Distribution"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Beats"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "class_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. PLOT F PATIENT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 5)
)

f_patient_counts.plot(
    kind="bar"
)

plt.title(
    "F-Class Beats per Patient"
)

plt.xlabel(
    "Patient ID"
)

plt.ylabel(
    "Number of F Beats"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "F_beats_per_patient.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. TOP FEATURE IMPORTANCE-STYLE PLOT
# ============================================================

top_plot = (
    feature_stats
    .head(15)
    .sort_values(
        "F_score"
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.barh(
    top_plot["feature"],
    top_plot["F_score"]
)

plt.title(
    "Top Features Distinguishing F from Non-F"
)

plt.xlabel(
    "ANOVA F-score"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "F_top_features.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("F-CLASS ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nTotal F beats: {len(f_df):,}"
)

print(
    f"F percentage: "
    f"{len(f_df) / len(df) * 100:.3f}%"
)

print(
    f"Patients containing F: "
    f"{len(f_patient_counts)}"
)

print("\nResults saved to:")

print(RESULT_DIR)

print("\nGenerated files:")

for file in sorted(
    os.listdir(RESULT_DIR)
):

    print(
        "  ",
        file
    )

print("\nDone.")