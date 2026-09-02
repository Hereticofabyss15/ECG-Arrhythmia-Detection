import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "results",
    "cross_validation",
    "patient_cv_out_of_fold_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "results",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading out-of-fold predictions...")

df = pd.read_csv(PREDICTION_FILE)

print(f"Samples: {len(df):,}")
print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# DETECT COLUMN NAMES
# ============================================================

true_col = None
pred_col = None
patient_col = None

for col in df.columns:
    c = col.lower()

    if c in ["true_label", "y_true", "true", "actual", "label"]:
        true_col = col

    if c in ["predicted_label", "y_pred", "pred", "prediction"]:
        pred_col = col

    if "patient" in c:
        patient_col = col

if true_col is None or pred_col is None:
    raise ValueError(
        "Could not identify true/predicted label columns. "
        "Check the printed column names."
    )

print(f"\nTrue label column: {true_col}")
print(f"Predicted label column: {pred_col}")

if patient_col:
    print(f"Patient column: {patient_col}")

# ============================================================
# BASIC ERROR ANALYSIS
# ============================================================

df["correct"] = df[true_col] == df[pred_col]
df["error"] = ~df["correct"]

accuracy = df["correct"].mean()

print("\n" + "=" * 70)
print("OVERALL ERROR ANALYSIS")
print("=" * 70)

print(f"Overall accuracy: {accuracy:.4f}")
print(f"Correct predictions: {df['correct'].sum():,}")
print(f"Incorrect predictions: {df['error'].sum():,}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = ["N", "SVEB", "VEB", "F", "Q"]

cm = pd.crosstab(
    df[true_col],
    df[pred_col],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
)

cm = cm.reindex(index=classes, columns=classes, fill_value=0)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print(cm)

cm.to_csv(
    os.path.join(OUTPUT_DIR, "error_analysis_confusion_matrix.csv")
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Patient-Level OOF Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "error_analysis_confusion_matrix.png"),
    dpi=300
)

plt.close()

# ============================================================
# PER-CLASS ERROR RATE
# ============================================================

class_results = []

for cls in classes:

    actual = df[true_col] == cls

    total = actual.sum()
    correct = ((df[true_col] == cls) &
               (df[pred_col] == cls)).sum()

    incorrect = total - correct

    recall = correct / total if total > 0 else 0
    error_rate = incorrect / total if total > 0 else 0

    class_results.append({
        "class": cls,
        "total_samples": total,
        "correct": correct,
        "incorrect": incorrect,
        "recall": recall,
        "error_rate": error_rate
    })

class_results = pd.DataFrame(class_results)

print("\n" + "=" * 70)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 70)
print(class_results.to_string(index=False))

class_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_class_error_analysis.csv"
    ),
    index=False
)

# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

errors = df[df["error"]].copy()

confusions = (
    errors
    .groupby([true_col, pred_col])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n" + "=" * 70)
print("MOST COMMON MISCLASSIFICATIONS")
print("=" * 70)

print(confusions.head(20).to_string(index=False))

confusions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "most_common_misclassifications.csv"
    ),
    index=False
)

# ============================================================
# F-CLASS ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("F-CLASS ERROR ANALYSIS")
print("=" * 70)

f_samples = df[df[true_col] == "F"]

print(f"Total F samples: {len(f_samples):,}")
print(
    f"Correctly classified F: "
    f"{(f_samples[pred_col] == 'F').sum():,}"
)

print("\nF predictions:")
print(
    f_samples[pred_col]
    .value_counts()
)

f_confusion = (
    f_samples[pred_col]
    .value_counts()
    .rename_axis("predicted_class")
    .reset_index(name="count")
)

f_confusion["percentage"] = (
    f_confusion["count"] /
    len(f_samples) *
    100
)

print("\nF misclassification distribution:")
print(f_confusion.to_string(index=False))

f_confusion.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "F_class_error_analysis.csv"
    ),
    index=False
)

# ============================================================
# PATIENT-LEVEL ERROR ANALYSIS
# ============================================================

if patient_col:

    patient_errors = (
        df.groupby(patient_col)
        .agg(
            total_samples=("correct", "size"),
            errors=("error", "sum")
        )
        .reset_index()
    )

    patient_errors["error_rate"] = (
        patient_errors["errors"] /
        patient_errors["total_samples"]
    )

    patient_errors = patient_errors.sort_values(
        "error_rate",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("PATIENTS WITH HIGHEST ERROR RATE")
    print("=" * 70)

    print(
        patient_errors.head(20)
        .to_string(index=False)
    )

    patient_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_error_rates.csv"
        ),
        index=False
    )

# ============================================================
# ERROR DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    class_results["class"],
    class_results["error_rate"]
)

plt.xlabel("Class")
plt.ylabel("Error Rate")
plt.title("Error Rate by Arrhythmia Class")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_error_rates.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# SAVE ALL ERRORS
# ============================================================

errors.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_misclassified_samples.csv"
    ),
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(file)