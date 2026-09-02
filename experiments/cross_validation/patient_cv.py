# ============================================================
# ECG ARRHYTHMIA CLASSIFICATION
# PATIENT-LEVEL CROSS-VALIDATION
#
# 54 handcrafted DSP features
# Patient-independent evaluation
# StratifiedGroupKFold
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "features.csv"

if not DATA_PATH.exists():

    print("features.csv not found at expected location.")
    print("Searching recursively...")

    matches = list(PROJECT_ROOT.rglob("features.csv"))

    if not matches:
        raise FileNotFoundError(
            "Could not find features.csv"
        )

    DATA_PATH = matches[0]


print("=" * 70)
print("ECG ARRHYTHMIA CLASSIFICATION")
print("PATIENT-LEVEL CROSS-VALIDATION")
print("=" * 70)

print("\nDataset:")
print(DATA_PATH)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total beats   : {len(df):,}")
print(f"Total columns : {len(df.columns)}")


# ============================================================
# 3. BASIC CHECKS
# ============================================================

required_columns = [
    "patient_id",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# 4. SELECT ALL 54 FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    "patient_id",
    "label",
    "database"
}

FEATURES = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

print(f"\nTotal features: {len(FEATURES)}")

if len(FEATURES) != 54:

    print(
        "\nWARNING:"
        f" Expected 54 features but found {len(FEATURES)}"
    )


# ============================================================
# 5. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["label"]
)

classes = label_encoder.classes_

print("\nClass mapping:")

for i, cls in enumerate(classes):

    print(
        f"   {i} -> {cls}"
    )


# ============================================================
# 6. DATA MATRICES
# ============================================================

X = df[FEATURES].copy()

groups = df["patient_id"].values


# ============================================================
# 7. PATIENT INFORMATION
# ============================================================

unique_patients = np.unique(groups)

print(
    f"\nTotal patients: {len(unique_patients)}"
)


# ============================================================
# 8. STRATIFIED GROUP K-FOLD
# ============================================================

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. XGBOOST CONFIGURATION
# ============================================================

# Best configuration obtained during
# previous hyperparameter optimization.

XGB_PARAMS = {

    "objective": "multi:softprob",

    "num_class": len(classes),

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


# ============================================================
# 10. RESULT STORAGE
# ============================================================

fold_results = []

all_predictions = []

all_true_labels = []


# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 70)


    # --------------------------------------------------------
    # Patient IDs
    # --------------------------------------------------------

    train_patients = np.unique(
        groups[train_idx]
    )

    val_patients = np.unique(
        groups[val_idx]
    )

    print(
        f"\nTraining patients   : "
        f"{len(train_patients)}"
    )

    print(
        f"Validation patients : "
        f"{len(val_patients)}"
    )


    # --------------------------------------------------------
    # Verify patient independence
    # --------------------------------------------------------

    overlap = set(train_patients).intersection(
        set(val_patients)
    )

    if overlap:

        raise RuntimeError(
            f"Patient leakage detected in fold {fold}"
        )

    print(
        "✓ No patient overlap"
    )


    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    X_train = X.iloc[train_idx]

    X_val = X.iloc[val_idx]

    y_train = y[train_idx]

    y_val = y[val_idx]


    print(
        f"\nTraining beats   : "
        f"{len(train_idx):,}"
    )

    print(
        f"Validation beats : "
        f"{len(val_idx):,}"
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\nTraining distribution:")

    train_distribution = pd.Series(
        label_encoder.inverse_transform(y_train)
    ).value_counts()

    print(train_distribution)


    print("\nValidation distribution:")

    val_distribution = pd.Series(
        label_encoder.inverse_transform(y_val)
    ).value_counts()

    print(val_distribution)


    # ========================================================
    # IMPUTATION
    # ========================================================

    print("\nFitting median imputer...")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_val = imputer.transform(
        X_val
    )


    # ========================================================
    # SCALING
    # ========================================================

    print("Fitting StandardScaler...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weight_dict = {
        cls: weight
        for cls, weight in zip(
            np.unique(y_train),
            class_weights
        )
    }

    print("\nClass weights:")

    for cls, weight in weight_dict.items():

        class_name = label_encoder.inverse_transform(
            [cls]
        )[0]

        print(
            f"   {class_name:<5}: "
            f"{weight:.4f}"
        )


    # ========================================================
    # SAMPLE WEIGHTS
    # ========================================================

    sample_weights = np.array([

        weight_dict[label]

        for label in y_train

    ])


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = XGBClassifier(
        **XGB_PARAMS
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        verbose=False
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print("Generating predictions...")

    y_pred = model.predict(
        X_val
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        y_pred
    )

    macro_precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\nFold Results")
    print("-" * 40)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=classes,
            zero_division=0
        )
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_results.append({

        "fold": fold,

        "train_patients":
            len(train_patients),

        "validation_patients":
            len(val_patients),

        "train_beats":
            len(train_idx),

        "validation_beats":
            len(val_idx),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1
    })


    # ========================================================
    # STORE PREDICTIONS
    # ========================================================

    all_predictions.extend(
        y_pred
    )

    all_true_labels.extend(
        y_val
    )


# ============================================================
# 12. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


# ============================================================
# 13. MEAN / STANDARD DEVIATION
# ============================================================

metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1"
]


summary_rows = []

for metric in metrics:

    summary_rows.append({

        "metric": metric,

        "mean":
            results_df[metric].mean(),

        "std":
            results_df[metric].std(),

        "min":
            results_df[metric].min(),

        "max":
            results_df[metric].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 14. OVERALL OUT-OF-FOLD PERFORMANCE
# ============================================================

all_true_labels = np.array(
    all_true_labels
)

all_predictions = np.array(
    all_predictions
)


overall_accuracy = accuracy_score(
    all_true_labels,
    all_predictions
)

overall_balanced_accuracy = balanced_accuracy_score(
    all_true_labels,
    all_predictions
)

overall_macro_precision = precision_score(
    all_true_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

overall_macro_recall = recall_score(
    all_true_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

overall_macro_f1 = f1_score(
    all_true_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 15. OUTPUT DIRECTORY
# ============================================================

RESULT_DIR = (
    PROJECT_ROOT /
    "experiments" /
    "results" /
    "cross_validation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 16. SAVE FOLD RESULTS
# ============================================================

fold_results_path = (
    RESULT_DIR /
    "patient_cv_fold_results.csv"
)

results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary_path = (
    RESULT_DIR /
    "patient_cv_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n\n")
print("=" * 80)
print("PATIENT-LEVEL CROSS-VALIDATION SUMMARY")
print("=" * 80)

print("\nPer-fold results:")

print(
    results_df[
        [
            "fold",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 80)
print("MEAN ± STANDARD DEVIATION")
print("=" * 80)

for _, row in summary_df.iterrows():

    print(
        f"{row['metric']:<20} : "
        f"{row['mean']:.4f} ± {row['std']:.4f}"
    )


print("\n")
print("=" * 80)
print("OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 80)

print(
    f"\nAccuracy          : "
    f"{overall_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{overall_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : "
    f"{overall_macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{overall_macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{overall_macro_f1:.4f}"
)


# ============================================================
# 19. FINAL CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 80)
print("OVERALL OUT-OF-FOLD CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        all_true_labels,
        all_predictions,
        target_names=classes,
        zero_division=0
    )
)


# ============================================================
# 20. SAVE OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_path = (
    RESULT_DIR /
    "patient_cv_out_of_fold_predictions.csv"
)

oof_df = pd.DataFrame({

    "true_label":
        label_encoder.inverse_transform(
            all_true_labels
        ),

    "predicted_label":
        label_encoder.inverse_transform(
            all_predictions
        )
})

oof_df.to_csv(
    oof_path,
    index=False
)


# ============================================================
# 21. FILE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"\n1. {fold_results_path}"
)

print(
    f"2. {summary_path}"
)

print(
    f"3. {oof_path}"
)

print("\n")
print("✓ Patient-level grouping preserved")
print("✓ No patient appears in both train and validation")
print("✓ Preprocessing fitted separately inside every fold")
print("✓ Class weights calculated separately inside every fold")
print("✓ Test/held-out set was NOT used")
print("✓ Hyperparameters remained fixed")
print("\nCROSS-VALIDATION COMPLETE")
print("=" * 80)