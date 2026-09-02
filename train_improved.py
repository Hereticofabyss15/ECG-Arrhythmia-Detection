# ======================================================================
# IMPROVED ECG ARRHYTHMIA CLASSIFICATION
# TWO-STAGE XGBOOST
# PATIENT-INDEPENDENT EVALUATION
#
# IMPORTANT:
# - features.csv is NOT modified
# - build_dataset.py is NOT executed
# - No new features are generated
# - Existing 54 features are used exactly as stored in features.csv
# - Stage 2 prediction indexing bug is fixed
# ======================================================================

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

# Existing dataset only
DATASET_NAME = "features.csv"

# Output directory
MODEL_DIR_NAME = "models"

# Labels
LABELS = ["N", "SVEB", "VEB", "F", "Q"]

# Stage 1 labels
NORMAL_LABEL = "Normal"
ABNORMAL_LABEL = "Abnormal"

# Number of trees
STAGE1_ESTIMATORS = 500
STAGE2_ESTIMATORS = 600

# XGBoost parameters
LEARNING_RATE = 0.05
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 3
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85
REG_ALPHA = 0.1
REG_LAMBDA = 2.0

# Threshold search
THRESHOLDS = np.round(
    np.arange(0.10, 0.51, 0.01),
    2
)


# ======================================================================
# PRINT HEADER
# ======================================================================

print("=" * 70)
print("IMPROVED ECG ARRHYTHMIA CLASSIFICATION")
print("TWO-STAGE XGBOOST")
print("PATIENT-INDEPENDENT EVALUATION")
print("=" * 70)


# ======================================================================
# FIND features.csv
# ======================================================================

print("\nSearching for features.csv...")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

possible_paths = [
    os.path.join(current_dir, DATASET_NAME),
    os.path.join(project_dir, DATASET_NAME),
    os.path.join(project_dir, "data", DATASET_NAME),
    os.path.join(current_dir, "data", DATASET_NAME),
]

dataset_path = None

for path in possible_paths:
    if os.path.isfile(path):
        dataset_path = path
        break


if dataset_path is None:

    # Recursive search as final fallback
    for root, dirs, files in os.walk(project_dir):

        # Ignore virtual environments
        dirs[:] = [
            d for d in dirs
            if d not in ["venv", ".venv", "__pycache__"]
        ]

        if DATASET_NAME in files:
            dataset_path = os.path.join(root, DATASET_NAME)
            break


if dataset_path is None:
    raise FileNotFoundError(
        "\nfeatures.csv not found.\n\n"
        "Expected locations include:\n"
        f"  {os.path.join(project_dir, 'data', 'features.csv')}\n\n"
        "Make sure your existing features.csv is present.\n"
        "This script does NOT build or modify the dataset."
    )


print("\n✓ Dataset found:")
print(dataset_path)


# ======================================================================
# LOAD DATASET
# ======================================================================

print("\nLoading dataset...")

df = pd.read_csv(dataset_path)

print(f"Total beats: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nDataset columns:")
print(df.columns.tolist())


# ======================================================================
# IDENTIFY PATIENT AND LABEL COLUMNS
# ======================================================================

possible_patient_columns = [
    "patient_id",
    "patient",
    "record",
    "record_id",
    "subject_id"
]

patient_column = None

for col in possible_patient_columns:
    if col in df.columns:
        patient_column = col
        break


if patient_column is None:
    raise ValueError(
        "Could not identify patient/record column."
    )


if "label" not in df.columns:
    raise ValueError(
        "Dataset does not contain a 'label' column."
    )


print(f"\n✓ Patient/record column: {patient_column}")
print("✓ Label column: label")


# ======================================================================
# NORMALIZE LABELS
# ======================================================================

df["label"] = df["label"].astype(str).str.strip()


# Keep only the five expected classes
df = df[df["label"].isin(LABELS)].copy()

df.reset_index(drop=True, inplace=True)


print("\nLabel distribution:")
print(df["label"].value_counts())


# ======================================================================
# FEATURE SELECTION
# ======================================================================
#
# IMPORTANT:
# We use ONLY existing numerical features.
#
# patient_id / database / label are metadata and are excluded.
#
# No new features are created.
# ======================================================================

EXCLUDED_COLUMNS = [
    patient_column,
    "database",
    "label"
]

feature_columns = [
    col
    for col in df.columns
    if col not in EXCLUDED_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]


if len(feature_columns) == 0:
    raise ValueError("No numerical feature columns found.")


print(f"\nTotal features: {len(feature_columns)}")

if len(feature_columns) != 54:
    print(
        "\nWARNING:"
        f" Expected 54 existing features, found {len(feature_columns)}."
    )


X_all = df[feature_columns].copy()
y_all = df["label"].copy()
patients_all = df[patient_column].copy()


# ======================================================================
# PATIENT-INDEPENDENT DS1 / DS2 SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("PATIENT-INDEPENDENT DS1 / DS2 SPLIT")
print("=" * 70)


unique_patients = np.array(
    sorted(patients_all.unique(), key=lambda x: str(x))
)

rng = np.random.RandomState(RANDOM_STATE)

shuffled_patients = unique_patients.copy()
rng.shuffle(shuffled_patients)

split_index = len(shuffled_patients) // 2

train_patients = set(shuffled_patients[:split_index])
test_patients = set(shuffled_patients[split_index:])


train_mask = patients_all.isin(train_patients).values
test_mask = patients_all.isin(test_patients).values


X_train_full = X_all.loc[train_mask].copy()
y_train_full = y_all.loc[train_mask].copy()
patients_train_full = patients_all.loc[train_mask].copy()

X_test = X_all.loc[test_mask].copy()
y_test = y_all.loc[test_mask].copy()
patients_test = patients_all.loc[test_mask].copy()


print(f"Training records: {len(train_patients)}")
print(f"Testing records : {len(test_patients)}")

print(f"Training beats: {len(X_train_full)}")
print(f"Testing beats : {len(X_test)}")


overlap = train_patients.intersection(test_patients)

if len(overlap) > 0:
    raise RuntimeError(
        "Patient overlap detected between training and testing!"
    )

print("\n✓ No patient overlap between training and testing.")


print("\nTraining distribution:")
print(y_train_full.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())


# ======================================================================
# CHECK INVALID VALUES
# ======================================================================

train_nan_count = X_train_full.isna().sum().sum()
test_nan_count = X_test.isna().sum().sum()

print("\nInvalid values:")
print(f"Training NaN: {train_nan_count}")
print(f"Testing NaN : {test_nan_count}")


# ======================================================================
# PATIENT-INDEPENDENT TRAIN / VALIDATION SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("PATIENT-INDEPENDENT TRAIN / VALIDATION SPLIT")
print("=" * 70)


development_patients = np.array(
    sorted(train_patients, key=lambda x: str(x))
)

rng = np.random.RandomState(RANDOM_STATE)

development_patients_shuffled = development_patients.copy()
rng.shuffle(development_patients_shuffled)


validation_count = max(
    1,
    int(round(len(development_patients_shuffled) * 0.20))
)

validation_patients = set(
    development_patients_shuffled[:validation_count]
)

actual_train_patients = set(
    development_patients_shuffled[validation_count:]
)


train_dev_mask = patients_train_full.isin(
    actual_train_patients
).values

validation_mask = patients_train_full.isin(
    validation_patients
).values


X_train = X_train_full.loc[train_dev_mask].copy()
y_train = y_train_full.loc[train_dev_mask].copy()
patients_train = patients_train_full.loc[train_dev_mask].copy()


X_val = X_train_full.loc[validation_mask].copy()
y_val = y_train_full.loc[validation_mask].copy()
patients_val = patients_train_full.loc[validation_mask].copy()


print(
    f"Patients available for development: "
    f"{len(development_patients)}"
)

print(
    f"\nTraining patients   : "
    f"{len(actual_train_patients)}"
)

print(
    f"Validation patients : "
    f"{len(validation_patients)}"
)

print(f"\nTraining beats      : {len(X_train)}")
print(f"Validation beats    : {len(X_val)}")


patient_overlap = (
    actual_train_patients.intersection(validation_patients)
)

if len(patient_overlap) > 0:
    raise RuntimeError(
        "Patient overlap detected between train and validation!"
    )

print("\n✓ No patient overlap between training and validation.")


print("\nTraining distribution:")
print(y_train.value_counts())

print("\nValidation distribution:")
print(y_val.value_counts())


print("\nTraining class proportions:")
print(y_train.value_counts(normalize=True))

print("\nValidation class proportions:")
print(y_val.value_counts(normalize=True))


missing_validation_classes = set(LABELS) - set(y_val.unique())

if missing_validation_classes:
    print(
        "\nWARNING: Validation set is missing:",
        missing_validation_classes
    )
else:
    print("\n✓ All five classes are present in validation.")


# ======================================================================
# IMPUTATION
# ======================================================================

print("\nFitting median imputer...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)


# ======================================================================
# STANDARDIZATION
# ======================================================================

print("Fitting StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)


# ======================================================================
# STAGE 1
# NORMAL VS ABNORMAL
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 1: NORMAL VS ABNORMAL")
print("=" * 70)


y_train_stage1 = np.where(
    y_train.values == "N",
    NORMAL_LABEL,
    ABNORMAL_LABEL
)

y_val_stage1 = np.where(
    y_val.values == "N",
    NORMAL_LABEL,
    ABNORMAL_LABEL
)

y_test_stage1 = np.where(
    y_test.values == "N",
    NORMAL_LABEL,
    ABNORMAL_LABEL
)


print("\nTraining distribution:")
print(pd.Series(y_train_stage1).value_counts())


# ======================================================================
# STAGE 1 CLASS WEIGHTS
# ======================================================================

stage1_counts = Counter(y_train_stage1)

normal_count = stage1_counts[NORMAL_LABEL]
abnormal_count = stage1_counts[ABNORMAL_LABEL]

stage1_total = normal_count + abnormal_count
stage1_classes = 2

normal_weight = (
    stage1_total /
    (stage1_classes * normal_count)
)

abnormal_weight = (
    stage1_total /
    (stage1_classes * abnormal_count)
)

stage1_weights = {
    NORMAL_LABEL: normal_weight,
    ABNORMAL_LABEL: abnormal_weight
}


print("\nStage 1 weights:")
print(stage1_weights)


stage1_sample_weights = np.array([
    stage1_weights[label]
    for label in y_train_stage1
])


# ======================================================================
# STAGE 1 ENCODING
# ======================================================================

stage1_mapping = {
    NORMAL_LABEL: 0,
    ABNORMAL_LABEL: 1
}

y_train_stage1_encoded = np.array([
    stage1_mapping[x]
    for x in y_train_stage1
])

y_val_stage1_encoded = np.array([
    stage1_mapping[x]
    for x in y_val_stage1
])

y_test_stage1_encoded = np.array([
    stage1_mapping[x]
    for x in y_test_stage1
])


# ======================================================================
# STAGE 1 XGBOOST
# ======================================================================

print("\nTraining Stage 1 XGBoost...")


stage1_model = XGBClassifier(
    n_estimators=STAGE1_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    min_child_weight=MIN_CHILD_WEIGHT,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    reg_alpha=REG_ALPHA,
    reg_lambda=REG_LAMBDA,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)


stage1_model.fit(
    X_train_scaled,
    y_train_stage1_encoded,
    sample_weight=stage1_sample_weights
)


print("Stage 1 training complete.")


# ======================================================================
# STAGE 2
# ABNORMAL CLASS CLASSIFICATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 2: ABNORMAL CLASS CLASSIFICATION")
print("=" * 70)


abnormal_train_mask = y_train.values != "N"
abnormal_val_mask = y_val.values != "N"
abnormal_test_mask = y_test.values != "N"


X_train_stage2 = X_train_scaled[abnormal_train_mask]
y_train_stage2 = y_train.values[abnormal_train_mask]

X_val_stage2 = X_val_scaled[abnormal_val_mask]
y_val_stage2 = y_val.values[abnormal_val_mask]

X_test_stage2 = X_test_scaled[abnormal_test_mask]
y_test_stage2 = y_test.values[abnormal_test_mask]


print(
    f"\nAbnormal training samples: "
    f"{len(X_train_stage2)}"
)

print(
    f"Abnormal validation samples: "
    f"{len(X_val_stage2)}"
)


print("\nTraining distribution:")
print(pd.Series(y_train_stage2).value_counts())


# ======================================================================
# STAGE 2 LABEL ENCODING
# ======================================================================

stage2_mapping = {
    "SVEB": 0,
    "VEB": 1,
    "F": 2,
    "Q": 3
}

stage2_reverse_mapping = {
    value: key
    for key, value in stage2_mapping.items()
}


y_train_stage2_encoded = np.array([
    stage2_mapping[x]
    for x in y_train_stage2
])

y_val_stage2_encoded = np.array([
    stage2_mapping[x]
    for x in y_val_stage2
])


# ======================================================================
# STAGE 2 CLASS WEIGHTS
# ======================================================================

stage2_counts = Counter(y_train_stage2)

stage2_total = len(y_train_stage2)
stage2_num_classes = 4


initial_stage2_weights = {}

for label in stage2_mapping:

    count = stage2_counts.get(label, 0)

    if count > 0:
        initial_stage2_weights[label] = (
            stage2_total /
            (stage2_num_classes * count)
        )
    else:
        initial_stage2_weights[label] = 1.0


print("\nInitial Stage 2 weights:")
print(initial_stage2_weights)


# ======================================================================
# MODERATE RARE CLASS BOOST
# ======================================================================
#
# F and Q can be extremely rare depending on the patient split.
#
# We prevent excessively large weights from completely dominating
# the XGBoost model.
#
# Dataset remains untouched.
# ======================================================================

stage2_weights = initial_stage2_weights.copy()


for label in stage2_weights:

    if label == "F":
        stage2_weights[label] *= 1.10

    elif label == "SVEB":
        stage2_weights[label] *= 1.35

    elif label == "Q":
        stage2_weights[label] *= 1.10


# Cap extreme weights
MAX_STAGE2_WEIGHT = 30.0

for label in stage2_weights:

    stage2_weights[label] = min(
        stage2_weights[label],
        MAX_STAGE2_WEIGHT
    )


print("\nAdjusted Stage 2 weights:")
print(stage2_weights)


stage2_sample_weights = np.array([
    stage2_weights[label]
    for label in y_train_stage2
])


# ======================================================================
# STAGE 2 XGBOOST
# ======================================================================

print("\nTraining Stage 2 XGBoost...")


stage2_model = XGBClassifier(
    n_estimators=STAGE2_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    min_child_weight=MIN_CHILD_WEIGHT,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    reg_alpha=REG_ALPHA,
    reg_lambda=REG_LAMBDA,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)


stage2_model.fit(
    X_train_stage2,
    y_train_stage2_encoded,
    sample_weight=stage2_sample_weights
)


print("Stage 2 training complete.")


# ======================================================================
# PREDICTION FUNCTION
# ======================================================================
#
# IMPORTANT BUG FIX
# -----------------
#
# Stage 2 only sees abnormal samples.
#
# Therefore:
#
# stage2_predictions has length = number of abnormal samples
#
# while final_predictions has length = number of ALL samples.
#
# We MUST map Stage 2 predictions back using abnormal_indices.
# ======================================================================

def two_stage_predict(
    X,
    stage1_model,
    stage2_model,
    stage2_reverse_mapping,
    threshold=0.50
):

    # --------------------------------------------------------------
    # Stage 1 probability
    # --------------------------------------------------------------

    stage1_probabilities = stage1_model.predict_proba(X)

    # Probability of Abnormal
    abnormal_probability = stage1_probabilities[:, 1]

    # --------------------------------------------------------------
    # Stage 1 decision
    # --------------------------------------------------------------

    stage1_abnormal = (
        abnormal_probability >= threshold
    )

    # --------------------------------------------------------------
    # Create output for ALL samples
    #
    # Default = Normal
    # --------------------------------------------------------------

    final_predictions = np.full(
        len(X),
        "N",
        dtype=object
    )

    # --------------------------------------------------------------
    # Get indices of samples sent to Stage 2
    # --------------------------------------------------------------

    abnormal_indices = np.where(
        stage1_abnormal
    )[0]

    # --------------------------------------------------------------
    # Nothing abnormal
    # --------------------------------------------------------------

    if len(abnormal_indices) == 0:

        return (
            final_predictions,
            abnormal_probability
        )

    # --------------------------------------------------------------
    # Stage 2 receives ONLY abnormal rows
    # --------------------------------------------------------------

    X_abnormal = X[abnormal_indices]

    stage2_probabilities = stage2_model.predict_proba(
        X_abnormal
    )

    stage2_predictions_encoded = np.argmax(
        stage2_probabilities,
        axis=1
    )

    stage2_predictions = np.array([
        stage2_reverse_mapping[int(pred)]
        for pred in stage2_predictions_encoded
    ])

    # --------------------------------------------------------------
    # CRITICAL FIX
    #
    # Put Stage 2 predictions back into their ORIGINAL positions.
    #
    # OLD BUG:
    #
    # final_predictions[i] = stage2_predictions[i]
    #
    # This fails because stage2_predictions is shorter.
    #
    # CORRECT:
    #
    # final_predictions[abnormal_indices] = stage2_predictions
    # --------------------------------------------------------------

    final_predictions[abnormal_indices] = stage2_predictions

    return (
        final_predictions,
        abnormal_probability
    )


# ======================================================================
# STAGE 1 THRESHOLD SEARCH
# ======================================================================

print("\n" + "=" * 70)
print("SEARCHING STAGE 1 THRESHOLD")
print("=" * 70)


threshold_results = []


for threshold in THRESHOLDS:

    predictions, abnormal_probabilities = two_stage_predict(
        X_val_scaled,
        stage1_model,
        stage2_model,
        stage2_reverse_mapping,
        threshold=threshold
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    normal_true = (
        y_val.values == "N"
    )

    abnormal_true = ~normal_true

    normal_pred = (
        predictions == "N"
    )

    abnormal_pred = ~normal_pred

    normal_recall = (
        np.sum(normal_true & normal_pred)
        /
        max(np.sum(normal_true), 1)
    )

    abnormal_recall = (
        np.sum(abnormal_true & abnormal_pred)
        /
        max(np.sum(abnormal_true), 1)
    )

    threshold_results.append({

        "threshold": threshold,

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "macro_f1": macro_f1,

        "normal_recall": normal_recall,

        "abnormal_recall": abnormal_recall
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n" + "=" * 70)
print("STAGE 1 THRESHOLD SWEEP")
print("=" * 70)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ======================================================================
# SELECT THRESHOLD
# ======================================================================

best_balanced_row = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]

best_macro_f1_row = threshold_df.loc[
    threshold_df["macro_f1"].idxmax()
]

best_abnormal_recall_row = threshold_df.loc[
    threshold_df["abnormal_recall"].idxmax()
]


print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)


print(
    f"\nBest Balanced Accuracy threshold : "
    f"{best_balanced_row['threshold']:.2f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_balanced_row['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1                         : "
    f"{best_balanced_row['macro_f1']:.4f}"
)

print(
    f"Normal Recall                    : "
    f"{best_balanced_row['normal_recall']:.4f}"
)

print(
    f"Abnormal Recall                  : "
    f"{best_balanced_row['abnormal_recall']:.4f}"
)


print(
    f"\nBest Macro F1 threshold          : "
    f"{best_macro_f1_row['threshold']:.2f}"
)

print(
    f"Macro F1                         : "
    f"{best_macro_f1_row['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_macro_f1_row['balanced_accuracy']:.4f}"
)


print(
    f"\nBest Abnormal Recall threshold   : "
    f"{best_abnormal_recall_row['threshold']:.2f}"
)

print(
    f"Abnormal Recall                  : "
    f"{best_abnormal_recall_row['abnormal_recall']:.4f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_abnormal_recall_row['balanced_accuracy']:.4f}"
)


# ======================================================================
# THRESHOLD SELECTION
# ======================================================================
#
# Primary objective:
# Balanced accuracy.
#
# This keeps the model from simply optimizing ordinary accuracy,
# which would heavily favor Normal beats.
# ======================================================================

best_threshold = float(
    best_balanced_row["threshold"]
)

print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}"
)


# ======================================================================
# VALIDATION PREDICTION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


validation_predictions, validation_abnormal_probabilities = (
    two_stage_predict(
        X_val_scaled,
        stage1_model,
        stage2_model,
        stage2_reverse_mapping,
        threshold=best_threshold
    )
)


# ======================================================================
# VALIDATION REPORT
# ======================================================================

print("\n" + "=" * 70)
print("TWO-STAGE XGBOOST - PATIENT-INDEPENDENT VALIDATION")
print("=" * 70)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        validation_predictions,
        labels=LABELS,
        zero_division=0
    )
)


validation_accuracy = accuracy_score(
    y_val,
    validation_predictions
)

validation_balanced_accuracy = balanced_accuracy_score(
    y_val,
    validation_predictions
)

validation_macro_precision = precision_score(
    y_val,
    validation_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

validation_macro_recall = recall_score(
    y_val,
    validation_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

validation_macro_f1 = f1_score(
    y_val,
    validation_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)


print(
    f"Accuracy          : "
    f"{validation_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{validation_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : "
    f"{validation_macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{validation_macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{validation_macro_f1:.4f}"
)


validation_cm = confusion_matrix(
    y_val,
    validation_predictions,
    labels=LABELS
)


print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        validation_cm,
        index=LABELS,
        columns=LABELS
    )
)


# ======================================================================
# FINAL DS2 TEST
# ======================================================================

print("\n" + "=" * 70)
print("FINAL PATIENT-INDEPENDENT DS2 TEST")
print("=" * 70)


print(
    f"\nUsing selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    "\nIMPORTANT:"
)

print(
    "DS2 was not used during model training, "
    "validation, or threshold selection."
)


# ======================================================================
# FINAL TEST PREDICTION
# ======================================================================

test_predictions, test_abnormal_probabilities = (
    two_stage_predict(
        X_test_scaled,
        stage1_model,
        stage2_model,
        stage2_reverse_mapping,
        threshold=best_threshold
    )
)


# ======================================================================
# FINAL TEST REPORT
# ======================================================================

print("\n" + "=" * 70)
print("TWO-STAGE XGBOOST - FINAL DS2 TEST")
print("=" * 70)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=LABELS,
        zero_division=0
    )
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
)

test_macro_precision = precision_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)


print(
    f"Accuracy          : "
    f"{test_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{test_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : "
    f"{test_macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{test_macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{test_macro_f1:.4f}"
)


# ======================================================================
# TEST CONFUSION MATRIX
# ======================================================================

test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=LABELS
)


print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        test_cm,
        index=LABELS,
        columns=LABELS
    )
)


# ======================================================================
# PER-CLASS METRICS
# ======================================================================

print("\nPer-class metrics:")


test_report_dict = classification_report(
    y_test,
    test_predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0
)


per_class_rows = []


for label in LABELS:

    true_positive = test_cm[
        LABELS.index(label),
        LABELS.index(label)
    ]

    false_negative = (
        test_cm[
            LABELS.index(label),
            :
        ].sum()
        - true_positive
    )

    false_positive = (
        test_cm[
            :,
            LABELS.index(label)
        ].sum()
        - true_positive
    )

    true_negative = (
        test_cm.sum()
        - true_positive
        - false_positive
        - false_negative
    )

    specificity = (
        true_negative /
        max(true_negative + false_positive, 1)
    )

    per_class_rows.append({

        "class": label,

        "precision": test_report_dict[label]["precision"],

        "sensitivity_recall":
            test_report_dict[label]["recall"],

        "f1":
            test_report_dict[label]["f1-score"],

        "support":
            test_report_dict[label]["support"],

        "specificity":
            specificity
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


print(
    per_class_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ======================================================================
# MACRO SPECIFICITY
# ======================================================================

macro_specificity = (
    per_class_df["specificity"].mean()
)


print(
    f"\nMacro Specificity : "
    f"{macro_specificity:.4f}"
)


# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================
#
# We use Stage 2 importance because it is responsible for distinguishing
# the abnormal arrhythmia classes.
# ======================================================================

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)


importance_values = (
    stage2_model.feature_importances_
)


feature_importance_df = pd.DataFrame({

    "feature": feature_columns,

    "importance": importance_values
})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    feature_importance_df.head(20).to_string(
        index=False
    )
)


# ======================================================================
# FINAL RESULTS TABLE
# ======================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


validation_macro_specificity = np.nan


validation_cm_for_specificity = validation_cm


validation_specificities = []


for i in range(len(LABELS)):

    tp = validation_cm_for_specificity[i, i]

    fn = (
        validation_cm_for_specificity[i, :].sum()
        - tp
    )

    fp = (
        validation_cm_for_specificity[:, i].sum()
        - tp
    )

    tn = (
        validation_cm_for_specificity.sum()
        - tp
        - fn
        - fp
    )

    specificity = (
        tn /
        max(tn + fp, 1)
    )

    validation_specificities.append(
        specificity
    )


validation_macro_specificity = np.mean(
    validation_specificities
)


results_df = pd.DataFrame([

    {

        "Model":
            "TWO-STAGE XGBOOST - "
            "PATIENT-INDEPENDENT VALIDATION",

        "Accuracy":
            validation_accuracy,

        "Balanced Accuracy":
            validation_balanced_accuracy,

        "Macro Precision":
            validation_macro_precision,

        "Macro Recall":
            validation_macro_recall,

        "Macro F1":
            validation_macro_f1,

        "Macro Specificity":
            validation_macro_specificity
    },

    {

        "Model":
            "TWO-STAGE XGBOOST - "
            "FINAL DS2 TEST",

        "Accuracy":
            test_accuracy,

        "Balanced Accuracy":
            test_balanced_accuracy,

        "Macro Precision":
            test_macro_precision,

        "Macro Recall":
            test_macro_recall,

        "Macro F1":
            test_macro_f1,

        "Macro Specificity":
            macro_specificity
    }

])


print(
    results_df.to_string(
        index=False
    )
)


# ======================================================================
# CREATE MODEL DIRECTORY
# ======================================================================

model_dir = os.path.join(
    project_dir,
    MODEL_DIR_NAME
)

os.makedirs(
    model_dir,
    exist_ok=True
)


# ======================================================================
# SAVE MODELS
# ======================================================================

joblib.dump(
    stage1_model,
    os.path.join(
        model_dir,
        "stage1_xgboost.joblib"
    )
)


joblib.dump(
    stage2_model,
    os.path.join(
        model_dir,
        "stage2_xgboost.joblib"
    )
)


joblib.dump(
    stage2_mapping,
    os.path.join(
        model_dir,
        "stage2_label_mapping.joblib"
    )
)


# ======================================================================
# SAVE PREPROCESSING
# ======================================================================

joblib.dump(
    imputer,
    os.path.join(
        model_dir,
        "imputer.joblib"
    )
)


joblib.dump(
    scaler,
    os.path.join(
        model_dir,
        "scaler.joblib"
    )
)


joblib.dump(
    feature_columns,
    os.path.join(
        model_dir,
        "feature_columns.joblib"
    )
)


joblib.dump(
    LABELS,
    os.path.join(
        model_dir,
        "labels.joblib"
    )
)


joblib.dump(
    best_threshold,
    os.path.join(
        model_dir,
        "best_threshold.joblib"
    )
)


# ======================================================================
# SAVE THRESHOLD SWEEP
# ======================================================================

threshold_df.to_csv(
    os.path.join(
        model_dir,
        "threshold_sweep.csv"
    ),
    index=False
)


# ======================================================================
# SAVE VALIDATION PREDICTIONS
# ======================================================================

validation_predictions_df = pd.DataFrame({

    "patient_id":
        patients_val.values,

    "true_label":
        y_val.values,

    "predicted_label":
        validation_predictions,

    "abnormal_probability":
        validation_abnormal_probabilities
})


validation_predictions_df.to_csv(
    os.path.join(
        model_dir,
        "two_stage_validation_predictions.csv"
    ),
    index=False
)


# ======================================================================
# SAVE TEST PREDICTIONS
# ======================================================================

test_predictions_df = pd.DataFrame({

    "patient_id":
        patients_test.values,

    "true_label":
        y_test.values,

    "predicted_label":
        test_predictions,

    "abnormal_probability":
        test_abnormal_probabilities
})


test_predictions_df.to_csv(
    os.path.join(
        model_dir,
        "two_stage_test_predictions.csv"
    ),
    index=False
)


# ======================================================================
# SAVE TEST CONFUSION MATRIX
# ======================================================================

np.save(
    os.path.join(
        model_dir,
        "two_stage_test_confusion_matrix.npy"
    ),
    test_cm
)


# ======================================================================
# SAVE FEATURE IMPORTANCE
# ======================================================================

feature_importance_df.to_csv(
    os.path.join(
        model_dir,
        "feature_importance.csv"
    ),
    index=False
)


# ======================================================================
# SAVE FINAL RESULTS
# ======================================================================

results_df.to_csv(
    os.path.join(
        model_dir,
        "two_stage_final_results.csv"
    ),
    index=False
)


# ======================================================================
# SAVE PER-CLASS METRICS
# ======================================================================

per_class_df.to_csv(
    os.path.join(
        model_dir,
        "per_class_metrics.csv"
    ),
    index=False
)


# ======================================================================
# SAVE CONFIGURATION
# ======================================================================

configuration = {

    "dataset_path":
        dataset_path,

    "dataset_modified":
        False,

    "dataset_rebuilt":
        False,

    "number_of_features":
        len(feature_columns),

    "features_generated":
        False,

    "patient_column":
        patient_column,

    "labels":
        LABELS,

    "random_state":
        RANDOM_STATE,

    "stage1_estimators":
        STAGE1_ESTIMATORS,

    "stage2_estimators":
        STAGE2_ESTIMATORS,

    "learning_rate":
        LEARNING_RATE,

    "max_depth":
        MAX_DEPTH,

    "min_child_weight":
        MIN_CHILD_WEIGHT,

    "subsample":
        SUBSAMPLE,

    "colsample_bytree":
        COLSAMPLE_BYTREE,

    "selected_threshold":
        best_threshold
}


joblib.dump(
    configuration,
    os.path.join(
        model_dir,
        "training_configuration.joblib"
    )
)


# ======================================================================
# FINAL MESSAGE
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print("\nModels and results saved in:")
print(model_dir)


print("\nImportant files:")

important_files = [

    "stage1_xgboost.joblib",

    "stage2_xgboost.joblib",

    "stage2_label_mapping.joblib",

    "imputer.joblib",

    "scaler.joblib",

    "feature_columns.joblib",

    "labels.joblib",

    "best_threshold.joblib",

    "threshold_sweep.csv",

    "two_stage_validation_predictions.csv",

    "two_stage_test_predictions.csv",

    "two_stage_test_confusion_matrix.npy",

    "feature_importance.csv",

    "per_class_metrics.csv",

    "two_stage_final_results.csv",

    "training_configuration.joblib"
]


for filename in important_files:
    print(f"- {filename}")


print("\n✓ Dataset was NOT modified.")
print("✓ features.csv was NOT rebuilt.")
print("✓ No new features were generated.")
print("✓ Patient-independent evaluation preserved.")
print("✓ Stage 2 prediction indexing bug fixed.")
print("✓ DS2 remains completely untouched until final evaluation.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)