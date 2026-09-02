# ================================================================
# ECG ARRHYTHMIA CLASSIFICATION
# XGBOOST HYPERPARAMETER OPTIMIZATION
# PATIENT-INDEPENDENT EVALUATION
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ================================================================
# CONFIGURATION
# ================================================================

print("=" * 70)
print("ECG ARRHYTHMIA CLASSIFICATION")
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
print("PATIENT-INDEPENDENT EVALUATION")
print("=" * 70)


# ------------------------------------------------
# Automatically find features.csv
# ------------------------------------------------

SEARCH_PATHS = [
    r"C:\Users\Lenovo\Downloads\ECG\_Arrhythmia\_Detection\ECG\_Arrhythmia\_Detection\data\features.csv",
    r"C:\Users\Lenovo\Downloads\ECG\_Arrhythmia\_Detection\data\features.csv",
    os.path.join("data", "features.csv"),
    os.path.join("..", "data", "features.csv"),
]

DATA_PATH = None

print("\nSearching for features.csv...")

for path in SEARCH_PATHS:

    if os.path.exists(path):
        DATA_PATH = os.path.abspath(path)
        break


if DATA_PATH is None:

    print("\nERROR: features.csv not found.")

    # Recursive search from current directory
    for root, dirs, files in os.walk("."):

        if "features.csv" in files:

            DATA_PATH = os.path.abspath(
                os.path.join(root, "features.csv")
            )

            break


if DATA_PATH is None:

    raise FileNotFoundError(
        "\nCould not locate features.csv.\n"
        "Please make sure the dataset exists."
    )


print("\nDataset found:")
print(DATA_PATH)


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Total beats :", len(df))
print("Total columns :", len(df.columns))


# ================================================================
# BASIC INFORMATION
# ================================================================

PATIENT_COL = "patient_id"
LABEL_COL = "label"

FEATURES = [
    c for c in df.columns
    if c not in [
        PATIENT_COL,
        LABEL_COL,
        "database"
    ]
]

print("\nTotal features:", len(FEATURES))

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())


# ================================================================
# PATIENT-LEVEL SPLIT
# ================================================================

print("\n" + "=" * 70)
print("PATIENT-INDEPENDENT SPLIT")
print("=" * 70)


patients = df[PATIENT_COL].unique()

print("\nTotal patients:", len(patients))


# ------------------------------------------------
# IMPORTANT:
# Use fixed random seed so all experiments
# use the SAME patients.
# ------------------------------------------------

rng = np.random.RandomState(42)

patients_shuffled = patients.copy()
rng.shuffle(patients_shuffled)


# 50% development / 50% final test
n_dev = len(patients_shuffled) // 2

development_patients = patients_shuffled[:n_dev]
test_patients = patients_shuffled[n_dev:]


development_df = df[
    df[PATIENT_COL].isin(development_patients)
].copy()

test_df = df[
    df[PATIENT_COL].isin(test_patients)
].copy()


print("\nDevelopment patients:", len(development_patients))
print("Test patients:", len(test_patients))

print("\nDevelopment beats:", len(development_df))
print("Test beats:", len(test_df))


# ================================================================
# DEVELOPMENT TRAIN / VALIDATION SPLIT
# ================================================================

print("\n" + "=" * 70)
print("DEVELOPMENT TRAIN / VALIDATION SPLIT")
print("=" * 70)


rng = np.random.RandomState(123)

dev_patients = development_patients.copy()
rng.shuffle(dev_patients)

n_train = int(len(dev_patients) * 0.80)

train_patients = dev_patients[:n_train]
validation_patients = dev_patients[n_train:]


train_df = development_df[
    development_df[PATIENT_COL].isin(train_patients)
].copy()

val_df = development_df[
    development_df[PATIENT_COL].isin(validation_patients)
].copy()


print("\nTraining patients:", len(train_patients))
print("Validation patients:", len(validation_patients))

print("\nTraining beats:", len(train_df))
print("Validation beats:", len(val_df))

print("\nTraining distribution:")
print(train_df[LABEL_COL].value_counts())

print("\nValidation distribution:")
print(val_df[LABEL_COL].value_counts())


# ================================================================
# VERIFY NO PATIENT OVERLAP
# ================================================================

assert len(
    set(train_patients) &
    set(validation_patients)
) == 0

assert len(
    set(train_patients) &
    set(test_patients)
) == 0

assert len(
    set(validation_patients) &
    set(test_patients)
) == 0

print("\n✓ No patient overlap.")


# ================================================================
# PREPROCESSING
# ================================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)


X_train = train_df[FEATURES]
X_val = val_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df[LABEL_COL]
y_val = val_df[LABEL_COL]
y_test = test_df[LABEL_COL]


# ------------------------------------------------
# Encode labels
# ------------------------------------------------

label_names = sorted(y_train.unique())

label_to_int = {
    label: i
    for i, label in enumerate(label_names)
}

int_to_label = {
    i: label
    for label, i in label_to_int.items()
}


y_train_encoded = y_train.map(label_to_int)
y_val_encoded = y_val.map(label_to_int)
y_test_encoded = y_test.map(label_to_int)


# ------------------------------------------------
# Imputation
# ------------------------------------------------

print("\nFitting median imputer...")

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)


# ------------------------------------------------
# Scaling
# ------------------------------------------------

print("Fitting StandardScaler...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ================================================================
# CLASS WEIGHTS
# ================================================================

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)


class_counts = y_train_encoded.value_counts()

total = len(y_train_encoded)
n_classes = len(class_counts)

class_weights = {
    cls: total / (n_classes * count)
    for cls, count in class_counts.items()
}


print("\nClass weights:")

for cls, weight in class_weights.items():

    print(
        f"{int_to_label[cls]:>5} : {weight:.4f}"
    )


sample_weights = np.array([
    class_weights[int(label)]
    for label in y_train_encoded
])


# ================================================================
# HYPERPARAMETER SEARCH
# ================================================================

print("\n" + "=" * 70)
print("XGBOOST HYPERPARAMETER SEARCH")
print("=" * 70)


# ------------------------------------------------
# Parameter combinations
# ------------------------------------------------

PARAMETER_GRID = [

    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0
    },

    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0
    },

    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0
    },

    {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0
    },

    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "gamma": 0
    },

    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0
    },

    {
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.1
    },

    {
        "max_depth": 7,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "gamma": 0.1
    },

    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1
    },

    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1
    }

]


# ------------------------------------------------
# Search
# ------------------------------------------------

results = []

best_model = None
best_score = -np.inf
best_params = None


for i, params in enumerate(PARAMETER_GRID):

    print("\n" + "-" * 70)
    print(f"EXPERIMENT {i + 1}/{len(PARAMETER_GRID)}")
    print("-" * 70)

    print(params)


    model = XGBClassifier(

        objective="multi:softprob",

        num_class=len(label_names),

        eval_metric="mlogloss",

        tree_method="hist",

        random_state=42,

        n_jobs=-1,

        **params
    )


    print("\nTraining...")

    model.fit(
        X_train,
        y_train_encoded,
        sample_weight=sample_weights
    )


    # ------------------------------------------------
    # Validation prediction
    # ------------------------------------------------

    y_pred = model.predict(X_val)


    accuracy = accuracy_score(
        y_val_encoded,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val_encoded,
        y_pred
    )

    macro_precision = precision_score(
        y_val_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )


    print("\nValidation results:")

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


    results.append({

        **params,

        "accuracy": accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1

    })


    # ------------------------------------------------
    # Select best based on Macro F1
    # ------------------------------------------------

    if macro_f1 > best_score:

        best_score = macro_f1

        best_model = model

        best_params = params.copy()


# ================================================================
# RESULTS
# ================================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "macro_f1",
    ascending=False
)


print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# BEST MODEL
# ================================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)


print("\nBest parameters:")

for key, value in best_params.items():

    print(
        f"{key:20s}: {value}"
    )


print(
    f"\nBest validation Macro F1: "
    f"{best_score:.4f}"
)


# ================================================================
# BEST MODEL VALIDATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("BEST MODEL VALIDATION REPORT")
print("=" * 70)


best_val_pred = best_model.predict(X_val)


print(
    classification_report(
        y_val_encoded,
        best_val_pred,
        target_names=[
            int_to_label[i]
            for i in range(len(label_names))
        ],
        zero_division=0
    )
)


# ================================================================
# FINAL DS2 TEST
# ================================================================

print("\n" + "=" * 70)
print("FINAL PATIENT-INDEPENDENT DS2 TEST")
print("=" * 70)


print(
    "\nIMPORTANT:"
    "\nDS2 was NOT used during hyperparameter optimization."
)


test_pred = best_model.predict(X_test)


test_accuracy = accuracy_score(
    y_test_encoded,
    test_pred
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test_encoded,
    test_pred
)

test_macro_precision = precision_score(
    y_test_encoded,
    test_pred,
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    y_test_encoded,
    test_pred,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test_encoded,
    test_pred,
    average="macro",
    zero_division=0
)


print("\nClassification Report:")

print(
    classification_report(
        y_test_encoded,
        test_pred,
        target_names=[
            int_to_label[i]
            for i in range(len(label_names))
        ],
        zero_division=0
    )
)


print(
    f"\nAccuracy          : {test_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {test_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : {test_macro_precision:.4f}"
)

print(
    f"Macro Recall      : {test_macro_recall:.4f}"
)

print(
    f"Macro F1          : {test_macro_f1:.4f}"
)


# ================================================================
# SAVE RESULTS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        DATA_PATH
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results_path = os.path.join(
    RESULTS_DIR,
    "xgboost_hyperparameter_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------
# Save final summary
# ------------------------------------------------

summary = pd.DataFrame([{

    "model":
        "XGBoost Hyperparameter Optimized",

    "accuracy":
        test_accuracy,

    "balanced_accuracy":
        test_balanced_accuracy,

    "macro_precision":
        test_macro_precision,

    "macro_recall":
        test_macro_recall,

    "macro_f1":
        test_macro_f1

}])


summary_path = os.path.join(
    RESULTS_DIR,
    "xgboost_optimized_final.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        best_model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


importance_path = os.path.join(
    RESULTS_DIR,
    "xgboost_optimized_feature_importance.csv"
)


importance.to_csv(
    importance_path,
    index=False
)


print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print("\nResults saved to:")

print(results_path)

print(summary_path)

print(importance_path)

print("\n✓ Patient-independent splitting preserved.")

print("✓ Validation used for hyperparameter selection.")

print("✓ DS2 kept untouched until final evaluation.")

print("✓ Same 54 features used.")

print("\nDONE")