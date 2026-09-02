# ============================================================================
# ECG ARRHYTHMIA CLASSIFICATION
# FEATURE ABLATION STUDY
# PATIENT-INDEPENDENT EVALUATION
#
# Experiments:
# 1. Time-domain features
# 2. RR / Heart-rate features
# 3. Spectral features
# 4. Wavelet features
# 5. All features
#
# DS2 is kept completely untouched until final evaluation.
# ============================================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

print("=" * 75)
print("ECG ARRHYTHMIA CLASSIFICATION")
print("FEATURE ABLATION STUDY")
print("PATIENT-INDEPENDENT EVALUATION")
print("=" * 75)


# ============================================================================
# FIND PROJECT ROOT
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# script is inside:
# ECG_Arrhythmia_Detection/experiments/feature_ablation/

PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "features.csv"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "results",
    "feature_ablation"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================================
# FIND DATASET IF EXPECTED PATH DOES NOT EXIST
# ============================================================================

print("\nSearching for features.csv...")

if not os.path.exists(DATA_PATH):

    search_paths = [
        os.path.join(PROJECT_ROOT, "data", "features.csv"),
        os.path.join(
            os.path.dirname(PROJECT_ROOT),
            "data",
            "features.csv"
        )
    ]

    candidates = []

    for path in search_paths:
        if os.path.exists(path):
            candidates.append(path)

    if len(candidates) > 0:
        DATA_PATH = candidates[0]

    else:
        recursive = glob.glob(
            os.path.join(
                os.path.dirname(PROJECT_ROOT),
                "**",
                "features.csv"
            ),
            recursive=True
        )

        if len(recursive) > 0:
            DATA_PATH = recursive[0]

        else:
            raise FileNotFoundError(
                "features.csv could not be found."
            )


print("\nDataset found:")
print(DATA_PATH)


# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Total beats :", len(df))
print("Total columns :", len(df.columns))


# ============================================================================
# BASIC INFORMATION
# ============================================================================

PATIENT_COL = "patient_id"
LABEL_COL = "label"
DATABASE_COL = "database"

print("\nLabels:")
print(df[LABEL_COL].value_counts())


# ============================================================================
# ALL FEATURE COLUMNS
# ============================================================================

META_COLUMNS = [
    PATIENT_COL,
    DATABASE_COL,
    LABEL_COL
]

ALL_FEATURES = [
    c for c in df.columns
    if c not in META_COLUMNS
]

print("\nTotal features:", len(ALL_FEATURES))


# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================

# --------------------------------------------------------------------------
# TIME DOMAIN
# --------------------------------------------------------------------------

TIME_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "rms",
    "ptp",
    "skewness",
    "kurtosis",
    "r_peak_amplitude",
    "r_peak_position",
    "r_peak_to_mean",
    "max_slope",
    "min_slope",
    "mean_abs_slope",
    "slope_std",
    "zero_crossing_rate",
    "absolute_area",
    "signal_energy",
    "qrs_width"
]


# --------------------------------------------------------------------------
# RR / HEART RATE
# --------------------------------------------------------------------------

RR_FEATURES = [
    "rr_interval",
    "prev_rr",
    "next_rr",
    "heart_rate",
    "mean_rr",
    "local_heart_rate",
    "rr_ratio",
    "next_rr_ratio",
    "rr_deviation",
    "rr_deviation_ratio"
]


# --------------------------------------------------------------------------
# SPECTRAL
# --------------------------------------------------------------------------

SPECTRAL_FEATURES = [
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_energy"
]


# --------------------------------------------------------------------------
# WAVELET
# --------------------------------------------------------------------------

WAVELET_FEATURES = [
    "wav_0_mean",
    "wav_0_std",
    "wav_0_energy",
    "wav_0_energy_ratio",

    "wav_1_mean",
    "wav_1_std",
    "wav_1_energy",
    "wav_1_energy_ratio",

    "wav_2_mean",
    "wav_2_std",
    "wav_2_energy",
    "wav_2_energy_ratio",

    "wav_3_mean",
    "wav_3_std",
    "wav_3_energy",
    "wav_3_energy_ratio",

    "wav_4_mean",
    "wav_4_std",
    "wav_4_energy",
    "wav_4_energy_ratio",

    "wavelet_entropy"
]


# ============================================================================
# CHECK FEATURE GROUPS
# ============================================================================

FEATURE_GROUPS = {
    "Time-domain": TIME_FEATURES,
    "RR-heart-rate": RR_FEATURES,
    "Spectral": SPECTRAL_FEATURES,
    "Wavelet": WAVELET_FEATURES,
    "All-features": ALL_FEATURES
}


print("\n" + "=" * 75)
print("FEATURE GROUPS")
print("=" * 75)

for name, features in FEATURE_GROUPS.items():

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if len(missing) > 0:

        print("\nWARNING:", name)
        print("Missing:", missing)

        FEATURE_GROUPS[name] = [
            f for f in features
            if f in df.columns
        ]

    print(
        f"{name:20s}: "
        f"{len(FEATURE_GROUPS[name])} features"
    )


# ============================================================================
# PATIENT-INDEPENDENT DEVELOPMENT / TEST SPLIT
# ============================================================================
#
# We have:
# MITDB = 63 patients
# SVDB  = 63 patients
#
# Since the database column contains MITDB/SVDB but not DS1/DS2 labels,
# we use a patient-level 50/50 development/test split.
#
# IMPORTANT:
# The same patient split is generated deterministically.
# ============================================================================

print("\n" + "=" * 75)
print("PATIENT-INDEPENDENT DEVELOPMENT / TEST SPLIT")
print("=" * 75)


patients = sorted(df[PATIENT_COL].unique())

print("\nTotal patients:", len(patients))


# deterministic split
rng = np.random.RandomState(42)

shuffled_patients = rng.permutation(patients)

n_test = len(shuffled_patients) // 2

test_patients = set(
    shuffled_patients[:n_test]
)

development_patients = set(
    shuffled_patients[n_test:]
)


dev_df = df[
    df[PATIENT_COL].isin(development_patients)
].copy()

test_df = df[
    df[PATIENT_COL].isin(test_patients)
].copy()


print("\nDevelopment patients:", len(development_patients))
print("Test patients:", len(test_patients))

print("\nDevelopment beats:", len(dev_df))
print("Test beats:", len(test_df))

print("\nDevelopment distribution:")
print(dev_df[LABEL_COL].value_counts())

print("\nTest distribution:")
print(test_df[LABEL_COL].value_counts())


overlap = (
    set(dev_df[PATIENT_COL])
    &
    set(test_df[PATIENT_COL])
)

if len(overlap) > 0:
    raise RuntimeError(
        "PATIENT LEAKAGE DETECTED!"
    )

print("\n✓ No patient overlap.")


# ============================================================================
# DEVELOPMENT TRAIN / VALIDATION SPLIT
# ============================================================================

print("\n" + "=" * 75)
print("DEVELOPMENT TRAIN / VALIDATION SPLIT")
print("=" * 75)


dev_patients = sorted(
    dev_df[PATIENT_COL].unique()
)

rng = np.random.RandomState(123)

dev_patients = rng.permutation(dev_patients)

n_validation = max(
    1,
    int(round(len(dev_patients) * 0.20))
)

validation_patients = set(
    dev_patients[:n_validation]
)

training_patients = set(
    dev_patients[n_validation:]
)


train_df = dev_df[
    dev_df[PATIENT_COL].isin(training_patients)
].copy()

val_df = dev_df[
    dev_df[PATIENT_COL].isin(validation_patients)
].copy()


print("\nTraining patients:", len(training_patients))
print("Validation patients:", len(validation_patients))

print("\nTraining beats:", len(train_df))
print("Validation beats:", len(val_df))

print("\nTraining distribution:")
print(train_df[LABEL_COL].value_counts())

print("\nValidation distribution:")
print(val_df[LABEL_COL].value_counts())


overlap = (
    set(train_df[PATIENT_COL])
    &
    set(val_df[PATIENT_COL])
)

if len(overlap) > 0:
    raise RuntimeError(
        "PATIENT LEAKAGE BETWEEN TRAINING AND VALIDATION!"
    )

print("\n✓ No patient overlap.")


# ============================================================================
# LABEL ENCODING
# ============================================================================

LABELS = [
    "N",
    "SVEB",
    "VEB",
    "F",
    "Q"
]

label_to_int = {
    label: i
    for i, label in enumerate(LABELS)
}

int_to_label = {
    i: label
    for label, i in label_to_int.items()
}


y_train = train_df[LABEL_COL].map(
    label_to_int
).values

y_val = val_df[LABEL_COL].map(
    label_to_int
).values

y_test = test_df[LABEL_COL].map(
    label_to_int
).values


# ============================================================================
# CLASS WEIGHTS
# ============================================================================
#
# Balanced weights:
#
# weight = total_samples / (number_of_classes * class_count)
#
# This prevents N from dominating the model.
# ============================================================================

class_counts = np.bincount(
    y_train,
    minlength=len(LABELS)
)

total_samples = len(y_train)
n_classes = len(LABELS)

class_weights = {}

for i, count in enumerate(class_counts):

    if count > 0:

        class_weights[i] = (
            total_samples /
            (n_classes * count)
        )


print("\n" + "=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

for i, label in enumerate(LABELS):

    print(
        f"{label:5s}: "
        f"{class_weights[i]:.4f}"
    )


sample_weights = np.array([
    class_weights[int(y)]
    for y in y_train
])


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
#
# These are the BEST hyperparameters from our previous experiment.
#
# max_depth = 7
# learning_rate = 0.03
# n_estimators = 600
# subsample = 0.8
# colsample_bytree = 0.8
# min_child_weight = 2
# gamma = 0.1
#
# IMPORTANT:
# We keep these fixed for every feature experiment.
# ============================================================================

MODEL_PARAMS = {
    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "gamma": 0.1,

    "objective": "multi:softprob",
    "num_class": 5,

    "eval_metric": "mlogloss",

    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
}


# ============================================================================
# FUNCTION: TRAIN + VALIDATE
# ============================================================================

def run_experiment(
    experiment_name,
    feature_list
):

    print("\n")
    print("=" * 75)
    print(
        "FEATURE EXPERIMENT:",
        experiment_name
    )
    print("=" * 75)

    print(
        "\nNumber of features:",
        len(feature_list)
    )

    print("\nFeatures:")

    for feature in feature_list:
        print("  ", feature)


    # ------------------------------------------------------------------------
    # EXTRACT FEATURES
    # ------------------------------------------------------------------------

    X_train = train_df[
        feature_list
    ].copy()

    X_val = val_df[
        feature_list
    ].copy()

    X_test = test_df[
        feature_list
    ].copy()


    # ------------------------------------------------------------------------
    # IMPUTATION
    # ------------------------------------------------------------------------

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

    X_test = imputer.transform(
        X_test
    )


    # ------------------------------------------------------------------------
    # SCALING
    # ------------------------------------------------------------------------

    print("Fitting StandardScaler...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )


    # ------------------------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------------------------

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        verbose=False
    )


    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    print("\nValidation evaluation...")

    val_pred = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_balanced_accuracy = balanced_accuracy_score(
        y_val,
        val_pred
    )

    val_macro_precision = precision_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    val_macro_recall = recall_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    print("\nValidation results:")

    print(
        f"Accuracy          : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{val_balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{val_macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{val_macro_recall:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{val_macro_f1:.4f}"
    )


    # ------------------------------------------------------------------------
    # VALIDATION REPORT
    # ------------------------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            val_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0
        )
    )


    # ------------------------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------------------------

    result = {

        "experiment": experiment_name,

        "num_features": len(feature_list),

        "accuracy": val_accuracy,

        "balanced_accuracy":
            val_balanced_accuracy,

        "macro_precision":
            val_macro_precision,

        "macro_recall":
            val_macro_recall,

        "macro_f1":
            val_macro_f1
    }


    # ------------------------------------------------------------------------
    # RETURN MODEL FOR POSSIBLE FINAL TEST
    # ------------------------------------------------------------------------

    return (
        result,
        model,
        imputer,
        scaler,
        X_test
    )


# ============================================================================
# RUN ALL FEATURE EXPERIMENTS
# ============================================================================

results = []

experiment_objects = {}


for experiment_name, feature_list in FEATURE_GROUPS.items():

    (
        result,
        model,
        imputer,
        scaler,
        X_test
    ) = run_experiment(
        experiment_name,
        feature_list
    )

    results.append(result)

    experiment_objects[
        experiment_name
    ] = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": feature_list,
        "X_test": X_test
    }


# ============================================================================
# RESULTS TABLE
# ============================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
)


print("\n")
print("=" * 75)
print("FEATURE ABLATION RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================================
# SAVE VALIDATION RESULTS
# ============================================================================

validation_results_path = os.path.join(
    RESULT_DIR,
    "feature_ablation_validation_results.csv"
)

results_df.to_csv(
    validation_results_path,
    index=False
)


# ============================================================================
# SELECT BEST FEATURE GROUP
# ============================================================================

best_experiment = results_df.iloc[0]

best_name = best_experiment[
    "experiment"
]

best_f1 = best_experiment[
    "macro_f1"
]


print("\n")
print("=" * 75)
print("BEST FEATURE GROUP")
print("=" * 75)

print(
    "\nBest feature group:",
    best_name
)

print(
    "Validation Macro F1:",
    f"{best_f1:.4f}"
)


# ============================================================================
# FINAL DS2 TEST
# ============================================================================
#
# IMPORTANT:
#
# DS2/test data has NOT been used in:
#
# - feature selection
# - hyperparameter optimization
# - model selection
#
# It is used ONLY here.
# ============================================================================

print("\n")
print("=" * 75)
print("FINAL PATIENT-INDEPENDENT DS2 TEST")
print("=" * 75)

print(
    "\nIMPORTANT:"
)

print(
    "Test data was NOT used during feature selection."
)

print(
    "Test data is now being evaluated once."
)


best_object = experiment_objects[
    best_name
]

best_model = best_object[
    "model"
]

X_test = best_object[
    "X_test"
]


# ------------------------------------------------------------------------
# TEST PREDICTION
# ------------------------------------------------------------------------

test_pred = best_model.predict(
    X_test
)


# ------------------------------------------------------------------------
# TEST METRICS
# ------------------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_pred
)

test_macro_precision = precision_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)


print("\nFinal DS2/Test results:")

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


# ============================================================================
# FINAL CLASSIFICATION REPORT
# ============================================================================

print("\n")
print("=" * 75)
print("FINAL DS2 CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        test_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=list(range(len(LABELS)))
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

print("\nConfusion Matrix:")

print(
    cm_df.to_string()
)


# ============================================================================
# SAVE CONFUSION MATRIX
# ============================================================================

cm_path = os.path.join(
    RESULT_DIR,
    "feature_ablation_best_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================================
# FINAL RESULTS TABLE
# ============================================================================

final_result = pd.DataFrame([
    {

        "best_feature_group":
            best_name,

        "num_features":
            len(
                FEATURE_GROUPS[
                    best_name
                ]
            ),

        "validation_macro_f1":
            best_f1,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "test_macro_precision":
            test_macro_precision,

        "test_macro_recall":
            test_macro_recall,

        "test_macro_f1":
            test_macro_f1
    }
])


final_results_path = os.path.join(
    RESULT_DIR,
    "feature_ablation_final_results.csv"
)

final_result.to_csv(
    final_results_path,
    index=False
)


# ============================================================================
# SAVE FEATURE GROUP INFORMATION
# ============================================================================

feature_group_rows = []

for group_name, features in FEATURE_GROUPS.items():

    for feature in features:

        feature_group_rows.append({

            "feature_group":
                group_name,

            "feature":
                feature
        })


feature_group_df = pd.DataFrame(
    feature_group_rows
)

feature_group_path = os.path.join(
    RESULT_DIR,
    "feature_groups.csv"
)

feature_group_df.to_csv(
    feature_group_path,
    index=False
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n")
print("=" * 75)
print("FEATURE ABLATION EXPERIMENT COMPLETE")
print("=" * 75)

print("\nValidation comparison:")

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest feature group:")
print(best_name)

print(
    "\nFinal DS2/Test Macro F1:",
    f"{test_macro_f1:.4f}"
)

print(
    "\nResults saved to:"
)

print(validation_results_path)
print(final_results_path)
print(cm_path)
print(feature_group_path)

print("\n✓ Patient-independent splitting preserved.")
print("✓ Same optimized XGBoost configuration used.")
print("✓ Only feature groups were changed.")
print("✓ Validation used for feature-group selection.")
print("✓ Test/DS2 remained untouched until final evaluation.")

print("\nDONE")