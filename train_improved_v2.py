# ======================================================================
# IMPROVED ECG ARRHYTHMIA CLASSIFICATION
# TWO-STAGE XGBOOST + CLASS-SPECIFIC THRESHOLDS
# PATIENT-INDEPENDENT EVALUATION
# ======================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

print("=" * 70)
print("IMPROVED ECG ARRHYTHMIA CLASSIFICATION")
print("TWO-STAGE XGBOOST + CLASS-SPECIFIC THRESHOLDS")
print("PATIENT-INDEPENDENT EVALUATION")
print("=" * 70)


# ======================================================================
# AUTOMATIC PATH DETECTION
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

print("\nSearching for features.csv...")

possible_paths = [
    PROJECT_DIR / "data" / "features.csv",
    SCRIPT_DIR.parent / "data" / "features.csv",
    SCRIPT_DIR / "data" / "features.csv",
    Path.cwd() / "data" / "features.csv",
]

DATA_PATH = None

for path in possible_paths:
    if path.exists():
        DATA_PATH = path
        break


# If not found, search nearby directories
if DATA_PATH is None:
    search_roots = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parent.parent,
        Path.cwd()
    ]

    for root in search_roots:
        try:
            matches = list(root.rglob("features.csv"))

            if matches:
                DATA_PATH = matches[0]
                break

        except Exception:
            pass


if DATA_PATH is None:
    print("\nERROR: features.csv could not be found.")
    print("\nYour project should contain a file named:")
    print("features.csv")
    print("\nExample:")
    print(r"C:\Users\Lenovo\Downloads\ECG\_Arrhythmia\_Detection\ECG\_Arrhythmia\_Detection\data\features.csv")

    raise FileNotFoundError("features.csv not found.")


print("\nDataset found:")
print(DATA_PATH)


# ======================================================================
# MODEL DIRECTORY
# ======================================================================

MODEL_DIR = PROJECT_DIR / "models"

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

print("\nModels will be saved to:")
print(MODEL_DIR)


# ======================================================================
# LOAD DATASET
# ======================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total beats: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nDataset columns:")
print(df.columns.tolist())


# ======================================================================
# IDENTIFY IMPORTANT COLUMNS
# ======================================================================

PATIENT_COLUMN = "patient_id"
LABEL_COLUMN = "label"

if PATIENT_COLUMN not in df.columns:
    raise ValueError(f"Missing patient column: {PATIENT_COLUMN}")

if LABEL_COLUMN not in df.columns:
    raise ValueError(f"Missing label column: {LABEL_COLUMN}")

print(f"\nPatient/record column: {PATIENT_COLUMN}")
print(f"Label column: {LABEL_COLUMN}")


# ======================================================================
# LABEL INFORMATION
# ======================================================================

print("\nLabel distribution:")
print(df[LABEL_COLUMN].value_counts())


LABELS = ["N", "SVEB", "VEB", "F", "Q"]

for label in LABELS:
    if label not in df[LABEL_COLUMN].unique():
        print(f"WARNING: {label} is missing from dataset.")


# ======================================================================
# FEATURE SELECTION
# ======================================================================

NON_FEATURE_COLUMNS = [
    PATIENT_COLUMN,
    LABEL_COLUMN
]

if "database" in df.columns:
    NON_FEATURE_COLUMNS.append("database")

FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in NON_FEATURE_COLUMNS
]

print(f"\nTotal features: {len(FEATURE_COLUMNS)}")


# ======================================================================
# CLEAN LABELS
# ======================================================================

df = df.dropna(subset=[LABEL_COLUMN, PATIENT_COLUMN]).copy()

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
df[PATIENT_COLUMN] = df[PATIENT_COLUMN].astype(str).str.strip()


# ======================================================================
# PATIENT-INDEPENDENT DS1 / DS2 SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("PATIENT-INDEPENDENT DS1 / DS2 SPLIT")
print("=" * 70)


# ----------------------------------------------------------------------
# Use database column if available
# ----------------------------------------------------------------------

if "database" in df.columns:

    databases = df["database"].astype(str).str.upper()

    ds1_mask = databases.isin(["DS1", "MITBIH", "MIT-BIH", "TRAIN"])
    ds2_mask = databases.isin(["DS2", "TEST"])

    if ds1_mask.sum() > 0 and ds2_mask.sum() > 0:

        ds1 = df[ds1_mask].copy()
        ds2 = df[ds2_mask].copy()

    else:

        print("\nDatabase column was found, but DS1/DS2 labels were not recognized.")
        print("Using patient-level split instead.")

        ds1 = None
        ds2 = None

else:
    ds1 = None
    ds2 = None


# ----------------------------------------------------------------------
# If database split unavailable, use patient-level 50/50 split
# ----------------------------------------------------------------------

if ds1 is None or ds2 is None:

    patients = df[PATIENT_COLUMN].unique()

    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(patients)

    midpoint = len(patients) // 2

    train_patients = patients[:midpoint]
    test_patients = patients[midpoint:]

    ds1 = df[df[PATIENT_COLUMN].isin(train_patients)].copy()
    ds2 = df[df[PATIENT_COLUMN].isin(test_patients)].copy()


print(f"\nTraining records: {ds1[PATIENT_COLUMN].nunique()}")
print(f"Testing records : {ds2[PATIENT_COLUMN].nunique()}")

print(f"Training beats: {len(ds1)}")
print(f"Testing beats : {len(ds2)}")


overlap = set(ds1[PATIENT_COLUMN]) & set(ds2[PATIENT_COLUMN])

if overlap:
    raise RuntimeError(
        f"Patient leakage detected! {len(overlap)} patients overlap."
    )

print("\n✓ No patient overlap between training and testing.")


print("\nTraining distribution:")
print(ds1[LABEL_COLUMN].value_counts())

print("\nTesting distribution:")
print(ds2[LABEL_COLUMN].value_counts())


# ======================================================================
# PATIENT-INDEPENDENT TRAIN / VALIDATION SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("PATIENT-INDEPENDENT TRAIN / VALIDATION SPLIT")
print("=" * 70)


development_patients = ds1[PATIENT_COLUMN].unique()

rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(development_patients)

n_train_patients = int(len(development_patients) * 0.80)

train_patients = development_patients[:n_train_patients]
val_patients = development_patients[n_train_patients:]


train_df = ds1[
    ds1[PATIENT_COLUMN].isin(train_patients)
].copy()

val_df = ds1[
    ds1[PATIENT_COLUMN].isin(val_patients)
].copy()


print(f"\nPatients available for development: {len(development_patients)}")

print(f"Training patients   : {len(train_patients)}")
print(f"Validation patients : {len(val_patients)}")

print(f"\nTraining beats      : {len(train_df)}")
print(f"Validation beats    : {len(val_df)}")


overlap = set(train_df[PATIENT_COLUMN]) & set(val_df[PATIENT_COLUMN])

if overlap:
    raise RuntimeError(
        f"Patient leakage detected between train and validation: {overlap}"
    )

print("\n✓ No patient overlap between training and validation.")


print("\nTraining distribution:")
print(train_df[LABEL_COLUMN].value_counts())

print("\nValidation distribution:")
print(val_df[LABEL_COLUMN].value_counts())


print("\nTraining class proportions:")
print(train_df[LABEL_COLUMN].value_counts(normalize=True))

print("\nValidation class proportions:")
print(val_df[LABEL_COLUMN].value_counts(normalize=True))


# ======================================================================
# CHECK VALIDATION CLASSES
# ======================================================================

missing_validation_classes = [
    label for label in LABELS
    if label not in val_df[LABEL_COLUMN].unique()
]

if missing_validation_classes:

    print(
        "\nWARNING: Missing validation classes:",
        missing_validation_classes
    )

else:

    print("\n✓ All five classes are present in validation.")


# ======================================================================
# FEATURE MATRICES
# ======================================================================

X_train_raw = train_df[FEATURE_COLUMNS].copy()
X_val_raw = val_df[FEATURE_COLUMNS].copy()
X_test_raw = ds2[FEATURE_COLUMNS].copy()

y_train = train_df[LABEL_COLUMN].values
y_val = val_df[LABEL_COLUMN].values
y_test = ds2[LABEL_COLUMN].values


# ======================================================================
# IMPUTATION
# ======================================================================

print("\nFitting median imputer...")

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train_raw)
X_val = imputer.transform(X_val_raw)
X_test = imputer.transform(X_test_raw)


# ======================================================================
# STANDARDIZATION
# ======================================================================

print("Fitting StandardScaler...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ======================================================================
# STAGE 1
# NORMAL VS ABNORMAL
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 1: NORMAL VS ABNORMAL")
print("=" * 70)


y_train_stage1 = np.where(y_train == "N", 0, 1)
y_val_stage1 = np.where(y_val == "N", 0, 1)


print("\nTraining distribution:")
print(
    pd.Series(
        np.where(
            y_train_stage1 == 0,
            "Normal",
            "Abnormal"
        )
    ).value_counts()
)


# ----------------------------------------------------------------------
# Stage 1 class weights
# ----------------------------------------------------------------------

stage1_classes = np.array([0, 1])

stage1_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=stage1_classes,
    y=y_train_stage1
)

stage1_weights = {
    0: stage1_weights_array[0],
    1: stage1_weights_array[1]
}

print("\nStage 1 weights:")
print(stage1_weights)


sample_weights_stage1 = np.array([
    stage1_weights[int(label)]
    for label in y_train_stage1
])


# ======================================================================
# STAGE 1 XGBOOST
# ======================================================================

print("\nTraining Stage 1 XGBoost...")


stage1_model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


stage1_model.fit(
    X_train,
    y_train_stage1,
    sample_weight=sample_weights_stage1
)


print("Stage 1 training complete.")


# ======================================================================
# STAGE 2
# ABNORMAL CLASS CLASSIFICATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 2: ABNORMAL CLASS CLASSIFICATION")
print("=" * 70)


abnormal_train_mask = y_train != "N"
abnormal_val_mask = y_val != "N"


X_train_stage2 = X_train[abnormal_train_mask]
X_val_stage2 = X_val[abnormal_val_mask]

y_train_stage2_text = y_train[abnormal_train_mask]
y_val_stage2_text = y_val[abnormal_val_mask]


print(f"\nAbnormal training samples: {len(X_train_stage2)}")
print(f"Abnormal validation samples: {len(X_val_stage2)}")


print("\nTraining distribution:")
print(pd.Series(y_train_stage2_text).value_counts())


# ======================================================================
# STAGE 2 LABEL ENCODING
# ======================================================================

STAGE2_LABELS = ["SVEB", "VEB", "F", "Q"]

stage2_label_mapping = {
    "SVEB": 0,
    "VEB": 1,
    "F": 2,
    "Q": 3
}

stage2_inverse_mapping = {
    0: "SVEB",
    1: "VEB",
    2: "F",
    3: "Q"
}


y_train_stage2 = np.array([
    stage2_label_mapping[label]
    for label in y_train_stage2_text
])

y_val_stage2 = np.array([
    stage2_label_mapping[label]
    for label in y_val_stage2_text
])


# ======================================================================
# STAGE 2 CLASS WEIGHTS
# ======================================================================

stage2_classes = np.array([0, 1, 2, 3])

stage2_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=stage2_classes,
    y=y_train_stage2
)

stage2_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        stage2_classes,
        stage2_weights_array
    )
}


print("\nInitial Stage 2 weights:")

print({
    stage2_inverse_mapping[k]: v
    for k, v in stage2_weights.items()
})


# ======================================================================
# EXTRA WEIGHT FOR F
# ======================================================================

# F is extremely rare. Give it additional importance.

stage2_weights[2] *= 1.10


# Slightly increase SVEB and Q
stage2_weights[0] *= 1.35
stage2_weights[3] *= 1.10


print("\nAdjusted Stage 2 weights:")

print({
    stage2_inverse_mapping[k]: v
    for k, v in stage2_weights.items()
})


sample_weights_stage2 = np.array([
    stage2_weights[int(label)]
    for label in y_train_stage2
])


# ======================================================================
# STAGE 2 XGBOOST
# ======================================================================

print("\nTraining Stage 2 XGBoost...")


stage2_model = XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


stage2_model.fit(
    X_train_stage2,
    y_train_stage2,
    sample_weight=sample_weights_stage2
)


print("Stage 2 training complete.")


# ======================================================================
# PREDICTION FUNCTION
# ======================================================================

def get_two_stage_probabilities(
    X,
    stage1_model,
    stage2_model
):

    # Stage 1 abnormal probability
    stage1_prob = stage1_model.predict_proba(X)[:, 1]

    abnormal_indices = np.where(
        stage1_prob >= 0
    )[0]

    # Stage 2 probabilities for ALL samples
    stage2_probabilities = stage2_model.predict_proba(X)

    return stage1_prob, stage2_probabilities


# ======================================================================
# TWO-STAGE PREDICTION
# ======================================================================

def two_stage_predict(
    X,
    stage1_model,
    stage2_model,
    threshold=0.5,
    class_thresholds=None
):

    n_samples = X.shape[0]

    stage1_prob = stage1_model.predict_proba(X)[:, 1]

    stage2_prob = stage2_model.predict_proba(X)

    predictions = np.full(
        n_samples,
        "N",
        dtype=object
    )

    for i in range(n_samples):

        # --------------------------------------------------------------
        # Stage 1 decision
        # --------------------------------------------------------------

        if stage1_prob[i] < threshold:
            predictions[i] = "N"
            continue

        # --------------------------------------------------------------
        # Stage 2
        # --------------------------------------------------------------

        probs = stage2_prob[i]

        # --------------------------------------------------------------
        # Class-specific thresholds
        # --------------------------------------------------------------

        if class_thresholds is not None:

            adjusted_probs = probs.copy()

            for class_id in range(len(probs)):

                label = stage2_inverse_mapping[class_id]

                if label in class_thresholds:

                    adjusted_probs[class_id] = (
                        probs[class_id] /
                        class_thresholds[label]
                    )

            predicted_class = np.argmax(adjusted_probs)

        else:

            predicted_class = np.argmax(probs)

        predictions[i] = stage2_inverse_mapping[
            int(predicted_class)
        ]

    return predictions


# ======================================================================
# THRESHOLD SEARCH
# ======================================================================

print("\n" + "=" * 70)
print("SEARCHING STAGE 1 THRESHOLD")
print("=" * 70)


threshold_results = []


for threshold in np.arange(0.10, 0.51, 0.01):

    predictions = two_stage_predict(
        X_val,
        stage1_model,
        stage2_model,
        threshold=threshold
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
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

    normal_mask = y_val == "N"
    abnormal_mask = y_val != "N"

    normal_recall = recall_score(
        y_val[normal_mask],
        predictions[normal_mask],
        labels=["N"],
        average="macro",
        zero_division=0
    )

    abnormal_recall = recall_score(
        abnormal_mask.astype(int),
        (predictions != "N").astype(int),
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "normal_recall": normal_recall,
        "abnormal_recall": abnormal_recall
    })


threshold_df = pd.DataFrame(threshold_results)


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
# SELECT BEST STAGE 1 THRESHOLD
# ======================================================================

best_balanced = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]

best_macro_f1 = threshold_df.loc[
    threshold_df["macro_f1"].idxmax()
]

best_abnormal_recall = threshold_df.loc[
    threshold_df["abnormal_recall"].idxmax()
]


print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)


print(
    f"\nBest Balanced Accuracy threshold : "
    f"{best_balanced['threshold']:.2f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_balanced['balanced_accuracy']:.4f}"
)

print(
    f"Macro F1                         : "
    f"{best_balanced['macro_f1']:.4f}"
)

print(
    f"Normal Recall                    : "
    f"{best_balanced['normal_recall']:.4f}"
)

print(
    f"Abnormal Recall                  : "
    f"{best_balanced['abnormal_recall']:.4f}"
)


print(
    f"\nBest Macro F1 threshold          : "
    f"{best_macro_f1['threshold']:.2f}"
)

print(
    f"Macro F1                         : "
    f"{best_macro_f1['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_macro_f1['balanced_accuracy']:.4f}"
)


print(
    f"\nBest Abnormal Recall threshold   : "
    f"{best_abnormal_recall['threshold']:.2f}"
)

print(
    f"Abnormal Recall                  : "
    f"{best_abnormal_recall['abnormal_recall']:.4f}"
)

print(
    f"Balanced Accuracy                : "
    f"{best_abnormal_recall['balanced_accuracy']:.4f}"
)


STAGE1_THRESHOLD = float(
    best_balanced["threshold"]
)

print(
    f"\nSelected threshold: "
    f"{STAGE1_THRESHOLD:.2f}"
)


# ======================================================================
# CLASS-SPECIFIC THRESHOLD SEARCH
# ======================================================================

print("\n" + "=" * 70)
print("CLASS-SPECIFIC THRESHOLD OPTIMIZATION")
print("=" * 70)


# Start with equal thresholds
class_thresholds = {
    "SVEB": 1.0,
    "VEB": 1.0,
    "F": 1.0,
    "Q": 1.0
}


# ----------------------------------------------------------------------
# Optimize one class at a time.
#
# Lower threshold = easier for that class to be selected.
# Higher threshold = harder for that class to be selected.
# ----------------------------------------------------------------------

threshold_candidates = np.arange(
    0.50,
    1.51,
    0.05
)


for target_class in STAGE2_LABELS:

    best_score = -1
    best_threshold = 1.0

    for candidate in threshold_candidates:

        trial_thresholds = class_thresholds.copy()

        trial_thresholds[target_class] = candidate

        predictions = two_stage_predict(
            X_val,
            stage1_model,
            stage2_model,
            threshold=STAGE1_THRESHOLD,
            class_thresholds=trial_thresholds
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        )

        if macro_f1 > best_score:

            best_score = macro_f1
            best_threshold = candidate

    class_thresholds[target_class] = best_threshold

    print(
        f"{target_class:>5} threshold = "
        f"{best_threshold:.2f} "
        f"(validation macro F1 = {best_score:.4f})"
    )


print("\nFinal class-specific thresholds:")
print(class_thresholds)


# ======================================================================
# VALIDATION EVALUATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


val_predictions = two_stage_predict(
    X_val,
    stage1_model,
    stage2_model,
    threshold=STAGE1_THRESHOLD,
    class_thresholds=class_thresholds
)


# ======================================================================
# VALIDATION METRICS
# ======================================================================

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    val_predictions
)

val_macro_precision = precision_score(
    y_val,
    val_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

val_macro_recall = recall_score(
    y_val,
    val_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 70)
print("TWO-STAGE XGBOOST - PATIENT-INDEPENDENT VALIDATION")
print("=" * 70)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        val_predictions,
        labels=LABELS,
        zero_division=0
    )
)


print(f"Accuracy          : {val_accuracy:.4f}")
print(f"Balanced Accuracy : {val_balanced_accuracy:.4f}")
print(f"Macro Precision   : {val_macro_precision:.4f}")
print(f"Macro Recall      : {val_macro_recall:.4f}")
print(f"Macro F1          : {val_macro_f1:.4f}")


# ======================================================================
# VALIDATION CONFUSION MATRIX
# ======================================================================

val_cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=LABELS
)


print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        val_cm,
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
    f"\nUsing Stage 1 threshold: "
    f"{STAGE1_THRESHOLD:.2f}"
)

print(
    f"Using class thresholds: "
    f"{class_thresholds}"
)

print("\nIMPORTANT:")
print(
    "DS2 was not used during model training, "
    "validation, threshold optimization, "
    "or class-specific threshold selection."
)


# ======================================================================
# TEST PREDICTION
# ======================================================================

test_predictions = two_stage_predict(
    X_test,
    stage1_model,
    stage2_model,
    threshold=STAGE1_THRESHOLD,
    class_thresholds=class_thresholds
)


# ======================================================================
# TEST METRICS
# ======================================================================

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


print(f"Accuracy          : {test_accuracy:.4f}")
print(f"Balanced Accuracy : {test_balanced_accuracy:.4f}")
print(f"Macro Precision   : {test_macro_precision:.4f}")
print(f"Macro Recall      : {test_macro_recall:.4f}")
print(f"Macro F1          : {test_macro_f1:.4f}")


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

per_class_precision = precision_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average=None,
    zero_division=0
)

per_class_recall = recall_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average=None,
    zero_division=0
)

per_class_f1 = f1_score(
    y_test,
    test_predictions,
    labels=LABELS,
    average=None,
    zero_division=0
)


per_class_specificity = []

for i, label in enumerate(LABELS):

    tp = test_cm[i, i]

    fn = test_cm[i, :].sum() - tp

    fp = test_cm[:, i].sum() - tp

    tn = test_cm.sum() - tp - fn - fp

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    per_class_specificity.append(
        specificity
    )


metrics_df = pd.DataFrame({

    "class": LABELS,

    "precision": per_class_precision,

    "sensitivity_recall": per_class_recall,

    "f1": per_class_f1,

    "support": [
        np.sum(y_test == label)
        for label in LABELS
    ],

    "specificity": per_class_specificity

})


print("\nPer-class metrics:")
print(metrics_df.to_string(index=False))


macro_specificity = np.mean(
    per_class_specificity
)

print(
    f"\nMacro Specificity : "
    f"{macro_specificity:.4f}"
)


# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)


feature_importance = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": stage2_model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)


print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ======================================================================
# FINAL RESULTS TABLE
# ======================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


results_df = pd.DataFrame({

    "Model": [
        "TWO-STAGE XGBOOST - PATIENT-INDEPENDENT VALIDATION",
        "TWO-STAGE XGBOOST - FINAL DS2 TEST"
    ],

    "Accuracy": [
        val_accuracy,
        test_accuracy
    ],

    "Balanced Accuracy": [
        val_balanced_accuracy,
        test_balanced_accuracy
    ],

    "Macro Precision": [
        val_macro_precision,
        test_macro_precision
    ],

    "Macro Recall": [
        val_macro_recall,
        test_macro_recall
    ],

    "Macro F1": [
        val_macro_f1,
        test_macro_f1
    ],

    "Macro Specificity": [
        np.mean([
            (
                (
                    val_cm.sum()
                    - val_cm[i, i]
                    - (val_cm[:, i].sum() - val_cm[i, i])
                    - (val_cm[i, :].sum() - val_cm[i, i])
                )
                /
                (
                    val_cm.sum()
                    - val_cm[i, :].sum()
                )
            )
            if (
                val_cm.sum()
                - val_cm[i, :].sum()
            ) > 0
            else 0
            for i in range(len(LABELS))
        ]),

        macro_specificity
    ]

})


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ======================================================================
# SAVE MODELS
# ======================================================================

print("\n" + "=" * 70)
print("SAVING MODELS AND RESULTS")
print("=" * 70)


joblib.dump(
    stage1_model,
    MODEL_DIR / "stage1_xgboost.joblib"
)

joblib.dump(
    stage2_model,
    MODEL_DIR / "stage2_xgboost.joblib"
)

joblib.dump(
    stage2_label_mapping,
    MODEL_DIR / "stage2_label_mapping.joblib"
)

joblib.dump(
    imputer,
    MODEL_DIR / "imputer.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib"
)

joblib.dump(
    FEATURE_COLUMNS,
    MODEL_DIR / "feature_columns.joblib"
)

joblib.dump(
    LABELS,
    MODEL_DIR / "labels.joblib"
)

joblib.dump(
    STAGE1_THRESHOLD,
    MODEL_DIR / "best_threshold.joblib"
)

joblib.dump(
    class_thresholds,
    MODEL_DIR / "class_specific_thresholds.joblib"
)


# ======================================================================
# SAVE THRESHOLD RESULTS
# ======================================================================

threshold_df.to_csv(
    MODEL_DIR / "threshold_sweep.csv",
    index=False
)


# ======================================================================
# SAVE VALIDATION PREDICTIONS
# ======================================================================

validation_predictions_df = val_df[
    [
        PATIENT_COLUMN,
        LABEL_COLUMN
    ]
].copy()


validation_predictions_df[
    "prediction"
] = val_predictions


validation_predictions_df.to_csv(
    MODEL_DIR / "two_stage_validation_predictions.csv",
    index=False
)


# ======================================================================
# SAVE TEST PREDICTIONS
# ======================================================================

test_predictions_df = ds2[
    [
        PATIENT_COLUMN,
        LABEL_COLUMN
    ]
].copy()


test_predictions_df[
    "prediction"
] = test_predictions


test_predictions_df.to_csv(
    MODEL_DIR / "two_stage_test_predictions.csv",
    index=False
)


# ======================================================================
# SAVE CONFUSION MATRIX
# ======================================================================

np.save(
    MODEL_DIR / "two_stage_test_confusion_matrix.npy",
    test_cm
)


# ======================================================================
# SAVE FEATURE IMPORTANCE
# ======================================================================

feature_importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)


# ======================================================================
# SAVE PER-CLASS METRICS
# ======================================================================

metrics_df.to_csv(
    MODEL_DIR / "per_class_metrics.csv",
    index=False
)


# ======================================================================
# SAVE FINAL RESULTS
# ======================================================================

results_df.to_csv(
    MODEL_DIR / "two_stage_final_results.csv",
    index=False
)


# ======================================================================
# SAVE CLASS THRESHOLDS
# ======================================================================

pd.DataFrame([
    {
        "class": k,
        "threshold": v
    }
    for k, v in class_thresholds.items()
]).to_csv(
    MODEL_DIR / "class_specific_thresholds.csv",
    index=False
)


# ======================================================================
# SAVE TRAINING CONFIGURATION
# ======================================================================

training_configuration = {

    "random_state": RANDOM_STATE,

    "dataset_path": str(DATA_PATH),

    "feature_count": len(FEATURE_COLUMNS),

    "feature_columns": FEATURE_COLUMNS,

    "labels": LABELS,

    "stage1_threshold": STAGE1_THRESHOLD,

    "class_specific_thresholds": class_thresholds,

    "stage1_parameters": stage1_model.get_params(),

    "stage2_parameters": stage2_model.get_params(),

    "training_patients": len(train_patients),

    "validation_patients": len(val_patients),

    "training_beats": len(train_df),

    "validation_beats": len(val_df),

    "test_beats": len(ds2),

    "patient_independent": True

}


joblib.dump(
    training_configuration,
    MODEL_DIR / "training_configuration.joblib"
)


# ======================================================================
# FINAL
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print("\nModels and results saved in:")

print(MODEL_DIR)


print("\nImportant files:")

files = [
    "stage1_xgboost.joblib",
    "stage2_xgboost.joblib",
    "stage2_label_mapping.joblib",
    "imputer.joblib",
    "scaler.joblib",
    "feature_columns.joblib",
    "labels.joblib",
    "best_threshold.joblib",
    "class_specific_thresholds.joblib",
    "threshold_sweep.csv",
    "two_stage_validation_predictions.csv",
    "two_stage_test_predictions.csv",
    "two_stage_test_confusion_matrix.npy",
    "feature_importance.csv",
    "per_class_metrics.csv",
    "two_stage_final_results.csv",
    "class_specific_thresholds.csv",
    "training_configuration.joblib"
]


for filename in files:
    print(f"- {filename}")


print("\n✓ Dataset was NOT modified.")
print("✓ features.csv was NOT rebuilt.")
print("✓ No new features were generated.")
print("✓ Patient-independent evaluation preserved.")
print("✓ Stage 2 prediction uses correct indexing.")
print("✓ DS2 remains untouched until final evaluation.")
print("✓ Class-specific thresholds optimized using validation only.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)