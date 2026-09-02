import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/features.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PATIENT-INDEPENDENT SPLIT
# ============================================================

TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",

    "800", "801", "802", "803", "804", "805", "806", "807",
    "808", "809", "810", "811", "812", "820", "821", "822",
    "823", "824", "825", "826", "827", "828", "829", "840",
    "841", "842", "843", "844", "845", "846", "847", "848",
    "849", "850", "851", "852", "853", "854", "855", "856",
    "857", "858", "859", "860", "861", "862", "863", "864"
]

TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",

    "865", "866", "867", "868", "869", "870", "871", "872",
    "873", "874", "875", "876", "877", "878", "879", "880",
    "881", "882", "883", "884", "885", "886", "887", "888",
    "889", "890", "891", "892", "893", "894"
]


# ============================================================
# FEATURES
# ============================================================

METADATA_COLUMNS = [
    "patient_id",
    "database",
    "label"
]

LABELS = [
    "N",
    "SVEB",
    "VEB",
    "F",
    "Q"
]


# ============================================================
# MODERATED CLASS WEIGHTS
# ============================================================

def calculate_moderated_weights(y):

    counts = pd.Series(y).value_counts()

    total = len(y)
    n_classes = len(counts)

    weights = {}

    for cls in LABELS:

        if cls not in counts:
            weights[cls] = 1.0
            continue

        raw_weight = total / (
            n_classes * counts[cls]
        )

        # Square-root moderation
        moderated = np.sqrt(raw_weight)

        # Prevent extreme values
        moderated = np.clip(
            moderated,
            0.5,
            8.0
        )

        weights[cls] = float(moderated)

    return weights


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(
    y_true,
    y_pred,
    labels
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    specificities = []

    total = np.sum(cm)

    for i in range(len(labels)):

        tp = cm[i, i]

        fn = np.sum(cm[i, :]) - tp

        fp = np.sum(cm[:, i]) - tp

        tn = total - tp - fn - fp

        if (tn + fp) > 0:

            specificity = tn / (
                tn + fp
            )

        else:

            specificity = 0.0

        specificities.append(
            specificity
        )

    return np.array(
        specificities
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test,
    labels
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    specificity = calculate_specificity(
        y_test,
        predictions,
        labels
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {precision.mean():.4f}"
    )

    print(
        f"Macro Recall      : {recall.mean():.4f}"
    )

    print(
        f"Macro F1          : {f1.mean():.4f}"
    )

    print(
        f"Macro Specificity : {specificity.mean():.4f}"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    results = pd.DataFrame({

        "class": labels,

        "precision": precision,

        "sensitivity_recall": recall,

        "f1": f1,

        "support": support,

        "specificity": specificity

    })

    print()
    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "true_label": y_test,

        "predicted_label": predictions

    })

    prediction_path = os.path.join(
        MODEL_DIR,
        f"{name.lower().replace(' ', '_')}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_path = os.path.join(
        MODEL_DIR,
        f"{name.lower().replace(' ', '_')}_confusion_matrix.npy"
    )

    np.save(
        cm_path,
        cm
    )

    return {

        "Model": name,

        "Accuracy": accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Macro Precision":
            precision.mean(),

        "Macro Recall":
            recall.mean(),

        "Macro F1":
            f1.mean(),

        "Macro Specificity":
            specificity.mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECG ARRHYTHMIA CLASSIFICATION")
    print("MITDB + SVDB")
    print("PATIENT-INDEPENDENT TRAINING")
    print("=" * 70)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Total beats: {len(df)}"
    )

    print(
        f"Total records: "
        f"{df['patient_id'].nunique()}"
    )

    feature_columns = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
    ]

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )


    # ========================================================
    # CLEAN PATIENT IDS
    # ========================================================

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # CHECK TRAIN / TEST OVERLAP
    # ========================================================

    overlap = set(TRAIN_RECORDS).intersection(
        set(TEST_RECORDS)
    )

    if len(overlap) > 0:

        raise ValueError(
            f"Patient leakage detected: {overlap}"
        )

    print()
    print(
        "No patient overlap between "
        "training and testing."
    )


    # ========================================================
    # CHECK AVAILABLE RECORDS
    # ========================================================

    available_records = set(
        df["patient_id"].unique()
    )

    train_available = [
        x for x in TRAIN_RECORDS
        if x in available_records
    ]

    test_available = [
        x for x in TEST_RECORDS
        if x in available_records
    ]

    missing_train = [
        x for x in TRAIN_RECORDS
        if x not in available_records
    ]

    missing_test = [
        x for x in TEST_RECORDS
        if x not in available_records
    ]

    print()
    print(
        f"Training records available: "
        f"{len(train_available)}"
    )

    print(
        f"Testing records available: "
        f"{len(test_available)}"
    )

    if missing_train:

        print(
            "\nWARNING - Missing training records:"
        )

        print(
            missing_train
        )

    if missing_test:

        print(
            "\nWARNING - Missing testing records:"
        )

        print(
            missing_test
        )


    # ========================================================
    # PATIENT-INDEPENDENT SPLIT
    # ========================================================

    train_df = df[
        df["patient_id"].isin(
            TRAIN_RECORDS
        )
    ].copy()

    test_df = df[
        df["patient_id"].isin(
            TEST_RECORDS
        )
    ].copy()


    print()
    print("=" * 70)
    print("PATIENT-INDEPENDENT SPLIT")
    print("=" * 70)

    print(
        f"Training records: "
        f"{train_df['patient_id'].nunique()}"
    )

    print(
        f"Testing records : "
        f"{test_df['patient_id'].nunique()}"
    )

    print(
        f"Training beats: "
        f"{len(train_df)}"
    )

    print(
        f"Testing beats : "
        f"{len(test_df)}"
    )


    # ========================================================
    # CHECK LABELS
    # ========================================================

    print()
    print("Training distribution:")

    print(
        train_df["label"].value_counts()
    )

    print()
    print("Testing distribution:")

    print(
        test_df["label"].value_counts()
    )


    # ========================================================
    # FEATURES / LABELS
    # ========================================================

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        "label"
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        "label"
    ].copy()


    # ========================================================
    # INVALID VALUES
    # ========================================================

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print()
    print("Invalid values:")

    print(
        "Training NaN:",
        X_train.isna().sum().sum()
    )

    print(
        "Testing NaN:",
        X_test.isna().sum().sum()
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT FIRST
    #
    # IMPORTANT:
    # We split BEFORE fitting imputer/scaler.
    # This prevents validation leakage.
    # ========================================================

    print()
    print(
        "Creating training / validation split..."
    )

    (
        X_tr_raw,
        X_val_raw,
        y_tr,
        y_val
    ) = train_test_split(

        X_train,

        y_train,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y_train
    )

    print(
        f"Training subset   : "
        f"{len(y_tr)}"
    )

    print(
        f"Validation subset : "
        f"{len(y_val)}"
    )


    # ========================================================
    # IMPUTATION
    # ========================================================

    print()
    print(
        "Fitting median imputer..."
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_tr_imputed = (
        imputer.fit_transform(
            X_tr_raw
        )
    )

    X_val_imputed = (
        imputer.transform(
            X_val_raw
        )
    )

    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )

    joblib.dump(
        imputer,
        os.path.join(
            MODEL_DIR,
            "imputer.joblib"
        )
    )


    # ========================================================
    # SCALING
    # ========================================================

    print(
        "Fitting StandardScaler..."
    )

    scaler = StandardScaler()

    X_tr = scaler.fit_transform(
        X_tr_imputed
    )

    X_val = scaler.transform(
        X_val_imputed
    )

    X_test_scaled = scaler.transform(
        X_test_imputed
    )

    joblib.dump(
        scaler,
        os.path.join(
            MODEL_DIR,
            "scaler.joblib"
        )
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print()
    print("=" * 70)
    print("MODERATED CLASS WEIGHTS")
    print("=" * 70)

    class_weights = (
        calculate_moderated_weights(
            y_tr
        )
    )

    print(
        class_weights
    )


    # ========================================================
    # LABEL ENCODING
    # ========================================================

    label_to_int = {
        label: i
        for i, label in enumerate(
            LABELS
        )
    }

    int_to_label = {
        i: label
        for label, i in
        label_to_int.items()
    }

    y_tr_int = np.array([
        label_to_int[x]
        for x in y_tr
    ])

    y_val_int = np.array([
        label_to_int[x]
        for x in y_val
    ])

    y_test_int = np.array([
        label_to_int[x]
        for x in y_test
    ])


    # ========================================================
    # XGBOOST SAMPLE WEIGHTS
    # ========================================================

    sample_weights = np.array([
        class_weights[x]
        for x in y_tr
    ])


    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    xgb = XGBClassifier(

        n_estimators=700,

        max_depth=5,

        learning_rate=0.04,

        min_child_weight=3,

        subsample=0.85,

        colsample_bytree=0.85,

        gamma=0.1,

        reg_alpha=0.05,

        reg_lambda=2.0,

        objective="multi:softprob",

        num_class=5,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    xgb.fit(

        X_tr,

        y_tr_int,

        sample_weight=sample_weights,

        eval_set=[
            (
                X_val,
                y_val_int
            )
        ],

        verbose=False
    )

    joblib.dump(
        xgb,
        os.path.join(
            MODEL_DIR,
            "xgboost.joblib"
        )
    )

    joblib.dump(
        label_to_int,
        os.path.join(
            MODEL_DIR,
            "xgb_label_mapping.joblib"
        )
    )

    print(
        "XGBoost saved."
    )


    # ========================================================
    # TRAIN SVM
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    SVM_TARGET = 8000

    svm_indices = []

    for cls in LABELS:

        cls_indices = np.where(
            y_tr.values == cls
        )[0]

        if len(cls_indices) > SVM_TARGET:

            selected = rng.choice(

                cls_indices,

                size=SVM_TARGET,

                replace=False
            )

        else:

            selected = cls_indices

        svm_indices.extend(
            selected.tolist()
        )

    svm_indices = np.array(
        svm_indices
    )

    rng.shuffle(
        svm_indices
    )

    X_svm = X_tr[
        svm_indices
    ]

    y_svm = y_tr.iloc[
        svm_indices
    ]

    print(
        f"SVM training samples: "
        f"{len(y_svm)}"
    )

    print()
    print(
        y_svm.value_counts()
    )

    svm = SVC(

        kernel="rbf",

        C=10,

        gamma="scale",

        class_weight="balanced",

        random_state=RANDOM_STATE
    )

    svm.fit(
        X_svm,
        y_svm
    )

    joblib.dump(
        svm,
        os.path.join(
            MODEL_DIR,
            "svm.joblib"
        )
    )

    print(
        "SVM saved."
    )


    # ========================================================
    # TRAIN MLP
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING MLP")
    print("=" * 70)

    MLP_TARGET = 8000

    mlp_indices = []

    for cls in LABELS:

        cls_indices = np.where(
            y_tr.values == cls
        )[0]

        if len(cls_indices) >= MLP_TARGET:

            selected = rng.choice(

                cls_indices,

                size=MLP_TARGET,

                replace=False
            )

        else:

            selected = rng.choice(

                cls_indices,

                size=MLP_TARGET,

                replace=True
            )

        mlp_indices.extend(
            selected.tolist()
        )

    mlp_indices = np.array(
        mlp_indices
    )

    rng.shuffle(
        mlp_indices
    )

    X_mlp = X_tr[
        mlp_indices
    ]

    y_mlp = y_tr.iloc[
        mlp_indices
    ]

    print(
        "\nMLP training distribution:"
    )

    print(
        y_mlp.value_counts()
    )

    mlp = MLPClassifier(

        hidden_layer_sizes=(
            128,
            64,
            32
        ),

        activation="relu",

        solver="adam",

        alpha=0.0005,

        batch_size=256,

        learning_rate_init=0.0005,

        max_iter=100,

        early_stopping=True,

        validation_fraction=0.15,

        n_iter_no_change=10,

        random_state=RANDOM_STATE,

        verbose=True
    )

    mlp.fit(
        X_mlp,
        y_mlp
    )

    joblib.dump(
        mlp,
        os.path.join(
            MODEL_DIR,
            "mlp.joblib"
        )
    )

    print(
        "MLP saved."
    )


    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL PATIENT-INDEPENDENT TEST")
    print("=" * 70)

    results = []


    # ========================================================
    # SVM
    # ========================================================

    results.append(
        evaluate_model(

            "SVM",

            svm,

            X_test_scaled,

            y_test,

            LABELS
        )
    )


    # ========================================================
    # XGBOOST WRAPPER
    # ========================================================

    class XGBWrapper:

        def predict(self, X):

            pred = xgb.predict(
                X
            ).astype(int)

            return np.array([

                int_to_label[int(x)]

                for x in pred

            ])


    results.append(
        evaluate_model(

            "XGBoost",

            XGBWrapper(),

            X_test_scaled,

            y_test,

            LABELS
        )
    )


    # ========================================================
    # MLP
    # ========================================================

    results.append(
        evaluate_model(

            "MLP",

            mlp,

            X_test_scaled,

            y_test,

            LABELS
        )
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(

        os.path.join(
            MODEL_DIR,
            "model_comparison.csv"
        ),

        index=False
    )


    # ========================================================
    # SAVE FEATURE NAMES
    # ========================================================

    joblib.dump(

        feature_columns,

        os.path.join(
            MODEL_DIR,
            "feature_columns.joblib"
        )
    )


    # ========================================================
    # SAVE LABEL INFORMATION
    # ========================================================

    joblib.dump(

        LABELS,

        os.path.join(
            MODEL_DIR,
            "labels.joblib"
        )
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Models saved in:",
        MODEL_DIR
    )

    print()
    print(
        "Saved files:"
    )

    print(
        "  - xgboost.joblib"
    )

    print(
        "  - svm.joblib"
    )

    print(
        "  - mlp.joblib"
    )

    print(
        "  - imputer.joblib"
    )

    print(
        "  - scaler.joblib"
    )

    print(
        "  - feature_columns.joblib"
    )

    print(
        "  - labels.joblib"
    )

    print(
        "  - model_comparison.csv"
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()