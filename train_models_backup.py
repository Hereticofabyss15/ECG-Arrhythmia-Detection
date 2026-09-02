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
from sklearn.utils.class_weight import compute_class_weight
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

# We deliberately split by RECORD/PATIENT rather than by beat.
# This prevents beats from the same patient appearing in both
# training and test sets.

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


# ============================================================
# CONTROLLED BALANCING
# ============================================================

def controlled_undersample_and_oversample(
    X,
    y,
    target_per_class=12000
):
    """
    Create a reasonably balanced training set.

    Majority class:
        N is randomly undersampled.

    Minority classes:
        Random oversampling with replacement.

    This is intentionally NOT applied to the test set.
    """

    rng = np.random.RandomState(RANDOM_STATE)

    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)

    X_parts = []
    y_parts = []

    print("\nBalancing training data...")
    print("--------------------------------")

    for cls in classes:

        indices = np.where(y == cls)[0]

        original_count = len(indices)

        if original_count >= target_per_class:

            selected = rng.choice(
                indices,
                size=target_per_class,
                replace=False
            )

        else:

            selected = rng.choice(
                indices,
                size=target_per_class,
                replace=True
            )

        X_parts.append(X[selected])
        y_parts.append(y[selected])

        print(
            f"{cls}: {original_count} -> "
            f"{len(selected)}"
        )

    X_balanced = np.vstack(X_parts)
    y_balanced = np.concatenate(y_parts)

    # Shuffle
    shuffle_idx = rng.permutation(
        len(y_balanced)
    )

    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    return X_balanced, y_balanced


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(y_true, y_pred, labels):

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

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        specificities.append(specificity)

    return np.array(specificities)


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

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
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

    # Per-class table

    results = pd.DataFrame({

        "class": labels,

        "precision": precision,

        "sensitivity_recall": recall,

        "f1": f1,

        "support": support,

        "specificity": specificity
    })

    print()
    print(results.to_string(index=False))

    # Save predictions

    prediction_df = pd.DataFrame({

        "true_label": y_test,

        "predicted_label": predictions
    })

    prediction_df.to_csv(
        os.path.join(
            MODEL_DIR,
            f"{name.lower().replace(' ', '_')}_predictions.csv"
        ),
        index=False
    )

    # Save confusion matrix

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    np.save(
        os.path.join(
            MODEL_DIR,
            f"{name.lower().replace(' ', '_')}_confusion_matrix.npy"
        ),
        cm
    )

    return {

        "Model": name,

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_accuracy,

        "Macro Precision": precision.mean(),

        "Macro Recall": recall.mean(),

        "Macro F1": f1.mean(),

        "Macro Specificity": specificity.mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECG ARRHYTHMIA CLASSIFICATION")
    print("MITDB + SVDB")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Total beats: {len(df)}"
    )

    print(
        f"Total records: {df['patient_id'].nunique()}"
    )

    print(
        f"Total features: "
        f"{len([c for c in df.columns if c not in METADATA_COLUMNS])}"
    )

    # Ensure patient IDs are strings

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Patient-independent split
    # --------------------------------------------------------

    train_df = df[
        df["patient_id"].isin(TRAIN_RECORDS)
    ].copy()

    test_df = df[
        df["patient_id"].isin(TEST_RECORDS)
    ].copy()

    print("\nPatient-independent split")
    print("-------------------------")

    print(
        f"Training records: "
        f"{train_df['patient_id'].nunique()}"
    )

    print(
        f"Testing records : "
        f"{test_df['patient_id'].nunique()}"
    )

    print(
        f"Training beats: {len(train_df)}"
    )

    print(
        f"Testing beats : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
    ]

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

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    labels = [
        "N",
        "SVEB",
        "VEB",
        "F",
        "Q"
    ]

    print("\nTraining distribution:")
    print(
        y_train.value_counts()
    )

    print("\nTesting distribution:")
    print(
        y_test.value_counts()
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("\nInvalid values:")
    print(
        "Training NaN:",
        X_train.isna().sum().sum()
    )

    print(
        "Testing NaN:",
        X_test.isna().sum().sum()
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    print("\nFitting median imputer...")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    joblib.dump(
        imputer,
        os.path.join(
            MODEL_DIR,
            "imputer.joblib"
        )
    )

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    print("Fitting scaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
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

    # --------------------------------------------------------
    # Create validation split
    #
    # This validation split is ONLY used for tuning.
    # The final DS2 test set remains untouched.
    # --------------------------------------------------------

    print(
        "\nCreating training validation split..."
    )

    (
        X_tr,
        X_val,
        y_tr,
        y_val
    ) = train_test_split(

        X_train_scaled,

        y_train,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y_train
    )

    # --------------------------------------------------------
    # Balance ONLY training portion
    # --------------------------------------------------------

    X_balanced, y_balanced = (
        controlled_undersample_and_oversample(
            X_tr,
            y_tr,
            target_per_class=12000
        )
    )

    print(
        "\nBalanced training distribution:"
    )

    print(
        pd.Series(y_balanced)
        .value_counts()
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_values = np.array(
        labels
    )

    present_classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train
    )

    class_weights = dict(
        zip(
            present_classes,
            weights
        )
    )

    print(
        "\nClass weights:"
    )

    print(class_weights)

    # ========================================================
    # SVM
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    svm = SVC(

        kernel="rbf",

        C=10,

        gamma="scale",

        class_weight="balanced",

        random_state=RANDOM_STATE
    )

    svm.fit(
        X_balanced,
        y_balanced
    )

    joblib.dump(
        svm,
        os.path.join(
            MODEL_DIR,
            "svm.joblib"
        )
    )

    print("SVM saved.")

    # ========================================================
    # XGBOOST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    # Encode labels

    label_to_int = {
        label: i
        for i, label in enumerate(labels)
    }

    int_to_label = {
        i: label
        for label, i in label_to_int.items()
    }

    y_balanced_xgb = np.array([
        label_to_int[x]
        for x in y_balanced
    ])

    y_val_xgb = np.array([
        label_to_int[x]
        for x in y_val
    ])

    y_test_xgb = np.array([
        label_to_int[x]
        for x in y_test
    ])

    xgb = XGBClassifier(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softprob",

        num_class=5,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    xgb.fit(

        X_balanced,

        y_balanced_xgb,

        eval_set=[
            (
                X_val,
                y_val_xgb
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

    print("XGBoost saved.")

    # ========================================================
    # MLP
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING MLP")
    print("=" * 70)

    mlp = MLPClassifier(

        hidden_layer_sizes=(
            128,
            64,
            32
        ),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        batch_size=256,

        learning_rate_init=0.001,

        max_iter=80,

        early_stopping=True,

        validation_fraction=0.15,

        n_iter_no_change=8,

        random_state=RANDOM_STATE,

        verbose=True
    )

    mlp.fit(
        X_balanced,
        y_balanced
    )

    joblib.dump(
        mlp,
        os.path.join(
            MODEL_DIR,
            "mlp.joblib"
        )
    )

    print("MLP saved.")

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DS2 EVALUATION")
    print("=" * 70)

    results = []

    # SVM

    results.append(
        evaluate_model(
            "SVM",
            svm,
            X_test_scaled,
            y_test,
            labels
        )
    )

    # XGBoost

    xgb_pred_int = xgb.predict(
        X_test_scaled
    ).astype(int)

    xgb_predictions = np.array([
        int_to_label[int(x)]
        for x in xgb_pred_int
    ])

    # Temporary wrapper so evaluate_model
    # can work with XGBoost.

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
            labels
        )
    )

    # MLP

    results.append(
        evaluate_model(
            "MLP",
            mlp,
            X_test_scaled,
            y_test,
            labels
        )
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

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

    print()
    print(
        "Model comparison saved to "
        "models/model_comparison.csv"
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()