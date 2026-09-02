import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "features.csv")

RESULT_DIR = os.path.join(
    BASE_DIR, "experiments", "results", "imbalance_comparison"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

TARGET = "label"
GROUP = "patient_id"

DROP_COLUMNS = [
    TARGET,
    GROUP,
    "database"
]

FEATURES = [c for c in df.columns if c not in DROP_COLUMNS]

X = df[FEATURES].values
y_labels = df[TARGET].values
groups = df[GROUP].values


# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {
    "N": 0,
    "SVEB": 1,
    "VEB": 2,
    "F": 3,
    "Q": 4
}

inverse_label_map = {
    0: "N",
    1: "SVEB",
    2: "VEB",
    3: "F",
    4: "Q"
}

y = np.array([label_map[label] for label in y_labels])


# ============================================================
# MODEL CONFIGURATION
# SAME AS BEST PREVIOUS XGBOOST
# ============================================================

MODEL_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 5,

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
# CLASS-WEIGHT STRATEGIES
# ============================================================

def get_class_weights(y_train, strategy):

    classes, counts = np.unique(y_train, return_counts=True)

    total = len(y_train)
    n_classes = len(classes)

    frequencies = dict(zip(classes, counts))

    # --------------------------------------------------------
    # 1. BASELINE
    # sklearn-style balanced weighting
    # --------------------------------------------------------

    if strategy == "baseline":

        weights = {
            c: total / (n_classes * frequencies[c])
            for c in classes
        }

    # --------------------------------------------------------
    # 2. MODERATE
    # Square-root of baseline weights
    # Reduces extreme minority weighting
    # --------------------------------------------------------

    elif strategy == "moderate":

        baseline = {
            c: total / (n_classes * frequencies[c])
            for c in classes
        }

        weights = {
            c: np.sqrt(baseline[c])
            for c in classes
        }

        # Normalize so average weight ~= 1
        mean_weight = np.mean(list(weights.values()))

        weights = {
            c: weights[c] / mean_weight
            for c in classes
        }

    # --------------------------------------------------------
    # 3. SQUARE ROOT
    # Inverse square-root frequency
    # --------------------------------------------------------

    elif strategy == "sqrt":

        raw = {
            c: 1 / np.sqrt(frequencies[c])
            for c in classes
        }

        mean_weight = np.mean(list(raw.values()))

        weights = {
            c: raw[c] / mean_weight
            for c in classes
        }

    # --------------------------------------------------------
    # 4. CAPPED
    # Balanced weights, but maximum weight limited
    # --------------------------------------------------------

    elif strategy == "capped":

        weights = {
            c: total / (n_classes * frequencies[c])
            for c in classes
        }

        MAX_WEIGHT = 15.0

        weights = {
            c: min(weights[c], MAX_WEIGHT)
            for c in classes
        }

        # Normalize
        mean_weight = np.mean(list(weights.values()))

        weights = {
            c: weights[c] / mean_weight
            for c in classes
        }

    else:
        raise ValueError("Unknown strategy")

    return weights


# ============================================================
# PATIENT-LEVEL 5-FOLD CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

strategies = [
    "baseline",
    "moderate",
    "sqrt",
    "capped"
]

all_results = []


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for strategy in strategies:

    print("\n" + "=" * 70)
    print("STRATEGY:", strategy.upper())
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        print(f"\nFold {fold}/5")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        groups_train = groups[train_idx]
        groups_val = groups[val_idx]

        # ----------------------------------------------------
        # SAFETY CHECK: NO PATIENT OVERLAP
        # ----------------------------------------------------

        overlap = set(groups_train) & set(groups_val)

        if len(overlap) != 0:
            raise RuntimeError(
                f"Patient leakage detected in fold {fold}"
            )

        print(
            f"Train patients: {len(np.unique(groups_train))}"
        )

        print(
            f"Validation patients: {len(np.unique(groups_val))}"
        )

        # ----------------------------------------------------
        # STANDARDIZATION
        # Fit ONLY on training data
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # ----------------------------------------------------
        # CLASS WEIGHTS
        # ----------------------------------------------------

        class_weights = get_class_weights(
            y_train,
            strategy
        )

        print("Class weights:")

        for c in sorted(class_weights):

            print(
                f"  {inverse_label_map[c]}: "
                f"{class_weights[c]:.4f}"
            )

        sample_weights = np.array([
            class_weights[label]
            for label in y_train
        ])

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = XGBClassifier(
            **MODEL_PARAMS
        )

        model.fit(
            X_train_scaled,
            y_train,
            sample_weight=sample_weights
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(X_val_scaled)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        balanced_acc = balanced_accuracy_score(
            y_val,
            y_pred
        )

        precision = precision_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
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

        per_class_f1 = f1_score(
            y_val,
            y_pred,
            average=None,
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )

        print(
            f"Accuracy:          {accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy: {balanced_acc:.4f}"
        )

        print(
            f"Macro F1:          {macro_f1:.4f}"
        )

        print(
            f"F F1:              {per_class_f1[3]:.4f}"
        )

        print(
            f"SVEB F1:           {per_class_f1[1]:.4f}"
        )

        fold_result = {
            "strategy": strategy,
            "fold": fold,

            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,

            "N_f1": per_class_f1[0],
            "SVEB_f1": per_class_f1[1],
            "VEB_f1": per_class_f1[2],
            "F_f1": per_class_f1[3],
            "Q_f1": per_class_f1[4],

            "train_patients":
                len(np.unique(groups_train)),

            "val_patients":
                len(np.unique(groups_val)),

            "val_F_support":
                np.sum(y_val == 3)
        }

        fold_results.append(fold_result)
        all_results.append(fold_result)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_df = pd.DataFrame(all_results)

fold_path = os.path.join(
    RESULT_DIR,
    "imbalance_fold_results.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "N_f1",
    "SVEB_f1",
    "VEB_f1",
    "F_f1",
    "Q_f1"
]

summary_rows = []

for strategy in strategies:

    subset = fold_df[
        fold_df["strategy"] == strategy
    ]

    row = {
        "strategy": strategy
    }

    for metric in metric_columns:

        row[f"{metric}_mean"] = (
            subset[metric].mean()
        )

        row[f"{metric}_std"] = (
            subset[metric].std()
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    RESULT_DIR,
    "imbalance_strategy_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 90)
print("FINAL CLASS-IMBALANCE COMPARISON")
print("=" * 90)

display_columns = [
    "strategy",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "SVEB_f1_mean",
    "VEB_f1_mean",
    "F_f1_mean",
    "N_f1_mean",
    "Q_f1_mean"
]

print(
    summary_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nResults saved to:")
print(fold_path)
print(summary_path)

print("\nExperiment complete.")