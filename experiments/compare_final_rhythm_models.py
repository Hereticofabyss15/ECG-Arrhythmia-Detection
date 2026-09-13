import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = "data/rhythm_multilead_fusion_features.csv"

RANDOM_STATE = 42
N_SPLITS = 5


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(DATA_PATH)

print("=" * 75)
print("FINAL RHYTHM MODEL COMPARISON")
print("=" * 75)

print(f"Dataset shape: {df.shape}")


# ================================================================
# FEATURE SELECTION
# ================================================================

metadata_cols = [
    "record_id",
    "rhythm",
    "split"
]

# Exact duplicate feature already identified
drop_feature = "rr_diff_std"

feature_cols = [
    col
    for col in df.columns
    if col not in metadata_cols + [drop_feature]
]

print(f"Number of model features: {len(feature_cols)}")


# ================================================================
# TRAIN + VALIDATION / TEST SPLIT
# ================================================================

train_val = df[
    df["split"].isin(["train", "val"])
].copy()

test = df[
    df["split"] == "test"
].copy()

X_train_val = train_val[feature_cols]
y_train_val = train_val["rhythm"]

X_test = test[feature_cols]
y_test = test["rhythm"]

print(f"Training + validation records: {len(train_val)}")
print(f"Test records: {len(test)}")

print("\nTrain + validation class distribution:")
print(y_train_val.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())


# ================================================================
# ENCODE LABELS FOR XGBOOST
# ================================================================

label_encoder = LabelEncoder()

y_train_val_encoded = label_encoder.fit_transform(y_train_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nXGBoost class mapping:")

for index, label in enumerate(label_encoder.classes_):
    print(f"{index} -> {label}")


# ================================================================
# MODELS
# ================================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ================================================================
# CROSS-VALIDATION ON TRAIN + VALIDATION ONLY
# ================================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}


comparison_results = []


print("\n" + "=" * 75)
print("5-FOLD CROSS-VALIDATION")
print("=" * 75)

print("Test set remains completely untouched.\n")


# ================================================================
# LOGISTIC REGRESSION, SVM, RANDOM FOREST
# ================================================================

for name in [
    "Logistic Regression",
    "SVM",
    "Random Forest"
]:

    model = models[name]

    print("-" * 75)
    print(name)
    print("-" * 75)

    scores = cross_validate(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    balanced_mean = scores["test_balanced_accuracy"].mean()
    balanced_std = scores["test_balanced_accuracy"].std()

    macro_f1_mean = scores["test_macro_f1"].mean()
    macro_f1_std = scores["test_macro_f1"].std()

    comparison_results.append({
        "model": name,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "balanced_accuracy_mean": balanced_mean,
        "balanced_accuracy_std": balanced_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std
    })

    print(
        f"Accuracy:          "
        f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_mean:.4f} ± {balanced_std:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1_mean:.4f} ± {macro_f1_std:.4f}"
    )


# ================================================================
# XGBOOST CROSS-VALIDATION
# ================================================================

print("-" * 75)
print("XGBoost")
print("-" * 75)

xgb_scores = cross_validate(
    models["XGBoost"],
    X_train_val,
    y_train_val_encoded,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
    return_train_score=False
)

accuracy_mean = xgb_scores["test_accuracy"].mean()
accuracy_std = xgb_scores["test_accuracy"].std()

balanced_mean = xgb_scores["test_balanced_accuracy"].mean()
balanced_std = xgb_scores["test_balanced_accuracy"].std()

macro_f1_mean = xgb_scores["test_macro_f1"].mean()
macro_f1_std = xgb_scores["test_macro_f1"].std()

comparison_results.append({
    "model": "XGBoost",
    "accuracy_mean": accuracy_mean,
    "accuracy_std": accuracy_std,
    "balanced_accuracy_mean": balanced_mean,
    "balanced_accuracy_std": balanced_std,
    "macro_f1_mean": macro_f1_mean,
    "macro_f1_std": macro_f1_std
})

print(
    f"Accuracy:          "
    f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_mean:.4f} ± {balanced_std:.4f}"
)

print(
    f"Macro F1:          "
    f"{macro_f1_mean:.4f} ± {macro_f1_std:.4f}"
)


# ================================================================
# MODEL COMPARISON
# ================================================================

comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df.sort_values(
    by="macro_f1_mean",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "accuracy_mean": "{:.4f}".format,
            "accuracy_std": "{:.4f}".format,
            "balanced_accuracy_mean": "{:.4f}".format,
            "balanced_accuracy_std": "{:.4f}".format,
            "macro_f1_mean": "{:.4f}".format,
            "macro_f1_std": "{:.4f}".format
        }
    )
)


# ================================================================
# SELECT BEST MODEL
# ================================================================

best_model_name = comparison_df.iloc[0]["model"]

print("\n" + "=" * 75)
print("SELECTED MODEL")
print("=" * 75)

print(
    f"Best model based on mean CV Macro F1: "
    f"{best_model_name}"
)


# ================================================================
# SAVE MODEL COMPARISON
# ================================================================

comparison_df.to_csv(
    "data/final_rhythm_model_comparison.csv",
    index=False
)

print(
    "\nSaved model comparison:"
    "\ndata/final_rhythm_model_comparison.csv"
)


# ================================================================
# FINAL MODEL TRAINING
# ================================================================

print("\n" + "=" * 75)
print("FINAL MODEL TRAINING")
print("=" * 75)


# XGBoost needs encoded labels
if best_model_name == "XGBoost":

    final_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print(
        f"Training {best_model_name} "
        "on complete Train + Validation data..."
    )

    final_model.fit(
        X_train_val,
        y_train_val_encoded
    )

    y_test_pred_encoded = final_model.predict(
        X_test
    )

    y_test_pred = label_encoder.inverse_transform(
        y_test_pred_encoded.astype(int)
    )

else:

    final_model = models[best_model_name]

    print(
        f"Training {best_model_name} "
        "on complete Train + Validation data..."
    )

    final_model.fit(
        X_train_val,
        y_train_val
    )

    y_test_pred = final_model.predict(
        X_test
    )


# ================================================================
# FINAL TEST EVALUATION
# ================================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

print(
    "Evaluating the selected model on the untouched test set..."
)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_test_pred
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)


print(
    f"\nFinal Test Accuracy:          "
    f"{test_accuracy:.4f}"
)

print(
    f"Final Test Balanced Accuracy: "
    f"{test_balanced_accuracy:.4f}"
)

print(
    f"Final Test Macro F1:          "
    f"{test_macro_f1:.4f}"
)


# ================================================================
# PER-CLASS RESULTS
# ================================================================

print("\n" + "=" * 75)
print("FINAL TEST CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=4
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n" + "=" * 75)
print("FINAL TEST CONFUSION MATRIX")
print("=" * 75)

print(cm_df)


# ================================================================
# SAVE FINAL TEST RESULTS
# ================================================================

final_results = pd.DataFrame([
    {
        "model": best_model_name,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_macro_f1": test_macro_f1
    }
])

final_results.to_csv(
    "data/final_rhythm_test_results.csv",
    index=False
)

cm_df.to_csv(
    "data/final_rhythm_confusion_matrix.csv"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(f"Selected model:          {best_model_name}")
print(f"Test Accuracy:           {test_accuracy:.4f}")
print(f"Test Balanced Accuracy:  {test_balanced_accuracy:.4f}")
print(f"Test Macro F1:           {test_macro_f1:.4f}")

print("\nSaved:")
print("data/final_rhythm_model_comparison.csv")
print("data/final_rhythm_test_results.csv")
print("data/final_rhythm_confusion_matrix.csv")

print("\nTEST SET IS NOW LOCKED.")
print("No further model selection or tuning should be performed.")