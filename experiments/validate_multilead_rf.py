import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "data/rhythm_multilead_fusion_features.csv"

print("=" * 70)
print("5-FOLD CROSS-VALIDATION - MULTI-LEAD RANDOM FOREST")
print("=" * 70)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

DROP_FEATURES = ["rr_diff_std"]

metadata = [
    "record_id",
    "rhythm",
    "split"
]

features = [
    col for col in df.columns
    if col not in metadata
    and col not in DROP_FEATURES
]

assert len(features) == 99

# IMPORTANT:
# Only the original training set is used for CV.
train = df[df["split"] == "train"].copy()

X = train[features]
y = train["rhythm"]

print(f"\nTraining records : {len(train)}")
print(f"Features         : {len(features)}")

print("\nClass distribution:")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# 5-fold stratified CV
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print(f"\nTraining fold {fold}/5...")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_val,
        y_pred
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1
    })

    all_true.extend(y_val)
    all_pred.extend(y_pred)

    print(
        f"Accuracy          : {accuracy:.4f}\n"
        f"Balanced Accuracy : {balanced_acc:.4f}\n"
        f"Macro F1          : {macro_f1:.4f}"
    )

# ------------------------------------------------------------
# Overall CV results
# ------------------------------------------------------------

results = pd.DataFrame(fold_results)

pooled_accuracy = accuracy_score(
    all_true,
    all_pred
)

pooled_balanced = balanced_accuracy_score(
    all_true,
    all_pred
)

pooled_macro_f1 = f1_score(
    all_true,
    all_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("5-FOLD CV RESULTS")
print("=" * 70)

print("\nFold results:")
print(
    results.to_string(
        index=False
    )
)

print("\nMean ± standard deviation:")

print(
    f"Accuracy          : "
    f"{results['accuracy'].mean():.4f} "
    f"± {results['accuracy'].std():.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{results['balanced_accuracy'].mean():.4f} "
    f"± {results['balanced_accuracy'].std():.4f}"
)

print(
    f"Macro F1          : "
    f"{results['macro_f1'].mean():.4f} "
    f"± {results['macro_f1'].std():.4f}"
)

print("\nPooled OOF results:")

print(
    f"Accuracy          : {pooled_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {pooled_balanced:.4f}"
)

print(
    f"Macro F1          : {pooled_macro_f1:.4f}"
)

# ------------------------------------------------------------
# Per-class OOF performance
# ------------------------------------------------------------

labels = sorted(y.unique())

print("\n" + "=" * 70)
print("PER-CLASS OOF RESULTS")
print("=" * 70)

print(
    classification_report(
        all_true,
        all_pred,
        labels=labels,
        digits=4
    )
)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=labels
)

print("Confusion matrix:")
print("Labels:", labels)
print(cm)

# ------------------------------------------------------------
# Compare with previous RF CV
# ------------------------------------------------------------

previous_macro_f1 = 0.6968

print("\n" + "=" * 70)
print("COMPARISON WITH PREVIOUS RF")
print("=" * 70)

print(
    f"Previous 28-feature RF CV Macro F1 : "
    f"{previous_macro_f1:.4f}"
)

print(
    f"New 99-feature RF CV Macro F1      : "
    f"{pooled_macro_f1:.4f}"
)

print(
    f"Improvement                        : "
    f"{pooled_macro_f1 - previous_macro_f1:+.4f}"
)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results.to_csv(
    "data/rhythm_multilead_rf_5fold_results.csv",
    index=False
)

oof = pd.DataFrame({
    "y_true": all_true,
    "y_pred": all_pred
})

oof.to_csv(
    "data/rhythm_multilead_rf_oof_predictions.csv",
    index=False
)

print("\nSaved:")
print("  data/rhythm_multilead_rf_5fold_results.csv")
print("  data/rhythm_multilead_rf_oof_predictions.csv")

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)