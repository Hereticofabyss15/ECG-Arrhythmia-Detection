import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "results",
    "roc_pr_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

FEATURE_COLS = [
    c for c in df.columns
    if c not in ["patient_id", "database", "label"]
]

X = df[FEATURE_COLS]
y = df["label"]
groups = df["patient_id"]

# Desired class order
classes = ["N", "SVEB", "VEB", "F", "Q"]

print(f"Samples: {len(df):,}")
print(f"Patients: {groups.nunique()}")
print(f"Features: {len(FEATURE_COLS)}")

# ============================================================
# ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

# Force the encoder to use all five classes
label_encoder.fit(classes)

y_encoded = label_encoder.transform(y)

print("\nClass encoding:")

for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# LabelEncoder order is:
# 0 -> F
# 1 -> N
# 2 -> Q
# 3 -> SVEB
# 4 -> VEB

# ============================================================
# MODEL PARAMETERS
# ============================================================

model_params = {
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
# 5-FOLD PATIENT-LEVEL OOF PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PATIENT-LEVEL OOF PROBABILITIES")
print("=" * 70)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []
all_prob = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y_encoded, groups=groups), 1
):

    print(f"\nFold {fold}/5")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_counts = pd.Series(y_train).value_counts()

    total = len(y_train)

    weights = {
        i: total /
        (len(classes) * class_counts[i])
        for i in range(len(classes))
    }

    sample_weights = np.array([
        weights[label]
        for label in y_train
    ])

    print(
        f"Train: {len(train_idx):,} | "
        f"Validation: {len(val_idx):,}"
    )

    print("Class weights:")

    for encoded_class, weight in weights.items():
        class_name = label_encoder.inverse_transform(
            [encoded_class]
        )[0]

        print(
            f"  {class_name}: {weight:.4f}"
        )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model = XGBClassifier(**model_params)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    # --------------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------------

    prob = model.predict_proba(X_val)

    # XGBoost probability columns follow encoded order:
    # F, N, Q, SVEB, VEB

    pred_encoded = np.argmax(prob, axis=1)

    true_labels = label_encoder.inverse_transform(y_val)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    all_true.extend(true_labels)
    all_pred.extend(pred_labels)
    all_prob.append(prob)

    print("Fold complete.")

# ============================================================
# COMBINE OOF RESULTS
# ============================================================

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_prob = np.vstack(all_prob)

print("\n" + "=" * 70)
print("OOF PROBABILITIES GENERATED")
print("=" * 70)

print(f"Samples: {len(all_true):,}")
print(f"Raw probability shape: {all_prob.shape}")

# ============================================================
# REORDER PROBABILITY COLUMNS
# ============================================================

# Current LabelEncoder/XGBoost order:
#
# 0 -> F
# 1 -> N
# 2 -> Q
# 3 -> SVEB
# 4 -> VEB
#
# Desired order:
#
# N, SVEB, VEB, F, Q

all_prob = all_prob[:, [1, 3, 4, 0, 2]]

print("\nProbability column order:")
print("N, SVEB, VEB, F, Q")

# ============================================================
# SAVE OOF PROBABILITIES
# ============================================================

oof = pd.DataFrame({
    "true_label": all_true,
    "predicted_label": all_pred,

    "prob_N": all_prob[:, 0],
    "prob_SVEB": all_prob[:, 1],
    "prob_VEB": all_prob[:, 2],
    "prob_F": all_prob[:, 3],
    "prob_Q": all_prob[:, 4]
})

oof.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "patient_cv_oof_probabilities.csv"
    ),
    index=False
)

print("\nOOF probability file saved.")

# ============================================================
# BINARIZE LABELS
# ============================================================

y_binary = label_binarize(
    all_true,
    classes=classes
)

# ============================================================
# ROC-AUC ANALYSIS
# ============================================================

roc_results = []

plt.figure(figsize=(8, 6))

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_binary[:, i],
        all_prob[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_results.append({
        "class": cls,
        "roc_auc": roc_auc
    })

    plt.plot(
        fpr,
        tpr,
        label=f"{cls} (AUC = {roc_auc:.4f})"
    )

# Macro ROC-AUC

macro_roc_auc = np.mean(
    [
        result["roc_auc"]
        for result in roc_results
    ]
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Patient-Level OOF ROC Curves"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves.png"
    ),
    dpi=300
)

plt.close()

roc_results.append({
    "class": "Macro Average",
    "roc_auc": macro_roc_auc
})

roc_df = pd.DataFrame(
    roc_results
)

roc_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "roc_auc_results.csv"
    ),
    index=False
)

# ============================================================
# PRECISION-RECALL ANALYSIS
# ============================================================

pr_results = []

plt.figure(figsize=(8, 6))

for i, cls in enumerate(classes):

    precision, recall, _ = precision_recall_curve(
        y_binary[:, i],
        all_prob[:, i]
    )

    pr_auc = average_precision_score(
        y_binary[:, i],
        all_prob[:, i]
    )

    pr_results.append({
        "class": cls,
        "pr_auc": pr_auc
    })

    plt.plot(
        recall,
        precision,
        label=f"{cls} (AP = {pr_auc:.4f})"
    )

# Macro PR-AUC

macro_pr_auc = np.mean(
    [
        result["pr_auc"]
        for result in pr_results
    ]
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Patient-Level OOF Precision-Recall Curves"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_curves.png"
    ),
    dpi=300
)

plt.close()

pr_results.append({
    "class": "Macro Average",
    "pr_auc": macro_pr_auc
})

pr_df = pd.DataFrame(
    pr_results
)

pr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pr_auc_results.csv"
    ),
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ROC-AUC RESULTS")
print("=" * 70)

print(
    roc_df.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("PR-AUC RESULTS")
print("=" * 70)

print(
    pr_df.to_string(
        index=False
    )
)

# ============================================================
# VERIFY PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY CHECK")
print("=" * 70)

print(
    "Minimum probability:",
    all_prob.min()
)

print(
    "Maximum probability:",
    all_prob.max()
)

print(
    "Average probability sum:",
    all_prob.sum(axis=1).mean()
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ROC/PR ANALYSIS COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(file)