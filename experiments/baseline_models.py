import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("=" * 75)
print("ECG ARRHYTHMIA CLASSIFICATION")
print("BASELINE MODEL COMPARISON")
print("PATIENT-INDEPENDENT EVALUATION")
print("=" * 75)


# ============================================================
# 1. FIND DATASET
# ============================================================

print("\nSearching for features.csv...")

possible_paths = [
    os.path.join("data", "features.csv"),
    os.path.join("..", "data", "features.csv"),
    os.path.join(
        r"C:\Users\Lenovo\Downloads\ECG_Arrhythmia_Detection",
        "data",
        "features.csv"
    )
]

DATA_PATH = None

for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = os.path.abspath(path)
        break

if DATA_PATH is None:
    matches = glob.glob(
        os.path.abspath(
            os.path.join(
                os.getcwd(),
                "..",
                "**",
                "features.csv"
            )
        ),
        recursive=True
    )

    if matches:
        DATA_PATH = matches[0]

if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find features.csv.\n"
        "Put this script inside the experiments folder."
    )

print("\nDataset found:")
print(DATA_PATH)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total beats : {len(df)}")
print(f"Total columns : {len(df.columns)}")

PATIENT_COL = "patient_id"
LABEL_COL = "label"

print("\nLabels:")
print(df[LABEL_COL].value_counts())


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

exclude = [
    PATIENT_COL,
    LABEL_COL,
    "database"
]

feature_columns = [
    c for c in df.columns
    if c not in exclude
]

print(f"\nTotal features: {len(feature_columns)}")


# ============================================================
# 4. REMOVE INVALID VALUES
# ============================================================

X = df[feature_columns].copy()
y = df[LABEL_COL].copy()

X = X.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 5. PATIENT-INDEPENDENT SPLIT
# ============================================================

print("\n" + "=" * 75)
print("PATIENT-INDEPENDENT DEVELOPMENT / TEST SPLIT")
print("=" * 75)

patients = df[PATIENT_COL].unique()

rng = np.random.RandomState(42)
rng.shuffle(patients)

n_train_patients = len(patients) // 2

train_patients = patients[:n_train_patients]
test_patients = patients[n_train_patients:]

train_mask = df[PATIENT_COL].isin(train_patients)
test_mask = df[PATIENT_COL].isin(test_patients)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print(f"\nTraining patients : {len(train_patients)}")
print(f"Testing patients  : {len(test_patients)}")

print(f"Training beats : {len(X_train)}")
print(f"Testing beats  : {len(X_test)}")

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())


# ============================================================
# 6. LABEL ENCODING
# ============================================================

labels = ["N", "SVEB", "VEB", "F", "Q"]

label_to_int = {
    label: i for i, label in enumerate(labels)
}

y_train_encoded = y_train.map(label_to_int)
y_test_encoded = y_test.map(label_to_int)


# ============================================================
# 7. PREPROCESSING
# ============================================================

print("\nFitting preprocessing...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

class_counts = y_train_encoded.value_counts().sort_index()

total = len(y_train_encoded)
n_classes = len(class_counts)

class_weights = {
    cls: total / (n_classes * count)
    for cls, count in class_counts.items()
}

sample_weights = y_train_encoded.map(class_weights).values


# ============================================================
# 9. MODEL STORAGE
# ============================================================

results = []


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, Xtr, Xte, use_weights=False):

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    if use_weights:
        model.fit(
            Xtr,
            y_train_encoded,
            sample_weight=sample_weights
        )
    else:
        model.fit(
            Xtr,
            y_train_encoded
        )

    predictions = model.predict(Xte)

    accuracy = accuracy_score(
        y_test_encoded,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_encoded,
        predictions
    )

    macro_precision = precision_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_encoded,
            predictions,
            target_names=labels,
            zero_division=0
        )
    )

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1
    })

    return model


# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================

logistic = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1
)

logistic = evaluate_model(
    "Logistic Regression",
    logistic,
    X_train_scaled,
    X_test_scaled
)


# ============================================================
# 12. RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

random_forest = evaluate_model(
    "Random Forest",
    random_forest,
    X_train_imp,
    X_test_imp
)


# ============================================================
# 13. XGBOOST
# ============================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb = evaluate_model(
    "XGBoost Baseline",
    xgb,
    X_train_imp,
    X_test_imp,
    use_weights=True
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)

output_file = os.path.join(
    output_dir,
    "baseline_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 15. DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 75)
print("BASELINE MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 75)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 75)