import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rhythm_qrs import detect_r_peaks, FS
from feature_extraction import extract_features


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(r"D:\Downloads\ECG_Arrhythmia_Dataset")

SPLIT_FILES = [
    Path("data/rhythm_train.csv"),
    Path("data/rhythm_val.csv"),
    Path("data/rhythm_test.csv"),
]

OUTPUT_PATH = Path("data/beat_derived_features.csv")

MODEL_PATH = Path("models/final_xgboost/final_xgboost_model.json")
ENCODER_PATH = Path("models/final_xgboost/label_encoder.pkl")
FEATURE_NAMES_PATH = Path("models/final_xgboost/feature_names.json")


# Same beat window used by the original beat-level training pipeline:
# 200 ms before R-peak + 400 ms after R-peak
LEFT_SAMPLES = int(0.20 * FS)
RIGHT_SAMPLES = int(0.40 * FS)


# ============================================================
# LOAD SPLITS
# ============================================================

split_data = []

for split_file in SPLIT_FILES:

    df = pd.read_csv(split_file)

    split_name = split_file.stem.replace(
        "rhythm_",
        ""
    )

    df["split"] = split_name

    split_data.append(df)


records = pd.concat(
    split_data,
    ignore_index=True
)

records = records[
    ["record_id", "rhythm", "split"]
]

print("=" * 60)
print("BATCH BEAT-DERIVED FEATURE EXTRACTION")
print("=" * 60)

print(f"Records: {len(records)}")
print(
    f"Unique records: "
    f"{records['record_id'].nunique()}"
)


# ============================================================
# LOAD BEAT MODEL
# ============================================================

with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = json.load(f)

label_encoder = joblib.load(
    ENCODER_PATH
)

class_names = label_encoder.classes_

model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_beat_features(
    ecg,
    r_peaks
):

    r_peaks = np.asarray(
        r_peaks,
        dtype=int
    )

    beats = []
    valid_peaks = []

    for peak in r_peaks:

        start = peak - LEFT_SAMPLES
        end = peak + RIGHT_SAMPLES

        if start < 0 or end > len(ecg):
            continue

        beats.append(
            ecg[start:end]
        )

        valid_peaks.append(
            peak
        )

    valid_peaks = np.asarray(
        valid_peaks,
        dtype=int
    )

    if len(beats) == 0:

        return None

    feature_rows = []

    for i, beat in enumerate(beats):

        rr_interval = np.nan

        if i >= 1:

            rr_interval = (
                valid_peaks[i]
                - valid_peaks[i - 1]
            ) / FS

        prev_rr = np.nan

        if i >= 2:

            prev_rr = (
                valid_peaks[i - 1]
                - valid_peaks[i - 2]
            ) / FS

        next_rr = np.nan

        if i + 1 < len(valid_peaks):

            next_rr = (
                valid_peaks[i + 1]
                - valid_peaks[i]
            ) / FS

        features = extract_features(
            beat,
            peak_index=None,
            fs=FS,
            rr_interval=rr_interval,
            prev_rr=prev_rr,
            next_rr=next_rr
        )

        feature_rows.append(
            features
        )

    beat_df = pd.DataFrame(
        feature_rows,
        columns=feature_names
    )

    beat_df = beat_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Match the preprocessing used in the test script.
    beat_df = beat_df.fillna(
        beat_df.median()
    )

    # If a column is completely NaN, use zero.
    beat_df = beat_df.fillna(0)

    probabilities = model.predict_proba(
        beat_df
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_labels = (
        label_encoder.inverse_transform(
            predicted_indices
        )
    )

    # --------------------------------------------------------
    # Soft beat-class probabilities
    # --------------------------------------------------------

    mean_probabilities = probabilities.mean(
        axis=0
    )

    result = {}

    for i, class_name in enumerate(class_names):

        result[
            f"beat_prob_{class_name}"
        ] = float(
            mean_probabilities[i]
        )

    # --------------------------------------------------------
    # Beat burden features
    # --------------------------------------------------------

    class_to_index = {
        name: i
        for i, name in enumerate(class_names)
    }

    if "SVEB" in class_to_index:

        result["SVEB_burden"] = float(
            mean_probabilities[
                class_to_index["SVEB"]
            ]
        )

    if "VEB" in class_to_index:

        result["VEB_burden"] = float(
            mean_probabilities[
                class_to_index["VEB"]
            ]
        )

    if "F" in class_to_index:

        result["F_burden"] = float(
            mean_probabilities[
                class_to_index["F"]
            ]
        )

    result["ectopic_burden"] = float(
        sum(
            result.get(
                key,
                0.0
            )
            for key in [
                "SVEB_burden",
                "VEB_burden",
                "F_burden"
            ]
        )
    )

    # --------------------------------------------------------
    # Hard beat sequence statistics
    # --------------------------------------------------------

    result["num_beats"] = int(
        len(predicted_labels)
    )

    # Maximum consecutive VEB run
    max_veb_run = 0
    current_veb_run = 0

    for label in predicted_labels:

        if label == "VEB":

            current_veb_run += 1

            max_veb_run = max(
                max_veb_run,
                current_veb_run
            )

        else:

            current_veb_run = 0

    result["max_VEB_run"] = int(
        max_veb_run
    )

    # Beat-class transition rate
    if len(predicted_labels) > 1:

        transitions = np.sum(
            predicted_labels[1:]
            != predicted_labels[:-1]
        )

        result["beat_transition_rate"] = float(
            transitions
            / (len(predicted_labels) - 1)
        )

    else:

        result["beat_transition_rate"] = 0.0

    # --------------------------------------------------------
    # Beat-class entropy
    # --------------------------------------------------------

    hard_counts = np.array([
        np.sum(
            predicted_labels == class_name
        )
        for class_name in class_names
    ])

    hard_probabilities = (
        hard_counts / len(predicted_labels)
    )

    nonzero = (
        hard_probabilities > 0
    )

    result["beat_class_entropy"] = float(
        -np.sum(
            hard_probabilities[nonzero]
            * np.log2(
                hard_probabilities[nonzero]
            )
        )
    )

    return result


# ============================================================
# PROCESS RECORDS
# ============================================================

results = []

failed = []

for index, row in records.iterrows():

    record_id = row["record_id"]
    rhythm = row["rhythm"]

    mat_path = (
        DATASET_PATH
        / rhythm
        / f"{record_id}.mat"
    )

    try:

        mat_data = loadmat(
            mat_path
        )

        ecg = np.asarray(
            mat_data["val"][1],
            dtype=float
        )

        detection = detect_r_peaks(
            ecg
        )

        r_peaks = np.asarray(
            detection[3],
            dtype=int
        )

        features = extract_beat_features(
            ecg,
            r_peaks
        )

        if features is None:

            raise ValueError(
                "No valid beats"
            )

        features["record_id"] = record_id
        features["rhythm"] = rhythm
        features["split"] = row["split"]

        results.append(
            features
        )

    except Exception as e:

        failed.append({
            "record_id": record_id,
            "rhythm": rhythm,
            "error": str(e)
        })

    if (index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(records)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(
    results
)

column_order = [
    "record_id",
    "rhythm",
    "split"
] + [
    column
    for column in result_df.columns
    if column not in [
        "record_id",
        "rhythm",
        "split"
    ]
]

result_df = result_df[
    column_order
]

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Successful records: "
    f"{len(result_df)}"
)

print(
    f"Failed records: "
    f"{len(failed)}"
)

print(
    f"Features generated: "
    f"{len(result_df.columns) - 3}"
)

print(
    f"Dataset shape: "
    f"{result_df.shape}"
)

print()
print("Saved to:")
print(OUTPUT_PATH)

if failed:

    failed_df = pd.DataFrame(
        failed
    )

    failed_path = Path(
        "data/beat_derived_failed.csv"
    )

    failed_df.to_csv(
        failed_path,
        index=False
    )

    print()
    print(
        f"Failed-record details saved to: "
        f"{failed_path}"
    )