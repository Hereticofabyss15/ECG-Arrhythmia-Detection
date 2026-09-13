import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import entropy

from rhythm_qrs import detect_r_peaks, FS


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Downloads\ECG_Arrhythmia_Dataset"
OUTPUT_PATH = r"data\rhythm_features.csv"

RHYTHMS = [
    "SB",
    "SR",
    "ST",
    "AFIB",
    "AF",
    "SA",
    "SVT"
]


# ============================================================
# LOAD ECG
# ============================================================

def load_ecg(record_path):

    mat = loadmat(record_path)
    signal = mat["val"]

    # Lead II
    ecg = signal[1].astype(float)

    return ecg


# ============================================================
# RR / HRV FEATURE EXTRACTION
# ============================================================

def extract_rhythm_features(r_peaks, fs):

    r_peaks = np.asarray(r_peaks)

    features = {}

    if len(r_peaks) < 2:
        return features

    # RR intervals in seconds
    rr = np.diff(r_peaks) / fs

    # --------------------------------------------------------
    # Basic heart-rate features
    # --------------------------------------------------------

    features["num_r_peaks"] = len(r_peaks)
    features["num_rr_intervals"] = len(rr)

    features["mean_rr"] = np.mean(rr)
    features["median_rr"] = np.median(rr)
    features["min_rr"] = np.min(rr)
    features["max_rr"] = np.max(rr)

    # More robust overall heart rate
    features["mean_hr"] = 60.0 / np.mean(rr)

    features["min_hr"] = np.min(60.0 / rr)
    features["max_hr"] = np.max(60.0 / rr)

    # --------------------------------------------------------
    # RR variability
    # --------------------------------------------------------

    if len(rr) > 1:
        features["sdnn"] = np.std(rr, ddof=1)
    else:
        features["sdnn"] = 0.0

    rr_diff = np.diff(rr)

    if len(rr_diff) > 0:

        features["rmssd"] = np.sqrt(
            np.mean(rr_diff ** 2)
        )

        features["sdsd"] = np.std(
            rr_diff,
            ddof=1
        ) if len(rr_diff) > 1 else 0.0

        features["mean_abs_rr_diff"] = np.mean(
            np.abs(rr_diff)
        )

        features["max_abs_rr_diff"] = np.max(
            np.abs(rr_diff)
        )

        features["pnn50"] = np.mean(
            np.abs(rr_diff) > 0.05
        )

        features["pnn100"] = np.mean(
            np.abs(rr_diff) > 0.10
        )

        features["rr_diff_std"] = np.std(
            rr_diff,
            ddof=1
        ) if len(rr_diff) > 1 else 0.0

    else:

        features["rmssd"] = 0.0
        features["sdsd"] = 0.0
        features["mean_abs_rr_diff"] = 0.0
        features["max_abs_rr_diff"] = 0.0
        features["pnn50"] = 0.0
        features["pnn100"] = 0.0
        features["rr_diff_std"] = 0.0

    # --------------------------------------------------------
    # RR coefficient of variation
    # --------------------------------------------------------

    if features["mean_rr"] > 0:

        features["rr_cv"] = (
            features["sdnn"] /
            features["mean_rr"]
        )

    else:

        features["rr_cv"] = 0.0

    # --------------------------------------------------------
    # RR distribution
    # --------------------------------------------------------

    q1, q3 = np.percentile(
        rr,
        [25, 75]
    )

    features["rr_iqr"] = q3 - q1

    # --------------------------------------------------------
    # RR entropy
    # --------------------------------------------------------

    hist, _ = np.histogram(
        rr,
        bins="auto"
    )

    hist = hist[hist > 0]

    if len(hist) > 1:

        probabilities = (
            hist /
            np.sum(hist)
        )

        features["rr_entropy"] = entropy(
            probabilities
        )

    else:

        features["rr_entropy"] = 0.0

    return features


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(record_path, rhythm):

    ecg = load_ecg(record_path)

    result = detect_r_peaks(ecg)

    # Return structure of detect_r_peaks():
    # result[0] -> preprocessed ECG
    # result[1] -> MOF scores
    # result[2] -> candidate locations
    # result[3] -> final R-peak locations

    r_peaks = result[3]

    features = extract_rhythm_features(
        r_peaks,
        FS
    )

    features["record_id"] = os.path.splitext(
        os.path.basename(record_path)
    )[0]

    features["rhythm"] = rhythm

    return features


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BATCH RHYTHM FEATURE EXTRACTION")
    print("=" * 70)

    all_features = []

    total_records = 0
    failed_records = 0

    for rhythm in RHYTHMS:

        rhythm_path = os.path.join(
            DATASET_PATH,
            rhythm
        )

        mat_files = sorted(
            [
                f for f in os.listdir(rhythm_path)
                if f.lower().endswith(".mat")
            ]
        )

        print()
        print(
            f"{rhythm}: {len(mat_files)} records"
        )

        for i, mat_file in enumerate(mat_files, start=1):

            record_path = os.path.join(
                rhythm_path,
                mat_file
            )

            try:

                features = process_record(
                    record_path,
                    rhythm
                )

                all_features.append(features)

                total_records += 1

                if i % 50 == 0 or i == len(mat_files):

                    print(
                        f"  Processed {i}/{len(mat_files)}"
                    )

            except Exception as e:

                failed_records += 1

                print(
                    f"  ERROR: {mat_file} -> {e}"
                )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(all_features)

    # Put identifiers first
    feature_columns = [
        col
        for col in df.columns
        if col not in ["record_id", "rhythm"]
    ]

    df = df[
        ["record_id", "rhythm"] +
        feature_columns
    ]

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = os.path.dirname(
        OUTPUT_PATH
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Records processed : {total_records}"
    )

    print(
        f"Failed records    : {failed_records}"
    )

    print(
        f"Features          : {len(feature_columns)}"
    )

    print(
        f"Dataset shape     : {df.shape}"
    )

    print()
    print("Class distribution:")
    print(
        df["rhythm"].value_counts().sort_index()
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("=" * 70)