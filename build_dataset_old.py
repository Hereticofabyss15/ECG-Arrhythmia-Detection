import os
import warnings
import numpy as np
import pandas as pd
import wfdb

from feature_extraction import extract_features


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "databases"
OUTPUT_DIR = "data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "features_combined.csv"
)


# ============================================================
# DATABASES
# ============================================================

DATABASES = {
    "mitdb": os.path.join(DATA_DIR, "mitdb"),
    "svdb": os.path.join(DATA_DIR, "svdb"),
}


# ============================================================
# AAMI MAPPING
# ============================================================
#
# N    = Normal
# SVEB = Supraventricular ectopic beat
# VEB  = Ventricular ectopic beat
# F    = Fusion
# Q    = Unknown / paced / other
#
# MITDB and SVDB use WFDB annotation symbols.
#
# We map only annotations that can reasonably be assigned
# to the five classes used by the existing project.
#
# Unknown annotations are ignored.
# ============================================================

AAMI_MAP = {

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    "N": "N",

    # --------------------------------------------------------
    # SUPRAVENTRICULAR ECTOPIC
    # --------------------------------------------------------

    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",

    # --------------------------------------------------------
    # VENTRICULAR ECTOPIC
    # --------------------------------------------------------

    "V": "VEB",
    "E": "VEB",

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    "F": "F",

    # --------------------------------------------------------
    # UNKNOWN / OTHER
    # --------------------------------------------------------

    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


# ============================================================
# RECORD LIST
# ============================================================

def get_records(database_path):

    records = []

    for filename in os.listdir(database_path):

        if filename.endswith(".hea"):

            record_name = filename[:-4]

            records.append(record_name)

    return sorted(records)


# ============================================================
# SAFE FEATURE EXTRACTION
# ============================================================

def safe_extract_features(signal, fs, r_peak_index, rr_interval):

    """
    Wrapper around the existing feature extraction function.

    Different versions of feature_extraction.py may have
    slightly different interfaces, so this function keeps the
    dataset-building logic isolated.
    """

    try:

        features = extract_features(
            signal,
            fs=fs,
            r_peak_index=r_peak_index,
            rr_interval=rr_interval
        )

        return features

    except TypeError:

        try:

            features = extract_features(
                signal,
                fs,
                r_peak_index,
                rr_interval
            )

            return features

        except Exception:

            return None

    except Exception:

        return None


# ============================================================
# PROCESS ONE DATABASE RECORD
# ============================================================

def process_record(database_name, database_path, record_name):

    print(
        f"Processing {database_name} record "
        f"{record_name}..."
    )

    try:

        record_path = os.path.join(
            database_path,
            record_name
        )

        record = wfdb.rdrecord(record_path)

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )

    except Exception as e:

        print(
            f"  ERROR reading record {record_name}: {e}"
        )

        return []

    signal = record.p_signal

    fs = float(record.fs)

    # --------------------------------------------------------
    # Use first ECG channel
    # --------------------------------------------------------

    if signal.ndim == 2:

        ecg = signal[:, 0]

    else:

        ecg = signal

    ecg = np.asarray(
        ecg,
        dtype=float
    )

    samples = annotation.sample
    symbols = annotation.symbol

    rows = []

    previous_peak = None

    # --------------------------------------------------------
    # Process annotations
    # --------------------------------------------------------

    for sample, symbol in zip(
        samples,
        symbols
    ):

        # Ignore annotations that aren't beat labels

        if symbol not in AAMI_MAP:

            continue

        label = AAMI_MAP[symbol]

        sample = int(sample)

        # ----------------------------------------------------
        # Window around beat
        # ----------------------------------------------------

        window_before = int(
            0.25 * fs
        )

        window_after = int(
            0.45 * fs
        )

        start = max(
            0,
            sample - window_before
        )

        end = min(
            len(ecg),
            sample + window_after
        )

        beat = ecg[start:end]

        # Need enough samples

        if len(beat) < 20:

            continue

        # ----------------------------------------------------
        # RR interval
        # ----------------------------------------------------

        if previous_peak is None:

            rr_interval = np.nan

        else:

            rr_interval = (
                sample - previous_peak
            ) / fs

        previous_peak = sample

        # ----------------------------------------------------
        # Extract 46 features
        # ----------------------------------------------------

        try:

            feature_dict = safe_extract_features(
                beat,
                fs,
                sample - start,
                rr_interval
            )

        except Exception:

            feature_dict = None

        if feature_dict is None:

            continue

        # ----------------------------------------------------
        # Convert feature output to dictionary
        # ----------------------------------------------------

        if isinstance(
            feature_dict,
            dict
        ):

            row = feature_dict.copy()

        else:

            try:

                row = dict(
                    feature_dict
                )

            except Exception:

                continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        row["label"] = label

        # Patient IDs must remain unique across databases.
        #
        # For example:
        # MITDB_100
        # SVDB_800
        #
        # This prevents accidental patient leakage if record
        # numbers overlap.
        # ----------------------------------------------------

        row["patient_id"] = (
            f"{database_name.upper()}_{record_name}"
        )

        row["database"] = database_name

        row["record"] = record_name

        rows.append(row)

    print(
        f"  -> {len(rows)} beats extracted"
    )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    all_rows = []

    # ========================================================
    # PROCESS DATABASES
    # ========================================================

    for database_name, database_path in DATABASES.items():

        print()
        print("=" * 60)
        print(
            f"Processing database: {database_name}"
        )
        print("=" * 60)

        if not os.path.exists(
            database_path
        ):

            print(
                f"WARNING: {database_path} does not exist."
            )

            continue

        records = get_records(
            database_path
        )

        print(
            f"Records found: {len(records)}"
        )

        for record_name in records:

            rows = process_record(
                database_name,
                database_path,
                record_name
            )

            all_rows.extend(rows)

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if len(all_rows) == 0:

        print(
            "\nERROR: No beats were extracted."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # REMOVE NON-FEATURE COLUMNS TEMPORARILY
    # ========================================================

    metadata_columns = [
        "label",
        "patient_id",
        "database",
        "record"
    ]

    feature_columns = [
        c for c in df.columns
        if c not in metadata_columns
    ]

    print()
    print(
        "=" * 60
    )

    print(
        f"Total beats: {len(df)}"
    )

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # HANDLE INF
    # ========================================================

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # REPORT INVALID VALUES
    # ========================================================

    nan_count = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"NaN/invalid feature values: {nan_count}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    # Keep NaNs here.
    #
    # Training script should fit the imputer ONLY on the
    # training patients to prevent data leakage.
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved combined dataset to:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print(
        "Class distribution:"
    )

    print(
        df["label"].value_counts()
    )

    # ========================================================
    # DATABASE DISTRIBUTION
    # ========================================================

    print()
    print(
        "Database distribution:"
    )

    print(
        df["database"].value_counts()
    )

    # ========================================================
    # PATIENT COUNT
    # ========================================================

    print()
    print(
        "Patients:"
    )

    print(
        df["patient_id"].nunique()
    )

    # ========================================================
    # SAVE FEATURE LIST
    # ========================================================

    feature_list_file = os.path.join(
        OUTPUT_DIR,
        "feature_names.txt"
    )

    with open(
        feature_list_file,
        "w"
    ) as f:

        for feature in feature_columns:

            f.write(
                feature + "\n"
            )

    print()
    print(
        f"Feature list saved to {feature_list_file}"
    )

    print()
    print(
        "Dataset construction complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()