import os
import warnings
import numpy as np
import pandas as pd
import wfdb

from feature_extraction import extract_features

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATABASES = {
    "MITDB": "databases/mitdb",
    "SVDB": "databases/svdb"
}

OUTPUT_PATH = "data/features.csv"


# ============================================================
# AAMI-STYLE CLASS MAPPING
# ============================================================

MITDB_MAPPING = {

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # --------------------------------------------------------
    # Supraventricular ectopic
    # --------------------------------------------------------
    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",

    # --------------------------------------------------------
    # Ventricular ectopic
    # --------------------------------------------------------
    "V": "VEB",
    "E": "VEB",

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------
    "F": "F",

    # --------------------------------------------------------
    # Unknown / paced / other
    # --------------------------------------------------------
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
    "|": "Q",
    "x": "Q",
    "P": "Q"
}


SVDB_MAPPING = {

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # --------------------------------------------------------
    # Supraventricular ectopic
    # --------------------------------------------------------
    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",

    # --------------------------------------------------------
    # Ventricular ectopic
    # --------------------------------------------------------
    "V": "VEB",
    "E": "VEB",

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------
    "F": "F",

    # --------------------------------------------------------
    # Unknown / paced / other
    # --------------------------------------------------------
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
    "|": "Q",
    "x": "Q",
    "P": "Q"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_record_names(database_path):
    """
    Get all WFDB record names from a local database directory.
    """

    if not os.path.exists(database_path):

        print(
            f"WARNING: Database not found: {database_path}"
        )

        return []

    records = []

    for filename in os.listdir(database_path):

        if filename.endswith(".hea"):

            record_name = filename[:-4]

            if record_name not in records:

                records.append(record_name)

    return sorted(records)


# ============================================================
# SAFE FLOAT CONVERSION
# ============================================================

def safe_float(value):
    """
    Convert feature value to float.

    Invalid values become NaN.
    """

    try:

        value = float(value)

        if np.isfinite(value):

            return value

        return np.nan

    except Exception:

        return np.nan


# ============================================================
# EXTRACT BEAT
# ============================================================

def extract_beat(signal, peak_index, fs=360):
    """
    Extract a fixed-size beat window around an R peak.

    Approximately:
        0.20 sec before R peak
        0.40 sec after R peak
    """

    left = int(0.20 * fs)
    right = int(0.40 * fs)

    start = peak_index - left
    end = peak_index + right

    # Reject beats too close to signal boundaries
    if start < 0 or end > len(signal):

        return None

    beat = signal[start:end]

    return beat


# ============================================================
# SELECT ECG CHANNEL
# ============================================================

def get_lead_signal(record):
    """
    Select the first valid ECG channel.
    """

    signal = record.p_signal

    if signal is None:

        return None

    valid_channels = []

    for ch in range(signal.shape[1]):

        channel = signal[:, ch]

        if np.isfinite(channel).sum() > 0:

            valid_channels.append(ch)

    if not valid_channels:

        return None

    # First valid channel
    channel_index = valid_channels[0]

    return signal[:, channel_index]


# ============================================================
# CALCULATE RR INTERVAL
# ============================================================

def calculate_rr(beat_samples, index, fs):
    """
    Calculate RR interval using ONLY valid beat annotations.

    This is important because WFDB annotation files contain
    non-beat events such as rhythm markers, waveform markers,
    comments, etc.

    beat_samples:
        Sample positions of actual annotated beats.
    """

    if index <= 0:

        return np.nan

    previous_peak = beat_samples[index - 1]

    current_peak = beat_samples[index]

    rr = (
        current_peak - previous_peak
    ) / fs

    if rr <= 0:

        return np.nan

    return rr


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(
    record_path,
    record_name,
    mapping,
    database_name
):

    rows = []

    try:

        print(
            f"Processing {database_name} record "
            f"{record_name}..."
        )

        # ----------------------------------------------------
        # Load ECG record
        # ----------------------------------------------------

        record = wfdb.rdrecord(record_path)

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )

        fs = record.fs

        # ----------------------------------------------------
        # Select ECG signal
        # ----------------------------------------------------

        signal = get_lead_signal(record)

        if signal is None:

            print(
                " -> No valid ECG signal"
            )

            return rows

        samples = np.asarray(
            annotation.sample
        )

        symbols = annotation.symbol

        # ====================================================
        # IMPORTANT FIX
        #
        # Build a list containing ONLY actual beat
        # annotations that are present in our mapping.
        #
        # RR intervals must be calculated from these beats,
        # NOT from every annotation event.
        # ====================================================

        beat_samples = []

        beat_symbols = []

        for sample, symbol in zip(
            samples,
            symbols
        ):

            if symbol in mapping:

                beat_samples.append(
                    int(sample)
                )

                beat_symbols.append(
                    symbol
                )

        beat_samples = np.asarray(
            beat_samples,
            dtype=int
        )

        print(
            f" -> Valid beat annotations: "
            f"{len(beat_samples)}"
        )

        extracted = 0

        # ====================================================
        # PROCESS EACH BEAT
        # ====================================================

        for i, (
            peak,
            symbol
        ) in enumerate(
            zip(
                beat_samples,
                beat_symbols
            )
        ):

            # ------------------------------------------------
            # Convert annotation symbol to AAMI class
            # ------------------------------------------------

            label = mapping[symbol]

            # ------------------------------------------------
            # Extract ECG beat
            # ------------------------------------------------

            beat = extract_beat(
                signal,
                peak,
                fs
            )

            if beat is None:

                continue

            # =================================================
            # RR CONTEXT
            # =================================================

            # Current RR
            rr_interval = calculate_rr(
                beat_samples,
                i,
                fs
            )

            # ------------------------------------------------
            # Previous RR
            #
            # For beat i:
            #
            # previous RR =
            # beat(i-1) - beat(i-2)
            # ------------------------------------------------

            prev_rr = np.nan

            if i >= 2:

                prev_rr = (
                    beat_samples[i - 1]
                    - beat_samples[i - 2]
                ) / fs

            # ------------------------------------------------
            # Next RR
            #
            # For beat i:
            #
            # next RR =
            # beat(i+1) - beat(i)
            # ------------------------------------------------

            next_rr = np.nan

            if i + 1 < len(beat_samples):

                next_rr = (
                    beat_samples[i + 1]
                    - beat_samples[i]
                ) / fs

            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            try:

                features = extract_features(
                    beat,
                    peak_index=None,
                    fs=fs,
                    rr_interval=rr_interval,
                    prev_rr=prev_rr,
                    next_rr=next_rr
                )

            except Exception:

                continue

            # ------------------------------------------------
            # Make sure all features are numeric
            # ------------------------------------------------

            clean_features = {}

            for key, value in features.items():

                clean_features[key] = safe_float(
                    value
                )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            clean_features["patient_id"] = record_name

            clean_features["label"] = label

            clean_features["database"] = database_name

            rows.append(
                clean_features
            )

            extracted += 1

        print(
            f" -> {extracted} beats extracted"
        )

    except Exception as e:

        print(
            f" -> ERROR processing "
            f"{record_name}: {e}"
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        "data",
        exist_ok=True
    )

    all_rows = []

    print("=" * 70)

    print(
        "BUILDING EXPANDED ECG DATASET"
    )

    print(
        "MIT-BIH Arrhythmia Database +"
    )

    print(
        "MIT-BIH Supraventricular Arrhythmia Database"
    )

    print("=" * 70)

    # ========================================================
    # PROCESS DATABASES
    # ========================================================

    for database_name, database_path in DATABASES.items():

        print()

        print("=" * 70)

        print(
            f"DATABASE: {database_name}"
        )

        print(
            f"PATH: {database_path}"
        )

        print("=" * 70)

        records = get_record_names(
            database_path
        )

        print(
            f"Found {len(records)} records"
        )

        # ----------------------------------------------------
        # Select mapping
        # ----------------------------------------------------

        if database_name == "MITDB":

            mapping = MITDB_MAPPING

        else:

            mapping = SVDB_MAPPING

        # ----------------------------------------------------
        # Process records
        # ----------------------------------------------------

        for record_name in records:

            record_path = os.path.join(
                database_path,
                record_name
            )

            rows = process_record(
                record_path,
                record_name,
                mapping,
                database_name
            )

            all_rows.extend(
                rows
            )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if len(all_rows) == 0:

        print()

        print(
            "ERROR: No beats were extracted."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # COLUMN ORDER
    # ========================================================

    metadata_columns = [
        "patient_id",
        "database",
        "label"
    ]

    feature_columns = [
        c for c in df.columns
        if c not in metadata_columns
    ]

    df = df[
        metadata_columns
        + feature_columns
    ]

    # ========================================================
    # REPLACE INF
    # ========================================================

    numeric_columns = [
        c for c in df.columns
        if c not in metadata_columns
    ]

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "DATASET CREATED"
    )

    print("=" * 70)

    print(
        f"Saved {len(df)} total beats "
        f"to {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Database distribution
    # --------------------------------------------------------

    print()

    print(
        "Databases:"
    )

    print(
        df["database"].value_counts()
    )

    # --------------------------------------------------------
    # Patient distribution
    # --------------------------------------------------------

    print()

    print(
        "Patients:"
    )

    print(
        f"Total patients/records: "
        f"{df['patient_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()

    print(
        "Class distribution:"
    )

    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    print()

    print(
        "Feature columns:",
        len(feature_columns)
    )

    print()

    print(
        "Feature names:"
    )

    print(
        feature_columns
    )

    # ========================================================
    # NaN CHECK
    # ========================================================

    nan_count = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    # ========================================================
    # INF CHECK
    # ========================================================

    inf_count = np.isinf(
        df[
            feature_columns
        ]
        .fillna(0)
        .values
    ).sum()

    print()

    print(
        f"NaN values: {nan_count}"
    )

    print(
        f"Inf values: {inf_count}"
    )

    # ========================================================
    # RR FEATURE CHECK
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RR FEATURE CHECK"
    )

    print("=" * 70)

    rr_columns = [
        "rr_interval",
        "prev_rr",
        "next_rr",
        "heart_rate",
        "mean_rr",
        "local_heart_rate",
        "rr_ratio",
        "next_rr_ratio",
        "rr_deviation",
        "rr_deviation_ratio"
    ]

    for column in rr_columns:

        if column in df.columns:

            missing = df[column].isna().sum()

            percentage = (
                missing
                / len(df)
                * 100
            )

            print(
                f"{column:25s} "
                f"NaN: {missing:8d} "
                f"({percentage:6.2f}%)"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "DONE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()