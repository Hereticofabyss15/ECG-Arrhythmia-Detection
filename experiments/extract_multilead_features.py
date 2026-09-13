import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import periodogram


DATASET_PATH = r"D:\Downloads\ECG_Arrhythmia_Dataset"
OUTPUT_PATH = "data/multilead_features.csv"

FS = 500

LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6"
]


def extract_lead_features(signal):
    signal = np.asarray(signal, dtype=np.float64)

    mean_value = np.mean(signal)
    std_value = np.std(signal)
    rms_value = np.sqrt(np.mean(signal ** 2))
    ptp_value = np.ptp(signal)
    energy_value = np.mean(signal ** 2)

    # Dominant frequency
    frequencies, power = periodogram(signal, fs=FS)

    # Ignore DC component
    if len(power) > 1:
        dominant_index = np.argmax(power[1:]) + 1
        dominant_frequency = frequencies[dominant_index]
    else:
        dominant_frequency = 0.0

    return [
        mean_value,
        std_value,
        rms_value,
        ptp_value,
        energy_value,
        dominant_frequency
    ]


def process_record(mat_path):

    mat_data = loadmat(mat_path)

    if "val" not in mat_data:
        raise ValueError("MAT file does not contain 'val' variable")

    ecg = np.asarray(mat_data["val"])

    if ecg.shape != (12, 5000):
        raise ValueError(
            f"Unexpected ECG shape: {ecg.shape}"
        )

    features = {}

    for lead_index, lead_name in enumerate(LEADS):

        signal = ecg[lead_index]

        values = extract_lead_features(signal)

        features[f"{lead_name}_mean"] = values[0]
        features[f"{lead_name}_std"] = values[1]
        features[f"{lead_name}_rms"] = values[2]
        features[f"{lead_name}_ptp"] = values[3]
        features[f"{lead_name}_energy"] = values[4]
        features[f"{lead_name}_dominant_freq"] = values[5]

    return features


print("=" * 70)
print("MULTI-LEAD ECG FEATURE EXTRACTION")
print("=" * 70)

print(f"Dataset : {DATASET_PATH}")
print(f"Sampling frequency : {FS} Hz")
print(f"Leads : {len(LEADS)}")
print(f"Features per lead : 6")
print(f"Total multi-lead features : {len(LEADS) * 6}")


# Collect record IDs from all rhythm folders
records = []

for rhythm in sorted(os.listdir(DATASET_PATH)):

    rhythm_path = os.path.join(
        DATASET_PATH,
        rhythm
    )

    if not os.path.isdir(rhythm_path):
        continue

    for filename in os.listdir(rhythm_path):

        if filename.lower().endswith(".mat"):

            record_id = os.path.splitext(filename)[0]

            records.append({
                "record_id": record_id,
                "rhythm": rhythm,
                "mat_path": os.path.join(
                    rhythm_path,
                    filename
                )
            })


records = sorted(
    records,
    key=lambda x: x["record_id"]
)


print(f"\nRecords found : {len(records)}")


rows = []
failed = []

for i, record in enumerate(records, start=1):

    try:

        features = process_record(
            record["mat_path"]
        )

        row = {
            "record_id": record["record_id"],
            "rhythm": record["rhythm"]
        }

        row.update(features)

        rows.append(row)

    except Exception as e:

        failed.append({
            "record_id": record["record_id"],
            "rhythm": record["rhythm"],
            "error": str(e)
        })

    if i % 250 == 0 or i == len(records):
        print(
            f"Processed {i}/{len(records)}"
        )


df = pd.DataFrame(rows)


# Remove duplicate record IDs if any exist
before_duplicates = len(df)

df = df.drop_duplicates(
    subset="record_id",
    keep="first"
)

duplicates_removed = (
    before_duplicates - len(df)
)


# Check missing values
feature_columns = [
    column
    for column in df.columns
    if column not in ["record_id", "rhythm"]
]

missing_cells = int(
    df[feature_columns].isna().sum().sum()
)

infinite_cells = int(
    np.isinf(
        df[feature_columns].to_numpy()
    ).sum()
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("MULTI-LEAD FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(f"Successful records : {len(df)}")
print(f"Failed records     : {len(failed)}")
print(f"Duplicates removed : {duplicates_removed}")
print(f"Features           : {len(feature_columns)}")
print(f"Dataset shape      : {df.shape}")
print(f"Missing cells      : {missing_cells}")
print(f"Infinite cells     : {infinite_cells}")

print("\nClass distribution:")

print(
    df["rhythm"].value_counts().sort_index()
)


print("\nFeature groups:")

for lead in LEADS:
    print(
        f"{lead:>3} : 6 features"
    )


if failed:

    failed_df = pd.DataFrame(failed)

    failed_df.to_csv(
        "data/multilead_failed_records.csv",
        index=False
    )

    print(
        "\nFailed records saved to:"
        " data/multilead_failed_records.csv"
    )


print(
    f"\nSaved to: {OUTPUT_PATH}"
)