import sys
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import loadmat
from scipy.signal import periodogram
from xgboost import XGBClassifier

from rhythm_qrs import detect_r_peaks
from rhythm_features import extract_rhythm_features
from feature_extraction import extract_features


# ================================================================
# CONFIGURATION
# ================================================================

FS = 500

PROJECT_DIR = Path(__file__).resolve().parent

# Final rhythm model
RHYTHM_MODEL_PATH = (
    PROJECT_DIR / "models" / "final_rhythm_rf_model.pkl"
)

RHYTHM_FEATURE_PATH = (
    PROJECT_DIR / "models" / "final_rhythm_feature_names.json"
)

# Final beat-level model
BEAT_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_xgboost"
    / "final_xgboost_model.json"
)

BEAT_ENCODER_PATH = (
    PROJECT_DIR
    / "models"
    / "final_xgboost"
    / "label_encoder.pkl"
)

BEAT_FEATURE_PATH = (
    PROJECT_DIR
    / "models"
    / "final_xgboost"
    / "feature_names.json"
)


# 12-lead order used during training
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


# Beat segmentation used during training
LEFT_SAMPLES = int(0.20 * FS)
RIGHT_SAMPLES = int(0.40 * FS)


# ================================================================
# LOAD MODELS
# ================================================================

def load_models():

    print("Loading models...")

    rhythm_model = joblib.load(
        RHYTHM_MODEL_PATH
    )

    with open(
        RHYTHM_FEATURE_PATH,
        "r"
    ) as f:

        rhythm_feature_names = json.load(f)

    beat_model = XGBClassifier()

    beat_model.load_model(
        BEAT_MODEL_PATH
    )

    beat_encoder = joblib.load(
        BEAT_ENCODER_PATH
    )

    with open(
        BEAT_FEATURE_PATH,
        "r"
    ) as f:

        beat_feature_names = json.load(f)

    print("Models loaded.")

    return (
        rhythm_model,
        rhythm_feature_names,
        beat_model,
        beat_encoder,
        beat_feature_names
    )


# ================================================================
# LOAD ECG FROM MAT FILE
# ================================================================

def load_ecg(mat_path):

    mat_path = Path(mat_path)

    if not mat_path.exists():

        raise FileNotFoundError(
            f"ECG file not found:\n{mat_path}"
        )

    data = loadmat(mat_path)

    if "val" not in data:

        raise ValueError(
            "The .mat file does not contain "
            "a 'val' variable."
        )

    ecg = np.asarray(
        data["val"]
    )

    if ecg.shape != (12, 5000):

        raise ValueError(
            f"Unexpected ECG shape: {ecg.shape}. "
            f"Expected (12, 5000)."
        )

    return ecg.astype(float)


# ================================================================
# RR / HRV FEATURES
# ================================================================

def extract_rr_features(r_peaks):

    features = extract_rhythm_features(
        r_peaks,
        FS
    )

    if not features:

        raise ValueError(
            "Could not extract RR/HRV features. "
            "At least two R-peaks are required."
        )

    return features


# ================================================================
# BEAT-LEVEL FEATURES
# ================================================================

def extract_beat_derived_features(
    lead_ii,
    r_peaks,
    beat_model,
    beat_encoder,
    beat_feature_names
):

    beat_rows = []

    n_samples = len(lead_ii)

    # ------------------------------------------------------------
    # Extract individual beats
    # ------------------------------------------------------------

    for i, peak in enumerate(r_peaks):

        start = (
            peak -
            LEFT_SAMPLES
        )

        end = (
            peak +
            RIGHT_SAMPLES
        )

        if start < 0 or end > n_samples:

            continue

        beat = lead_ii[
            start:end
        ]

        if len(beat) != (
            LEFT_SAMPLES +
            RIGHT_SAMPLES
        ):

            continue

        rr_interval = None
        prev_rr = None
        next_rr = None

        if i > 0:

            rr_interval = (
                r_peaks[i] -
                r_peaks[i - 1]
            ) / FS

        if i > 1:

            prev_rr = (
                r_peaks[i - 1] -
                r_peaks[i - 2]
            ) / FS

        if i < len(r_peaks) - 1:

            next_rr = (
                r_peaks[i + 1] -
                r_peaks[i]
            ) / FS

        features = extract_features(
            beat,
            peak_index=None,
            fs=FS,
            rr_interval=rr_interval,
            prev_rr=prev_rr,
            next_rr=next_rr
        )

        beat_rows.append(
            features
        )

    if not beat_rows:

        raise ValueError(
            "No valid beats could be extracted."
        )

    # ------------------------------------------------------------
    # Build beat feature matrix
    # ------------------------------------------------------------

    X_beats = pd.DataFrame(
        beat_rows,
        columns=beat_feature_names
    )

    X_beats = X_beats.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Same imputation used during training
    X_beats = X_beats.fillna(
        X_beats.median()
    )

    X_beats = X_beats.fillna(0)

    # ------------------------------------------------------------
    # Beat XGBoost prediction
    # ------------------------------------------------------------

    beat_probabilities = (
        beat_model.predict_proba(
            X_beats
        )
    )

    beat_predictions_encoded = (
        beat_model.predict(
            X_beats
        )
    )

    beat_predictions = (
        beat_encoder.inverse_transform(
            beat_predictions_encoded.astype(int)
        )
    )

    class_names = list(
        beat_encoder.classes_
    )

    probability_df = pd.DataFrame(
        beat_probabilities,
        columns=class_names
    )

    # ------------------------------------------------------------
    # Beat probability features
    # ------------------------------------------------------------

    beat_prob_F = (
        probability_df["F"].mean()
    )

    beat_prob_N = (
        probability_df["N"].mean()
    )

    beat_prob_Q = (
        probability_df["Q"].mean()
    )

    beat_prob_SVEB = (
        probability_df["SVEB"].mean()
    )

    beat_prob_VEB = (
        probability_df["VEB"].mean()
    )

    # ------------------------------------------------------------
    # Hard beat-class burdens
    # ------------------------------------------------------------

    SVEB_burden = np.mean(
        beat_predictions == "SVEB"
    )

    VEB_burden = np.mean(
        beat_predictions == "VEB"
    )

    F_burden = np.mean(
        beat_predictions == "F"
    )

    ectopic_burden = (
        SVEB_burden +
        VEB_burden +
        F_burden
    )

    num_beats = len(
        beat_predictions
    )

    # ------------------------------------------------------------
    # Maximum VEB run
    # ------------------------------------------------------------

    max_VEB_run = 0
    current_run = 0

    for label in beat_predictions:

        if label == "VEB":

            current_run += 1

            max_VEB_run = max(
                max_VEB_run,
                current_run
            )

        else:

            current_run = 0

    # ------------------------------------------------------------
    # Beat transition rate
    # ------------------------------------------------------------

    if len(beat_predictions) > 1:

        transitions = np.sum(
            beat_predictions[1:] !=
            beat_predictions[:-1]
        )

        beat_transition_rate = (
            transitions /
            (len(beat_predictions) - 1)
        )

    else:

        beat_transition_rate = 0.0

    # ------------------------------------------------------------
    # Beat-class entropy
    # ------------------------------------------------------------

    counts = probability_df.mean().values

    counts = counts[
        counts > 0
    ]

    if len(counts) > 0:

        probabilities = (
            counts /
            counts.sum()
        )

        beat_class_entropy = -np.sum(
            probabilities *
            np.log(probabilities)
        )

    else:

        beat_class_entropy = 0.0

    # ------------------------------------------------------------
    # Final beat-derived features
    # ------------------------------------------------------------

    features = {

        "beat_prob_F":
            beat_prob_F,

        "beat_prob_N":
            beat_prob_N,

        "beat_prob_Q":
            beat_prob_Q,

        "beat_prob_SVEB":
            beat_prob_SVEB,

        "beat_prob_VEB":
            beat_prob_VEB,

        "ectopic_burden":
            ectopic_burden,

        "num_beats":
            num_beats,

        "max_VEB_run":
            max_VEB_run,

        "beat_transition_rate":
            beat_transition_rate,

        "beat_class_entropy":
            beat_class_entropy
    }

    return (
        features,
        beat_predictions
    )


# ================================================================
# MULTI-LEAD FEATURES
# ================================================================

def extract_multilead_features(ecg):

    features = {}

    for lead_index, lead_name in enumerate(LEADS):

        signal = ecg[
            lead_index
        ]

        mean_value = np.mean(
            signal
        )

        std_value = np.std(
            signal
        )

        rms_value = np.sqrt(
            np.mean(
                signal ** 2
            )
        )

        ptp_value = np.ptp(
            signal
        )

        energy_value = np.mean(
            signal ** 2
        )

        frequencies, power = periodogram(
            signal,
            fs=FS
        )

        # Ignore DC component
        if len(power) > 1:

            dominant_index = (
                np.argmax(
                    power[1:]
                ) + 1
            )

            dominant_frequency = (
                frequencies[
                    dominant_index
                ]
            )

        else:

            dominant_frequency = 0.0

        features[
            f"{lead_name}_mean"
        ] = mean_value

        features[
            f"{lead_name}_std"
        ] = std_value

        features[
            f"{lead_name}_rms"
        ] = rms_value

        features[
            f"{lead_name}_ptp"
        ] = ptp_value

        features[
            f"{lead_name}_energy"
        ] = energy_value

        features[
            f"{lead_name}_dominant_freq"
        ] = dominant_frequency

    return features


# ================================================================
# BUILD FINAL 99-FEATURE VECTOR
# ================================================================

def build_feature_vector(
    ecg,
    r_peaks,
    beat_model,
    beat_encoder,
    beat_feature_names,
    final_feature_names
):

    # ------------------------------------------------------------
    # RR / HRV
    # ------------------------------------------------------------

    rr_features = extract_rr_features(
        r_peaks
    )

    rr_features.pop(
        "num_r_peaks",
        None
    )

    rr_features.pop(
        "num_rr_intervals",
        None
    )

    rr_features.pop(
        "rr_diff_std",
        None
    )

    # ------------------------------------------------------------
    # Beat-derived
    # ------------------------------------------------------------

    (
        beat_features,
        beat_predictions
    ) = extract_beat_derived_features(
        ecg[1],
        r_peaks,
        beat_model,
        beat_encoder,
        beat_feature_names
    )

    # ------------------------------------------------------------
    # Multi-lead
    # ------------------------------------------------------------

    multilead_features = (
        extract_multilead_features(
            ecg
        )
    )

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    all_features = {}

    all_features.update(
        rr_features
    )

    all_features.update(
        beat_features
    )

    all_features.update(
        multilead_features
    )

    # ------------------------------------------------------------
    # Verify exact feature set
    # ------------------------------------------------------------

    missing_features = [
        feature
        for feature in final_feature_names
        if feature not in all_features
    ]

    if missing_features:

        raise ValueError(
            "Missing final model features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X = pd.DataFrame(
        [[
            all_features[feature]
            for feature in final_feature_names
        ]],
        columns=final_feature_names
    )

    if X.shape[1] != 99:

        raise ValueError(
            f"Expected 99 features, "
            f"got {X.shape[1]}."
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Final feature vector contains "
            "NaN or infinite values."
        )

    return (
        X,
        beat_predictions,
        all_features
    )


# ================================================================
# MAIN ECG INFERENCE FUNCTION
# ================================================================

def predict_rhythm_from_ecg(
    ecg,
    display_input="ECG waveform"
):

    print("\n" + "=" * 70)
    print("ECG RHYTHM INFERENCE")
    print("=" * 70)

    print(
        f"\nInput: {display_input}"
    )

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    (
        rhythm_model,
        rhythm_feature_names,
        beat_model,
        beat_encoder,
        beat_feature_names
    ) = load_models()

    # ------------------------------------------------------------
    # Validate ECG
    # ------------------------------------------------------------

    ecg = np.asarray(
        ecg,
        dtype=float
    )

    if ecg.shape != (12, 5000):

        raise ValueError(
            f"Unexpected ECG shape: {ecg.shape}. "
            f"Expected (12, 5000)."
        )

    print(
        f"\nECG shape: {ecg.shape}"
    )

    print(
        f"Sampling frequency: {FS} Hz"
    )

    print(
        f"Duration: "
        f"{ecg.shape[1] / FS:.1f} seconds"
    )

    # ------------------------------------------------------------
    # R-peak detection
    # ------------------------------------------------------------

    print(
        "\nDetecting R-peaks..."
    )

    (
        _,
        _,
        candidates,
        r_peaks
    ) = detect_r_peaks(
        ecg[1]
    )

    print(
        f"Candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Final R-peaks: "
        f"{len(r_peaks)}"
    )

    if len(r_peaks) < 2:

        raise ValueError(
            "Too few R-peaks detected "
            "for rhythm analysis."
        )

    # ------------------------------------------------------------
    # Build 99 features
    # ------------------------------------------------------------

    print(
        "\nExtracting final 99 features..."
    )

    (
        X,
        beat_predictions,
        all_features
    ) = build_feature_vector(
        ecg,
        r_peaks,
        beat_model,
        beat_encoder,
        beat_feature_names,
        rhythm_feature_names
    )

    print(
        f"Feature vector shape: "
        f"{X.shape}"
    )

    # ------------------------------------------------------------
    # Final rhythm prediction
    # ------------------------------------------------------------

    prediction = rhythm_model.predict(
        X
    )[0]

    probabilities = (
        rhythm_model.predict_proba(
            X
        )[0]
    )

    class_names = (
        rhythm_model.classes_
    )

    probability_table = (
        pd.DataFrame({
            "Rhythm": class_names,
            "Probability": probabilities
        })
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RHYTHM PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPredicted rhythm: "
        f"{prediction}"
    )

    print(
        f"R-peaks detected: "
        f"{len(r_peaks)}"
    )

    print(
        f"Estimated mean HR: "
        f"{all_features['mean_hr']:.2f} BPM"
    )

    print(
        "\nRhythm probabilities:"
    )

    for _, row in (
        probability_table.iterrows()
    ):

        print(
            f"  {row['Rhythm']:5s} : "
            f"{row['Probability'] * 100:6.2f}%"
        )

    print(
        "\nBeat-level summary:"
    )

    beat_counts = pd.Series(
        beat_predictions
    ).value_counts()

    for label in [
        "N",
        "SVEB",
        "VEB",
        "F",
        "Q"
    ]:

        count = beat_counts.get(
            label,
            0
        )

        print(
            f"  {label:5s}: "
            f"{count}"
        )

    print(
        "\n" + "=" * 70
    )

    # ------------------------------------------------------------
    # Return structured results
    # ------------------------------------------------------------

    return {
        "prediction":
            prediction,

        "probabilities":
            probability_table,

        "r_peaks":
            r_peaks,

        "candidates":
            candidates,

        "mean_hr":
            all_features["mean_hr"],

        "beat_predictions":
            beat_predictions,

        "beat_features":
            all_features,

        "feature_vector":
            X,

        "ecg":
            ecg
    }


# ================================================================
# MAT FILE INFERENCE
# ================================================================

def predict_rhythm(mat_path):

    ecg = load_ecg(
        mat_path
    )

    return predict_rhythm_from_ecg(
        ecg,
        display_input=str(
            mat_path
        )
    )


# ================================================================
# COMMAND-LINE INTERFACE
# ================================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python rhythm_inference.py "
            "<path_to_mat_file>"
        )

        sys.exit(1)

    mat_file = sys.argv[1]

    predict_rhythm(
        mat_file
    )