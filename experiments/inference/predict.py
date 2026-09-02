import os
import json
import argparse
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "final_xgboost"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.json"
)

ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading model...")

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    with open(FEATURE_FILE, "r") as f:
        feature_names = json.load(f)

    label_encoder = joblib.load(ENCODER_FILE)

    print("Model loaded successfully.")
    print(f"Features required: {len(feature_names)}")

    return model, feature_names, label_encoder


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ECG Arrhythmia Prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing ECG features"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, feature_names, label_encoder = load_model()

    # --------------------------------------------------------
    # Load input CSV
    # --------------------------------------------------------

    print(f"\nLoading input file: {args.input}")

    df = pd.read_csv(args.input)

    print(f"Input samples: {len(df)}")

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        print("\nERROR: Missing features:")

        for feature in missing_features:
            print(f"  - {feature}")

        return

    # --------------------------------------------------------
    # Select features in EXACT training order
    # --------------------------------------------------------

    X = df[feature_names]

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    predicted_labels = label_encoder.inverse_transform(
        predictions
    )

    classes = label_encoder.classes_

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ECG ARRHYTHMIA PREDICTIONS")
    print("=" * 70)

    for i in range(len(df)):

        print(f"\nBeat {i + 1}")

        print(
            f"Prediction: {predicted_labels[i]}"
        )

        print(
            f"Confidence: "
            f"{probabilities[i].max() * 100:.2f}%"
        )

        print("\nClass probabilities:")

        for j, class_name in enumerate(classes):

            print(
                f"  {class_name:5s}: "
                f"{probabilities[i][j] * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result_df = df.copy()

    result_df["predicted_label"] = predicted_labels

    for j, class_name in enumerate(classes):

        result_df[
            f"prob_{class_name}"
        ] = probabilities[:, j]

    output_file = os.path.join(
        os.path.dirname(args.input),
        "predictions.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"Predictions saved to:\n{output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()