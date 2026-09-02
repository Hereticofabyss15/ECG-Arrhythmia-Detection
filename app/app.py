import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "final_xgboost")

MODEL_PATH = os.path.join(MODEL_DIR, "final_xgboost_model.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECG Arrhythmia Classification System",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    return model, encoder, feature_names


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_INFO = {
    "N": "Normal Beat",
    "SVEB": "Supraventricular Ectopic Beat",
    "VEB": "Ventricular Ectopic Beat",
    "F": "Fusion Beat",
    "Q": "Unknown / Other"
}


# ============================================================
# HEADER
# ============================================================

st.title("❤️ ECG Arrhythmia Classification System")

st.markdown("### Patient-Independent ECG Beat Classification")

st.write(
    "This application uses 54 pre-extracted DSP-based ECG features "
    "and a trained XGBoost classifier to classify individual ECG beats "
    "into five categories."
)

st.info(
    "Input requirement: Upload a CSV containing all 54 extracted "
    "ECG/DSP features for each beat. Additional columns are allowed "
    "and will be ignored."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, encoder, feature_names = load_model()

except Exception as e:

    st.error("Failed to load the trained model.")

    st.exception(e)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input Features",
            len(feature_names)
        )

    with col2:
        st.metric(
            "Output Classes",
            len(encoder.classes_)
        )

    with col3:
        st.metric(
            "Model",
            "XGBoost"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV containing the 54 extracted ECG/DSP features per beat",
    type=["csv"]
)


# ============================================================
# PROCESS INPUT
# ============================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------------
        # READ CSV
        # --------------------------------------------------------

        df = pd.read_csv(uploaded_file)

        st.success(
            f"File loaded successfully: "
            f"{df.shape[0]:,} rows × {df.shape[1]:,} columns"
        )


        # --------------------------------------------------------
        # CHECK REQUIRED FEATURES
        # --------------------------------------------------------

        missing_features = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]


        # --------------------------------------------------------
        # MISSING FEATURE COLUMNS
        # --------------------------------------------------------

        if missing_features:

            st.error(
                f"Prediction cannot be performed because "
                f"{len(missing_features)} required feature(s) are missing."
            )

            st.write("### Missing Features")

            st.code(
                "\n".join(missing_features)
            )

            st.warning(
                "The model requires all 54 features used during training. "
                "Please provide the missing feature columns. "
                "The application will not guess or artificially create "
                "missing features."
            )

            st.stop()


        # --------------------------------------------------------
        # SELECT EXACT MODEL FEATURES
        # --------------------------------------------------------

        X = df[feature_names].copy()


        # --------------------------------------------------------
        # CHECK NON-NUMERIC VALUES
        # --------------------------------------------------------

        non_numeric_columns = X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric_columns:

            st.error(
                "Some required feature columns contain non-numeric data."
            )

            st.write("### Problematic Columns")

            st.code(
                "\n".join(non_numeric_columns)
            )

            st.warning(
                "All 54 model input features must contain numeric values."
            )

            st.stop()


        # --------------------------------------------------------
        # CHECK MISSING VALUES
        # --------------------------------------------------------

        missing_values = X.isnull().sum()

        missing_value_columns = missing_values[
            missing_values > 0
        ]


        if len(missing_value_columns) > 0:

            st.error(
                "Missing values (NaN) were found in the input data."
            )

            missing_table = pd.DataFrame({
                "Feature": missing_value_columns.index,
                "Missing Values": missing_value_columns.values
            })

            st.dataframe(
                missing_table,
                use_container_width=True
            )

            st.warning(
                "Please remove or properly preprocess missing values "
                "before running the classifier."
            )

            st.stop()


        # --------------------------------------------------------
        # VALID INPUT
        # --------------------------------------------------------

        st.success(
            "All 54 required features are present and valid."
        )


        # --------------------------------------------------------
        # CLASSIFICATION BUTTON
        # --------------------------------------------------------

        if st.button(
            "🔍 Classify ECG Beats",
            type="primary"
        ):

            with st.spinner(
                "Running ECG classification..."
            ):

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                predictions_encoded = model.predict(X)

                probabilities = model.predict_proba(X)

                predictions = encoder.inverse_transform(
                    predictions_encoded.astype(int)
                )


            # ====================================================
            # RESULTS
            # ====================================================

            st.markdown("---")

            st.header("Classification Results")


            # ----------------------------------------------------
            # RESULTS DATAFRAME
            # ----------------------------------------------------

            results = pd.DataFrame()

            results["Beat"] = np.arange(
                1,
                len(predictions) + 1
            )

            results["Prediction"] = predictions

            results["Class"] = [
                CLASS_INFO.get(
                    label,
                    "Unknown"
                )
                for label in predictions
            ]

            results["Model Confidence (%)"] = (
                np.max(probabilities, axis=1) * 100
            ).round(2)


            # ----------------------------------------------------
            # SUMMARY
            # ----------------------------------------------------

            total_beats = len(predictions)

            class_counts = pd.Series(
                predictions
            ).value_counts()


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Total Beats",
                    f"{total_beats:,}"
                )


            with col2:

                most_common = class_counts.index[0]

                st.metric(
                    "Most Common Class",
                    most_common
                )


            with col3:

                average_confidence = (
                    np.max(probabilities, axis=1).mean()
                    * 100
                )

                st.metric(
                    "Average Model Confidence",
                    f"{average_confidence:.2f}%"
                )


            # ----------------------------------------------------
            # CLASS DISTRIBUTION
            # ----------------------------------------------------

            st.subheader(
                "Predicted Class Distribution"
            )

            distribution = (
                pd.Series(predictions)
                .value_counts()
                .rename_axis("Class")
                .reset_index(name="Count")
            )

            distribution["Class Name"] = distribution[
                "Class"
            ].map(CLASS_INFO)

            distribution["Percentage (%)"] = (
                distribution["Count"]
                / total_beats
                * 100
            ).round(2)

            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True
            )


            # ----------------------------------------------------
            # BEAT-BY-BEAT PREDICTIONS
            # ----------------------------------------------------

            st.subheader(
                "Beat-by-Beat Predictions"
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


            # ----------------------------------------------------
            # PROBABILITY DISTRIBUTION
            # ----------------------------------------------------

            st.subheader(
                "Class Probability Distribution"
            )

            probability_df = pd.DataFrame(
                probabilities,
                columns=encoder.classes_
            )

            probability_df.index = np.arange(
                1,
                len(probability_df) + 1
            )

            probability_df.index.name = "Beat"

            st.line_chart(
                probability_df
            )


            # ----------------------------------------------------
            # INDIVIDUAL BEAT ANALYSIS
            # ----------------------------------------------------

            st.subheader(
                "Individual Beat Analysis"
            )

            selected_beat = st.number_input(
                "Select Beat Number",
                min_value=1,
                max_value=total_beats,
                value=1,
                step=1
            )

            selected_index = selected_beat - 1


            selected_prediction = predictions[
                selected_index
            ]

            selected_probabilities = probabilities[
                selected_index
            ]


            st.write(
                f"### Prediction: **{selected_prediction}**"
            )

            st.write(
                f"**{CLASS_INFO.get(selected_prediction, 'Unknown')}**"
            )

            st.write(
                f"Model Confidence: "
                f"**{np.max(selected_probabilities) * 100:.2f}%**"
            )


            # ----------------------------------------------------
            # INDIVIDUAL PROBABILITY TABLE
            # ----------------------------------------------------

            individual_probability_df = pd.DataFrame({

                "Class": encoder.classes_,

                "Class Name": [
                    CLASS_INFO.get(
                        c,
                        "Unknown"
                    )
                    for c in encoder.classes_
                ],

                "Probability (%)": (
                    selected_probabilities * 100
                ).round(4)

            })

            st.dataframe(
                individual_probability_df,
                use_container_width=True,
                hide_index=True
            )


            # ----------------------------------------------------
            # CONFIDENCE DISCLAIMER
            # ----------------------------------------------------

            st.caption(
                "Model confidence represents the XGBoost predicted "
                "probability for the selected class. It should not be "
                "interpreted as clinical certainty or a medical diagnosis."
            )


            # ----------------------------------------------------
            # ADD PROBABILITIES TO RESULTS
            # ----------------------------------------------------

            for i, class_name in enumerate(
                encoder.classes_
            ):

                results[
                    f"Probability_{class_name}"
                ] = (
                    probabilities[:, i] * 100
                ).round(4)


            # ----------------------------------------------------
            # DOWNLOAD RESULTS
            # ----------------------------------------------------

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="ecg_predictions.csv",
                mime="text/csv"
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the uploaded file."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ECG Arrhythmia Classification System | "
    "DSP Features + XGBoost | "
    "Research/Educational Use Only — Not a Clinical Diagnostic Tool"
)