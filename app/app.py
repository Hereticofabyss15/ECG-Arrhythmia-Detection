import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rhythm_inference import (
    load_ecg,
    predict_rhythm_from_ecg
)


# ================================================================
# CONFIGURATION
# ================================================================

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


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)


# ================================================================
# TITLE
# ================================================================

st.title(
    "Hierarchical ECG Arrhythmia Detection"
)

st.markdown(
    """
    **DSP-based ECG analysis with beat-level and rhythm-level
    machine learning**
    """
)

st.divider()


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("ECG Input")

    uploaded_file = st.file_uploader(
        "Upload a 12-lead ECG waveform",
        type=["mat"]
    )

    st.markdown(
        """
        **Supported format**

        `.mat` containing a `val` array with:

        - 12 ECG leads
        - 500 Hz sampling rate
        - 10 seconds duration
        - 5000 samples per lead
        """
    )

    st.divider()

    st.caption(
        "Research / decision-support prototype. "
        "Not intended for clinical diagnosis."
    )


# ================================================================
# WAIT FOR FILE
# ================================================================

if uploaded_file is None:

    st.info(
        "Upload a 12-lead ECG `.mat` file from the sidebar "
        "to begin analysis."
    )

    st.stop()


# ================================================================
# SAVE UPLOADED FILE TEMPORARILY
# ================================================================

temp_dir = PROJECT_DIR / "app" / "temp"

temp_dir.mkdir(
    parents=True,
    exist_ok=True
)

temp_path = (
    temp_dir /
    uploaded_file.name
)

with open(
    temp_path,
    "wb"
) as f:

    f.write(
        uploaded_file.getbuffer()
    )


# ================================================================
# LOAD ECG
# ================================================================

try:

    ecg = load_ecg(
        temp_path
    )

except Exception as e:

    st.error(
        f"Could not load ECG: {e}"
    )

    st.stop()


# ================================================================
# FILE INFORMATION
# ================================================================

st.subheader("ECG Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Leads",
        ecg.shape[0]
    )

with col2:

    st.metric(
        "Samples / Lead",
        ecg.shape[1]
    )

with col3:

    st.metric(
        "Sampling Rate",
        f"{FS} Hz"
    )

with col4:

    st.metric(
        "Duration",
        f"{ecg.shape[1] / FS:.1f} s"
    )


# ================================================================
# ECG WAVEFORM
# ================================================================

st.subheader("12-Lead ECG Waveform")

time = np.arange(
    ecg.shape[1]
) / FS


fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 22),
    sharex=True
)


for i, lead in enumerate(LEADS):

    axes[i].plot(
        time,
        ecg[i],
        linewidth=0.8
    )

    axes[i].set_ylabel(
        lead,
        rotation=0,
        labelpad=25
    )

    axes[i].grid(
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Time (seconds)"
)

fig.suptitle(
    "12-Lead ECG",
    fontsize=16
)

fig.tight_layout(
    rect=[0, 0, 1, 0.99]
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ================================================================
# RUN ANALYSIS
# ================================================================

st.divider()

run_analysis = st.button(
    "Run ECG Analysis",
    type="primary",
    use_container_width=True
)


if run_analysis:

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    with st.spinner(
        "Processing ECG and running the hierarchical model..."
    ):

        try:

            results = predict_rhythm_from_ecg(
                ecg,
                display_input=uploaded_file.name
            )

        except Exception as e:

            st.error(
                f"ECG analysis failed: {e}"
            )

            st.stop()


    # ------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------

    prediction = results[
        "prediction"
    ]

    probability_table = results[
        "probabilities"
    ]

    r_peaks = results[
        "r_peaks"
    ]

    mean_hr = results[
        "mean_hr"
    ]

    beat_predictions = results[
        "beat_predictions"
    ]


    # ============================================================
    # MAIN RESULT
    # ============================================================

    st.divider()

    st.subheader(
        "Rhythm Analysis Result"
    )

    result_col1, result_col2 = st.columns(
        [1, 2]
    )

    with result_col1:

        st.metric(
            "Predicted Rhythm",
            prediction
        )

    with result_col2:

        st.metric(
            "Estimated Mean Heart Rate",
            f"{mean_hr:.2f} BPM"
        )


    # ============================================================
    # PROBABILITY DISTRIBUTION
    # ============================================================

    st.subheader(
        "Rhythm Probability"
    )

    probability_display = (
        probability_table.copy()
    )

    probability_display[
        "Probability"
    ] = (
        probability_display[
            "Probability"
        ] * 100
    ).round(2)

    probability_display = (
        probability_display.rename(
            columns={
                "Probability":
                    "Probability (%)"
            }
        )
    )

    st.dataframe(
        probability_display,
        hide_index=True,
        use_container_width=True
    )


    # ============================================================
    # R-PEAK DETECTION
    # ============================================================

    st.subheader(
        "R-Peak Detection"
    )

    st.write(
        f"Detected **{len(r_peaks)} R-peaks** "
        f"in the 10-second ECG."
    )

    lead_ii = ecg[1]

    fig_r, ax_r = plt.subplots(
        figsize=(14, 4)
    )

    ax_r.plot(
        time,
        lead_ii,
        linewidth=0.8,
        label="Lead II"
    )

    peak_times = (
        np.asarray(r_peaks) / FS
    )

    peak_values = (
        lead_ii[
            np.asarray(r_peaks)
        ]
    )

    ax_r.scatter(
        peak_times,
        peak_values,
        s=35,
        zorder=3,
        label="Detected R-peaks"
    )

    ax_r.set_xlabel(
        "Time (seconds)"
    )

    ax_r.set_ylabel(
        "Amplitude"
    )

    ax_r.set_title(
        "Lead II with Detected R-Peaks"
    )

    ax_r.grid(
        alpha=0.25
    )

    ax_r.legend()

    fig_r.tight_layout()

    st.pyplot(
        fig_r,
        use_container_width=True
    )

    plt.close(fig_r)


    # ============================================================
    # BEAT-LEVEL SUMMARY
    # ============================================================

    st.subheader(
        "Beat-Level Classification"
    )

    beat_counts = (
        pd.Series(
            beat_predictions
        )
        .value_counts()
    )

    beat_labels = [
        "N",
        "SVEB",
        "VEB",
        "F",
        "Q"
    ]

    beat_cols = st.columns(
        len(beat_labels)
    )

    for col, label in zip(
        beat_cols,
        beat_labels
    ):

        with col:

            st.metric(
                label,
                int(
                    beat_counts.get(
                        label,
                        0
                    )
                )
            )


    # ============================================================
    # MODEL PIPELINE
    # ============================================================

    st.divider()

    st.subheader(
        "Analysis Pipeline"
    )

    st.markdown(
        """
        **12-lead ECG**
        → **R-peak detection**
        → **RR/HRV features**

        **Lead II beats**
        → **54 DSP/morphological features**
        → **Beat-level XGBoost**

        **All 12 leads**
        → **72 multi-lead signal features**

        **Combined 99-feature representation**
        → **Final Random Forest**
        → **Rhythm classification**
        """
    )


    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.warning(
        "This system is a research/decision-support prototype "
        "and has not been validated for clinical diagnosis."
    )