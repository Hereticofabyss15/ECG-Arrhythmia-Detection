import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from pymof import MOF


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = r"D:\Downloads\ECG_Arrhythmia_Dataset"

FS = 500

# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

FILTER_ORDER = 5
LOWCUT = 0.5
HIGHCUT = 40.0

POWERLINE_FREQUENCY = 50.0

# ------------------------------------------------------------
# MOW-ECG windowing
# ------------------------------------------------------------

WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)

OVERLAP_SECONDS = 1.0
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * FS)

STRIDE_SAMPLES = WINDOW_SAMPLES - OVERLAP_SAMPLES

# ------------------------------------------------------------
# MOW-ECG parameters
# ------------------------------------------------------------

MOF_THRESHOLD = 2.0

# Initial refractory period estimate.
#
# The paper tunes this parameter using grid search.
# We use 0.40 s as an initial global value rather than
# using record-specific ground-truth-optimized values.
INITIAL_REFRACTORY = 0.25

# Minimum RR interval used during post-processing.
MIN_RR_INTERVAL = 0.30

# Dynamic search region around the MOF candidate.
#
# The paper specifies that the search-window width is
# proportional to the initial refractory period but does
# not give the exact multiplier in the published text.
SEARCH_HALF_WIDTH_FACTOR = 0.50


# ============================================================
# Signal loading
# ============================================================

def load_lead_ii(mat_file):

    data = loadmat(mat_file)

    signal = data["val"]

    return signal[1].astype(float)


# ============================================================
# Preprocessing
# ============================================================

def butterworth_filter(signal):

    nyquist = FS / 2.0

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    sos = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def powerline_filter(signal):

    # One period of a 50 Hz signal.
    kernel_size = max(
        1,
        int(round(FS / POWERLINE_FREQUENCY))
    )

    kernel = np.ones(kernel_size) / kernel_size

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


def preprocess_ecg(signal):

    filtered = butterworth_filter(signal)

    filtered = powerline_filter(filtered)

    return filtered


# ============================================================
# Z-score normalization
# ============================================================

def zscore_signal(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-12:
        return np.zeros_like(signal)

    return (signal - mean) / std


# ============================================================
# MOF calculation
# ============================================================

def calculate_window_mof(window):

    normalized = zscore_signal(window)

    # pymof expects:
    #
    # shape = (n_samples, n_features)
    #
    # Our ECG is one-dimensional, therefore:
    #
    # (n_samples, 1)

    data = normalized.reshape(-1, 1)

    model = MOF()

    model.fit(
        data,
        Window=len(data),
        KeepMassRatio=False
    )

    return model.decision_scores_


# ============================================================
# Overlapping-window MOF
# ============================================================

def calculate_mof_scores(signal):

    n = len(signal)

    scores = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)

    start = 0

    while start < n:

        end = min(
            start + WINDOW_SAMPLES,
            n
        )

        window = signal[start:end]

        # Very short final windows are not useful for MOF.
        if len(window) < max(100, WINDOW_SAMPLES // 2):
            break

        window_scores = calculate_window_mof(window)

        scores[start:end] += window_scores
        counts[start:end] += 1.0

        if end == n:
            break

        start += STRIDE_SAMPLES

    valid = counts > 0

    scores[valid] /= counts[valid]

    return scores


# ============================================================
# Dynamic search window
# ============================================================

def get_search_window(candidate, signal_length):

    half_width = int(
        SEARCH_HALF_WIDTH_FACTOR
        * INITIAL_REFRACTORY
        * FS
    )

    start = max(
        0,
        candidate - half_width
    )

    end = min(
        signal_length,
        candidate + half_width + 1
    )

    return start, end


# ============================================================
# Iterative MOW-ECG candidate identification
# ============================================================

def detect_r_peaks_mow_ecg(
    filtered_signal,
    mof_scores
):

    working_scores = mof_scores.copy()
    working_signal = filtered_signal.copy()

    detected_peaks = []

    signal_length = len(filtered_signal)

    refractory_samples = int(
        INITIAL_REFRACTORY * FS
    )

    # The paper estimates an upper bound on the number
    # of possible candidates using signal length / refractory.
    max_iterations = int(
        np.ceil(
            signal_length
            / refractory_samples
        )
    )

    for _ in range(max_iterations):

        if len(working_scores) == 0:
            break

        candidate = int(
            np.argmax(working_scores)
        )

        candidate_score = working_scores[candidate]

        # MOW-ECG threshold from the paper.
        if candidate_score < MOF_THRESHOLD:
            break

        # ----------------------------------------------------
        # Dynamic search region around MOF candidate
        # ----------------------------------------------------

        start, end = get_search_window(
            candidate,
            signal_length
        )

        local_signal = working_signal[
            start:end
        ]

        if len(local_signal) == 0:
            break

        # ----------------------------------------------------
        # Morphological validation
        #
        # The paper uses the maximum absolute ECG amplitude
        # within the dynamic search region.
        # ----------------------------------------------------

        local_peak = int(
            np.argmax(
                np.abs(local_signal)
            )
        )

        r_peak = start + local_peak

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        detected_peaks.append(r_peak)

        # ----------------------------------------------------
        # Dynamic refractory reinforcement
        #
        # Nullify both MOF scores and ECG amplitudes around
        # the detected R-peak.
        # ----------------------------------------------------

        blank_start = max(
            0,
            r_peak - refractory_samples
        )

        blank_end = min(
            signal_length,
            r_peak + refractory_samples + 1
        )

        working_scores[
            blank_start:blank_end
        ] = -np.inf

        working_signal[
            blank_start:blank_end
        ] = 0.0

    if len(detected_peaks) == 0:
        return np.array([], dtype=int)

    detected_peaks = np.array(
        detected_peaks,
        dtype=int
    )

    return detected_peaks


# ============================================================
# Physiological post-processing
# ============================================================

def postprocess_r_peaks(
    r_peaks,
    min_rr_interval=MIN_RR_INTERVAL
):

    if len(r_peaks) <= 1:
        return r_peaks

    # Restore chronological order.
    r_peaks = np.sort(
        np.unique(r_peaks)
    )

    rr_intervals = (
        np.diff(r_peaks) / FS
    )

    if len(rr_intervals) == 0:
        return r_peaks

    # Paper:
    #
    # lower RR bound =
    # min(
    #     empirically derived minimum RR,
    #     0.5 * median(candidate RR intervals)
    # )
    #
    adaptive_rr_lower_bound = min(
        min_rr_interval,
        0.5 * np.median(rr_intervals)
    )

    valid_peaks = [
        r_peaks[0]
    ]

    for peak in r_peaks[1:]:

        rr = (
            peak - valid_peaks[-1]
        ) / FS

        if rr >= adaptive_rr_lower_bound:
            valid_peaks.append(peak)

    return np.array(
        valid_peaks,
        dtype=int
    )


# ============================================================
# Complete R-peak detection pipeline
# ============================================================

def detect_r_peaks(signal):

    # --------------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------------

    filtered = preprocess_ecg(
        signal
    )

    # --------------------------------------------------------
    # 2. Overlapping-window MOF
    # --------------------------------------------------------

    mof_scores = calculate_mof_scores(
        filtered
    )

    # --------------------------------------------------------
    # 3. Iterative candidate identification
    # --------------------------------------------------------

    candidates = detect_r_peaks_mow_ecg(
        filtered,
        mof_scores
    )

    # --------------------------------------------------------
    # 4. Physiological post-processing
    # --------------------------------------------------------

    r_peaks = postprocess_r_peaks(
        candidates
    )

    return (
        filtered,
        mof_scores,
        candidates,
        r_peaks
    )


# ============================================================
# Record analysis
# ============================================================

def analyze_record(mat_file):

    signal = load_lead_ii(
        mat_file
    )

    (
        filtered,
        mof_scores,
        candidates,
        r_peaks
    ) = detect_r_peaks(
        signal
    )

    if len(r_peaks) >= 2:

        rr_intervals = (
            np.diff(r_peaks) / FS
        )

        mean_rr = np.mean(
            rr_intervals
        )

        heart_rate = (
            60.0 / mean_rr
        )

    else:

        rr_intervals = np.array([])

        mean_rr = np.nan

        heart_rate = np.nan

    return (
        signal,
        filtered,
        mof_scores,
        candidates,
        r_peaks,
        rr_intervals,
        mean_rr,
        heart_rate
    )


# ============================================================
# Plot
# ============================================================

def plot_record(
    rhythm,
    mat_file,
    signal,
    filtered,
    mof_scores,
    candidates,
    r_peaks
):

    time = (
        np.arange(len(signal))
        / FS
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=True
    )

    # --------------------------------------------------------
    # Raw ECG
    # --------------------------------------------------------

    axes[0].plot(
        time,
        signal,
        linewidth=0.8,
        label="Raw Lead II"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].set_title(
        f"{rhythm} — "
        f"{os.path.basename(mat_file)}"
    )

    axes[0].legend()
    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Filtered ECG + detected R-peaks
    # --------------------------------------------------------

    axes[1].plot(
        time,
        filtered,
        linewidth=0.8,
        label="Preprocessed Lead II"
    )

    if len(r_peaks) > 0:

        axes[1].plot(
            time[r_peaks],
            filtered[r_peaks],
            "ro",
            markersize=6,
            label="Detected R-peaks"
        )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].legend()
    axes[1].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # MOF score
    # --------------------------------------------------------

    axes[2].plot(
        time,
        mof_scores,
        linewidth=0.9,
        label="MOF Score"
    )

    axes[2].axhline(
        MOF_THRESHOLD,
        linestyle="--",
        label="MOF threshold"
    )

    if len(candidates) > 0:

        axes[2].plot(
            time[candidates],
            mof_scores[candidates],
            "ko",
            markersize=5,
            label="MOW-ECG candidates"
        )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "MOF"
    )

    axes[2].legend()
    axes[2].grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    rhythms = [
        "SB",
        "SR",
        "ST",
        "AFIB",
        "AF",
        "SA",
        "SVT"
    ]

    print()
    print(
        "MOW-ECG R-Peak Detection"
    )
    print("=" * 70)

    print(
        f"Sampling frequency : {FS} Hz"
    )

    print(
        f"Window size        : "
        f"{WINDOW_SAMPLES} samples "
        f"({WINDOW_SECONDS:.1f} s)"
    )

    print(
        f"Window overlap     : "
        f"{OVERLAP_SAMPLES} samples "
        f"({OVERLAP_SECONDS:.1f} s)"
    )

    print(
        f"MOF threshold      : "
        f"{MOF_THRESHOLD}"
    )

    print(
        f"Initial refractory : "
        f"{INITIAL_REFRACTORY:.2f} s"
    )

    print(
        f"Minimum RR         : "
        f"{MIN_RR_INTERVAL:.2f} s"
    )

    print("=" * 70)

    for rhythm in rhythms:

        folder = os.path.join(
            DATASET_PATH,
            rhythm
        )

        mat_files = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.endswith(".mat")
            ]
        )

        if not mat_files:
            continue

        mat_file = os.path.join(
            folder,
            mat_files[0]
        )

        (
            signal,
            filtered,
            mof_scores,
            candidates,
            r_peaks,
            rr_intervals,
            mean_rr,
            heart_rate
        ) = analyze_record(
            mat_file
        )

        print(
            f"{rhythm:<6} | "
            f"{mat_files[0]:<12} | "
            f"Candidates: "
            f"{len(candidates):<3} | "
            f"R-peaks: "
            f"{len(r_peaks):<3} | "
            f"HR: "
            f"{heart_rate:.1f} BPM"
        )
        if rhythm == "AFIB":
          print("R-peak times (s):")
          print(np.round(np.array(r_peaks) / FS, 3))

        plot_record(
            rhythm,
            mat_file,
            signal,
            filtered,
            mof_scores,
            candidates,
            r_peaks
        )

    print("=" * 70)


if __name__ == "__main__":
    main()