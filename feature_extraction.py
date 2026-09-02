"""
Enhanced ECG Feature Extraction

Extracts:
- Time-domain features
- ECG morphology features
- R-peak features
- RR interval/context features
- Frequency-domain features
- Wavelet features
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(
    beat,
    peak_index=None,
    fs=360,
    rr_interval=None,
    prev_rr=None,
    next_rr=None
):
    """
    Extract comprehensive ECG features from one beat.

    Parameters
    ----------
    beat : array-like
        Segmented ECG beat.

    peak_index : int, optional
        R-peak location inside the beat.

    fs : float
        ECG sampling frequency.

    rr_interval : float, optional
        Current RR interval in seconds.

    prev_rr : float, optional
        Previous RR interval in seconds.

    next_rr : float, optional
        Next RR interval in seconds.

    Returns
    -------
    dict
        Extracted ECG features.
    """

    beat = np.asarray(beat, dtype=float)

    if len(beat) == 0:
        return {}

    # --------------------------------------------------------
    # Replace invalid ECG samples
    # --------------------------------------------------------

    beat = np.nan_to_num(
        beat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    features = {}

    # ========================================================
    # 1. TIME-DOMAIN FEATURES
    # ========================================================

    features["mean"] = np.mean(beat)
    features["std"] = np.std(beat)
    features["max"] = np.max(beat)
    features["min"] = np.min(beat)
    features["rms"] = np.sqrt(np.mean(beat ** 2))
    features["ptp"] = np.ptp(beat)

    # scipy can occasionally produce NaN for constant signals
    sk = skew(beat)
    ku = kurtosis(beat)

    features["skewness"] = (
        float(sk) if np.isfinite(sk) else 0.0
    )

    features["kurtosis"] = (
        float(ku) if np.isfinite(ku) else 0.0
    )

    # ========================================================
    # 2. R-PEAK FEATURES
    # ========================================================

    if peak_index is not None:

        peak_pos = int(peak_index)

        if peak_pos < 0 or peak_pos >= len(beat):
            peak_pos = int(np.argmax(np.abs(beat)))

    else:

        peak_pos = int(np.argmax(np.abs(beat)))

    features["r_peak_amplitude"] = beat[peak_pos]

    features["r_peak_position"] = (
        peak_pos / max(len(beat) - 1, 1)
    )

    features["r_peak_to_mean"] = (
        beat[peak_pos] - np.mean(beat)
    )

    # ========================================================
    # 3. RR INTERVAL / HEART RATE FEATURES
    # ========================================================

    # Current RR
    if (
        rr_interval is None
        or not np.isfinite(rr_interval)
        or rr_interval <= 0
    ):
        current_rr = np.nan
    else:
        current_rr = float(rr_interval)

    # Previous RR
    if (
        prev_rr is None
        or not np.isfinite(prev_rr)
        or prev_rr <= 0
    ):
        previous_rr = np.nan
    else:
        previous_rr = float(prev_rr)

    # Next RR
    if (
        next_rr is None
        or not np.isfinite(next_rr)
        or next_rr <= 0
    ):
        following_rr = np.nan
    else:
        following_rr = float(next_rr)

    features["rr_interval"] = current_rr
    features["prev_rr"] = previous_rr
    features["next_rr"] = following_rr

    # Heart rate
    if np.isfinite(current_rr):
        features["heart_rate"] = 60.0 / current_rr
    else:
        features["heart_rate"] = np.nan

    # Mean RR from available neighboring intervals
    rr_values = [
        x for x in [
            previous_rr,
            current_rr,
            following_rr
        ]
        if np.isfinite(x)
    ]

    if len(rr_values) > 0:

        mean_rr = np.mean(rr_values)

    else:

        mean_rr = np.nan

    features["mean_rr"] = mean_rr

    # Local heart rate
    if np.isfinite(mean_rr) and mean_rr > 0:

        features["local_heart_rate"] = (
            60.0 / mean_rr
        )

    else:

        features["local_heart_rate"] = np.nan

    # --------------------------------------------------------
    # RR ratios
    # --------------------------------------------------------

    if (
        np.isfinite(current_rr)
        and np.isfinite(previous_rr)
        and previous_rr > 0
    ):

        features["rr_ratio"] = (
            current_rr / previous_rr
        )

    else:

        features["rr_ratio"] = np.nan

    if (
        np.isfinite(current_rr)
        and np.isfinite(following_rr)
        and following_rr > 0
    ):

        features["next_rr_ratio"] = (
            following_rr / current_rr
        )

    else:

        features["next_rr_ratio"] = np.nan

    # --------------------------------------------------------
    # RR deviation
    # --------------------------------------------------------

    if (
        np.isfinite(current_rr)
        and np.isfinite(mean_rr)
    ):

        features["rr_deviation"] = (
            current_rr - mean_rr
        )

        if mean_rr > 0:

            features["rr_deviation_ratio"] = (
                current_rr / mean_rr
            )

        else:

            features["rr_deviation_ratio"] = np.nan

    else:

        features["rr_deviation"] = np.nan
        features["rr_deviation_ratio"] = np.nan

    # ========================================================
    # 4. MORPHOLOGY FEATURES
    # ========================================================

    derivative = np.diff(beat)

    if len(derivative) > 0:

        features["max_slope"] = np.max(
            derivative
        )

        features["min_slope"] = np.min(
            derivative
        )

        features["mean_abs_slope"] = np.mean(
            np.abs(derivative)
        )

        features["slope_std"] = np.std(
            derivative
        )

    else:

        features["max_slope"] = 0.0
        features["min_slope"] = 0.0
        features["mean_abs_slope"] = 0.0
        features["slope_std"] = 0.0

    # Zero crossing rate
    signs = np.sign(beat)

    zero_crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    features["zero_crossing_rate"] = (
        zero_crossings
        / max(len(beat) - 1, 1)
    )

    # Absolute area
    features["absolute_area"] = (
        np.sum(np.abs(beat)) / fs
    )

    # Signal energy
    features["signal_energy"] = (
        np.sum(beat ** 2)
    )

    # ========================================================
    # 5. APPROXIMATE QRS WIDTH
    # ========================================================

    peak_amplitude = abs(
        beat[peak_pos]
    )

    if peak_amplitude > 0:

        threshold = 0.5 * peak_amplitude

        above_threshold = np.where(
            np.abs(beat) >= threshold
        )[0]

        if len(above_threshold) >= 2:

            qrs_width_samples = (
                above_threshold[-1]
                - above_threshold[0]
            )

            features["qrs_width"] = (
                qrs_width_samples / fs
            )

        else:

            features["qrs_width"] = 0.0

    else:

        features["qrs_width"] = 0.0

    # ========================================================
    # 6. FREQUENCY-DOMAIN FEATURES
    # ========================================================

    n = len(beat)

    if n > 1:

        frequencies = np.fft.rfftfreq(
            n,
            d=1.0 / fs
        )

        spectrum = np.abs(
            np.fft.rfft(beat)
        )

        power = spectrum ** 2

        total_power = np.sum(power)

        if total_power > 0:

            # Ignore DC when searching dominant frequency
            if len(power) > 1:

                dominant_index = (
                    np.argmax(power[1:]) + 1
                )

            else:

                dominant_index = 0

            features["dominant_freq"] = (
                frequencies[dominant_index]
            )

            spectral_centroid = (
                np.sum(
                    frequencies * power
                ) / total_power
            )

            features["spectral_centroid"] = (
                spectral_centroid
            )

            spectral_bandwidth = np.sqrt(
                np.sum(
                    (
                        frequencies
                        - spectral_centroid
                    ) ** 2 * power
                ) / total_power
            )

            features["spectral_bandwidth"] = (
                spectral_bandwidth
            )

            features["spectral_energy"] = (
                total_power
            )

        else:

            features["dominant_freq"] = 0.0
            features["spectral_centroid"] = 0.0
            features["spectral_bandwidth"] = 0.0
            features["spectral_energy"] = 0.0

    else:

        features["dominant_freq"] = 0.0
        features["spectral_centroid"] = 0.0
        features["spectral_bandwidth"] = 0.0
        features["spectral_energy"] = 0.0

    # ========================================================
    # 7. WAVELET FEATURES
    # ========================================================

    # We always create the same 5 wavelet groups.
    # If a particular level cannot be calculated,
    # its features are set to zero.

    wavelet = "db4"

    # Fixed feature groups:
    # wav_0, wav_1, wav_2, wav_3, wav_4

    wavelet_groups = range(5)

    # Initialize all wavelet features
    for i in wavelet_groups:

        features[f"wav_{i}_mean"] = 0.0
        features[f"wav_{i}_std"] = 0.0
        features[f"wav_{i}_energy"] = 0.0
        features[f"wav_{i}_energy_ratio"] = 0.0

    features["wavelet_entropy"] = 0.0

    try:

        max_level = pywt.dwt_max_level(
            len(beat),
            pywt.Wavelet(wavelet).dec_len
        )

        level = min(5, max_level)

        if level >= 1:

            coeffs = pywt.wavedec(
                beat,
                wavelet,
                level=level
            )

            energies = []

            for i, coeff in enumerate(coeffs):

                if i >= 5:
                    break

                coeff = np.asarray(
                    coeff,
                    dtype=float
                )

                energy = np.sum(
                    coeff ** 2
                )

                mean_value = np.mean(
                    coeff
                )

                std_value = np.std(
                    coeff
                )

                if not np.isfinite(
                    mean_value
                ):
                    mean_value = 0.0

                if not np.isfinite(
                    std_value
                ):
                    std_value = 0.0

                if not np.isfinite(
                    energy
                ):
                    energy = 0.0

                features[
                    f"wav_{i}_mean"
                ] = mean_value

                features[
                    f"wav_{i}_std"
                ] = std_value

                features[
                    f"wav_{i}_energy"
                ] = energy

                energies.append(
                    energy
                )

            # Pad energy list
            while len(energies) < 5:

                energies.append(0.0)

            total_wavelet_energy = np.sum(
                energies
            )

            # Energy ratios
            if total_wavelet_energy > 0:

                for i in range(5):

                    features[
                        f"wav_{i}_energy_ratio"
                    ] = (
                        energies[i]
                        / total_wavelet_energy
                    )

                # Wavelet entropy
                probabilities = (
                    np.asarray(energies)
                    / total_wavelet_energy
                )

                probabilities = probabilities[
                    probabilities > 0
                ]

                if len(probabilities) > 0:

                    features[
                        "wavelet_entropy"
                    ] = -np.sum(
                        probabilities
                        * np.log2(
                            probabilities
                        )
                    )

    except Exception:

        # Keep all initialized zero values.
        pass

    # ========================================================
    # FINAL SAFETY CHECK
    # ========================================================

    # IMPORTANT:
    # Any remaining invalid feature becomes NaN here.
    # build_dataset.py will later perform controlled
    # imputation for genuine boundary RR values.

    for key, value in features.items():

        try:

            value = float(value)

            if not np.isfinite(value):

                features[key] = np.nan

            else:

                features[key] = value

        except Exception:

            features[key] = np.nan

    return features


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    beats,
    peaks,
    fs
):
    """
    Extract features from all segmented beats.

    RR context is calculated from neighboring R peaks.
    """

    feature_rows = []

    peaks = np.asarray(
        peaks,
        dtype=float
    )

    for i, beat in enumerate(beats):

        # ----------------------------------------------------
        # Previous RR
        # ----------------------------------------------------

        prev_rr = np.nan

        if i >= 2:

            prev_rr = (
                peaks[i - 1]
                - peaks[i - 2]
            ) / fs

        # ----------------------------------------------------
        # Current RR
        # ----------------------------------------------------

        rr_interval = np.nan

        if i >= 1:

            rr_interval = (
                peaks[i]
                - peaks[i - 1]
            ) / fs

        # ----------------------------------------------------
        # Next RR
        # ----------------------------------------------------

        next_rr = np.nan

        if i + 1 < len(peaks):

            next_rr = (
                peaks[i + 1]
                - peaks[i]
            ) / fs

        # ----------------------------------------------------
        # R peak is centered in our beat segmentation
        # ----------------------------------------------------

        peak_index = len(beat) // 3

        features = extract_features(
            beat,
            peak_index=peak_index,
            fs=fs,
            rr_interval=rr_interval,
            prev_rr=prev_rr,
            next_rr=next_rr
        )

        feature_rows.append(
            features
        )

    return feature_rows


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":

    fs = 360

    t = np.arange(216) / fs

    test_beat = (
        0.1
        * np.sin(
            2 * np.pi * 1.2 * t
        )
    )

    # Simulated R peak
    test_beat[72] += 1.0

    features = extract_features(
        test_beat,
        peak_index=72,
        fs=fs,
        rr_interval=0.8,
        prev_rr=0.82,
        next_rr=0.78
    )

    df = pd.DataFrame(
        [features]
    )

    print(
        "Feature extraction self-test OK"
    )

    print(
        "Number of features:",
        len(features)
    )

    print(
        "Feature names:"
    )

    print(
        list(features.keys())
    )

    print(
        "NaN count:",
        df.isna().sum().sum()
    )