"""
Module 2: QRS / R-peak Detection (Pan-Tompkins algorithm)

Pipeline: bandpass filter -> derivative -> squaring ->
          moving-window integration -> adaptive thresholding -> R-peaks
"""
import numpy as np
from scipy.signal import butter, filtfilt


def pan_tompkins_detect(signal, fs):
    """
    Returns indices of detected R-peaks in `signal`.
    `signal` should already be preprocessed (baseline removed).
    """
    # 1. Bandpass filter (5-15 Hz, standard for QRS emphasis)
    nyq = 0.5 * fs
    b, a = butter(1, [5 / nyq, 15 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)

    # 2. Derivative (5-point derivative approximation)
    derivative = np.diff(filtered, prepend=filtered[0])

    # 3. Squaring
    squared = derivative ** 2

    # 4. Moving-window integration (~150ms window)
    win_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win_size) / win_size, mode='same')

    # 5. Adaptive thresholding + peak finding
    r_peaks = _find_peaks_adaptive(integrated, fs)

    # 6. Refine: snap each detected peak to the true local max in the
    #    ORIGINAL signal within a small window (integration shifts peaks)
    refined_peaks = _refine_peaks(signal, r_peaks, fs)

    return refined_peaks


def _find_peaks_adaptive(integrated_signal, fs, refractory_ms=200):
    """Simple adaptive-threshold peak picker with a refractory period
    to avoid double-counting a single QRS complex."""
    threshold = 0.3 * np.max(integrated_signal)
    min_distance = int((refractory_ms / 1000) * fs)

    peaks = []
    i = 1
    n = len(integrated_signal)
    while i < n - 1:
        if integrated_signal[i] > threshold and \
           integrated_signal[i] > integrated_signal[i - 1] and \
           integrated_signal[i] >= integrated_signal[i + 1]:
            if not peaks or (i - peaks[-1]) > min_distance:
                peaks.append(i)
            i += min_distance
        else:
            i += 1
    return np.array(peaks)


def _refine_peaks(signal, rough_peaks, fs, window_ms=75):
    """Snap rough peak locations to the true local maximum in the
    original (preprocessed) signal — corrects the delay introduced
    by filtering/integration."""
    window = int((window_ms / 1000) * fs)
    refined = []
    for p in rough_peaks:
        lo = max(0, p - window)
        hi = min(len(signal), p + window)
        local_max_idx = lo + np.argmax(signal[lo:hi])
        refined.append(local_max_idx)
    return np.array(sorted(set(refined)))


if __name__ == "__main__":
    from preprocessing import preprocess_ecg

    fs = 360
    t = np.arange(0, 10, 1 / fs)
    heartbeats = np.zeros_like(t)
    true_beat_times = np.arange(0.5, 10, 0.8)
    for beat_time in true_beat_times:
        heartbeats += np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    noise = 0.02 * np.random.randn(len(t))
    raw = heartbeats + baseline_wander + noise

    filtered = preprocess_ecg(raw, fs)
    peaks = pan_tompkins_detect(filtered, fs)

    print(f"True number of beats: {len(true_beat_times)}")
    print(f"Detected R-peaks: {len(peaks)}")
    print("Detected peak times (s):", np.round(peaks / fs, 2))
