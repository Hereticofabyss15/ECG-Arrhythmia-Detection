"""
Module 1: ECG Preprocessing
- Baseline wander removal (median filter, two-stage)
- Bandpass filtering (Butterworth, 0.5-40 Hz)
"""
import numpy as np
from scipy.signal import medfilt, butter, filtfilt


def remove_baseline_wander(signal, fs):
    """
    Two-stage median filter baseline removal.
    Classic approach: ~200ms window then ~600ms window, subtract from signal.
    """
    win1 = int(0.2 * fs)
    if win1 % 2 == 0:
        win1 += 1
    win2 = int(0.6 * fs)
    if win2 % 2 == 0:
        win2 += 1

    baseline1 = medfilt(signal, kernel_size=win1)
    baseline2 = medfilt(baseline1, kernel_size=win2)

    return signal - baseline2


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=2):
    """Butterworth bandpass filter, zero-phase (filtfilt)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def preprocess_ecg(signal, fs):
    """Full preprocessing pipeline: baseline removal -> bandpass filter."""
    no_baseline = remove_baseline_wander(signal, fs)
    filtered = bandpass_filter(no_baseline, fs)
    return filtered


if __name__ == "__main__":
    # Quick self-test with synthetic ECG-like signal
    fs = 360
    t = np.arange(0, 10, 1 / fs)
    heartbeats = np.zeros_like(t)
    for beat_time in np.arange(0.5, 10, 0.8):
        heartbeats += np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    noise = 0.02 * np.random.randn(len(t))
    raw = heartbeats + baseline_wander + noise

    filtered = preprocess_ecg(raw, fs)
    print("Preprocessing self-test OK")
    print("Raw signal std:", np.std(raw))
    print("Filtered signal std:", np.std(filtered))
