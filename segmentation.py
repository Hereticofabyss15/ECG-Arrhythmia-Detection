"""
Module 3: Beat Segmentation

Takes R-peak locations and slices a fixed-size window around each one,
turning a continuous ECG recording into a set of labeled beat examples.
"""
import numpy as np


def segment_beats(signal, r_peaks, fs, before_ms=250, after_ms=250):
    """
    Extract a window around each R-peak.

    before_ms / after_ms: how much signal (in ms) to take before/after
    the peak. Default 250ms + 250ms = 500ms window, which comfortably
    covers a QRS complex at typical heart rates.

    Returns:
        beats: list of 1D numpy arrays (may vary by 1 sample near edges)
        valid_peaks: the r_peaks that had a full window available
    """
    before = int((before_ms / 1000) * fs)
    after = int((after_ms / 1000) * fs)

    beats = []
    valid_peaks = []
    for p in r_peaks:
        lo = p - before
        hi = p + after
        if lo < 0 or hi > len(signal):
            continue  # skip beats too close to the recording edges
        beats.append(signal[lo:hi])
        valid_peaks.append(p)

    return beats, np.array(valid_peaks)


def label_beats(valid_peaks, ann_samples, ann_symbols, tolerance=36):
    """
    Assign a label to each segmented beat by matching it to the nearest
    annotation sample within `tolerance` samples (default ~100ms at 360Hz).

    ann_samples: array of annotation sample indices (from the .atr file)
    ann_symbols: array of annotation symbols, same length as ann_samples

    Returns a list of labels (one per beat), 'U' (unknown) if no match found.
    """
    labels = []
    for p in valid_peaks:
        diffs = np.abs(ann_samples - p)
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance:
            labels.append(ann_symbols[idx])
        else:
            labels.append('U')
    return labels


# AAMI 5-class grouping, as used by the mondejar/ecg-classification repo
AAMI_CLASSES = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',            # Normal
    'A': 'SVEB', 'a': 'SVEB', 'J': 'SVEB', 'S': 'SVEB',           # Supraventricular ectopic
    'V': 'VEB', 'E': 'VEB',                                       # Ventricular ectopic
    'F': 'F',                                                     # Fusion
    'P': 'Q', '/': 'Q', 'f': 'Q', 'u': 'Q',                       # Unknown/paced
}


def map_to_aami(labels):
    """Map raw MIT-BIH annotation symbols to the 5 AAMI classes."""
    return [AAMI_CLASSES.get(l, 'Q') for l in labels]


if __name__ == "__main__":
    from preprocessing import preprocess_ecg
    from qrs_detection import pan_tompkins_detect

    fs = 360
    t = np.arange(0, 10, 1 / fs)
    heartbeats = np.zeros_like(t)
    for beat_time in np.arange(0.5, 10, 0.8):
        heartbeats += np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
    raw = heartbeats + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * np.random.randn(len(t))

    filtered = preprocess_ecg(raw, fs)
    peaks = pan_tompkins_detect(filtered, fs)
    beats, valid_peaks = segment_beats(filtered, peaks, fs)

    print(f"Segmented {len(beats)} beats, each of length {len(beats[0])} samples")

    # fake annotations for the self-test (pretend every beat is Normal)
    fake_labels = label_beats(valid_peaks, valid_peaks, ['N'] * len(valid_peaks))
    print("Labels:", fake_labels)
    print("AAMI classes:", map_to_aami(fake_labels))
