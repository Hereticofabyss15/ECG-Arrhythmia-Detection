"""
Generate the standard report figures:
1. Raw ECG
2. Filtered ECG
3. Detected R-peaks
4. Segmented beats overlay
5. Confusion matrix (per model)
6. Model accuracy comparison bar chart
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns


def plot_raw_vs_filtered(t, raw, filtered, out_path='results/filtered_ecg.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, raw, color='tab:red')
    axes[0].set_title('Raw ECG')
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(t, filtered, color='tab:blue')
    axes[1].set_title('Filtered ECG (baseline removed + bandpassed)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_r_peaks(t, filtered, peaks, out_path='results/qrs_detection.png'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, filtered, color='tab:blue', label='ECG')
    ax.scatter(t[peaks], filtered[peaks], color='red', marker='x', s=80, label='Detected R-peaks')
    ax.set_title('QRS / R-peak Detection')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_beat_segments(beats, out_path='results/beat_segments.png', n_show=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in beats[:n_show]:
        ax.plot(b, alpha=0.7)
    ax.set_title(f'Segmented Heartbeats (first {min(n_show, len(beats))} shown)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, model_name, out_path=None):
    if out_path is None:
        out_path = f'results/confusion_matrix_{model_name.lower()}.png'
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison_df, out_path='results/model_comparison.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(
        kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import joblib
    import pandas as pd
    from preprocessing import preprocess_ecg
    from qrs_detection import pan_tompkins_detect
    from segmentation import segment_beats
    from train_models import patient_level_split, prepare_xy
    from evaluate import evaluate_model, compare_models

    fs = 360
    t = np.arange(0, 10, 1 / fs)
    heartbeats = np.zeros_like(t)
    for beat_time in np.arange(0.5, 10, 0.8):
        heartbeats += np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
    raw = heartbeats + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * np.random.randn(len(t))
    filtered = preprocess_ecg(raw, fs)
    peaks = pan_tompkins_detect(filtered, fs)
    beats, valid_peaks = segment_beats(filtered, peaks, fs)

    plot_raw_vs_filtered(t, raw, filtered)
    plot_r_peaks(t, filtered, peaks)
    plot_beat_segments(beats)

    # Reuse the synthetic multi-class dataset + trained models for the
    # confusion matrix / comparison plots
    np.random.seed(42)
    n_patients = 10
    rows = []
    classes = ['N', 'SVEB', 'VEB', 'F', 'Q']
    class_weights = [0.7, 0.1, 0.12, 0.05, 0.03]
    for pid in range(n_patients):
        labels = np.random.choice(classes, size=200, p=class_weights)
        for lbl in labels:
            base = {'N': 0, 'SVEB': 1, 'VEB': 2, 'F': 3, 'Q': 4}[lbl]
            row = {f'f{i}': base * 0.5 + np.random.randn() for i in range(24)}
            row['label'] = lbl
            row['patient_id'] = pid
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith('f')]
    _, test_df = patient_level_split(df, list(range(0, 7)), list(range(7, 10)))

    le = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_test, y_test, _ = prepare_xy(test_df, feature_cols, label_encoder=le)

    models = {
        'SVM': (joblib.load('models/svm.pkl'), True),
        'RandomForest': (joblib.load('models/randomforest.pkl'), False),
        'XGBoost': (joblib.load('models/xgboost.pkl'), False),
    }
    results = {}
    for name, (model, needs_scaling) in models.items():
        results[name] = evaluate_model(model, X_test, y_test, le.classes_,
                                        needs_scaling=needs_scaling, scaler=scaler)
        plot_confusion_matrix(results[name]['confusion_matrix'], le.classes_, name)

    comparison = compare_models(results)
    plot_model_comparison(comparison)

    print("All figures saved to results/")
