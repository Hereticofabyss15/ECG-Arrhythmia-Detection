# Hierarchical ECG Arrhythmia Detection and Classification

A DSP and machine-learning based system for hierarchical ECG analysis, combining beat-level morphology, rhythm-level dynamics, and 12-lead ECG signal characteristics.

The project evolved from a beat-level arrhythmia classifier into a complete raw-ECG rhythm classification pipeline with a Streamlit inference application.

---

## Project Overview

The system analyzes a 12-lead, 10-second ECG waveform and performs:

1. R-peak detection using a model-free adaptive method based on Mass Ratio Variance Outlier Factor (MOW-ECG).
2. RR interval and HRV feature extraction.
3. Beat-level arrhythmia classification using DSP/morphological features and XGBoost.
4. Multi-lead signal feature extraction from all 12 ECG leads.
5. Feature-level fusion of rhythm, beat-derived, and multi-lead information.
6. Final rhythm classification using a Random Forest classifier.
7. Raw ECG inference through a Streamlit application.

The final system classifies seven rhythm categories:

- Sinus Bradycardia (SB)
- Sinus Rhythm (SR)
- Sinus Tachycardia (ST)
- Atrial Fibrillation (AFIB)
- Atrial Flutter (AF)
- Sinus Arrhythmia (SA)
- Supraventricular Tachycardia (SVT)

> Note: In the selected PhysioNet ECG Arrhythmia dataset, `AF` corresponds to **atrial flutter**, while `AFIB` corresponds to **atrial fibrillation**.

---

# Final System Architecture

'
                         12-Lead ECG
                         500 Hz / 10 s
                              │
                              ▼
                    ┌────────────────────┐
                    │  R-Peak Detection  │
                    │     MOW-ECG        │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        RR / HRV        Beat-Level ML      Multi-Lead
        Features         XGBoost           Features
             │                │                │
             │          54 DSP Features       │
             │                │                │
             │                ▼                │
             │        Beat Probabilities       │
             │        + Beat Statistics        │
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    99-Feature Fusion
                              │
                              ▼
                  Final Random Forest
                              │
                              ▼
                   Rhythm Classification
                              │
                              ▼
                 Streamlit ECG Application
1. Beat-Level Arrhythmia Classification

The original stage of the project focused on classifying individual ECG beats.

Beat-Level Pipeline
ECG
 ↓
Preprocessing
 ↓
R-Peak Detection
 ↓
Beat Segmentation
 ↓
DSP / Morphological Feature Extraction
 ↓
XGBoost
 ↓
Beat Classification

The beat classifier predicts five AAMI-style categories:

N — Normal
SVEB — Supraventricular Ectopic Beat
VEB — Ventricular Ectopic Beat
F — Fusion Beat
Q — Unknown / other
Beat-Level Features

Each beat is represented using 54 features covering:

Time-domain statistics
R-peak characteristics
RR interval information
Heart-rate features
Morphological features
QRS width
Slope characteristics
Zero-crossing information
Signal area and energy
Frequency-domain features
Wavelet features
Wavelet entropy

The final beat-level XGBoost model was trained using the resulting feature representation.

Beat-Level Model

The final XGBoost model is stored in:

models/final_xgboost/

The trained model is also used during rhythm inference to generate beat-level probability information.

Rather than simply using hard beat labels, the rhythm classifier uses aggregate beat-level information such as:

Mean probability of each beat class
Ectopic burden
Number of valid beats
Maximum VEB run
Beat transition rate
Beat-class entropy

This allows beat-level information to contribute to rhythm classification without treating every beat prediction as perfectly reliable.

2. Rhythm-Level Classification

The second stage focuses on rhythm rather than individual beats.

The rhythm dataset was obtained from the PhysioNet ECG Arrhythmia Database, consisting of 12-lead, 500 Hz, 10-second ECG recordings.

The original dataset contains a large number of ECG records with diagnostic labels.

A balanced subset was constructed for this project.

Final Rhythm Dataset

After removing:

Records for which reliable R-peak detection could not be obtained
Multi-label/duplicate records that did not provide a unique rhythm label

the final balanced dataset contained:

3,115 ECG records
7 rhythm classes
445 records per class

Dataset split:

Split	Records
Train	2,180
Validation	467
Test	468
Total	3,115

The test set was kept untouched during model selection.

3. R-Peak Detection

R-peaks are detected using a model-free adaptive approach inspired by the MOW-ECG method.

The implementation uses:

Butterworth filtering
50 Hz powerline-noise suppression
Overlapping windows
Window-level Z-score normalization
Mass Ratio Variance Outlier Factor
Adaptive candidate selection
Dynamic search around candidate locations
Physiological RR-interval postprocessing

The implementation was designed to operate on 500 Hz ECG recordings.

The final detector uses:

Window length:       3 seconds
Window overlap:      1 second
MOF threshold:       2
Initial refractory:  0.25 seconds
Minimum RR interval: 0.30 seconds

The detector was tested across representative examples from all seven rhythm classes.

4. RR / HRV Features

The detected R-peaks are converted into RR intervals.

The rhythm representation contains features such as:

Mean RR
Median RR
Minimum RR
Maximum RR
Mean heart rate
Minimum heart rate
Maximum heart rate
SDNN
RMSSD
SDSD
Mean absolute RR difference
Maximum absolute RR difference
pNN50
pNN100
RR coefficient of variation
RR IQR
RR entropy

These features capture rhythm regularity, heart-rate variability, and beat-to-beat dynamics.

5. Beat-Derived Rhythm Features

The existing beat-level XGBoost classifier is integrated into the rhythm pipeline.

Each ECG is processed beat-by-beat and the resulting predictions/probabilities are aggregated into record-level features.

The final rhythm model uses:

Beat class probabilities
Ectopic burden
Number of beats
Maximum VEB run
Beat transition rate
Beat-class entropy

This provides a connection between:

Beat morphology
        ↓
Beat classifier
        ↓
Record-level rhythm representation
6. Multi-Lead ECG Features

Instead of using only Lead II, the final system also extracts signal-level information from all 12 ECG leads:

I
II
III
aVR
aVL
aVF
V1
V2
V3
V4
V5
V6

Six features are extracted from every lead:

Mean
Standard deviation
RMS
Peak-to-peak amplitude
Signal energy
Dominant frequency

Therefore:

12 leads × 6 features = 72 multi-lead features

These features provide spatial information that cannot be obtained from RR intervals alone.

7. Feature Fusion

Three information sources are combined:

Rhythm dynamics

RR / HRV features

Beat morphology

Beat-level XGBoost probability and aggregate features

Multi-lead signal information

12-lead signal statistics

The resulting representation initially contained 100 features.

One exact duplicate feature was removed:

rr_diff_std

because it was identical to:

sdsd

The final model therefore uses:

99 features
8. Model Comparison

Four models were compared using 5-fold cross-validation on the combined training and validation data.

The test set was not used for model selection.

Model	Accuracy	Balanced Accuracy	Macro F1
Logistic Regression	0.6607 ± 0.0180	0.6608 ± 0.0180	0.6553 ± 0.0191
SVM	0.6706 ± 0.0269	0.6707 ± 0.0271	0.6551 ± 0.0281
Random Forest	0.7344 ± 0.0244	0.7345 ± 0.0247	0.7317 ± 0.0244
XGBoost	0.7348 ± 0.0100	0.7349 ± 0.0103	0.7308 ± 0.0100

Random Forest was selected because it achieved the highest mean cross-validation Macro F1.

9. Final Test Evaluation

After model selection, the final Random Forest was retrained using all training and validation records:

Training records: 2,647
Features:         99
Classes:          7

The previously untouched test set was then evaluated once.

Overall Performance
Metric	Test Result
Accuracy	71.79%
Balanced Accuracy	71.71%
Macro F1	71.92%
Per-Class F1
Rhythm	F1
AF	0.3465
AFIB	0.5000
SA	0.7385
SB	0.8611
SR	0.8889
ST	0.8125
SVT	0.8872

The main remaining limitation is distinguishing:

Atrial Flutter (AF)
vs
Atrial Fibrillation (AFIB)

The sinus and tachyarrhythmia classes are considerably easier for the current feature representation.

10. Feature Importance

Feature importance was analyzed using the Random Forest trained on the training data.

The three major feature families contributed approximately:

Feature Family	Relative RF Importance
RR / HRV	47.63%
Multi-lead ECG	35.93%
Beat-derived	16.45%

The most important individual features were dominated by RR/HRV characteristics such as:

Median RR
Mean RR
Maximum RR
Mean heart rate
Minimum heart rate
RR IQR
RR CV
SDNN
Ectopic burden

Multi-lead features also contributed substantially, particularly features derived from the precordial leads.

Feature importance is interpreted as model-level importance and is not treated as evidence of physiological causality.

11. Experiments That Were Investigated but Not Included

Several approaches were explored during development.

These experiments are documented in the repository, but they are not part of the final model.

PCA-Based Atrial Activity Extraction

A PCA-based QRST cancellation approach was investigated to improve the difficult AF/AFIB distinction.

The approach attempted to:

Detect R-peaks.
Extract multi-lead beat segments.
Construct beat matrices.
Apply PCA.
Reconstruct the ventricular component.
Subtract the reconstruction.
Analyze the residual in the 3–9 Hz band.
Extract spectral and complexity features.

Candidate features included:

Dominant frequency
Spectral entropy
Organization index
Atrial-band power
Atrial-band power ratio
Residual RMS
Result

The approach was not sufficiently reliable for the complete dataset.

The extraction failed or produced incomplete features for a substantial and highly class-dependent portion of the data.

Therefore:

PCA atrial features were not included in the final 99-feature model.

Relevant experimental files include:

experiments/extract_pca_features.py
experiments/test_pca_atrial_activity.py
12. ASVC QRST Cancellation

An alternative adaptive singular-value cancellation approach was also investigated.

The goal was to remove the ventricular QRST component while preserving atrial activity.

Several cancellation configurations were tested, including different correlation thresholds and neighbor-selection strategies.

Result

The method showed unstable behavior across records and leads.

Some ECGs produced useful cancellation, while others produced severe residual artifacts and unreliable ventricular suppression.

Broader validation demonstrated that the method was not sufficiently stable for inclusion in the final pipeline.

Therefore:

ASVC atrial features were rejected and are not used by the final model.

Relevant experimental files include:

experiments/test_asvc_atrial_activity.py
experiments/validate_asvc_broader.py
13. Atrial Feature Experiments

Additional AF/AFIB feature analysis was performed to determine whether the existing 12-lead representation contained useful discriminative information.

Training-only analysis showed that several limb-lead features exhibited measurable differences between atrial flutter and atrial fibrillation.

However, the effect sizes were modest and the current broad signal statistics did not provide a sufficiently reliable atrial-specific representation.

Rather than adding unstable or weakly validated features, the final system was locked using the 99-feature representation.

14. What Is NOT Implemented

The following approaches are intentionally not part of the final system:

CNN-based ECG classification
LSTM / recurrent neural networks
End-to-end deep learning from raw ECG
PCA-based atrial features
ASVC-based atrial features
Atrial-specific QRST cancellation in the final model
Clinical diagnosis or clinical decision-making
Continuous long-term ECG monitoring
Real-time streaming ECG inference
Automatic clinical report generation

These may be possible future extensions, but they are outside the current finalized pipeline.

15. Streamlit Application

A Streamlit application is implemented for raw ECG inference.

The application accepts a:

12-lead .mat ECG waveform

with:

12 leads
500 Hz
5000 samples per lead
10 seconds

The user does not need to provide the 99 engineered features.

The application performs the complete inference pipeline automatically:

Raw ECG
 ↓
R-peak detection
 ↓
RR / HRV extraction
 ↓
Beat-level XGBoost
 ↓
12-lead feature extraction
 ↓
99-feature fusion
 ↓
Final Random Forest
 ↓
Rhythm prediction

The application currently displays:

12-lead ECG waveform
Detected R-peaks on Lead II
Predicted rhythm
Estimated mean heart rate
Rhythm probabilities
Beat-level classification summary
Analysis pipeline

The UI is intentionally kept simple at the current stage and can be redesigned independently of the inference pipeline.

16. Repository Structure
ECG-Arrhythmia-Detection/
│
├── app/
│   └── app.py
│
├── experiments/
│   ├── analyze_multilead_rf_features.py
│   ├── build_multilead_fusion_dataset.py
│   ├── build_rhythm_fusion_dataset.py
│   ├── compare_final_rhythm_models.py
│   ├── extract_beat_derived_features.py
│   ├── extract_multilead_features.py
│   ├── filter_multilead_features.py
│   ├── train_final_rhythm_model.py
│   └── validate_multilead_rf.py
│
├── models/
│   ├── final_xgboost/
│   ├── final_rhythm_rf_model.pkl
│   └── final_rhythm_feature_names.json
│
├── rhythm_features.py
├── rhythm_inference.py
├── rhythm_qrs.py
├── feature_extraction.py
├── qrs_detection.py
│
├── data/
│   └── generated datasets and experiment outputs
│
├── .gitignore
└── README.md

Large ECG datasets and generated feature datasets are excluded from version control.

17. Running the Streamlit Application

Create/activate the project virtual environment and install the required dependencies.

Then run:

..\venv\Scripts\python.exe -m streamlit run app\app.py

Upload a compatible 12-lead ECG .mat file through the application.

18. Reproducibility and Evaluation Strategy

The project follows a strict model-selection workflow:

Train + Validation
        │
        ▼
5-fold cross-validation
        │
        ▼
Model selection
        │
        ▼
Retrain selected model
        │
        ▼
Untouched test set
        │
        ▼
Final evaluation

The test set was not used for:

Feature selection
Model selection
Hyperparameter tuning
Feature engineering decisions

This separation was maintained to avoid using the final test set as a development signal.

19. Current Limitations

The main limitations of the current system are:

AF and AFIB remain difficult to distinguish.
The current multi-lead features are broad signal statistics rather than atrial-specific representations.
R-peak detection can fail on a small number of difficult ECGs.
The rhythm dataset uses 10-second recordings rather than long-term ambulatory ECG.
The system has not undergone clinical validation.
The Streamlit application is a research prototype rather than a clinical device.
20. Future Work

Possible future directions include:

More robust atrial-activity extraction
Better AF/AFIB discrimination
ECG morphology features beyond broad signal statistics
Deep learning models
Raw waveform CNN architectures
Temporal sequence models
Longer ECG recordings
External dataset validation
Real-time ECG processing
More advanced visualization and reporting

These are future directions rather than components of the current final model.

Disclaimer

This project is intended for research and educational purposes only.

It is not a medical device and should not be used for clinical diagnosis, treatment decisions, or patient management.

Author

Prashant Kumar Moharana
Electronics and Communication Engineering
VIT Chennai
