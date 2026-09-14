# Hierarchical ECG Arrhythmia Detection and Classification

### Patient-independent ECG rhythm classification using DSP, multi-lead ECG features, and machine learning

An end-to-end ECG analysis system that combines **digital signal processing (DSP)**, **beat-level machine learning**, **RR/HRV analysis**, and **12-lead ECG features** for hierarchical cardiac rhythm classification.

The system operates at two complementary levels:

- **Beat level:** ECG beats are segmented and represented using 54 DSP-based features, followed by an XGBoost classifier that estimates beat-level morphology classes.
- **Rhythm level:** RR/HRV features, beat-derived statistics, and multi-lead ECG features are fused into a 99-feature representation and classified using a Random Forest model.

The final rhythm classifier distinguishes seven rhythm categories:

**SB, SR, ST, AFIB, AF, SA, and SVT**

The project focuses on patient-independent evaluation at the beat level, record-level evaluation at the rhythm level, severe class-imbalance analysis, feature ablation, model comparison, multi-lead analysis, error analysis, and deployment through a Streamlit application.

> **Note:** This is a research/educational project and is not intended for clinical diagnosis.

---

## Project Overview

ECG rhythm classification is an important application of biomedical signal processing and machine learning.

Rather than relying on a single feature family or a single ECG beat, this project uses a **hierarchical analysis pipeline** that combines beat morphology, beat-to-beat timing, and information from all 12 ECG leads.

The overall system is:

**12-Lead ECG → Preprocessing → QRS/R-Peak Detection → Beat-Level Analysis + Rhythm-Level Analysis → Feature Fusion → Random Forest Rhythm Classification**

The system consists of three major components:

1. Beat-level analysis
2. Rhythm-level analysis
3. Multi-lead feature fusion

---

## Hierarchical System Architecture

```text
                         12-Lead ECG
                              │
                              ▼
                    ┌───────────────────┐
                    │   Preprocessing   │
                    │                   │
                    │ Bandpass filtering│
                    │ Powerline removal │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  R-Peak / QRS     │
                    │     Detection     │
                    └─────────┬─────────┘
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
                 ▼            ▼             ▼
        Beat-Level Path   Rhythm Path   Multi-Lead Path
                 │            │             │
                 ▼            ▼             ▼
        Beat Segmentation  RR Intervals   12-Lead
                 │            │           Statistics
                 ▼            ▼             │
          54 DSP Features  RR / HRV       │
                 │          Features       │
                 ▼            │             │
             XGBoost          │             │
                 │            │             │
                 ▼            │             │
          Beat Predictions   │             │
                 │            │             │
                 ▼            ▼             │
          Beat-Derived   Rhythm Features   │
             Features          │            │
                 │             │            │
                 └─────────────┼────────────┘
                               ▼
                    ┌────────────────────┐
                    │ 99-Feature Fusion  │
                    │                    │
                    │ RR / HRV           │
                    │ Beat-derived       │
                    │ Multi-lead ECG     │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │   Random Forest    │
                    │  Rhythm Classifier │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Rhythm Prediction  │
                    │                    │
                    │ SB / SR / ST       │
                    │ AFIB / AF / SA     │
                    │ SVT                │
                    └────────────────────┘
```

---

## 1. Beat-Level Analysis

The first level of the system analyzes individual ECG beats.

The beat-level pipeline is:

```text
ECG Signal
    │
    ▼
Preprocessing
    │
    ▼
QRS / R-Peak Detection
    │
    ▼
Beat Segmentation
    │
    ▼
54 DSP Features
    │
    ▼
XGBoost
    │
    ▼
Beat-Level Prediction
```

### Beat-Level Classes

The beat-level classifier distinguishes five beat categories:

| Label | Description |
|---|---|
| N | Normal Beat |
| SVEB | Supraventricular Ectopic Beat |
| VEB | Ventricular Ectopic Beat |
| F | Fusion Beat |
| Q | Unknown / Other |

The beat classifier is used as an intermediate component of the final rhythm pipeline.

---

## 2. Beat-Level Dataset

The beat-level system was developed using:

- MIT-BIH Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database

The final extracted beat-level feature dataset contains:

- 296,525 ECG beats
- 126 patients
- 54 DSP features
- 5 beat classes

### Beat Class Distribution

| Class | Beats |
|---|---|
| N | 252,890 |
| VEB | 17,176 |
| SVEB | 14,978 |
| Q | 10,656 |
| F | 825 |

The dataset is highly imbalanced, with the F class representing approximately 0.28% of all beats.

Raw datasets and large generated feature files are intentionally excluded from the repository.

---

## 3. Beat-Level DSP Feature Extraction

The beat classifier uses 54 engineered ECG features.

### 3.1 Time-Domain Features

Examples include:

- Mean
- Standard deviation
- Minimum
- Maximum
- Skewness
- Kurtosis
- Signal energy
- Amplitude-related measurements

These features describe the statistical and morphological properties of each ECG beat.

### 3.2 RR / Heart-Rate Features

Examples include:

- RR interval
- Next RR interval
- Mean RR
- Heart rate
- Local heart rate
- RR deviation
- RR deviation ratio
- Next RR ratio

These features capture temporal relationships between consecutive heartbeats.

### 3.3 QRS / Morphological Features

Examples include:

- R-peak amplitude
- R-peak position
- R-peak-to-mean ratio
- QRS width
- Slope statistics
- Maximum/minimum slope
- Mean absolute slope

These features capture ECG morphology around the QRS complex.

### 3.4 Spectral Features

Examples include:

- Spectral energy
- Spectral bandwidth
- Frequency-domain energy characteristics

### 3.5 Wavelet Features

Wavelet decomposition is used to capture ECG characteristics at different frequency scales.

Examples include:

- Wavelet energy
- Wavelet energy ratios
- Wavelet standard deviation
- Wavelet entropy
- Multi-level wavelet coefficients

---

## 4. Beat-Level Machine Learning Model

The beat-level classifier uses an XGBoost multiclass model.

### Final Configuration

```text
max_depth        = 7
learning_rate    = 0.03
n_estimators     = 600
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 2
gamma            = 0.1
reg_alpha        = 0
reg_lambda       = 1
```

Class weighting using inverse class frequency was used to reduce the effect of severe class imbalance.

The final XGBoost model is trained on the complete beat-level feature dataset.

---

## 5. Beat-Level Evaluation

### Patient-Grouped Cross-Validation

To reduce patient-level data leakage, the beat-level evaluation uses:

- 5-fold patient-grouped cross-validation

Beats belonging to the same patient are kept within the same fold.

### Cross-Validation Performance

The final fixed XGBoost configuration achieved:

| Metric | Mean ± Std |
|---|---|
| Macro F1 | 0.622 ± 0.066 |

The variation between folds is strongly influenced by the extremely small and uneven distribution of the F class across patients.

### Out-of-Fold Performance

Pooling predictions from all five validation folds gives:

| Metric | Score |
|---|---|
| Accuracy | 0.8831 |
| Balanced Accuracy | 0.6888 |
| Macro F1 | 0.6183 |

### Per-Class F1

| Class | F1 |
|---|---|
| N | 0.94 |
| SVEB | 0.49 |
| VEB | 0.69 |
| F | 0.07 |
| Q | 0.90 |

The model performs strongly on N, VEB and Q, while SVEB is more challenging and F remains the most difficult class.

---

## 6. Beat-Level ROC-AUC and Precision-Recall Analysis

One-vs-rest ROC and Precision-Recall curves were generated using patient-grouped out-of-fold predictions.

### ROC-AUC

| Class | ROC-AUC |
|---|---|
| N | 0.9491 |
| SVEB | 0.9105 |
| VEB | 0.9792 |
| F | 0.9167 |
| Q | 0.9959 |
| **Macro** | **0.9503** |

### PR-AUC

| Class | PR-AUC |
|---|---|
| N | 0.9884 |
| SVEB | 0.4790 |
| VEB | 0.8571 |
| F | 0.0344 |
| Q | 0.9542 |
| **Macro** | **0.6626** |

The difference between ROC-AUC and PR-AUC for F highlights the effect of extreme class imbalance.

---

## 7. Beat-Level Error Analysis

Out-of-fold predictions were analyzed to understand where the classifier fails.

### Most Common Misclassifications

| Actual → Predicted | Samples |
|---|---|
| N → SVEB | 13,524 |
| N → VEB | 8,851 |
| SVEB → N | 3,443 |
| N → F | 2,058 |
| SVEB → VEB | 1,722 |
| VEB → SVEB | 1,416 |
| Q → N | 939 |
| N → Q | 854 |

The largest non-F confusion occurs between N, SVEB and VEB, indicating that some ectopic beats have feature characteristics overlapping with other classes.

The F class is especially difficult to generalize because of its extreme rarity and concentration among a small number of patients.

---

## 8. F-Class Analysis

The F class is the primary limitation of the beat-level system.

Only:

- 825 / 296,525 = 0.278%

of all beats belong to F.

Only 23 patients contain F beats, and the class is highly concentrated among a small number of patients.

This creates a difficult patient-independent learning problem.

The model can learn useful signal characteristics from the available F examples, but those characteristics are difficult to generalize to unseen patients.

This is reflected by:

- F-class PR-AUC = 0.0344
- F-class OOF F1 = 0.07

Therefore, overall accuracy should not be interpreted as evidence of equally strong performance across all beat classes.

---

## 9. Beat-Level Feature Ablation Study

Different groups of DSP features were evaluated to determine their contribution.

| Feature Group | Validation Macro F1 |
|---|---|
| Time-domain | 0.4115 |
| RR / Heart Rate | 0.4923 |
| Wavelet | 0.2760 |
| Spectral | 0.2364 |
| **All Features** | **0.6503** |

The combined feature set performed best in the ablation experiment.

Among individual feature groups, RR/heart-rate features were the strongest standalone group, highlighting the importance of beat-to-beat temporal information.

The ablation experiment used a separate patient split, so its absolute scores should not be directly compared with the final 5-fold cross-validation scores.

---

## 10. Transition to Rhythm-Level Classification

Beat-level classification alone does not fully describe cardiac rhythm.

A rhythm is characterized not only by individual beat morphology, but also by:

- Heart rate
- RR interval variability
- Beat-to-beat irregularity
- Ectopic beat burden
- Rhythm transitions
- Information distributed across multiple ECG leads

Therefore, the project extends the beat-level system into a second-stage rhythm classifier.

The rhythm-level pipeline is:

```text
12-Lead ECG
     │
     ├───────────────┐
     │               │
     ▼               ▼
Lead II         All 12 Leads
     │               │
     ▼               ▼
R-Peak Detection   Multi-Lead
     │             Features
     ▼               │
RR / HRV Features    │
     │               │
     ▼               │
Beat-Level XGBoost   │
     │               │
     ▼               │
Beat-Derived         │
Features             │
     │               │
     └───────┬───────┘
             ▼
      Feature Fusion
             │
             ▼
       99 Features
             │
             ▼
      Random Forest
             │
             ▼
     Rhythm Prediction
```

---

## 11. Rhythm-Level Dataset

The rhythm-level system uses the large-scale PhysioNet ECG Arrhythmia dataset.

The dataset contains:

- 12-lead ECG recordings
- 500 Hz sampling frequency
- 10-second recordings
- 5000 samples per lead
- Professional diagnostic labels

Each selected recording contains:

- 12 leads × 5000 samples

The project uses seven rhythm classes:

| Label | Rhythm |
|---|---|
| SB | Sinus Bradycardia |
| SR | Sinus Rhythm |
| ST | Sinus Tachycardia |
| AFIB | Atrial Fibrillation |
| AF | Atrial Flutter |
| SA | Sinus Arrhythmia |
| SVT | Supraventricular Tachycardia |

**Important:** In the source dataset, AF corresponds to atrial flutter, while AFIB corresponds to atrial fibrillation. They are treated as separate rhythm classes.

A balanced subset was constructed for model development.

Final dataset:

- 3,115 recordings
- 445 recordings per rhythm class
- 7 rhythm classes
- 3,115 unique record IDs
- No record overlap between train, validation and test sets

---

## 12. R-Peak Detection

The rhythm pipeline uses a model-free adaptive R-peak detection approach inspired by recent robust R-peak detection research.

The implementation uses:

- Butterworth bandpass filtering
- Powerline-noise suppression
- Windowed processing
- Per-window normalization
- Mass-ratio-based outlier scoring
- Adaptive candidate localization
- Dynamic refractory handling
- Physiological RR-interval postprocessing

The implementation was developed specifically for the 500 Hz, 10-second rhythm dataset.

The detector is used to derive the RR intervals required for rhythm-level analysis.

---

## 13. RR / HRV Feature Extraction

RR intervals are used to construct rhythm-level temporal features.

The extracted feature family includes:

- Mean RR
- Median RR
- Minimum RR
- Maximum RR
- Mean heart rate
- Minimum heart rate
- Maximum heart rate
- SDNN
- RMSSD
- SDSD
- Mean absolute RR difference
- Maximum absolute RR difference
- pNN50
- pNN100
- RR coefficient of variation
- RR IQR
- RR entropy

These features capture the temporal structure and irregularity of the rhythm.

---

## 14. Beat-Derived Rhythm Features

The beat-level XGBoost model is used as an intermediate feature generator.

For each rhythm recording, beat-level predictions are aggregated into rhythm-level statistics.

The final beat-derived feature family includes:

- Beat probability for N
- Beat probability for SVEB
- Beat probability for VEB
- Beat probability for F
- Beat probability for Q
- Ectopic burden
- Number of beats
- Maximum VEB run
- Beat transition rate
- Beat-class entropy

These features provide information about the composition and temporal organization of beat-level classifications within the recording.

---

## 15. Multi-Lead ECG Features

Because rhythm information may be distributed differently across ECG leads, the rhythm classifier also uses statistics extracted independently from all 12 ECG leads.

For each lead, the following features are extracted:

- Mean
- Standard deviation
- RMS
- Peak-to-peak amplitude
- Signal energy
- Dominant frequency

This produces:

- 12 leads × 6 features = 72 multi-lead features

The leads include:

- I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

---

## 16. Final 99-Feature Fusion

The final rhythm classifier combines three feature families:

| Feature Family | Features |
|---|---|
| RR / HRV | 17 |
| Beat-derived | 10 |
| Multi-lead ECG | 72 |
| **Total** | **99** |

One exact duplicate feature, `rr_diff_std`, was removed because it was identical to `sdsd`.

The final representation therefore contains 99 model inputs.

---

## 17. Rhythm Model Comparison

Four machine learning approaches were evaluated using the final fused representation:

- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost

Five-fold cross-validation was performed on the combined training and validation data.

### 5-Fold Cross-Validation

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.6607 ± 0.0180 | 0.6608 ± 0.0180 | 0.6553 ± 0.0191 |
| SVM | 0.6706 ± 0.0269 | 0.6707 ± 0.0271 | 0.6551 ± 0.0281 |
| Random Forest | 0.7344 ± 0.0244 | 0.7345 ± 0.0247 | 0.7317 ± 0.0244 |
| XGBoost | 0.7348 ± 0.0100 | 0.7349 ± 0.0103 | 0.7308 ± 0.0100 |

Random Forest was selected as the final rhythm classifier because it achieved the highest mean cross-validation Macro F1.

The difference between Random Forest and XGBoost is small, but Random Forest achieved the highest mean Macro F1.

---

## 18. Multi-Lead Feature Contribution

The addition of multi-lead features improved rhythm classification performance compared with the earlier RR/HRV + beat-derived representation.

Five-fold cross-validation results:

- **28-feature rhythm representation:** Macro F1 = 0.6968
- **99-feature multi-lead fusion:** Macro F1 = 0.7263
- **Improvement:** +0.0295 Macro F1

This supports the usefulness of combining temporal rhythm information with information distributed across multiple ECG leads.

Feature-family importance in the final Random Forest was approximately:

| Feature Family | Relative Importance |
|---|---|
| RR / HRV | 47.63% |
| Multi-lead ECG | 35.93% |
| Beat-derived | 16.45% |

The model therefore relies heavily on rhythm dynamics while also benefiting substantially from multi-lead ECG information.

---

## 19. Final Rhythm Model

The selected final rhythm classifier is:

**Random Forest**

Configuration:

```text
n_estimators = 400
class_weight = balanced
random_state = 42
n_jobs = -1
```

The model was trained on the complete training + validation set:

- Training + Validation = 2,647 recordings

The final test set remained completely untouched during model selection.

---

## 20. Final Test Evaluation

The final model was evaluated once on the locked test set containing:

- 468 ECG recordings

### Overall Performance

| Metric | Score |
|---|---|
| Accuracy | 0.7179 |
| Balanced Accuracy | 0.7171 |
| Macro F1 | 0.7192 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AF | 0.3607 | 0.3333 | 0.3465 |
| AFIB | 0.4568 | 0.5522 | 0.5000 |
| SA | 0.7619 | 0.7164 | 0.7385 |
| SB | 0.8052 | 0.9254 | 0.8611 |
| SR | 0.9492 | 0.8358 | 0.8889 |
| ST | 0.8525 | 0.7761 | 0.8125 |
| SVT | 0.8939 | 0.8806 | 0.8872 |

The model performs particularly well on SB, SR, ST and SVT.

AF and AFIB remain substantially more difficult to distinguish using the current feature representation.

---

## 21. Final Test Confusion Matrix

```text
              Predicted
             AF AFIB SA SB SR ST SVT

Actual AF    22  34  3  1  1  4  1
Actual AFIB  23  37  6  1  0  0  0
Actual SA     4   2 48 11  2  0  0
Actual SB     0   1  4 62  0  0  0
Actual SR     4   1  1  2 56  2  1
Actual ST     4   5  1  0  0 52  5
Actual SVT    4   1  0  0  0  3 59
```

The largest challenge is distinguishing AF from AFIB, where the two classes show substantial overlap in the current feature representation.

---

## 22. AF vs AFIB Analysis

AF and AFIB were analyzed separately because their temporal statistics are highly similar.

The rhythm-level RR/HRV features show considerable overlap between the two classes.

For example, mean heart rate was approximately:

- AF: 95.74 BPM
- AFIB: 96.69 BPM

and the number of detected beats was also very similar.

Multi-lead analysis showed stronger class separation in several leads, particularly:

- III
- aVF
- II
- aVL
- V4
- V5

This motivated the use of multi-lead ECG features in the final model.

However, the current representation does not explicitly model atrial activity, which remains an important limitation for AF/AFIB discrimination.

---

## 23. Approaches Explored During Development

Several additional approaches were investigated during development.

These experiments were used to understand the problem and inform the final system design, but they are not included in the deployment repository.

### PCA-Based Atrial Activity Features

PCA-based QRST cancellation and atrial-activity features were explored using selected ECG leads and frequency-domain characteristics.

The approach was motivated by research showing that atrial fibrillatory activity can provide useful information for distinguishing atrial rhythms.

However, the batch extraction process produced substantial incompleteness and class-dependent failure patterns.

The approach was therefore not included in the final model.

### Adaptive Singular Value Cancellation

An ASVC-based atrial activity extraction approach was also investigated.

The method was unstable across recordings and parameter settings, with inconsistent signal reduction and unreliable behavior across leads.

It was therefore rejected from the final pipeline.

These experiments demonstrated an important practical limitation:

> More specialized signal-processing features do not automatically provide more robust classification performance.

The final system therefore uses a simpler and more reproducible feature representation based on RR/HRV dynamics, beat-derived information, and multi-lead ECG statistics.

---

## 24. Feature Importance

The final Random Forest model identifies several highly important rhythm features.

Examples include:

- median RR
- mean RR
- maximum RR
- mean heart rate
- minimum heart rate
- beat probability for SVEB
- RR IQR
- minimum RR
- RR coefficient of variation
- ectopic burden
- number of beats
- SDNN
- mean absolute RR difference
- RR entropy
- pNN50
- maximum heart rate
- RMSSD

The most important feature families are:

1. RR / HRV → strongest contribution
2. Multi-lead ECG → second strongest
3. Beat-derived → complementary contribution

Feature importance indicates model usage but does not establish clinical causality or clinical significance.

---

## 25. Streamlit Application

A Streamlit application is included for interactive inference.

The final application accepts a raw 12-lead ECG `.mat` file.

Expected ECG format:

- 12 leads × 5000 samples
- Sampling frequency = 500 Hz
- Duration = 10 seconds

The application performs:

```text
Raw 12-Lead ECG
       │
       ▼
Preprocessing
       │
       ▼
R-Peak Detection
       │
       ├───────────────┐
       │               │
       ▼               ▼
 Beat-Level       RR / HRV
 XGBoost          Features
       │               │
       ▼               │
Beat-Derived          │
Features              │
       │               │
       └───────┬───────┘
               │
        Multi-Lead Features
               │
               ▼
         99-Feature Vector
               │
               ▼
       Final Random Forest
               │
               ▼
        Rhythm Prediction
```

The application displays:

- 12-lead ECG waveform
- Lead II R-peaks
- Estimated mean heart rate
- Predicted rhythm
- Rhythm probabilities
- Beat-level prediction summary
- Processing information
- Research/educational disclaimer

---

## 26. Example Inference

For a representative sinus-rhythm recording:

**Input:**
- 12-lead ECG
- 500 Hz
- 10 seconds

**Detected R-peaks:** 10

**Estimated mean HR:** 59.95 BPM

**Predicted rhythm:** SR

**Model probability:** SR → 84.00%

This is an inference demonstration only and should not be interpreted as an accuracy measurement.

The official model performance is reported using the untouched test set described above.

---

## 27. Running the Project

### Requirements

- Python 3.12+
- NumPy
- SciPy
- Pandas
- Scikit-learn
- PyWavelets
- pymof
- XGBoost
- Matplotlib
- Streamlit

Install the exact runtime dependencies using:

```bash
pip install -r requirements.txt
```

---

## 28. Running Raw ECG Inference

The main inference script is:

```text
rhythm_inference.py
```

It expects a `.mat` ECG file containing:

- `val`

with shape:

```text
(12, 5000)
```

Example:

```bash
python rhythm_inference.py "path\to\record.mat"
```

The script performs the complete hierarchical inference pipeline and reports:

- R-peak count
- Mean heart rate
- Final rhythm prediction
- Rhythm probabilities
- Beat-level prediction summary

---

## 29. Running the Streamlit Application

From the project root:

```bash
python -m streamlit run app\app.py
```

The application will start locally and provide a browser interface for uploading ECG recordings.

---

## 30. Repository Structure

```text
ECG-Arrhythmia-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── final_rhythm_confusion_matrix.csv
│   ├── final_rhythm_model_comparison.csv
│   ├── final_rhythm_test_results.csv
│   └── rhythm_dataset_manifest.csv
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
│   ├── final_rhythm_rf_model.pkl
│   ├── final_rhythm_feature_names.json
│   │
│   └── final_xgboost/
│       ├── final_xgboost_model.json
│       ├── feature_names.json
│       ├── label_encoder.pkl
│       ├── model_config.json
│       └── final_feature_importance.csv
│
├── feature_extraction.py
├── preprocessing.py
├── qrs_detection.py
├── rhythm_features.py
├── rhythm_inference.py
├── rhythm_qrs.py
├── segmentation.py
├── requirements.txt
├── README.md
└── .gitignore
```

Large ECG datasets, generated feature datasets, intermediate experimental outputs, and temporary files are excluded using `.gitignore`.

---

## 31. Final Model Files

The repository contains the trained models required for inference.

### Beat-Level Model

```text
models/final_xgboost/
```

Contains:

- XGBoost model
- Feature names
- Label encoder
- Model configuration
- Feature importance

### Rhythm-Level Model

```text
models/final_rhythm_rf_model.pkl
models/final_rhythm_feature_names.json
```

The rhythm model expects the final 99-feature representation.

---

## 32. Reproducibility

The repository contains the primary preprocessing, feature extraction, model comparison, evaluation and inference components used during development.

Large datasets and generated feature files are excluded from GitHub because of their size.

The final trained models and selected evaluation artifacts are included.

The locked rhythm test set was not used during model selection.

---

## 33. Evaluation Considerations

The beat-level and rhythm-level tasks use different datasets and evaluation strategies.

### Beat-Level System

Uses:

- MIT-BIH Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database
- Patient-grouped cross-validation

### Rhythm-Level System

Uses:

- PhysioNet ECG Arrhythmia dataset
- Record-level train/validation/test split
- Five-fold cross-validation for model selection
- One final evaluation on an untouched test set

The rhythm dataset does not provide an explicit patient identifier suitable for patient-level grouping in the selected subset.

Therefore, the rhythm results should be described as record-level evaluation, not as strict patient-independent evaluation.

---

## 34. Limitations

The current system has several important limitations:

- The beat-level F class is extremely rare and highly concentrated among a small number of patients.
- AF and AFIB remain difficult to distinguish using the current feature representation.
- The rhythm-level feature representation does not explicitly model atrial activity or fibrillatory-wave morphology.
- The R-peak detector is a project implementation inspired by recent model-free adaptive R-peak detection research; it should not be considered a clinical-grade detector.
- The rhythm dataset does not provide a suitable explicit patient identifier for patient-level grouping in this project.
- The final rhythm model is evaluated at the record level rather than using a strict patient-independent split.
- The final rhythm accuracy is approximately 71.8%, meaning substantial classification errors remain.
- Model probabilities should not be interpreted as calibrated clinical confidence.
- The system has not undergone external clinical validation.
- The system has not been evaluated as a medical device.
- The project should be considered a research/educational prototype rather than a clinical diagnostic system.

---

## 35. Future Work

Potential future improvements include:

- Increasing the diversity of minority rhythm examples
- Improving AF vs AFIB discrimination
- Incorporating atrial-specific ECG representations
- Evaluating additional external ECG datasets
- Testing calibrated probabilities
- Improving R-peak detection robustness
- Exploring temporal deep-learning architectures
- Extending the feature representation beyond simple multi-lead statistics
- Adding additional rhythm classes
- Improving the Streamlit interface
- Cloud deployment and scalable inference

---

## 36. Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- PyWavelets
- pymof
- XGBoost
- Matplotlib
- Streamlit
- Git / GitHub

---

## Author

**Prashant Kumar Moharana**

Electronics and Communication Engineering
VIT Chennai

---

## Disclaimer

This project is intended strictly for research and educational purposes.

It is not a medical device and should not be used to make clinical decisions, diagnose patients, or replace professional medical evaluation.

The predictions generated by the system represent machine-learning outputs from research datasets and should not be interpreted as clinical diagnoses.
