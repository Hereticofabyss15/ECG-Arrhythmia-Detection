# ECG Arrhythmia Detection

### Patient-independent ECG arrhythmia classification using DSP features and XGBoost

An end-to-end machine learning project for classifying ECG beats into five arrhythmia categories using **digital signal processing (DSP) features** and **XGBoost**.

The project focuses on patient-independent evaluation, class imbalance analysis, error analysis, ROC/PR analysis, and deployment through a Streamlit application.

> **Note:** This is a research/educational project and is not intended for clinical diagnosis.

---

## Project Overview

ECG arrhythmia detection is an important application of biomedical signal processing and machine learning.

In this project, ECG beats are transformed into a set of **54 DSP-based features**, which are then classified using an XGBoost multiclass classifier.

The system performs:

**ECG Signal → Preprocessing → Beat Segmentation → DSP Feature Extraction → XGBoost Classification → Arrhythmia Prediction**

The five classes used are:

| Label | Description |
|---|---|
| N | Normal Beat |
| SVEB | Supraventricular Ectopic Beat |
| VEB | Ventricular Ectopic Beat |
| F | Fusion Beat |
| Q | Unknown / Other |

---

## Key Features

- ECG preprocessing and filtering
- QRS detection and beat segmentation
- Extraction of **54 time-domain, RR-interval, spectral, and wavelet features**
- Patient-independent model evaluation
- Stratified Group K-Fold cross-validation
- XGBoost multiclass classification
- Class imbalance analysis
- Feature ablation study
- Error and confusion-matrix analysis
- ROC-AUC and Precision-Recall analysis
- Final model trained on the complete dataset
- Feature-level inference pipeline
- Streamlit web application
- GitHub-ready reproducible project structure

---

## System Architecture

'
                    ECG Signal
                        │
                        ▼
              ┌───────────────────┐
              │  Preprocessing    │
              │ Baseline Removal  │
              │   Bandpass Filter │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │   QRS Detection   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Beat Segmentation │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ 54 DSP Features   │
              │                   │
              │ Time-domain       │
              │ RR / Heart Rate   │
              │ Spectral          │
              │ Wavelet           │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │     XGBoost       │
              │  Multiclass Model │
              └─────────┬─────────┘
                        │
                        ▼
             ┌────────────────────┐
             │ Arrhythmia Class   │
             │ N / SVEB / VEB / F │
             │       / Q          │
             └────────────────────┘
Dataset
The project uses ECG recordings from:
- MIT-BIH Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database
The final extracted feature dataset contains:
- 296,525 ECG beats
- 126 patients
- 54 DSP features
- 5 classes
Class Distribution
Class	Beats
N	252,890
VEB	17,176
SVEB	14,978
Q	10,656
F	825


The dataset is highly imbalanced, with the F class representing only approximately 0.28% of all beats.
Raw datasets and the large feature CSV are intentionally excluded from this repository.
DSP Feature Extraction
The classifier uses 54 engineered ECG features.
Feature Categories
1. Time-Domain Features
Examples include:
- Mean
- Standard deviation
- Minimum
- Maximum
- Skewness
- Kurtosis
- Signal energy
- Amplitude-related measurements
2. RR / Heart-Rate Features
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
3. QRS / Morphological Features
Examples include:
- R-peak amplitude
- R-peak position
- R-peak-to-mean ratio
- QRS width
- Slope statistics
- Maximum/minimum slope
- Mean absolute slope
4. Spectral Features
Examples include:
- Spectral energy
- Spectral bandwidth
- Frequency-domain energy characteristics
5. Wavelet Features
Wavelet decomposition is used to capture ECG characteristics at different frequency scales.
Examples include:
- Wavelet energy
- Wavelet energy ratios
- Wavelet standard deviation
- Wavelet entropy
- Multi-level wavelet coefficients
Machine Learning Model
The final classifier is an XGBoost multiclass model.
Final Configuration
max_depth        = 7
learning_rate    = 0.03
n_estimators     = 600
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 2
gamma            = 0.1
reg_alpha        = 0
reg_lambda       = 1
Class weighting using inverse class frequency was used to reduce the effect of severe class imbalance.
Evaluation
Patient-Independent Cross-Validation
To reduce patient-level data leakage, evaluation uses:
StratifiedGroupKFold with 5 folds
where patient ID is used as the grouping variable.
This ensures that beats from the same patient are not simultaneously used for training and validation within a fold.
5-Fold Results
Metric	Mean ± Std
Accuracy	0.8831 ± 0.0488
Balanced Accuracy	0.6755 ± 0.0585
Macro Precision	0.6094 ± 0.0633
Macro Recall	0.6755 ± 0.0585
Macro F1	0.6220 ± 0.0656


The primary cross-validation estimate is therefore:
Macro F1 = 0.622 ± 0.066
The variation between folds is largely influenced by the extremely small and uneven distribution of the F class across patients.
Out-of-Fold Performance
Pooling predictions from all five validation folds gives:
Metric	Score
Accuracy	0.8831
Balanced Accuracy	0.6888
Macro Precision	0.5813
Macro Recall	0.6888
Macro F1	0.6183


Per-Class Performance
Class	Precision	Recall	F1
N	0.98	0.90	0.94
SVEB	0.39	0.65	0.49
VEB	0.58	0.87	0.69
F	0.05	0.14	0.07
Q	0.91	0.88	0.90


The model performs strongly on N, VEB and Q, while SVEB is more challenging and F remains the most difficult class.
ROC-AUC and Precision-Recall Analysis
One-vs-rest ROC and Precision-Recall curves were generated using patient-independent out-of-fold predictions.
ROC-AUC
Class	ROC-AUC
N	0.9491
SVEB	0.9105
VEB	0.9792
F	0.9167
Q	0.9959
Macro	0.9503


PR-AUC
Class	PR-AUC
N	0.9884
SVEB	0.4790
VEB	0.8571
F	0.0344
Q	0.9542
Macro	0.6626


The difference between ROC-AUC and PR-AUC for F highlights the effect of extreme class imbalance.
Error Analysis
Out-of-fold predictions were analyzed to understand where the classifier fails.
Most Common Misclassifications
Actual → Predicted	Samples
N → SVEB	13,524
N → VEB	8,851
SVEB → N	3,443
N → F	2,058
SVEB → VEB	1,722
VEB → SVEB	1,416
Q → N	939
N → Q	854


The largest non-F confusion occurs between N, SVEB and VEB, indicating that some ectopic beats have feature characteristics overlapping with other classes.
For F beats specifically:
- 60.36% were predicted as N
- 24.61% were predicted as VEB
- 13.82% were correctly classified as F
F-Class Analysis
The F class is the primary limitation of the current system.
Only:
825 / 296,525 = 0.278%
of all beats belong to F.
More importantly, the F class is highly concentrated among a small number of patients:
- Only 23 patients contain F beats.
- The two largest contributing patients account for approximately 89% of all F beats.
- The top three patients account for approximately 91%.
- The top five account for approximately 94%.
This creates a difficult patient-independent learning problem: the model can learn useful signal characteristics from the available F examples, but those characteristics are difficult to generalize to unseen patients.
This is also reflected by:
F-class PR-AUC = 0.0344
and
F-class OOF F1 = 0.07
Therefore, the overall accuracy should not be interpreted as evidence of equally strong performance across all arrhythmia classes.
Feature Ablation Study
Different groups of DSP features were evaluated to determine their contribution.
Feature Group	Validation Macro F1
Time-domain	0.4115
RR / Heart Rate	0.4923
Wavelet	0.2760
Spectral	0.2364
All Features	0.6503


The combined feature set performed best in the ablation experiment.
Among individual feature groups, RR/heart-rate features were the strongest standalone group, highlighting the importance of beat-to-beat temporal information.
The ablation experiment used a separate patient split, so its absolute scores should not be directly compared with the 5-fold cross-validation scores.

Important Features
The final XGBoost model identifies several high-importance features, including:
- wav_4_std
- wav_4_energy
- wav_4_energy_ratio
- heart_rate
- rr_deviation_ratio
- spectral_energy
- wavelet_entropy
- next_rr
- rr_interval
- wav_0_std
This suggests that the model relies heavily on a combination of:
wavelet characteristics + heart-rate/RR information + ECG morphology/slope information.
Feature importance indicates model usage but does not establish clinical causality or clinical significance.
Streamlit Application
A Streamlit application is included for interactive inference.
The application allows users to:
- Upload a CSV containing the 54 extracted ECG/DSP features
- Validate feature columns and missing values
- Predict the arrhythmia class for each beat
- View class probabilities
- View overall prediction statistics
- Inspect individual beat predictions
- Download prediction results as CSV
Application Pipeline
CSV containing 54 ECG/DSP features
                │
                ▼
        Feature Validation
                │
                ▼
        Final XGBoost Model
                │
                ▼
       Class Probabilities
                │
                ▼
       Arrhythmia Prediction
The current application operates on pre-extracted ECG features, not directly on raw ECG waveform files.

Running the Project
Requirements
- Python 3.12+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- XGBoost
- PyWavelets
- Matplotlib
- Streamlit
- WFDB
Install dependencies using:
pip install numpy scipy pandas scikit-learn matplotlib seaborn pywavelets xgboost wfdb streamlit
Running Inference
The inference script is located at:
experiments/inference/predict.py
It expects a CSV containing the 54 required feature columns.
Example:
python experiments/inference/predict.py --input experiments/inference/test_input.csv
The prediction output is saved as:
experiments/inference/predictions.csv
Running the Streamlit Application
From the project root:
streamlit run app/app.py
The application will open in a browser.
Project Structure
ECG-Arrhythmia-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   └── features.csv
│
├── databases/
│   └── ...
│
├── experiments/
│   │
│   ├── cross_validation/
│   │   └── patient_cv.py
│   │
│   ├── error_analysis/
│   │   └── error_analysis.py
│   │
│   ├── feature_ablation/
│   │   └── feature_ablation.py
│   │
│   ├── f_class_analysis/
│   │   └── f_class_analysis.py
│   │
│   ├── imbalance_comparison/
│   │   └── imbalance_comparison.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   └── train_final_model.py
│   │
│   ├── roc_pr_analysis/
│   │   └── roc_pr_analysis.py
│   │
│   └── xgboost_tuning.py
│
├── models/
│   └── final_xgboost/
│       ├── final_xgboost_model.json
│       ├── feature_names.json
│       ├── label_encoder.pkl
│       ├── model_config.json
│       └── final_feature_importance.csv
│
├── build_dataset.py
├── feature_extraction.py
├── preprocessing.py
├── qrs_detection.py
├── segmentation.py
├── train_models.py
├── README.md
└── .gitignore
Limitations
The current system has several important limitations:
1. The model operates on extracted ECG features rather than directly consuming raw ECG waveforms in the deployed application.
2. The F class contains very few samples and is concentrated among a small number of patients.
3. Patient-independent evaluation demonstrates substantially more challenging performance than random beat-level splitting.
4. The reported cross-validation is a grouped patient-independent evaluation and should not be described as the official MIT-BIH AAMI DS1/DS2 protocol.
5. The model has not undergone clinical validation.
6. Model confidence represents XGBoost predicted probability and should not be interpreted as clinical certainty.
7. The system should be considered a research/educational prototype rather than a clinical diagnostic system.
Reproducibility
The repository contains the main preprocessing, feature extraction, training, evaluation, error analysis and inference scripts used to develop the system.
Large datasets, raw ECG recordings and large generated prediction files are excluded using .gitignore.
The final trained XGBoost model and its associated metadata are included.
Future Work
Potential future improvements include:
- Increasing the number and diversity of F-class examples
- Evaluating additional patient-independent datasets
- Improving minority-class generalization
- Exploring calibrated probabilities
- Testing additional machine learning architectures
- Extending the application to support raw ECG input
- Cloud deployment and scalable inference
Technologies Used
- Python
- NumPy
- Pandas
- SciPy
- WFDB
- PyWavelets
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Git / GitHub
Author
Prashant Kumar Moharana
Electronics and Communication Engineering
VIT Chennai
Disclaimer
This project is intended strictly for research and educational purposes.
It is not a medical device and should not be used to make clinical decisions or diagnose patients.
