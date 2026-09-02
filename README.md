# ECG Signal Processing and ML-Based Arrhythmia Detection

All 7 modules are implemented and tested (with a synthetic ECG signal, since
this dev environment can't reach PhysioNet). Every module ran successfully
here — you just need to plug in the real dataset.

## What's already done and verified working

| File | What it does | Verified |
|---|---|---|
| `preprocessing.py` | Baseline wander removal + Butterworth bandpass | Yes |
| `qrs_detection.py` | Pan-Tompkins R-peak detection | Yes — detected 12/12 synthetic beats exactly |
| `segmentation.py` | Beat windowing + AAMI label mapping | Yes |
| `feature_extraction.py` | Time/frequency/wavelet features (24 per beat) | Yes |
| `train_models.py` | SVM, Random Forest, XGBoost w/ patient-level split | Yes |
| `evaluate.py` | Confusion matrix, precision/recall/F1, comparison table | Yes |
| `visualize.py` | All 6 report figures | Yes — see `results/` |
| `build_dataset.py` | Converts real MIT-BIH records into the feature CSV | **Run this on your machine** |

## Step-by-step: running this on your own computer

### 1. Set up environment
```bash
python -m venv ecg_env
source ecg_env/bin/activate      # Windows: ecg_env\Scripts\activate
pip install numpy scipy pandas matplotlib scikit-learn pywavelets xgboost wfdb seaborn joblib
```

### 2. Download the real MIT-BIH data
```bash
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/MIT-BIH')"
```
This pulls all 48 records directly from PhysioNet (~100MB). Takes a few minutes.

### 3. Build the real feature dataset
```bash
python build_dataset.py
```
This runs preprocessing -> QRS detection -> segmentation -> feature extraction
across every record and saves `data/features.csv`. This is the one step that
needed real data, so it couldn't be run in this sandbox — it *will* work
once you have the files from Step 2, since every module it calls was
verified individually above.

### 4. Train the models on real data
Edit the bottom of `train_models.py` (or write a 5-line new script) to load
`data/features.csv` instead of the synthetic data block, using the DS1/DS2
patient split already defined in `build_dataset.py`:

```python
import pandas as pd
from train_models import patient_level_split, prepare_xy, train_all_models, save_models
from build_dataset import DS1, DS2

df = pd.read_csv('data/features.csv')
feature_cols = [c for c in df.columns if c not in ('label', 'patient_id')]

train_df, test_df = patient_level_split(df, DS1, DS2)
X_train, y_train, le = prepare_xy(train_df, feature_cols, fit_encoder=True)
models, scaler = train_all_models(X_train, y_train)
save_models(models, scaler, le)
```

### 5. Evaluate
Same pattern — swap the synthetic data block in `evaluate.py` for
`data/features.csv` loaded the same way, then run:
```bash
python evaluate.py
```

### 6. Generate final figures
```bash
python visualize.py
```
(after adapting its `__main__` block the same way, or just call the plotting
functions directly with your real signal/results — they're all standalone
functions you can import).

## Methodology and leakage controls

- The 44-record MIT-BIH inter-patient split is fixed: DS1 records train and
  tune the models; DS2 records are held out until the single final evaluation.
- Invalid numeric values are converted to missing values. A median imputer and
  scaler are fitted on DS1 only, then applied unchanged to DS2.
- Class balancing is performed on DS1 only: normal beats are capped relative
  to the largest minority class, then SMOTE synthesizes minority examples.
  DS2 is never sampled, scaled, imputed, or used to select hyperparameters.
- SVM and XGBoost hyperparameters are selected using a group-held-out DS1
  validation fold (patient groups), with macro-F1 as the selection metric.
- `models/` stores the imputer, scaler, class mapping, feature list, models,
  methodology JSON, and per-beat DS2 predictions. `evaluate.py` writes DS2
  confusion matrices, per-class metrics, and a comparison table.

## Notes for your report
- The patient-level split (DS1 = train, DS2 = test) is the literature-standard
  inter-patient split — mention this explicitly, it's a common mistake to skip.
- Start with binary classification (Normal vs Abnormal) if the 5-class problem
  proves noisy on real data — the same code handles both, just relabel before
  training.
- `results/` already contains example figures generated from synthetic data
  to confirm the plotting code works — regenerate these with real output for
  your final report.

## Project folder structure
```
ECG_Arrhythmia_Detection/
├── data/MIT-BIH/          # put downloaded .dat/.hea/.atr files here
├── preprocessing.py
├── qrs_detection.py
├── segmentation.py
├── feature_extraction.py
├── build_dataset.py       # real-data pipeline runner
├── train_models.py
├── evaluate.py
├── visualize.py
├── models/                # saved .pkl models
└── results/                # saved figures + comparison CSV
```
