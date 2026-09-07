# PCA–ANN Formalin Detection — 25-Experiment Workflow

Training, evaluation, and Raspberry Pi inference for an electronic nose model
combining Principal Component Analysis (PCA) and an Artificial Neural Network (ANN).
The all-25 workflow retains all 25 experiments from the original research dataset.

## Repository contents

The source layout and computational workflow follow `PCA_ANN_All25.zip`.
Documentation, comments, docstrings, and program messages have been translated
into English. Dataset headers, feature names, artifact identifiers, and original
input filenames are retained for compatibility. Run commands from the repository root.
The original repository license is retained in `LICENSE`.

```text
pca-ann-formalin/
├── pca_ann_pipeline.py
├── train_all25.py
├── seed_stability_analysis.py
├── deployment_validation.py
├── predict_raw.py
├── raspi_predict_excel.py
├── tests/
│   ├── test_pca_ann_pipeline.py
│   ├── test_dataset_integration.py
│   └── test_deployment_artifact.py
├── requirements.txt
├── requirements-raspi.txt
├── README.md
├── ALL25_RESULTS.md
├── LICENSE
└── .gitignore
```

| Script | Purpose |
|---|---|
| `train_all25.py` | Complete all-25 training and validation workflow |
| `pca_ann_pipeline.py` | Raw preprocessing, 13-feature extraction, PCA, ANN, cross-validation, plots, and model export |
| `seed_stability_analysis.py` | Repeated evaluation with 10 random seeds |
| `deployment_validation.py` | Replay, synthetic-input, invalid-input, and model-reload checks |
| `predict_raw.py` | Inference on exactly one raw recording |
| `raspi_predict_excel.py` | Excel-based Raspberry Pi inference, with model hash verification before loading |

[ALL25_RESULTS.md](ALL25_RESULTS.md) records the results supplied in the archive;
it does not claim a new training run was performed for this repository update.

The dataset `Data Validasi & Pengujian (1).xlsx`, research document `Resume.pdf`,
trained models in `model/`, and generated files in `outputs_all25/` are excluded
from Git. They remain local research materials, binary artifacts, or generated
outputs. Copy the dataset from your local archive to the repository root before
training. The program does not download it automatically.

## Methodological notes

The all-25 model uses `short_window_policy=keep`, retaining `5mL_rep1` even though
its effective baseline coverage is only 42.475 seconds against a 60-second target.
All 13 available baseline readings are used, without extending or fabricating
data, and the sample receives a `baseline_duration_short` warning.

`5mL_rep4` receives an `exposure_window_gap_over_10s` warning. Overall, 23 samples
have QC status `ok` and two have status `warning`. These flags remain visible in
`features_13.csv`. Treat the all-25 variant as a sensitivity analysis. The strict
24-experiment variant more closely follows the baseline requirement in the
research document. Its `outputs/` folder is included in neither the ZIP nor this repository.

## Training workflow

```text
Raw timestamps and sensor readings
        ↓
Remove rows with missing experiment metadata
        ↓
Use Baseline and Exposure; ignore Purging
        ↓
Group by concentration and replication
        ↓
Last 60 seconds of Baseline + first 120 seconds of Exposure
        ↓
Extract 13 features
        ↓
Median imputation → Z-score scaling → PCA (3 components) → ANN
        ↓
Leave-one-replication-out evaluation
        ↓
Fit final models on all 25 samples
```

## Installation

Use Windows 10/11 or 64-bit Linux. Python 3.10 is recommended for consistency with
the original environment. Allow space for a virtual environment and generated plots.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is optional because these commands invoke the environment directly.

### Linux

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

For the PowerShell examples below, Linux users should use `.venv/bin/python`
and backslashes for line continuation instead of PowerShell backticks.

## Run all-25 training

```powershell
.\.venv\Scripts\python.exe train_all25.py
```

This runs preprocessing, feature extraction, the ANN versus PCA–ANN comparison,
leave-one-replication-out cross-validation, final training, 10-seed stability
analysis, and deployment validation.

To choose paths:

```powershell
.\.venv\Scripts\python.exe train_all25.py `
  --input "Data Validasi & Pengujian (1).xlsx" `
  --sheet Data `
  --output-dir outputs_all25
```

To run only the underlying training pipeline with the same configuration:

```powershell
.\.venv\Scripts\python.exe pca_ann_pipeline.py `
  --input "Data Validasi & Pengujian (1).xlsx" `
  --sheet Data `
  --output-dir outputs_all25 `
  --baseline-seconds 60 `
  --baseline-anchor tail `
  --exposure-seconds 120 `
  --short-window-policy keep `
  --pca-components 3 `
  --hidden-layers 8 `
  --alpha 0.1 `
  --max-iter 5000 `
  --cv-mode replication `
  --random-state 42
```

`--short-window-policy keep` is required for all-25 reproduction. The pipeline
defaults to `drop`, the strict variant. The direct pipeline command does not run
the separate stability and deployment-validation stages.

## Output files

| Local file under `outputs_all25/` | Purpose |
|---|---|
| `model_pca_ann.pkl` | Deployment pipeline containing the imputer, scaler, PCA, and ANN |
| `model_manifest.json` | Feature contract, configuration, versions, hashes, and QC notes |
| `features_13.csv` | One row per experiment, with features, audit metadata, windows, and QC |
| `pca_scores.csv` | PC1, PC2, and PC3 scores from the final PCA |
| `pca_loadings.csv` | Original feature weights for each component |
| `pca_explained_variance.csv` | Explained variance per component |
| `predictions_oof.csv` | Held-out predictions used for evaluation |
| `metrics_summary.csv`, `fold_metrics.csv` | Overall and per-fold metrics |
| `seed_stability_*.csv` | Results across random seeds |
| `cleaned_rows.csv` | Preprocessed Baseline and Exposure rows |
| `excluded_samples.csv` | Excluded samples; header only for the all-25 run |
| `deployment_tests/` | Replay, synthetic-input, and invalid-input validation results |

The archive also includes `PCA_ANN_All25.xlsx`, a consolidated result workbook,
and two Raspberry Pi demo JSON files. Training does not automatically create that
workbook or run those separate Raspberry Pi demos.

PCA loadings are not a new training dataset. Their signs indicate direction and
their absolute magnitudes describe relative contributions to each component.

## Automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dataset tests are skipped if the workbook is absent. The supplied artifact tests
target the strict variant in `outputs/` and are skipped if its required artifacts
are absent. All-25 deployment validation runs through `train_all25.py` or the command below.

## Separate validation commands

### Seed stability

```powershell
.\.venv\Scripts\python.exe seed_stability_analysis.py `
  --features outputs_all25/features_13.csv `
  --output-dir outputs_all25
```

This measures sensitivity to ANN initialization on the same dataset and folds,
not generalization to new field experiments.

### Deployment validation

```powershell
.\.venv\Scripts\python.exe deployment_validation.py `
  --workbook "Data Validasi & Pengujian (1).xlsx" `
  --model outputs_all25/model_pca_ann.pkl `
  --output-dir outputs_all25/deployment_tests
```

Replay and synthetic inputs check preprocessing, serialization, and inference.
They do not replace validation on new days, batches, or devices.

## Using archived models

The ZIP stores four model files in `model/`; the examples use `outputs_all25/`,
the destination for newly trained models. To use archived models without
retraining, copy those four files to `outputs_all25/`, together with
`model_manifest.json` from the archive's `outputs_all25/` folder. Verify hashes
against the manifest before loading. These files remain excluded from Git.

## Raspberry Pi setup

Use 64-bit Raspberry Pi OS and a virtual environment. Match the Python and library
versions recorded in the manifest, because scikit-learn pickle compatibility
across versions is not guaranteed. The original environment used Python 3.10.11.
If compatible packages are unavailable for your architecture, use a compatible
environment or another deployment format instead of forcing the pickle to load.

Copy these source files and locally obtained artifacts to the Pi:

```text
pca_ann_pipeline.py
predict_raw.py
raspi_predict_excel.py
requirements-raspi.txt
outputs_all25/model_pca_ann.pkl
outputs_all25/model_manifest.json
```

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-raspi.txt
```

### Raw Excel format

Required headers:

```text
Timestamp
Fase
HCHO
MQ-138
TGS822
HUMIDITY
```

`Fase` means phase. Keep original dataset headers and filenames for compatibility.
`Konsentrasi` means concentration and `Replikasi` means replication. These and
other extra columns are not model features. Production input must contain exactly
one Baseline–Exposure cycle. Purging rows may be present but are ignored.

### Demo on the original dataset

Select one experiment from the multi-experiment workbook:

```bash
.venv/bin/python raspi_predict_excel.py \
  --input "Data Validasi & Pengujian (1).xlsx" \
  --sheet Data \
  --model outputs_all25/model_pca_ann.pkl \
  --manifest outputs_all25/model_manifest.json \
  --demo-concentration-ml 0 \
  --demo-replication 5 \
  --sample-id demo_0mL_rep5 \
  --output demo_0mL_rep5_result.json
```

Concentration and replication select rows only; they are not model features.
This is training-data replay, not independent validation.

### Predict a new recording

```bash
.venv/bin/python raspi_predict_excel.py \
  --input new_recording.xlsx \
  --sheet Data \
  --model outputs_all25/model_pca_ann.pkl \
  --manifest outputs_all25/model_manifest.json \
  --sample-id pi01_20260725_140501 \
  --output pi01_20260725_140501_result.json
```

The program verifies the model hash, reads the workbook, ignores Purging, selects
the required windows, extracts 13 features, and runs the saved pipeline. It writes
the class, formalin probability, QC status, windows, and features to JSON.

The archive includes `raspi_demo_0mL_rep5.json` and `raspi_demo_15mL_rep5.json`
under `outputs_all25/deployment_tests/`. They predict class 0 for 0 mL and class 1
for 15 mL, with both passing QC. They remain training-data replays.

Do not use `--allow-qc-warnings` for field decisions. It is a diagnostic option
that can permit short recordings, gaps, or problematic sensor values.

## Interpret predictions

```json
{
  "predicted_label": 1,
  "predicted_class": "formalin",
  "probability_formalin": 0.87,
  "decision_threshold": 0.5,
  "qc_status": "ok"
}
```

`probability_formalin` is a model score, not a measurement of concentration or a
guarantee of certainty. This proof-of-concept does not replace laboratory testing
or independent field validation.
