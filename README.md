# PCA–ANN Formalin Detection using Electronic Nose

Implementation of a **Principal Component Analysis (PCA)** and **Artificial Neural Network (ANN)** pipeline for formalin detection using Electronic Nose (E-Nose) sensor data.

This repository contains the complete workflow from raw sensor preprocessing, feature extraction, model training, evaluation, deployment validation, and Raspberry Pi inference.

> **Status:** Research Implementation (Workflow 25 Experiments)

---

# Overview

This project implements an end-to-end machine learning pipeline for formalin detection using Electronic Nose measurements.

The workflow includes:

- Raw sensor preprocessing
- Baseline and exposure segmentation
- 13-feature extraction
- Missing value imputation
- Z-score normalization
- Principal Component Analysis (3 components)
- Artificial Neural Network classification
- Leave-One-Replication-Out Cross Validation
- Final model training
- Deployment validation
- Raspberry Pi inference

---

# Repository Structure

```text
PCA_ANN_All25/
│
├── README.md
├── ALL25_RESULTS.md
├── requirements.txt
├── requirements-raspi.txt
│
├── train_all25.py
├── pca_ann_pipeline.py
├── predict_raw.py
├── deployment_validation.py
├── seed_stability_analysis.py
├── raspi_predict_excel.py
│
├── data/
│   └── Data Validasi & Pengujian (1).xlsx
│
├── models/
│   ├── model_pca_ann.pkl
│   ├── model_pca_ann.joblib
│   ├── model_ann_13_fitur.pkl
│   └── model_ann_13_fitur.joblib
│
├── results/
│   ├── metrics_summary.csv
│   ├── fold_metrics.csv
│   ├── pca_scores.csv
│   ├── pca_loadings.csv
│   ├── pca_explained_variance.csv
│   ├── confusion_matrices.png
│   ├── deployment_tests/
│   └── ...
│
└── tests/
    ├── test_dataset_integration.py
    ├── test_deployment_artifact.py
    └── test_pca_ann_pipeline.py
```

---

# Methodological Notes

This repository corresponds to the **all-25 experiment** workflow.

The pipeline uses:

```python
short_window_policy = "keep"
```

This policy retains all 25 experiments, including samples with quality-control warnings.

Quality Control Summary:

- QC OK : 23 samples
- QC Warning : 2 samples

Warnings remain recorded inside the generated feature table and are **not hidden** from subsequent analysis.

---

# Machine Learning Workflow

```text
Raw Sensor Data
        │
        ▼
Metadata Cleaning
        │
        ▼
Baseline & Exposure Segmentation
        │
        ▼
13 Feature Extraction
        │
        ▼
Median Imputer
        │
        ▼
Z-score Standardization
        │
        ▼
Principal Component Analysis
(3 Components)
        │
        ▼
Artificial Neural Network
        │
        ▼
Leave-One-Replication-Out
Cross Validation
        │
        ▼
Final Model Training
        │
        ▼
Deployment Validation
```

---

# Installation

## Windows

```powershell
py -3.10 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Linux

```bash
python3.10 -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install -r requirements.txt
```

---

# Training

Run the complete workflow:

```bash
python train_all25.py
```

The script automatically performs:

- Data preprocessing
- Feature extraction
- PCA transformation
- ANN training
- Cross-validation
- Final model generation
- Seed stability analysis
- Deployment validation

---

# Deployment Validation

Run:

```bash
python deployment_validation.py
```

Validation includes:

- Replay testing
- Dummy testing
- Negative testing
- Model reload verification

---

# Raspberry Pi Deployment

Deployment is provided through:

```text
raspi_predict_excel.py
```

The deployment pipeline:

1. Verify model hash.
2. Read raw Excel measurement.
3. Perform preprocessing.
4. Extract 13 features.
5. Execute PCA transformation.
6. Run ANN inference.
7. Export prediction to JSON.

---

# Output Files

| File | Description |
|------|-------------|
| metrics_summary.csv | Overall evaluation metrics |
| fold_metrics.csv | Metrics for each CV fold |
| features_13.csv | Extracted features |
| pca_scores.csv | PCA scores |
| pca_loadings.csv | PCA loading matrix |
| pca_explained_variance.csv | Explained variance |
| predictions_oof.csv | Out-of-fold predictions |
| cleaned_rows.csv | Processed sensor rows |
| excluded_samples.csv | Excluded samples |
| deployment_tests/ | Deployment validation outputs |

---

# Running Unit Tests

```bash
python -m unittest discover -s tests -v
```

---

# Requirements

Python 3.10 is recommended.

Install dependencies using:

```bash
pip install -r requirements.txt
```

For Raspberry Pi:

```bash
pip install -r requirements-raspi.txt
```

---

# Research Notes

This implementation represents the software component of a PCA–ANN formalin detection study using Electronic Nose sensor measurements.

The repository is intended for:

- Research reproducibility
- Software validation
- Deployment demonstration
- Academic documentation

Deployment validation verifies software functionality and serialization consistency. It does **not** replace independent field validation or laboratory confirmation.

---

# Intellectual Property Notice

This repository contains software developed as part of an academic research project.

The source code, trained models, documentation, and associated materials are protected under applicable copyright laws. Unauthorized reproduction, modification, or redistribution outside the applicable license or without the author's permission is prohibited.

---

# Author

**Firda Anandhita**

Research Project:

**PCA–ANN Based Formalin Detection using Electronic Nose**

---

# Citation

If this repository contributes to your research, please cite the associated thesis, publication, or software documentation.
