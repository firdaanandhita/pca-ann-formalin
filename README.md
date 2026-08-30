# PCA–ANN Formalin Detection using Electronic Nose

Implementation of a **Principal Component Analysis (PCA)** and **Artificial Neural Network (ANN)** pipeline for formalin detection using Electronic Nose (E-Nose) sensor data.

This repository contains the source code implementing the complete software workflow, including raw sensor preprocessing, feature extraction, model training, prediction, deployment validation, and Raspberry Pi inference.

> **Status:** Public source code repository accompanying an academic research project.

---

# Repository Scope

This repository contains the software implementation of the PCA–ANN pipeline for formalin detection using Electronic Nose sensor data.

To keep the repository focused on the software implementation, **research datasets, trained model artifacts, and experimental outputs are intentionally excluded**. The methodology, experimental results, and performance evaluation are documented separately in the associated thesis and supporting research documents.

---

# Overview

The implemented software performs an end-to-end machine learning workflow consisting of:

- Raw sensor preprocessing
- Baseline and exposure segmentation
- 13-feature extraction
- Missing value imputation
- Z-score normalization
- Principal Component Analysis (PCA)
- Artificial Neural Network (ANN) classification
- Leave-One-Replication-Out Cross Validation
- Final model training
- Deployment validation
- Raspberry Pi inference

---

# Repository Structure

```text
pca-ann-formalin/
│
├── src/
│   ├── pca_ann_pipeline.py
│   ├── train_all25.py
│   ├── predict_raw.py
│   └── raspi_predict_excel.py
│
├── validation/
│   └── deployment_validation.py
│
├── README.md
├── requirements.txt
├── requirements-raspi.txt
└── .gitignore
```

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
Median Imputation
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
        │
        ▼
Prediction
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

Run the complete training workflow:

```bash
python src/train_all25.py
```

The training workflow performs:

- Data preprocessing
- Feature extraction
- PCA transformation
- ANN training
- Cross-validation
- Final model generation

---

# Prediction

Run prediction using the trained PCA–ANN model:

```bash
python src/predict_raw.py
```

The prediction pipeline automatically:

- Loads the trained model
- Performs preprocessing
- Extracts 13 features
- Applies PCA transformation
- Executes ANN inference
- Produces prediction results

---

# Deployment Validation

Run the deployment validation workflow:

```bash
python validation/deployment_validation.py
```

The validation includes:

- Model loading verification
- Replay testing
- Dummy testing
- Input validation
- Model reload verification
- Pipeline consistency checking

Deployment validation verifies software functionality and serialization consistency. It does **not** replace independent field validation or laboratory confirmation.

---

# Raspberry Pi Deployment

The Raspberry Pi implementation is provided through:

```text
src/raspi_predict_excel.py
```

The deployment workflow:

1. Load the trained model.
2. Read raw Excel measurements.
3. Perform preprocessing.
4. Extract the 13 engineered features.
5. Apply PCA transformation.
6. Execute ANN inference.
7. Export prediction results.

---

# Requirements

Python **3.10** is recommended.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

For Raspberry Pi deployment:

```bash
pip install -r requirements-raspi.txt
```

---

# Research Notes

This repository contains the software implementation developed as part of an academic research project on formalin detection using an Electronic Nose and a PCA–ANN classification approach.

The repository is intended for:

- Software implementation reference
- Research reproducibility
- Deployment demonstration
- Academic documentation
- Intellectual property (software copyright) support

---

# Intellectual Property Notice

This repository contains software developed as part of an academic research project.

The source code is protected under applicable copyright laws. Research datasets, trained model artifacts, and experimental outputs are maintained separately from this public repository.

Unauthorized reproduction, modification, or redistribution outside the applicable license or without the author's permission is prohibited.

---

# Author

**Firda Anandhita**

Research Project:

**PCA-ANN-Based E-Nose for Formalin Detection in Tofu**

---

# Citation

If this repository contributes to your research, please cite the associated thesis, publication, or software documentation.
