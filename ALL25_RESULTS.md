# Training Results for All 25 Experiments

These results are translated from the report in `PCA_ANN_All25.zip`. They describe
the archived run, not a new execution performed while preparing this repository.

## Configuration

- Dataset: `Data Validasi & Pengujian (1).xlsx`, sheet `Data`
- Phases: Baseline and Exposure; Purging is ignored
- Baseline: last 60 seconds; Exposure: first 120 seconds
- Short-window policy: `keep`
- Features: 13; standardization: Z-score
- PCA: PC1, PC2, and PC3
- ANN: 8-neuron hidden layer, ReLU, LBFGS solver, alpha 0.1
- Evaluation: leave-one-replication-out, 5 folds
- Classes: 5 non-formalin and 20 formalin samples

## Data quality

All 25 experiments are retained. `5mL_rep1` has 13 baseline readings spanning
39.239 seconds, with effective coverage of 42.475 seconds. It does not meet the
60-second baseline target. No temporal extension, interpolation, or imputation
is performed; all available readings are used and flagged `baseline_duration_short`.

There are 23 samples with QC status `ok` and two with status `warning`:

- `5mL_rep1`: `baseline_duration_short`
- `5mL_rep4`: `exposure_window_gap_over_10s`

These deviations make the all-25 results a non-strict sensitivity analysis.
The strict 24-experiment variant more closely follows the baseline requirement
in the research document (`Resume.pdf`).

## Out-of-fold evaluation

| Model | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANN, 13 features | 96.00% | 90.00% | 95.24% | 100.00% | 80.00% | 97.56% | 97.00% |
| PCA–ANN | 96.00% | 97.50% | 100.00% | 95.00% | 100.00% | 97.44% | 100.00% |

## Confusion matrices

### ANN with 13 features

| Actual / Predicted | Non-formalin | Formalin |
|---|---:|---:|
| Non-formalin | 4 | 1 |
| Formalin | 0 | 20 |

TN = 4, FP = 1, FN = 0, TP = 20.
`0mL_rep1` is misclassified as formalin, with formalin probability 0.956380.

### PCA–ANN

| Actual / Predicted | Non-formalin | Formalin |
|---|---:|---:|
| Non-formalin | 5 | 0 |
| Formalin | 1 | 19 |

TN = 5, FP = 0, FN = 1, TP = 19.
`1mL_rep3` is misclassified as non-formalin, with formalin probability 0.424723.

## PCA

| Component | Explained variance | Cumulative |
|---|---:|---:|
| PC1 | 68.68% | 68.68% |
| PC2 | 14.81% | 83.49% |
| PC3 | 7.79% | 91.27% |

The three components retain 91.27% of the variance in the 13 features.

## Random-seed stability

Across 10 seeds, labels and classification metrics remain stable. PCA–ANN retains
96.00% accuracy, 97.50% balanced accuracy, 95.00% recall, 100.00% specificity,
97.44% F1-score, and 100.00% ROC-AUC for every tested seed. This indicates stability
on the same dataset and folds, not generalization to new field experiments.

## Artifact validation

The archived run reports that all 16 technical deployment checks passed, including:

- Successful pickle loading and the `imputer -> scaler -> pca -> ann` sequence.
- A valid 13-feature contract.
- Replay of 25 experiments reproducing training features, with maximum difference `7.105e-15`.
- Identical predictions before and after pickle reload.
- Unchanged probabilities when Purging data is removed.
- Correct synthetic non-formalin prediction, with formalin probability 0.005732.
- Correct synthetic formalin prediction, with formalin probability close to 1.
- Rejection of incomplete or invalid inputs.

SHA-256 of the archived `model_pca_ann.pkl`:

```text
a9f529e9fc2b87501dd7f41a6f58723ce214a159bd0e95f676d4d3b218911a8c
```

## Comparison with the strict 24-experiment model

Adding `5mL_rep1` does not change the number of errors. Accuracy changes from
95.83% to 96.00% because the denominator increases from 24 to 25. PCA–ANN still
misclassifies `1mL_rep3`. Balanced accuracy changes from 97.37% to 97.50%, while
cumulative explained variance changes from 91.01% to 91.27%.

The improvement is small and largely reflects one additional true positive.
This variant is useful as a sensitivity analysis, but must not be described as
fully compliant with the 60-second baseline protocol.
