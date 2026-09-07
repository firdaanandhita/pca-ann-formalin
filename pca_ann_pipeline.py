"""Convert raw sensor recordings into PCA-ANN models.

Read Excel/CSV data, retain Baseline and Exposure, group by concentration and
replication, extract 13 features, apply Z-score scaling, compare ANN with
PCA-ANN using out-of-fold predictions, and refit final models on retained samples.
For all 25 experiments, use train_all25.py: its keep policy retains short
baselines with QC warnings, without adding or interpolating readings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


FEATURE_COLUMNS = [
    "HCHO_baseline_mean",
    "HCHO_exposure_mean",
    "HCHO_exposure_max",
    "HCHO_delta_max",
    "MQ138_baseline_mean",
    "MQ138_exposure_mean",
    "MQ138_exposure_max",
    "MQ138_delta_max",
    "TGS822_baseline_mean",
    "TGS822_exposure_mean",
    "TGS822_exposure_max",
    "TGS822_delta_max",
    "RH_mean",
]

METADATA_COLUMNS = [
    "sample_id",
    "concentration_original",
    "concentration_ml",
    "replication_id",
    "label",
]

COLUMN_ALIASES = {
    "timestamp": ["Timestamp", "Time", "Datetime", "Date Time", "Waktu"],
    "hcho": ["HCHO", "HCHO Sensor", "Sensor HCHO"],
    "mq138": ["MQ-138", "MQ138", "MQ_138", "MQ 138"],
    "tgs822": ["TGS822", "TGS-822", "TGS_822", "TGS 822"],
    "humidity": [
        "HUMIDITY",
        "Humidity",
        "RH",
        "RH%",
        "RH (%)",
        "Kelembapan",
    ],
    "concentration": [
        "Konsentrasi",
        "Concentration",
        "Kadar",
        "Konsentrasi Formalin",
    ],
    "replication": ["Replikasi", "Replication", "Replicate", "Ulangan"],
    "phase": ["Fase", "Phase", "Tahap"],
}

PHASE_ALIASES = {
    "baseline": {
        "baseline",
        "base",
        "awal",
        "kondisiawal",
    },
    "exposure": {
        "exposure",
        "paparan",
        "expose",
        "sampling",
        "sample",
    },
    "purging": {
        "purging",
        "purge",
        "pembersihan",
        "recovery",
        "cleaning",
    },
}

INTERNAL_COLUMNS = {
    "timestamp": "__timestamp",
    "hcho": "__hcho",
    "mq138": "__mq138",
    "tgs822": "__tgs822",
    "humidity": "__humidity",
}


@dataclass
class PreprocessingResult:
    """Cleaned rows and audit statistics describing preprocessing."""

    cleaned: pd.DataFrame
    source_rows: int
    dropped_missing_metadata: int
    rows_after_metadata_filter: int
    column_mapping: dict[str, str]
    invalid_numeric_counts: dict[str, int]
    missing_numeric_counts: dict[str, int]
    phase_counts: dict[str, int]
    phase_counts_before_filter: dict[str, int]
    ignored_phase_counts: dict[str, int]
    timestamp_quality: dict[str, Any]


@dataclass
class FeatureExtractionResult:
    """One feature row per experiment, together with excluded samples."""

    features: pd.DataFrame
    excluded_samples: list[dict[str, Any]]


@dataclass
class InferencePreprocessingResult:
    """Clean inference data, column mappings, and quality warnings."""

    cleaned: pd.DataFrame
    source_rows: int
    ignored_phase_counts: dict[str, int]
    phase_counts_used: dict[str, int]
    column_mapping: dict[str, str]
    input_warnings: list[str]


def _normalise_key(value: Any) -> str:
    """Normalize case, whitespace, and punctuation for column matching."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def resolve_columns(
    columns: Iterable[Any], required: Iterable[str] | None = None
) -> dict[str, str]:
    """Map Excel header aliases to internal column names.

    MQ-138, MQ138, and MQ 138 refer to the same sensor. Missing required
    columns or ambiguous matches raise errors to avoid selecting incorrect data."""

    required_columns = list(required or COLUMN_ALIASES.keys())
    unknown = [
        canonical
        for canonical in required_columns
        if canonical not in COLUMN_ALIASES
    ]
    if unknown:
        raise ValueError(f"Unknown canonical column names: {unknown}")

    normalised: dict[str, list[str]] = {}
    for column in columns:
        normalised.setdefault(_normalise_key(column), []).append(str(column))

    duplicates = {
        key: values for key, values in normalised.items() if len(values) > 1
    }
    if duplicates:
        raise ValueError(
            "Column names are ambiguous after normalization: "
            + json.dumps(duplicates, ensure_ascii=False)
        )

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical in required_columns:
        aliases = COLUMN_ALIASES[canonical]
        matches: list[str] = []
        for alias in aliases:
            matches.extend(normalised.get(_normalise_key(alias), []))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            mapping[canonical] = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple columns match '{canonical}': {matches}"
            )
        else:
            missing.append(canonical)

    if missing:
        raise ValueError(
            "Required columns not found: "
            + ", ".join(missing)
            + f". Available columns: {list(map(str, columns))}"
        )
    return mapping


def read_dataset(input_path: Path, sheet: str | int = "Data") -> pd.DataFrame:
    """Read an XLSX/XLSM/XLS, CSV, or TSV dataset without transforming values."""

    suffix = input_path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".tsv":
        return pd.read_csv(input_path, sep="\t")
    raise ValueError(
        f"Format '{input_path.suffix}' is not supported. Use XLSX, XLSM, XLS, CSV, or TSV."
    )


def _empty_metadata_mask(series: pd.Series) -> pd.Series:
    """Identify NaN, empty strings, and whitespace-only metadata."""

    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("").fillna(True)


def parse_concentration_ml(value: Any) -> float:
    """Parse values such as 1 mL or 1,5 into numeric milliliters."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Concentration is missing.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        text = str(value).strip()
        match = re.fullmatch(
            r"\s*([-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+))\s*"
            r"(?:ml|milliliter|milliliters)?\s*",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Cannot parse concentration: {value!r}")
        number = float(match.group(1).replace(",", "."))

    if not np.isfinite(number):
        raise ValueError(f"Concentration is not finite: {value!r}")
    if number < 0:
        raise ValueError(f"Negative concentration is invalid: {value!r}")
    return 0.0 if np.isclose(number, 0.0, atol=1e-12) else number


def _normalise_replication(value: Any) -> str:
    """Normalize replication IDs used as metadata, never model features."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Invalid replication: {value!r}")
        if number.is_integer():
            return str(int(number))
        return f"{number:.12g}"

    text = str(value).strip()
    if not text:
        raise ValueError("Replication is missing.")
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def _normalise_phase(value: Any) -> str:
    """Map phase aliases to baseline, exposure, or purging."""

    key = _normalise_key(value)
    for canonical, aliases in PHASE_ALIASES.items():
        if key in aliases:
            return canonical
    return str(value).strip().casefold()


def _to_numeric_locale(series: pd.Series) -> pd.Series:
    """Parse decimal commas as numbers and invalid values as NaN."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def preprocess_rows(raw: pd.DataFrame) -> PreprocessingResult:
    """Clean training rows and prepare metadata for feature extraction.

    Drop missing concentration, replication, or phase metadata. Ignore Purging
    and other phases. Assign class 0 to 0 mL and class 1 to positive concentrations.
    The 60/120-second windows are selected later by extract_features."""

    # Step 1: resolve actual column names, accepting reasonable Excel header aliases.
    mapping = resolve_columns(raw.columns)

    # Step 2: drop rows missing essential experiment metadata. Count missing sensor
    # values so QC and the model imputer can handle them transparently.
    metadata_source_columns = [
        mapping["concentration"],
        mapping["replication"],
        mapping["phase"],
    ]
    missing_metadata = pd.DataFrame(
        {
            column: _empty_metadata_mask(raw[column])
            for column in metadata_source_columns
        }
    ).any(axis=1)

    metadata_cleaned = raw.loc[~missing_metadata].copy()
    if metadata_cleaned.empty:
        raise ValueError(
            "No rows remain after dropping missing concentration, replication, and phase metadata."
        )

    # Step 3: normalize phases and retain only Baseline and Exposure.
    # Count Purging for audit purposes, never as a feature.
    metadata_cleaned["phase_normalized"] = metadata_cleaned[
        mapping["phase"]
    ].map(_normalise_phase)
    phase_counts_before_filter = {
        str(key): int(value)
        for key, value in metadata_cleaned["phase_normalized"]
        .value_counts()
        .items()
    }
    feature_phase_mask = metadata_cleaned["phase_normalized"].isin(
        ["baseline", "exposure"]
    )
    ignored_phase_counts = {
        str(key): int(value)
        for key, value in metadata_cleaned.loc[
            ~feature_phase_mask, "phase_normalized"
        ]
        .value_counts()
        .items()
    }
    cleaned = metadata_cleaned.loc[feature_phase_mask].copy()
    if cleaned.empty:
        raise ValueError(
            "No Baseline or Exposure rows remain after preprocessing."
        )

    # Step 4: parse concentration and derive binary labels.
    # Keep concentration as research metadata, not classifier input.
    concentration_values: list[float] = []
    concentration_errors: list[tuple[int, Any, str]] = []
    for index, value in cleaned[mapping["concentration"]].items():
        try:
            concentration_values.append(parse_concentration_ml(value))
        except ValueError as exc:
            concentration_values.append(np.nan)
            concentration_errors.append((int(index), value, str(exc)))
    if concentration_errors:
        preview = concentration_errors[:10]
        raise ValueError(
            "Some non-empty concentration values are invalid. "
            f"Examples (index, value, reason): {preview}"
        )

    cleaned["concentration_original"] = cleaned[mapping["concentration"]].astype(
        "string"
    )
    cleaned["concentration_ml"] = concentration_values
    cleaned["replication_id"] = cleaned[mapping["replication"]].map(
        _normalise_replication
    )
    cleaned["label"] = (cleaned["concentration_ml"] > 0).astype(int)

    # Step 5: parse timestamps and four numeric signals.
    # Record invalid values instead of silently discarding them.
    cleaned[INTERNAL_COLUMNS["timestamp"]] = pd.to_datetime(
        cleaned[mapping["timestamp"]], errors="coerce"
    )
    invalid_timestamp_count = int(
        cleaned[INTERNAL_COLUMNS["timestamp"]].isna().sum()
    )
    if invalid_timestamp_count:
        raise ValueError(
            f"Found {invalid_timestamp_count} invalid timestamps in rows with complete metadata."
        )

    invalid_numeric_counts: dict[str, int] = {}
    missing_numeric_counts: dict[str, int] = {}
    for canonical in ("hcho", "mq138", "tgs822", "humidity"):
        internal = INTERNAL_COLUMNS[canonical]
        converted = _to_numeric_locale(cleaned[mapping[canonical]]).replace(
            [np.inf, -np.inf], np.nan
        )
        newly_invalid = converted.isna() & ~cleaned[mapping[canonical]].isna()
        invalid_numeric_counts[canonical] = int(newly_invalid.sum())
        missing_numeric_counts[canonical] = int(converted.isna().sum())
        cleaned[internal] = converted

    cleaned["sample_id"] = cleaned.apply(
        lambda row: (
            f"{row['concentration_ml']:g}mL_rep{row['replication_id']}"
        ),
        axis=1,
    )

    # Step 6: audit timestamps within each experiment and phase.
    # Large gaps and reversed timestamps support subsequent QC.
    phase_counts = {
        str(key): int(value)
        for key, value in cleaned["phase_normalized"].value_counts().items()
    }
    within_phase_deltas: list[float] = []
    timestamp_anomalies: list[dict[str, Any]] = []
    for (sample_id, phase), group in cleaned.groupby(
        ["sample_id", "phase_normalized"], sort=False
    ):
        ordered = group.sort_index()
        deltas = ordered[INTERNAL_COLUMNS["timestamp"]].diff().dt.total_seconds()
        finite_deltas = deltas.dropna()
        within_phase_deltas.extend(finite_deltas.tolist())
        anomalous = finite_deltas.loc[
            (finite_deltas < 0) | (finite_deltas > 10)
        ]
        for row_index, delta_seconds in anomalous.items():
            timestamp_anomalies.append(
                {
                    "row_index": int(row_index),
                    "sample_id": str(sample_id),
                    "phase": str(phase),
                    "delta_seconds": float(delta_seconds),
                    "type": (
                        "timestamp_reversal"
                        if delta_seconds < 0
                        else "gap_over_10_seconds"
                    ),
                }
            )

    positive_deltas = [
        delta for delta in within_phase_deltas if np.isfinite(delta) and delta > 0
    ]
    timestamp_quality = {
        "duplicate_timestamp_count": int(
            cleaned[INTERNAL_COLUMNS["timestamp"]].duplicated().sum()
        ),
        "median_positive_interval_seconds": (
            float(np.median(positive_deltas)) if positive_deltas else None
        ),
        "timestamp_reversal_count_within_sample_phase": sum(
            anomaly["type"] == "timestamp_reversal"
            for anomaly in timestamp_anomalies
        ),
        "gap_over_10_seconds_count_within_sample_phase": sum(
            anomaly["type"] == "gap_over_10_seconds"
            for anomaly in timestamp_anomalies
        ),
        "anomalies": timestamp_anomalies,
    }

    return PreprocessingResult(
        cleaned=cleaned,
        source_rows=int(len(raw)),
        dropped_missing_metadata=int(missing_metadata.sum()),
        rows_after_metadata_filter=int(len(metadata_cleaned)),
        column_mapping=mapping,
        invalid_numeric_counts=invalid_numeric_counts,
        missing_numeric_counts=missing_numeric_counts,
        phase_counts=phase_counts,
        phase_counts_before_filter=phase_counts_before_filter,
        ignored_phase_counts=ignored_phase_counts,
        timestamp_quality=timestamp_quality,
    )


def preprocess_inference_rows(
    raw: pd.DataFrame, sample_id: str = "inference_sample"
) -> InferencePreprocessingResult:
    """Clean one recording without requiring labels or concentration.

    Required fields are Timestamp, Fase, HCHO, MQ-138, TGS822, and HUMIDITY.
    Temporary concentration and label metadata only adapt the shared extractor;
    they are not among the 13 features and never enter the ANN."""

    sample_id = str(sample_id).strip()
    if not sample_id:
        raise ValueError("Inference sample_id must not be empty.")

    # Inference does not require concentration or replication: these are
    # unknown when measuring a new sample.
    required = [
        "timestamp",
        "hcho",
        "mq138",
        "tgs822",
        "humidity",
        "phase",
    ]
    mapping = resolve_columns(raw.columns, required=required)
    working = raw.copy()
    working["phase_normalized"] = working[mapping["phase"]].map(
        _normalise_phase
    )

    # As during training, ignore Purging and other unused phases.
    feature_phase_mask = working["phase_normalized"].isin(
        ["baseline", "exposure"]
    )
    ignored_phase_counts = {
        str(key): int(value)
        for key, value in working.loc[
            ~feature_phase_mask, "phase_normalized"
        ]
        .replace("", "<blank>")
        .value_counts()
        .items()
    }
    cleaned = working.loc[feature_phase_mask].copy()
    if cleaned.empty:
        raise ValueError(
            "Inference input has no Baseline or Exposure rows."
        )

    cleaned[INTERNAL_COLUMNS["timestamp"]] = pd.to_datetime(
        cleaned[mapping["timestamp"]], errors="coerce"
    )
    invalid_timestamp_count = int(
        cleaned[INTERNAL_COLUMNS["timestamp"]].isna().sum()
    )
    if invalid_timestamp_count:
        raise ValueError(
            f"Found {invalid_timestamp_count} invalid timestamps in Baseline/Exposure."
        )

    for canonical in ("hcho", "mq138", "tgs822", "humidity"):
        cleaned[INTERNAL_COLUMNS[canonical]] = _to_numeric_locale(
            cleaned[mapping[canonical]]
        ).replace([np.inf, -np.inf], np.nan)

    # Record timestamp issues as warnings; prediction decides whether to reject them.
    input_warnings: list[str] = []
    timestamp_column = INTERNAL_COLUMNS["timestamp"]
    duplicate_count = int(cleaned[timestamp_column].duplicated().sum())
    if duplicate_count:
        input_warnings.append(f"duplicate_timestamps:{duplicate_count}")

    reversal_count = 0
    for _, phase_group in cleaned.groupby("phase_normalized", sort=False):
        deltas = (
            phase_group.sort_index()[timestamp_column]
            .diff()
            .dt.total_seconds()
            .dropna()
        )
        reversal_count += int((deltas < 0).sum())
    if reversal_count:
        input_warnings.append(f"timestamp_reversals:{reversal_count}")

    # Synthetic metadata only adapts the shared extractor.
    # Zero here does NOT predict that the sample is non-formalin.
    cleaned["concentration_original"] = "<unknown>"
    cleaned["concentration_ml"] = 0.0
    cleaned["replication_id"] = sample_id
    cleaned["label"] = 0
    cleaned["sample_id"] = sample_id

    phase_counts_used = {
        str(key): int(value)
        for key, value in cleaned["phase_normalized"].value_counts().items()
    }
    missing_required_phases = [
        phase
        for phase in ("baseline", "exposure")
        if phase_counts_used.get(phase, 0) == 0
    ]
    if missing_required_phases:
        raise ValueError(
            "missing_required_phase: " + ",".join(missing_required_phases)
        )
    return InferencePreprocessingResult(
        cleaned=cleaned,
        source_rows=int(len(raw)),
        ignored_phase_counts=ignored_phase_counts,
        phase_counts_used=phase_counts_used,
        column_mapping=mapping,
        input_warnings=input_warnings,
    )


def _select_time_window(
    rows: pd.DataFrame,
    timestamp_column: str,
    seconds: float,
    anchor: str,
) -> pd.DataFrame:
    """Select a time window without padding, interpolation, or new rows.

    Head selects the beginning and tail selects the end of a phase. If the
    recording is too short, retain available rows and let QC handle the shortfall."""

    rows = rows.sort_values(timestamp_column)
    if rows.empty or seconds <= 0:
        return rows

    if anchor == "head":
        cutoff = rows[timestamp_column].min() + pd.Timedelta(seconds=seconds)
        return rows.loc[rows[timestamp_column] <= cutoff]
    if anchor == "tail":
        cutoff = rows[timestamp_column].max() - pd.Timedelta(seconds=seconds)
        return rows.loc[rows[timestamp_column] >= cutoff]
    raise ValueError(f"Unknown window anchor: {anchor}")


def _duration_seconds(rows: pd.DataFrame, timestamp_column: str) -> float:
    """Measure the span between the first and last timestamps."""

    if len(rows) < 2:
        return 0.0
    duration = rows[timestamp_column].max() - rows[timestamp_column].min()
    return float(duration.total_seconds())


def _max_positive_gap_seconds(
    rows: pd.DataFrame, timestamp_column: str
) -> float:
    """Find the largest forward gap to detect interrupted recordings."""

    if len(rows) < 2:
        return 0.0
    deltas = (
        rows.sort_values(timestamp_column)[timestamp_column]
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    positive = deltas.loc[deltas > 0]
    return float(positive.max()) if not positive.empty else 0.0


def _median_positive_interval_seconds(
    rows: pd.DataFrame, timestamp_column: str
) -> float:
    """Estimate the typical sampling interval from positive timestamp differences."""

    if len(rows) < 2:
        return 0.0
    deltas = (
        rows.sort_values(timestamp_column)[timestamp_column]
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    positive = deltas.loc[deltas > 0]
    return float(positive.median()) if not positive.empty else 0.0


def _effective_coverage_seconds(
    rows: pd.DataFrame, timestamp_column: str
) -> float:
    """Estimate coverage as the time span plus one typical sampling interval."""

    return _duration_seconds(
        rows, timestamp_column
    ) + _median_positive_interval_seconds(rows, timestamp_column)


def _phase_run_count(phases: pd.Series, target: str) -> int:
    """Count separate phase blocks to prevent merging multiple cycles."""

    matches = phases.eq(target)
    run_starts = matches & ~matches.shift(fill_value=False)
    return int(run_starts.sum())


def _safe_mean(series: pd.Series) -> float:
    """Compute the mean ignoring partial NaNs; return NaN if all values are missing."""

    return float(series.mean()) if series.notna().any() else np.nan


def _safe_max(series: pd.Series) -> float:
    """Compute the maximum ignoring partial NaNs; return NaN if all values are missing."""

    return float(series.max()) if series.notna().any() else np.nan


def extract_features(
    cleaned: pd.DataFrame,
    baseline_seconds: float = 60.0,
    exposure_seconds: float = 120.0,
    baseline_anchor: str = "tail",
    short_window_policy: str = "drop",
) -> FeatureExtractionResult:
    """Extract one row of 13 features per concentration-replication experiment.

    Use the final 60 seconds of Baseline and initial 120 seconds of Exposure.
    Each gas sensor contributes baseline mean, exposure mean, exposure maximum,
    and maximum-minus-baseline delta. Exposure humidity contributes RH_mean.
    The short_window_policy drops, rejects, or keeps short phases with warnings.
    All-25 uses keep without synthetic readings, padding, or interpolation.
    The short baseline of 5mL_rep1 makes its baseline and delta features part
    of a non-strict sensitivity analysis."""

    if short_window_policy not in {"keep", "drop", "error"}:
        raise ValueError(
            "short_window_policy must be one of: keep, drop, error."
        )

    records: list[dict[str, Any]] = []
    excluded_samples: list[dict[str, Any]] = []
    timestamp_column = INTERNAL_COLUMNS["timestamp"]

    # Each concentration-replication pair is one candidate sample.
    # Thus roughly 8,000 raw rows can become just 25 feature rows.
    group_columns = ["concentration_ml", "replication_id"]
    for (concentration_ml, replication_id), group in cleaned.groupby(
        group_columns, sort=False, dropna=False
    ):
        # Reject multiple separate Baseline or Exposure cycles within one pair;
        # independent cycles must not be merged.
        group_in_source_order = group.sort_index()
        baseline_run_count = _phase_run_count(
            group_in_source_order["phase_normalized"], "baseline"
        )
        exposure_run_count = _phase_run_count(
            group_in_source_order["phase_normalized"], "exposure"
        )
        if baseline_run_count > 1 or exposure_run_count > 1:
            excluded_samples.append(
                {
                    "sample_id": str(group["sample_id"].iloc[0]),
                    "reason": "repeated_required_phase_blocks",
                    "details": (
                        f"baseline_blocks={baseline_run_count}, "
                        f"exposure_blocks={exposure_run_count}"
                    ),
                }
            )
            continue

        group = group.sort_values(timestamp_column)
        sample_id = str(group["sample_id"].iloc[0])
        baseline_all = group.loc[group["phase_normalized"] == "baseline"].copy()
        exposure_all = group.loc[group["phase_normalized"] == "exposure"].copy()

        missing_phases = []
        if baseline_all.empty:
            missing_phases.append("baseline")
        if exposure_all.empty:
            missing_phases.append("exposure")
        if missing_phases:
            excluded_samples.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_required_phase",
                    "details": ",".join(missing_phases),
                }
            )
            continue

        # Select the Resume.pdf protocol windows. Do not average across experiments
        # or add synthetic timestamps.
        baseline = _select_time_window(
            baseline_all,
            timestamp_column=timestamp_column,
            seconds=baseline_seconds,
            anchor=baseline_anchor,
        )
        exposure = _select_time_window(
            exposure_all,
            timestamp_column=timestamp_column,
            seconds=exposure_seconds,
            anchor="head",
        )
        if baseline.empty or exposure.empty:
            excluded_samples.append(
                {
                    "sample_id": sample_id,
                    "reason": "empty_analysis_window",
                    "details": (
                        f"baseline_rows={len(baseline)}, exposure_rows={len(exposure)}"
                    ),
                }
            )
            continue

        concentration_original = str(group["concentration_original"].iloc[0])
        label_values = group["label"].drop_duplicates().tolist()
        if len(label_values) != 1:
            raise ValueError(f"Inconsistent labels in sample {sample_id}.")

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "concentration_original": concentration_original,
            "concentration_ml": float(concentration_ml),
            "replication_id": str(replication_id),
            "label": int(label_values[0]),
            "sample_start": group[timestamp_column].min(),
            "sample_end": group[timestamp_column].max(),
            "baseline_rows_total": int(len(baseline_all)),
            "exposure_rows_total": int(len(exposure_all)),
            "baseline_rows_used": int(len(baseline)),
            "exposure_rows_used": int(len(exposure)),
            "baseline_duration_available_seconds": _duration_seconds(
                baseline_all, timestamp_column
            ),
            "exposure_duration_available_seconds": _duration_seconds(
                exposure_all, timestamp_column
            ),
            "baseline_effective_coverage_seconds": _effective_coverage_seconds(
                baseline_all, timestamp_column
            ),
            "exposure_effective_coverage_seconds": _effective_coverage_seconds(
                exposure_all, timestamp_column
            ),
            "baseline_window_duration_seconds": _duration_seconds(
                baseline, timestamp_column
            ),
            "exposure_window_duration_seconds": _duration_seconds(
                exposure, timestamp_column
            ),
            "baseline_window_max_gap_seconds": _max_positive_gap_seconds(
                baseline, timestamp_column
            ),
            "exposure_window_max_gap_seconds": _max_positive_gap_seconds(
                exposure, timestamp_column
            ),
            "baseline_window_start": baseline[timestamp_column].min(),
            "baseline_window_end": baseline[timestamp_column].max(),
            "exposure_window_start": exposure[timestamp_column].min(),
            "exposure_window_end": exposure[timestamp_column].max(),
        }

        # Check duration and time continuity before using features.
        qc_flags: list[str] = []
        if (
            baseline_seconds > 0
            and record["baseline_effective_coverage_seconds"] < baseline_seconds
        ):
            qc_flags.append("baseline_duration_short")
        if (
            exposure_seconds > 0
            and record["exposure_effective_coverage_seconds"] < exposure_seconds
        ):
            qc_flags.append("exposure_duration_short")
        if record["baseline_window_max_gap_seconds"] > 10:
            qc_flags.append("baseline_window_gap_over_10s")
        if record["exposure_window_max_gap_seconds"] > 10:
            qc_flags.append("exposure_window_gap_over_10s")

        # Four features per gas sensor: baseline mean, exposure mean,
        # exposure maximum, and exposure maximum minus baseline mean.
        sensor_specs = [
            ("HCHO", INTERNAL_COLUMNS["hcho"]),
            ("MQ138", INTERNAL_COLUMNS["mq138"]),
            ("TGS822", INTERNAL_COLUMNS["tgs822"]),
        ]
        for feature_prefix, internal_column in sensor_specs:
            baseline_mean = _safe_mean(baseline[internal_column])
            exposure_mean = _safe_mean(exposure[internal_column])
            exposure_max = _safe_max(exposure[internal_column])
            record[f"{feature_prefix}_baseline_mean"] = baseline_mean
            record[f"{feature_prefix}_exposure_mean"] = exposure_mean
            record[f"{feature_prefix}_exposure_max"] = exposure_max
            record[f"{feature_prefix}_delta_max"] = (
                exposure_max - baseline_mean
                if np.isfinite(exposure_max) and np.isfinite(baseline_mean)
                else np.nan
            )
            missing_baseline = int(baseline[internal_column].isna().sum())
            missing_exposure = int(exposure[internal_column].isna().sum())
            record[f"{feature_prefix}_missing_baseline"] = missing_baseline
            record[f"{feature_prefix}_missing_exposure"] = missing_exposure
            if missing_baseline or missing_exposure:
                qc_flags.append(f"{feature_prefix}_missing_values")

        # Summarize humidity as its mean during Exposure only.
        record["RH_mean"] = _safe_mean(exposure[INTERNAL_COLUMNS["humidity"]])
        rh_missing = int(exposure[INTERNAL_COLUMNS["humidity"]].isna().sum())
        record["RH_missing_exposure"] = rh_missing
        if rh_missing:
            qc_flags.append("RH_missing_values")

        missing_features = [
            feature for feature in FEATURE_COLUMNS if pd.isna(record[feature])
        ]
        if missing_features:
            qc_flags.append("feature_imputation_required")

        record["qc_status"] = "warning" if qc_flags else "ok"
        record["qc_flags"] = ";".join(dict.fromkeys(qc_flags))

        # The all-25 keep policy preserves QC warnings in the output
        # so limitations remain visible.
        duration_flags = [
            flag
            for flag in qc_flags
            if flag in {"baseline_duration_short", "exposure_duration_short"}
        ]
        if duration_flags and short_window_policy == "error":
            raise ValueError(
                f"Sample {sample_id} does not meet the Resume.pdf duration requirement: "
                + ",".join(duration_flags)
            )
        if duration_flags and short_window_policy == "drop":
            excluded_samples.append(
                {
                    "sample_id": sample_id,
                    "reason": "insufficient_phase_duration",
                    "details": ";".join(duration_flags),
                    "baseline_effective_coverage_seconds": record[
                        "baseline_effective_coverage_seconds"
                    ],
                    "exposure_effective_coverage_seconds": record[
                        "exposure_effective_coverage_seconds"
                    ],
                }
            )
            continue

        records.append(record)

    if not records:
        raise ValueError(
            "No samples with paired baseline and exposure phases can be processed."
        )

    features = pd.DataFrame.from_records(records).sort_values(
        ["sample_start", "concentration_ml", "replication_id"]
    )
    features = features.reset_index(drop=True)
    return FeatureExtractionResult(
        features=features,
        excluded_samples=excluded_samples,
    )


def parse_pca_components(value: str) -> int | float:
    """Parse a fixed PC count or a target variance fraction between 0 and 1."""

    value = value.strip()
    if re.fullmatch(r"\d+", value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("The PCA component count must be at least 1.")
        return parsed
    try:
        parsed_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "PCA components must be an integer or a proportion between 0 and 1."
        ) from exc
    if not 0 < parsed_float < 1:
        raise argparse.ArgumentTypeError(
            "PCA variance proportion must be greater than 0 and less than 1."
        )
    return parsed_float


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    """Parse ANN architecture; 8,4 specifies layers of 8 and 4 neurons."""

    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be specified as '8' or '8,4'."
        ) from exc
    if not layers or any(layer < 1 for layer in layers):
        raise argparse.ArgumentTypeError(
            "Each hidden layer size must be a positive integer."
        )
    return layers


def build_model(
    *,
    use_pca: bool,
    pca_components: int | float,
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Build a pipeline with median imputation, Z-score scaling, optional PCA, and ANN.

    During cross-validation, all learned preprocessing is fitted only on training
    samples to prevent test leakage. The 13-feature ANN omits the PCA step."""

    # A single sklearn Pipeline keeps training and inference preprocessing aligned.
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    svd_solver="full",
                ),
            )
        )
    steps.append(
        (
            "ann",
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="lbfgs",
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def _binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probability: Sequence[float],
) -> dict[str, float]:
    """Compute binary metrics with class 1 (formalin) as the positive class.

    Accuracy measures overall correctness; balanced accuracy averages class
    recalls. Precision measures positive prediction correctness, recall measures
    formalin detection, specificity measures non-formalin detection, F1 balances
    precision and recall, and AUC evaluates score ranking."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    if len(np.unique(y_true_array)) == 2:
        roc_auc = roc_auc_score(y_true_array, np.asarray(y_probability))
    else:
        roc_auc = np.nan
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "roc_auc": float(roc_auc),
    }


def _validate_binary_problem(y: pd.Series) -> None:
    """Require classes 0 and 1, with at least two samples per class."""

    class_counts = y.value_counts().sort_index()
    if set(class_counts.index) != {0, 1}:
        raise ValueError(
            f"Binary classification requires classes 0 and 1. Found: {class_counts.to_dict()}"
        )
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Each class requires at least two samples for evaluation."
        )


def _cross_validation_splits(
    y: pd.Series,
    replication_groups: pd.Series,
    cv_mode: str,
    cv_folds: int,
    random_state: int,
):
    """Build train-test splits without fitting on held-out samples.

    Replication mode holds out all samples sharing a replication ID together.
    Stratified mode preserves class proportions but does not preserve groups
    and is provided only as a comparison."""

    if cv_mode == "replication":
        if replication_groups.nunique() < 2:
            raise ValueError(
                "Leave-one-replication-out requires at least two distinct replication IDs."
            )
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(np.zeros(len(y)), y, replication_groups))
    elif cv_mode == "stratified":
        smallest_class = int(y.value_counts().min())
        n_splits = min(cv_folds, smallest_class)
        if n_splits < 2:
            raise ValueError("Stratified CV requires at least two folds.")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        raise ValueError(f"Unknown CV mode: {cv_mode}")

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        if len(np.unique(y.iloc[train_index])) < 2:
            raise ValueError(
                f"Fold {fold_number} has only one class in its training data."
            )
    return splits


def evaluate_models(
    feature_table: pd.DataFrame,
    *,
    pca_components: int | float,
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
    cv_mode: str,
    cv_folds: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, Pipeline],
]:
    """Evaluate ANN and PCA-ANN out of fold, then fit final deployment models.

    Each fold fits imputation, scaling, PCA, class weights, and ANN on training
    samples only. Each experiment receives one held-out prediction for metrics
    and confusion matrices. Final models are then refitted on all retained
    samples; reported metrics remain OOF scores, not final-model training scores."""

    # Only these 13 columns are numeric model inputs. Concentration, replication,
    # sample_id, and labels are never included as features.
    X = feature_table[FEATURE_COLUMNS]
    y = feature_table["label"].astype(int)
    groups = feature_table["replication_id"].astype(str)
    _validate_binary_problem(y)

    splits = _cross_validation_splits(
        y=y,
        replication_groups=groups,
        cv_mode=cv_mode,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Compare direct 13-feature ANN and PCA-ANN using identical folds.
    model_templates = {
        "ANN_13_fitur": build_model(
            use_pca=False,
            pca_components=pca_components,
            hidden_layers=hidden_layers,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        ),
        "PCA_ANN": build_model(
            use_pca=True,
            pca_components=pca_components,
            hidden_layers=hidden_layers,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        ),
    }

    prediction_records: list[dict[str, Any]] = []
    fold_metric_records: list[dict[str, Any]] = []

    # Generate held-out predictions for each fold. Compute class weights
    # from y_train only to avoid leaking held-out labels.
    for model_name, template in model_templates.items():
        for fold_number, (train_index, test_index) in enumerate(splits, start=1):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            model = clone(template)
            sample_weight = compute_sample_weight(
                class_weight="balanced", y=y_train
            )
            model.fit(X_train, y_train, ann__sample_weight=sample_weight)
            predicted = model.predict(X_test).astype(int)
            probability = model.predict_proba(X_test)[:, 1]

            fold_metrics = _binary_metrics(y_test, predicted, probability)
            fold_test_groups = sorted(groups.iloc[test_index].unique().tolist())
            fold_metric_records.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "train_samples": int(len(train_index)),
                    "test_samples": int(len(test_index)),
                    "test_replications": ",".join(fold_test_groups),
                    **fold_metrics,
                }
            )

            for local_position, row_index in enumerate(test_index):
                source_row = feature_table.iloc[row_index]
                prediction_records.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "sample_id": source_row["sample_id"],
                        "concentration_ml": source_row["concentration_ml"],
                        "replication_id": source_row["replication_id"],
                        "true_label": int(y_test.iloc[local_position]),
                        "predicted_label": int(predicted[local_position]),
                        "probability_formalin": float(probability[local_position]),
                        "correct": bool(
                            int(predicted[local_position])
                            == int(y_test.iloc[local_position])
                        ),
                    }
                )

    # Combine held-out predictions into an OOF table and compute metrics.
    predictions = pd.DataFrame.from_records(prediction_records)
    fold_metrics = pd.DataFrame.from_records(fold_metric_records)
    summary_records: list[dict[str, Any]] = []
    confusion_matrices: dict[str, np.ndarray] = {}

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "roc_auc",
    ]
    for model_name in model_templates:
        model_predictions = predictions.loc[predictions["model"] == model_name]
        overall = _binary_metrics(
            model_predictions["true_label"],
            model_predictions["predicted_label"],
            model_predictions["probability_formalin"],
        )
        model_fold_metrics = fold_metrics.loc[fold_metrics["model"] == model_name]
        record: dict[str, Any] = {
            "model": model_name,
            "evaluation": (
                "leave-one-replication-out"
                if cv_mode == "replication"
                else "stratified-k-fold"
            ),
            "folds": int(len(model_fold_metrics)),
            "samples": int(len(model_predictions)),
            **overall,
        }
        for metric_name in metric_names:
            record[f"{metric_name}_fold_mean"] = float(
                model_fold_metrics[metric_name].mean()
            )
            record[f"{metric_name}_fold_std"] = float(
                model_fold_metrics[metric_name].std(ddof=1)
            )
        summary_records.append(record)
        confusion_matrices[model_name] = confusion_matrix(
            model_predictions["true_label"],
            model_predictions["predicted_label"],
            labels=[0, 1],
        )

    # After evaluation, fit deployment models on all retained samples.
    # This final fit is not used to calculate OOF metrics.
    final_models: dict[str, Pipeline] = {}
    final_sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    for model_name, template in model_templates.items():
        fitted = clone(template)
        fitted.fit(X, y, ann__sample_weight=final_sample_weight)
        final_models[model_name] = fitted

    return (
        pd.DataFrame.from_records(summary_records),
        fold_metrics,
        predictions,
        confusion_matrices,
        final_models,
    )


def _sha256(path: Path) -> str:
    """Fingerprint file contents with SHA-256.

    A matching hash does not establish pickle safety; use a trusted source."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy values, timestamps, collections, and NaN to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


# Load Matplotlib only when saving training plots. Raspberry Pi inference
# does not need a plotting dependency just to run the model.
def _get_pyplot():
    """Load Matplotlib with a headless backend and return pyplot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save_confusion_matrices(
    matrices: dict[str, np.ndarray], output_path: Path
) -> None:
    """Save OOF confusion matrices with actual rows and predicted columns."""

    plt = _get_pyplot()
    names = list(matrices)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        matrix = matrices[name]
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column] > matrix.max() / 2
                        else "black"
                    ),
                    fontsize=12,
                )
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["0: non-formalin", "1: formalin"],
            yticklabels=["0: non-formalin", "1: formalin"],
            xlabel="Predicted",
            ylabel="Actual",
            title=name.replace("_", " "),
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrix — Out-of-Fold Predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pca_plots(
    pca_scores: pd.DataFrame,
    explained_variance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save descriptive PCA plots from the final model fitted on all samples.

    These plots describe patterns and variance; they are not held-out evaluation."""

    plt = _get_pyplot()
    if {"PC1", "PC2"}.issubset(pca_scores.columns):
        fig, axis = plt.subplots(figsize=(8, 6))
        colours = {0: "#1f77b4", 1: "#d62728"}
        labels = {0: "0: non-formalin", 1: "1: formalin"}
        for label, subset in pca_scores.groupby("label"):
            axis.scatter(
                subset["PC1"],
                subset["PC2"],
                s=65,
                alpha=0.85,
                color=colours[int(label)],
                label=labels[int(label)],
                edgecolor="white",
                linewidth=0.6,
            )
        for _, row in pca_scores.iterrows():
            axis.annotate(
                str(row["sample_id"]),
                (row["PC1"], row["PC2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )
        pc1_variance = explained_variance.loc[
            explained_variance["component"] == "PC1", "explained_variance_ratio"
        ].iloc[0]
        pc2_variance = explained_variance.loc[
            explained_variance["component"] == "PC2", "explained_variance_ratio"
        ].iloc[0]
        axis.set_xlabel(f"PC1 ({pc1_variance:.1%} variance)")
        axis.set_ylabel(f"PC2 ({pc2_variance:.1%} variance)")
        axis.set_title("PCA projection of all samples (final model)")
        axis.axhline(0, color="#bbbbbb", linewidth=0.8)
        axis.axvline(0, color="#bbbbbb", linewidth=0.8)
        axis.legend()
        axis.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        explained_variance["component"],
        explained_variance["explained_variance_ratio"],
        color="#4472C4",
        label="Variance per PC",
    )
    axis.plot(
        explained_variance["component"],
        explained_variance["cumulative_explained_variance"],
        color="#ED7D31",
        marker="o",
        label="Cumulative",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Variance proportion")
    axis.set_title("PCA Explained Variance (final model)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "pca_explained_variance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_sensor_response_plot(
    feature_table: pd.DataFrame, output_path: Path
) -> None:
    """Plot sensor delta mean and standard deviation by concentration.

    Concentration is only a research visualization axis, never a classifier input."""

    plt = _get_pyplot()
    delta_features = [
        "HCHO_delta_max",
        "MQ138_delta_max",
        "TGS822_delta_max",
    ]
    colours = {
        "HCHO_delta_max": "#D62728",
        "MQ138_delta_max": "#2CA02C",
        "TGS822_delta_max": "#1F77B4",
    }
    labels = {
        "HCHO_delta_max": "HCHO",
        "MQ138_delta_max": "MQ138",
        "TGS822_delta_max": "TGS822",
    }
    grouped = feature_table.groupby("concentration_ml")[delta_features].agg(
        ["mean", "std"]
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    x_values = grouped.index.to_numpy(dtype=float)
    for feature in delta_features:
        means = grouped[(feature, "mean")].to_numpy(dtype=float)
        deviations = grouped[(feature, "std")].fillna(0).to_numpy(dtype=float)
        axis.errorbar(
            x_values,
            means,
            yerr=deviations,
            marker="o",
            linewidth=2,
            capsize=4,
            color=colours[feature],
            label=labels[feature],
        )
    axis.set_xlabel("Concentration (mL)")
    axis.set_ylabel("Maximum delta (exposure max − baseline mean)")
    axis.set_title("Sensor response by concentration (mean ± SD across replications)")
    axis.set_xticks(x_values)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    *,
    input_path: Path,
    sheet: str | int,
    output_dir: Path,
    preprocessing: PreprocessingResult,
    extraction: FeatureExtractionResult,
    metrics_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion_matrices: dict[str, np.ndarray],
    final_models: dict[str, Pipeline],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Save audit data, evaluation, PCA, fitted models, and a manifest.

    CSV files support tabular audits and PNG files provide visualizations. Pickle
    and joblib bundles contain the full pipeline fitted on retained samples.
    OOF metrics are stored separately and are not independent field scores.
    Load only trusted pickle artifacts after checking their hashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table = extraction.features

    # Part 1: save cleaned rows, features, exclusions, metrics, OOF predictions,
    # and confusion matrices as auditable CSV tables.
    export_cleaned = preprocessing.cleaned.drop(
        columns=[column for column in INTERNAL_COLUMNS.values() if column in preprocessing.cleaned],
        errors="ignore",
    )
    export_cleaned.to_csv(output_dir / "cleaned_rows.csv", index=False)
    feature_table.to_csv(output_dir / "features_13.csv", index=False)
    excluded_columns = [
        "sample_id",
        "reason",
        "details",
        "baseline_effective_coverage_seconds",
        "exposure_effective_coverage_seconds",
    ]
    pd.DataFrame.from_records(
        extraction.excluded_samples, columns=excluded_columns
    ).to_csv(output_dir / "excluded_samples.csv", index=False)
    concentration_summary = feature_table.groupby("concentration_ml")[
        FEATURE_COLUMNS
    ].agg(["mean", "std"])
    concentration_summary.columns = [
        f"{feature}_{statistic}"
        for feature, statistic in concentration_summary.columns
    ]
    concentration_summary.to_csv(
        output_dir / "feature_summary_by_concentration.csv"
    )
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions_oof.csv", index=False)

    for model_name, matrix in confusion_matrices.items():
        pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(output_dir / f"confusion_matrix_{model_name.lower()}.csv")

    # Part 2: plots support presentation; CSV files remain the canonical values.
    _save_confusion_matrices(
        confusion_matrices, output_dir / "confusion_matrices.png"
    )
    _save_sensor_response_plot(
        feature_table, output_dir / "sensor_delta_by_concentration.png"
    )

    # Part 3: bundles include all preprocessing, so Raspberry Pi users
    # do not need to run PCA separately.
    source_sha256 = _sha256(input_path)
    class_distribution_for_bundle = {
        str(key): int(value)
        for key, value in feature_table["label"].value_counts().sort_index().items()
    }
    metric_lookup = {
        str(row["model"]): row
        for row in metrics_summary.to_dict(orient="records")
    }
    model_artifacts: list[dict[str, Any]] = []
    for model_name, model in final_models.items():
        bundle = {
            "artifact_version": 1,
            "model_name": model_name,
            "pipeline": model,
            "feature_columns": FEATURE_COLUMNS,
            "decision_threshold": 0.5,
            "label_definition": {
                "0": "non-formalin (concentration = 0 mL)",
                "1": "formalin (concentration > 0 mL)",
            },
            "config": config,
            "training_summary": {
                "source_sha256": source_sha256,
                "samples": int(len(feature_table)),
                "class_distribution": class_distribution_for_bundle,
                "excluded_samples": extraction.excluded_samples,
                "oof_metrics": metric_lookup[model_name],
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "security_note": (
                "Load pickle/joblib artifacts only from trusted sources."
            ),
        }
        joblib_path = output_dir / f"model_{model_name.lower()}.joblib"
        pickle_path = output_dir / f"model_{model_name.lower()}.pkl"
        joblib.dump(bundle, joblib_path)
        with pickle_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_artifacts.extend(
            [
                {
                    "model": model_name,
                    "format": "joblib",
                    "path": joblib_path.name,
                    "sha256": _sha256(joblib_path),
                },
                {
                    "model": model_name,
                    "format": "pickle",
                    "path": pickle_path.name,
                    "sha256": _sha256(pickle_path),
                },
            ]
        )

    # Part 4: transform features with the final PCA to obtain descriptive
    # scores, explained variance, and loadings.
    pca_model = final_models["PCA_ANN"]
    X = feature_table[FEATURE_COLUMNS]
    imputed = pca_model.named_steps["imputer"].transform(X)
    scaled = pca_model.named_steps["scaler"].transform(imputed)
    pca_step: PCA = pca_model.named_steps["pca"]
    scores = pca_step.transform(scaled)
    pc_names = [f"PC{index + 1}" for index in range(scores.shape[1])]

    pca_scores = feature_table[METADATA_COLUMNS].copy()
    for index, pc_name in enumerate(pc_names):
        pca_scores[pc_name] = scores[:, index]
    pca_scores.to_csv(output_dir / "pca_scores.csv", index=False)

    explained_variance = pd.DataFrame(
        {
            "component": pc_names,
            "explained_variance_ratio": pca_step.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca_step.explained_variance_ratio_
            ),
        }
    )
    explained_variance.to_csv(
        output_dir / "pca_explained_variance.csv", index=False
    )

    loadings = pd.DataFrame(
        pca_step.components_.T,
        index=FEATURE_COLUMNS,
        columns=pc_names,
    )
    loadings.index.name = "feature"
    loadings.to_csv(output_dir / "pca_loadings.csv")
    _save_pca_plots(pca_scores, explained_variance, output_dir)

    # Part 5: record data provenance, preprocessing, configuration, library
    # versions, and QC warnings in JSON for reproducibility.
    class_distribution = {
        str(key): int(value)
        for key, value in feature_table["label"].value_counts().sort_index().items()
    }
    concentration_distribution = {
        f"{float(key):g} mL": int(value)
        for key, value in feature_table["concentration_ml"]
        .value_counts()
        .sort_index()
        .items()
    }
    qc_distribution = {
        str(key): int(value)
        for key, value in feature_table["qc_status"].value_counts().items()
    }
    metadata = {
        "source": {
            "path": str(input_path.resolve()),
            "sha256": source_sha256,
            "sheet": sheet,
            "rows": preprocessing.source_rows,
            "column_mapping": preprocessing.column_mapping,
        },
        "preprocessing": {
            "dropped_missing_konsentrasi_replikasi_fase": (
                preprocessing.dropped_missing_metadata
            ),
            "rows_after_metadata_filter": (
                preprocessing.rows_after_metadata_filter
            ),
            "ignored_non_feature_phase_rows": int(
                sum(preprocessing.ignored_phase_counts.values())
            ),
            "ignored_phase_counts": preprocessing.ignored_phase_counts,
            "baseline_exposure_rows": int(len(preprocessing.cleaned)),
            "invalid_numeric_counts": preprocessing.invalid_numeric_counts,
            "missing_numeric_counts": preprocessing.missing_numeric_counts,
            "phase_counts_before_filter": (
                preprocessing.phase_counts_before_filter
            ),
            "phase_counts_used": preprocessing.phase_counts,
            "timestamp_quality": preprocessing.timestamp_quality,
        },
        "feature_extraction": {
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "sample_count": int(len(feature_table)),
            "excluded_samples": extraction.excluded_samples,
            "class_distribution": class_distribution,
            "concentration_distribution": concentration_distribution,
            "qc_distribution": qc_distribution,
            "qc_warnings": feature_table.loc[
                feature_table["qc_status"] != "ok",
                ["sample_id", "qc_flags"],
            ].to_dict(orient="records"),
        },
        "model": {
            "comparison": ["ANN_13_fitur", "PCA_ANN"],
            "artifacts": model_artifacts,
            "metrics": metrics_summary.to_dict(orient="records"),
            "pca_explained_variance": explained_variance.to_dict(
                orient="records"
            ),
            "config": config,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, ensure_ascii=False, indent=2)
    # The deployment manifest records artifact hashes. Hashes detect changes;
    # they do not make untrusted pickle files safe.
    model_manifest = {
        "artifact_version": 1,
        "primary_model": "model_pca_ann.pkl",
        "artifact_format": "pickle",
        "source_sha256": source_sha256,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": 0.5,
        "label_definition": {
            "0": "non-formalin (concentration = 0 mL)",
            "1": "formalin (concentration > 0 mL)",
        },
        "resume_contract": {
            "phases_used": ["baseline", "exposure"],
            "purging_used": False,
            "baseline_seconds": config["baseline_seconds"],
            "baseline_anchor": config["baseline_anchor"],
            "exposure_seconds": config["exposure_seconds"],
            "short_window_policy": config["short_window_policy"],
            "features": 13,
            "standardization": "Z-score via StandardScaler",
            "pca_components": config["pca_components"],
            "classifier": "MLPClassifier (binary ANN)",
        },
        "training": {
            "samples": int(len(feature_table)),
            "class_distribution": class_distribution_for_bundle,
            "excluded_samples": extraction.excluded_samples,
            "qc_distribution": metadata["feature_extraction"]["qc_distribution"],
            "qc_warnings": metadata["feature_extraction"]["qc_warnings"],
            "evaluation": metrics_summary.to_dict(orient="records"),
        },
        "environment": metadata["environment"],
        "artifacts": model_artifacts,
        "security_note": (
            "Pickle files can execute code when loaded. Load only trusted artifacts and verify SHA-256."
        ),
    }
    with (output_dir / "model_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe(model_manifest),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return metadata


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Read, clean, extract features, evaluate, refit, and save outputs.

    The all-25 wrapper supplies short_window_policy=keep. Replay, synthetic-input
    checks, and OOF evaluation remain internal checks, not independent field
    validation under new conditions."""

    # Locate the dataset and read the selected sheet.
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()

    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)
    # Clean rows and extract 13 features per experiment.
    preprocessing = preprocess_rows(raw)
    extraction = extract_features(
        preprocessing.cleaned,
        baseline_seconds=args.baseline_seconds,
        exposure_seconds=args.exposure_seconds,
        baseline_anchor=args.baseline_anchor,
        short_window_policy=args.short_window_policy,
    )

    # Limit the PCA size to the smallest training fold.
    if args.cv_mode == "replication":
        largest_test_fold = int(
            extraction.features["replication_id"].value_counts().max()
        )
    else:
        smallest_class = int(
            extraction.features["label"].value_counts().min()
        )
        actual_folds = min(args.cv_folds, smallest_class)
        largest_test_fold = int(math.ceil(len(extraction.features) / actual_folds))
    minimum_training_samples = len(extraction.features) - largest_test_fold
    max_components = min(len(FEATURE_COLUMNS), minimum_training_samples)
    if isinstance(args.pca_components, int) and args.pca_components > max_components:
        raise ValueError(
            f"PCA {args.pca_components} components exceed the fold size. "
            f"Use <= {max_components}."
        )

    # Evaluate both models out of fold and fit final artifacts.
    (
        metrics_summary,
        fold_metrics,
        predictions,
        confusion_matrices,
        final_models,
    ) = evaluate_models(
        extraction.features,
        pca_components=args.pca_components,
        hidden_layers=args.hidden_layers,
        alpha=args.alpha,
        max_iter=args.max_iter,
        random_state=args.random_state,
        cv_mode=args.cv_mode,
        cv_folds=args.cv_folds,
    )

    config = {
        "baseline_seconds": args.baseline_seconds,
        "baseline_anchor": args.baseline_anchor,
        "exposure_seconds": args.exposure_seconds,
        "short_window_policy": args.short_window_policy,
        "pca_components": args.pca_components,
        "hidden_layers": args.hidden_layers,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "cv_mode": args.cv_mode,
        "cv_folds": args.cv_folds,
        "class_balancing": "balanced sample weights, training folds only",
    }
    # Save artifacts and the audit trail.
    metadata = save_outputs(
        input_path=input_path,
        sheet=sheet,
        output_dir=output_dir,
        preprocessing=preprocessing,
        extraction=extraction,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        predictions=predictions,
        confusion_matrices=confusion_matrices,
        final_models=final_models,
        config=config,
    )

    print(f"Pipeline completed. Output: {output_dir}")
    print(
        "Rows: "
        f"{preprocessing.source_rows} source, "
        f"{preprocessing.dropped_missing_metadata} dropped for missing metadata, "
        f"{sum(preprocessing.ignored_phase_counts.values())} Purging/other-phase rows ignored, "
        f"{len(preprocessing.cleaned)} Baseline+Exposure rows used."
    )
    print(
        f"Feature samples: {len(extraction.features)}; "
        f"label distribution: {metadata['feature_extraction']['class_distribution']}"
    )
    print(
        metrics_summary[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
            ]
        ].to_string(index=False)
    )
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    """Define training command-line options.

    The default short-window policy is drop. Use train_all25.py to select keep
    and outputs_all25 explicitly for the 25-experiment variant."""

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess sensor data and perform binary formalin classification with PCA-ANN."
        )
    )
    parser.add_argument(
        "--input",
        default="Data Validasi & Pengujian (1).xlsx",
        help="Path to the XLSX/CSV/TSV dataset.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Excel sheet name or index (default: Data).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory (default: outputs).",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=60.0,
        help="Baseline window duration in seconds; 0 uses the entire phase.",
    )
    parser.add_argument(
        "--baseline-anchor",
        choices=["tail", "head"],
        default="tail",
        help=(
            "Select the baseline window from the phase end (tail, closest to exposure) or beginning (head)."
        ),
    )
    parser.add_argument(
        "--exposure-seconds",
        type=float,
        default=120.0,
        help="Exposure window duration from the phase start; 0 uses the entire phase.",
    )
    parser.add_argument(
        "--short-window-policy",
        choices=["drop", "keep", "error"],
        default="drop",
        help=(
            "Policy for samples shorter than the Resume.pdf protocol: drop (default), keep with warnings, or error."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=parse_pca_components,
        default=3,
        help="Number of PCs (e.g. 3) or target variance (e.g. 0.95).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(8,),
        help="ANN hidden layer sizes, e.g. 8 or 8,4.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="ANN L2 regularization (default: 0.1).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum ANN iterations (default: 5000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Reproducibility seed (default: 42).",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["replication", "stratified"],
        default="replication",
        help=(
            "Leave-one-replication-out evaluation (default, grouped to prevent leakage) or stratified k-fold."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for --cv-mode stratified (default: 5).",
    )
    return parser


def main() -> None:
    """Validate basic arguments and run the training pipeline."""

    parser = build_argument_parser()
    args = parser.parse_args()
    if args.baseline_seconds < 0 or args.exposure_seconds < 0:
        parser.error("Window duration must not be negative.")
    if args.alpha < 0:
        parser.error("Alpha must not be negative.")
    if args.max_iter < 1:
        parser.error("Maximum iterations must be at least 1.")
    if args.cv_folds < 2:
        parser.error("CV requires at least 2 folds.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
