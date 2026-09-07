"""Infer formalin class from one raw sensor experiment.

This module does not train models. It extracts 13 features from Baseline and
Exposure and runs the saved PCA-ANN pipeline. Each call accepts one experiment,
not multiple cycles. Purging is ignored. QC precedes prediction.
probability_formalin is a model class probability, not concentration in mL,
ppm, or any chemical unit. Pickle can execute code: use trusted models and
verify SHA-256 before calling load_model_bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pca_ann_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    preprocess_inference_rows,
    read_dataset,
)


class InputQualityError(ValueError):
    """Indicate readable input that fails quality checks."""

    pass


def file_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load and inspect a trusted PCA-ANN pickle bundle.

    Check required keys, feature order, and pipeline steps. The caller must
    verify the checksum against the manifest before this function opens pickle.
    Args:
        model_path: Path to a trusted .pkl artifact.
    Returns:
        A dictionary containing the pipeline and inference metadata.
    Raises:
        FileNotFoundError: The model file is missing.
        ValueError: The format or structure violates the contract."""

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix.casefold() != ".pkl":
        raise ValueError("This inference program expects a .pkl model.")

    # PICKLE SECURITY: loading can execute embedded code. Verify provenance
    # and checksum BEFORE calling this function. Do not load arbitrary uploads.
    with path.open("rb") as handle:
        bundle = pickle.load(handle)

    # Deployment bundles include the learned imputer, Z-score scaler, PCA,
    # and ANN together, avoiding manual reconstruction during inference.
    required_keys = {
        "artifact_version",
        "model_name",
        "pipeline",
        "feature_columns",
        "decision_threshold",
        "config",
    }
    missing = sorted(required_keys - set(bundle))
    if missing:
        raise ValueError(f"Incomplete model bundle; missing keys: {missing}")
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature order does not match the 13-feature contract."
        )

    pipeline = bundle["pipeline"]
    expected_steps = ["imputer", "scaler", "pca", "ann"]
    if list(pipeline.named_steps) != expected_steps:
        raise ValueError(
            "Invalid PCA-ANN pipeline. "
            f"Expected {expected_steps}, found {list(pipeline.named_steps)}."
        )
    return bundle


def _json_safe(value: Any) -> Any:
    """Convert NumPy and pandas values to JSON-safe types."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def predict_dataframe(
    raw: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    sample_id: str = "inference_sample",
    allow_qc_warnings: bool = False,
) -> dict[str, Any]:
    """Predict one experiment from raw sensor rows.

    Args:
        raw: Timestamp, Fase, HCHO, MQ-138, TGS822, and HUMIDITY readings.
            Concentration is not required.
        bundle: Model bundle returned by load_model_bundle.
        sample_id: Audit identifier for this experiment.
        allow_qc_warnings: Permit warnings for diagnostics only.
    Returns:
        Predicted class, formalin probability, QC, windows, and 13 features.
    Raises:
        InputQualityError: Extraction does not yield exactly one experiment,
            or QC fails while warnings are disallowed.
        ValueError: Required columns, phases, or basic values are invalid."""

    # ONE FILE = ONE EXPERIMENT. Normalize headers and phases, retaining
    # Baseline and Exposure only. Reject multiple cycles instead of
    # silently choosing one.
    preprocessing = preprocess_inference_rows(raw, sample_id=sample_id)
    config = bundle["config"]

    # Use the same windows and 13-feature formulas as training. Keep only
    # allows calculation of all QC flags; warnings are rejected below unless
    # allow_qc_warnings explicitly enables diagnostics.
    extraction = extract_features(
        preprocessing.cleaned,
        baseline_seconds=float(config["baseline_seconds"]),
        exposure_seconds=float(config["exposure_seconds"]),
        baseline_anchor=str(config["baseline_anchor"]),
        short_window_policy="keep",
    )
    if extraction.excluded_samples:
        raise InputQualityError(
            "Input cannot be extracted as one experiment: "
            + json.dumps(extraction.excluded_samples, ensure_ascii=False)
        )
    if len(extraction.features) != 1:
        raise InputQualityError(
            f"Inference requires exactly one experiment; found {len(extraction.features)}."
        )

    # QC guards against short recordings, timestamp problems, large gaps,
    # and missing sensor readings. Questionable inputs receive no class by default.
    feature_row = extraction.features.iloc[0]
    qc_text = feature_row.get("qc_flags", "")
    qc_flags = (
        []
        if pd.isna(qc_text) or not str(qc_text).strip()
        else str(qc_text).split(";")
    )
    all_warnings = list(dict.fromkeys(preprocessing.input_warnings + qc_flags))
    if all_warnings and not allow_qc_warnings:
        raise InputQualityError(
            "Input failed quality checks: " + ", ".join(all_warnings)
        )

    feature_frame = pd.DataFrame(
        [[float(feature_row[column]) for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    pipeline = bundle["pipeline"]

    # Run imputer -> StandardScaler -> PCA -> ANN. The result is a formalin
    # CLASS probability, not a measured chemical concentration.
    probability_formalin = float(pipeline.predict_proba(feature_frame)[0, 1])
    threshold = float(bundle.get("decision_threshold", 0.5))
    predicted_label = int(probability_formalin >= threshold)

    # Return QC, windows, and features so each decision can be audited.
    result = {
        "sample_id": sample_id,
        "model_name": bundle["model_name"],
        "predicted_label": predicted_label,
        "predicted_class": (
            "formalin" if predicted_label == 1 else "non-formalin"
        ),
        "probability_formalin": probability_formalin,
        "decision_threshold": threshold,
        "qc_status": "warning" if all_warnings else "ok",
        "qc_warnings": all_warnings,
        "input_rows": int(preprocessing.source_rows),
        "ignored_phase_counts": preprocessing.ignored_phase_counts,
        "phase_counts_used": preprocessing.phase_counts_used,
        "window": {
            "baseline_rows_total": int(feature_row["baseline_rows_total"]),
            "baseline_rows_used": int(feature_row["baseline_rows_used"]),
            "baseline_effective_coverage_seconds": float(
                feature_row["baseline_effective_coverage_seconds"]
            ),
            "exposure_rows_total": int(feature_row["exposure_rows_total"]),
            "exposure_rows_used": int(feature_row["exposure_rows_used"]),
            "exposure_effective_coverage_seconds": float(
                feature_row["exposure_effective_coverage_seconds"]
            ),
        },
        "features": {
            column: float(feature_row[column]) for column in FEATURE_COLUMNS
        },
    }
    return _json_safe(result)


def build_parser() -> argparse.ArgumentParser:
    """Build the inference command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict formalin from one raw Baseline/Exposure recording without a concentration column."
        )
    )
    parser.add_argument("--input", required=True, help="XLSX/CSV/TSV file containing one experiment.")
    parser.add_argument(
        "--model",
        default="outputs/model_pca_ann.pkl",
        help="Trusted PCA-ANN .pkl model.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Sheet name or index for Excel input.",
    )
    parser.add_argument(
        "--sample-id",
        default="inference_sample",
        help="Identifier for this experiment.",
    )
    parser.add_argument(
        "--output",
        help="Optionally save the prediction as JSON.",
    )
    parser.add_argument(
        "--allow-qc-warnings",
        action="store_true",
        help="Allow prediction when duration, gaps, or sensor values trigger warnings.",
    )
    return parser


def main() -> None:
    """Read one recording, load the model, and print inference JSON."""

    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)
    model_path = Path(args.model).expanduser().resolve()

    # The CLI expects a trusted local model. It reports a hash for auditing;
    # the caller must still verify the manifest before opening pickle.
    bundle = load_model_bundle(model_path)
    result = predict_dataframe(
        raw,
        bundle,
        sample_id=args.sample_id,
        allow_qc_warnings=args.allow_qc_warnings,
    )
    result["model_path"] = str(model_path)
    result["model_sha256"] = file_sha256(model_path)
    result["input_path"] = str(input_path)

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
