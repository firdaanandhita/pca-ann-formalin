"""Run PCA-ANN inference on raw Excel data on Raspberry Pi.

Function docstrings and nearby comments explain each processing step."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pca_ann_pipeline import parse_concentration_ml, read_dataset, resolve_columns
from predict_raw import InputQualityError, load_model_bundle, predict_dataframe


# Compute SHA-256 to verify that the copied model matches its manifest
# and was not corrupted during transfer.
def calculate_sha256(path: Path) -> str:
    """Compute a SHA-256 fingerprint for file integrity verification."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Read model_manifest.json and compare hashes before opening pickle.
# Only load trusted artifacts because pickle can execute code.
def verify_model_before_loading(model_path: Path, manifest_path: Path) -> str:
    """Verify the model hash against the manifest before opening pickle."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("path") == model_path.name
        ),
        None,
    )
    if artifact is None:
        raise ValueError(
            f"Hash for {model_path.name} was not found in the manifest."
        )

    actual_hash = calculate_sha256(model_path)
    expected_hash = str(artifact["sha256"]).casefold()
    if actual_hash.casefold() != expected_hash:
        raise ValueError(
            "Model hash mismatch. Do not load the model: the file may be corrupted, modified, or the wrong artifact."
        )
    return actual_hash


# Map missing or invalid concentration cells to NaN so demo selection
# excludes them. Concentration selects a known experiment only;
# it is never passed to the model as a feature.
def parse_concentration_or_nan(value: Any) -> float:
    """Parse demo concentration, returning NaN for invalid values."""

    try:
        return parse_concentration_ml(value)
    except (TypeError, ValueError):
        return np.nan


# Select one concentration-replication pair from a multi-experiment workbook
# for demonstration. Production files must contain exactly one cycle
# and do not need this selection step.
def select_demo_experiment(
    raw: pd.DataFrame,
    concentration_ml: float,
    replication: int,
) -> pd.DataFrame:
    """Select one concentration-replication pair from the original workbook."""

    mapping = resolve_columns(
        raw.columns,
        required=["concentration", "replication"],
    )
    concentrations = raw[mapping["concentration"]].map(parse_concentration_or_nan)
    replications = pd.to_numeric(
        raw[mapping["replication"]],
        errors="coerce",
    )
    selected = raw.loc[
        concentrations.eq(float(concentration_ml))
        & replications.eq(float(replication))
    ].copy()
    if selected.empty:
        raise ValueError(
            "Demo experiment not found for "
            f"{concentration_ml:g} mL replication {replication}."
        )
    return selected


# Demo options must be supplied together. Without them, treat the
# whole file as one new production recording.
def build_parser() -> argparse.ArgumentParser:
    """Define production and training-dataset demo command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Raspberry Pi example for predicting one raw Excel recording with the all-25 PCA-ANN model."
        )
    )
    parser.add_argument("--input", required=True, help="Raw XLSX/CSV/TSV file.")
    parser.add_argument("--sheet", default="Data", help="Input Excel sheet name.")
    parser.add_argument(
        "--model",
        default="outputs_all25/model_pca_ann.pkl",
        help="All-25 PCA-ANN model file.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs_all25/model_manifest.json",
        help="Manifest containing the expected model hash.",
    )
    parser.add_argument(
        "--sample-id",
        default="raspi_measurement",
        help="Measurement identifier included in the result.",
    )
    parser.add_argument(
        "--output",
        default="raspi_prediction_result.json",
        help="JSON file for saving prediction results.",
    )
    parser.add_argument(
        "--demo-concentration-ml",
        type=float,
        help="Concentration to select from a multi-experiment workbook.",
    )
    parser.add_argument(
        "--demo-replication",
        type=int,
        help="Replication number to select from a multi-experiment workbook.",
    )
    parser.add_argument(
        "--allow-qc-warnings",
        action="store_true",
        help=(
            "Allow prediction despite QC warnings. Use only for diagnostics, not field decisions."
        ),
    )
    return parser


# Validate arguments, verify hashes, read input, optionally select a demo,
# extract 13 features, run the pipeline, and save JSON.
def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    """Verify artifacts, extract features, predict, and export JSON."""

    model_path = Path(args.model).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    demo_options = (
        args.demo_concentration_ml is not None,
        args.demo_replication is not None,
    )
    if demo_options[0] != demo_options[1]:
        raise ValueError(
            "--demo-concentration-ml and --demo-replication must be used together."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    model_hash = verify_model_before_loading(model_path, manifest_path)
    bundle = load_model_bundle(model_path)
    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)

    mode = "production"
    expected_label: int | None = None
    if all(demo_options):
        raw = select_demo_experiment(
            raw,
            concentration_ml=float(args.demo_concentration_ml),
            replication=int(args.demo_replication),
        )
        mode = "training_dataset_demo"
        # Use the known label only to check the selected demo result.
        # Do not pass it to predict_dataframe or the model pipeline.
        expected_label = int(float(args.demo_concentration_ml) > 0)

    result = predict_dataframe(
        raw,
        bundle,
        sample_id=args.sample_id,
        allow_qc_warnings=args.allow_qc_warnings,
    )
    result["mode"] = mode
    result["model_sha256"] = model_hash
    result["input_sha256"] = calculate_sha256(input_path)
    result["input_path"] = str(input_path)
    result["model_path"] = str(model_path)
    if mode == "training_dataset_demo":
        result["demo_warning"] = (
            "This recording comes from the training dataset and is not independent validation."
        )
        result["expected_label_from_metadata"] = expected_label
        result["prediction_matches_expected_label"] = (
            result["predicted_label"] == expected_label
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


# Report clear CLI errors for schema, model integrity, and recording QC failures.
def main() -> None:
    """Run the CLI and display readable input-quality errors."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_prediction(args)
    except InputQualityError as exc:
        parser.exit(4, f"Input QC failed: {exc}\n")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(2, f"Invalid input or artifact: {exc}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
