"""Check technical readiness of the PCA-ANN inference pipeline.

Verify model reload, consistent 13-feature extraction, Purging invariance,
invalid-input rejection, and two end-to-end synthetic smoke tests derived
from existing recordings. Passing does not establish field validity: new
days, devices, or batches are needed. For all-25, use the model and matching
features_13.csv in outputs_all25."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pca_ann_pipeline import FEATURE_COLUMNS, read_dataset, resolve_columns
from predict_raw import (
    InputQualityError,
    file_sha256,
    load_model_bundle,
    predict_dataframe,
)


def _actual_run(
    raw: pd.DataFrame, concentration: str, replication: int
) -> pd.DataFrame:
    """Select an existing recording for technical replay.

    Concentration and replication locate a known experiment. Returned columns
    exclude concentration and labels, matching unlabeled production inputs."""

    mapping = resolve_columns(raw.columns)
    mask = (
        raw[mapping["concentration"]].astype("string").str.strip().eq(concentration)
        & pd.to_numeric(raw[mapping["replication"]], errors="coerce").eq(
            replication
        )
    )
    selected = raw.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            f"Run not found: concentration={concentration}, replication={replication}"
        )
    keep = [
        mapping["timestamp"],
        mapping["hcho"],
        mapping["mq138"],
        mapping["tgs822"],
        mapping["humidity"],
        mapping["phase"],
    ]
    return selected[keep].copy()


def _window_rows(
    phase_rows: pd.DataFrame,
    timestamp_column: str,
    seconds: float,
    anchor: str,
) -> pd.DataFrame:
    """Select the beginning or end of a phase by timestamp.

    Tail selects the end of Baseline nearest exposure; the other internal anchor
    selects the start of Exposure. No interpolation or extra rows are introduced."""

    rows = phase_rows.sort_values(timestamp_column).copy()
    timestamps = pd.to_datetime(rows[timestamp_column], errors="raise")
    if anchor == "tail":
        return rows.loc[timestamps >= timestamps.max() - pd.Timedelta(seconds=seconds)]
    return rows.loc[timestamps <= timestamps.min() + pd.Timedelta(seconds=seconds)]


def _resample_phase(
    phase_rows: pd.DataFrame,
    *,
    mapping: dict[str, str],
    phase_name: str,
    periods: int,
    start_timestamp: pd.Timestamp,
    rng: np.random.Generator,
    noise_fraction: float,
) -> pd.DataFrame:
    """Create a synthetic phase at one-second intervals from an existing pattern.

    Interpolate sensor readings and add small controlled random noise. This is
    a derivative of an existing recording, not a new sample measurement."""

    timestamp_column = mapping["timestamp"]
    rows = phase_rows.sort_values(timestamp_column).copy()
    timestamps = pd.to_datetime(rows[timestamp_column], errors="raise")
    elapsed = (timestamps - timestamps.min()).dt.total_seconds().to_numpy()
    target = np.arange(periods, dtype=float)

    output: dict[str, Any] = {
        "Timestamp": pd.date_range(start_timestamp, periods=periods, freq="1s"),
        "Fase": phase_name,
    }
    output_names = {
        "hcho": "HCHO",
        "mq138": "MQ-138",
        "tgs822": "TGS822",
        "humidity": "HUMIDITY",
    }

    # Apply the same treatment to gas sensors and humidity. Interpolate partial
    # missing values along the phase pattern; reject an entirely missing sensor.
    for canonical, output_name in output_names.items():
        values = pd.to_numeric(rows[mapping[canonical]], errors="coerce")
        values = values.interpolate(limit_direction="both")
        if values.isna().all():
            raise ValueError(f"Sensor {canonical} is entirely missing in the synthetic input source.")
        interpolated = np.interp(
            target,
            elapsed,
            values.to_numpy(dtype=float),
        )
        scale = max(
            float(np.std(interpolated)),
            abs(float(np.mean(interpolated))) * 0.01,
            1e-6,
        )

        # Scale noise to signal variation, with a floor for nearly flat signals.
        noisy = interpolated + rng.normal(
            loc=0.0,
            scale=scale * noise_fraction,
            size=periods,
        )
        output[output_name] = noisy
    return pd.DataFrame(output)


def make_noisy_dummy(
    actual: pd.DataFrame,
    *,
    seed: int,
    noise_fraction: float = 0.02,
) -> pd.DataFrame:
    """Build a synthetic cycle with 60 Baseline and 120 Exposure rows.

    Use the original protocol windows, resample to 1 Hz, and assign fictional
    2030 timestamps. The seed makes the input reproducible. Correct predictions
    only exercise preprocessing and inference, not generalization to field samples."""

    required = [
        "timestamp",
        "hcho",
        "mq138",
        "tgs822",
        "humidity",
        "phase",
    ]
    mapping = resolve_columns(actual.columns, required=required)
    phases = actual[mapping["phase"]].astype("string").str.strip().str.casefold()
    baseline_all = actual.loc[phases.eq("baseline")].copy()
    exposure_all = actual.loc[phases.eq("exposure")].copy()
    if baseline_all.empty or exposure_all.empty:
        raise ValueError("Synthetic input source must contain Baseline and Exposure.")

    # Use the training windows: final 60 seconds of Baseline and
    # initial 120 seconds of Exposure.
    baseline = _window_rows(
        baseline_all, mapping["timestamp"], seconds=60, anchor="tail"
    )
    exposure = _window_rows(
        exposure_all, mapping["timestamp"], seconds=120, anchor="head"
    )
    rng = np.random.default_rng(seed)

    # Use fictional timestamps: features depend on sensor patterns and phase
    # sequence, not calendar dates.
    start = pd.Timestamp("2030-01-01 00:00:00")
    baseline_dummy = _resample_phase(
        baseline,
        mapping=mapping,
        phase_name="Baseline",
        periods=60,
        start_timestamp=start,
        rng=rng,
        noise_fraction=noise_fraction,
    )
    exposure_dummy = _resample_phase(
        exposure,
        mapping=mapping,
        phase_name="Exposure",
        periods=120,
        start_timestamp=start + pd.Timedelta(seconds=60),
        rng=rng,
        noise_fraction=noise_fraction,
    )
    return pd.concat([baseline_dummy, exposure_dummy], ignore_index=True)


def _expect_failure(callback, expected_text: str) -> dict[str, Any]:
    """Check that invalid input raises an exception containing expected_text.

    This tests input quality guards rather than classification accuracy."""

    try:
        callback()
    except Exception as exc:
        message = str(exc)
        return {
            "passed": expected_text.casefold() in message.casefold(),
            "exception": type(exc).__name__,
            "message": message,
        }
    return {
        "passed": False,
        "exception": None,
        "message": "Input was not rejected.",
    }


def run_validation(
    *,
    workbook_path: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run deployment checks and save their report.

    Check replay feature parity, known examples, Purging invariance, noisy inputs,
    invalid-input rejection, and pickle reload consistency. Write synthetic CSVs
    and deployment_test_results.json; raise AssertionError if any check fails.
    Read features_13.csv beside the model so both artifacts use the same workflow."""

    # Step 1: load the model, raw data, and matching feature table
    # from the same workflow.
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(model_path)
    raw = read_dataset(workbook_path, sheet="Data")
    training_features = pd.read_csv(model_path.parent / "features_13.csv")

    replay_feature_max_difference = 0.0
    replay_feature_rows_checked = 0

    # Step 2: replay every training experiment and recompute its 13 features.
    # Require parity with features_13.csv. Permit warnings only for this check
    # so all-25 sample 5mL_rep1 is audited without hiding its QC warnings.
    for _, expected_row in training_features.iterrows():
        concentration = f"{float(expected_row['concentration_ml']):g} mL"
        replication = int(float(expected_row["replication_id"]))
        replay_raw = _actual_run(raw, concentration, replication)
        replay_result = predict_dataframe(
            replay_raw,
            bundle,
            sample_id=f"parity_{concentration}_{replication}",
            allow_qc_warnings=True,
        )
        expected_values = expected_row[FEATURE_COLUMNS].to_numpy(dtype=float)
        actual_values = np.array(
            [replay_result["features"][feature] for feature in FEATURE_COLUMNS],
            dtype=float,
        )
        replay_feature_max_difference = max(
            replay_feature_max_difference,
            float(np.max(np.abs(expected_values - actual_values))),
        )
        replay_feature_rows_checked += 1

    # Step 3: replay known non-formalin and formalin examples.
    # These are existing recordings, not independent tests.
    replay_non = _actual_run(raw, concentration="0 mL", replication=5)
    replay_formal = _actual_run(raw, concentration="15 mL", replication=5)
    replay_non_result = predict_dataframe(
        replay_non, bundle, sample_id="replay_0mL_rep5"
    )
    replay_formal_result = predict_dataframe(
        replay_formal, bundle, sample_id="replay_15mL_rep5"
    )

    # Step 4: predictions must be invariant to removing Purging because
    # only Baseline and Exposure contribute features.
    replay_non_without_purging = replay_non.loc[
        replay_non["Fase"].astype("string").str.casefold().isin(
            ["baseline", "exposure"]
        )
    ].copy()
    replay_non_no_purge_result = predict_dataframe(
        replay_non_without_purging,
        bundle,
        sample_id="replay_0mL_rep5_no_purging",
    )
    purging_probability_difference = abs(
        replay_non_result["probability_formalin"]
        - replay_non_no_purge_result["probability_formalin"]
    )

    # Step 5: save two noisy synthetic inputs derived from existing recordings
    # so the smoke tests can be inspected and repeated.
    dummy_non = make_noisy_dummy(replay_non, seed=42)
    dummy_formal = make_noisy_dummy(replay_formal, seed=43)
    dummy_non_path = output_dir / "dummy_non_formalin.csv"
    dummy_formal_path = output_dir / "dummy_formalin.csv"
    dummy_non.to_csv(dummy_non_path, index=False)
    dummy_formal.to_csv(dummy_formal_path, index=False)
    dummy_non_result = predict_dataframe(
        dummy_non, bundle, sample_id="dummy_non_formalin"
    )
    dummy_formal_result = predict_dataframe(
        dummy_formal, bundle, sample_id="dummy_formalin"
    )

    # Step 6: build deliberately invalid inputs. Reject recordings that violate
    # phase, duration, or sensor requirements instead of forcing predictions.
    short_baseline = pd.concat(
        [
            dummy_non.loc[dummy_non["Fase"].eq("Baseline")].head(30),
            dummy_non.loc[dummy_non["Fase"].eq("Exposure")],
        ],
        ignore_index=True,
    )
    missing_exposure = dummy_non.loc[dummy_non["Fase"].eq("Baseline")].copy()
    missing_sensor = dummy_non.drop(columns=["HCHO"])
    only_purging = replay_non.loc[
        replay_non["Fase"].astype("string").str.casefold().eq("purging")
    ].copy()
    all_hcho_missing = dummy_non.copy()
    all_hcho_missing["HCHO"] = np.nan
    multiple_cycles = pd.concat(
        [
            dummy_non,
            dummy_non.assign(
                Timestamp=pd.to_datetime(dummy_non["Timestamp"])
                + pd.Timedelta(hours=1)
            ),
        ],
        ignore_index=True,
    )

    # Inference extraction uses keep to calculate QC, but predict_dataframe
    # rejects warnings by default. The short-baseline input must fail.
    short_check = _expect_failure(
        lambda: predict_dataframe(
            short_baseline, bundle, sample_id="invalid_short_baseline"
        ),
        "baseline_duration_short",
    )
    phase_check = _expect_failure(
        lambda: predict_dataframe(
            missing_exposure, bundle, sample_id="invalid_missing_exposure"
        ),
        "missing_required_phase",
    )
    sensor_check = _expect_failure(
        lambda: predict_dataframe(
            missing_sensor, bundle, sample_id="invalid_missing_hcho"
        ),
        "hcho",
    )
    purging_only_check = _expect_failure(
        lambda: predict_dataframe(
            only_purging, bundle, sample_id="invalid_only_purging"
        ),
        "baseline",
    )
    empty_check = _expect_failure(
        lambda: predict_dataframe(
            dummy_non.iloc[0:0].copy(),
            bundle,
            sample_id="invalid_empty",
        ),
        "baseline",
    )
    missing_values_check = _expect_failure(
        lambda: predict_dataframe(
            all_hcho_missing,
            bundle,
            sample_id="invalid_hcho_nan",
        ),
        "feature_imputation_required",
    )
    multiple_cycles_check = _expect_failure(
        lambda: predict_dataframe(
            multiple_cycles,
            bundle,
            sample_id="invalid_multiple_cycles",
        ),
        "no samples",
    )

    # Step 7: reload pickle and check identical probabilities.
    # This checks serialization, not field accuracy.
    reloaded_bundle = load_model_bundle(model_path)
    reloaded_result = predict_dataframe(
        dummy_formal, reloaded_bundle, sample_id="dummy_formalin_reload"
    )
    reload_probability_difference = abs(
        dummy_formal_result["probability_formalin"]
        - reloaded_result["probability_formalin"]
    )

    metrics_path = model_path.parent / "metrics_summary.csv"
    metrics = pd.read_csv(metrics_path).to_dict(orient="records")

    # Step 8: summarize checks as booleans. Keep replay and synthetic tests
    # separate from OOF training metrics.
    checks = {
        "pickle_loaded": True,
        "feature_contract_13": len(bundle["feature_columns"]) == 13,
        "replay_non_formalin_correct": (
            replay_non_result["predicted_label"] == 0
        ),
        "replay_formalin_correct": (
            replay_formal_result["predicted_label"] == 1
        ),
        "dummy_non_formalin_correct": (
            dummy_non_result["predicted_label"] == 0
        ),
        "dummy_formalin_correct": (
            dummy_formal_result["predicted_label"] == 1
        ),
        "all_replay_features_identical": (
            replay_feature_rows_checked == len(training_features)
            and replay_feature_max_difference <= 1e-12
        ),
        "purging_invariant": purging_probability_difference <= 1e-12,
        "short_baseline_rejected": short_check["passed"],
        "missing_exposure_rejected": phase_check["passed"],
        "missing_sensor_rejected": sensor_check["passed"],
        "purging_only_rejected": purging_only_check["passed"],
        "empty_input_rejected": empty_check["passed"],
        "all_nan_sensor_rejected": missing_values_check["passed"],
        "multiple_cycles_rejected": multiple_cycles_check["passed"],
        "pickle_reload_identical": reload_probability_difference <= 1e-12,
    }
    all_passed = all(checks.values())

    # State interpretation limits so smoke-test success is not mistaken
    # for real-world validation.
    report = {
        "all_technical_checks_passed": all_passed,
        "checks": checks,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": file_sha256(model_path),
            "model_name": bundle["model_name"],
            "feature_count": len(bundle["feature_columns"]),
        },
        "out_of_fold_real_data_evaluation": metrics,
        "replay_results": {
            "non_formalin": replay_non_result,
            "formalin": replay_formal_result,
        },
        "dummy_results": {
            "non_formalin": dummy_non_result,
            "formalin": dummy_formal_result,
            "files": [str(dummy_non_path), str(dummy_formal_path)],
        },
        "invariance": {
            "purging_probability_difference": purging_probability_difference,
            "pickle_reload_probability_difference": (
                reload_probability_difference
            ),
            "replay_feature_rows_checked": replay_feature_rows_checked,
            "replay_feature_max_absolute_difference": (
                replay_feature_max_difference
            ),
        },
        "negative_tests": {
            "short_baseline": short_check,
            "missing_exposure": phase_check,
            "missing_hcho": sensor_check,
            "purging_only": purging_only_check,
            "empty_input": empty_check,
            "all_hcho_nan": missing_values_check,
            "multiple_cycles": multiple_cycles_check,
        },
        "scope_limitations": [
            (
                "Replay and synthetic tests verify the input contract, feature extraction, serialization, and inference execution; not field generalization."
            ),
            (
                "Real-world validation requires external data never used for training, ideally from different days, devices, or batches."
            ),
        ],
    }
    report_path = output_dir / "deployment_test_results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Do not report success when a guard fails. Include failed check names
    # in the raised exception.
    if not all_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Deployment checks failed: {failed}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Define workbook, model, and report options.

    Defaults target outputs. For all-25, select outputs_all25/model_pca_ann.pkl
    and an appropriate output directory."""

    parser = argparse.ArgumentParser(description="Validate PCA-ANN deployment.")
    parser.add_argument(
        "--workbook",
        default="Data Validasi & Pengujian (1).xlsx",
    )
    parser.add_argument(
        "--model",
        default="outputs/model_pca_ann.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deployment_tests",
    )
    return parser


def main() -> None:
    """Run validation and print check status and synthetic predictions.

    run_validation saves the complete report."""

    args = build_parser().parse_args()
    report = run_validation(
        workbook_path=Path(args.workbook).expanduser().resolve(),
        model_path=Path(args.model).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "all_technical_checks_passed": report[
                    "all_technical_checks_passed"
                ],
                "checks": report["checks"],
                "dummy_results": {
                    key: {
                        "predicted_class": value["predicted_class"],
                        "probability_formalin": value["probability_formalin"],
                    }
                    for key, value in report["dummy_results"].items()
                    if key in {"non_formalin", "formalin"}
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
