

from __future__ import annotations

import argparse
from pathlib import Path

from deployment_validation import run_validation
from pca_ann_pipeline import run_pipeline
from seed_stability_analysis import run_analysis


# Use all-25 CLI defaults so users need not specify every training parameter.
def build_parser() -> argparse.ArgumentParser:
    """Define command-line options with 25-experiment defaults."""

    parser = argparse.ArgumentParser(
        description=(
            "Complete PCA-ANN training using all 25 experiments. Short-baseline samples are retained with QC warnings."
        )
    )
    parser.add_argument(
        "--input",
        default="Data Validasi & Pengujian (1).xlsx",
        help="Raw XLSX/CSV/TSV file to process.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Excel sheet containing raw data.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_all25",
        help="Directory for models, evaluation tables, and plots.",
    )
    parser.add_argument(
        "--skip-seed-test",
        action="store_true",
        help="Skip the 10-seed stability analysis.",
    )
    parser.add_argument(
        "--skip-deployment-test",
        action="store_true",
        help="Skip replay, synthetic-input, and invalid-input model checks.",
    )
    return parser


# Build the pipeline configuration with short_window_policy=keep.
# Use available data without fabricating or extending short phases,
# and retain the associated QC warnings.
def make_pipeline_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Build the research-protocol configuration for the all-25 variant."""

    return argparse.Namespace(
        input=args.input,
        sheet=args.sheet,
        output_dir=args.output_dir,
        baseline_seconds=60.0,
        baseline_anchor="tail",
        exposure_seconds=120.0,
        short_window_policy="keep",
        pca_components=3,
        hidden_layers=(8,),
        alpha=0.1,
        max_iter=5000,
        random_state=42,
        cv_mode="replication",
        cv_folds=5,
    )


# Check file availability before training to report a clear missing-dataset error.
def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Check dataset availability and resolve the output directory."""

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    return input_path, output_dir


# Run preprocessing, features, cross-validation, final training, and export;
# then test seed stability and saved-model inference. Seed, replay, and
# synthetic-input checks do not replace new independent field data.
def run_all25_workflow(args: argparse.Namespace) -> dict:
    """Run training, seed stability, and deployment smoke tests."""

    input_path, output_dir = validate_paths(args)
    pipeline_arguments = make_pipeline_arguments(args)
    metadata = run_pipeline(pipeline_arguments)

    if not args.skip_seed_test:
        run_analysis(
            feature_path=output_dir / "features_13.csv",
            output_dir=output_dir,
            seeds=[7, 11, 19, 23, 31, 42, 53, 67, 79, 97],
        )

    if not args.skip_deployment_test:
        run_validation(
            workbook_path=input_path,
            model_path=output_dir / "model_pca_ann.pkl",
            output_dir=output_dir / "deployment_tests",
        )

    return metadata


# Parse user options, run the workflow, and print key artifact locations.
def main() -> None:
    """Parse arguments, run the workflow, and print key artifact locations."""

    args = build_parser().parse_args()
    metadata = run_all25_workflow(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_count = metadata["feature_extraction"]["sample_count"]
    class_distribution = metadata["feature_extraction"]["class_distribution"]
    print(f"All-25 training completed: {sample_count} samples.")
    print(f"Class distribution: {class_distribution}")
    print(f"Main model: {output_dir / 'model_pca_ann.pkl'}")
    print(f"Metrics: {output_dir / 'metrics_summary.csv'}")
    print(f"Confusion matrix: {output_dir / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
