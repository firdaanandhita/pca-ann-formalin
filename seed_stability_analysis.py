"""Check ANN stability across random initializations.

random_state determines initial weights, so identical data and configuration
can yield different results across seeds. Repeated evaluation on the same
feature table yields means, standard deviations, minima, and maxima.
This is not external validation. For all-25, use outputs_all25/features_13.csv
and set the output directory to outputs_all25."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ann_pipeline import evaluate_models


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "roc_auc",
]


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated integer seeds, such as 7,11,42.

    Require at least two seeds to compare multiple runs."""

    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be a list of integers.") from exc
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("Use at least two seeds.")
    return seeds


def run_analysis(
    feature_path: Path,
    output_dir: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the same model configuration for each seed.

    Only ANN initialization changes. Features, replication folds, PCA size, and
    ANN configuration remain fixed. Save per-seed results and a summary as CSVs.
    These are within-dataset stability measures, not external accuracy or
    confidence intervals for field use."""

    # One row per experiment with 13 extracted features. All-25 contains
    # 5 non-formalin and 20 formalin samples, including retained QC warnings.
    features = pd.read_csv(feature_path)
    records = []

    # Keep data, architecture, and folds fixed; only the ANN seed changes.
    for seed in seeds:
        summary, _, _, _, _ = evaluate_models(
            features,
            pca_components=3,
            hidden_layers=(8,),
            alpha=0.1,
            max_iter=5000,
            random_state=seed,
            cv_mode="replication",
            cv_folds=5,
        )
        summary.insert(1, "random_state", seed)
        records.append(summary)

    # Combine detailed results so individual seed metrics remain visible.
    runs = pd.concat(records, ignore_index=True)
    aggregated_rows = []

    # Summarize the 13-feature ANN and PCA-ANN separately. Small standard
    # deviations indicate consistency across seeds.
    for model_name, group in runs.groupby("model", sort=False):
        row = {
            "model": model_name,
            "seeds": len(group),
            "seed_values": ",".join(str(seed) for seed in seeds),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        aggregated_rows.append(row)
    aggregate = pd.DataFrame.from_records(aggregated_rows)

    # Save both files so individual runs remain auditable alongside averages.
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "seed_stability_runs.csv", index=False)
    aggregate.to_csv(output_dir / "seed_stability_summary.csv", index=False)
    return runs, aggregate


def build_parser() -> argparse.ArgumentParser:
    """Define feature paths, output location, and seeds.

    Defaults target outputs; override them with outputs_all25 paths for all-25."""

    parser = argparse.ArgumentParser(description="Analyze ANN random-seed stability.")
    parser.add_argument(
        "--features",
        default="outputs/features_13.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=[7, 11, 19, 23, 31, 42, 53, 67, 79, 97],
        help="Comma-separated list of seeds.",
    )
    return parser


def main() -> None:
    """Run the analysis from the command line and print key metrics."""

    args = build_parser().parse_args()

    # run_analysis saves complete CSVs; print selected metrics as a compact summary.
    _, summary = run_analysis(
        feature_path=Path(args.features).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        seeds=args.seeds,
    )
    display_columns = [
        "model",
        "seeds",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "recall_mean",
        "recall_std",
        "specificity_mean",
        "specificity_std",
        "f1_score_mean",
        "f1_score_std",
    ]
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
