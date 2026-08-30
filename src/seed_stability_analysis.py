"""Periksa apakah hasil ANN stabil ketika titik awal acaknya diubah.

ANN memulai proses belajar dari bobot awal yang sebagian ditentukan oleh
``random_state`` atau *seed*. Karena itu, data dan konfigurasi yang sama dapat
memberikan hasil yang sedikit berbeda untuk seed yang berbeda. Program ini
menjalankan evaluasi berulang pada tabel fitur yang sama, lalu merangkum
rata-rata, simpangan baku, nilai minimum, dan nilai maksimum setiap metrik.

Analisis ini tidak memakai data lapangan baru dan bukan pengganti validasi
eksternal. Untuk workflow 25 sampel, arahkan ``--features`` ke
``outputs_all25/features_13.csv`` dan ``--output-dir`` ke ``outputs_all25``.
"""

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
    """Ubah daftar seed seperti ``"7,11,42"`` menjadi daftar bilangan bulat.

    Minimal dua seed diwajibkan karena kestabilan tidak dapat dibandingkan jika
    model hanya dijalankan satu kali.
    """

    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds harus berupa daftar integer.") from exc
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("Gunakan minimal dua seed.")
    return seeds


def run_analysis(
    feature_path: Path,
    output_dir: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluasi konfigurasi model yang sama untuk setiap seed.

    Hanya titik awal acak ANN yang berubah. Tabel fitur, pembagian
    leave-one-replication-out, jumlah komponen PCA, dan konfigurasi ANN tetap
    sama. Hasil per seed dan ringkasannya disimpan sebagai dua file CSV.

    Nilai yang dihasilkan menunjukkan kestabilan algoritme pada dataset yang
    sama. Nilai tersebut bukan akurasi pada data eksternal dan bukan confidence
    interval untuk penggunaan di lingkungan nyata.
    """

    # Tabel ini sudah berisi satu baris per eksperimen dan 13 fitur hasil
    # ekstraksi. Pada workflow all-25, tabel ini berisi 5 sampel non-formalin
    # dan 20 sampel formalin, termasuk sampel yang dipertahankan dengan QC.
    features = pd.read_csv(feature_path)
    records = []

    # Setiap putaran memakai data, arsitektur, dan fold yang sama. Dengan
    # demikian, perbedaan hasil antarputaran hanya berasal dari seed ANN.
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

    # Gabungkan hasil rinci agar pembaca dapat melihat metrik setiap seed,
    # bukan hanya angka rata-ratanya.
    runs = pd.concat(records, ignore_index=True)
    aggregated_rows = []

    # Ringkasan dihitung terpisah untuk ANN 13 fitur dan PCA-ANN. Simpangan
    # baku yang kecil berarti hasil relatif konsisten terhadap perubahan seed.
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

    # Kedua file disimpan agar hasil individual tetap dapat diaudit ketika
    # ringkasan rata-rata terlihat terlalu sederhana.
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "seed_stability_runs.csv", index=False)
    aggregate.to_csv(output_dir / "seed_stability_summary.csv", index=False)
    return runs, aggregate


def build_parser() -> argparse.ArgumentParser:
    """Buat opsi command line untuk lokasi fitur, keluaran, dan daftar seed.

    Nilai default menunjuk folder ``outputs``. Workflow all-25 harus
    menggantinya dengan path di ``outputs_all25`` ketika program dijalankan.
    """

    parser = argparse.ArgumentParser(description="Analisis stabilitas seed ANN.")
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
        help="Daftar seed dipisahkan koma.",
    )
    return parser


def main() -> None:
    """Jalankan analisis dari command line dan tampilkan metrik utama."""

    args = build_parser().parse_args()

    # File CSV lengkap tetap disimpan oleh run_analysis. Terminal hanya
    # menampilkan kolom yang paling mudah dibaca sebagai ringkasan cepat.
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
