

from __future__ import annotations

import argparse
from pathlib import Path

from deployment_validation import run_validation
from pca_ann_pipeline import run_pipeline
from seed_stability_analysis import run_analysis


# Fungsi ini menyiapkan pilihan yang dapat diberikan saat program dijalankan
# dari Terminal. Nilai default sengaja diarahkan ke model 25 eksperimen agar
# pengguna tidak perlu mengingat seluruh parameter training satu per satu.
def build_parser() -> argparse.ArgumentParser:
    """Siapkan opsi Terminal dengan nilai default untuk 25 eksperimen."""

    parser = argparse.ArgumentParser(
        description=(
            "Training lengkap PCA-ANN menggunakan seluruh 25 eksperimen. "
            "Sampel dengan Baseline pendek tetap dipakai dan diberi warning QC."
        )
    )
    parser.add_argument(
        "--input",
        default="Data Validasi & Pengujian (1).xlsx",
        help="File raw XLSX/CSV/TSV yang akan diproses.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Nama sheet Excel yang berisi data raw.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_all25",
        help="Folder tempat model, tabel evaluasi, dan grafik disimpan.",
    )
    parser.add_argument(
        "--skip-seed-test",
        action="store_true",
        help="Lewati pengujian kestabilan 10 random seed.",
    )
    parser.add_argument(
        "--skip-deployment-test",
        action="store_true",
        help="Lewati replay, dummy, dan negative test artefak model.",
    )
    return parser


# Fungsi ini membentuk kumpulan parameter yang dibutuhkan pipeline utama.
# Parameter pentingnya adalah short_window_policy='keep'. Artinya, data yang
# kurang dari 60/120 detik tidak dibuat-buat atau diperpanjang; data yang
# tersedia tetap dihitung dan diberi tanda warning pada hasil QC.
def make_pipeline_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Bentuk konfigurasi metode yang mengikuti Resume dan varian all-25."""

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


# Fungsi ini memeriksa keberadaan file sebelum training dimulai. Pemeriksaan
# sederhana ini membuat pesan kesalahan lebih mudah dipahami dibandingkan jika
# program baru gagal ketika sedang membaca workbook.
def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Pastikan dataset ada dan tentukan lokasi folder hasil."""

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {input_path}")
    return input_path, output_dir


# Fungsi ini menjalankan seluruh workflow all-25. Tahap pertama melakukan
# preprocessing, ekstraksi 13 fitur, cross-validation, PCA, ANN, dan penyimpanan
# model. Tahap kedua mengulang evaluasi pada beberapa random seed. Tahap ketiga
# menguji apakah model yang tersimpan dapat dimuat dan menerima data raw.
#
# Pengujian seed, replay, dan dummy hanya memeriksa kestabilan internal serta
# jalur perangkat lunak. Tahap tersebut bukan pengganti data lapangan baru.
def run_all25_workflow(args: argparse.Namespace) -> dict:
    """Jalankan training, seed stability, lalu deployment smoke test."""

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


# Fungsi main adalah titik awal program. Fungsi ini membaca pilihan pengguna,
# menjalankan workflow, lalu menampilkan lokasi hasil yang paling penting.
def main() -> None:
    """Baca argumen, jalankan workflow, dan tampilkan lokasi artefak utama."""

    args = build_parser().parse_args()
    metadata = run_all25_workflow(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_count = metadata["feature_extraction"]["sample_count"]
    class_distribution = metadata["feature_extraction"]["class_distribution"]
    print(f"Training all-25 selesai: {sample_count} sampel.")
    print(f"Distribusi kelas: {class_distribution}")
    print(f"Model utama: {output_dir / 'model_pca_ann.pkl'}")
    print(f"Metrik: {output_dir / 'metrics_summary.csv'}")
    print(f"Confusion matrix: {output_dir / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
