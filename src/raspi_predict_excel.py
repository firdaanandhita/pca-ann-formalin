"""Contoh menjalankan model PCA-ANN dari raw Excel pada Raspberry Pi.

Penjelasan setiap fungsi tersedia sebagai komentar dekat kode dan docstring.
"""

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


# Fungsi ini menghitung sidik jari SHA-256 sebuah file. Nilai hash dipakai untuk
# memastikan model yang disalin ke Raspberry Pi sama persis dengan model yang
# tercatat di manifest dan tidak rusak selama proses pemindahan.
def calculate_sha256(path: Path) -> str:
    """Hitung fingerprint SHA-256 file untuk verifikasi integritas."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Fungsi ini membaca model_manifest.json dan membandingkan hash model sebelum
# pickle dibuka. Pickle hanya boleh dimuat dari sumber tepercaya karena format
# tersebut dapat mengeksekusi kode ketika dibaca.
def verify_model_before_loading(model_path: Path, manifest_path: Path) -> str:
    """Cocokkan hash model dengan manifest sebelum membuka pickle."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

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
            f"Hash untuk {model_path.name} tidak ditemukan di manifest."
        )

    actual_hash = calculate_sha256(model_path)
    expected_hash = str(artifact["sha256"]).casefold()
    if actual_hash.casefold() != expected_hash:
        raise ValueError(
            "Hash model tidak cocok. Jangan memuat model karena file mungkin "
            "rusak, berubah, atau bukan artefak yang benar."
        )
    return actual_hash


# Fungsi ini mencoba membaca konsentrasi satu sel. Baris yang kosong atau tidak
# valid diberi NaN agar tidak terpilih pada mode demonstrasi. Nilai konsentrasi
# hanya dipakai untuk memilih satu eksperimen dari workbook lama dan tidak
# pernah diberikan sebagai fitur kepada model.
def parse_concentration_or_nan(value: Any) -> float:
    """Baca konsentrasi demo; nilai tak valid dikembalikan sebagai NaN."""

    try:
        return parse_concentration_ml(value)
    except (TypeError, ValueError):
        return np.nan


# Fungsi ini memilih satu pasangan konsentrasi dan replikasi dari workbook raw
# yang berisi banyak eksperimen, seperti file dataset awal. Mode ini hanya untuk
# demonstrasi. Pada penggunaan lapangan, satu file sebaiknya berisi tepat satu
# siklus Baseline dan Exposure sehingga fungsi ini tidak diperlukan.
def select_demo_experiment(
    raw: pd.DataFrame,
    concentration_ml: float,
    replication: int,
) -> pd.DataFrame:
    """Pilih tepat satu pasangan konsentrasi-replikasi dari workbook awal."""

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
            "Eksperimen demo tidak ditemukan untuk "
            f"{concentration_ml:g} mL replikasi {replication}."
        )
    return selected


# Fungsi ini menyiapkan pilihan Terminal. Opsi demo harus diberikan berpasangan.
# Tanpa opsi demo, seluruh file dianggap sebagai satu rekaman lapangan baru.
def build_parser() -> argparse.ArgumentParser:
    """Siapkan opsi untuk mode produksi dan mode demo dataset awal."""

    parser = argparse.ArgumentParser(
        description=(
            "Contoh Raspberry Pi untuk memprediksi satu rekaman raw Excel "
            "menggunakan model PCA-ANN all-25."
        )
    )
    parser.add_argument("--input", required=True, help="File raw XLSX/CSV/TSV.")
    parser.add_argument("--sheet", default="Data", help="Nama sheet input Excel.")
    parser.add_argument(
        "--model",
        default="outputs_all25/model_pca_ann.pkl",
        help="File model PCA-ANN all-25.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs_all25/model_manifest.json",
        help="Manifest yang menyimpan hash resmi model.",
    )
    parser.add_argument(
        "--sample-id",
        default="raspi_measurement",
        help="Identitas pengukuran yang akan ditulis pada hasil.",
    )
    parser.add_argument(
        "--output",
        default="hasil_prediksi_raspi.json",
        help="File JSON tempat hasil prediksi disimpan.",
    )
    parser.add_argument(
        "--demo-concentration-ml",
        type=float,
        help="Konsentrasi yang dipilih dari workbook multi-eksperimen.",
    )
    parser.add_argument(
        "--demo-replication",
        type=int,
        help="Nomor replikasi yang dipilih dari workbook multi-eksperimen.",
    )
    parser.add_argument(
        "--allow-qc-warnings",
        action="store_true",
        help=(
            "Paksa prediksi meskipun ada warning QC. Gunakan hanya untuk "
            "diagnostik, bukan keputusan lapangan."
        ),
    )
    return parser


# Fungsi ini menjalankan inferensi dari awal sampai akhir: memeriksa argumen,
# memverifikasi hash, membaca workbook, memilih eksperimen demo bila diminta,
# mengekstraksi 13 fitur, menjalankan scaler-PCA-ANN, dan menyimpan JSON.
def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    """Jalankan verifikasi, ekstraksi fitur, prediksi, dan ekspor JSON."""

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
            "--demo-concentration-ml dan --demo-replication harus dipakai bersama."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    model_hash = verify_model_before_loading(model_path, manifest_path)
    bundle = load_model_bundle(model_path)
    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)

    mode = "produksi"
    expected_label: int | None = None
    if all(demo_options):
        raw = select_demo_experiment(
            raw,
            concentration_ml=float(args.demo_concentration_ml),
            replication=int(args.demo_replication),
        )
        mode = "demo_dataset_training"
        # Label ini hanya dipakai setelah baris demo dipilih untuk mengecek
        # contoh. Label tidak dikirim ke predict_dataframe atau pipeline model.
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
    if mode == "demo_dataset_training":
        result["demo_warning"] = (
            "Rekaman berasal dari dataset training dan bukan validasi independen."
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


# Fungsi main menangani pesan kesalahan agar pengguna Raspberry Pi mendapat
# alasan yang jelas jika schema, hash model, atau kualitas rekaman bermasalah.
def main() -> None:
    """Jalankan CLI dan ubah kegagalan QC menjadi pesan yang mudah dibaca."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_prediction(args)
    except InputQualityError as exc:
        parser.exit(4, f"QC input gagal: {exc}\n")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(2, f"Input atau artefak tidak valid: {exc}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
