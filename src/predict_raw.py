"""Inferensi formalin dari satu eksperimen sensor mentah.

Modul ini tidak melatih model. Tugasnya adalah membaca satu rekaman eksperimen,
memakai fase Baseline dan Exposure untuk menghitung 13 fitur, lalu menjalankan
pipeline lengkap yang tersimpan di artefak PCA-ANN.

Hal penting untuk pengguna:

- Satu pemanggilan hanya untuk satu eksperimen, bukan beberapa siklus sekaligus.
- Data Purging boleh ada, tetapi tidak dipakai untuk membentuk fitur.
- Pemeriksaan kualitas (QC) dilakukan sebelum prediksi agar data yang tidak
  memenuhi protokol tidak dipaksa menjadi kelas 0 atau 1.
- ``probability_formalin`` adalah probabilitas kelas menurut model, bukan kadar
  formalin dalam mL, ppm, atau satuan kimia lainnya.
- File pickle dapat mengeksekusi kode saat dibuka. Muat hanya model tepercaya
  dan verifikasi checksum SHA-256 sebelum memanggil ``load_model_bundle``.
"""

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
    """Menandai input yang dapat dibaca tetapi gagal pemeriksaan kualitas."""

    pass


def file_sha256(path: Path) -> str:
    """Menghitung checksum SHA-256 sebuah file tanpa mengubah isinya."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Memuat dan memeriksa struktur dasar bundle model PCA-ANN tepercaya.

    Pemeriksaan di fungsi ini memastikan key, urutan 13 fitur, dan urutan tahap
    pipeline sesuai kontrak program. Fungsi ini tidak membandingkan checksum
    dengan manifest; pemanggil harus melakukannya sebelum file pickle dibuka.

    Args:
        model_path: Lokasi artefak model dengan ekstensi ``.pkl``.

    Returns:
        Dictionary bundle yang berisi pipeline lengkap dan metadata inferensi.

    Raises:
        FileNotFoundError: Jika file model tidak ditemukan.
        ValueError: Jika format atau struktur bundle tidak sesuai kontrak.
    """

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    if path.suffix.casefold() != ".pkl":
        raise ValueError("Program inferensi ini mengharapkan model berformat .pkl.")

    # KEAMANAN PICKLE:
    # Membuka pickle dapat menjalankan kode yang tertanam di dalam file.
    # Karena itu, checksum dan asal file harus dipastikan tepercaya SEBELUM
    # fungsi ini dipanggil. Jangan memuat model dari unggahan sembarang.
    with path.open("rb") as handle:
        bundle = pickle.load(handle)

    # Bundle deployment harus membawa pipeline lengkap. Dengan demikian,
    # imputer, Z-score, PCA, dan ANN yang dipakai saat inferensi sama dengan
    # yang sudah dipelajari saat training; pengguna tidak perlu melakukannya
    # secara manual satu per satu.
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
        raise ValueError(f"Bundle model tidak lengkap; key hilang: {missing}")
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Urutan fitur model tidak sama dengan kontrak 13 fitur program."
        )

    pipeline = bundle["pipeline"]
    expected_steps = ["imputer", "scaler", "pca", "ann"]
    if list(pipeline.named_steps) != expected_steps:
        raise ValueError(
            "Pipeline PCA-ANN tidak sesuai. "
            f"Expected {expected_steps}, found {list(pipeline.named_steps)}."
        )
    return bundle


def _json_safe(value: Any) -> Any:
    """Mengubah tipe NumPy/pandas menjadi nilai yang aman ditulis sebagai JSON."""

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
    """Memprediksi satu eksperimen dari baris sensor mentah.

    Args:
        raw: Data mentah yang memuat Timestamp, Fase, HCHO, MQ-138, TGS822,
            dan HUMIDITY. Kolom Konsentrasi tidak diperlukan untuk inferensi.
        bundle: Bundle model hasil ``load_model_bundle``.
        sample_id: Identitas audit untuk satu eksperimen yang sedang diuji.
        allow_qc_warnings: Jika ``True``, prediksi tetap dijalankan saat ada
            warning QC. Opsi ini hanya cocok untuk diagnostik, bukan mode
            deployment normal.

    Returns:
        Dictionary berisi kelas prediksi, probabilitas kelas formalin, hasil
        QC, informasi window, dan 13 fitur yang dipakai model.

    Raises:
        InputQualityError: Jika eksperimen tidak dapat diekstrak tepat satu kali
            atau gagal QC saat warning tidak diizinkan.
        ValueError: Jika kolom/fase wajib atau nilai dasar input tidak valid.
    """

    # SATU FILE = SATU EKSPERIMEN:
    # Tahap ini menormalkan nama kolom dan fase, lalu hanya mempertahankan
    # Baseline serta Exposure. Purging sengaja diabaikan sesuai Resume.
    # Jika file berisi beberapa siklus, proses berikutnya harus menolaknya
    # daripada memilih salah satu siklus secara diam-diam.
    preprocessing = preprocess_inference_rows(raw, sample_id=sample_id)
    config = bundle["config"]

    # Ekstraksi memakai window dan rumus 13 fitur yang sama dengan training.
    # Kebijakan "keep" di sini hanya agar semua flag kualitas dapat dihitung;
    # warning tersebut tetap ditolak di bawah kecuali pengguna secara eksplisit
    # mengaktifkan mode diagnostik allow_qc_warnings.
    extraction = extract_features(
        preprocessing.cleaned,
        baseline_seconds=float(config["baseline_seconds"]),
        exposure_seconds=float(config["exposure_seconds"]),
        baseline_anchor=str(config["baseline_anchor"]),
        short_window_policy="keep",
    )
    if extraction.excluded_samples:
        raise InputQualityError(
            "Input tidak dapat diekstrak sebagai satu pengujian: "
            + json.dumps(extraction.excluded_samples, ensure_ascii=False)
        )
    if len(extraction.features) != 1:
        raise InputQualityError(
            f"Inferensi memerlukan tepat satu pengujian, ditemukan {len(extraction.features)}."
        )

    # QC melindungi pengguna dari prediksi berbasis rekaman yang terlalu pendek,
    # timestamp bermasalah, gap besar, atau nilai sensor yang hilang. Secara
    # default model tidak memberikan kelas jika kualitas input meragukan.
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
            "Input gagal pemeriksaan kualitas: " + ", ".join(all_warnings)
        )

    feature_frame = pd.DataFrame(
        [[float(feature_row[column]) for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    pipeline = bundle["pipeline"]

    # Pipeline menjalankan imputer -> StandardScaler -> PCA -> ANN secara utuh.
    # Angka yang dihasilkan adalah probabilitas untuk KELAS formalin menurut
    # model. Angka ini bukan hasil pengukuran konsentrasi atau kadar formalin.
    probability_formalin = float(pipeline.predict_proba(feature_frame)[0, 1])
    threshold = float(bundle.get("decision_threshold", 0.5))
    predicted_label = int(probability_formalin >= threshold)

    # Informasi QC, window, dan fitur ikut dikembalikan agar setiap keputusan
    # dapat diaudit, bukan hanya menampilkan label akhir tanpa konteks.
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
    """Membangun parser argumen untuk antarmuka command-line inferensi."""

    parser = argparse.ArgumentParser(
        description=(
            "Prediksi formalin dari satu rekaman raw Baseline/Exposure "
            "tanpa kolom Konsentrasi."
        )
    )
    parser.add_argument("--input", required=True, help="XLSX/CSV/TSV satu pengujian.")
    parser.add_argument(
        "--model",
        default="outputs/model_pca_ann.pkl",
        help="Model PCA-ANN .pkl yang tepercaya.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Nama atau indeks sheet untuk input Excel.",
    )
    parser.add_argument(
        "--sample-id",
        default="inference_sample",
        help="ID untuk pengujian ini.",
    )
    parser.add_argument(
        "--output",
        help="Opsional: simpan hasil prediksi sebagai JSON.",
    )
    parser.add_argument(
        "--allow-qc-warnings",
        action="store_true",
        help="Tetap prediksi jika durasi/gap/nilai sensor memicu warning.",
    )
    return parser


def main() -> None:
    """Menjalankan alur CLI: baca satu rekaman, muat model, dan cetak JSON."""

    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input tidak ditemukan: {input_path}")

    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)
    model_path = Path(args.model).expanduser().resolve()

    # CLI mengharapkan model lokal yang sudah dipercaya. Hash model dicantumkan
    # pada hasil untuk audit, tetapi pengguna/deployer tetap harus mencocokkannya
    # dengan manifest sebelum pickle dibuka.
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
