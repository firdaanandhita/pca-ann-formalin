"""Pipeline lengkap untuk mengubah rekaman sensor mentah menjadi model PCA-ANN.

Alur program ini adalah:
1. membaca baris mentah dari Excel/CSV;
2. memakai fase Baseline dan Exposure saja;
3. menggabungkan ribuan baris menjadi satu sampel untuk setiap pasangan
   konsentrasi-replikasi;
4. menghitung 13 fitur ringkas;
5. menstandarkan fitur dengan Z-score;
6. membandingkan ANN 13 fitur dengan PCA-ANN;
7. mengevaluasi model memakai prediksi out-of-fold; dan
8. melatih ulang model final pada seluruh sampel yang dipertahankan.

Untuk analisis 25 eksperimen, gunakan ``train_all25.py``. Program tersebut
memilih kebijakan ``keep`` untuk baseline pendek: data tidak ditambah atau
diinterpolasi, tetapi sampel tetap dipakai dan diberi peringatan QC.
"""

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
    """Wadah hasil pembersihan baris dan angka audit yang menjelaskan prosesnya."""

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
    """Wadah tabel satu baris per eksperimen serta sampel yang dikeluarkan."""

    features: pd.DataFrame
    excluded_samples: list[dict[str, Any]]


@dataclass
class InferencePreprocessingResult:
    """Wadah data inferensi bersih, pemetaan kolom, dan peringatan kualitas."""

    cleaned: pd.DataFrame
    source_rows: int
    ignored_phase_counts: dict[str, int]
    phase_counts_used: dict[str, int]
    column_mapping: dict[str, str]
    input_warnings: list[str]


def _normalise_key(value: Any) -> str:
    """Samakan kapital, spasi, dan tanda baca untuk mencocokkan nama kolom."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def resolve_columns(
    columns: Iterable[Any], required: Iterable[str] | None = None
) -> dict[str, str]:
    """Cocokkan variasi nama kolom Excel dengan nama internal program.

    Misalnya, ``MQ-138``, ``MQ138``, dan ``MQ 138`` dianggap sebagai sensor
    yang sama. Program sengaja berhenti bila suatu kolom wajib tidak ditemukan
    atau lebih dari satu kolom tampak sama, agar tidak memakai data yang salah.
    """

    required_columns = list(required or COLUMN_ALIASES.keys())
    unknown = [
        canonical
        for canonical in required_columns
        if canonical not in COLUMN_ALIASES
    ]
    if unknown:
        raise ValueError(f"Nama kolom canonical tidak dikenal: {unknown}")

    normalised: dict[str, list[str]] = {}
    for column in columns:
        normalised.setdefault(_normalise_key(column), []).append(str(column))

    duplicates = {
        key: values for key, values in normalised.items() if len(values) > 1
    }
    if duplicates:
        raise ValueError(
            "Ada nama kolom yang menjadi ambigu setelah normalisasi: "
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
                f"Lebih dari satu kolom cocok untuk '{canonical}': {matches}"
            )
        else:
            missing.append(canonical)

    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + f". Kolom tersedia: {list(map(str, columns))}"
        )
    return mapping


def read_dataset(input_path: Path, sheet: str | int = "Data") -> pd.DataFrame:
    """Baca dataset XLSX/XLSM/XLS, CSV, atau TSV tanpa mengolah nilainya."""

    suffix = input_path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".tsv":
        return pd.read_csv(input_path, sep="\t")
    raise ValueError(
        f"Format '{input_path.suffix}' belum didukung. Gunakan XLSX, XLSM, XLS, CSV, atau TSV."
    )


def _empty_metadata_mask(series: pd.Series) -> pd.Series:
    """Tandai NaN, teks kosong, atau teks berisi spasi saja sebagai data kosong."""

    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("").fillna(True)


def parse_concentration_ml(value: Any) -> float:
    """Ubah nilai seperti ``1 mL`` atau ``1,5`` menjadi angka satuan mililiter."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Konsentrasi kosong.")

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
            raise ValueError(f"Konsentrasi tidak dapat dibaca: {value!r}")
        number = float(match.group(1).replace(",", "."))

    if not np.isfinite(number):
        raise ValueError(f"Konsentrasi bukan angka hingga: {value!r}")
    if number < 0:
        raise ValueError(f"Konsentrasi negatif tidak valid: {value!r}")
    return 0.0 if np.isclose(number, 0.0, atol=1e-12) else number


def _normalise_replication(value: Any) -> str:
    """Buat ID replikasi konsisten; ID ini hanya metadata, bukan fitur model."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Replikasi tidak valid: {value!r}")
        if number.is_integer():
            return str(int(number))
        return f"{number:.12g}"

    text = str(value).strip()
    if not text:
        raise ValueError("Replikasi kosong.")
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def _normalise_phase(value: Any) -> str:
    """Ubah berbagai penulisan fase menjadi baseline, exposure, atau purging."""

    key = _normalise_key(value)
    for canonical, aliases in PHASE_ALIASES.items():
        if key in aliases:
            return canonical
    return str(value).strip().casefold()


def _to_numeric_locale(series: pd.Series) -> pd.Series:
    """Ubah angka berkoma desimal menjadi numerik dan nilai rusak menjadi NaN."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def preprocess_rows(raw: pd.DataFrame) -> PreprocessingResult:
    """Bersihkan baris training dan siapkan metadata untuk ekstraksi fitur.

    Konsentrasi, replikasi, dan fase yang kosong dibuang sesuai permintaan.
    Purging serta fase lain tidak menjadi masukan model. Label dibuat dari
    konsentrasi: tepat 0 mL menjadi kelas 0, sedangkan nilai di atas 0 mL
    menjadi kelas 1. Fungsi ini belum memilih window 60/120 detik; keputusan
    durasi dilakukan oleh :func:`extract_features`.
    """

    # Tahap 1: cari nama kolom sebenarnya agar program tetap dapat membaca
    # variasi penamaan yang wajar pada file Excel.
    mapping = resolve_columns(raw.columns)

    # Tahap 2: buang hanya baris yang kehilangan metadata eksperimen penting.
    # Nilai sensor yang kosong tidak langsung dibuang; jumlahnya dicatat agar
    # QC dan imputer model dapat menanganinya secara transparan.
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
            "Tidak ada baris tersisa setelah Konsentrasi, Replikasi, dan Fase kosong dihapus."
        )

    # Tahap 3: samakan nama fase, lalu pertahankan Baseline dan Exposure saja.
    # Purging dihitung untuk audit, tetapi tidak pernah menjadi fitur.
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
            "Tidak ada baris Baseline atau Exposure setelah preprocessing."
        )

    # Tahap 4: baca konsentrasi dan turunkan label biner. Nilai konsentrasi
    # disimpan sebagai metadata penelitian, bukan sebagai input classifier.
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
            "Ada Konsentrasi non-kosong yang tidak valid. "
            f"Contoh (index, nilai, alasan): {preview}"
        )

    cleaned["concentration_original"] = cleaned[mapping["concentration"]].astype(
        "string"
    )
    cleaned["concentration_ml"] = concentration_values
    cleaned["replication_id"] = cleaned[mapping["replication"]].map(
        _normalise_replication
    )
    cleaned["label"] = (cleaned["concentration_ml"] > 0).astype(int)

    # Tahap 5: ubah waktu dan empat sinyal yang diperlukan menjadi tipe data
    # yang dapat dihitung. Nilai rusak dicatat, bukan diam-diam dihilangkan.
    cleaned[INTERNAL_COLUMNS["timestamp"]] = pd.to_datetime(
        cleaned[mapping["timestamp"]], errors="coerce"
    )
    invalid_timestamp_count = int(
        cleaned[INTERNAL_COLUMNS["timestamp"]].isna().sum()
    )
    if invalid_timestamp_count:
        raise ValueError(
            f"Ada {invalid_timestamp_count} Timestamp tidak valid pada baris ber-metadata lengkap."
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

    # Tahap 6: audit urutan waktu per eksperimen dan fase. Lompatan waktu besar
    # atau timestamp mundur nantinya menjadi dasar pemeriksaan kualitas.
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
    """Bersihkan satu rekaman baru tanpa membutuhkan label atau konsentrasi.

    Data lapangan hanya perlu Timestamp, Fase, HCHO, MQ-138, TGS822, dan
    HUMIDITY. Nilai konsentrasi dan label sementara di bawah ini hanya dibuat
    agar fungsi ekstraksi training dapat dipakai ulang. Nilai tersebut tidak
    termasuk dalam 13 fitur dan tidak pernah masuk ke ANN.
    """

    sample_id = str(sample_id).strip()
    if not sample_id:
        raise ValueError("sample_id inferensi tidak boleh kosong.")

    # Inferensi sengaja tidak meminta kolom Konsentrasi dan Replikasi karena
    # keduanya tidak diketahui saat alat sedang memeriksa sampel baru.
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

    # Seperti saat training, Purging atau fase lain diabaikan.
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
            "Input inferensi tidak memiliki baris Baseline atau Exposure."
        )

    cleaned[INTERNAL_COLUMNS["timestamp"]] = pd.to_datetime(
        cleaned[mapping["timestamp"]], errors="coerce"
    )
    invalid_timestamp_count = int(
        cleaned[INTERNAL_COLUMNS["timestamp"]].isna().sum()
    )
    if invalid_timestamp_count:
        raise ValueError(
            f"Ada {invalid_timestamp_count} Timestamp tidak valid pada Baseline/Exposure."
        )

    for canonical in ("hcho", "mq138", "tgs822", "humidity"):
        cleaned[INTERNAL_COLUMNS[canonical]] = _to_numeric_locale(
            cleaned[mapping[canonical]]
        ).replace([np.inf, -np.inf], np.nan)

    # Masalah timestamp disimpan sebagai warning. Fungsi prediksi akan
    # memutuskan apakah warning tersebut harus menolak input.
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

    # Metadata tiruan ini hanya "adapter" internal untuk extractor bersama.
    # Angka nol di sini BUKAN tebakan bahwa sampel pasti non-formalin.
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
    """Pilih potongan waktu tanpa padding, interpolasi, atau membuat baris baru.

    ``head`` mengambil bagian awal fase; ``tail`` mengambil bagian akhir fase.
    Jika rekaman lebih pendek dari target, semua baris yang tersedia dikembalikan
    dan kekurangan durasinya akan ditangani oleh aturan QC.
    """

    rows = rows.sort_values(timestamp_column)
    if rows.empty or seconds <= 0:
        return rows

    if anchor == "head":
        cutoff = rows[timestamp_column].min() + pd.Timedelta(seconds=seconds)
        return rows.loc[rows[timestamp_column] <= cutoff]
    if anchor == "tail":
        cutoff = rows[timestamp_column].max() - pd.Timedelta(seconds=seconds)
        return rows.loc[rows[timestamp_column] >= cutoff]
    raise ValueError(f"Anchor window tidak dikenal: {anchor}")


def _duration_seconds(rows: pd.DataFrame, timestamp_column: str) -> float:
    """Hitung rentang waktu dari timestamp pertama sampai terakhir."""

    if len(rows) < 2:
        return 0.0
    duration = rows[timestamp_column].max() - rows[timestamp_column].min()
    return float(duration.total_seconds())


def _max_positive_gap_seconds(
    rows: pd.DataFrame, timestamp_column: str
) -> float:
    """Cari jeda maju terbesar untuk mendeteksi rekaman yang terputus."""

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
    """Hitung interval sampling tipikal dari selisih timestamp yang positif."""

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
    """Perkirakan cakupan fase sebagai span waktu ditambah satu interval tipikal."""

    return _duration_seconds(
        rows, timestamp_column
    ) + _median_positive_interval_seconds(rows, timestamp_column)


def _phase_run_count(phases: pd.Series, target: str) -> int:
    """Hitung berapa blok fase terpisah agar dua siklus tidak tergabung."""

    matches = phases.eq(target)
    run_starts = matches & ~matches.shift(fill_value=False)
    return int(run_starts.sum())


def _safe_mean(series: pd.Series) -> float:
    """Hitung rata-rata sambil mengabaikan sebagian NaN; semua NaN tetap NaN."""

    return float(series.mean()) if series.notna().any() else np.nan


def _safe_max(series: pd.Series) -> float:
    """Hitung maksimum sambil mengabaikan sebagian NaN; semua NaN tetap NaN."""

    return float(series.max()) if series.notna().any() else np.nan


def extract_features(
    cleaned: pd.DataFrame,
    baseline_seconds: float = 60.0,
    exposure_seconds: float = 120.0,
    baseline_anchor: str = "tail",
    short_window_policy: str = "drop",
) -> FeatureExtractionResult:
    """Ringkas ribuan baris menjadi satu baris berisi 13 fitur per eksperimen.

    Satu eksperimen ditentukan oleh pasangan konsentrasi dan replikasi. Untuk
    setiap eksperimen, program mengambil 60 detik terakhir Baseline dan
    120 detik pertama Exposure. Tiga sensor gas masing-masing menghasilkan
    empat fitur (baseline mean, exposure mean, exposure max, dan delta max),
    lalu kelembapan menghasilkan ``RH_mean``. Totalnya adalah 13 fitur.

    ``short_window_policy`` menentukan nasib fase yang kurang panjang:
    ``drop`` mengeluarkan sampel, ``error`` menghentikan program, dan ``keep``
    tetap menghitung fitur dari data yang benar-benar tersedia sambil memberi
    warning. Mode 25 eksperimen memakai ``keep``; tidak ada data buatan,
    padding, maupun interpolasi. Pada dataset sekarang, ``5mL_rep1`` memakai
    baseline pendek sehingga fitur baseline dan delta-nya perlu dibaca sebagai
    hasil non-strict/sensitivity analysis.
    """

    if short_window_policy not in {"keep", "drop", "error"}:
        raise ValueError(
            "short_window_policy harus salah satu dari: keep, drop, error."
        )

    records: list[dict[str, Any]] = []
    excluded_samples: list[dict[str, Any]] = []
    timestamp_column = INTERNAL_COLUMNS["timestamp"]

    # Setiap pasangan konsentrasi-replikasi menjadi tepat satu calon sampel.
    # Ini sebabnya sekitar 8.000 baris mentah dapat menjadi hanya 25 baris fitur.
    group_columns = ["concentration_ml", "replication_id"]
    for (concentration_ml, replication_id), group in cleaned.groupby(
        group_columns, sort=False, dropna=False
    ):
        # Pastikan satu pasangan tidak berisi lebih dari satu siklus Baseline
        # atau Exposure yang terpisah, karena dua siklus tidak boleh digabung.
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

        # Ambil window sesuai Resume.pdf. Baris tidak dirata-ratakan lintas
        # eksperimen dan tidak ada titik waktu sintetis yang ditambahkan.
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
            raise ValueError(f"Label tidak konsisten dalam sampel {sample_id}.")

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

        # QC durasi dan kontinuitas waktu dilakukan sebelum fitur dipakai.
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

        # Empat fitur untuk setiap sensor gas:
        # 1) rata-rata baseline, 2) rata-rata exposure, 3) puncak exposure,
        # 4) selisih puncak exposure dengan rata-rata baseline.
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

        # Kelembapan hanya diringkas menjadi rata-rata selama Exposure.
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

        # Kebijakan `keep` khusus all-25 tidak menyamarkan warning: status QC
        # tetap tersimpan agar keterbatasannya terlihat pada output.
        duration_flags = [
            flag
            for flag in qc_flags
            if flag in {"baseline_duration_short", "exposure_duration_short"}
        ]
        if duration_flags and short_window_policy == "error":
            raise ValueError(
                f"Sampel {sample_id} tidak memenuhi durasi Resume.pdf: "
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
            "Tidak ada sampel dengan pasangan fase baseline dan exposure yang dapat diproses."
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
    """Baca jumlah PC tetap, atau target proporsi varians antara 0 dan 1."""

    value = value.strip()
    if re.fullmatch(r"\d+", value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("Jumlah komponen PCA minimal 1.")
        return parsed
    try:
        parsed_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "PCA components harus bilangan bulat atau proporsi antara 0 dan 1."
        ) from exc
    if not 0 < parsed_float < 1:
        raise argparse.ArgumentTypeError(
            "Proporsi varians PCA harus lebih dari 0 dan kurang dari 1."
        )
    return parsed_float


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    """Baca arsitektur ANN; contoh ``8,4`` berarti dua layer berisi 8 dan 4 neuron."""

    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Hidden layers harus seperti '8' atau '8,4'."
        ) from exc
    if not layers or any(layer < 1 for layer in layers):
        raise argparse.ArgumentTypeError(
            "Setiap hidden layer harus berupa integer positif."
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
    """Susun seluruh preprocessing dan classifier sebagai satu objek model.

    Urutannya adalah median imputer, StandardScaler (Z-score), PCA bila
    diminta, lalu MLPClassifier sebagai ANN biner. Ketika objek ini dilatih
    di dalam fold, imputer, scaler, dan PCA juga hanya belajar dari bagian
    training. Hal itu penting untuk mencegah kebocoran informasi dari test.
    ANN 13 fitur memakai urutan yang sama tetapi tanpa langkah PCA.
    """

    # Menyatukan semua transformasi dalam sklearn Pipeline membuat urutan
    # preprocessing saat training dan inferensi selalu sama.
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
    """Hitung metrik biner dengan kelas 1 (formalin) sebagai kelas positif.

    Accuracy adalah proporsi semua tebakan benar. Balanced accuracy merata-ratakan
    kinerja per kelas. Precision mengukur ketepatan alarm formalin, recall
    mengukur formalin yang berhasil ditemukan, specificity mengukur non-formalin
    yang benar, F1 menyeimbangkan precision-recall, dan AUC menilai urutan skor.
    """

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
    """Pastikan kelas 0 dan 1 tersedia dengan minimal dua sampel per kelas."""

    class_counts = y.value_counts().sort_index()
    if set(class_counts.index) != {0, 1}:
        raise ValueError(
            f"Model biner memerlukan kelas 0 dan 1. Ditemukan: {class_counts.to_dict()}"
        )
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Setiap kelas minimal memerlukan dua sampel untuk evaluasi."
        )


def _cross_validation_splits(
    y: pd.Series,
    replication_groups: pd.Series,
    cv_mode: str,
    cv_folds: int,
    random_state: int,
):
    """Buat pembagian train-test tanpa memakai baris dari fold test saat fit.

    Mode ``replication`` memakai Leave-One-Group-Out: seluruh sampel dengan
    nomor replikasi yang sama ditahan sebagai test secara bersamaan. Ini lebih
    aman daripada memecah baris sensor secara acak karena replikasi terkait
    tidak tersebar ke train dan test. Mode ``stratified`` menjaga proporsi kelas,
    tetapi tidak menjaga grup replikasi sehingga hanya menjadi opsi pembanding.
    """

    if cv_mode == "replication":
        if replication_groups.nunique() < 2:
            raise ValueError(
                "Leave-one-replication-out memerlukan minimal dua Replikasi unik."
            )
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(np.zeros(len(y)), y, replication_groups))
    elif cv_mode == "stratified":
        smallest_class = int(y.value_counts().min())
        n_splits = min(cv_folds, smallest_class)
        if n_splits < 2:
            raise ValueError("Stratified CV memerlukan minimal dua fold.")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        raise ValueError(f"CV mode tidak dikenal: {cv_mode}")

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        if len(np.unique(y.iloc[train_index])) < 2:
            raise ValueError(
                f"Fold {fold_number} hanya memiliki satu kelas pada data latih."
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
    """Evaluasi ANN dan PCA-ANN secara out-of-fold lalu fit model final.

    Untuk setiap fold, seluruh pipeline belajar hanya dari sampel training,
    termasuk median, Z-score, PCA, bobot kelas, dan ANN. Setiap eksperimen
    memperoleh satu prediksi ketika berada di fold test. Prediksi out-of-fold
    tersebut digunakan untuk confusion matrix dan metrik evaluasi.

    Setelah evaluasi selesai, setiap model dilatih ulang pada seluruh sampel
    yang dipertahankan untuk menghasilkan artefak deployment. Oleh karena itu,
    metrik yang dilaporkan adalah metrik OOF, bukan skor training model final.
    """

    # Hanya 13 kolom ini yang menjadi input numerik. Konsentrasi, replikasi,
    # sample_id, dan label tidak pernah ikut masuk sebagai fitur.
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

    # Dua skenario dibandingkan secara adil dengan pembagian fold yang sama:
    # ANN langsung dari 13 fitur dan ANN dari ringkasan PCA.
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

    # Loop pertama menghasilkan prediksi test dari setiap fold. Bobot kelas
    # selalu dihitung dari y_train saja supaya label fold test tidak bocor.
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

    # Gabungkan seluruh prediksi test menjadi satu tabel OOF dan hitung metrik.
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

    # Sesudah evaluasi terkunci, fit model deployment pada seluruh 25 sampel.
    # Langkah ini wajar untuk artefak final, tetapi tidak dipakai menghitung OOF.
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
    """Hitung fingerprint file untuk mendeteksi perubahan isi.

    SHA-256 bukan jaminan bahwa pickle aman; file model tetap harus berasal
    dari sumber tepercaya.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Ubah NumPy, timestamp, koleksi, dan NaN menjadi nilai yang aman untuk JSON."""

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


# Matplotlib hanya dibutuhkan ketika training menyimpan grafik. Import dibuat
# secara "lazy" agar program prediksi Raspberry Pi tidak perlu memasang paket
# plotting yang berat hanya untuk menjalankan model.
def _get_pyplot():
    """Muat Matplotlib dalam mode tanpa layar dan kembalikan modul pyplot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save_confusion_matrices(
    matrices: dict[str, np.ndarray], output_path: Path
) -> None:
    """Simpan confusion matrix OOF; baris aktual dan kolom hasil prediksi."""

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
            xlabel="Prediksi",
            ylabel="Aktual",
            title=name.replace("_", " "),
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrix — Prediksi Out-of-Fold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pca_plots(
    pca_scores: pd.DataFrame,
    explained_variance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Simpan visual PCA deskriptif dari model final yang memakai seluruh data.

    Grafik ini membantu memahami pola dan varians, tetapi bukan evaluasi
    held-out karena PCA final sudah di-fit pada seluruh sampel.
    """

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
        axis.set_xlabel(f"PC1 ({pc1_variance:.1%} varians)")
        axis.set_ylabel(f"PC2 ({pc2_variance:.1%} varians)")
        axis.set_title("Proyeksi PCA seluruh sampel (model final)")
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
        label="Varians per PC",
    )
    axis.plot(
        explained_variance["component"],
        explained_variance["cumulative_explained_variance"],
        color="#ED7D31",
        marker="o",
        label="Kumulatif",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Proporsi varians")
    axis.set_title("Explained Variance PCA (model final)")
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
    """Gambar mean ± SD delta sensor menurut konsentrasi.

    Konsentrasi dipakai hanya sebagai sumbu visualisasi penelitian dan tidak
    pernah menjadi masukan classifier.
    """

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
    axis.set_xlabel("Konsentrasi (mL)")
    axis.set_ylabel("Delta maksimum (exposure max − baseline mean)")
    axis.set_title("Respons sensor menurut konsentrasi (mean ± SD antar replikasi)")
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
    """Simpan data audit, evaluasi, PCA, model final, dan manifest.

    CSV menyimpan data tabular agar mudah diaudit. PNG menyimpan visualisasi.
    Bundle ``.pkl`` dan ``.joblib`` berisi pipeline yang sudah di-fit pada
    seluruh sampel yang dipertahankan, termasuk imputer, scaler, PCA, dan ANN.
    Metrik OOF disimpan terpisah dan tidak boleh disalahartikan sebagai skor
    data lapangan independen. File pickle hanya boleh dimuat dari sumber
    tepercaya setelah hash-nya diperiksa.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table = extraction.features

    # Bagian 1: simpan baris bersih, 13 fitur, sampel yang dikeluarkan, metrik,
    # prediksi OOF, dan confusion matrix sebagai tabel CSV yang mudah diperiksa.
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

    # Bagian 2: gambar ini membantu presentasi hasil, tetapi angka kanoniknya
    # tetap berada dalam CSV.
    _save_confusion_matrices(
        confusion_matrices, output_dir / "confusion_matrices.png"
    )
    _save_sensor_response_plot(
        feature_table, output_dir / "sensor_delta_by_concentration.png"
    )

    # Bagian 3: setiap bundle menyimpan keseluruhan preprocessing model. Karena
    # itu pengguna Raspberry Pi tidak perlu menjalankan PCA secara terpisah.
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
                "0": "non-formalin (konsentrasi = 0 mL)",
                "1": "formalin (konsentrasi > 0 mL)",
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
                "Pickle/joblib hanya boleh dimuat dari sumber tepercaya."
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

    # Bagian 4: transformasi seluruh 13 fitur dengan model PCA final untuk
    # menghasilkan skor PC, explained variance, dan loading yang deskriptif.
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

    # Bagian 5: metadata JSON merekam asal data, keputusan preprocessing,
    # konfigurasi, versi pustaka, serta peringatan QC agar run dapat diaudit.
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
    # Manifest adalah kontrak deployment ringkas dan mencantumkan hash setiap
    # artefak. Hash mendeteksi perubahan file, bukan membuat pickle tak berisiko.
    model_manifest = {
        "artifact_version": 1,
        "primary_model": "model_pca_ann.pkl",
        "artifact_format": "pickle",
        "source_sha256": source_sha256,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": 0.5,
        "label_definition": {
            "0": "non-formalin (konsentrasi = 0 mL)",
            "1": "formalin (konsentrasi > 0 mL)",
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
            "classifier": "MLPClassifier (ANN biner)",
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
            "File .pkl dapat mengeksekusi kode saat dimuat. "
            "Muat hanya artefak yang dipercaya dan cocokkan SHA-256."
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
    """Jalankan alur baca → bersihkan → fitur → CV → fit final → simpan.

    Versi 25 eksperimen bergantung pada ``short_window_policy=keep`` yang
    diberikan oleh ``train_all25.py``. Hasil replay, dummy, dan OOF merupakan
    pemeriksaan internal; semuanya belum menggantikan validasi sampel lapangan
    independen yang dikumpulkan pada waktu/kondisi berbeda.
    """

    # Temukan dataset dan baca sheet yang dipilih.
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()

    sheet: str | int
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw = read_dataset(input_path, sheet=sheet)
    # Bersihkan baris lalu ubah setiap eksperimen menjadi 13 fitur.
    preprocessing = preprocess_rows(raw)
    extraction = extract_features(
        preprocessing.cleaned,
        baseline_seconds=args.baseline_seconds,
        exposure_seconds=args.exposure_seconds,
        baseline_anchor=args.baseline_anchor,
        short_window_policy=args.short_window_policy,
    )

    # Cegah jumlah komponen PCA melampaui ukuran data training terkecil.
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
            f"PCA {args.pca_components} komponen terlalu besar untuk ukuran fold. "
            f"Gunakan <= {max_components}."
        )

    # Evaluasi kedua model secara out-of-fold dan fit artefak final.
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
    # Simpan seluruh artefak dan audit trail.
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

    print(f"Pipeline selesai. Output: {output_dir}")
    print(
        "Baris: "
        f"{preprocessing.source_rows} sumber, "
        f"{preprocessing.dropped_missing_metadata} dihapus karena metadata kosong, "
        f"{sum(preprocessing.ignored_phase_counts.values())} Purging/fase lain diabaikan, "
        f"{len(preprocessing.cleaned)} Baseline+Exposure dipakai."
    )
    print(
        f"Sampel fitur: {len(extraction.features)}; "
        f"distribusi label: {metadata['feature_extraction']['class_distribution']}"
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
    """Definisikan opsi command line dalam istilah yang mudah dibaca.

    Default kebijakan window adalah ``drop``. Untuk reproduksi khusus 25
    eksperimen, lebih aman memakai wrapper ``python train_all25.py`` yang
    secara eksplisit memilih ``keep`` dan folder ``outputs_all25``.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Preprocessing data sensor dan klasifikasi biner formalin dengan PCA-ANN."
        )
    )
    parser.add_argument(
        "--input",
        default="Data Validasi & Pengujian (1).xlsx",
        help="Path dataset XLSX/CSV/TSV.",
    )
    parser.add_argument(
        "--sheet",
        default="Data",
        help="Nama sheet Excel atau indeks sheet (default: Data).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder keluaran (default: outputs).",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=60.0,
        help="Durasi window baseline dalam detik; 0 memakai seluruh fase.",
    )
    parser.add_argument(
        "--baseline-anchor",
        choices=["tail", "head"],
        default="tail",
        help=(
            "Ambil window baseline dari akhir fase (tail, paling dekat exposure) "
            "atau awal fase (head)."
        ),
    )
    parser.add_argument(
        "--exposure-seconds",
        type=float,
        default=120.0,
        help="Durasi window exposure dari awal fase; 0 memakai seluruh fase.",
    )
    parser.add_argument(
        "--short-window-policy",
        choices=["drop", "keep", "error"],
        default="drop",
        help=(
            "Kebijakan sampel yang tidak mencapai durasi Resume.pdf: "
            "drop (default), keep dengan warning, atau error."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=parse_pca_components,
        default=3,
        help="Jumlah PC (mis. 3) atau target varians (mis. 0.95).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(8,),
        help="Ukuran hidden layer ANN, mis. 8 atau 8,4.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Regularisasi L2 ANN (default: 0.1).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Iterasi maksimum ANN (default: 5000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed reproduksibilitas (default: 42).",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["replication", "stratified"],
        default="replication",
        help=(
            "Evaluasi leave-one-replication-out (default, anti kebocoran) "
            "atau stratified k-fold."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Jumlah fold bila --cv-mode stratified (default: 5).",
    )
    return parser


def main() -> None:
    """Validasi argumen dasar lalu jalankan pipeline training."""

    parser = build_argument_parser()
    args = parser.parse_args()
    if args.baseline_seconds < 0 or args.exposure_seconds < 0:
        parser.error("Durasi window tidak boleh negatif.")
    if args.alpha < 0:
        parser.error("Alpha tidak boleh negatif.")
    if args.max_iter < 1:
        parser.error("Max iter minimal 1.")
    if args.cv_folds < 2:
        parser.error("CV folds minimal 2.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
