"""Uji kesiapan teknis pipeline PCA-ANN sebelum dipakai untuk inferensi.

Program ini memeriksa bahwa model dapat dimuat ulang, ekstraksi 13 fitur tetap
konsisten, Purging benar-benar diabaikan, input rusak ditolak, dan dua data
dummy dapat melewati alur prediksi dari awal sampai akhir. Data dummy dibuat
dari rekaman yang sudah ada sehingga hanya cocok sebagai *smoke test* teknis.

Lulusnya semua pemeriksaan tidak membuktikan bahwa model telah tervalidasi di
lingkungan nyata. Validasi lapangan tetap memerlukan data baru dari hari,
perangkat, atau batch yang tidak pernah dipakai untuk training. Untuk workflow
25 sampel, gunakan model di ``outputs_all25``; file ``features_13.csv`` yang
menjadi pasangannya akan dibaca dari folder model tersebut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Memungkinkan validasi dijalankan langsung dari root repository dengan
# ``python validation/deployment_validation.py``.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    """Ambil satu rekaman lama sebagai bahan replay pengujian teknis.

    Konsentrasi dan replikasi hanya dipakai untuk menemukan rekaman yang sudah
    diketahui di workbook. Kolom yang dikembalikan tidak memuat konsentrasi
    atau label, sehingga fungsi prediksi tetap menerima bentuk input seperti
    kondisi nyata ketika kelas sampel belum diketahui.
    """

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
            f"Run tidak ditemukan: concentration={concentration}, replication={replication}"
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
    """Pilih bagian awal atau akhir suatu fase berdasarkan timestamp.

    ``tail`` dipakai untuk bagian akhir Baseline yang paling dekat dengan
    paparan. Nilai anchor lainnya pada pemakaian internal fungsi ini berarti
    bagian awal fase, yang digunakan untuk Exposure. Fungsi hanya memilih
    baris; tidak melakukan interpolasi atau menambah data.
    """

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
    """Bentuk fase dummy berinterval satu detik dari pola rekaman asli.

    Nilai sensor diinterpolasi ke timestamp satu detik, kemudian diberi noise
    acak kecil dan terkontrol. Dummy yang dihasilkan masih merupakan turunan
    rekaman lama, bukan pengukuran sampel baru.
    """

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

    # Perlakuan yang sama diterapkan pada ketiga sensor gas dan kelembapan.
    # Nilai kosong sebagian diisi sepanjang pola fase agar interpolasi dapat
    # dilakukan; sensor yang seluruhnya kosong tetap dianggap kesalahan.
    for canonical, output_name in output_names.items():
        values = pd.to_numeric(rows[mapping[canonical]], errors="coerce")
        values = values.interpolate(limit_direction="both")
        if values.isna().all():
            raise ValueError(f"Sensor {canonical} seluruhnya kosong pada dummy source.")
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

        # Besar noise mengikuti variasi sinyal, dengan batas bawah agar sinyal
        # yang sangat datar tetap mendapat gangguan kecil untuk smoke test.
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
    """Buat satu dummy lengkap: 60 baris Baseline dan 120 baris Exposure.

    Bagian akhir Baseline dan bagian awal Exposure mengikuti kontrak Resume.
    Keduanya diubah menjadi data 1 Hz dengan timestamp fiktif tahun 2030.
    ``seed`` membuat dummy dapat dibuat ulang secara identik untuk audit.

    Dummy hanya menguji apakah preprocessing, ekstraksi fitur, dan inferensi
    berjalan. Prediksi yang benar pada dummy tidak membuktikan generalisasi
    model terhadap sampel lapangan baru.
    """

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
        raise ValueError("Source dummy harus memiliki Baseline dan Exposure.")

    # Ambil window yang sama seperti proses training: akhir Baseline selama
    # 60 detik dan awal Exposure selama 120 detik.
    baseline = _window_rows(
        baseline_all, mapping["timestamp"], seconds=60, anchor="tail"
    )
    exposure = _window_rows(
        exposure_all, mapping["timestamp"], seconds=120, anchor="head"
    )
    rng = np.random.default_rng(seed)

    # Timestamp sengaja dibuat fiktif. Model memakai pola sensor dan urutan
    # fase, bukan tanggal kalender, untuk membentuk 13 fitur.
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
    """Periksa bahwa input buruk ditolak dengan alasan yang diharapkan.

    Pemeriksaan dinyatakan lulus bila callback melempar exception dan pesannya
    mengandung ``expected_text``. Ini menguji pagar kualitas input, bukan
    ketepatan klasifikasi formalin.
    """

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
        "message": "Input tidak ditolak.",
    }


def run_validation(
    *,
    workbook_path: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Jalankan seluruh pemeriksaan teknis deployment dan simpan laporannya.

    Pemeriksaan mencakup kesamaan fitur hasil replay, prediksi contoh yang
    sudah diketahui, invariansi terhadap Purging, dummy noisy, penolakan input
    buruk, dan konsistensi hasil setelah pickle dimuat ulang. Fungsi menulis
    dummy CSV serta ``deployment_test_results.json`` dan akan gagal dengan
    ``AssertionError`` jika sedikitnya satu pemeriksaan tidak lulus.

    ``features_13.csv`` dibaca dari folder yang sama dengan model. Karena itu,
    model all-25 harus dipasangkan dengan file fitur dari ``outputs_all25``.
    """

    # Tahap 1: muat model, data raw, dan tabel fitur yang menjadi pasangan
    # artefak model. Ketiganya harus berasal dari workflow yang sama.
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(model_path)
    raw = read_dataset(workbook_path, sheet="Data")
    training_features = pd.read_csv(model_path.parent / "features_13.csv")

    replay_feature_max_difference = 0.0
    replay_feature_rows_checked = 0

    # Tahap 2: replay setiap eksperimen training dari data raw dan hitung ulang
    # 13 fiturnya. Hasil harus identik dengan features_13.csv. QC warning
    # diizinkan khusus pada parity ini agar sampel 5mL_rep1 dalam workflow
    # all-25 tetap dapat diperiksa tanpa menyembunyikan warning-nya.
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

    # Tahap 3: jalankan inferensi pada satu contoh non-formalin dan satu contoh
    # formalin yang label aslinya sudah diketahui. Ini adalah replay data lama,
    # bukan pengujian independen.
    replay_non = _actual_run(raw, concentration="0 mL", replication=5)
    replay_formal = _actual_run(raw, concentration="15 mL", replication=5)
    replay_non_result = predict_dataframe(
        replay_non, bundle, sample_id="replay_0mL_rep5"
    )
    replay_formal_result = predict_dataframe(
        replay_formal, bundle, sample_id="replay_15mL_rep5"
    )

    # Tahap 4: bandingkan prediksi sebelum dan setelah Purging dihapus. Nilai
    # harus sama karena pipeline hanya boleh memakai Baseline dan Exposure.
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

    # Tahap 5: buat dua dummy noisy dari pola contoh lama dan simpan CSV-nya
    # agar input smoke test dapat dilihat serta diuji ulang.
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

    # Tahap 6: siapkan berbagai input yang sengaja salah. Tujuannya memastikan
    # sistem menolak data yang tidak memenuhi kontrak, bukan memaksakan
    # prediksi ketika fase, durasi, atau sensor tidak layak.
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

    # Walaupun extractor inferensi memakai kebijakan keep agar dapat menghitung
    # QC, predict_dataframe secara default tetap menolak warning. Karena itu,
    # dummy Baseline pendek di bawah harus gagal dengan alasan durasi pendek.
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
        "tidak ada sampel",
    )

    # Tahap 7: muat ulang file pickle dan pastikan probabilitas tidak berubah.
    # Ini mendeteksi masalah serialisasi, bukan mengukur akurasi lapangan.
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

    # Tahap 8: rangkum setiap pemeriksaan menjadi nilai True/False yang mudah
    # dibaca. Replay dan dummy dinilai terpisah dari metrik OOF training.
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

    # Laporan lengkap juga menyatakan batas interpretasi agar kelulusan smoke
    # test tidak disalahartikan sebagai validasi real environment.
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
                "Replay dan dummy membuktikan kontrak input, ekstraksi fitur, "
                "serialisasi, serta eksekusi inferensi; bukan generalisasi lapangan."
            ),
            (
                "Validasi real environment tetap memerlukan data eksternal yang "
                "tidak pernah dipakai training, idealnya hari/perangkat/batch berbeda."
            ),
        ],
    }
    report_path = output_dir / "deployment_test_results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Jangan diam-diam menghasilkan laporan "sukses" bila ada guardrail yang
    # gagal. Nama pemeriksaan yang gagal disertakan dalam exception.
    if not all_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Deployment checks gagal: {failed}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Buat opsi command line untuk workbook, model, dan folder laporan.

    Nilai default menunjuk folder ``outputs``. Untuk workflow all-25, berikan
    ``--model outputs_all25/model_pca_ann.pkl`` dan folder output yang sesuai.
    """

    parser = argparse.ArgumentParser(description="Validasi deployment PCA-ANN.")
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
    """Jalankan validasi dan tampilkan ringkasan hasil di terminal.

    Laporan lengkap tetap disimpan oleh ``run_validation``; terminal hanya
    menampilkan status pemeriksaan dan ringkasan prediksi dummy.
    """

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
