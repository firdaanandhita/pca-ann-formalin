from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pca_ann_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    parse_concentration_ml,
    preprocess_inference_rows,
    preprocess_rows,
    resolve_columns,
)


class ConcentrationParserTests(unittest.TestCase):
    def test_zero_and_positive_values(self):
        self.assertEqual(parse_concentration_ml("0 mL"), 0.0)
        self.assertEqual(parse_concentration_ml("0ml"), 0.0)
        self.assertEqual(parse_concentration_ml(0), 0.0)
        self.assertEqual(parse_concentration_ml("1 mL"), 1.0)
        self.assertEqual(parse_concentration_ml("15mL"), 15.0)
        self.assertEqual(parse_concentration_ml("1,5 ml"), 1.5)

    def test_negative_and_non_numeric_values_fail(self):
        with self.assertRaises(ValueError):
            parse_concentration_ml("-1 mL")
        with self.assertRaises(ValueError):
            parse_concentration_ml("tidak diketahui")
        with self.assertRaises(ValueError):
            parse_concentration_ml("1 liter")


class ColumnResolutionTests(unittest.TestCase):
    def test_actual_header_aliases(self):
        columns = [
            "Timestamp",
            "HCHO",
            "MQ-138",
            "TGS822",
            "HUMIDITY",
            "Konsentrasi",
            "Replikasi",
            "Fase",
        ]
        mapping = resolve_columns(columns)
        self.assertEqual(mapping["mq138"], "MQ-138")
        self.assertEqual(mapping["humidity"], "HUMIDITY")


class PreprocessingAndFeatureTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> pd.DataFrame:
        baseline_time = pd.date_range(
            "2026-01-01 00:00:00", periods=10, freq="10s"
        )
        exposure_time = pd.date_range(
            "2026-01-01 00:02:00", periods=15, freq="10s"
        )
        baseline = pd.DataFrame(
            {
                "Timestamp": baseline_time,
                "HCHO": np.arange(10, dtype=float),
                "MQ-138": np.arange(10, dtype=float) + 10,
                "TGS822": np.arange(10, dtype=float) + 20,
                "HUMIDITY": np.arange(10, dtype=float) + 50,
                "Konsentrasi": "0 mL",
                "Replikasi": 1,
                "Fase": "Baseline",
            }
        )
        exposure = pd.DataFrame(
            {
                "Timestamp": exposure_time,
                "HCHO": np.arange(15, dtype=float) + 100,
                "MQ-138": np.arange(15, dtype=float) + 200,
                "TGS822": np.arange(15, dtype=float) + 300,
                "HUMIDITY": np.arange(15, dtype=float) + 60,
                "Konsentrasi": "0 mL",
                "Replikasi": 1,
                "Fase": "Exposure",
            }
        )
        missing_metadata = exposure.iloc[[0]].copy()
        missing_metadata[["Konsentrasi", "Replikasi", "Fase"]] = np.nan
        return pd.concat(
            [missing_metadata, baseline, exposure], ignore_index=True
        )

    def test_drop_metadata_and_binary_label(self):
        raw = self._fixture()
        result = preprocess_rows(raw)
        self.assertEqual(result.source_rows, 26)
        self.assertEqual(result.dropped_missing_metadata, 1)
        self.assertEqual(len(result.cleaned), 25)
        self.assertEqual(set(result.cleaned["label"]), {0})

    def test_time_based_feature_windows_and_delta(self):
        cleaned = preprocess_rows(self._fixture()).cleaned
        result = extract_features(
            cleaned,
            baseline_seconds=60,
            exposure_seconds=120,
            baseline_anchor="tail",
        )
        self.assertEqual(len(result.features), 1)
        row = result.features.iloc[0]
        self.assertEqual(list(result.features[FEATURE_COLUMNS].columns), FEATURE_COLUMNS)

        expected_baseline_hcho = np.arange(3, 10, dtype=float).mean()
        expected_exposure_hcho = np.arange(13, dtype=float).mean() + 100
        expected_exposure_max = 112.0
        self.assertAlmostEqual(row["HCHO_baseline_mean"], expected_baseline_hcho)
        self.assertAlmostEqual(row["HCHO_exposure_mean"], expected_exposure_hcho)
        self.assertAlmostEqual(row["HCHO_exposure_max"], expected_exposure_max)
        self.assertAlmostEqual(
            row["HCHO_delta_max"],
            expected_exposure_max - expected_baseline_hcho,
        )
        self.assertAlmostEqual(row["RH_mean"], np.arange(13).mean() + 60)

    def test_purging_is_excluded_from_preprocessed_rows(self):
        raw = self._fixture()
        purging = raw.loc[raw["Fase"] == "Exposure"].iloc[[0]].copy()
        purging["Fase"] = "Purging"
        purging["Timestamp"] = pd.Timestamp("2026-01-01 00:10:00")
        result = preprocess_rows(pd.concat([raw, purging], ignore_index=True))

        self.assertNotIn("purging", set(result.cleaned["phase_normalized"]))
        self.assertEqual(result.ignored_phase_counts, {"purging": 1})

    def test_unlabeled_inference_rows_do_not_require_concentration(self):
        raw = self._fixture().drop(columns=["Konsentrasi", "Replikasi"])
        prepared = preprocess_inference_rows(raw, sample_id="uji_lapangan")
        extracted = extract_features(
            prepared.cleaned,
            baseline_seconds=60,
            exposure_seconds=120,
            short_window_policy="error",
        )
        self.assertEqual(len(extracted.features), 1)
        self.assertEqual(extracted.features.iloc[0]["sample_id"], "uji_lapangan")

    def test_one_hz_endpoint_convention_meets_resume_duration(self):
        baseline = pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    "2026-01-01", periods=60, freq="1s"
                ),
                "HCHO": 0.1,
                "MQ-138": 0.2,
                "TGS822": 0.3,
                "HUMIDITY": 50.0,
                "Konsentrasi": "0 mL",
                "Replikasi": 1,
                "Fase": "Baseline",
            }
        )
        exposure = pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    "2026-01-01 00:01:00", periods=120, freq="1s"
                ),
                "HCHO": 0.15,
                "MQ-138": 0.25,
                "TGS822": 0.35,
                "HUMIDITY": 51.0,
                "Konsentrasi": "0 mL",
                "Replikasi": 1,
                "Fase": "Exposure",
            }
        )
        cleaned = preprocess_rows(
            pd.concat([baseline, exposure], ignore_index=True)
        ).cleaned
        result = extract_features(
            cleaned,
            short_window_policy="error",
        )
        row = result.features.iloc[0]
        self.assertAlmostEqual(row["baseline_effective_coverage_seconds"], 60.0)
        self.assertAlmostEqual(row["exposure_effective_coverage_seconds"], 120.0)


if __name__ == "__main__":
    unittest.main()
