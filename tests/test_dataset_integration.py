from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pca_ann_pipeline import (
    FEATURE_COLUMNS,
    extract_features,
    preprocess_rows,
    read_dataset,
)


DATASET = Path(__file__).resolve().parents[1] / "Data Validasi & Pengujian (1).xlsx"


@unittest.skipUnless(DATASET.exists(), "Dataset workbook tidak tersedia.")
class ActualDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = read_dataset(DATASET, sheet="Data")
        cls.preprocessing = preprocess_rows(raw)
        cls.extraction = extract_features(
            cls.preprocessing.cleaned,
            baseline_seconds=60,
            exposure_seconds=120,
            baseline_anchor="tail",
        )

    def test_expected_row_and_phase_counts(self):
        self.assertEqual(self.preprocessing.source_rows, 8629)
        self.assertEqual(self.preprocessing.dropped_missing_metadata, 274)
        self.assertEqual(self.preprocessing.rows_after_metadata_filter, 8355)
        self.assertEqual(len(self.preprocessing.cleaned), 5186)
        self.assertEqual(
            self.preprocessing.phase_counts,
            {"exposure": 3455, "baseline": 1731},
        )
        self.assertEqual(
            self.preprocessing.phase_counts_before_filter,
            {"exposure": 3455, "purging": 3169, "baseline": 1731},
        )
        self.assertEqual(
            self.preprocessing.ignored_phase_counts,
            {"purging": 3169},
        )
        self.assertEqual(
            self.preprocessing.missing_numeric_counts,
            {"hcho": 0, "mq138": 0, "tgs822": 0, "humidity": 3},
        )

    def test_expected_sample_and_class_counts(self):
        features = self.extraction.features
        self.assertEqual(len(features), 24)
        self.assertEqual(features["label"].value_counts().to_dict(), {1: 19, 0: 5})
        self.assertEqual(
            set(
                item["sample_id"]
                for item in self.extraction.excluded_samples
                if item["reason"] == "insufficient_phase_duration"
            ),
            {"5mL_rep1"},
        )
        self.assertFalse(
            features["qc_flags"].str.contains(
                "baseline_duration_short|exposure_duration_short",
                regex=True,
                na=False,
            ).any()
        )

    def test_all_13_features_are_finite_and_deltas_match(self):
        features = self.extraction.features
        matrix = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(matrix).all())
        for sensor in ("HCHO", "MQ138", "TGS822"):
            expected = (
                features[f"{sensor}_exposure_max"]
                - features[f"{sensor}_baseline_mean"]
            )
            np.testing.assert_allclose(
                features[f"{sensor}_delta_max"],
                expected,
                rtol=1e-12,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
