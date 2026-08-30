from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from predict_raw import (
    InputQualityError,
    file_sha256,
    load_model_bundle,
    predict_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs" / "model_pca_ann.pkl"
MANIFEST = ROOT / "outputs" / "model_manifest.json"
DEPLOYMENT_DIR = ROOT / "outputs" / "deployment_tests"
DUMMY_NON = DEPLOYMENT_DIR / "dummy_non_formalin.csv"
DUMMY_FORMAL = DEPLOYMENT_DIR / "dummy_formalin.csv"


@unittest.skipUnless(
    all(path.exists() for path in [MODEL, MANIFEST, DUMMY_NON, DUMMY_FORMAL]),
    "Artefak deployment belum dibuat.",
)
class SavedDeploymentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(MODEL)
        cls.non_formalin = pd.read_csv(DUMMY_NON)
        cls.formalin = pd.read_csv(DUMMY_FORMAL)

    def test_pickle_hash_matches_manifest(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        record = next(
            artifact
            for artifact in manifest["artifacts"]
            if artifact["path"] == MODEL.name
        )
        self.assertEqual(file_sha256(MODEL), record["sha256"])

    def test_dummy_predictions_match_expected_classes(self):
        non_result = predict_dataframe(
            self.non_formalin,
            self.bundle,
            sample_id="unittest_dummy_non",
        )
        formal_result = predict_dataframe(
            self.formalin,
            self.bundle,
            sample_id="unittest_dummy_formal",
        )
        self.assertEqual(non_result["predicted_label"], 0)
        self.assertEqual(formal_result["predicted_label"], 1)

    def test_short_baseline_is_rejected_by_default(self):
        short = pd.concat(
            [
                self.non_formalin.loc[
                    self.non_formalin["Fase"].eq("Baseline")
                ].head(30),
                self.non_formalin.loc[
                    self.non_formalin["Fase"].eq("Exposure")
                ],
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(
            InputQualityError, "baseline_duration_short"
        ):
            predict_dataframe(
                short,
                self.bundle,
                sample_id="unittest_short_baseline",
            )


if __name__ == "__main__":
    unittest.main()
