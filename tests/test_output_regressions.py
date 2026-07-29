"""Regression tests against tracked compact golden result tables."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.config import (
    EXPECTED_TEST_MATCHES,
    FIRST_APPEARANCE_GOLDEN,
)


ROOT = Path(__file__).resolve().parents[1]


class GoldenOutputRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.overall = pd.read_csv(
            ROOT / "outputs" / "meeting6" / "33_overall_model_metrics.csv"
        )
        cls.early = pd.read_csv(
            ROOT
            / "outputs"
            / "meeting7"
            / "34_cumulative_threshold_model_performance.csv"
        )
        cls.entry = pd.read_csv(
            ROOT
            / "outputs"
            / "meeting8_technical"
            / "42_entry_cohort_scale_summary.csv"
        )

    def test_overall_core_model_metrics_are_frozen(self) -> None:
        metrics = self.overall.set_index("model")
        expected = {
            "Glicko_low_fixed": (0.187603868924, 0.5517787243),
            "Validation_best_Elo": (0.19007264006, 0.556534255949),
            "Glicko_C0_fixed": (0.195707693341, 0.571957573912),
            "Default_Elo": (0.194155553153, 0.567632617282),
            "Conservative_Elo": (0.201418477877, 0.585927563619),
        }
        for model, (brier, log_loss) in expected.items():
            self.assertEqual(int(metrics.loc[model, "evaluation_games"]), EXPECTED_TEST_MATCHES)
            self.assertAlmostEqual(float(metrics.loc[model, "brier"]), brier, places=12)
            self.assertAlmostEqual(
                float(metrics.loc[model, "log_loss"]),
                log_loss,
                places=12,
            )

    def test_first_appearance_headline_values(self) -> None:
        first = self.early[
            (self.early["group_type"] == "cumulative_threshold")
            & (self.early["group"] == "first_1")
            & (self.early["model"] == "Glicko_low_fixed")
        ].iloc[0]
        self.assertEqual(int(first["appearances"]), 76)
        self.assertAlmostEqual(
            float(first["mean_predicted_probability"]),
            FIRST_APPEARANCE_GOLDEN.mean_current_probability,
            places=12,
        )
        self.assertAlmostEqual(
            float(first["empirical_win_rate"]),
            FIRST_APPEARANCE_GOLDEN.empirical_win_rate,
            places=12,
        )
        self.assertAlmostEqual(
            float(first["brier"]),
            FIRST_APPEARANCE_GOLDEN.brier_score,
            places=6,
        )

    def test_entry_scale_headline_values(self) -> None:
        row = self.entry[
            self.entry["group"] == "test_year_recorded_entry"
        ].iloc[0]
        self.assertAlmostEqual(
            float(row["mean_opponent_rating"]),
            FIRST_APPEARANCE_GOLDEN.mean_opponent_rating,
            places=6,
        )
        self.assertAlmostEqual(
            float(row["mean_initial_minus_opponent_rating"]),
            FIRST_APPEARANCE_GOLDEN.mean_initial_minus_opponent_gap,
            places=6,
        )
        self.assertAlmostEqual(
            float(row["mean_initial_minus_contemporaneous_median"]),
            FIRST_APPEARANCE_GOLDEN.mean_initial_minus_contemporaneous_median_gap,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
