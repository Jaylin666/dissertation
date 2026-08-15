"""Tests for chronological adaptive-K Elo."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.pipelines.adaptive_k_pipeline import (
    ADAPTIVE_CANDIDATES,
    RETAINED_MODEL,
    run_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveKPipelineTests(unittest.TestCase):
    def test_counts_are_prematch_and_updates_can_be_asymmetric(self) -> None:
        prior = pd.DataFrame(
            {
                "fcode": range(1, 101),
                "year": [2024] * 100,
                "event": [1] * 100,
                "winner": [1] * 100,
                "loser": range(1000, 1100),
            }
        )
        current = pd.DataFrame(
            {"fcode": [101], "year": [2025], "event": [2], "winner": [1], "loser": [5000]}
        )
        candidate = next(item for item in ADAPTIVE_CANDIDATES if item.name == RETAINED_MODEL)
        predictions, _ = run_candidate(pd.concat([prior, current], ignore_index=True), candidate)
        row = predictions.iloc[0]
        self.assertEqual(int(row["winner_previous_total_games"]), 100)
        self.assertEqual(int(row["winner_previous_year_games"]), 100)
        self.assertEqual(int(row["loser_previous_total_games"]), 0)
        self.assertEqual(float(row["winner_k"]), 10.0)
        self.assertEqual(float(row["loser_k"]), 30.0)
        self.assertNotAlmostEqual(
            float(row["winner_rating_change"] + row["loser_rating_change"]),
            0.0,
        )

    def test_retained_candidate_metrics_are_frozen(self) -> None:
        metrics = pd.read_csv(
            ROOT / "outputs" / "dissertation_evidence" / "chapter4" / "overall_model_metrics.csv"
        ).set_index("model")
        row = metrics.loc["best_AdaptiveK"]
        self.assertAlmostEqual(float(row["brier"]), 0.190781422536, places=12)
        self.assertAlmostEqual(float(row["log_loss"]), 0.559184791547, places=12)
        self.assertAlmostEqual(float(row["accuracy"]), 0.706213199754, places=12)
        self.assertEqual(RETAINED_MODEL, "AdaptiveK_PreviousYearGames_Elo_scale300")


if __name__ == "__main__":
    unittest.main()
