"""Tests for grouped Glicko rating periods."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.pipelines.glicko_rating_period import run_rating_period


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs" / "dissertation_evidence" / "chapter4" / "glicko_rating_period_metrics.csv"


class GlickoRatingPeriodTests(unittest.TestCase):
    def test_grouped_period_uses_opening_state_for_every_prediction(self) -> None:
        matches = pd.DataFrame(
            {
                "fcode": [1, 2],
                "year": [2025, 2025],
                "event": [10, 10],
                "event_order_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                "winner": [1, 1],
                "loser": [2, 2],
            }
        )
        grouped, _ = run_rating_period(matches, "event_level")
        sequential, _ = run_rating_period(matches, "match_by_match")
        self.assertEqual(grouped["pred_a_win"].tolist(), [0.5, 0.5])
        self.assertGreater(float(sequential.iloc[1]["pred_a_win"]), 0.5)
        self.assertTrue((grouped["player_a_rd_before_period"] == 350.0).all())

    def test_frozen_period_metrics(self) -> None:
        metrics = pd.read_csv(METRICS_PATH).set_index("period_type")
        expected = {
            "match_by_match": (0.1957076933405641, 0.5719575739115588, 0.6937340715352843),
            "event_level": (0.2012320209189822, 0.5881669000419171, 0.6847701907021707),
            "monthly": (0.20214477628937463, 0.5899327343359905, 0.6851217154407241),
            "yearly": (0.2102503776945106, 0.6107045994167629, 0.6671939537744969),
        }
        for period, values in expected.items():
            self.assertAlmostEqual(float(metrics.loc[period, "brier_score"]), values[0], places=15)
            self.assertAlmostEqual(float(metrics.loc[period, "log_loss"]), values[1], places=15)
            self.assertAlmostEqual(float(metrics.loc[period, "accuracy"]), values[2], places=15)


if __name__ == "__main__":
    unittest.main()
