"""Tests for prematch features and unified comparison validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from code.config import EXPECTED_TEST_MATCHES
from code.pipelines.comparison_inputs import (
    build_prematch_features,
    validate_comparison_input,
)


class ComparisonInputTests(unittest.TestCase):
    def test_features_are_saved_before_current_game_enters_history(self) -> None:
        matches = pd.DataFrame(
            {
                "fcode": [1, 2, 3],
                "year": [2025, 2025, 2025],
                "event": [1, 2, 3],
                "event_order_date": pd.to_datetime(["2025-01-01", "2025-01-02", None]),
                "event_date_ordering_method": ["parsed_full_date", "parsed_full_date", "fallback_no_date"],
                "winner": [1, 1, 1],
                "loser": [2, 3, 4],
            }
        )
        long, games = build_prematch_features(matches)
        player_one = long.loc[long["player_id"].eq(1)].sort_values("match_sequence")
        self.assertEqual(player_one["total_games_before"].tolist(), [0, 1, 2])
        self.assertEqual(player_one["recorded_appearance_number"].tolist(), [1, 2, 3])
        self.assertFalse(bool(player_one.iloc[2]["date_features_available"]))
        self.assertEqual(player_one.iloc[2]["date_quality"], "missing")
        self.assertEqual(len(games), 3)
        self.assertTrue((games["player_a_id"] < games["player_b_id"]).all())

    def test_unified_validator_requires_11379_unique_canonical_games(self) -> None:
        player_a = np.arange(1, EXPECTED_TEST_MATCHES + 1)
        table = pd.DataFrame(
            {
                "match_id": player_a,
                "match_sequence": player_a,
                "player_a_id": player_a,
                "player_b_id": player_a + 20_000,
                "winner_id": player_a,
                "loser_id": player_a + 20_000,
                "outcome_a": np.ones(EXPECTED_TEST_MATCHES, dtype=int),
                "a_total_games_before": np.zeros(EXPECTED_TEST_MATCHES, dtype=int),
                "b_total_games_before": np.zeros(EXPECTED_TEST_MATCHES, dtype=int),
                "p_a_Conservative_Elo": 0.5,
                "p_a_Default_Elo": 0.5,
                "p_a_Validation_best_Elo": 0.5,
                "p_a_best_AdaptiveK": 0.5,
                "p_a_Glicko_C0_fixed": 0.5,
                "p_a_Glicko_low_fixed": 0.5,
            }
        )
        validate_comparison_input(table)


if __name__ == "__main__":
    unittest.main()
