"""Tests for common initial-rating translation invariance."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from code.analysis.initial_rating_sensitivity import (
    INITIAL_RATING_CANDIDATES,
    run_candidate,
)


class InitialRatingSensitivityTests(unittest.TestCase):
    def test_candidate_set_is_frozen(self) -> None:
        self.assertEqual(INITIAL_RATING_CANDIDATES, (1000, 1100, 1200, 1300, 1400, 1500))

    def test_common_shift_preserves_probabilities(self) -> None:
        matches = pd.DataFrame(
            {
                "fcode": [1, 2, 3, 4],
                "year": [2023] * 4,
                "event": [1, 1, 2, 2],
                "inactivity_period_index": [1, 1, 2, 2],
                "winner": [1, 2, 1, 3],
                "loser": [2, 3, 3, 2],
            }
        )
        low_predictions, low_ratings = run_candidate(matches, 1000, 0.0)
        high_predictions, high_ratings = run_candidate(matches, 1500, 0.0)
        np.testing.assert_array_equal(
            low_predictions["pred_a_win"].to_numpy(),
            high_predictions["pred_a_win"].to_numpy(),
        )
        aligned = low_ratings.merge(high_ratings, on="player_id", suffixes=("_low", "_high"))
        np.testing.assert_allclose(aligned["rating_high"] - aligned["rating_low"], 500.0)
        np.testing.assert_array_equal(aligned["rd_low"], aligned["rd_high"])


if __name__ == "__main__":
    unittest.main()
