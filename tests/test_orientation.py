"""Tests for canonical and focal probability orientation."""

from __future__ import annotations

import unittest

from code.analysis.orientation import (
    canonical_outcome,
    canonical_players,
    direct_player_a_glicko_probability,
    focal_outcome,
    focal_probability,
)


class OrientationTests(unittest.TestCase):
    def test_canonical_players_do_not_depend_on_outcome(self) -> None:
        self.assertEqual(canonical_players(20, 10), (10, 20))
        self.assertEqual(canonical_players(10, 20), (10, 20))
        self.assertEqual(canonical_outcome(10, 10, 20), 1)
        self.assertEqual(canonical_outcome(20, 10, 20), 0)

    def test_focal_orientation_matches_frozen_step34_convention(self) -> None:
        self.assertEqual(focal_outcome(1, "a"), 1)
        self.assertEqual(focal_outcome(1, "b"), 0)
        self.assertEqual(focal_outcome(0, "a"), 0)
        self.assertEqual(focal_outcome(0, "b"), 1)
        self.assertAlmostEqual(focal_probability(0.73, "a"), 0.73)
        self.assertAlmostEqual(focal_probability(0.73, "b"), 0.27)

    def test_step33_direct_player_a_probability_uses_opponent_rd(self) -> None:
        probability = direct_player_a_glicko_probability(
            rating_a=1500.0,
            rating_b=1200.0,
            rd_b=80.0,
        )
        self.assertGreater(probability, 0.5)
        reverse_with_different_rd = direct_player_a_glicko_probability(
            rating_a=1200.0,
            rating_b=1500.0,
            rd_b=350.0,
        )
        self.assertNotAlmostEqual(
            probability + reverse_with_different_rd,
            1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
