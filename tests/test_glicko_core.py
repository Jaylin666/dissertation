"""Tests for the single canonical Glicko-1 implementation."""

from __future__ import annotations

import math
import unittest

from code.config import GLICKO_LOW_INFLATION
from code.models.glicko import (
    DEFAULT_RATING,
    DEFAULT_RD,
    MAX_RD,
    MIN_RD,
    expected_score,
    update_two_players_single_game,
)


class GlickoCoreTests(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(DEFAULT_RATING, 1500.0)
        self.assertEqual(DEFAULT_RD, 350.0)
        self.assertEqual(MIN_RD, 30.0)
        self.assertEqual(MAX_RD, 350.0)

    def test_expected_probability_bounds(self) -> None:
        for rating in (500.0, 1000.0, 1500.0, 2000.0, 2500.0):
            probability = expected_score(rating, 1500.0, DEFAULT_RD)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
        self.assertAlmostEqual(expected_score(1500.0, 1500.0, 350.0), 0.5)

    def test_single_game_update_directions(self) -> None:
        update = update_two_players_single_game(
            1500.0,
            350.0,
            1500.0,
            350.0,
            1.0,
        )
        self.assertGreater(update.player1_rating_after, update.player1_rating_before)
        self.assertLess(update.player2_rating_after, update.player2_rating_before)
        self.assertLess(update.player1_rd_after, update.player1_rd_before)
        self.assertLess(update.player2_rd_after, update.player2_rd_before)

    def test_low_inflation_configuration_is_unchanged(self) -> None:
        expected = math.sqrt(((350.0**2) - (30.0**2)) / 240.0)
        self.assertAlmostEqual(GLICKO_LOW_INFLATION.inactivity_c, expected, places=15)
        self.assertAlmostEqual(
            GLICKO_LOW_INFLATION.inactivity_c,
            22.50925735484551,
            places=12,
        )
        self.assertEqual(GLICKO_LOW_INFLATION.rating_period, "match_by_match")
        self.assertEqual(GLICKO_LOW_INFLATION.inactivity_unit, "month")


if __name__ == "__main__":
    unittest.main()
