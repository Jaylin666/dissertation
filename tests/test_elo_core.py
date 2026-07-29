"""Tests for the canonical Elo equations."""

from __future__ import annotations

import unittest

from code.config import DEFAULT_ELO, VALIDATION_BEST_ELO
from code.models.elo import (
    adaptive_k_previous_year,
    adaptive_k_total,
    expected_score,
    update_with_config,
)


class EloCoreTests(unittest.TestCase):
    def test_expected_probability_bounds_and_symmetry(self) -> None:
        for difference in (-2000.0, -500.0, 0.0, 500.0, 2000.0):
            probability = expected_score(1500.0 + difference, 1500.0, 300.0)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
        self.assertAlmostEqual(expected_score(1500.0, 1500.0, 500.0), 0.5)
        probability = expected_score(1600.0, 1400.0, 300.0)
        complement = expected_score(1400.0, 1600.0, 300.0)
        self.assertAlmostEqual(probability + complement, 1.0, places=15)

    def test_winner_and_loser_move_in_expected_directions(self) -> None:
        update = update_with_config(1500.0, 1500.0, DEFAULT_ELO)
        self.assertGreater(update.winner_rating_after, update.winner_rating_before)
        self.assertLess(update.loser_rating_after, update.loser_rating_before)
        self.assertAlmostEqual(update.rating_change, 10.0, places=15)
        self.assertAlmostEqual(
            update.winner_rating_after + update.loser_rating_after,
            3000.0,
            places=15,
        )

    def test_frozen_validation_best_parameters(self) -> None:
        self.assertEqual(VALIDATION_BEST_ELO.k_factor, 30.0)
        self.assertEqual(VALIDATION_BEST_ELO.scale, 300.0)
        self.assertEqual(VALIDATION_BEST_ELO.initial_rating, 1500.0)

    def test_historical_adaptive_k_boundaries(self) -> None:
        self.assertEqual(adaptive_k_total(19), 30.0)
        self.assertEqual(adaptive_k_total(20), 20.0)
        self.assertEqual(adaptive_k_total(100), 10.0)
        self.assertEqual(adaptive_k_previous_year(5), 30.0)
        self.assertEqual(adaptive_k_previous_year(6), 20.0)
        self.assertEqual(adaptive_k_previous_year(31), 10.0)


if __name__ == "__main__":
    unittest.main()
