"""Regression tests for burn-in and first-recorded-entry definitions."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.analysis.entry_diagnostics import cohort_for_year
from code.config import (
    EXPECTED_BOTH_DEBUT_MATCHES,
    EXPECTED_DEBUT_APPEARANCES,
    EXPECTED_DEBUT_MATCHES,
    EXPECTED_DEBUT_PLAYERS,
    EXPECTED_EXACTLY_ONE_DEBUT_MATCHES,
    EXPECTED_POST_BURN_IN_PLAYERS_1990_2024,
    EXPECTED_SYSTEM_START_PLAYERS,
    EXPECTED_TEST_YEAR_ENTRANTS,
    EXPECTED_UNIQUE_PLAYERS,
    EXPECTED_WITHIN_BURN_IN_PLAYERS,
)


ROOT = Path(__file__).resolve().parents[1]
MEETING8 = ROOT / "outputs" / "meeting8_technical"


class EntryDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohorts = pd.read_csv(MEETING8 / "42_entry_cohort_scale_summary.csv")
        cls.audit = pd.read_csv(MEETING8 / "42_2025_crossfile_entry_audit.csv")

    def test_cohort_boundaries(self) -> None:
        self.assertEqual(cohort_for_year(1985), "system_start_left_censored")
        self.assertEqual(cohort_for_year(1989), "within_5y_burn_in_recorded_entry")
        self.assertEqual(cohort_for_year(1990), "post_burn_in_recorded_entry")
        self.assertEqual(cohort_for_year(2024), "post_burn_in_recorded_entry")
        self.assertEqual(cohort_for_year(2025), "test_year_recorded_entry")

    def test_frozen_burnin_cohort_counts(self) -> None:
        counts = self.cohorts.set_index("group")["n_unique_players"].astype(int)
        self.assertEqual(
            int(counts["system_start_left_censored"]),
            EXPECTED_SYSTEM_START_PLAYERS,
        )
        self.assertEqual(
            int(counts["within_5y_burn_in_recorded_entry"]),
            EXPECTED_WITHIN_BURN_IN_PLAYERS,
        )
        self.assertEqual(
            int(counts["post_burn_in_recorded_entry"]),
            EXPECTED_POST_BURN_IN_PLAYERS_1990_2024,
        )
        self.assertEqual(
            int(counts["test_year_recorded_entry"]),
            EXPECTED_TEST_YEAR_ENTRANTS,
        )
        total = (
            EXPECTED_SYSTEM_START_PLAYERS
            + EXPECTED_WITHIN_BURN_IN_PLAYERS
            + EXPECTED_POST_BURN_IN_PLAYERS_1990_2024
            + EXPECTED_TEST_YEAR_ENTRANTS
        )
        self.assertEqual(total, EXPECTED_UNIQUE_PLAYERS)

    def test_frozen_first_appearance_audit(self) -> None:
        self.assertEqual(len(self.audit), EXPECTED_DEBUT_APPEARANCES)
        self.assertEqual(self.audit["player_id"].nunique(), EXPECTED_DEBUT_PLAYERS)
        self.assertEqual(self.audit["match_id"].nunique(), EXPECTED_DEBUT_MATCHES)
        match_counts = self.audit.groupby("match_id").size()
        self.assertEqual(int((match_counts == 1).sum()), EXPECTED_EXACTLY_ONE_DEBUT_MATCHES)
        self.assertEqual(int((match_counts == 2).sum()), EXPECTED_BOTH_DEBUT_MATCHES)

    def test_step33_step34_and_full_history_keys_agree(self) -> None:
        self.assertTrue(self.audit["in_full_history"].astype(bool).all())
        self.assertTrue(self.audit["in_step33"].astype(bool).all())
        self.assertTrue(self.audit["in_step34"].astype(bool).all())
        self.assertTrue(self.audit["match_sequence_agrees"].astype(bool).all())
        self.assertTrue(self.audit["opponent_id_agrees"].astype(bool).all())
        self.assertTrue(self.audit["focal_side_agrees"].astype(bool).all())
        self.assertTrue(self.audit["all_sources_agree"].astype(bool).all())


if __name__ == "__main__":
    unittest.main()
