"""Checks for the tracked name-free processed dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    ROOT
    / "data"
    / "processed"
    / "association_croquet_games_1985_2025_no_names.csv.gz"
)
MANIFEST = DATASET.with_name("manifest.json")
REQUIRED_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
FORBIDDEN_COLUMNS = {
    "winner_name",
    "loser_name",
    "player_name",
    "surname",
    "firstname",
    "initials",
    "oldname",
    "altfirstname",
    "oldfirstname",
    "birthy",
    "birthm",
    "birthd",
    "deathy",
    "deathm",
    "deathd",
    "namechange",
    "firstnamechange",
}


class PublicDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.columns = pd.read_csv(DATASET, nrows=0).columns.tolist()
        cls.games = pd.read_csv(DATASET, usecols=REQUIRED_COLUMNS, low_memory=False)

    def test_dataset_exists_and_is_readable(self) -> None:
        self.assertTrue(DATASET.is_file())
        self.assertGreater(DATASET.stat().st_size, 0)
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(self.columns))

    def test_frozen_counts_and_year_range(self) -> None:
        players = pd.concat([self.games["winner"], self.games["loser"]])
        self.assertEqual(len(self.games), 456_382)
        self.assertEqual(players.nunique(), 5_143)
        self.assertEqual(int(self.games["year"].eq(2025).sum()), 11_379)
        self.assertEqual(
            (int(self.games["year"].min()), int(self.games["year"].max())),
            (1985, 2025),
        )

    def test_game_and_player_identifiers(self) -> None:
        self.assertTrue(self.games["fcode"].is_unique)
        self.assertFalse(self.games["fcode"].isna().any())
        self.assertFalse(self.games["winner"].isna().any())
        self.assertFalse(self.games["loser"].isna().any())
        self.assertTrue(self.games["winner"].ne(self.games["loser"]).all())
        player_ids = self.games[["winner", "loser"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        self.assertFalse(player_ids.isna().any().any())
        self.assertTrue(player_ids.mod(1).eq(0).all().all())

    def test_direct_personal_columns_are_absent(self) -> None:
        self.assertFalse(FORBIDDEN_COLUMNS.intersection(self.columns))

    def test_manifest_sha256_matches(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with DATASET.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.assertEqual(digest.hexdigest(), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
