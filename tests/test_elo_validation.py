"""Tests for the frozen Elo validation workflow."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.pipelines.elo_validation import (
    K_VALUES,
    SCALE_VALUES,
    select_best_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "outputs" / "dissertation_evidence" / "chapter4" / "elo_validation_grid.csv"


class EloValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = pd.read_csv(GRID_PATH)

    def test_frozen_grid_has_28_rows_and_21_ratios(self) -> None:
        self.assertEqual(len(K_VALUES) * len(SCALE_VALUES), 28)
        self.assertEqual(len(self.grid), 28)
        self.assertEqual((self.grid["K"] / self.grid["scale"]).nunique(), 21)

    def test_selected_parameters_and_equivalent_ratio(self) -> None:
        selected = select_best_parameters(self.grid)
        self.assertEqual((int(selected["K"]), int(selected["scale"])), (30, 300))
        equivalent = self.grid.loc[self.grid["K"].eq(40) & self.grid["scale"].eq(400)].iloc[0]
        self.assertEqual(selected["validation_log_loss"], equivalent["validation_log_loss"])
        self.assertEqual(selected["validation_brier_score"], equivalent["validation_brier_score"])

    def test_tie_break_prefers_smaller_k_then_scale(self) -> None:
        rows = pd.DataFrame(
            {
                "validation_log_loss": [0.5, 0.5, 0.5],
                "validation_brier_score": [0.2, 0.2, 0.2],
                "K": [30, 20, 20],
                "scale": [300, 500, 400],
                "test_log_loss": [0.1, 0.9, 0.2],
            }
        )
        selected = select_best_parameters(rows)
        self.assertEqual((int(selected["K"]), int(selected["scale"])), (20, 400))

    def test_selection_does_not_use_2025_metrics(self) -> None:
        altered = self.grid.copy()
        altered["test_log_loss"] = range(len(altered), 0, -1)
        before = select_best_parameters(self.grid)
        after = select_best_parameters(altered)
        self.assertEqual((before["K"], before["scale"]), (after["K"], after["scale"]))


if __name__ == "__main__":
    unittest.main()
