"""Tests for compact-output regression checks."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from code.validation_utils import REPRODUCTION_EVIDENCE_TABLES, compare_csv_tables


ROOT = Path(__file__).resolve().parents[1]


class ReproductionValidationTests(unittest.TestCase):
    def test_every_mapped_evidence_table_is_tracked(self) -> None:
        names = [tracked for _, tracked, _ in REPRODUCTION_EVIDENCE_TABLES]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 19)
        for name in names:
            self.assertTrue((ROOT / "outputs" / "dissertation_evidence" / name).exists())

    def test_table_comparison_preserves_row_order(self) -> None:
        reference = pd.DataFrame({"id": [1, 2], "value": [0.1, 0.2]})
        same_values = pd.DataFrame({"id": [1, 2], "value": [0.1 + 1e-12, 0.2]})
        reversed_rows = reference.iloc[::-1].reset_index(drop=True)
        self.assertEqual(compare_csv_tables(same_values, reference), [])
        self.assertTrue(compare_csv_tables(reversed_rows, reference))


if __name__ == "__main__":
    unittest.main()
