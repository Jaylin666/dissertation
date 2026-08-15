"""Tests for active final-figure generators."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from code.analysis.early_game import (
    STAGE_LABELS,
    create_predicted_vs_empirical_replot,
    create_stage_brier_replot,
)
from code.analysis.entry_diagnostics import create_prematch_scale_alignment_replot
from code.pipelines.comparison_pipeline import (
    create_overall_brier_figure,
    create_player_a_calibration_gap_figure,
)


ROOT = Path(__file__).resolve().parents[1]
CHAPTER4 = ROOT / "outputs" / "dissertation_evidence" / "chapter4"
CHAPTER5 = ROOT / "outputs" / "dissertation_evidence" / "chapter5"


class FigureGeneratorTests(unittest.TestCase):
    def test_chapter4_figure_generators_use_expected_series(self) -> None:
        metrics = pd.read_csv(CHAPTER4 / "overall_model_metrics.csv")
        calibration = pd.read_csv(CHAPTER4 / "calibration_bins.csv")
        self.assertEqual(metrics["model"].nunique(), 6)
        overall_bins = calibration.loc[calibration["sample"].eq("Overall")]
        self.assertEqual(overall_bins["bin_label"].nunique(), 10)
        with TemporaryDirectory() as directory:
            brier = Path(directory) / "overall_brier_zoomed.png"
            calibration_path = Path(directory) / "calibration_player_a.png"
            create_overall_brier_figure(metrics, brier)
            create_player_a_calibration_gap_figure(calibration, calibration_path)
            self.assertGreater(brier.stat().st_size, 0)
            self.assertGreater(calibration_path.stat().st_size, 0)

    def test_chapter5_figure_generators_use_expected_stages_and_metrics(self) -> None:
        stage = pd.read_csv(CHAPTER5 / "early_game_stage_core.csv", dtype={"appearance_stage": str})
        scale = pd.read_csv(CHAPTER5 / "prematch_scale_alignment_2025_core.csv")
        self.assertEqual(stage["appearance_stage"].tolist(), STAGE_LABELS)
        self.assertIn("mean_actual_first_opponent_rating", set(scale["metric"]))
        with TemporaryDirectory() as directory:
            stage_path = Path(directory) / "stage_brier_replot.png"
            predicted_path = Path(directory) / "predicted_vs_empirical_replot.png"
            scale_path = Path(directory) / "prematch_scale_alignment_2025_replot.png"
            create_stage_brier_replot(stage, stage_path)
            create_predicted_vs_empirical_replot(stage, predicted_path)
            create_prematch_scale_alignment_replot(scale, scale_path)
            for path in (stage_path, predicted_path, scale_path):
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
