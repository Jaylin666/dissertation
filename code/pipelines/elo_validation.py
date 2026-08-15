"""Chronological Elo parameter validation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from code.analysis.orientation import canonical_outcome, canonical_players
from code.config import EXPECTED_VALIDATION_MATCHES
from code.io_utils import PROJECT_ROOT, ensure_directory
from code.models.elo import expected_score
from code.pipelines.elo_pipeline import FULL_HISTORY_PATH, load_matches


START_YEAR = 2015
END_YEAR = 2025
HISTORY_YEARS = tuple(range(2015, 2023))
VALIDATION_YEARS = (2023, 2024)
TEST_YEARS = (2025,)
K_VALUES = (10, 15, 20, 25, 30, 35, 40)
SCALE_VALUES = (300, 400, 500, 600)
DEFAULT_RATING = 1500.0
EPSILON = 1e-15


def _hash_ids(values: Iterable[int]) -> str:
    text = ",".join(str(int(value)) for value in values)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _metrics(predictions: pd.DataFrame, years: tuple[int, ...], prefix: str) -> dict[str, Any]:
    sample = predictions.loc[predictions["year"].isin(years)]
    if sample.empty:
        raise ValueError(f"No predictions for {prefix} years {years}")
    outcomes = sample["outcome_a"].to_numpy(dtype=float)
    probabilities = sample["pred_a_win"].to_numpy(dtype=float)
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    observed = float(outcomes.mean())
    return {
        f"{prefix}_log_loss": float(
            -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
        ),
        f"{prefix}_brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        f"{prefix}_accuracy": float(np.mean((probabilities >= 0.5) == (outcomes == 1.0))),
        f"{prefix}_baseline_accuracy": max(observed, 1.0 - observed),
        f"{prefix}_games": int(len(sample)),
        f"{prefix}_actual_a_win_count_0": int((outcomes == 0).sum()),
        f"{prefix}_actual_a_win_count_1": int((outcomes == 1).sum()),
        f"{prefix}_pred_a_win_min": float(probabilities.min()),
        f"{prefix}_pred_a_win_max": float(probabilities.max()),
        f"{prefix}_pred_a_win_out_of_range_count": int(
            ((probabilities < 0.0) | (probabilities > 1.0)).sum()
        ),
        f"{prefix}_fcode_hash": _hash_ids(sample["fcode"].astype(int)),
    }


def run_candidate(
    matches: pd.DataFrame,
    k_factor: float,
    scale: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run one parameter pair with prediction before update."""

    ratings: dict[int, float] = {}
    changes: list[float] = []
    rows: list[dict[str, Any]] = []
    skipped = 0

    columns = ["fcode", "year", "event", "winner", "loser"]
    sample = matches.loc[matches["year"].between(START_YEAR, END_YEAR), columns]
    for match in sample.itertuples(index=False):
        if pd.isna(match.winner) or pd.isna(match.loser):
            skipped += 1
            continue
        winner = int(match.winner)
        loser = int(match.loser)
        winner_before = ratings.get(winner, DEFAULT_RATING)
        loser_before = ratings.get(loser, DEFAULT_RATING)
        probability_winner = expected_score(winner_before, loser_before, scale)
        player_a, player_b = canonical_players(winner, loser)
        outcome_a = canonical_outcome(winner, player_a, player_b)
        probability_a = probability_winner if player_a == winner else 1.0 - probability_winner

        change = float(k_factor) * (1.0 - probability_winner)
        ratings[winner] = winner_before + change
        ratings[loser] = loser_before - change
        changes.append(abs(change))

        year = int(match.year)
        if year in VALIDATION_YEARS or year in TEST_YEARS:
            rows.append(
                {
                    "fcode": int(match.fcode),
                    "year": year,
                    "event": int(match.event),
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": probability_a,
                    "pred_winner_win": probability_winner,
                    "K": float(k_factor),
                    "scale": float(scale),
                }
            )

    final_values = pd.Series(ratings, dtype=float)
    return pd.DataFrame(rows), {
        "average_abs_rating_change_all_years": float(np.mean(changes)),
        "final_rating_mean": float(final_values.mean()),
        "final_rating_std": float(final_values.std()),
        "skipped_games": int(skipped),
    }


def select_best_parameters(results: pd.DataFrame) -> pd.Series:
    """Select without consulting held-out 2025 metrics."""

    required = ["validation_log_loss", "validation_brier_score", "K", "scale"]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError(f"Validation results are missing columns: {missing}")
    return results.sort_values(required, kind="mergesort").iloc[0]


def run_parameter_grid(
    matches: pd.DataFrame,
    k_values: Iterable[int] = K_VALUES,
    scale_values: Iterable[int] = SCALE_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen 28-pair grid."""

    result_rows: list[dict[str, Any]] = []
    predictions_by_pair: dict[tuple[int, int], pd.DataFrame] = {}
    for k_factor in k_values:
        for scale in scale_values:
            predictions, run_stats = run_candidate(matches, k_factor, scale)
            row: dict[str, Any] = {
                "K": int(k_factor),
                "scale": int(scale),
                "default_rating": DEFAULT_RATING,
                "start_year": START_YEAR,
                "end_year": END_YEAR,
                "validation_years": ",".join(map(str, VALIDATION_YEARS)),
                "test_years": ",".join(map(str, TEST_YEARS)),
                **run_stats,
            }
            row.update(_metrics(predictions, VALIDATION_YEARS, "validation"))
            row.update(_metrics(predictions, TEST_YEARS, "test"))
            result_rows.append(row)
            predictions_by_pair[(int(k_factor), int(scale))] = predictions

    results = pd.DataFrame(result_rows).sort_values(
        ["validation_log_loss", "validation_brier_score", "K", "scale"],
        kind="mergesort",
    ).reset_index(drop=True)
    results["rank_by_validation_log_loss"] = np.arange(1, len(results) + 1)
    best = select_best_parameters(results)
    if int(best["validation_games"]) != EXPECTED_VALIDATION_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_MATCHES} validation games, "
            f"found {int(best['validation_games'])}"
        )
    predictions = predictions_by_pair[(int(best["K"]), int(best["scale"]))]
    return results, predictions


def selected_model_table(results: pd.DataFrame) -> pd.DataFrame:
    best = select_best_parameters(results)
    columns = [
        "K",
        "scale",
        "validation_games",
        "validation_log_loss",
        "validation_brier_score",
        "validation_accuracy",
        "rank_by_validation_log_loss",
    ]
    return best[columns].to_frame().T


def run_pipeline(
    matches_path: str | Path = FULL_HISTORY_PATH,
    output_root: str | Path = "outputs/reproduction",
) -> dict[str, Path]:
    matches = load_matches(matches_path)
    results, predictions = run_parameter_grid(matches)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = ensure_directory(destination.resolve() / "elo_validation")
    paths = {
        "grid": destination / "elo_validation_grid.csv",
        "selected": destination / "elo_validation_selected_model.csv",
        "predictions": destination / "elo_validation_selected_predictions.csv",
    }
    results.to_csv(paths["grid"], index=False)
    selected_model_table(results).to_csv(paths["selected"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    return paths
