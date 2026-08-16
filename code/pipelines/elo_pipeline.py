"""Chronological Elo pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from code.analysis.orientation import canonical_outcome, canonical_players
from code.config import (
    ELO_CONFIGURATIONS,
    EXPECTED_FULL_HISTORY_MATCHES,
    EXPECTED_TEST_MATCHES,
    EXPECTED_UNIQUE_PLAYERS,
    FULL_HISTORY_END_YEAR,
    FULL_HISTORY_START_YEAR,
    EloConfig,
)
from code.io_utils import (
    PROJECT_ROOT,
    PUBLIC_MATCHES_PATH,
    add_event_ordering_columns,
    ensure_directory,
    read_csv_checked,
    stable_match_sort,
)
from code.models.elo import expected_score, update_with_config


FULL_HISTORY_PATH = PUBLIC_MATCHES_PATH
REQUIRED_MATCH_COLUMNS = ("fcode", "year", "event", "winner", "loser")
OPTIONAL_MATCH_COLUMNS = (
    "winner_name",
    "loser_name",
    "event_date_raw",
    "event_date_parsed",
)
EPSILON = 1e-15


def player_code(value: Any) -> int:
    return int(float(value))


def load_matches(path: str | Path = FULL_HISTORY_PATH) -> pd.DataFrame:
    """Load the checked historical games in frozen order."""

    matches = read_csv_checked(
        path,
        REQUIRED_MATCH_COLUMNS,
        low_memory=False,
    )
    for column in REQUIRED_MATCH_COLUMNS:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")
    if matches[list(REQUIRED_MATCH_COLUMNS)].isna().any().any():
        raise ValueError("Required match columns contain missing values")
    for column in OPTIONAL_MATCH_COLUMNS:
        if column not in matches.columns:
            matches[column] = pd.NA
    matches = matches[
        matches["year"].between(FULL_HISTORY_START_YEAR, FULL_HISTORY_END_YEAR)
    ].copy()
    matches = add_event_ordering_columns(matches)
    matches = stable_match_sort(matches)

    unique_players = pd.concat([matches["winner"], matches["loser"]]).nunique()
    test_matches = int((matches["year"] == FULL_HISTORY_END_YEAR).sum())
    if len(matches) != EXPECTED_FULL_HISTORY_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_FULL_HISTORY_MATCHES} matches, found {len(matches)}"
        )
    if int(unique_players) != EXPECTED_UNIQUE_PLAYERS:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_PLAYERS} players, found {int(unique_players)}"
        )
    if test_matches != EXPECTED_TEST_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_MATCHES} test matches, found {test_matches}"
        )
    return matches


def run_elo(
    matches: pd.DataFrame,
    config: EloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run one Elo configuration with prediction before each update."""

    started = time.perf_counter()
    ratings: dict[int, float] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    names: dict[int, str] = {}
    predictions: list[dict[str, Any]] = []

    columns = [
        "fcode",
        "year",
        "event",
        "winner",
        "loser",
        "winner_name",
        "loser_name",
    ]
    for row in matches[columns].itertuples(index=False):
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        winner_before = ratings.get(winner, config.initial_rating)
        loser_before = ratings.get(loser, config.initial_rating)

        player_a, player_b = canonical_players(winner, loser)
        outcome_a = canonical_outcome(winner, player_a, player_b)
        rating_a = ratings.get(player_a, config.initial_rating)
        rating_b = ratings.get(player_b, config.initial_rating)
        probability_a = expected_score(rating_a, rating_b, config.scale)

        update = update_with_config(winner_before, loser_before, config)
        ratings[winner] = update.winner_rating_after
        ratings[loser] = update.loser_rating_after
        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

        if winner not in names and pd.notna(row.winner_name):
            names[winner] = str(row.winner_name).strip()
        if loser not in names and pd.notna(row.loser_name):
            names[loser] = str(row.loser_name).strip()

        if int(row.year) == FULL_HISTORY_END_YEAR:
            predictions.append(
                {
                    "model": config.name,
                    "year": int(row.year),
                    "game_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "event": int(row.event),
                    "winner": winner,
                    "loser": loser,
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": probability_a,
                    "pred_winner_win": update.predicted_winner_win,
                    "winner_rating_before": winner_before,
                    "loser_rating_before": loser_before,
                    "rating_change": update.rating_change,
                }
            )

    rating_rows = [
        {
            "model": config.name,
            "player_id": player,
            "player_name": names.get(player, pd.NA),
            "rating": rating,
            "games_played": games_played[player],
            "wins": wins[player],
            "losses": losses[player],
        }
        for player, rating in sorted(ratings.items())
    ]
    final_ratings = pd.DataFrame(rating_rows)
    final_ratings["rank_by_rating"] = (
        final_ratings["rating"].rank(method="min", ascending=False).astype(int)
    )
    final_ratings = final_ratings.sort_values(
        ["model", "rank_by_rating", "player_id"]
    ).reset_index(drop=True)
    return pd.DataFrame(predictions), final_ratings, time.perf_counter() - started


def player_a_accuracy(predictions: pd.DataFrame) -> float:
    """Treat a probability of 0.5 as a Player A prediction."""

    probabilities = predictions["pred_a_win"].to_numpy(dtype=float)
    outcomes = predictions["outcome_a"].to_numpy(dtype=int)
    correct = (probabilities >= 0.5).astype(int) == outcomes
    return float(np.mean(correct))


def metric_row(predictions: pd.DataFrame, config: EloConfig, runtime: float) -> dict[str, Any]:
    """Calculate one model metric row."""

    if len(predictions) != EXPECTED_TEST_MATCHES:
        raise ValueError(
            f"{config.name} produced {len(predictions)} test rows; "
            f"expected {EXPECTED_TEST_MATCHES}"
        )
    probabilities = np.clip(
        predictions["pred_winner_win"].to_numpy(dtype=float),
        EPSILON,
        1.0 - EPSILON,
    )
    return {
        "model": config.name,
        "initial_rating": config.initial_rating,
        "k_factor": config.k_factor,
        "scale": config.scale,
        "evaluation_year": FULL_HISTORY_END_YEAR,
        "n_games": len(probabilities),
        "brier": float(np.mean((1.0 - probabilities) ** 2)),
        "log_loss": float(-np.mean(np.log(probabilities))),
        "accuracy": player_a_accuracy(predictions),
        "runtime_seconds": float(runtime),
    }


def run_configurations(
    matches: pd.DataFrame,
    configurations: Iterable[EloConfig] = ELO_CONFIGURATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen Elo configurations."""

    prediction_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for config in configurations:
        predictions, ratings, runtime = run_elo(matches, config)
        prediction_frames.append(predictions)
        rating_frames.append(ratings)
        metrics.append(metric_row(predictions, config, runtime))
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(rating_frames, ignore_index=True),
        pd.DataFrame(metrics),
    )


def write_outputs(
    predictions: pd.DataFrame,
    ratings: pd.DataFrame,
    metrics: pd.DataFrame,
    output_root: str | Path,
) -> dict[str, Path]:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    destination = ensure_directory(root.resolve() / "elo_pipeline")
    paths = {
        "predictions": destination / "elo_predictions_2025.csv",
        "ratings": destination / "elo_final_ratings.csv",
        "metrics": destination / "elo_metrics_2025.csv",
    }
    predictions.to_csv(paths["predictions"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    return paths


def run_pipeline(
    output_root: str | Path = "outputs/refactor_validation",
) -> dict[str, Path]:
    """Run the Elo pipeline and write its outputs."""

    matches = load_matches()
    predictions, ratings, metrics = run_configurations(matches)
    return write_outputs(predictions, ratings, metrics, output_root)
