"""Chronological adaptive-K Elo candidates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.analysis.orientation import canonical_outcome, canonical_players
from code.config import EXPECTED_TEST_MATCHES
from code.io_utils import PROJECT_ROOT, ensure_directory
from code.models.elo import adaptive_k_previous_year, adaptive_k_total, expected_score
from code.pipelines.elo_pipeline import FULL_HISTORY_PATH, load_matches


EPSILON = 1e-15
RETAINED_MODEL = "AdaptiveK_PreviousYearGames_Elo_scale300"


@dataclass(frozen=True)
class AdaptiveCandidate:
    name: str
    rule: str
    scale: float


ADAPTIVE_CANDIDATES = (
    AdaptiveCandidate("AdaptiveK_TotalGames_Elo", "total_previous_games", 500.0),
    AdaptiveCandidate("AdaptiveK_PreviousYearGames_Elo", "previous_year_games", 500.0),
    AdaptiveCandidate("AdaptiveK_TotalGames_Elo_scale300", "total_previous_games", 300.0),
    AdaptiveCandidate(
        "AdaptiveK_PreviousYearGames_Elo_scale300",
        "previous_year_games",
        300.0,
    ),
)


def build_player_year_counts(matches: pd.DataFrame) -> dict[tuple[int, int], int]:
    appearances = pd.concat(
        [
            matches[["year", "winner"]].rename(columns={"winner": "player_id"}),
            matches[["year", "loser"]].rename(columns={"loser": "player_id"}),
        ],
        ignore_index=True,
    )
    counts = appearances.groupby(["player_id", "year"]).size()
    return {
        (int(player), int(year)): int(count)
        for (player, year), count in counts.items()
    }


def choose_k(candidate: AdaptiveCandidate, total_games: int, previous_year_games: int) -> float:
    if candidate.rule == "total_previous_games":
        return adaptive_k_total(total_games)
    if candidate.rule == "previous_year_games":
        return adaptive_k_previous_year(previous_year_games)
    raise ValueError(f"Unknown adaptive-K rule: {candidate.rule}")


def run_candidate(
    matches: pd.DataFrame,
    candidate: AdaptiveCandidate,
    player_year_counts: dict[tuple[int, int], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one adaptive candidate with pre-game activity counts."""

    yearly = player_year_counts or build_player_year_counts(matches)
    ratings: dict[int, float] = {}
    total_games: defaultdict[int, int] = defaultdict(int)
    predictions: list[dict[str, Any]] = []

    columns = ["fcode", "year", "event", "winner", "loser"]
    for row in matches[columns].itertuples(index=False):
        year = int(row.year)
        winner = int(row.winner)
        loser = int(row.loser)
        winner_total = total_games[winner]
        loser_total = total_games[loser]
        winner_previous_year = yearly.get((winner, year - 1), 0)
        loser_previous_year = yearly.get((loser, year - 1), 0)
        winner_k = choose_k(candidate, winner_total, winner_previous_year)
        loser_k = choose_k(candidate, loser_total, loser_previous_year)
        winner_rating = ratings.get(winner, 1500.0)
        loser_rating = ratings.get(loser, 1500.0)
        probability_winner = expected_score(winner_rating, loser_rating, candidate.scale)

        player_a, player_b = canonical_players(winner, loser)
        outcome_a = canonical_outcome(winner, player_a, player_b)
        rating_a = ratings.get(player_a, 1500.0)
        rating_b = ratings.get(player_b, 1500.0)
        probability_a = expected_score(rating_a, rating_b, candidate.scale)

        winner_change = winner_k * (1.0 - probability_winner)
        loser_change = -loser_k * (1.0 - probability_winner)
        ratings[winner] = winner_rating + winner_change
        ratings[loser] = loser_rating + loser_change

        if year == 2025:
            predictions.append(
                {
                    "model": candidate.name,
                    "rule": candidate.rule,
                    "scale": candidate.scale,
                    "year": year,
                    "fcode": int(row.fcode),
                    "event": int(row.event),
                    "winner": winner,
                    "loser": loser,
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": probability_a,
                    "pred_winner_win": probability_winner,
                    "player_a_rating_before": rating_a,
                    "player_b_rating_before": rating_b,
                    "winner_k": winner_k,
                    "loser_k": loser_k,
                    "winner_previous_total_games": winner_total,
                    "loser_previous_total_games": loser_total,
                    "winner_previous_year_games": winner_previous_year,
                    "loser_previous_year_games": loser_previous_year,
                    "winner_rating_change": winner_change,
                    "loser_rating_change": loser_change,
                }
            )

        total_games[winner] += 1
        total_games[loser] += 1

    final = pd.DataFrame(
        {
            "model": candidate.name,
            "player_id": list(sorted(ratings)),
            "rating": [ratings[player] for player in sorted(ratings)],
            "games_played": [total_games[player] for player in sorted(ratings)],
        }
    )
    return pd.DataFrame(predictions), final


def candidate_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if len(predictions) != EXPECTED_TEST_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_MATCHES} evaluation games, found {len(predictions)}"
        )
    outcomes = predictions["outcome_a"].to_numpy(dtype=float)
    probabilities = predictions["pred_a_win"].to_numpy(dtype=float)
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return {
        "evaluation_games": len(predictions),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
        "log_loss": float(
            -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
        ),
        "accuracy": float(np.mean((probabilities >= 0.5) == (outcomes == 1.0))),
        "mean_predicted_probability": float(probabilities.mean()),
    }


def run_all_candidates(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yearly = build_player_year_counts(matches)
    predictions: list[pd.DataFrame] = []
    ratings: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for candidate in ADAPTIVE_CANDIDATES:
        model_predictions, model_ratings = run_candidate(matches, candidate, yearly)
        metrics.append(
            {
                "model": candidate.name,
                "rule": candidate.rule,
                "scale": candidate.scale,
                "retained_exploratory_comparator": candidate.name == RETAINED_MODEL,
                **candidate_metrics(model_predictions),
            }
        )
        predictions.append(model_predictions)
        ratings.append(model_ratings)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(ratings, ignore_index=True),
        pd.DataFrame(metrics),
    )


def run_pipeline(
    matches_path: str | Path = FULL_HISTORY_PATH,
    output_root: str | Path = "outputs/reproduction",
) -> dict[str, Path]:
    matches = load_matches(matches_path)
    predictions, ratings, metrics = run_all_candidates(matches)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = ensure_directory(destination.resolve() / "adaptive_k")
    paths = {
        "metrics": destination / "adaptive_k_metrics_2025.csv",
        "predictions": destination / "adaptive_k_predictions_2025.csv",
        "ratings": destination / "adaptive_k_final_ratings.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    return paths
