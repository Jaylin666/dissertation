"""Common initial-rating translation sensitivity for low-inflation Glicko."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.analysis.orientation import canonical_outcome, canonical_players
from code.io_utils import PROJECT_ROOT, ensure_directory
from code.models.glicko import DEFAULT_RD, expected_score, update_two_players_single_game
from code.pipelines.glicko_pipeline import (
    build_variants,
    inflate_rd_for_inactivity,
    load_matches,
)


INITIAL_RATING_CANDIDATES = (1000, 1100, 1200, 1300, 1400, 1500)
REFERENCE_INITIAL_RATING = 1500
VALIDATION_YEARS = (2023, 2024)
EVALUATION_YEARS = (2023, 2024, 2025)
EPSILON = 1e-15


def run_candidate(
    matches: pd.DataFrame,
    common_initial_rating: float,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the full low-inflation history from one common origin."""

    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_period: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    columns = [
        "fcode",
        "year",
        "event",
        "inactivity_period_index",
        "winner",
        "loser",
    ]
    for row in matches[columns].itertuples(index=False):
        winner = int(row.winner)
        loser = int(row.loser)
        period = int(row.inactivity_period_index)
        for player in (winner, loser):
            if player not in ratings:
                ratings[player] = float(common_initial_rating)
                rds[player] = DEFAULT_RD
            else:
                elapsed = period - last_period[player]
                rds[player] = inflate_rd_for_inactivity(rds[player], elapsed, c_value)

        winner_rating = ratings[winner]
        winner_rd = rds[winner]
        loser_rating = ratings[loser]
        loser_rd = rds[loser]
        player_a, player_b = canonical_players(winner, loser)
        outcome_a = canonical_outcome(winner, player_a, player_b)
        if player_a == winner:
            rating_a, rd_a = winner_rating, winner_rd
            rating_b, rd_b = loser_rating, loser_rd
        else:
            rating_a, rd_a = loser_rating, loser_rd
            rating_b, rd_b = winner_rating, winner_rd
        probability_a = expected_score(rating_a, rating_b, rd_b)
        if int(row.year) in EVALUATION_YEARS:
            rows.append(
                {
                    "common_initial_rating": int(common_initial_rating),
                    "fcode": int(row.fcode),
                    "year": int(row.year),
                    "event": int(row.event),
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": probability_a,
                    "rating_a": rating_a,
                    "rating_b": rating_b,
                    "rd_a": rd_a,
                    "rd_b": rd_b,
                }
            )

        update = update_two_players_single_game(
            winner_rating,
            winner_rd,
            loser_rating,
            loser_rd,
            1.0,
        )
        ratings[winner] = update.player1_rating_after
        rds[winner] = update.player1_rd_after
        ratings[loser] = update.player2_rating_after
        rds[loser] = update.player2_rd_after
        last_period[winner] = max(last_period.get(winner, period), period)
        last_period[loser] = max(last_period.get(loser, period), period)

    final = pd.DataFrame(
        {
            "common_initial_rating": int(common_initial_rating),
            "player_id": list(sorted(ratings)),
            "rating": [ratings[player] for player in sorted(ratings)],
            "rd": [rds[player] for player in sorted(ratings)],
        }
    )
    return pd.DataFrame(rows), final


def validation_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    sample = predictions.loc[predictions["year"].isin(VALIDATION_YEARS)]
    probabilities = sample["pred_a_win"].to_numpy(dtype=float)
    outcomes = sample["outcome_a"].to_numpy(dtype=float)
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return {
        "validation_brier": float(np.mean((probabilities - outcomes) ** 2)),
        "validation_log_loss": float(
            -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
        ),
        "validation_accuracy": float(
            np.mean((probabilities >= 0.5) == (outcomes == 1.0))
        ),
    }


def build_invariance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions.loc[
        predictions["common_initial_rating"].eq(REFERENCE_INITIAL_RATING),
        ["fcode", "pred_a_win"],
    ].rename(columns={"pred_a_win": "reference_probability"})
    rows: list[dict[str, Any]] = []
    for candidate in INITIAL_RATING_CANDIDATES:
        sample = predictions.loc[predictions["common_initial_rating"].eq(candidate)]
        aligned = sample.merge(reference, on="fcode", validate="one_to_one")
        maximum = float(
            np.max(np.abs(aligned["pred_a_win"] - aligned["reference_probability"]))
        )
        rows.append(
            {
                "common_initial_rating": candidate,
                **validation_metrics(sample),
                "max_difference_across_candidates": maximum,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(
    matches_path: str | Path | None = None,
    output_root: str | Path = "outputs/reproduction",
) -> dict[str, Path]:
    matches, inactivity_unit, _ = load_matches(matches_path)
    low = next(
        variant for variant in build_variants(inactivity_unit) if variant["variant"] == "low_inflation"
    )
    prediction_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    for candidate in INITIAL_RATING_CANDIDATES:
        predictions, ratings = run_candidate(matches, candidate, float(low["c_value"]))
        prediction_frames.append(predictions)
        rating_frames.append(ratings)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_ratings = pd.concat(rating_frames, ignore_index=True)
    invariance = build_invariance_table(all_predictions)
    maximum = float(invariance["max_difference_across_candidates"].max())
    if maximum > 1e-12:
        raise ValueError("Common initial-rating candidates produced different probabilities")
    invariance["max_difference_across_candidates"] = 0.0

    destination = Path(output_root)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = ensure_directory(destination.resolve() / "initial_rating_sensitivity")
    paths = {
        "summary": destination / "initial_rating_invariance_core.csv",
        "predictions": destination / "initial_rating_predictions.csv",
        "ratings": destination / "initial_rating_final_ratings.csv",
    }
    invariance.to_csv(paths["summary"], index=False)
    all_predictions.to_csv(paths["predictions"], index=False)
    all_ratings.to_csv(paths["ratings"], index=False)
    return paths
