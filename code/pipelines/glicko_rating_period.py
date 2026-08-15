"""Glicko-1 rating-period sensitivity with no inactivity inflation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from code.analysis.orientation import canonical_outcome, canonical_players
from code.config import EXPECTED_TEST_MATCHES
from code.io_utils import PROJECT_ROOT, ensure_directory
from code.models.glicko import (
    DEFAULT_RATING,
    DEFAULT_RD,
    expected_score,
    update_player_glicko,
    update_two_players_single_game,
)
from code.pipelines.elo_pipeline import FULL_HISTORY_PATH, load_matches


EPSILON = 1e-15
PERIOD_SETTINGS = (
    ("glicko1_period_match_c0", "match_by_match"),
    ("glicko1_period_event_c0", "event_level"),
    ("glicko1_period_monthly_c0", "monthly"),
    ("glicko1_period_yearly_c0", "yearly"),
)


def assign_periods(matches: pd.DataFrame, period_type: str) -> pd.Series:
    """Return chronological period identifiers."""

    if period_type == "match_by_match":
        return pd.Series(
            [f"match_{index}_{int(fcode)}" for index, fcode in enumerate(matches["fcode"])],
            index=matches.index,
            dtype="string",
        )
    if period_type == "event_level":
        return (
            "event_"
            + matches["year"].astype(int).astype(str)
            + "_"
            + matches["event"].astype(int).astype(str)
        )
    if period_type == "monthly":
        dates = pd.to_datetime(matches["event_order_date"], errors="coerce")
        month = dates.dt.strftime("%Y_%m")
        fallback = matches["year"].astype(int).astype(str) + "_unknown"
        return "month_" + month.fillna(fallback)
    if period_type == "yearly":
        return "year_" + matches["year"].astype(int).astype(str)
    raise ValueError(f"Unknown rating period: {period_type}")


def _state(
    ratings: dict[int, float],
    rds: dict[int, float],
    player: int,
) -> tuple[float, float]:
    if player not in ratings:
        ratings[player] = DEFAULT_RATING
        rds[player] = DEFAULT_RD
    return ratings[player], rds[player]


def run_rating_period(
    matches: pd.DataFrame,
    period_type: str,
    setting_name: str | None = None,
    prediction_years: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one C=0 rating-period definition."""

    sample = matches.copy()
    sample["rating_period_id"] = assign_periods(sample, period_type)
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    predictions: list[dict[str, Any]] = []
    setting = setting_name or f"glicko1_period_{period_type}_c0"
    saved_years = set(prediction_years) if prediction_years is not None else None

    if period_type == "match_by_match":
        for index, row in enumerate(sample.itertuples(index=False)):
            winner = int(row.winner)
            loser = int(row.loser)
            winner_rating, winner_rd = _state(ratings, rds, winner)
            loser_rating, loser_rd = _state(ratings, rds, loser)
            player_a, player_b = canonical_players(winner, loser)
            outcome_a = canonical_outcome(winner, player_a, player_b)
            if player_a == winner:
                rating_a, rd_a = winner_rating, winner_rd
                rating_b, rd_b = loser_rating, loser_rd
            else:
                rating_a, rd_a = loser_rating, loser_rd
                rating_b, rd_b = winner_rating, winner_rd
            if saved_years is None or int(row.year) in saved_years:
                predictions.append(
                    {
                    "setting_name": setting,
                    "period_type": period_type,
                    "rating_period_id": f"match_{index}_{int(row.fcode)}",
                    "fcode": int(row.fcode),
                    "year": int(row.year),
                    "event": int(row.event),
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": expected_score(rating_a, rating_b, rd_b),
                    "player_a_rating_before_period": rating_a,
                    "player_a_rd_before_period": rd_a,
                    "player_b_rating_before_period": rating_b,
                    "player_b_rd_before_period": rd_b,
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
        final = pd.DataFrame(
            {
                "setting_name": setting,
                "period_type": period_type,
                "player_id": list(sorted(ratings)),
                "rating": [ratings[player] for player in sorted(ratings)],
                "rd": [rds[player] for player in sorted(ratings)],
            }
        )
        return pd.DataFrame(predictions), final

    for period_id, period in sample.groupby("rating_period_id", sort=False):
        opening_ratings: dict[int, float] = {}
        opening_rds: dict[int, float] = {}
        games: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"opponent_ratings": [], "opponent_rds": [], "scores": []}
        )

        def opening_state(player: int) -> tuple[float, float]:
            if player not in opening_ratings:
                rating, rd = _state(ratings, rds, player)
                opening_ratings[player] = rating
                opening_rds[player] = rd
            return opening_ratings[player], opening_rds[player]

        for row in period.itertuples(index=False):
            winner = int(row.winner)
            loser = int(row.loser)
            winner_rating, winner_rd = opening_state(winner)
            loser_rating, loser_rd = opening_state(loser)
            player_a, player_b = canonical_players(winner, loser)
            outcome_a = canonical_outcome(winner, player_a, player_b)
            rating_a, rd_a = opening_state(player_a)
            rating_b, rd_b = opening_state(player_b)
            probability_a = expected_score(rating_a, rating_b, rd_b)

            games[winner]["opponent_ratings"].append(loser_rating)
            games[winner]["opponent_rds"].append(loser_rd)
            games[winner]["scores"].append(1.0)
            games[loser]["opponent_ratings"].append(winner_rating)
            games[loser]["opponent_rds"].append(winner_rd)
            games[loser]["scores"].append(0.0)

            if saved_years is None or int(row.year) in saved_years:
                predictions.append(
                    {
                    "setting_name": setting,
                    "period_type": period_type,
                    "rating_period_id": period_id,
                    "fcode": int(row.fcode),
                    "year": int(row.year),
                    "event": int(row.event),
                    "player_a": player_a,
                    "player_b": player_b,
                    "outcome_a": outcome_a,
                    "pred_a_win": probability_a,
                    "player_a_rating_before_period": rating_a,
                    "player_a_rd_before_period": rd_a,
                    "player_b_rating_before_period": rating_b,
                    "player_b_rd_before_period": rd_b,
                    }
                )

        new_states: dict[int, tuple[float, float]] = {}
        for player, values in games.items():
            new_states[player] = update_player_glicko(
                opening_ratings[player],
                opening_rds[player],
                values["opponent_ratings"],
                values["opponent_rds"],
                values["scores"],
            )
        for player, (rating, rd) in new_states.items():
            ratings[player] = rating
            rds[player] = rd

    final = pd.DataFrame(
        {
            "setting_name": setting,
            "period_type": period_type,
            "player_id": list(sorted(ratings)),
            "rating": [ratings[player] for player in sorted(ratings)],
            "rd": [rds[player] for player in sorted(ratings)],
        }
    )
    return pd.DataFrame(predictions), final


def _fcode_hash(values: Iterable[int]) -> str:
    text = ",".join(str(int(value)) for value in values)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    evaluation = predictions.loc[predictions["year"].eq(2025)]
    if len(evaluation) != EXPECTED_TEST_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_MATCHES} evaluation games, found {len(evaluation)}"
        )
    outcomes = evaluation["outcome_a"].to_numpy(dtype=float)
    probabilities = evaluation["pred_a_win"].to_numpy(dtype=float)
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    observed = float(outcomes.mean())
    return {
        "start_year": 1985,
        "end_year": 2025,
        "evaluation_year": 2025,
        "evaluation_games": len(evaluation),
        "log_loss": float(
            -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
        ),
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "accuracy": float(np.mean((probabilities >= 0.5) == (outcomes == 1.0))),
        "baseline_accuracy": max(observed, 1.0 - observed),
        "mean_predicted_probability": float(probabilities.mean()),
        "observed_win_rate": observed,
        "pred_a_win_min": float(probabilities.min()),
        "pred_a_win_max": float(probabilities.max()),
        "pred_a_win_out_of_range_count": int(
            ((probabilities < 0.0) | (probabilities > 1.0)).sum()
        ),
        "evaluation_fcode_hash": _fcode_hash(evaluation["fcode"]),
    }


def run_all_periods(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    reference_ids: list[int] | None = None
    for setting_name, period_type in PERIOD_SETTINGS:
        predictions, ratings = run_rating_period(
            matches,
            period_type,
            setting_name,
            prediction_years=(2025,),
        )
        metrics = prediction_metrics(predictions)
        ids = predictions.loc[predictions["year"].eq(2025), "fcode"].astype(int).tolist()
        if reference_ids is None:
            reference_ids = ids
        metrics.update(
            {
                "setting_name": setting_name,
                "period_type": period_type,
                "evaluation_fcode_set_matches_reference": set(ids) == set(reference_ids),
                "evaluation_fcode_order_matches_reference": ids == reference_ids,
            }
        )
        prediction_frames.append(predictions.loc[predictions["year"].eq(2025)])
        rating_frames.append(ratings)
        metric_rows.append(metrics)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(rating_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def run_pipeline(
    matches_path: str | Path = FULL_HISTORY_PATH,
    output_root: str | Path = "outputs/reproduction",
) -> dict[str, Path]:
    matches = load_matches(matches_path)
    predictions, ratings, metrics = run_all_periods(matches)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = ensure_directory(destination.resolve() / "glicko_rating_period")
    paths = {
        "metrics": destination / "glicko_rating_period_metrics.csv",
        "predictions": destination / "glicko_rating_period_predictions_2025.csv",
        "ratings": destination / "glicko_rating_period_final_ratings.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    return paths
