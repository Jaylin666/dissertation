"""Leakage-free 2025 features and model comparison inputs."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.analysis.orientation import direct_player_a_glicko_probability
from code.config import EXPECTED_TEST_MATCHES
from code.io_utils import PROJECT_ROOT, ensure_directory, require_columns
from code.models.glicko import expected_score as glicko_expected_score
from code.pipelines.adaptive_k_pipeline import RETAINED_MODEL
from code.pipelines.elo_pipeline import FULL_HISTORY_PATH, load_matches


ELO_MODELS = (
    "Conservative_Elo",
    "Default_Elo",
    "Validation_best_Elo",
)
GLICKO_VARIANTS = {
    "C0_no_inflation": "Glicko_C0",
    "low_inflation": "Glicko_low",
}


def date_quality(method: Any) -> str:
    if method == "parsed_full_date":
        return "exact"
    if method == "month_year_imputed":
        return "project_fallback"
    return "missing"


def _initial_state() -> dict[str, Any]:
    return {
        "total_games": 0,
        "first_date": None,
        "last_date": None,
        "recent_dates": deque(),
        "games_by_year": defaultdict(int),
        "missing_dated_games": 0,
    }


def _recent_count(dates: deque[pd.Timestamp], current: pd.Timestamp, days: int) -> int:
    lower = current - pd.Timedelta(days=days)
    return int(sum(date >= lower for date in dates))


def _player_features(
    state: dict[str, Any],
    player: int,
    opponent: int,
    side: str,
    row: Any,
) -> dict[str, Any]:
    current = pd.to_datetime(row.event_order_date, errors="coerce")
    has_date = pd.notna(current)
    previous = int(state["total_games"])
    has_history = previous > 0
    available = bool(
        has_date
        and state["missing_dated_games"] == 0
        and (not has_history or state["last_date"] is not None)
    )
    if available:
        games_90 = _recent_count(state["recent_dates"], current, 90)
        games_365 = _recent_count(state["recent_dates"], current, 365)
        if has_history:
            days_since = int((current - state["last_date"]).days)
            career_days = int((current - state["first_date"]).days)
        else:
            days_since = np.nan
            career_days = np.nan
    else:
        games_90 = np.nan
        games_365 = np.nan
        days_since = np.nan
        career_days = np.nan
    quality = date_quality(row.event_date_ordering_method)
    return {
        "match_id": int(row.fcode),
        "match_sequence": int(row.match_sequence),
        "year": int(row.year),
        "event_id": int(row.event),
        "match_date": current.date().isoformat() if has_date else pd.NA,
        "date_quality": quality,
        "elapsed_time_quality": quality if available else "missing",
        "player_id": player,
        "opponent_id": opponent,
        "player_side": side,
        "total_games_before": previous,
        "recorded_appearance_number": previous + 1,
        "games_last_90_days": games_90,
        "games_last_365_days": games_365,
        "games_previous_calendar_year": int(state["games_by_year"].get(int(row.year) - 1, 0)),
        "days_since_last_game": days_since,
        "career_days_before": career_days,
        "is_debut": previous == 0,
        "has_previous_history": has_history,
        "date_features_available": available,
    }


def _update_state(state: dict[str, Any], year: int, event_order_date: Any) -> None:
    state["total_games"] += 1
    state["games_by_year"][year] += 1
    current = pd.to_datetime(event_order_date, errors="coerce")
    if pd.isna(current):
        state["missing_dated_games"] += 1
        return
    if state["first_date"] is None:
        state["first_date"] = current
    state["last_date"] = current
    state["recent_dates"].append(current)
    cutoff = current - pd.Timedelta(days=365)
    while state["recent_dates"] and state["recent_dates"][0] < cutoff:
        state["recent_dates"].popleft()


def build_prematch_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two player rows and one game row for every 2025 game."""

    ordered = matches.copy().reset_index(drop=True)
    ordered["match_sequence"] = np.arange(1, len(ordered) + 1, dtype=int)
    ordered["event_order_date"] = pd.to_datetime(ordered["event_order_date"], errors="coerce")
    states: defaultdict[int, dict[str, Any]] = defaultdict(_initial_state)
    player_rows: list[dict[str, Any]] = []

    columns = [
        "fcode",
        "match_sequence",
        "year",
        "event",
        "event_order_date",
        "event_date_ordering_method",
        "winner",
        "loser",
    ]
    for row in ordered[columns].itertuples(index=False):
        winner = int(row.winner)
        loser = int(row.loser)
        player_a = min(winner, loser)
        player_b = max(winner, loser)
        if int(row.year) == 2025:
            player_rows.append(_player_features(states[player_a], player_a, player_b, "A", row))
            player_rows.append(_player_features(states[player_b], player_b, player_a, "B", row))
        _update_state(states[winner], int(row.year), row.event_order_date)
        _update_state(states[loser], int(row.year), row.event_order_date)

    long = pd.DataFrame(player_rows)
    games_2025 = ordered.loc[ordered["year"].eq(2025)].copy()
    a_rows = long.loc[long["player_side"].eq("A")].set_index("match_id")
    b_rows = long.loc[long["player_side"].eq("B")].set_index("match_id")
    game_rows: list[dict[str, Any]] = []
    numeric_features = (
        ("total_games_before", "total_games_before"),
        ("games_last_90_days", "games_last_90_days"),
        ("games_last_365_days", "games_last_365_days"),
        ("previous_year_games", "games_previous_calendar_year"),
    )

    for match in games_2025.itertuples(index=False):
        match_id = int(match.fcode)
        a = a_rows.loc[match_id]
        b = b_rows.loc[match_id]
        winner = int(match.winner)
        row: dict[str, Any] = {
            "match_id": match_id,
            "fcode": match_id,
            "match_sequence": int(match.match_sequence),
            "year": int(match.year),
            "event_id": int(match.event),
            "event_key": f"{int(match.year)}_{int(match.event)}",
            "match_date": a["match_date"],
            "player_a_id": int(a["player_id"]),
            "player_b_id": int(b["player_id"]),
            "winner_id": winner,
            "loser_id": int(match.loser),
            "outcome_a": int(int(a["player_id"]) == winner),
        }
        for side, values in (("a", a), ("b", b)):
            for column in (
                "total_games_before",
                "recorded_appearance_number",
                "games_last_90_days",
                "games_last_365_days",
                "games_previous_calendar_year",
                "days_since_last_game",
                "career_days_before",
                "is_debut",
                "has_previous_history",
                "date_features_available",
                "date_quality",
                "elapsed_time_quality",
            ):
                row[f"{side}_{column}"] = values[column]
        for output_name, source_name in numeric_features:
            value_a = row[f"a_{source_name}"]
            value_b = row[f"b_{source_name}"]
            if pd.isna(value_a) or pd.isna(value_b):
                minimum = maximum = difference = np.nan
            else:
                minimum = float(min(value_a, value_b))
                maximum = float(max(value_a, value_b))
                difference = float(abs(value_a - value_b))
            row[f"min_{output_name}"] = minimum
            row[f"max_{output_name}"] = maximum
            row[f"abs_diff_{output_name}"] = difference
        row["either_player_debut"] = bool(row["a_is_debut"] or row["b_is_debut"])
        row["both_players_have_history"] = bool(
            row["a_has_previous_history"] and row["b_has_previous_history"]
        )
        if row["either_player_debut"]:
            row["min_days_since_last_game"] = np.nan
            row["max_days_since_last_game"] = np.nan
            row["either_player_inactive_365d"] = False
            row["either_player_inactive_730d"] = False
        else:
            days_a = row["a_days_since_last_game"]
            days_b = row["b_days_since_last_game"]
            if pd.isna(days_a) or pd.isna(days_b):
                row["min_days_since_last_game"] = np.nan
                row["max_days_since_last_game"] = np.nan
            else:
                row["min_days_since_last_game"] = float(min(days_a, days_b))
                row["max_days_since_last_game"] = float(max(days_a, days_b))
            row["either_player_inactive_365d"] = bool(
                (pd.notna(days_a) and days_a >= 365) or (pd.notna(days_b) and days_b >= 365)
            )
            row["either_player_inactive_730d"] = bool(
                (pd.notna(days_a) and days_a >= 730) or (pd.notna(days_b) and days_b >= 730)
            )
        row["both_players_active_last_365d"] = bool(
            pd.notna(row["a_games_last_365_days"])
            and pd.notna(row["b_games_last_365_days"])
            and row["a_games_last_365_days"] > 0
            and row["b_games_last_365_days"] > 0
        )
        row["either_player_low_recent_activity"] = bool(
            pd.notna(row["min_games_last_365_days"])
            and row["min_games_last_365_days"] <= 5
        )
        game_rows.append(row)
    return long, pd.DataFrame(game_rows)


def _glicko_table(predictions: pd.DataFrame, variant: str, prefix: str) -> pd.DataFrame:
    sample = predictions.loc[predictions["variant"].eq(variant)].copy()
    if len(sample) != EXPECTED_TEST_MATCHES:
        raise ValueError(f"{variant} has {len(sample)} rows; expected {EXPECTED_TEST_MATCHES}")
    a_is_winner = sample["winner"].astype(int) < sample["loser"].astype(int)
    sample[f"rating_a_{prefix}"] = np.where(
        a_is_winner, sample["pre_rating_winner"], sample["pre_rating_loser"]
    )
    sample[f"rating_b_{prefix}"] = np.where(
        a_is_winner, sample["pre_rating_loser"], sample["pre_rating_winner"]
    )
    sample[f"rd_a_{prefix}"] = np.where(
        a_is_winner, sample["pre_rd_winner"], sample["pre_rd_loser"]
    )
    sample[f"rd_b_{prefix}"] = np.where(
        a_is_winner, sample["pre_rd_loser"], sample["pre_rd_winner"]
    )
    sample[f"p_a_{prefix}_fixed"] = [
        direct_player_a_glicko_probability(rating_a, rating_b, rd_b)
        for rating_a, rating_b, rd_b in zip(
            sample[f"rating_a_{prefix}"],
            sample[f"rating_b_{prefix}"],
            sample[f"rd_b_{prefix}"],
        )
    ]
    sample[f"p_b_{prefix}_direct"] = [
        glicko_expected_score(rating_b, rating_a, rd_a)
        for rating_a, rating_b, rd_a in zip(
            sample[f"rating_a_{prefix}"],
            sample[f"rating_b_{prefix}"],
            sample[f"rd_a_{prefix}"],
        )
    ]
    sample[f"{prefix}_complement_gap"] = (
        sample[f"p_a_{prefix}_fixed"] + sample[f"p_b_{prefix}_direct"] - 1.0
    )
    columns = [
        "fcode",
        f"rating_a_{prefix}",
        f"rating_b_{prefix}",
        f"rd_a_{prefix}",
        f"rd_b_{prefix}",
        f"p_a_{prefix}_fixed",
        f"p_b_{prefix}_direct",
        f"{prefix}_complement_gap",
    ]
    return sample[columns]


def build_unified_comparison(
    features: pd.DataFrame,
    elo_predictions: pd.DataFrame,
    glicko_predictions: pd.DataFrame,
    adaptive_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge canonical features and prematch model states by game ID."""

    elo = elo_predictions.loc[elo_predictions["model"].isin(ELO_MODELS)].pivot(
        index="fcode", columns="model", values="pred_a_win"
    )
    elo = elo.rename(columns={model: f"p_a_{model}" for model in ELO_MODELS}).reset_index()
    adaptive = adaptive_predictions.loc[
        adaptive_predictions["model"].eq(RETAINED_MODEL), ["fcode", "pred_a_win"]
    ].rename(columns={"pred_a_win": "p_a_best_AdaptiveK"})
    result = features.merge(elo, left_on="match_id", right_on="fcode", validate="one_to_one")
    result = result.drop(columns=["fcode_y"]).rename(columns={"fcode_x": "fcode"})
    result = result.merge(adaptive, on="fcode", validate="one_to_one")
    for variant, prefix in GLICKO_VARIANTS.items():
        result = result.merge(
            _glicko_table(glicko_predictions, variant, prefix),
            on="fcode",
            validate="one_to_one",
        )
    result["p_a_Glicko_C0_fixed"] = result["p_a_Glicko_C0_fixed"]
    result["p_a_Glicko_low_fixed"] = result["p_a_Glicko_low_fixed"]
    result["max_prematch_rd"] = result[["rd_a_Glicko_low", "rd_b_Glicko_low"]].max(axis=1)
    result["min_prematch_rd"] = result[["rd_a_Glicko_low", "rd_b_Glicko_low"]].min(axis=1)
    result["mean_prematch_rd"] = result[["rd_a_Glicko_low", "rd_b_Glicko_low"]].mean(axis=1)
    return result.sort_values("match_sequence").reset_index(drop=True)


def validate_comparison_input(table: pd.DataFrame) -> None:
    probability_columns = [
        "p_a_Conservative_Elo",
        "p_a_Default_Elo",
        "p_a_Validation_best_Elo",
        "p_a_best_AdaptiveK",
        "p_a_Glicko_C0_fixed",
        "p_a_Glicko_low_fixed",
    ]
    require_columns(
        table,
        [
            "match_id",
            "match_sequence",
            "player_a_id",
            "player_b_id",
            "winner_id",
            "loser_id",
            "outcome_a",
            "a_total_games_before",
            "b_total_games_before",
            *probability_columns,
        ],
        "comparison input",
    )
    if len(table) != EXPECTED_TEST_MATCHES:
        raise ValueError(f"Expected {EXPECTED_TEST_MATCHES} rows, found {len(table)}")
    if table["match_id"].duplicated().any():
        raise ValueError("Comparison input contains duplicate game IDs")
    if not (table["player_a_id"] < table["player_b_id"]).all():
        raise ValueError("Player A must be the smaller database player ID")
    expected_outcome = table["player_a_id"].eq(table["winner_id"]).astype(int)
    if not table["outcome_a"].astype(int).equals(expected_outcome):
        raise ValueError("Player A outcomes do not match the canonical player IDs")
    if not table[probability_columns].apply(lambda values: values.between(0.0, 1.0).all()).all():
        raise ValueError("Comparison probabilities must lie in [0, 1]")


def run_pipeline(
    matches_path: str | Path = FULL_HISTORY_PATH,
    output_root: str | Path = "outputs/reproduction",
    elo_predictions_path: str | Path | None = None,
    glicko_predictions_path: str | Path | None = None,
    adaptive_predictions_path: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    elo_path = Path(elo_predictions_path) if elo_predictions_path else root / "elo_pipeline" / "elo_predictions_2025.csv"
    glicko_path = Path(glicko_predictions_path) if glicko_predictions_path else root / "glicko_inflation" / "glicko_inflation_predictions_2025.csv"
    adaptive_path = Path(adaptive_predictions_path) if adaptive_predictions_path else root / "adaptive_k" / "adaptive_k_predictions_2025.csv"
    matches = load_matches(matches_path)
    long, features = build_prematch_features(matches)
    elo_predictions = pd.read_csv(elo_path, low_memory=False)
    glicko_predictions = pd.read_csv(glicko_path, low_memory=False)
    adaptive_predictions = pd.read_csv(adaptive_path, low_memory=False)
    comparison = build_unified_comparison(
        features, elo_predictions, glicko_predictions, adaptive_predictions
    )
    validate_comparison_input(comparison)
    destination = ensure_directory(root / "comparison_inputs")
    paths = {
        "player_features": destination / "prematch_player_features_2025.csv",
        "match_features": destination / "prematch_match_features_2025.csv",
        "comparison": destination / "comparison_input_2025.csv",
    }
    long.to_csv(paths["player_features"], index=False)
    features.to_csv(paths["match_features"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    return paths
