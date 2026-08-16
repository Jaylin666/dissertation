"""This script compares different Glicko-1 rating-period assumptions for croquet data: match-by-match, event-level, monthly, and yearly.

The experiment isolates the rating-period assumption. All settings use the
same Glicko-1 formula, same full-history croquet dataset, same initial
rating/RD, and C=0. Inactivity RD inflation is intentionally not included here;
it should be tested separately after the rating-period diagnostic.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from glicko_core import (  # noqa: E402
    DEFAULT_RATING as CORE_DEFAULT_RATING,
    DEFAULT_RD as CORE_DEFAULT_RD,
    MAX_RD as CORE_MAX_RD,
    MIN_RD as CORE_MIN_RD,
    expected_score,
    update_player_glicko,
    update_two_players_single_game,
)


DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
MIN_RD = 30.0
MAX_RD = 350.0
C = 0.0
START_YEAR = 1985
END_YEAR = 2025
EXPECTED_2025_GAMES = 11_379
EPS = 1e-15

PERIOD_SETTINGS = [
    {
        "setting_name": "glicko1_period_match_c0",
        "period_type": "match_by_match",
        "label": "Match-by-match",
    },
    {
        "setting_name": "glicko1_period_event_c0",
        "period_type": "event_level",
        "label": "Event-level",
    },
    {
        "setting_name": "glicko1_period_monthly_c0",
        "period_type": "monthly",
        "label": "Monthly",
    },
    {
        "setting_name": "glicko1_period_yearly_c0",
        "period_type": "yearly",
        "label": "Yearly",
    },
]

MATCHES_PATH = PROJECT_ROOT / "outputs" / "elo_optimization" / "matches_1985_2025_checked.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "glicko_implementation"

PREDICTIONS_PATH = OUTPUT_DIR / "glicko_rating_period_predictions_1985_2025.csv"
METRICS_PATH = OUTPUT_DIR / "glicko_rating_period_metrics_2025.csv"
CALIBRATION_PATH = OUTPUT_DIR / "glicko_rating_period_calibration_2025.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "glicko_rating_period_confidence_2025.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "glicko_rating_period_final_ratings.csv"
RD_SUMMARY_PATH = OUTPUT_DIR / "glicko_rating_period_rd_summary.csv"
YEARLY_RD_SUMMARY_PATH = OUTPUT_DIR / "glicko_rating_period_yearly_rd_summary.csv"
PERIOD_SUMMARY_PATH = OUTPUT_DIR / "glicko_rating_period_period_summary.csv"
LIST_SIMILARITY_PATH = OUTPUT_DIR / "glicko_rating_period_list_similarity.csv"
DATE_ORDERING_SUMMARY_PATH = OUTPUT_DIR / "glicko_rating_period_date_ordering_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "glicko_rating_period_sensitivity_summary.md"

REQUIRED_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

PREDICTION_COLUMNS = [
    "setting_name",
    "period_type",
    "rating_period_id",
    "rating_period_start_order",
    "start_year",
    "end_year",
    "fcode",
    "code",
    "year",
    "event",
    "event_key",
    "eventname",
    "event_date_raw",
    "event_date_parsed",
    "event_order_date",
    "event_date_ordering_method",
    "winner",
    "loser",
    "player_a",
    "player_b",
    "actual_a_win",
    "pred_a_win",
    "pred_winner_win",
    "winner_rating_before_period",
    "winner_rd_before_period",
    "loser_rating_before_period",
    "loser_rd_before_period",
    "player_a_rating_before_period",
    "player_a_rd_before_period",
    "player_b_rating_before_period",
    "player_b_rd_before_period",
    "winner_rating_after_period",
    "winner_rd_after_period",
    "loser_rating_after_period",
    "loser_rd_after_period",
    "player_a_rating_after_period",
    "player_a_rd_after_period",
    "player_b_rating_after_period",
    "player_b_rd_after_period",
]

ACTIVE_PLAYER_SUBSETS = [
    ("all_common_players", 0),
    ("active_2025_games_ge1", 1),
    ("active_2025_games_ge5", 5),
    ("active_2025_games_ge10", 10),
]


def format_code_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        value_float = float(value)
        if value_float.is_integer():
            return str(int(value_float))
    except (TypeError, ValueError):
        pass
    return str(value)


def player_code(value: Any) -> int:
    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    if code in player_names or pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date and event_date_ordering_method without modifying raw dates."""

    matches = matches.copy()
    if "event_order_date" in matches.columns and "event_date_ordering_method" in matches.columns:
        matches["event_order_date"] = pd.to_datetime(matches["event_order_date"], errors="coerce")
        return matches

    parsed = pd.to_datetime(matches["event_date_parsed"], errors="coerce")
    matches["event_order_date"] = parsed
    matches["event_date_ordering_method"] = np.where(
        parsed.notna(),
        "parsed_full_date",
        "fallback_no_date",
    )

    missing_parsed = parsed.isna()
    raw = matches.loc[missing_parsed, "event_date_raw"].astype("string").str.strip()
    extracted = raw.str.extract(r"^(?P<month>\d{1,2})\.(?P<year>\d{2}|\d{4})$")

    valid_month_year = extracted["month"].notna()
    if valid_month_year.any():
        months = pd.to_numeric(extracted.loc[valid_month_year, "month"], errors="coerce")
        raw_years = extracted.loc[valid_month_year, "year"].astype(str)
        years = raw_years.astype(int)
        years = np.where(
            raw_years.str.len().eq(2),
            np.where(years >= 85, 1900 + years, 2000 + years),
            years,
        )

        valid_month = months.between(1, 12).fillna(False)
        valid_month_mask = valid_month.to_numpy(dtype=bool)
        valid_index = extracted.loc[valid_month_year].index[valid_month_mask]
        imputed_dates = pd.to_datetime(
            {
                "year": np.asarray(years)[valid_month_mask],
                "month": months.loc[valid_index].astype(int).to_numpy(),
                "day": np.repeat(15, len(valid_index)),
            },
            errors="coerce",
        )

        matches.loc[valid_index, "event_order_date"] = imputed_dates.to_numpy()
        matches.loc[valid_index, "event_date_ordering_method"] = "month_year_imputed"

    return matches


def load_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MATCHES_PATH}. Run code/13_build_full_history_match_dataset.py first."
        )

    matches = pd.read_csv(MATCHES_PATH, low_memory=False)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{MATCHES_PATH.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    missing_ids = matches[REQUIRED_COLUMNS].isna().sum()
    if int(missing_ids.sum()) > 0:
        raise ValueError(f"Required ID columns contain missing values:\n{missing_ids}")

    matches = add_event_ordering_columns(matches)
    matches["event_order_date_missing"] = matches["event_order_date"].isna()
    sort_cols = ["year", "event_order_date_missing", "event_order_date", "event", "code", "fcode"]
    matches = (
        matches.sort_values(sort_cols, na_position="last")
        .drop(columns=["event_order_date_missing"])
        .reset_index(drop=True)
    )
    matches["chronological_order"] = np.arange(len(matches), dtype=np.int64)
    matches["event_key"] = (
        matches["year"].astype(int).astype(str) + "_" + matches["event"].astype(int).astype(str)
    )

    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Loaded dataset: {MATCHES_PATH}")
    print(f"Matches: {len(matches):,}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    print(f"Players: {players:,}")
    return matches


def make_date_ordering_summary(matches: pd.DataFrame) -> pd.DataFrame:
    base = (
        matches.groupby("event_date_ordering_method", dropna=False)
        .size()
        .reset_index(name="match_count")
        .sort_values("event_date_ordering_method")
        .reset_index(drop=True)
    )
    base["share_of_matches"] = base["match_count"] / len(matches)

    rows = []
    for setting in PERIOD_SETTINGS:
        for row in base.itertuples(index=False):
            rows.append(
                {
                    "setting_name": setting["setting_name"],
                    "period_type": setting["period_type"],
                    "start_year": START_YEAR,
                    "end_year": END_YEAR,
                    "event_date_ordering_method": row.event_date_ordering_method,
                    "match_count": int(row.match_count),
                    "share_of_matches": float(row.share_of_matches),
                }
            )
    return pd.DataFrame(rows)


def assign_rating_period_id(matches: pd.DataFrame, period_type: str) -> pd.DataFrame:
    """Assign a Glicko rating-period id according to the requested assumption."""

    matches = matches.copy()
    if period_type == "match_by_match":
        if matches["fcode"].duplicated().any():
            matches["rating_period_id"] = (
                "match_"
                + matches["chronological_order"].astype(str)
                + "_"
                + matches["fcode"].astype(int).astype(str)
            )
        else:
            matches["rating_period_id"] = "match_" + matches["fcode"].astype(int).astype(str)
    elif period_type == "event_level":
        matches["rating_period_id"] = "event_" + matches["event_key"].astype(str)
    elif period_type == "monthly":
        event_order_date = pd.to_datetime(matches["event_order_date"], errors="coerce")
        month_id = event_order_date.dt.strftime("%Y_%m")
        fallback = matches["year"].astype(int).astype(str) + "_unknown"
        matches["rating_period_id"] = "month_" + month_id.fillna(fallback)
    elif period_type == "yearly":
        matches["rating_period_id"] = "year_" + matches["year"].astype(int).astype(str)
    else:
        raise ValueError(f"Unknown period_type: {period_type}")

    matches["rating_period_start_order"] = matches.groupby("rating_period_id", sort=False).ngroup()
    return matches


def get_player_state(
    ratings: dict[int, float],
    rds: dict[int, float],
    code: int,
    default_rating: float,
    default_rd: float,
) -> tuple[float, float]:
    if code not in ratings:
        ratings[code] = default_rating
        rds[code] = default_rd
    return ratings[code], rds[code]


def build_prediction_row(
    row: Any,
    setting_name: str,
    period_type: str,
    start_year: int,
    end_year: int,
    winner: int,
    loser: int,
    player_a: int,
    player_b: int,
    actual_a_win: int,
    pred_a_win: float,
    pred_winner_win: float,
    winner_rating_before: float,
    winner_rd_before: float,
    loser_rating_before: float,
    loser_rd_before: float,
    player_a_rating_before: float,
    player_a_rd_before: float,
    player_b_rating_before: float,
    player_b_rd_before: float,
) -> dict[str, Any]:
    return {
        "setting_name": setting_name,
        "period_type": period_type,
        "rating_period_id": getattr(row, "rating_period_id"),
        "rating_period_start_order": int(getattr(row, "rating_period_start_order")),
        "start_year": start_year,
        "end_year": end_year,
        "fcode": int(row.fcode),
        "code": int(row.code),
        "year": int(row.year),
        "event": int(row.event),
        "event_key": getattr(row, "event_key"),
        "eventname": getattr(row, "eventname", pd.NA),
        "event_date_raw": getattr(row, "event_date_raw", pd.NA),
        "event_date_parsed": getattr(row, "event_date_parsed", pd.NA),
        "event_order_date": getattr(row, "event_order_date", pd.NaT),
        "event_date_ordering_method": getattr(row, "event_date_ordering_method", pd.NA),
        "winner": winner,
        "loser": loser,
        "player_a": player_a,
        "player_b": player_b,
        "actual_a_win": actual_a_win,
        "pred_a_win": pred_a_win,
        "pred_winner_win": pred_winner_win,
        "winner_rating_before_period": winner_rating_before,
        "winner_rd_before_period": winner_rd_before,
        "loser_rating_before_period": loser_rating_before,
        "loser_rd_before_period": loser_rd_before,
        "player_a_rating_before_period": player_a_rating_before,
        "player_a_rd_before_period": player_a_rd_before,
        "player_b_rating_before_period": player_b_rating_before,
        "player_b_rd_before_period": player_b_rd_before,
    }


def attach_after_period_fields(
    prediction_row: dict[str, Any],
    new_ratings: dict[int, float],
    new_rds: dict[int, float],
) -> None:
    winner = prediction_row["winner"]
    loser = prediction_row["loser"]
    player_a = prediction_row["player_a"]
    player_b = prediction_row["player_b"]

    prediction_row["winner_rating_after_period"] = new_ratings[winner]
    prediction_row["winner_rd_after_period"] = new_rds[winner]
    prediction_row["loser_rating_after_period"] = new_ratings[loser]
    prediction_row["loser_rd_after_period"] = new_rds[loser]
    prediction_row["player_a_rating_after_period"] = new_ratings[player_a]
    prediction_row["player_a_rd_after_period"] = new_rds[player_a]
    prediction_row["player_b_rating_after_period"] = new_ratings[player_b]
    prediction_row["player_b_rd_after_period"] = new_rds[player_b]


def make_yearly_snapshot(
    setting_name: str,
    period_type: str,
    year: int,
    active_players: set[int],
    ratings: dict[int, float],
    rds: dict[int, float],
) -> dict[str, Any]:
    active_rds = np.asarray([rds[player] for player in active_players], dtype=float)
    active_ratings = np.asarray([ratings[player] for player in active_players], dtype=float)
    return {
        "setting_name": setting_name,
        "period_type": period_type,
        "year": year,
        "n_active_players_in_year": len(active_players),
        "mean_rd_active_players": float(np.mean(active_rds)),
        "median_rd_active_players": float(np.median(active_rds)),
        "p10_rd_active_players": float(np.quantile(active_rds, 0.10)),
        "p90_rd_active_players": float(np.quantile(active_rds, 0.90)),
        "mean_rating_active_players": float(np.mean(active_ratings)),
        "median_rating_active_players": float(np.median(active_ratings)),
    }


def validate_core_constants(default_rating: float, default_rd: float, min_rd: float, max_rd: float, c: float) -> None:
    if (
        default_rating != CORE_DEFAULT_RATING
        or default_rd != CORE_DEFAULT_RD
        or min_rd != CORE_MIN_RD
        or max_rd != CORE_MAX_RD
    ):
        raise ValueError(
            "This script uses glicko_core.py constants for the baseline setting only. "
            "Update glicko_core.py before running alternative default rating/RD bounds."
        )
    if c != 0.0:
        raise ValueError("This experiment intentionally keeps C=0; inactivity RD inflation is out of scope.")


def run_match_by_match_period(
    period_matches: pd.DataFrame,
    setting_name: str,
    period_type: str,
    start_year: int,
    end_year: int,
    default_rating: float,
    default_rd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    player_names: dict[int, str] = {}
    games_played: dict[int, int] = defaultdict(int)
    wins: dict[int, int] = defaultdict(int)
    losses: dict[int, int] = defaultdict(int)
    games_2025: dict[int, int] = defaultdict(int)
    wins_2025: dict[int, int] = defaultdict(int)
    losses_2025: dict[int, int] = defaultdict(int)

    prediction_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    current_year: int | None = None
    current_year_active: set[int] = set()

    for row in period_matches.itertuples(index=False):
        year = int(row.year)
        if current_year is None:
            current_year = year
        elif year != current_year:
            yearly_rows.append(
                make_yearly_snapshot(setting_name, period_type, current_year, current_year_active, ratings, rds)
            )
            print(
                f"  Year {current_year}: active_players={len(current_year_active):,}, "
                f"median_RD_active={yearly_rows[-1]['median_rd_active_players']:.3f}"
            )
            current_year = year
            current_year_active = set()

        winner = player_code(row.winner)
        loser = player_code(row.loser)
        update_player_name(player_names, winner, getattr(row, "winner_name", pd.NA))
        update_player_name(player_names, loser, getattr(row, "loser_name", pd.NA))

        winner_rating_before, winner_rd_before = get_player_state(ratings, rds, winner, default_rating, default_rd)
        loser_rating_before, loser_rd_before = get_player_state(ratings, rds, loser, default_rating, default_rd)

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        if player_a == winner:
            player_a_rating_before = winner_rating_before
            player_a_rd_before = winner_rd_before
            player_b_rating_before = loser_rating_before
            player_b_rd_before = loser_rd_before
        else:
            player_a_rating_before = loser_rating_before
            player_a_rd_before = loser_rd_before
            player_b_rating_before = winner_rating_before
            player_b_rd_before = winner_rd_before

        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = expected_score(player_a_rating_before, player_b_rating_before, player_b_rd_before)
        pred_winner_win = pred_a_win if winner == player_a else 1.0 - pred_a_win

        update = update_two_players_single_game(
            winner_rating_before,
            winner_rd_before,
            loser_rating_before,
            loser_rd_before,
            1.0,
        )
        ratings[winner] = update.player1_rating_after
        rds[winner] = update.player1_rd_after
        ratings[loser] = update.player2_rating_after
        rds[loser] = update.player2_rd_after

        prediction_row = build_prediction_row(
            row,
            setting_name,
            period_type,
            start_year,
            end_year,
            winner,
            loser,
            player_a,
            player_b,
            actual_a_win,
            pred_a_win,
            pred_winner_win,
            winner_rating_before,
            winner_rd_before,
            loser_rating_before,
            loser_rd_before,
            player_a_rating_before,
            player_a_rd_before,
            player_b_rating_before,
            player_b_rd_before,
        )
        attach_after_period_fields(prediction_row, ratings, rds)
        prediction_rows.append(prediction_row)

        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1
        if year == END_YEAR:
            games_2025[winner] += 1
            games_2025[loser] += 1
            wins_2025[winner] += 1
            losses_2025[loser] += 1
        current_year_active.update([winner, loser])

        abs_changes = [
            abs(update.player1_rating_after - winner_rating_before),
            abs(update.player2_rating_after - loser_rating_before),
        ]
        rd_changes = [
            update.player1_rd_after - winner_rd_before,
            update.player2_rd_after - loser_rd_before,
        ]
        period_rows.append(
            {
                "setting_name": setting_name,
                "period_type": period_type,
                "rating_period_id": getattr(row, "rating_period_id"),
                "rating_period_start_order": int(getattr(row, "rating_period_start_order")),
                "period_start_date": getattr(row, "event_order_date", pd.NaT),
                "period_end_date": getattr(row, "event_order_date", pd.NaT),
                "year": year,
                "month": pd.to_datetime(getattr(row, "event_order_date", pd.NaT), errors="coerce").month
                if pd.notna(getattr(row, "event_order_date", pd.NaT))
                else pd.NA,
                "event": int(row.event),
                "eventname": getattr(row, "eventname", pd.NA),
                "number_of_matches": 1,
                "number_of_players": 2,
                "mean_games_per_player_in_period": 1.0,
                "max_games_by_one_player_in_period": 1,
                "mean_abs_rating_change_in_period": float(np.mean(abs_changes)),
                "median_abs_rating_change_in_period": float(np.median(abs_changes)),
                "p90_abs_rating_change_in_period": float(np.quantile(abs_changes, 0.90)),
                "mean_rd_change_in_period": float(np.mean(rd_changes)),
                "median_rd_change_in_period": float(np.median(rd_changes)),
            }
        )

    if current_year is not None:
        yearly_rows.append(make_yearly_snapshot(setting_name, period_type, current_year, current_year_active, ratings, rds))
        print(
            f"  Year {current_year}: active_players={len(current_year_active):,}, "
            f"median_RD_active={yearly_rows[-1]['median_rd_active_players']:.3f}"
        )

    return (
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        build_final_ratings(
            setting_name,
            period_type,
            start_year,
            end_year,
            ratings,
            rds,
            player_names,
            games_played,
            wins,
            losses,
            games_2025,
            wins_2025,
            losses_2025,
        ),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(period_rows),
    )


def run_batch_period(
    period_matches: pd.DataFrame,
    setting_name: str,
    period_type: str,
    start_year: int,
    end_year: int,
    default_rating: float,
    default_rd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    player_names: dict[int, str] = {}
    games_played: dict[int, int] = defaultdict(int)
    wins: dict[int, int] = defaultdict(int)
    losses: dict[int, int] = defaultdict(int)
    games_2025: dict[int, int] = defaultdict(int)
    wins_2025: dict[int, int] = defaultdict(int)
    losses_2025: dict[int, int] = defaultdict(int)

    prediction_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    current_year: int | None = None
    current_year_active: set[int] = set()

    for _, group in period_matches.groupby("rating_period_id", sort=False):
        period_year = int(group["year"].iloc[0])
        if current_year is None:
            current_year = period_year
        elif period_year != current_year:
            yearly_rows.append(
                make_yearly_snapshot(setting_name, period_type, current_year, current_year_active, ratings, rds)
            )
            print(
                f"  Year {current_year}: active_players={len(current_year_active):,}, "
                f"median_RD_active={yearly_rows[-1]['median_rd_active_players']:.3f}"
            )
            current_year = period_year
            current_year_active = set()

        period_start_ratings: dict[int, float] = {}
        period_start_rds: dict[int, float] = {}
        period_games: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"opponent_ratings": [], "opponent_rds": [], "scores": []}
        )
        period_games_count: dict[int, int] = defaultdict(int)
        period_prediction_rows: list[dict[str, Any]] = []

        def start_state(code: int) -> tuple[float, float]:
            if code not in period_start_ratings:
                rating, rd = get_player_state(ratings, rds, code, default_rating, default_rd)
                period_start_ratings[code] = rating
                period_start_rds[code] = rd
            return period_start_ratings[code], period_start_rds[code]

        for row in group.itertuples(index=False):
            winner = player_code(row.winner)
            loser = player_code(row.loser)
            year = int(row.year)
            update_player_name(player_names, winner, getattr(row, "winner_name", pd.NA))
            update_player_name(player_names, loser, getattr(row, "loser_name", pd.NA))

            winner_rating_before, winner_rd_before = start_state(winner)
            loser_rating_before, loser_rd_before = start_state(loser)

            player_a = min(winner, loser)
            player_b = max(winner, loser)
            if player_a == winner:
                player_a_rating_before = winner_rating_before
                player_a_rd_before = winner_rd_before
                player_b_rating_before = loser_rating_before
                player_b_rd_before = loser_rd_before
            else:
                player_a_rating_before = loser_rating_before
                player_a_rd_before = loser_rd_before
                player_b_rating_before = winner_rating_before
                player_b_rd_before = winner_rd_before

            actual_a_win = 1 if player_a == winner else 0
            pred_a_win = expected_score(player_a_rating_before, player_b_rating_before, player_b_rd_before)
            pred_winner_win = pred_a_win if winner == player_a else 1.0 - pred_a_win

            period_games[winner]["opponent_ratings"].append(loser_rating_before)
            period_games[winner]["opponent_rds"].append(loser_rd_before)
            period_games[winner]["scores"].append(1.0)
            period_games[loser]["opponent_ratings"].append(winner_rating_before)
            period_games[loser]["opponent_rds"].append(winner_rd_before)
            period_games[loser]["scores"].append(0.0)
            period_games_count[winner] += 1
            period_games_count[loser] += 1

            games_played[winner] += 1
            games_played[loser] += 1
            wins[winner] += 1
            losses[loser] += 1
            if year == END_YEAR:
                games_2025[winner] += 1
                games_2025[loser] += 1
                wins_2025[winner] += 1
                losses_2025[loser] += 1
            current_year_active.update([winner, loser])

            period_prediction_rows.append(
                build_prediction_row(
                    row,
                    setting_name,
                    period_type,
                    start_year,
                    end_year,
                    winner,
                    loser,
                    player_a,
                    player_b,
                    actual_a_win,
                    pred_a_win,
                    pred_winner_win,
                    winner_rating_before,
                    winner_rd_before,
                    loser_rating_before,
                    loser_rd_before,
                    player_a_rating_before,
                    player_a_rd_before,
                    player_b_rating_before,
                    player_b_rd_before,
                )
            )

        new_ratings: dict[int, float] = {}
        new_rds: dict[int, float] = {}
        abs_rating_changes = []
        rd_changes = []
        for code, game_lists in period_games.items():
            old_rating = period_start_ratings[code]
            old_rd = period_start_rds[code]
            new_rating, new_rd = update_player_glicko(
                old_rating,
                old_rd,
                game_lists["opponent_ratings"],
                game_lists["opponent_rds"],
                game_lists["scores"],
            )
            new_ratings[code] = new_rating
            new_rds[code] = new_rd
            abs_rating_changes.append(abs(new_rating - old_rating))
            rd_changes.append(new_rd - old_rd)

        for code, new_rating in new_ratings.items():
            ratings[code] = new_rating
            rds[code] = new_rds[code]

        for prediction_row in period_prediction_rows:
            attach_after_period_fields(prediction_row, new_ratings, new_rds)
        prediction_rows.extend(period_prediction_rows)

        period_start_date = pd.to_datetime(group["event_order_date"], errors="coerce").min()
        period_end_date = pd.to_datetime(group["event_order_date"], errors="coerce").max()
        games_per_player = np.asarray(list(period_games_count.values()), dtype=float)
        first = group.iloc[0]
        month_value = (
            int(period_start_date.month)
            if pd.notna(period_start_date)
            else ("unknown" if period_type == "monthly" else pd.NA)
        )
        period_rows.append(
            {
                "setting_name": setting_name,
                "period_type": period_type,
                "rating_period_id": first["rating_period_id"],
                "rating_period_start_order": int(first["rating_period_start_order"]),
                "period_start_date": period_start_date,
                "period_end_date": period_end_date,
                "year": int(first["year"]),
                "month": month_value,
                "event": int(first["event"]) if period_type == "event_level" else pd.NA,
                "eventname": first["eventname"] if period_type == "event_level" else pd.NA,
                "number_of_matches": len(group),
                "number_of_players": len(period_games_count),
                "mean_games_per_player_in_period": float(games_per_player.mean()),
                "max_games_by_one_player_in_period": int(games_per_player.max()),
                "mean_abs_rating_change_in_period": float(np.mean(abs_rating_changes)),
                "median_abs_rating_change_in_period": float(np.median(abs_rating_changes)),
                "p90_abs_rating_change_in_period": float(np.quantile(abs_rating_changes, 0.90)),
                "mean_rd_change_in_period": float(np.mean(rd_changes)),
                "median_rd_change_in_period": float(np.median(rd_changes)),
            }
        )

    if current_year is not None:
        yearly_rows.append(make_yearly_snapshot(setting_name, period_type, current_year, current_year_active, ratings, rds))
        print(
            f"  Year {current_year}: active_players={len(current_year_active):,}, "
            f"median_RD_active={yearly_rows[-1]['median_rd_active_players']:.3f}"
        )

    return (
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        build_final_ratings(
            setting_name,
            period_type,
            start_year,
            end_year,
            ratings,
            rds,
            player_names,
            games_played,
            wins,
            losses,
            games_2025,
            wins_2025,
            losses_2025,
        ),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(period_rows),
    )


def build_final_ratings(
    setting_name: str,
    period_type: str,
    start_year: int,
    end_year: int,
    ratings: dict[int, float],
    rds: dict[int, float],
    player_names: dict[int, str],
    games_played: dict[int, int],
    wins: dict[int, int],
    losses: dict[int, int],
    games_2025: dict[int, int],
    wins_2025: dict[int, int],
    losses_2025: dict[int, int],
) -> pd.DataFrame:
    rows = []
    for code in sorted(ratings):
        rows.append(
            {
                "setting_name": setting_name,
                "period_type": period_type,
                "start_year": start_year,
                "end_year": end_year,
                "player_code": code,
                "player_name": player_names.get(code, pd.NA),
                "final_rating": ratings[code],
                "final_rd": rds[code],
                "conservative_rating": ratings[code] - 2.0 * rds[code],
                "games_played_in_run": games_played.get(code, 0),
                "wins_in_run": wins.get(code, 0),
                "losses_in_run": losses.get(code, 0),
                "games_played_2025": games_2025.get(code, 0),
                "wins_2025": wins_2025.get(code, 0),
                "losses_2025": losses_2025.get(code, 0),
            }
        )
    final_ratings = pd.DataFrame(rows)
    final_ratings["final_rank_by_rating"] = (
        final_ratings["final_rating"].rank(method="min", ascending=False).astype(int)
    )
    final_ratings["final_rank_by_conservative_rating"] = (
        final_ratings["conservative_rating"].rank(method="min", ascending=False).astype(int)
    )
    return final_ratings.sort_values(["setting_name", "final_rank_by_rating"]).reset_index(drop=True)


def run_glicko_by_rating_period(
    matches: pd.DataFrame,
    period_type: str,
    setting_name: str,
    start_year: int,
    end_year: int,
    default_rating: float = 1500.0,
    default_rd: float = 350.0,
    min_rd: float = 30.0,
    max_rd: float = 350.0,
    c: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Glicko-1 by the requested rating-period assumption."""

    validate_core_constants(default_rating, default_rd, min_rd, max_rd, c)
    period_matches = matches[(matches["year"] >= start_year) & (matches["year"] <= end_year)].copy()
    if period_matches.empty:
        raise ValueError(f"No matches found for {start_year}-{end_year}")

    period_matches = assign_rating_period_id(period_matches, period_type)
    n_periods = period_matches["rating_period_id"].nunique()
    print(f"  Number of rating periods: {n_periods:,}")

    if period_type == "match_by_match":
        return run_match_by_match_period(
            period_matches,
            setting_name,
            period_type,
            start_year,
            end_year,
            default_rating,
            default_rd,
        )

    return run_batch_period(
        period_matches,
        setting_name,
        period_type,
        start_year,
        end_year,
        default_rating,
        default_rd,
    )


def hash_fcodes(fcodes: list[int]) -> str:
    text = ",".join(str(int(fcode)) for fcode in fcodes)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_prediction_metrics(
    predictions: pd.DataFrame,
    setting_name: str,
    period_type: str,
    reference_fcode_set: set[int] | None,
    reference_fcode_order: list[int] | None,
) -> tuple[pd.DataFrame, set[int], list[int], list[str]]:
    eval_df = predictions[predictions["year"] == END_YEAR].copy()
    if eval_df.empty:
        raise ValueError(f"No {END_YEAR} games found for {setting_name}")

    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)
    clipped = pred.clip(EPS, 1.0 - EPS)
    log_loss = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))
    observed_win_rate = y.mean()
    fcode_order = eval_df["fcode"].astype(int).tolist()
    fcode_set = set(fcode_order)

    warnings = []
    if len(eval_df) != EXPECTED_2025_GAMES:
        warnings.append(f"{setting_name}: expected {EXPECTED_2025_GAMES} 2025 games but found {len(eval_df)}")
    set_matches = True if reference_fcode_set is None else fcode_set == reference_fcode_set
    order_matches = True if reference_fcode_order is None else fcode_order == reference_fcode_order
    if not set_matches:
        warnings.append(f"{setting_name}: 2025 fcode set does not match reference")

    row = {
        "setting_name": setting_name,
        "period_type": period_type,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "evaluation_year": END_YEAR,
        "evaluation_games": len(eval_df),
        "log_loss": float(log_loss),
        "brier_score": float(brier_score),
        "accuracy": float(accuracy),
        "baseline_accuracy": float(max(observed_win_rate, 1.0 - observed_win_rate)),
        "mean_predicted_probability": float(pred.mean()),
        "observed_win_rate": float(observed_win_rate),
        "pred_a_win_min": float(pred.min()),
        "pred_a_win_max": float(pred.max()),
        "pred_a_win_out_of_range_count": int(((pred < 0) | (pred > 1)).sum()),
        "evaluation_fcode_hash": hash_fcodes(fcode_order),
        "evaluation_fcode_set_matches_reference": set_matches,
        "evaluation_fcode_order_matches_reference": order_matches,
    }
    return pd.DataFrame([row]), fcode_set, fcode_order, warnings


def make_calibration_table(predictions: pd.DataFrame, setting_name: str, period_type: str) -> pd.DataFrame:
    eval_df = predictions[predictions["year"] == END_YEAR].copy()
    eval_df["actual_a_win"] = eval_df["actual_a_win"].astype(float)
    eval_df["pred_a_win"] = eval_df["pred_a_win"].astype(float)
    eval_df["brier_component"] = (eval_df["pred_a_win"] - eval_df["actual_a_win"]) ** 2

    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    eval_df["bin"] = pd.cut(eval_df["pred_a_win"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        eval_df.groupby("bin", observed=False)
        .agg(
            n_games=("actual_a_win", "size"),
            mean_predicted_probability=("pred_a_win", "mean"),
            observed_win_rate=("actual_a_win", "mean"),
            brier_score=("brier_component", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "period_type", period_type)
    grouped.insert(0, "setting_name", setting_name)
    return grouped


def make_confidence_table(predictions: pd.DataFrame, setting_name: str, period_type: str) -> pd.DataFrame:
    eval_df = predictions[predictions["year"] == END_YEAR].copy()
    eval_df["actual_a_win"] = eval_df["actual_a_win"].astype(float)
    eval_df["pred_a_win"] = eval_df["pred_a_win"].astype(float)
    eval_df["confidence"] = np.maximum(eval_df["pred_a_win"], 1.0 - eval_df["pred_a_win"])
    eval_df["correct"] = (eval_df["pred_a_win"] >= 0.5) == (eval_df["actual_a_win"] == 1.0)
    eval_df["brier_component"] = (eval_df["pred_a_win"] - eval_df["actual_a_win"]) ** 2

    bins = np.arange(0.50, 1.00001, 0.05)
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    eval_df["confidence_bin"] = pd.cut(
        eval_df["confidence"], bins=bins, labels=labels, include_lowest=True
    )
    grouped = (
        eval_df.groupby("confidence_bin", observed=False)
        .agg(
            n_games=("actual_a_win", "size"),
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            brier_score=("brier_component", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "period_type", period_type)
    grouped.insert(0, "setting_name", setting_name)
    return grouped


def make_rd_summary(final_ratings: pd.DataFrame, setting_name: str, period_type: str) -> pd.DataFrame:
    rd = final_ratings["final_rd"].astype(float)
    n_players = len(final_ratings)
    return pd.DataFrame(
        [
            {
                "setting_name": setting_name,
                "period_type": period_type,
                "n_players": n_players,
                "mean_final_rd": float(rd.mean()),
                "median_final_rd": float(rd.median()),
                "p10_final_rd": float(rd.quantile(0.10)),
                "p25_final_rd": float(rd.quantile(0.25)),
                "p75_final_rd": float(rd.quantile(0.75)),
                "p90_final_rd": float(rd.quantile(0.90)),
                "min_final_rd": float(rd.min()),
                "max_final_rd": float(rd.max()),
                "number_at_min_rd": int(np.isclose(rd, MIN_RD).sum()),
                "number_at_max_rd": int(np.isclose(rd, MAX_RD).sum()),
                "proportion_at_min_rd": float(np.isclose(rd, MIN_RD).sum() / n_players),
                "proportion_at_max_rd": float(np.isclose(rd, MAX_RD).sum() / n_players),
            }
        ]
    )


def top_overlap(ref: pd.DataFrame, comp: pd.DataFrame, n: int) -> float:
    if ref.empty or comp.empty:
        return np.nan
    k = min(n, len(ref), len(comp))
    if k == 0:
        return np.nan
    ref_top = set(ref.sort_values("final_rating_ref", ascending=False).head(k)["player_code"])
    comp_top = set(comp.sort_values("final_rating_comp", ascending=False).head(k)["player_code"])
    return len(ref_top & comp_top) / k


def compare_rating_lists(
    all_final_ratings: pd.DataFrame,
    active_2025_counts: dict[int, int],
) -> pd.DataFrame:
    ref = all_final_ratings[all_final_ratings["period_type"] == "match_by_match"].copy()
    ref = ref.rename(
        columns={
            "final_rating": "final_rating_ref",
            "final_rank_by_rating": "rank_ref",
        }
    )

    rows = []
    for period_type in ["event_level", "monthly", "yearly"]:
        comp = all_final_ratings[all_final_ratings["period_type"] == period_type].copy()
        comp_setting = comp["setting_name"].iloc[0]
        comp = comp.rename(
            columns={
                "final_rating": "final_rating_comp",
                "final_rank_by_rating": "rank_comp",
            }
        )
        merged = ref[["player_code", "final_rating_ref", "rank_ref"]].merge(
            comp[["player_code", "final_rating_comp", "rank_comp"]],
            on="player_code",
            how="inner",
        )
        merged["games_played_2025"] = merged["player_code"].map(active_2025_counts).fillna(0).astype(int)
        merged["abs_rating_difference"] = (
            merged["final_rating_ref"] - merged["final_rating_comp"]
        ).abs()
        merged["abs_rank_difference"] = (merged["rank_ref"] - merged["rank_comp"]).abs()

        for subset_name, min_games in ACTIVE_PLAYER_SUBSETS:
            subset = merged if min_games == 0 else merged[merged["games_played_2025"] >= min_games]
            common_players = len(subset)
            if common_players >= 2:
                pearson = subset["final_rating_ref"].corr(subset["final_rating_comp"], method="pearson")
                spearman = subset["final_rating_ref"].corr(subset["final_rating_comp"], method="spearman")
            else:
                pearson = np.nan
                spearman = np.nan

            rows.append(
                {
                    "reference_setting_name": "glicko1_period_match_c0",
                    "reference_period_type": "match_by_match",
                    "comparison_setting_name": comp_setting,
                    "comparison_period_type": period_type,
                    "player_subset": subset_name,
                    "min_2025_games": min_games,
                    "common_players": common_players,
                    "pearson_rating_correlation": pearson,
                    "spearman_rank_correlation": spearman,
                    "mean_abs_rating_difference": float(subset["abs_rating_difference"].mean())
                    if common_players
                    else np.nan,
                    "median_abs_rating_difference": float(subset["abs_rating_difference"].median())
                    if common_players
                    else np.nan,
                    "p90_abs_rating_difference": float(subset["abs_rating_difference"].quantile(0.90))
                    if common_players
                    else np.nan,
                    "max_abs_rating_difference": float(subset["abs_rating_difference"].max())
                    if common_players
                    else np.nan,
                    "mean_abs_rank_difference": float(subset["abs_rank_difference"].mean())
                    if common_players
                    else np.nan,
                    "median_abs_rank_difference": float(subset["abs_rank_difference"].median())
                    if common_players
                    else np.nan,
                    "top10_overlap": top_overlap(subset, subset, 10),
                    "top25_overlap": top_overlap(subset, subset, 25),
                    "top50_overlap": top_overlap(subset, subset, 50),
                    "top100_overlap": top_overlap(subset, subset, 100),
                }
            )
    return pd.DataFrame(rows)


def append_predictions(predictions: pd.DataFrame, first_write: bool) -> None:
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig" if first_write else "utf-8",
        mode="w" if first_write else "a",
        header=first_write,
    )


def write_summary(
    matches: pd.DataFrame,
    metrics: pd.DataFrame,
    rd_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    date_summary: pd.DataFrame,
    warnings: list[str],
) -> None:
    metric_lines = []
    for row in metrics.sort_values("period_type").itertuples(index=False):
        metric_lines.append(
            f"- {row.setting_name}: log loss {row.log_loss:.6f}, "
            f"Brier {row.brier_score:.6f}, accuracy {row.accuracy:.6f}"
        )

    rd_lines = []
    for row in rd_summary.sort_values("period_type").itertuples(index=False):
        rd_lines.append(
            f"- {row.setting_name}: median RD {row.median_final_rd:.3f}, "
            f"mean RD {row.mean_final_rd:.3f}, at MIN_RD {int(row.number_at_min_rd):,} "
            f"({row.proportion_at_min_rd:.1%})"
        )

    active_similarity = similarity[
        (similarity["player_subset"] == "active_2025_games_ge5")
    ].sort_values("comparison_period_type")
    similarity_lines = []
    for row in active_similarity.itertuples(index=False):
        similarity_lines.append(
            f"- {row.comparison_setting_name} vs match-by-match, active 2025 games >=5: "
            f"Spearman {row.spearman_rank_correlation:.4f}, "
            f"Top50 overlap {row.top50_overlap:.3f}, Top100 overlap {row.top100_overlap:.3f}, "
            f"mean abs rating difference {row.mean_abs_rating_difference:.2f}"
        )

    date_lines = []
    one_setting_date = date_summary[date_summary["setting_name"] == PERIOD_SETTINGS[0]["setting_name"]]
    for row in one_setting_date.itertuples(index=False):
        date_lines.append(
            f"- {row.event_date_ordering_method}: {int(row.match_count):,} matches ({row.share_of_matches:.2%})"
        )

    warning_lines = ["- None"] if not warnings else [f"- {warning}" for warning in warnings]
    best_metric = metrics.sort_values("log_loss").iloc[0]

    lines = [
        "# Glicko-1 Rating-Period Sensitivity Summary",
        "",
        "## Aim",
        "",
        "This experiment tests how different Glicko rating-period assumptions map onto croquet data.",
        "",
        "## Why Rating Period Matters",
        "",
        "Classic Glicko is period-based: ratings and RDs are held fixed within a rating period, all games in that period are collected, and player states are updated at the end of the period. Changing the period therefore changes both predictions and RD behaviour.",
        "",
        "## Rating-Period Assumptions Tested",
        "",
        "- Match-by-match: each game is one rating period.",
        "- Event-level: all games in the same year-event are one rating period.",
        "- Monthly: all games in the same calendar month are one rating period.",
        "- Yearly: all games in the same calendar year are one rating period.",
        "",
        "## Controlled Design",
        "",
        "All settings use the same Glicko-1 formula, same full-history dataset, same initial rating/RD, and C=0. This is deliberate: the experiment compares rating-period assumptions only. Inactivity RD inflation is not included here and should be tested separately.",
        "",
        "## Data Used",
        "",
        f"- Dataset: `outputs/elo_optimization/matches_1985_2025_checked.csv`",
        f"- Years: {int(matches['year'].min())}-{int(matches['year'].max())}",
        f"- Matches: {len(matches):,}",
        f"- Unique players: {pd.concat([matches['winner'], matches['loser']]).dropna().astype(int).nunique():,}",
        f"- 2025 test games per setting: {EXPECTED_2025_GAMES:,}",
        "",
        "Date ordering:",
        "",
        *date_lines,
        "",
        "## Prediction Results",
        "",
        *metric_lines,
        "",
        f"The best 2025 log loss in this diagnostic is from `{best_metric['setting_name']}`. This should not be over-interpreted yet; it only answers how period assumptions behave under C=0.",
        "",
        "## Final Rating List Similarity",
        "",
        *similarity_lines,
        "",
        "## RD Behaviour",
        "",
        *rd_lines,
        "",
        "## Interpretation for Supervisor Question",
        "",
        "If match-by-match and event-level results are very similar, that supports match-by-match updates as a pragmatic baseline for game-level croquet data. If monthly or yearly results differ more, that suggests very coarse periods may lose useful chronological information. There may not be a single theoretically preferred answer; the choice can be justified pragmatically and experimentally.",
        "",
        "## Important Limitations",
        "",
        "- This experiment keeps C=0, so inactivity RD inflation is not tested yet.",
        "- This is still Glicko-1, not Glicko-2.",
        "- It does not yet compare against Elo.",
        "- Yearly period is included mainly as a coarse diagnostic, not necessarily as a realistic candidate.",
        "",
        "## Warnings",
        "",
        *warning_lines,
        "",
        "## Next Step",
        "",
        "Next, either add inactivity RD inflation sensitivity for the most plausible period assumptions, or prepare a fair Elo-vs-Glicko comparison using the selected Glicko period assumption.",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    matches = load_matches()
    date_summary = make_date_ordering_summary(matches)

    print()
    print("Rating-period settings:")
    for setting in PERIOD_SETTINGS:
        print(f"  {setting['setting_name']} ({setting['period_type']})")

    print()
    print("Event date ordering summary:")
    print(
        date_summary[date_summary["setting_name"] == PERIOD_SETTINGS[0]["setting_name"]][
            ["event_date_ordering_method", "match_count", "share_of_matches"]
        ].to_string(index=False)
    )

    all_metrics = []
    all_calibration = []
    all_confidence = []
    all_final_ratings = []
    all_rd_summary = []
    all_yearly_rd = []
    all_period_summary = []
    reference_fcode_set: set[int] | None = None
    reference_fcode_order: list[int] | None = None

    first_prediction_write = True
    for setting in PERIOD_SETTINGS:
        print()
        print(f"Running {setting['setting_name']} ({setting['period_type']})...")
        predictions, final_ratings, yearly_rd, period_summary = run_glicko_by_rating_period(
            matches=matches,
            period_type=setting["period_type"],
            setting_name=setting["setting_name"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            default_rating=DEFAULT_RATING,
            default_rd=DEFAULT_RD,
            min_rd=MIN_RD,
            max_rd=MAX_RD,
            c=C,
        )

        metrics, fcode_set, fcode_order, metric_warnings = compute_prediction_metrics(
            predictions,
            setting["setting_name"],
            setting["period_type"],
            reference_fcode_set,
            reference_fcode_order,
        )
        warnings.extend(metric_warnings)
        if reference_fcode_set is None:
            reference_fcode_set = fcode_set
            reference_fcode_order = fcode_order

        calibration = make_calibration_table(predictions, setting["setting_name"], setting["period_type"])
        confidence = make_confidence_table(predictions, setting["setting_name"], setting["period_type"])
        rd_summary = make_rd_summary(final_ratings, setting["setting_name"], setting["period_type"])

        append_predictions(predictions, first_prediction_write)
        first_prediction_write = False

        all_metrics.append(metrics)
        all_calibration.append(calibration)
        all_confidence.append(confidence)
        all_final_ratings.append(final_ratings)
        all_rd_summary.append(rd_summary)
        all_yearly_rd.append(yearly_rd)
        all_period_summary.append(period_summary)

        metric = metrics.iloc[0]
        rd = rd_summary.iloc[0]
        print(
            f"  2025 games={int(metric['evaluation_games']):,}, "
            f"log_loss={metric['log_loss']:.6f}, "
            f"brier={metric['brier_score']:.6f}, "
            f"accuracy={metric['accuracy']:.6f}"
        )
        print(
            f"  final median RD={rd['median_final_rd']:.3f}, "
            f"players at MIN_RD={int(rd['number_at_min_rd']):,}"
        )

        del predictions

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    calibration_df = pd.concat(all_calibration, ignore_index=True)
    confidence_df = pd.concat(all_confidence, ignore_index=True)
    final_ratings_df = pd.concat(all_final_ratings, ignore_index=True)
    rd_summary_df = pd.concat(all_rd_summary, ignore_index=True)
    yearly_rd_df = pd.concat(all_yearly_rd, ignore_index=True)
    period_summary_df = pd.concat(all_period_summary, ignore_index=True)

    active_2025_matches = matches[matches["year"] == END_YEAR]
    active_2025_counts = (
        pd.concat([active_2025_matches["winner"], active_2025_matches["loser"]])
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )
    similarity_df = compare_rating_lists(final_ratings_df, active_2025_counts)

    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    confidence_df.to_csv(CONFIDENCE_PATH, index=False, encoding="utf-8-sig")
    final_ratings_df.to_csv(FINAL_RATINGS_PATH, index=False, encoding="utf-8-sig")
    rd_summary_df.to_csv(RD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    yearly_rd_df.to_csv(YEARLY_RD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    period_summary_df.to_csv(PERIOD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    similarity_df.to_csv(LIST_SIMILARITY_PATH, index=False, encoding="utf-8-sig")
    date_summary.to_csv(DATE_ORDERING_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    write_summary(matches, metrics_df, rd_summary_df, similarity_df, date_summary, warnings)

    print()
    print("2025 fcode set consistency:")
    print(metrics_df[["setting_name", "evaluation_games", "evaluation_fcode_set_matches_reference"]].to_string(index=False))

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("Warnings: none")

    print()
    print("Output files:")
    for path in [
        PREDICTIONS_PATH,
        METRICS_PATH,
        CALIBRATION_PATH,
        CONFIDENCE_PATH,
        FINAL_RATINGS_PATH,
        RD_SUMMARY_PATH,
        YEARLY_RD_SUMMARY_PATH,
        PERIOD_SUMMARY_PATH,
        LIST_SIMILARITY_PATH,
        DATE_ORDERING_SUMMARY_PATH,
        SUMMARY_MD_PATH,
    ]:
        print(f"  {path}")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
