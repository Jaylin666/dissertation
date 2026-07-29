"""Meeting 5 Glicko rating-period runtime comparison.

This script compares how the Glicko rating-period choice affects prediction
performance, computation time, and active-player rating-list similarity.

The experiment fixes the Glicko-1 formula, initial rating/RD, RD bounds, the
full-history match dataset, and the low_inflation inactivity RD setting from
the previous sensitivity step. Only the rating-period setting changes:
match-by-match, event-level, monthly, and yearly.

This script does not run an Elo-vs-Glicko final comparison and does not
implement adaptive-K Elo.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from glicko_core import (  # noqa: E402
    DEFAULT_RATING,
    DEFAULT_RD,
    MAX_RD,
    MIN_RD,
    expected_score,
    update_player_glicko,
)


START_YEAR = 1985
END_YEAR = 2025
EXPECTED_2025_GAMES = 11_379
EPS = 1e-15
NEAR_MAX_RD_THRESHOLD = MAX_RD - 5.0

RD_INFLATION_VARIANT = "low_inflation"
LOW_INFLATION_TARGET_MONTHS = 240

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting5_glicko_rating_period_runtime"

METRICS_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_runtime_metrics.csv"
SIMILARITY_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_similarity.csv"
RUNTIME_DETAILS_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_runtime_details.csv"
PREDICTIONS_2025_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_predictions_2025.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_final_ratings.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_runtime_summary.md"

BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_brier_bar.png"
LOGLOSS_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_logloss_bar.png"
RUNTIME_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_runtime_bar.png"
RUNTIME_VS_BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_runtime_vs_brier.png"
SIMILARITY_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rating_period_similarity_bar.png"

REQUIRED_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

RATING_PERIOD_SETTINGS = [
    {"rating_period": "match_by_match", "label": "Match-by-match"},
    {"rating_period": "event_level", "label": "Event-level"},
    {"rating_period": "monthly", "label": "Monthly"},
    {"rating_period": "yearly", "label": "Yearly"},
]

SIMILARITY_GROUPS = [
    ("all_common_players", 0, 0),
    ("active_2025_games_ge5", 5, 0),
    ("total_games_ge100", 0, 100),
    ("active_2025_games_ge5_and_total_games_ge100", 5, 100),
]

LOW_INFLATION_EXPECTED = {
    "log_loss": 0.552154,
    "brier": 0.187724,
    "accuracy": 0.711574,
}


def find_file(filename: str, preferred_relative: str | None = None) -> Path | None:
    """Find a project file using a preferred path first, then Path.rglob."""

    if preferred_relative:
        preferred = PROJECT_ROOT / preferred_relative
        if preferred.exists():
            return preferred
    matches = sorted(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def get_low_inflation_setting() -> tuple[float, str, int, str]:
    """Read low_inflation C from step 3 output, or recalculate it if unavailable."""

    metrics_path = find_file(
        "meeting5_glicko_rd_inflation_metrics_2025.csv",
        "outputs/meeting5_glicko_rd_inflation/meeting5_glicko_rd_inflation_metrics_2025.csv",
    )
    if metrics_path is not None:
        metrics = pd.read_csv(metrics_path)
        row = metrics.loc[metrics["variant"] == RD_INFLATION_VARIANT]
        if not row.empty:
            first = row.iloc[0]
            return (
                float(first["c_value"]),
                str(first.get("inactivity_unit", "month")),
                int(first.get("target_periods", LOW_INFLATION_TARGET_MONTHS)),
                f"read from {metrics_path}",
            )

    c_value = math.sqrt(((MAX_RD**2) - (MIN_RD**2)) / LOW_INFLATION_TARGET_MONTHS)
    return c_value, "month", LOW_INFLATION_TARGET_MONTHS, "recalculated from MAX_RD/MIN_RD and 240 months"


def format_code_value(value: Any) -> str:
    """Return a stable text representation for IDs."""

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
    """Convert a player code into an integer key."""

    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    """Store the first non-empty player name seen."""

    if code in player_names or pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ordering date columns without modifying raw event date fields."""

    matches = matches.copy()
    if "event_date_raw" not in matches.columns:
        matches["event_date_raw"] = pd.NA
    if "event_date_parsed" not in matches.columns:
        matches["event_date_parsed"] = pd.NA

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
        years_numeric = raw_years.astype(int)
        years = np.where(
            raw_years.str.len().eq(2),
            np.where(years_numeric >= 85, 1900 + years_numeric, 2000 + years_numeric),
            years_numeric,
        )
        valid_month = months.between(1, 12).fillna(False)
        valid_mask = valid_month.to_numpy(dtype=bool)
        valid_index = extracted.loc[valid_month_year].index[valid_mask]
        imputed_dates = pd.to_datetime(
            {
                "year": np.asarray(years)[valid_mask],
                "month": months.loc[valid_index].astype(int).to_numpy(),
                "day": np.repeat(15, len(valid_index)),
            },
            errors="coerce",
        )
        matches.loc[valid_index, "event_order_date"] = imputed_dates.to_numpy()
        matches.loc[valid_index, "event_date_ordering_method"] = "month_year_imputed"

    return matches


def add_calendar_period_index(matches: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Create month-based inactivity period index, with a year fallback."""

    matches = matches.copy()
    dates = pd.to_datetime(matches["event_order_date"], errors="coerce")
    if dates.notna().any():
        period_index = (dates.dt.year * 12 + dates.dt.month).astype("Float64")
        fallback_mask = dates.isna()
        period_index.loc[fallback_mask] = matches.loc[fallback_mask, "year"].astype(int) * 12 + 12
        matches["calendar_period_index"] = period_index.astype(int)
        matches["calendar_period_source"] = np.where(
            fallback_mask,
            "year_fallback_december",
            "month_from_event_order_date",
        )
        return matches, "month"

    matches["calendar_period_index"] = matches["year"].astype(int)
    matches["calendar_period_source"] = "year_only_fallback"
    return matches, "year"


def load_matches() -> tuple[pd.DataFrame, str, Path]:
    """Load and prepare the full-history match dataset."""

    dataset_path = find_file(
        "matches_1985_2025_checked.csv",
        "outputs/elo_optimization/matches_1985_2025_checked.csv",
    )
    if dataset_path is None:
        raise FileNotFoundError("Could not find matches_1985_2025_checked.csv.")

    matches = pd.read_csv(dataset_path, low_memory=False)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{dataset_path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    missing_ids = matches[REQUIRED_COLUMNS].isna().sum()
    if int(missing_ids.sum()) > 0:
        raise ValueError(f"Required ID columns contain missing values:\n{missing_ids}")

    matches = matches[(matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)].copy()
    matches = add_event_ordering_columns(matches)
    matches["event_order_date_missing"] = matches["event_order_date"].isna()
    matches = (
        matches.sort_values(
            ["year", "event_order_date_missing", "event_order_date", "event", "fcode"],
            na_position="last",
        )
        .drop(columns=["event_order_date_missing"])
        .reset_index(drop=True)
    )
    matches["chronological_order"] = np.arange(len(matches), dtype=np.int64)
    matches["event_key"] = (
        matches["year"].astype(int).astype(str) + "_" + matches["event"].astype(int).astype(str)
    )
    matches, inactivity_unit = add_calendar_period_index(matches)

    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Loaded dataset: {dataset_path}")
    print(f"Matches: {len(matches):,}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    print(f"Players: {players:,}")
    print(f"Inactivity unit: {inactivity_unit}")
    return matches, inactivity_unit, dataset_path


def assign_rating_period_key(matches: pd.DataFrame, rating_period: str) -> pd.DataFrame:
    """Assign rating period keys for the requested Glicko period setting."""

    data = matches.copy()
    if rating_period == "match_by_match":
        data["rating_period_key"] = "match_" + data["chronological_order"].astype(str)
    elif rating_period == "event_level":
        data["rating_period_key"] = "event_" + data["event_key"].astype(str)
    elif rating_period == "monthly":
        dates = pd.to_datetime(data["event_order_date"], errors="coerce")
        month_id = dates.dt.strftime("%Y_%m")
        fallback = data["year"].astype(int).astype(str) + "_unknown"
        data["rating_period_key"] = "month_" + month_id.fillna(fallback)
    elif rating_period == "yearly":
        data["rating_period_key"] = "year_" + data["year"].astype(int).astype(str)
    else:
        raise ValueError(f"Unknown rating period: {rating_period}")

    data["rating_period_order"] = data.groupby("rating_period_key", sort=False).ngroup()
    return data


def inflate_rd_for_inactivity(
    rd: float,
    elapsed_periods: float,
    c_value: float,
    min_rd: float = MIN_RD,
    max_rd: float = MAX_RD,
) -> float:
    """Inflate RD after inactivity, bounded by configured RD limits."""

    if c_value <= 0.0 or pd.isna(elapsed_periods) or elapsed_periods <= 0:
        return float(rd)
    inflated = math.sqrt((float(rd) ** 2) + (float(c_value) ** 2) * float(elapsed_periods))
    return min(max_rd, max(min_rd, inflated))


def evaluate_winner_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate predictions stored from the actual winner's perspective."""

    p = predictions["pred_winner_win"].astype(float).to_numpy()
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return {
        "evaluation_games_2025": int(len(predictions)),
        "log_loss": float(-np.mean(np.log(clipped))) if len(p) else float("nan"),
        "brier": float(np.mean((p - 1.0) ** 2)) if len(p) else float("nan"),
        "accuracy": float(np.mean(p >= 0.5)) if len(p) else float("nan"),
        "pred_min": float(np.min(p)) if len(p) else float("nan"),
        "pred_max": float(np.max(p)) if len(p) else float("nan"),
        "pred_nan_count": int(np.isnan(p).sum()),
        "pred_inf_count": int(np.isinf(p).sum()),
        "pred_out_of_range_count": int(((p < 0.0) | (p > 1.0)).sum()) if len(p) else 0,
    }


def build_final_ratings(
    rating_period: str,
    ratings: dict[int, float],
    rds: dict[int, float],
    games_played: dict[int, int],
    wins: dict[int, int],
    losses: dict[int, int],
    last_calendar_period_index: dict[int, int],
    player_names: dict[int, str],
) -> pd.DataFrame:
    """Build a final player state dataframe for one rating-period setting."""

    final_rows = []
    for player in sorted(ratings):
        final_rows.append(
            {
                "rating_period": rating_period,
                "rd_inflation_variant": RD_INFLATION_VARIANT,
                "player_id": player,
                "player_name": player_names.get(player, pd.NA),
                "rating": ratings[player],
                "rd": rds[player],
                "games_played": games_played[player],
                "wins": wins[player],
                "losses": losses[player],
                "last_calendar_period_index": last_calendar_period_index.get(player, pd.NA),
            }
        )
    final_ratings_df = pd.DataFrame(final_rows)
    final_ratings_df["rank_by_rating"] = final_ratings_df["rating"].rank(
        method="min", ascending=False
    ).astype(int)
    return final_ratings_df.sort_values(["rank_by_rating", "player_id"]).reset_index(drop=True)


def summarise_metrics_and_details(
    rating_period: str,
    c_value: float,
    number_of_games: int,
    number_of_periods: int,
    update_operations: int,
    runtime_seconds: float,
    predictions_df: pd.DataFrame,
    final_ratings_df: pd.DataFrame,
    detail_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create metrics and runtime detail dictionaries for one setting."""

    metrics = evaluate_winner_predictions(predictions_df)
    rds_array = final_ratings_df["rd"].astype(float).to_numpy()
    metrics.update(
        {
            "rating_period": rating_period,
            "rd_inflation_variant": RD_INFLATION_VARIANT,
            "c_value": c_value,
            "number_of_rating_periods": number_of_periods,
            "number_of_games": number_of_games,
            "number_of_update_operations": update_operations,
            "mean_games_per_rating_period": float(number_of_games / number_of_periods),
            "mean_players_updated_per_period": float(update_operations / number_of_periods),
            "runtime_seconds": runtime_seconds,
            "final_players": int(len(final_ratings_df)),
            "final_mean_rd": float(np.mean(rds_array)),
            "final_median_rd": float(np.median(rds_array)),
            "final_min_rd": float(np.min(rds_array)),
            "final_max_rd": float(np.max(rds_array)),
            "players_at_min_rd": int(np.sum(rds_array <= MIN_RD + 1e-9)),
            "players_near_max_rd": int(np.sum(rds_array >= NEAR_MAX_RD_THRESHOLD)),
            "notes": "",
        }
    )

    details = {
        "rating_period": rating_period,
        "periods": number_of_periods,
        "games": number_of_games,
        "update_operations": update_operations,
        "mean_games_per_period": float(number_of_games / number_of_periods),
        "median_games_per_period": np.nan,
        "max_games_per_period": np.nan,
        "mean_players_updated_per_period": float(update_operations / number_of_periods),
        "median_players_updated_per_period": np.nan,
        "max_players_updated_per_period": np.nan,
        "runtime_seconds": runtime_seconds,
    }
    if detail_overrides:
        details.update(detail_overrides)
    return metrics, details


def run_match_by_match_fast(matches: pd.DataFrame, c_value: float) -> dict[str, Any]:
    """Run match-by-match Glicko directly without grouping 456k one-match periods."""

    rating_period = "match_by_match"
    print(f"\nRunning rating period: {rating_period}")
    start = time.perf_counter()

    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_calendar_period_index: dict[int, int] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}
    predictions_2025: list[dict[str, Any]] = []

    cols = [
        "fcode",
        "year",
        "event",
        "event_key",
        "calendar_period_index",
        "chronological_order",
        "winner",
        "loser",
        "winner_name",
        "loser_name",
    ]

    for row in matches[cols].itertuples(index=False):
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        calendar_period_index = int(row.calendar_period_index)
        rating_period_key = f"match_{int(row.chronological_order)}"

        update_player_name(player_names, winner, row.winner_name)
        update_player_name(player_names, loser, row.loser_name)

        if winner not in ratings:
            ratings[winner] = DEFAULT_RATING
            rds[winner] = DEFAULT_RD
        else:
            elapsed = calendar_period_index - last_calendar_period_index[winner]
            rds[winner] = inflate_rd_for_inactivity(rds[winner], elapsed, c_value)

        if loser not in ratings:
            ratings[loser] = DEFAULT_RATING
            rds[loser] = DEFAULT_RD
        else:
            elapsed = calendar_period_index - last_calendar_period_index[loser]
            rds[loser] = inflate_rd_for_inactivity(rds[loser], elapsed, c_value)

        winner_rating_before = ratings[winner]
        winner_rd_before = rds[winner]
        loser_rating_before = ratings[loser]
        loser_rd_before = rds[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, loser_rd_before)

        winner_rating_after, winner_rd_after = update_player_glicko(
            winner_rating_before,
            winner_rd_before,
            [loser_rating_before],
            [loser_rd_before],
            [1.0],
        )
        loser_rating_after, loser_rd_after = update_player_glicko(
            loser_rating_before,
            loser_rd_before,
            [winner_rating_before],
            [winner_rd_before],
            [0.0],
        )

        ratings[winner] = winner_rating_after
        rds[winner] = winner_rd_after
        ratings[loser] = loser_rating_after
        rds[loser] = loser_rd_after
        last_calendar_period_index[winner] = calendar_period_index
        last_calendar_period_index[loser] = calendar_period_index

        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

        if int(row.year) == END_YEAR:
            predictions_2025.append(
                {
                    "rating_period": rating_period,
                    "rd_inflation_variant": RD_INFLATION_VARIANT,
                    "year": int(row.year),
                    "game_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "event_id": format_code_value(row.event),
                    "event_key": row.event_key,
                    "winner": winner,
                    "loser": loser,
                    "pred_winner_win": pred_winner_win,
                    "pre_rating_winner": winner_rating_before,
                    "pre_rd_winner": winner_rd_before,
                    "pre_rating_loser": loser_rating_before,
                    "pre_rd_loser": loser_rd_before,
                    "calendar_period_index": calendar_period_index,
                    "rating_period_key": rating_period_key,
                }
            )

    runtime_seconds = time.perf_counter() - start
    predictions_df = pd.DataFrame(predictions_2025)
    final_ratings_df = build_final_ratings(
        rating_period,
        ratings,
        rds,
        games_played,
        wins,
        losses,
        last_calendar_period_index,
        player_names,
    )
    number_of_games = int(len(matches))
    number_of_periods = number_of_games
    update_operations = number_of_games * 2
    metrics, details = summarise_metrics_and_details(
        rating_period,
        c_value,
        number_of_games,
        number_of_periods,
        update_operations,
        runtime_seconds,
        predictions_df,
        final_ratings_df,
        {
            "median_games_per_period": 1.0,
            "max_games_per_period": 1,
            "median_players_updated_per_period": 2.0,
            "max_players_updated_per_period": 2,
        },
    )

    print(
        f"  2025 games={metrics['evaluation_games_2025']:,}, "
        f"log_loss={metrics['log_loss']:.6f}, "
        f"brier={metrics['brier']:.6f}, accuracy={metrics['accuracy']:.6f}"
    )
    print(
        f"  runtime={runtime_seconds:.1f}s, periods={number_of_periods:,}, "
        f"updates={update_operations:,}, median RD={metrics['final_median_rd']:.3f}"
    )
    return {
        "metrics": metrics,
        "details": details,
        "predictions_2025": predictions_df,
        "final_ratings": final_ratings_df,
    }


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def run_glicko_rating_period(
    matches: pd.DataFrame,
    rating_period: str,
    c_value: float,
) -> dict[str, Any]:
    """Run Glicko with one rating-period setting and fixed low inflation."""

    if rating_period == "match_by_match":
        return run_match_by_match_fast(matches, c_value)

    print(f"\nRunning rating period: {rating_period}")
    start = time.perf_counter()
    data = assign_rating_period_key(matches, rating_period)
    number_of_periods = int(data["rating_period_key"].nunique())
    print(f"  Number of rating periods: {number_of_periods:,}")

    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_calendar_period_index: dict[int, int] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}

    predictions_2025: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    update_operations = 0

    cols = [
        "fcode",
        "year",
        "event",
        "event_key",
        "eventname",
        "event_date_raw",
        "event_date_parsed",
        "event_order_date",
        "event_date_ordering_method",
        "calendar_period_index",
        "rating_period_key",
        "rating_period_order",
        "winner",
        "loser",
        "winner_name",
        "loser_name",
    ]

    for period_key, period_df in data[cols].groupby("rating_period_key", sort=False):
        period_calendar_index = int(period_df["calendar_period_index"].min())
        period_order = int(period_df["rating_period_order"].iloc[0])
        winners = period_df["winner"].astype(int).tolist()
        losers = period_df["loser"].astype(int).tolist()
        active_players = set(winners) | set(losers)

        for player in active_players:
            if player not in ratings:
                ratings[player] = DEFAULT_RATING
                rds[player] = DEFAULT_RD
            else:
                elapsed = period_calendar_index - last_calendar_period_index[player]
                rds[player] = inflate_rd_for_inactivity(rds[player], elapsed, c_value)

        period_start_ratings = {player: ratings[player] for player in active_players}
        period_start_rds = {player: rds[player] for player in active_players}
        player_period_games: dict[int, dict[str, list[float]]] = {
            player: {"opponent_ratings": [], "opponent_rds": [], "scores": []}
            for player in active_players
        }
        games_per_player: defaultdict[int, int] = defaultdict(int)

        for row in period_df.itertuples(index=False):
            winner = player_code(row.winner)
            loser = player_code(row.loser)
            update_player_name(player_names, winner, row.winner_name)
            update_player_name(player_names, loser, row.loser_name)

            winner_rating_before = period_start_ratings[winner]
            winner_rd_before = period_start_rds[winner]
            loser_rating_before = period_start_ratings[loser]
            loser_rd_before = period_start_rds[loser]
            pred_winner_win = expected_score(
                winner_rating_before,
                loser_rating_before,
                loser_rd_before,
            )

            player_period_games[winner]["opponent_ratings"].append(loser_rating_before)
            player_period_games[winner]["opponent_rds"].append(loser_rd_before)
            player_period_games[winner]["scores"].append(1.0)
            player_period_games[loser]["opponent_ratings"].append(winner_rating_before)
            player_period_games[loser]["opponent_rds"].append(winner_rd_before)
            player_period_games[loser]["scores"].append(0.0)

            games_played[winner] += 1
            games_played[loser] += 1
            wins[winner] += 1
            losses[loser] += 1
            games_per_player[winner] += 1
            games_per_player[loser] += 1

            if int(row.year) == END_YEAR:
                predictions_2025.append(
                    {
                        "rating_period": rating_period,
                        "rd_inflation_variant": RD_INFLATION_VARIANT,
                        "year": int(row.year),
                        "game_id": int(row.fcode),
                        "fcode": int(row.fcode),
                        "event_id": format_code_value(row.event),
                        "event_key": row.event_key,
                        "winner": winner,
                        "loser": loser,
                        "pred_winner_win": pred_winner_win,
                        "pre_rating_winner": winner_rating_before,
                        "pre_rd_winner": winner_rd_before,
                        "pre_rating_loser": loser_rating_before,
                        "pre_rd_loser": loser_rd_before,
                        "calendar_period_index": period_calendar_index,
                        "rating_period_key": period_key,
                    }
                )

        rating_changes = []
        rd_changes = []
        for player, games in player_period_games.items():
            new_rating, new_rd = update_player_glicko(
                period_start_ratings[player],
                period_start_rds[player],
                games["opponent_ratings"],
                games["opponent_rds"],
                games["scores"],
            )
            rating_changes.append(abs(new_rating - period_start_ratings[player]))
            rd_changes.append(new_rd - period_start_rds[player])
            ratings[player] = new_rating
            rds[player] = new_rd
            last_calendar_period_index[player] = period_calendar_index

        update_operations += len(player_period_games)
        games_per_player_values = np.asarray(list(games_per_player.values()), dtype=float)
        period_rows.append(
            {
                "rating_period": rating_period,
                "period_key": period_key,
                "period_order": period_order,
                "calendar_period_index": period_calendar_index,
                "games": int(len(period_df)),
                "players_updated": int(len(player_period_games)),
                "mean_games_per_player_in_period": float(games_per_player_values.mean()),
                "max_games_by_one_player_in_period": int(games_per_player_values.max()),
                "mean_abs_rating_change_in_period": float(np.mean(rating_changes)),
                "median_abs_rating_change_in_period": float(np.median(rating_changes)),
                "p90_abs_rating_change_in_period": percentile(np.asarray(rating_changes), 90),
                "mean_rd_change_in_period": float(np.mean(rd_changes)),
                "median_rd_change_in_period": float(np.median(rd_changes)),
            }
        )

    runtime_seconds = time.perf_counter() - start
    predictions_df = pd.DataFrame(predictions_2025)
    final_rows = []
    for player in sorted(ratings):
        final_rows.append(
            {
                "rating_period": rating_period,
                "rd_inflation_variant": RD_INFLATION_VARIANT,
                "player_id": player,
                "player_name": player_names.get(player, pd.NA),
                "rating": ratings[player],
                "rd": rds[player],
                "games_played": games_played[player],
                "wins": wins[player],
                "losses": losses[player],
                "last_calendar_period_index": last_calendar_period_index.get(player, pd.NA),
            }
        )
    final_ratings_df = pd.DataFrame(final_rows)
    final_ratings_df["rank_by_rating"] = final_ratings_df["rating"].rank(
        method="min", ascending=False
    ).astype(int)
    final_ratings_df = final_ratings_df.sort_values(["rank_by_rating", "player_id"]).reset_index(
        drop=True
    )

    period_summary_df = pd.DataFrame(period_rows)
    metrics = evaluate_winner_predictions(predictions_df)
    rds_array = final_ratings_df["rd"].astype(float).to_numpy()
    metrics.update(
        {
            "rating_period": rating_period,
            "rd_inflation_variant": RD_INFLATION_VARIANT,
            "c_value": c_value,
            "number_of_rating_periods": number_of_periods,
            "number_of_games": int(len(data)),
            "number_of_update_operations": int(update_operations),
            "mean_games_per_rating_period": float(len(data) / number_of_periods),
            "mean_players_updated_per_period": float(update_operations / number_of_periods),
            "runtime_seconds": runtime_seconds,
            "final_players": int(len(final_ratings_df)),
            "final_mean_rd": float(np.mean(rds_array)),
            "final_median_rd": float(np.median(rds_array)),
            "final_min_rd": float(np.min(rds_array)),
            "final_max_rd": float(np.max(rds_array)),
            "players_at_min_rd": int(np.sum(rds_array <= MIN_RD + 1e-9)),
            "players_near_max_rd": int(np.sum(rds_array >= NEAR_MAX_RD_THRESHOLD)),
            "notes": "",
        }
    )

    details = {
        "rating_period": rating_period,
        "periods": number_of_periods,
        "games": int(len(data)),
        "update_operations": int(update_operations),
        "mean_games_per_period": float(period_summary_df["games"].mean()),
        "median_games_per_period": float(period_summary_df["games"].median()),
        "max_games_per_period": int(period_summary_df["games"].max()),
        "mean_players_updated_per_period": float(period_summary_df["players_updated"].mean()),
        "median_players_updated_per_period": float(period_summary_df["players_updated"].median()),
        "max_players_updated_per_period": int(period_summary_df["players_updated"].max()),
        "runtime_seconds": runtime_seconds,
    }

    print(
        f"  2025 games={metrics['evaluation_games_2025']:,}, "
        f"log_loss={metrics['log_loss']:.6f}, "
        f"brier={metrics['brier']:.6f}, accuracy={metrics['accuracy']:.6f}"
    )
    print(
        f"  runtime={runtime_seconds:.1f}s, periods={number_of_periods:,}, "
        f"updates={update_operations:,}, median RD={metrics['final_median_rd']:.3f}"
    )
    return {
        "metrics": metrics,
        "details": details,
        "predictions_2025": predictions_df,
        "final_ratings": final_ratings_df,
        "period_summary": period_summary_df,
    }


def top_overlap(merged: pd.DataFrame, n: int) -> float:
    """Return top-N overlap between reference and comparison ratings within a subset."""

    if merged.empty:
        return float("nan")
    k = min(n, len(merged))
    if k == 0:
        return float("nan")
    ref_top = set(merged.sort_values("rating_ref", ascending=False).head(k)["player_id"])
    comp_top = set(merged.sort_values("rating_comp", ascending=False).head(k)["player_id"])
    return float(len(ref_top & comp_top) / k)


def make_similarity_table(final_ratings: pd.DataFrame, active_2025_counts: dict[int, int]) -> pd.DataFrame:
    """Compare final rating lists against match-by-match as reference."""

    ref = final_ratings.loc[final_ratings["rating_period"] == "match_by_match"].copy()
    ref = ref[["player_id", "rating", "rank_by_rating", "games_played"]].rename(
        columns={"rating": "rating_ref", "rank_by_rating": "rank_ref", "games_played": "total_games"}
    )

    rows = []
    for rating_period in ["event_level", "monthly", "yearly"]:
        comp = final_ratings.loc[final_ratings["rating_period"] == rating_period].copy()
        comp = comp[["player_id", "rating", "rank_by_rating"]].rename(
            columns={"rating": "rating_comp", "rank_by_rating": "rank_comp"}
        )
        merged = ref.merge(comp, on="player_id", how="inner")
        merged["active_2025_games"] = (
            merged["player_id"].map(active_2025_counts).fillna(0).astype(int)
        )
        merged["abs_rank_diff"] = (merged["rank_ref"] - merged["rank_comp"]).abs()
        merged["abs_rating_diff"] = (merged["rating_ref"] - merged["rating_comp"]).abs()

        for group_name, min_active_2025_games, min_total_games in SIMILARITY_GROUPS:
            subset = merged.copy()
            if min_active_2025_games > 0:
                subset = subset[subset["active_2025_games"] >= min_active_2025_games]
            if min_total_games > 0:
                subset = subset[subset["total_games"] >= min_total_games]

            players = int(len(subset))
            if players >= 2:
                pearson = subset["rating_ref"].corr(subset["rating_comp"], method="pearson")
                spearman = subset["rating_ref"].corr(subset["rating_comp"], method="spearman")
                ref_centered = subset["rating_ref"] - subset["rating_ref"].mean()
                comp_centered = subset["rating_comp"] - subset["rating_comp"].mean()
                mean_abs_centered_rating_diff = float((ref_centered - comp_centered).abs().mean())
            else:
                pearson = np.nan
                spearman = np.nan
                mean_abs_centered_rating_diff = np.nan

            rows.append(
                {
                    "comparison": f"{rating_period}_vs_match_by_match",
                    "reference_rating_period": "match_by_match",
                    "comparison_rating_period": rating_period,
                    "group": group_name,
                    "players": players,
                    "spearman": float(spearman) if not pd.isna(spearman) else np.nan,
                    "pearson": float(pearson) if not pd.isna(pearson) else np.nan,
                    "top50_overlap": top_overlap(subset, 50),
                    "top100_overlap": top_overlap(subset, 100),
                    "mean_abs_rank_diff": float(subset["abs_rank_diff"].mean()) if players else np.nan,
                    "mean_abs_rating_diff": float(subset["abs_rating_diff"].mean()) if players else np.nan,
                    "mean_abs_centered_rating_diff": mean_abs_centered_rating_diff,
                }
            )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact markdown table."""

    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_bar_plot(metrics: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    """Save a bar chart by rating period."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = metrics.sort_values("rating_period")
    ax.bar(data["rating_period"], data[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Rating period")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_runtime_vs_brier_plot(metrics: pd.DataFrame) -> None:
    """Save runtime/Brier trade-off scatter plot."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(metrics["runtime_seconds"], metrics["brier"], color="#F58518", s=70)
    for row in metrics.itertuples(index=False):
        ax.annotate(row.rating_period, (row.runtime_seconds, row.brier), xytext=(5, 4), textcoords="offset points")
    ax.set_title("Prediction-runtime trade-off")
    ax.set_xlabel("Runtime seconds")
    ax.set_ylabel("2025 Brier score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RUNTIME_VS_BRIER_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_similarity_plot(similarity: pd.DataFrame) -> None:
    """Save similarity plot for active 2025 players."""

    data = similarity.loc[similarity["group"] == "active_2025_games_ge5"].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(data))
    width = 0.36
    ax.bar(x - width / 2, data["spearman"], width=width, label="Spearman", color="#4C78A8")
    ax.bar(x + width / 2, data["top100_overlap"], width=width, label="Top100 overlap", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(data["comparison_rating_period"], rotation=20)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Active 2025 rating-list similarity vs match-by-match")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SIMILARITY_PLOT_PATH, dpi=160)
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    similarity: pd.DataFrame,
    runtime_details: pd.DataFrame,
    c_value: float,
    inactivity_unit: str,
    c_source: str,
    warnings: list[str],
    dataset_path: Path,
) -> None:
    """Write the meeting-ready markdown summary."""

    best_brier = metrics.sort_values(["brier", "log_loss"]).iloc[0]
    active_similarity = similarity.loc[similarity["group"] == "active_2025_games_ge5"].copy()
    total_high_similarity = similarity.loc[similarity["group"] == "total_games_ge100"].copy()

    if best_brier["rating_period"] == "match_by_match":
        recommendation = (
            "match_by_match is the candidate main Glicko rating-period choice in this run: "
            "it has the best 2025 prediction metrics, and the runtime is manageable."
        )
    elif best_brier["rating_period"] in {"event_level", "monthly"}:
        recommendation = (
            f"{best_brier['rating_period']} is computationally defensible in this run because "
            "it gives the best Brier/log-loss trade-off under fixed low_inflation."
        )
    else:
        recommendation = (
            "yearly is best on one metric in this run, but it should be treated cautiously because "
            "yearly periods are very coarse and less realistic as a main Glicko candidate."
        )

    warning_lines = ["- None"] if not warnings else [f"- {warning}" for warning in warnings]

    lines = [
        "# Meeting 5 Glicko Rating-Period Runtime Comparison",
        "",
        "## Purpose",
        "",
        (
            "This experiment asks how the Glicko rating-period choice affects prediction "
            "performance, computation time, and active-player final rating-list similarity."
        ),
        "",
        "## Experimental Design",
        "",
        f"- Dataset: `{dataset_path.name}` covering {START_YEAR}-{END_YEAR}.",
        "- Evaluation set: 2025 games.",
        "- Glicko formula: existing validated Glicko-1 core.",
        f"- Initial rating/RD: {DEFAULT_RATING:.0f} / {DEFAULT_RD:.0f}.",
        f"- RD bounds: MIN_RD={MIN_RD:.0f}, MAX_RD={MAX_RD:.0f}.",
        "- Prediction rule: prediction is recorded before rating/RD update.",
        "- The only model-design change is the rating-period setting.",
        "",
        "## Rating Periods Compared",
        "",
        "- `match_by_match`: one rating period per match.",
        "- `event_level`: one rating period per year-event.",
        "- `monthly`: one rating period per calendar month.",
        "- `yearly`: one rating period per calendar year.",
        "",
        "## Inactivity RD Inflation Setting",
        "",
        (
            f"All rating-period settings use `{RD_INFLATION_VARIANT}` with "
            f"`c={c_value:.6f}` and `{inactivity_unit}`-based inactivity. "
            f"The C value was {c_source}."
        ),
        "",
        (
            "Rating period and inactivity period are separate concepts here: rating period "
            "controls how games are grouped for Glicko batch updates; inactivity period measures "
            "how long a player has been absent before their next rating period."
        ),
        "",
        "## Runtime Measurement",
        "",
        markdown_table(
            runtime_details.sort_values("rating_period"),
            [
                "rating_period",
                "periods",
                "games",
                "update_operations",
                "mean_games_per_period",
                "mean_players_updated_per_period",
                "runtime_seconds",
            ],
        ),
        "",
        "## Main 2025 Prediction Results",
        "",
        markdown_table(
            metrics.sort_values("rating_period"),
            [
                "rating_period",
                "evaluation_games_2025",
                "log_loss",
                "brier",
                "accuracy",
                "runtime_seconds",
                "final_median_rd",
                "players_at_min_rd",
            ],
        ),
        "",
        "## Runtime and Update-Operation Results",
        "",
        (
            "Match-by-match has the largest number of rating periods and update operations. "
            "Event/month/year periods combine many games into fewer update blocks, so they can "
            "be computationally lighter, although their predictions use coarser period-start states."
        ),
        "",
        "## Active-Player Rating-List Similarity",
        "",
        "Active 2025 players with at least 5 games:",
        "",
        markdown_table(
            active_similarity.sort_values("comparison_rating_period"),
            [
                "comparison",
                "players",
                "spearman",
                "pearson",
                "top50_overlap",
                "top100_overlap",
                "mean_abs_rank_diff",
                "mean_abs_centered_rating_diff",
            ],
        ),
        "",
        "Players with at least 100 total games:",
        "",
        markdown_table(
            total_high_similarity.sort_values("comparison_rating_period"),
            [
                "comparison",
                "players",
                "spearman",
                "pearson",
                "top50_overlap",
                "top100_overlap",
                "mean_abs_rank_diff",
                "mean_abs_centered_rating_diff",
            ],
        ),
        "",
        "## Interpretation For Supervisor",
        "",
        (
            "This directly answers the runtime/rating-period question. Shorter rating periods "
            "usually require more period updates, while coarser periods reuse the same period-start "
            "ratings for more games. The relevant decision is therefore a prediction-runtime "
            "trade-off, not only which method is theoretically neatest."
        ),
        "",
        "## Recommended Rating-Period Choice",
        "",
        recommendation,
        "",
        "Use cautious wording: this is a candidate main rating-period choice and a sensitivity result, not a final best model.",
        "",
        "## Remaining Limitations",
        "",
        "- This is still Glicko-1, not Glicko-2.",
        "- It does not compare Glicko against Elo.",
        "- It does not implement adaptive-K Elo.",
        "- Runtime depends on this local implementation and machine, so relative differences are more important than exact seconds.",
        "- Yearly is included as a coarse diagnostic, not necessarily a realistic main candidate.",
        "",
        "## Warnings",
        "",
        *warning_lines,
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the rating-period runtime comparison."""

    total_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    c_value, inactivity_unit, target_periods, c_source = get_low_inflation_setting()
    matches, detected_inactivity_unit, dataset_path = load_matches()
    if inactivity_unit != detected_inactivity_unit:
        print(
            f"WARNING: low_inflation output says inactivity_unit={inactivity_unit}, "
            f"but dataset parsing detected {detected_inactivity_unit}. Using detected unit."
        )
        inactivity_unit = detected_inactivity_unit

    print(f"Fixed RD inflation variant: {RD_INFLATION_VARIANT}")
    print(f"c_value: {c_value:.6f}")
    print(f"target_periods: {target_periods}")
    print(f"c source: {c_source}")
    print("Rating periods:")
    for setting in RATING_PERIOD_SETTINGS:
        print(f"  {setting['rating_period']}")

    event_order_summary = matches["event_date_ordering_method"].value_counts(dropna=False).to_dict()
    period_source_summary = matches["calendar_period_source"].value_counts(dropna=False).to_dict()
    print(f"Event date ordering summary: {event_order_summary}")
    print(f"Calendar inactivity period source summary: {period_source_summary}")

    all_metrics = []
    all_details = []
    all_predictions = []
    all_final_ratings = []

    for setting in RATING_PERIOD_SETTINGS:
        result = run_glicko_rating_period(matches, setting["rating_period"], c_value)
        all_metrics.append(result["metrics"])
        all_details.append(result["details"])
        all_predictions.append(result["predictions_2025"])
        all_final_ratings.append(result["final_ratings"])

    metrics_df = pd.DataFrame(all_metrics)
    runtime_details_df = pd.DataFrame(all_details)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    final_ratings_df = pd.concat(all_final_ratings, ignore_index=True)

    active_2025_counts = (
        pd.concat(
            [
                matches.loc[matches["year"] == END_YEAR, "winner"],
                matches.loc[matches["year"] == END_YEAR, "loser"],
            ]
        )
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )
    similarity_df = make_similarity_table(final_ratings_df, active_2025_counts)

    warnings = []
    if not (metrics_df["evaluation_games_2025"] == EXPECTED_2025_GAMES).all():
        warnings.append("At least one rating period does not have 11,379 evaluation games.")
    if int(metrics_df["pred_out_of_range_count"].sum()) > 0:
        warnings.append("Some prediction probabilities are outside [0, 1].")
    if int(metrics_df["pred_nan_count"].sum()) > 0 or int(metrics_df["pred_inf_count"].sum()) > 0:
        warnings.append("Some prediction probabilities are NaN or infinite.")

    match_row = metrics_df.loc[metrics_df["rating_period"] == "match_by_match"].iloc[0]
    low_deltas = {
        key: abs(float(match_row[key]) - value)
        for key, value in LOW_INFLATION_EXPECTED.items()
    }
    if not (
        low_deltas["log_loss"] < 1e-4
        and low_deltas["brier"] < 1e-4
        and low_deltas["accuracy"] < 1e-4
    ):
        warnings.append(f"match_by_match + low_inflation differs from previous step: {low_deltas}")
    else:
        metrics_df.loc[
            metrics_df["rating_period"] == "match_by_match", "notes"
        ] = "Matches previous low_inflation metrics within tolerance."

    ordered_metric_cols = [
        "rating_period",
        "rd_inflation_variant",
        "c_value",
        "inactivity_unit",
        "evaluation_games_2025",
        "log_loss",
        "brier",
        "accuracy",
        "runtime_seconds",
        "number_of_rating_periods",
        "number_of_games",
        "number_of_update_operations",
        "mean_games_per_rating_period",
        "mean_players_updated_per_period",
        "final_players",
        "final_mean_rd",
        "final_median_rd",
        "final_min_rd",
        "final_max_rd",
        "players_at_min_rd",
        "players_near_max_rd",
        "notes",
    ]
    metrics_df["rd_inflation_variant"] = RD_INFLATION_VARIANT
    metrics_df["inactivity_unit"] = inactivity_unit
    for col in ["notes"]:
        if col not in metrics_df.columns:
            metrics_df[col] = ""

    metrics_df[ordered_metric_cols].to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    similarity_df.to_csv(SIMILARITY_PATH, index=False, encoding="utf-8-sig")
    runtime_details_df.to_csv(RUNTIME_DETAILS_PATH, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(PREDICTIONS_2025_PATH, index=False, encoding="utf-8-sig")
    final_ratings_df.to_csv(FINAL_RATINGS_PATH, index=False, encoding="utf-8-sig")

    save_bar_plot(metrics_df, "brier", BRIER_PLOT_PATH, "2025 Brier score by rating period", "Brier score")
    save_bar_plot(metrics_df, "log_loss", LOGLOSS_PLOT_PATH, "2025 log loss by rating period", "Log loss")
    save_bar_plot(metrics_df, "runtime_seconds", RUNTIME_PLOT_PATH, "Runtime by rating period", "Runtime seconds")
    save_runtime_vs_brier_plot(metrics_df)
    save_similarity_plot(similarity_df)

    write_summary(
        metrics_df,
        similarity_df,
        runtime_details_df,
        c_value,
        inactivity_unit,
        c_source,
        warnings,
        dataset_path,
    )

    print("\nConsistency checks:")
    print(metrics_df[["rating_period", "evaluation_games_2025", "log_loss", "brier", "accuracy"]].to_string(index=False))
    print(f"match_by_match low_inflation deltas: {low_deltas}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("Warnings: none")

    print("\nOutput files:")
    for path in [
        METRICS_PATH,
        SIMILARITY_PATH,
        RUNTIME_DETAILS_PATH,
        PREDICTIONS_2025_PATH,
        FINAL_RATINGS_PATH,
        SUMMARY_MD_PATH,
        BRIER_PLOT_PATH,
        LOGLOSS_PLOT_PATH,
        RUNTIME_PLOT_PATH,
        RUNTIME_VS_BRIER_PLOT_PATH,
        SIMILARITY_PLOT_PATH,
    ]:
        print(f"  {path}")

    print("No Elo-vs-Glicko final comparison was run.")
    print("No adaptive-K Elo was run.")
    print(f"Total runtime: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
