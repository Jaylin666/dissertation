"""This script applies the validated Glicko-1 core formula to the full croquet match dataset using a match-by-match rating-period assumption.

This is the first full-data Glicko-1 baseline. It uses the already validated
Glicko core functions, treats each croquet match as one rating period, records
pre-match predictions, evaluates 2025 test games, and outputs final ratings
with RD. It does not compare against Elo, tune Glicko parameters, implement
Glicko-2, or test event/monthly/yearly rating periods.
"""

from __future__ import annotations

from collections import defaultdict
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


SETTING_NAME = "glicko1_match_by_match_rd350_c0"
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
MIN_RD = 30.0
MAX_RD = 350.0
C = 0.0
START_YEAR = 1985
END_YEAR = 2025
EXPECTED_2025_GAMES = 11_379
EPS = 1e-15

SETTINGS = [
    {
        "setting_name": SETTING_NAME,
        "default_rating": DEFAULT_RATING,
        "default_rd": DEFAULT_RD,
        "min_rd": MIN_RD,
        "max_rd": MAX_RD,
        "c": C,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
    }
]

MATCHES_PATH = PROJECT_ROOT / "outputs" / "elo_optimization" / "matches_1985_2025_checked.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "glicko_implementation"

PREDICTIONS_PATH = OUTPUT_DIR / "glicko_mbm_predictions_1985_2025.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "glicko_mbm_final_ratings_1985_2025.csv"
METRICS_PATH = OUTPUT_DIR / "glicko_mbm_metrics_2025.csv"
CALIBRATION_PATH = OUTPUT_DIR / "glicko_mbm_calibration_2025.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "glicko_mbm_confidence_2025.csv"
RD_SUMMARY_PATH = OUTPUT_DIR / "glicko_mbm_rd_summary.csv"
YEARLY_RD_SUMMARY_PATH = OUTPUT_DIR / "glicko_mbm_yearly_rd_summary.csv"
DATE_ORDERING_SUMMARY_PATH = OUTPUT_DIR / "glicko_mbm_date_ordering_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "glicko_mbm_baseline_summary.md"

REQUIRED_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]


def format_code_value(value: Any) -> str:
    """Return a stable string for event keys and reporting."""

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
    """Convert a player code from the dataset into an integer key."""

    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    """Store the first non-missing name observed for a player."""

    if code in player_names:
        return
    if pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date and event_date_ordering_method without changing raw date columns."""

    matches = matches.copy()
    if "event_order_date" in matches.columns and "event_date_ordering_method" in matches.columns:
        matches["event_order_date"] = pd.to_datetime(matches["event_order_date"], errors="coerce")
        return matches

    matches["event_date_parsed_for_ordering"] = pd.to_datetime(
        matches["event_date_parsed"], errors="coerce"
    )
    matches["event_order_date"] = matches["event_date_parsed_for_ordering"]
    matches["event_date_ordering_method"] = np.where(
        matches["event_date_parsed_for_ordering"].notna(),
        "parsed_full_date",
        "fallback_no_date",
    )

    missing_parsed = matches["event_date_parsed_for_ordering"].isna()
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

    matches = matches.drop(columns=["event_date_parsed_for_ordering"])
    return matches


def make_date_ordering_summary(matches: pd.DataFrame, setting_name: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Summarise how match ordering dates were obtained."""

    summary = (
        matches.groupby("event_date_ordering_method", dropna=False)
        .size()
        .reset_index(name="match_count")
        .sort_values("event_date_ordering_method")
        .reset_index(drop=True)
    )
    summary.insert(0, "end_year", end_year)
    summary.insert(0, "start_year", start_year)
    summary.insert(0, "setting_name", setting_name)
    summary["share_of_matches"] = summary["match_count"] / len(matches)
    return summary


def load_matches() -> pd.DataFrame:
    """Load and chronologically sort the full-history match dataset."""

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
    matches = matches.sort_values(sort_cols, na_position="last").drop(
        columns=["event_order_date_missing"]
    )
    matches = matches.reset_index(drop=True)

    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Loaded dataset: {MATCHES_PATH}")
    print(f"Matches: {len(matches):,}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    print(f"Players: {players:,}")
    return matches


def get_player_state(
    ratings: dict[int, float],
    rds: dict[int, float],
    code: int,
    default_rating: float,
    default_rd: float,
) -> tuple[float, float]:
    """Return current rating and RD for a player, creating defaults if needed."""

    if code not in ratings:
        ratings[code] = default_rating
        rds[code] = default_rd
    return ratings[code], rds[code]


def make_yearly_snapshot(
    year: int,
    active_players: set[int],
    ratings: dict[int, float],
    rds: dict[int, float],
) -> dict[str, Any]:
    """Summarise ratings/RDs for players active in a completed year."""

    active_rds = np.asarray([rds[player] for player in active_players], dtype=float)
    active_ratings = np.asarray([ratings[player] for player in active_players], dtype=float)
    return {
        "year": year,
        "n_active_players_in_year": len(active_players),
        "mean_rd_active_players": float(np.mean(active_rds)),
        "median_rd_active_players": float(np.median(active_rds)),
        "p10_rd_active_players": float(np.quantile(active_rds, 0.10)),
        "p90_rd_active_players": float(np.quantile(active_rds, 0.90)),
        "mean_rating_active_players": float(np.mean(active_ratings)),
        "median_rating_active_players": float(np.median(active_ratings)),
    }


def run_glicko_match_by_match(
    matches: pd.DataFrame,
    start_year: int,
    end_year: int,
    setting_name: str,
    default_rating: float = 1500.0,
    default_rd: float = 350.0,
    min_rd: float = 30.0,
    max_rd: float = 350.0,
    c: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run match-by-match Glicko-1 over a selected period."""

    _ = update_player_glicko
    if (
        default_rating != CORE_DEFAULT_RATING
        or default_rd != CORE_DEFAULT_RD
        or min_rd != CORE_MIN_RD
        or max_rd != CORE_MAX_RD
    ):
        raise ValueError(
            "Current glicko_core.py constants support only rating=1500, RD=350, min RD=30, max RD=350. "
            "Update glicko_core.py before running alternative Glicko settings."
        )
    if c != 0.0:
        raise ValueError("This baseline intentionally supports C=0 only; inactivity RD inflation is out of scope.")

    period_matches = matches[(matches["year"] >= start_year) & (matches["year"] <= end_year)].copy()
    if period_matches.empty:
        raise ValueError(f"No matches found for {start_year}-{end_year}")

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
    yearly_rows: list[dict[str, Any]] = []

    total_processed = 0
    for year, year_matches in period_matches.groupby("year", sort=True):
        year_int = int(year)
        active_players: set[int] = set()
        matches_in_year = len(year_matches)

        for row in year_matches.itertuples(index=False):
            winner = player_code(row.winner)
            loser = player_code(row.loser)
            event = int(row.event)
            code = int(row.code)
            fcode = int(row.fcode)
            event_key = f"{year_int}_{format_code_value(event)}"

            update_player_name(player_names, winner, getattr(row, "winner_name", pd.NA))
            update_player_name(player_names, loser, getattr(row, "loser_name", pd.NA))

            winner_rating_before, winner_rd_before = get_player_state(
                ratings, rds, winner, default_rating, default_rd
            )
            loser_rating_before, loser_rd_before = get_player_state(
                ratings, rds, loser, default_rating, default_rd
            )

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
            pred_a_win = expected_score(
                player_a_rating_before,
                player_b_rating_before,
                player_b_rd_before,
            )
            pred_winner_win = pred_a_win if winner == player_a else 1.0 - pred_a_win

            update = update_two_players_single_game(
                winner_rating_before,
                winner_rd_before,
                loser_rating_before,
                loser_rd_before,
                1.0,
            )
            winner_rating_after = update.player1_rating_after
            winner_rd_after = update.player1_rd_after
            loser_rating_after = update.player2_rating_after
            loser_rd_after = update.player2_rd_after

            ratings[winner] = winner_rating_after
            rds[winner] = winner_rd_after
            ratings[loser] = loser_rating_after
            rds[loser] = loser_rd_after

            if player_a == winner:
                player_a_rating_after = winner_rating_after
                player_a_rd_after = winner_rd_after
                player_b_rating_after = loser_rating_after
                player_b_rd_after = loser_rd_after
            else:
                player_a_rating_after = loser_rating_after
                player_a_rd_after = loser_rd_after
                player_b_rating_after = winner_rating_after
                player_b_rd_after = winner_rd_after

            games_played[winner] += 1
            games_played[loser] += 1
            wins[winner] += 1
            losses[loser] += 1
            if year_int == 2025:
                games_2025[winner] += 1
                games_2025[loser] += 1
                wins_2025[winner] += 1
                losses_2025[loser] += 1

            active_players.add(winner)
            active_players.add(loser)
            total_processed += 1

            prediction_rows.append(
                {
                    "setting_name": setting_name,
                    "start_year": start_year,
                    "end_year": end_year,
                    "fcode": fcode,
                    "code": code,
                    "year": year_int,
                    "event": event,
                    "event_key": event_key,
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
                    "winner_rating_before": winner_rating_before,
                    "winner_rd_before": winner_rd_before,
                    "loser_rating_before": loser_rating_before,
                    "loser_rd_before": loser_rd_before,
                    "player_a_rating_before": player_a_rating_before,
                    "player_a_rd_before": player_a_rd_before,
                    "player_b_rating_before": player_b_rating_before,
                    "player_b_rd_before": player_b_rd_before,
                    "winner_rating_after": winner_rating_after,
                    "winner_rd_after": winner_rd_after,
                    "loser_rating_after": loser_rating_after,
                    "loser_rd_after": loser_rd_after,
                    "player_a_rating_after": player_a_rating_after,
                    "player_a_rd_after": player_a_rd_after,
                    "player_b_rating_after": player_b_rating_after,
                    "player_b_rd_after": player_b_rd_after,
                    "winner_rating_change": winner_rating_after - winner_rating_before,
                    "loser_rating_change": loser_rating_after - loser_rating_before,
                    "winner_rd_change": winner_rd_after - winner_rd_before,
                    "loser_rd_change": loser_rd_after - loser_rd_before,
                    "abs_winner_rating_change": abs(winner_rating_after - winner_rating_before),
                    "abs_loser_rating_change": abs(loser_rating_after - loser_rating_before),
                }
            )

        yearly_snapshot = make_yearly_snapshot(year_int, active_players, ratings, rds)
        yearly_snapshot["setting_name"] = setting_name
        yearly_snapshot["start_year"] = start_year
        yearly_snapshot["end_year"] = end_year
        yearly_snapshot["matches_processed_in_year"] = matches_in_year
        yearly_rows.append(yearly_snapshot)
        print(
            f"Year {year_int}: matches={matches_in_year:,}, "
            f"active_players={yearly_snapshot['n_active_players_in_year']:,}, "
            f"median_RD_active={yearly_snapshot['median_rd_active_players']:.3f}"
        )

    predictions = pd.DataFrame(prediction_rows)

    final_rows = []
    for code_value in sorted(ratings):
        final_rows.append(
            {
                "setting_name": setting_name,
                "start_year": start_year,
                "end_year": end_year,
                "player_code": code_value,
                "player_name": player_names.get(code_value, pd.NA),
                "final_rating": ratings[code_value],
                "final_rd": rds[code_value],
                "conservative_rating": ratings[code_value] - 2.0 * rds[code_value],
                "games_played_in_run": games_played.get(code_value, 0),
                "wins_in_run": wins.get(code_value, 0),
                "losses_in_run": losses.get(code_value, 0),
                "games_played_2025": games_2025.get(code_value, 0),
                "wins_2025": wins_2025.get(code_value, 0),
                "losses_2025": losses_2025.get(code_value, 0),
            }
        )

    final_ratings = pd.DataFrame(final_rows)
    final_ratings["final_rank_by_rating"] = (
        final_ratings["final_rating"].rank(method="min", ascending=False).astype(int)
    )
    final_ratings["final_rank_by_conservative_rating"] = (
        final_ratings["conservative_rating"].rank(method="min", ascending=False).astype(int)
    )
    final_ratings = final_ratings.sort_values("final_rank_by_rating").reset_index(drop=True)

    yearly_rd_summary = pd.DataFrame(yearly_rows)
    yearly_rd_summary = yearly_rd_summary[
        [
            "setting_name",
            "start_year",
            "end_year",
            "year",
            "matches_processed_in_year",
            "n_active_players_in_year",
            "mean_rd_active_players",
            "median_rd_active_players",
            "p10_rd_active_players",
            "p90_rd_active_players",
            "mean_rating_active_players",
            "median_rating_active_players",
        ]
    ]

    print(f"Total matches processed for {setting_name}: {total_processed:,}")
    return predictions, final_ratings, yearly_rd_summary


def compute_prediction_metrics(predictions: pd.DataFrame, setting: dict[str, Any]) -> pd.DataFrame:
    """Evaluate 2025 prediction metrics using the Elo pipeline definitions."""

    eval_df = predictions[predictions["year"] == END_YEAR].copy()
    if eval_df.empty:
        raise ValueError("No 2025 games found for evaluation.")

    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)
    clipped = pred.clip(EPS, 1.0 - EPS)

    log_loss = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))
    observed_win_rate = y.mean()

    warning = ""
    if len(eval_df) != EXPECTED_2025_GAMES:
        warning = f"WARNING: expected {EXPECTED_2025_GAMES} 2025 games but found {len(eval_df)}."

    return pd.DataFrame(
        [
            {
                "setting_name": setting["setting_name"],
                "start_year": setting["start_year"],
                "end_year": setting["end_year"],
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
                "warning": warning,
            }
        ]
    )


def make_calibration_table(predictions: pd.DataFrame, setting_name: str) -> pd.DataFrame:
    """Create 2025 calibration table by pred_a_win bins."""

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
    grouped.insert(0, "setting_name", setting_name)
    return grouped


def make_confidence_table(predictions: pd.DataFrame, setting_name: str) -> pd.DataFrame:
    """Create 2025 confidence table."""

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
    grouped.insert(0, "setting_name", setting_name)
    return grouped


def make_rd_summary(final_ratings: pd.DataFrame, setting: dict[str, Any]) -> pd.DataFrame:
    """Summarise final RD distribution."""

    rd = final_ratings["final_rd"].astype(float)
    return pd.DataFrame(
        [
            {
                "setting_name": setting["setting_name"],
                "start_year": setting["start_year"],
                "end_year": setting["end_year"],
                "n_players": len(final_ratings),
                "mean_final_rd": float(rd.mean()),
                "median_final_rd": float(rd.median()),
                "p10_final_rd": float(rd.quantile(0.10)),
                "p25_final_rd": float(rd.quantile(0.25)),
                "p75_final_rd": float(rd.quantile(0.75)),
                "p90_final_rd": float(rd.quantile(0.90)),
                "min_final_rd": float(rd.min()),
                "max_final_rd": float(rd.max()),
                "number_at_min_rd": int(np.isclose(rd, setting["min_rd"]).sum()),
                "number_at_max_rd": int(np.isclose(rd, setting["max_rd"]).sum()),
                "number_near_max_rd_within_5": int((rd >= setting["max_rd"] - 5.0).sum()),
            }
        ]
    )


def write_summary(
    matches: pd.DataFrame,
    setting: dict[str, Any],
    metrics: pd.DataFrame,
    rd_summary: pd.DataFrame,
    date_summary: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Write meeting-ready markdown summary."""

    metric = metrics.iloc[0]
    rd = rd_summary.iloc[0]
    date_lines = [
        f"- {row.event_date_ordering_method}: {int(row.match_count):,} matches ({row.share_of_matches:.2%})"
        for row in date_summary.itertuples(index=False)
    ]

    warning_lines = ["- None"] if not warnings else [f"- {warning}" for warning in warnings]
    near_max = int(rd["number_near_max_rd_within_5"])

    lines = [
        "# Glicko-1 Match-by-Match Baseline Summary",
        "",
        "## Aim",
        "",
        "This is the first full-data Glicko-1 baseline. It applies the validated Glicko-1 core formula to the full croquet match dataset and records pre-match predictions for later fair comparison work.",
        "",
        "## Rating-Period Assumption",
        "",
        "This first baseline uses a match-by-match rating-period assumption. Croquet data are available at game level, so this is the most pragmatic starting point. Event-level, monthly, or yearly rating periods should be tested later as sensitivity analysis.",
        "",
        "## Data Used",
        "",
        f"- Dataset: `outputs/elo_optimization/matches_1985_2025_checked.csv`",
        f"- Years: {int(matches['year'].min())}-{int(matches['year'].max())}",
        f"- Matches: {len(matches):,}",
        f"- Unique players: {pd.concat([matches['winner'], matches['loser']]).dropna().astype(int).nunique():,}",
        "",
        "## Glicko Setting",
        "",
        "- Model: classic Glicko-1",
        f"- Setting name: `{setting['setting_name']}`",
        f"- Initial rating: {setting['default_rating']:.1f}",
        f"- Initial RD: {setting['default_rd']:.1f}",
        f"- Minimum RD: {setting['min_rd']:.1f}",
        f"- Maximum RD: {setting['max_rd']:.1f}",
        f"- Inactivity RD inflation C: {setting['c']:.1f}",
        "- No inactivity RD inflation is applied in this first baseline.",
        "",
        "## Data Ordering",
        "",
        "The script preserves the raw date columns and creates ordering helper columns only where needed. Full parsed dates are used directly. Month-year-only dates are imputed to the 15th of the month for ordering only. Rows with no usable date fall back to year/event/code/fcode ordering.",
        "",
        *date_lines,
        "",
        "## 2025 Prediction Results",
        "",
        f"- Evaluation games: {int(metric['evaluation_games']):,}",
        f"- Log loss: {metric['log_loss']:.6f}",
        f"- Brier score: {metric['brier_score']:.6f}",
        f"- Accuracy: {metric['accuracy']:.6f}",
        f"- Baseline accuracy: {metric['baseline_accuracy']:.6f}",
        "",
        "These metrics are not compared directly against Elo in this step. They are saved so that a later fair-comparison script can use the same match list, ordering, and metric definitions.",
        "",
        "## RD Behaviour",
        "",
        f"- Players with final ratings: {int(rd['n_players']):,}",
        f"- Median final RD: {rd['median_final_rd']:.3f}",
        f"- Mean final RD: {rd['mean_final_rd']:.3f}",
        f"- Minimum final RD: {rd['min_final_rd']:.3f}",
        f"- Maximum final RD: {rd['max_final_rd']:.3f}",
        f"- Number of players at MIN_RD: {int(rd['number_at_min_rd']):,}",
        f"- Number of players at MAX_RD: {int(rd['number_at_max_rd']):,}",
        f"- Number of players within 5 RD points of MAX_RD: {near_max:,}",
        "",
        "Because C=0, inactive players do not become more uncertain over time. RD shrinkage should therefore be interpreted as a first pipeline diagnostic, not the final uncertainty model.",
        "",
        "## Important Limitations",
        "",
        "- This is only the first match-by-match Glicko baseline.",
        "- It does not yet test alternative rating periods.",
        "- It does not yet include inactivity RD inflation.",
        "- It does not yet compare directly against Elo.",
        "- Because C=0, inactive players do not become more uncertain over time; this will be addressed later.",
        "",
        "## Warnings",
        "",
        *warning_lines,
        "",
        "## Next Step",
        "",
        "Next, inspect this baseline's metrics and RD behaviour before deciding whether to prioritise inactivity/RD inflation, rating-period sensitivity, or Elo-vs-Glicko fair comparison preparation.",
    ]

    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    matches = load_matches()
    setting = SETTINGS[0]

    print()
    print("Glicko setting:")
    for key, value in setting.items():
        print(f"  {key}: {value}")

    date_summary = make_date_ordering_summary(
        matches, setting["setting_name"], setting["start_year"], setting["end_year"]
    )
    print()
    print("Event date ordering summary:")
    print(date_summary[["event_date_ordering_method", "match_count", "share_of_matches"]].to_string(index=False))

    print()
    print(f"Running {setting['setting_name']}...")
    predictions, final_ratings, yearly_rd_summary = run_glicko_match_by_match(matches, **setting)

    metrics = compute_prediction_metrics(predictions, setting)
    calibration = make_calibration_table(predictions, setting["setting_name"])
    confidence = make_confidence_table(predictions, setting["setting_name"])
    rd_summary = make_rd_summary(final_ratings, setting)

    metric = metrics.iloc[0]
    if int(metric["evaluation_games"]) != EXPECTED_2025_GAMES:
        warning = (
            f"Expected {EXPECTED_2025_GAMES:,} 2025 evaluation games but found "
            f"{int(metric['evaluation_games']):,}."
        )
        warnings.append(warning)
        print(f"WARNING: {warning}")

    print()
    print(f"2025 evaluation games: {int(metric['evaluation_games']):,}")
    print(
        "2025 metrics: "
        f"log_loss={metric['log_loss']:.6f}, "
        f"brier={metric['brier_score']:.6f}, "
        f"accuracy={metric['accuracy']:.6f}, "
        f"baseline_accuracy={metric['baseline_accuracy']:.6f}"
    )

    rd = rd_summary.iloc[0]
    print()
    print("Final RD summary:")
    print(
        f"n_players={int(rd['n_players']):,}, "
        f"median_RD={rd['median_final_rd']:.3f}, "
        f"mean_RD={rd['mean_final_rd']:.3f}, "
        f"min_RD={rd['min_final_rd']:.3f}, "
        f"max_RD={rd['max_final_rd']:.3f}, "
        f"at_min={int(rd['number_at_min_rd']):,}, "
        f"at_max={int(rd['number_at_max_rd']):,}"
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    final_ratings.to_csv(FINAL_RATINGS_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    calibration.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    confidence.to_csv(CONFIDENCE_PATH, index=False, encoding="utf-8-sig")
    rd_summary.to_csv(RD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    yearly_rd_summary.to_csv(YEARLY_RD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    date_summary.to_csv(DATE_ORDERING_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    write_summary(matches, setting, metrics, rd_summary, date_summary, warnings)

    print()
    print("Output files:")
    for path in [
        PREDICTIONS_PATH,
        FINAL_RATINGS_PATH,
        METRICS_PATH,
        CALIBRATION_PATH,
        CONFIDENCE_PATH,
        RD_SUMMARY_PATH,
        YEARLY_RD_SUMMARY_PATH,
        DATE_ORDERING_SUMMARY_PATH,
        SUMMARY_MD_PATH,
    ]:
        print(f"  {path}")

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
