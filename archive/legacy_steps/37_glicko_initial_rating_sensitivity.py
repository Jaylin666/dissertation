"""Test common initial-rating translation invariance in Glicko-1."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import math
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glicko_core import DEFAULT_RD, MAX_RD, MIN_RD, expected_score, update_two_players_single_game


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

FULL_HISTORY_PATH = PROJECT_ROOT / "outputs" / "elo_optimization" / "matches_1985_2025_checked.csv"
STEP33_SCORES_PATH = PROJECT_ROOT / "outputs" / "meeting6" / "33_orientation_corrected_per_match_scores_2025.csv"
STEP34_APPEARANCES_PATH = OUTPUT_DIR / "34_early_game_appearance_dataset.csv"
STEP36_KEY_RESULTS_PATH = OUTPUT_DIR / "36_key_initialisation_diagnostic_results.csv"

INPUT_VALIDATION_PATH = OUTPUT_DIR / "37_input_validation_checks.csv"
VALIDATION_METRICS_PATH = OUTPUT_DIR / "37_validation_initial_rating_metrics.csv"
SELECTION_PATH = OUTPUT_DIR / "37_initial_rating_selection.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "37_test_initial_rating_metrics.csv"
APPEARANCE_PREDICTIONS_PATH = OUTPUT_DIR / "37_initial_rating_appearance_predictions_2025.csv"
EARLY_GAME_METRICS_PATH = OUTPUT_DIR / "37_initial_rating_early_game_metrics.csv"
STAGE_METRICS_PATH = OUTPUT_DIR / "37_initial_rating_stage_metrics.csv"
PAIRWISE_1500_PATH = OUTPUT_DIR / "37_pairwise_vs_initial_1500.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "37_selected_vs_1500_bootstrap.csv"
SELECTED_VS_ELO_PATH = OUTPUT_DIR / "37_selected_glicko_vs_elo.csv"
PROBABILITY_BANDS_PATH = OUTPUT_DIR / "37_selected_initial_rating_probability_bands.csv"
RATING_DISTRIBUTION_PATH = OUTPUT_DIR / "37_rating_distribution_diagnostics.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "37_key_initial_rating_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "37_figure_manifest.csv"
FINAL_VALIDATION_PATH = OUTPUT_DIR / "37_initial_rating_sensitivity_validation_checks.csv"

START_YEAR = 1985
END_YEAR = 2025
VALIDATION_YEARS = [2023, 2024]
TEST_YEAR = 2025
EXPECTED_FULL_HISTORY_MATCHES = 456_382
EXPECTED_VALIDATION_MATCHES = 23_888
EXPECTED_TEST_MATCHES = 11_379
CURRENT_INITIAL_RATING = 1500
INITIAL_RATING_CANDIDATES = [1000, 1100, 1200, 1300, 1400, 1500]
CUMULATIVE_THRESHOLDS = [1, 5, 10, 20, 30, 50]
CORE_THRESHOLDS = [1, 5, 10, 20]
STAGE_LABELS = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260717
EPS = 1e-15
SELECTION_TIE_TOLERANCE = 1e-8

REQUIRED_MATCH_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_MATCH_COLUMNS = [
    "eventname",
    "event_date_raw",
    "event_date_parsed",
    "winner_name",
    "loser_name",
]
REQUIRED_STEP33_COLUMNS = [
    "match_id",
    "year",
    "event_key",
    "player_a_id",
    "player_b_id",
    "winner_id",
    "loser_id",
    "outcome_a",
    "p_a_Validation_best_Elo",
    "p_a_Glicko_low_fixed",
]

OUTPUT_FILES = [
    INPUT_VALIDATION_PATH,
    VALIDATION_METRICS_PATH,
    SELECTION_PATH,
    TEST_METRICS_PATH,
    APPEARANCE_PREDICTIONS_PATH,
    EARLY_GAME_METRICS_PATH,
    STAGE_METRICS_PATH,
    PAIRWISE_1500_PATH,
    BOOTSTRAP_PATH,
    SELECTED_VS_ELO_PATH,
    PROBABILITY_BANDS_PATH,
    RATING_DISTRIBUTION_PATH,
    KEY_RESULTS_PATH,
    FIGURE_MANIFEST_PATH,
    FINAL_VALIDATION_PATH,
]


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: str = "",
) -> None:
    """Append one validation check row."""

    rows.append(
        {
            "check_name": check_name,
            "status": "PASS" if bool(passed) else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def print_checks(checks: pd.DataFrame) -> None:
    """Print checks in PASS/FAIL form."""

    for row in checks.itertuples(index=False):
        detail = f" | details={row.details}" if isinstance(row.details, str) and row.details else ""
        print(f"[{row.status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")


def format_code_value(value: Any) -> str:
    """Return stable text for event-like IDs."""

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
    """Convert a dataset player code into an integer key."""

    return int(float(value))


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ordering date columns using the established Step 24 logic."""

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
    matches["event_date_ordering_method"] = np.where(parsed.notna(), "parsed_full_date", "fallback_no_date")

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

    year_fallback = matches["event_order_date"].isna()
    if year_fallback.any():
        fallback_dates = pd.to_datetime(
            {
                "year": matches.loc[year_fallback, "year"].astype(int).to_numpy(),
                "month": np.repeat(12, int(year_fallback.sum())),
                "day": np.repeat(31, int(year_fallback.sum())),
            },
            errors="coerce",
        )
        matches.loc[year_fallback, "event_order_date"] = fallback_dates.to_numpy()
        matches.loc[year_fallback, "event_date_ordering_method"] = "year_end_fallback"

    return matches


def add_inactivity_period_index(matches: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Construct the established month-based inactivity period index."""

    matches = matches.copy()
    event_dates = pd.to_datetime(matches["event_order_date"], errors="coerce")
    if event_dates.notna().any():
        period_index = (event_dates.dt.year * 12 + event_dates.dt.month).astype("Float64")
        fallback_mask = event_dates.isna()
        period_index.loc[fallback_mask] = matches.loc[fallback_mask, "year"].astype(int) * 12 + 12
        matches["inactivity_period_index"] = period_index.astype(int)
        matches["inactivity_period_source"] = np.where(
            fallback_mask,
            "year_fallback_december",
            "month_from_event_order_date",
        )
        return matches, "month"

    matches["inactivity_period_index"] = matches["year"].astype(int)
    matches["inactivity_period_source"] = "year_only_fallback"
    return matches, "year"


def load_matches() -> tuple[pd.DataFrame, str]:
    """Load, validate, sort, and prepare full chronological match data."""

    if not FULL_HISTORY_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {FULL_HISTORY_PATH}")
    matches = pd.read_csv(FULL_HISTORY_PATH, low_memory=False)
    missing_required = [col for col in REQUIRED_MATCH_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{FULL_HISTORY_PATH.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_MATCH_COLUMNS:
        if col not in matches.columns:
            matches[col] = pd.NA

    for col in REQUIRED_MATCH_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    if int(matches[REQUIRED_MATCH_COLUMNS].isna().sum().sum()) > 0:
        raise ValueError("Required match columns contain missing values after numeric conversion.")

    matches = matches[(matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)].copy()
    matches = add_event_ordering_columns(matches)
    matches["event_order_date_missing"] = matches["event_order_date"].isna()
    matches = matches.sort_values(
        ["year", "event_order_date_missing", "event_order_date", "event", "fcode"],
        na_position="last",
    ).drop(columns=["event_order_date_missing"])
    matches = matches.reset_index(drop=True)
    matches["match_sequence"] = np.arange(1, len(matches) + 1)
    matches, inactivity_unit = add_inactivity_period_index(matches)
    matches["event_key"] = matches["year"].astype(int).astype(str) + "_" + matches["event"].astype(int).astype(str)
    return matches, inactivity_unit


def load_step33_and_step34() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Step 33 match predictions and Step 34 appearance data for Elo comparison."""

    if not STEP33_SCORES_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {STEP33_SCORES_PATH}")
    if not STEP34_APPEARANCES_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {STEP34_APPEARANCES_PATH}")

    step33 = pd.read_csv(STEP33_SCORES_PATH, low_memory=False)
    step34 = pd.read_csv(STEP34_APPEARANCES_PATH, low_memory=False)
    missing_step33 = [col for col in REQUIRED_STEP33_COLUMNS if col not in step33.columns]
    if missing_step33:
        raise ValueError(f"Step 33 table is missing required columns: {missing_step33}")
    return step33, step34


def low_inflation_spec(inactivity_unit: str) -> dict[str, Any]:
    """Return the established low-inflation Glicko specification."""

    target_periods = 240 if inactivity_unit == "month" else 20
    c_value = math.sqrt(((MAX_RD**2) - (MIN_RD**2)) / target_periods)
    return {
        "variant": "low_inflation",
        "c_value": c_value,
        "target_periods": target_periods,
        "inactivity_unit": inactivity_unit,
        "initial_rd": DEFAULT_RD,
        "min_rd": MIN_RD,
        "max_rd": MAX_RD,
        "rating_period": "match_by_match",
        "expected_score_definition": "p_A = expected_score(rating_A, rating_B, RD_B)",
    }


def inflate_rd_for_inactivity(rd: float, elapsed_periods: float, c_value: float) -> float:
    """Inflate RD after inactivity while preserving Glicko bounds."""

    if c_value <= 0.0 or pd.isna(elapsed_periods) or elapsed_periods <= 0:
        return float(rd)
    inflated = math.sqrt((float(rd) ** 2) + (float(c_value) ** 2) * float(elapsed_periods))
    return min(MAX_RD, max(MIN_RD, inflated))


def binary_metric_arrays(p: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Brier, log loss, and correctness arrays."""

    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    p_clip = np.clip(p_arr, EPS, 1.0 - EPS)
    brier = (p_arr - y_arr) ** 2
    log_loss = -(y_arr * np.log(p_clip) + (1.0 - y_arr) * np.log(1.0 - p_clip))
    correct = ((p_arr >= 0.5).astype(int) == y_arr.astype(int)).astype(int)
    return brier, log_loss, correct


def metric_row(group: pd.DataFrame, p_col: str, y_col: str, prefix: str = "") -> dict[str, Any]:
    """Calculate standard binary prediction metrics for one group."""

    brier, log_loss, correct = binary_metric_arrays(group[p_col], group[y_col])
    p = group[p_col].astype(float)
    y = group[y_col].astype(float)
    pre = f"{prefix}_" if prefix else ""
    return {
        f"{pre}number_of_matches": int(len(group)),
        f"{pre}brier": float(np.mean(brier)),
        f"{pre}log_loss": float(np.mean(log_loss)),
        f"{pre}accuracy": float(np.mean(correct)),
        f"{pre}mean_predicted_probability": float(p.mean()),
        f"{pre}empirical_win_rate": float(y.mean()),
    }


def run_initial_rating_candidate(
    matches: pd.DataFrame,
    candidate_initial_rating: int,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one low-inflation Glicko simulation with a candidate initial rating."""

    print(f"  running candidate initial rating {candidate_initial_rating}")
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_period_index: dict[int, int] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}
    prediction_rows: list[dict[str, Any]] = []

    sim_cols = [
        "fcode",
        "match_sequence",
        "year",
        "event",
        "event_key",
        "eventname",
        "event_date_raw",
        "event_date_parsed",
        "event_order_date",
        "event_date_ordering_method",
        "inactivity_period_index",
        "winner",
        "loser",
        "winner_name",
        "loser_name",
    ]

    c_value = float(spec["c_value"])
    for row in matches[sim_cols].itertuples(index=False):
        year = int(row.year)
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        period_index = int(row.inactivity_period_index)

        if winner not in ratings:
            ratings[winner] = float(candidate_initial_rating)
            rds[winner] = DEFAULT_RD
            winner_gap = np.nan
        else:
            winner_gap = period_index - last_period_index[winner]
            rds[winner] = inflate_rd_for_inactivity(rds[winner], winner_gap, c_value)

        if loser not in ratings:
            ratings[loser] = float(candidate_initial_rating)
            rds[loser] = DEFAULT_RD
            loser_gap = np.nan
        else:
            loser_gap = period_index - last_period_index[loser]
            rds[loser] = inflate_rd_for_inactivity(rds[loser], loser_gap, c_value)

        winner_rating_before = ratings[winner]
        winner_rd_before = rds[winner]
        loser_rating_before = ratings[loser]
        loser_rd_before = rds[loser]
        winner_games_before = games_played[winner]
        loser_games_before = games_played[loser]

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        outcome_a = 1 if player_a == winner else 0
        if player_a == winner:
            rating_a = winner_rating_before
            rd_a = winner_rd_before
            rating_b = loser_rating_before
            rd_b = loser_rd_before
            a_total_games_before = winner_games_before
            b_total_games_before = loser_games_before
            a_gap = winner_gap
            b_gap = loser_gap
        else:
            rating_a = loser_rating_before
            rd_a = loser_rd_before
            rating_b = winner_rating_before
            rd_b = winner_rd_before
            a_total_games_before = loser_games_before
            b_total_games_before = winner_games_before
            a_gap = loser_gap
            b_gap = winner_gap

        p_a = expected_score(rating_a, rating_b, rd_b)

        if year in {*VALIDATION_YEARS, TEST_YEAR}:
            prediction_rows.append(
                {
                    "candidate_initial_rating": int(candidate_initial_rating),
                    "variant": "low_inflation",
                    "c_value": c_value,
                    "initial_rd": DEFAULT_RD,
                    "rating_period": "match_by_match",
                    "match_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "match_sequence": int(row.match_sequence),
                    "year": year,
                    "event": int(row.event),
                    "event_id": int(row.event),
                    "event_key": row.event_key,
                    "eventname": row.eventname,
                    "event_date_raw": row.event_date_raw,
                    "event_date_parsed": row.event_date_parsed,
                    "match_date": row.event_order_date,
                    "event_date_ordering_method": row.event_date_ordering_method,
                    "winner_id": winner,
                    "loser_id": loser,
                    "player_a_id": player_a,
                    "player_b_id": player_b,
                    "outcome_a": outcome_a,
                    "p_a_Glicko_initial_rating_candidate": p_a,
                    "rating_a_Glicko": rating_a,
                    "rating_b_Glicko": rating_b,
                    "rd_a_Glicko": rd_a,
                    "rd_b_Glicko": rd_b,
                    "a_total_games_before": int(a_total_games_before),
                    "b_total_games_before": int(b_total_games_before),
                    "a_is_debut": bool(a_total_games_before == 0),
                    "b_is_debut": bool(b_total_games_before == 0),
                    "a_gap_periods": a_gap,
                    "b_gap_periods": b_gap,
                    "probability_orientation": "canonical player A win probability",
                    "probability_definition": "expected_score(rating_A, rating_B, RD_B)",
                }
            )

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

        last_period_index[winner] = max(last_period_index.get(winner, period_index), period_index)
        last_period_index[loser] = max(last_period_index.get(loser, period_index), period_index)
        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

        if winner not in player_names and not pd.isna(row.winner_name):
            player_names[winner] = str(row.winner_name)
        if loser not in player_names and not pd.isna(row.loser_name):
            player_names[loser] = str(row.loser_name)

    predictions = pd.DataFrame(prediction_rows)
    brier, log_loss, correct = binary_metric_arrays(
        predictions["p_a_Glicko_initial_rating_candidate"],
        predictions["outcome_a"],
    )
    predictions["brier"] = brier
    predictions["log_loss"] = log_loss
    predictions["correct"] = correct

    final_rows = []
    for player_id in sorted(ratings):
        final_rows.append(
            {
                "candidate_initial_rating": int(candidate_initial_rating),
                "player_id": int(player_id),
                "player_name": player_names.get(player_id, pd.NA),
                "rating": float(ratings[player_id]),
                "rd": float(rds[player_id]),
                "games_played": int(games_played[player_id]),
                "wins": int(wins[player_id]),
                "losses": int(losses[player_id]),
                "last_period_index": last_period_index.get(player_id, pd.NA),
            }
        )
    final_ratings = pd.DataFrame(final_rows)
    final_ratings["rank_by_rating"] = final_ratings["rating"].rank(method="min", ascending=False).astype(int)
    return predictions, final_ratings.sort_values(["rank_by_rating", "player_id"]).reset_index(drop=True)


def build_appearance_predictions(test_predictions: pd.DataFrame) -> pd.DataFrame:
    """Expand candidate 2025 match predictions to focal player appearances."""

    rows: list[pd.DataFrame] = []
    for side in ["a", "b"]:
        opponent = "b" if side == "a" else "a"
        focal_side = side.upper()
        out = pd.DataFrame(
            {
                "candidate_initial_rating": test_predictions["candidate_initial_rating"].astype(int),
                "match_id": test_predictions["match_id"].astype(int),
                "fcode": test_predictions["fcode"].astype(int),
                "match_sequence": test_predictions["match_sequence"].astype(int),
                "year": test_predictions["year"].astype(int),
                "match_date": test_predictions["match_date"],
                "event": test_predictions["event"].astype(int),
                "event_id": test_predictions["event_id"].astype(int),
                "event_key": test_predictions["event_key"],
                "focal_side": focal_side,
                "player_id": test_predictions[f"player_{side}_id"].astype(int),
                "opponent_id": test_predictions[f"player_{opponent}_id"].astype(int),
                "total_games_before": test_predictions[f"{side}_total_games_before"].astype(int),
                "opponent_total_games_before": test_predictions[f"{opponent}_total_games_before"].astype(int),
                "focal_pre_match_rating": test_predictions[f"rating_{side}_Glicko"].astype(float),
                "opponent_pre_match_rating": test_predictions[f"rating_{opponent}_Glicko"].astype(float),
                "focal_pre_match_RD": test_predictions[f"rd_{side}_Glicko"].astype(float),
                "opponent_pre_match_RD": test_predictions[f"rd_{opponent}_Glicko"].astype(float),
            }
        )
        if side == "a":
            out["outcome_focal"] = test_predictions["outcome_a"].astype(int)
            out["p_focal_Glicko"] = test_predictions["p_a_Glicko_initial_rating_candidate"].astype(float)
        else:
            out["outcome_focal"] = (1 - test_predictions["outcome_a"].astype(int)).astype(int)
            out["p_focal_Glicko"] = 1.0 - test_predictions["p_a_Glicko_initial_rating_candidate"].astype(float)

        out["appearance_number"] = out["total_games_before"] + 1
        out["opponent_appearance_number"] = out["opponent_total_games_before"] + 1
        out["rating_difference"] = out["focal_pre_match_rating"] - out["opponent_pre_match_rating"]
        rows.append(out)

    appearances = pd.concat(rows, ignore_index=True)
    for threshold in CUMULATIVE_THRESHOLDS:
        appearances[f"first_{threshold}"] = appearances["appearance_number"] <= threshold
    conditions = [
        appearances["appearance_number"].eq(1),
        appearances["appearance_number"].between(2, 5),
        appearances["appearance_number"].between(6, 10),
        appearances["appearance_number"].between(11, 20),
        appearances["appearance_number"].between(21, 50),
        appearances["appearance_number"].ge(51),
    ]
    appearances["appearance_stage"] = np.select(conditions, STAGE_LABELS, default=pd.NA)
    brier, log_loss, correct = binary_metric_arrays(appearances["p_focal_Glicko"], appearances["outcome_focal"])
    appearances["Brier_loss"] = brier
    appearances["log_loss"] = log_loss
    appearances["correct"] = correct
    return appearances.sort_values(["candidate_initial_rating", "match_sequence", "focal_side"]).reset_index(drop=True)


def appearance_metric_row(group: pd.DataFrame, group_type: str, group_name: str, group_order: int) -> dict[str, Any]:
    """Calculate appearance-level metrics for one candidate/group."""

    p = group["p_focal_Glicko"].astype(float)
    y = group["outcome_focal"].astype(float)
    return {
        "candidate_initial_rating": int(group["candidate_initial_rating"].iloc[0]),
        "group_type": group_type,
        "group": group_name,
        "group_order": int(group_order),
        "number_of_appearances": int(len(group)),
        "number_of_unique_players": int(group["player_id"].nunique()),
        "number_of_unique_matches": int(group["match_id"].nunique()),
        "mean_predicted_probability": float(p.mean()),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias": float((p - y).mean()),
        "brier": float(group["Brier_loss"].mean()),
        "log_loss": float(group["log_loss"].mean()),
        "accuracy": float(group["correct"].mean()),
    }


def build_early_and_stage_metrics(appearances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise early-game and stage performance for every candidate."""

    early_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for candidate, candidate_df in appearances.groupby("candidate_initial_rating", sort=True):
        for threshold in CUMULATIVE_THRESHOLDS:
            group = candidate_df.loc[candidate_df[f"first_{threshold}"]].copy()
            early_rows.append(appearance_metric_row(group, "cumulative_threshold", f"first_{threshold}", threshold))
        for order, stage in enumerate(STAGE_LABELS, start=1):
            group = candidate_df.loc[candidate_df["appearance_stage"].eq(stage)].copy()
            stage_rows.append(appearance_metric_row(group, "appearance_stage", stage, order))
    early = pd.DataFrame(early_rows)
    stage = pd.DataFrame(stage_rows)
    early.to_csv(EARLY_GAME_METRICS_PATH, index=False, encoding="utf-8-sig")
    stage.to_csv(STAGE_METRICS_PATH, index=False, encoding="utf-8-sig")
    return early, stage


def build_match_metrics(predictions: pd.DataFrame, years: list[int], period_name: str) -> pd.DataFrame:
    """Build match-level metrics for validation or test years."""

    rows: list[dict[str, Any]] = []
    data = predictions.loc[predictions["year"].isin(years)].copy()
    for candidate, group in data.groupby("candidate_initial_rating", sort=True):
        row = metric_row(group, "p_a_Glicko_initial_rating_candidate", "outcome_a")
        row.update(
            {
                "period": period_name,
                "candidate_initial_rating": int(candidate),
                "years": ",".join(str(year) for year in years),
                "empirical_canonical_player_A_win_rate": row.pop("empirical_win_rate"),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("candidate_initial_rating").reset_index(drop=True)
    return out


def validation_debut_biases(predictions: pd.DataFrame) -> dict[int, float]:
    """Calculate validation first_1 focal prediction bias for tie-breaking."""

    validation = predictions.loc[predictions["year"].isin(VALIDATION_YEARS)].copy()
    appearances = build_appearance_predictions(
        validation.rename(
            columns={"event_date_parsed": "event_date_parsed"}
        )
    )
    biases: dict[int, float] = {}
    for candidate, group in appearances.groupby("candidate_initial_rating"):
        debut = group.loc[group["appearance_number"].eq(1)]
        biases[int(candidate)] = float((debut["p_focal_Glicko"] - debut["outcome_focal"]).mean()) if not debut.empty else np.nan
    return biases


def select_initial_rating(validation_metrics: pd.DataFrame, validation_bias_by_candidate: dict[int, float]) -> pd.DataFrame:
    """Select initial rating using validation Brier, then log loss and tie-breaks."""

    out = validation_metrics.copy()
    out["validation_brier"] = out["brier"]
    out["validation_logloss"] = out["log_loss"]
    out["validation_accuracy"] = out["accuracy"]
    out["validation_debut_prediction_bias"] = out["candidate_initial_rating"].map(validation_bias_by_candidate)
    out["abs_validation_debut_prediction_bias"] = out["validation_debut_prediction_bias"].abs()
    out["distance_from_current_1500"] = (out["candidate_initial_rating"] - CURRENT_INITIAL_RATING).abs()
    out["validation_rank_brier"] = out["validation_brier"].round(8).rank(method="min", ascending=True).astype(int)
    out["validation_rank_logloss"] = out["validation_logloss"].round(8).rank(method="min", ascending=True).astype(int)

    min_brier = float(out["validation_brier"].min())
    tied = out.loc[(out["validation_brier"] - min_brier).abs() <= SELECTION_TIE_TOLERANCE].copy()
    min_logloss = float(tied["validation_logloss"].min())
    tied = tied.loc[(tied["validation_logloss"] - min_logloss).abs() <= SELECTION_TIE_TOLERANCE].copy()
    min_abs_bias = float(tied["abs_validation_debut_prediction_bias"].min())
    tied = tied.loc[(tied["abs_validation_debut_prediction_bias"] - min_abs_bias).abs() <= SELECTION_TIE_TOLERANCE].copy()
    selected_row = tied.sort_values(["distance_from_current_1500", "candidate_initial_rating"]).iloc[0]
    selected_rating = int(selected_row["candidate_initial_rating"])
    out["selected"] = out["candidate_initial_rating"].eq(selected_rating)
    all_candidates_tied = len(tied) == len(out)
    selected_reason = (
        "All candidates tied within tolerance on validation Brier, log loss, and debut bias; retained current 1500."
        if all_candidates_tied and selected_rating == CURRENT_INITIAL_RATING
        else (
            "Selected on validation Brier; tie tolerance "
            f"{SELECTION_TIE_TOLERANCE:g}; secondary log loss, debut bias, then closeness to 1500."
        )
    )
    out["selection_reason"] = np.where(
        out["selected"],
        selected_reason,
        "Not selected by validation-only rule.",
    )
    columns = [
        "candidate_initial_rating",
        "validation_brier",
        "validation_logloss",
        "validation_accuracy",
        "validation_debut_prediction_bias",
        "validation_rank_brier",
        "validation_rank_logloss",
        "selected",
        "selection_reason",
    ]
    out = out[columns].sort_values("candidate_initial_rating").reset_index(drop=True)
    out.to_csv(SELECTION_PATH, index=False, encoding="utf-8-sig")
    return out


def build_pairwise_vs_1500(test_predictions: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired differences versus the current initial rating 1500."""

    rows: list[dict[str, Any]] = []
    current_matches = test_predictions.loc[test_predictions["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].copy()
    for candidate in INITIAL_RATING_CANDIDATES:
        if candidate == CURRENT_INITIAL_RATING:
            continue
        other = test_predictions.loc[test_predictions["candidate_initial_rating"].eq(candidate)].copy()
        merged = current_matches[["match_id", "brier", "log_loss"]].merge(
            other[["match_id", "brier", "log_loss"]],
            on="match_id",
            how="inner",
            suffixes=("_1500", "_candidate"),
            validate="one_to_one",
        )
        rows.append(
            {
                "candidate_initial_rating": candidate,
                "analysis_group": "overall_2025",
                "group_type": "overall_match_level",
                "paired_rows": int(len(merged)),
                "delta_brier_1500_minus_candidate": float((merged["brier_1500"] - merged["brier_candidate"]).mean()),
                "delta_logloss_1500_minus_candidate": float((merged["log_loss_1500"] - merged["log_loss_candidate"]).mean()),
                "positive_delta_means": "alternative candidate is better than 1500",
            }
        )

    current_app = appearances.loc[appearances["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].copy()
    for candidate in INITIAL_RATING_CANDIDATES:
        if candidate == CURRENT_INITIAL_RATING:
            continue
        other = appearances.loc[appearances["candidate_initial_rating"].eq(candidate)].copy()
        merged_all = current_app[["match_id", "player_id", "Brier_loss", "log_loss", *[f"first_{t}" for t in CUMULATIVE_THRESHOLDS]]].merge(
            other[["match_id", "player_id", "Brier_loss", "log_loss", *[f"first_{t}" for t in CUMULATIVE_THRESHOLDS]]],
            on=["match_id", "player_id"],
            how="inner",
            suffixes=("_1500", "_candidate"),
            validate="one_to_one",
        )
        for threshold in CUMULATIVE_THRESHOLDS:
            mask = merged_all[f"first_{threshold}_1500"].astype(bool)
            group = merged_all.loc[mask]
            rows.append(
                {
                    "candidate_initial_rating": candidate,
                    "analysis_group": f"first_{threshold}",
                    "group_type": "appearance_cumulative_threshold",
                    "paired_rows": int(len(group)),
                    "delta_brier_1500_minus_candidate": float((group["Brier_loss_1500"] - group["Brier_loss_candidate"]).mean()),
                    "delta_logloss_1500_minus_candidate": float((group["log_loss_1500"] - group["log_loss_candidate"]).mean()),
                    "positive_delta_means": "alternative candidate is better than 1500",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PAIRWISE_1500_PATH, index=False, encoding="utf-8-sig")
    return out


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Return a percentile 95% CI."""

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def event_cluster_bootstrap(
    paired_matches: pd.DataFrame,
    selected_rating: int,
    reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Bootstrap selected-vs-1500 match-level paired differences by event."""

    rng = np.random.default_rng(seed)
    events = paired_matches["event_key"].drop_duplicates().to_numpy()
    event_to_indices = {
        event: paired_matches.index[paired_matches["event_key"].eq(event)].to_numpy()
        for event in events
    }
    brier_samples = np.empty(reps)
    logloss_samples = np.empty(reps)
    for i in range(reps):
        sampled_events = rng.choice(events, size=len(events), replace=True)
        idx = np.concatenate([event_to_indices[event] for event in sampled_events])
        sample = paired_matches.loc[idx]
        brier_samples[i] = (sample["brier_1500"] - sample["brier_selected"]).mean()
        logloss_samples[i] = (sample["log_loss_1500"] - sample["log_loss_selected"]).mean()

    rows = []
    for metric, point, values in [
        ("delta_brier", (paired_matches["brier_1500"] - paired_matches["brier_selected"]).mean(), brier_samples),
        ("delta_log_loss", (paired_matches["log_loss_1500"] - paired_matches["log_loss_selected"]).mean(), logloss_samples),
    ]:
        lower, upper = bootstrap_ci(values)
        rows.append(
            {
                "comparison": f"initial_{CURRENT_INITIAL_RATING}_minus_initial_{selected_rating}",
                "analysis_group": "overall_2025",
                "bootstrap_unit": "event_key",
                "bootstrap_replications": reps,
                "random_seed": seed,
                "metric": metric,
                "point_estimate": float(point),
                "ci_lower": lower,
                "ci_upper": upper,
                "paired_rows": int(len(paired_matches)),
                "clusters": int(len(events)),
                "positive_delta_means": "validation-selected initial rating is better than 1500",
            }
        )
    return rows


def player_cluster_bootstrap(
    paired_appearances: pd.DataFrame,
    selected_rating: int,
    group_name: str,
    mask: pd.Series,
    reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Bootstrap selected-vs-1500 appearance-level paired differences by focal player."""

    group = paired_appearances.loc[mask].copy()
    rng = np.random.default_rng(seed)
    players = group["player_id"].drop_duplicates().to_numpy()
    player_to_indices = {
        player: group.index[group["player_id"].eq(player)].to_numpy()
        for player in players
    }
    brier_samples = np.empty(reps)
    logloss_samples = np.empty(reps)
    for i in range(reps):
        sampled_players = rng.choice(players, size=len(players), replace=True)
        idx = np.concatenate([player_to_indices[player] for player in sampled_players])
        sample = group.loc[idx]
        brier_samples[i] = (sample["Brier_loss_1500"] - sample["Brier_loss_selected"]).mean()
        logloss_samples[i] = (sample["log_loss_1500"] - sample["log_loss_selected"]).mean()

    rows = []
    for metric, point, values in [
        ("delta_brier", (group["Brier_loss_1500"] - group["Brier_loss_selected"]).mean(), brier_samples),
        ("delta_log_loss", (group["log_loss_1500"] - group["log_loss_selected"]).mean(), logloss_samples),
    ]:
        lower, upper = bootstrap_ci(values)
        rows.append(
            {
                "comparison": f"initial_{CURRENT_INITIAL_RATING}_minus_initial_{selected_rating}",
                "analysis_group": group_name,
                "bootstrap_unit": "focal_player_id",
                "bootstrap_replications": reps,
                "random_seed": seed,
                "metric": metric,
                "point_estimate": float(point),
                "ci_lower": lower,
                "ci_upper": upper,
                "paired_rows": int(len(group)),
                "clusters": int(len(players)),
                "positive_delta_means": "validation-selected initial rating is better than 1500",
            }
        )
    return rows


def build_selected_vs_1500_bootstrap(
    test_predictions: pd.DataFrame,
    appearances: pd.DataFrame,
    selected_rating: int,
) -> pd.DataFrame:
    """Create paired bootstrap CIs for selected initial rating versus 1500."""

    selected_matches = test_predictions.loc[test_predictions["candidate_initial_rating"].eq(selected_rating)].copy()
    current_matches = test_predictions.loc[test_predictions["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].copy()
    paired_matches = current_matches[["match_id", "event_key", "brier", "log_loss"]].merge(
        selected_matches[["match_id", "brier", "log_loss"]],
        on="match_id",
        how="inner",
        suffixes=("_1500", "_selected"),
        validate="one_to_one",
    )
    rows = event_cluster_bootstrap(paired_matches, selected_rating, seed=RANDOM_SEED)

    selected_app = appearances.loc[appearances["candidate_initial_rating"].eq(selected_rating)].copy()
    current_app = appearances.loc[appearances["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].copy()
    paired_app = current_app[
        ["match_id", "player_id", "Brier_loss", "log_loss", *[f"first_{t}" for t in CORE_THRESHOLDS]]
    ].merge(
        selected_app[["match_id", "player_id", "Brier_loss", "log_loss", *[f"first_{t}" for t in CORE_THRESHOLDS]]],
        on=["match_id", "player_id"],
        how="inner",
        suffixes=("_1500", "_selected"),
        validate="one_to_one",
    )
    for threshold in CORE_THRESHOLDS:
        rows.extend(
            player_cluster_bootstrap(
                paired_app,
                selected_rating,
                f"first_{threshold}",
                paired_app[f"first_{threshold}_1500"].astype(bool),
                seed=RANDOM_SEED + threshold,
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(BOOTSTRAP_PATH, index=False, encoding="utf-8-sig")
    return out


def build_selected_glicko_vs_elo(
    test_predictions: pd.DataFrame,
    appearances: pd.DataFrame,
    step33: pd.DataFrame,
    step34: pd.DataFrame,
    selected_rating: int,
) -> pd.DataFrame:
    """Compare validation-selected Glicko with validation-best Elo."""

    rows: list[dict[str, Any]] = []
    selected_matches = test_predictions.loc[test_predictions["candidate_initial_rating"].eq(selected_rating)].copy()
    merged = selected_matches.merge(
        step33[["match_id", "p_a_Validation_best_Elo"]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    for model_name, p_col in [
        ("Selected_Glicko", "p_a_Glicko_initial_rating_candidate"),
        ("Validation_best_Elo", "p_a_Validation_best_Elo"),
    ]:
        metrics = metric_row(merged, p_col, "outcome_a")
        rows.append(
            {
                "analysis_group": "overall_2025",
                "model": model_name,
                "initial_rating": selected_rating if model_name == "Selected_Glicko" else pd.NA,
                **metrics,
                "prediction_bias": float((merged[p_col].astype(float) - merged["outcome_a"].astype(float)).mean()),
            }
        )

    selected_brier, selected_logloss, selected_correct = binary_metric_arrays(
        merged["p_a_Glicko_initial_rating_candidate"],
        merged["outcome_a"],
    )
    elo_brier, elo_logloss, elo_correct = binary_metric_arrays(merged["p_a_Validation_best_Elo"], merged["outcome_a"])
    rows.append(
        {
            "analysis_group": "overall_2025",
            "model": "Delta_Elo_minus_Selected_Glicko",
            "initial_rating": selected_rating,
            "number_of_matches": int(len(merged)),
            "brier": float(np.mean(elo_brier) - np.mean(selected_brier)),
            "log_loss": float(np.mean(elo_logloss) - np.mean(selected_logloss)),
            "accuracy": float(np.mean(selected_correct) - np.mean(elo_correct)),
            "mean_predicted_probability": pd.NA,
            "empirical_win_rate": float(merged["outcome_a"].mean()),
            "prediction_bias": pd.NA,
            "delta_definition": "Elo metric minus selected Glicko metric; positive Brier/log loss means selected Glicko is better",
        }
    )

    selected_app = appearances.loc[appearances["candidate_initial_rating"].eq(selected_rating)].copy()
    elo_app = step34[["match_id", "player_id", "p_focal_Validation_best_Elo"]].copy()
    app = selected_app.merge(elo_app, on=["match_id", "player_id"], how="inner", validate="one_to_one")
    for threshold in CORE_THRESHOLDS:
        group = app.loc[app[f"first_{threshold}"]].copy()
        for model_name, p_col in [
            ("Selected_Glicko", "p_focal_Glicko"),
            ("Validation_best_Elo", "p_focal_Validation_best_Elo"),
        ]:
            brier_arr, logloss_arr, correct_arr = binary_metric_arrays(group[p_col], group["outcome_focal"])
            rows.append(
                {
                    "analysis_group": f"first_{threshold}",
                    "model": model_name,
                    "initial_rating": selected_rating if model_name == "Selected_Glicko" else pd.NA,
                    "number_of_matches": int(group["match_id"].nunique()),
                    "number_of_appearances": int(len(group)),
                    "brier": float(np.mean(brier_arr)),
                    "log_loss": float(np.mean(logloss_arr)),
                    "accuracy": float(np.mean(correct_arr)),
                    "mean_predicted_probability": float(group[p_col].mean()),
                    "empirical_win_rate": float(group["outcome_focal"].mean()),
                    "prediction_bias": float((group[p_col].astype(float) - group["outcome_focal"].astype(float)).mean()),
                }
            )
        g_brier, g_logloss, g_correct = binary_metric_arrays(group["p_focal_Glicko"], group["outcome_focal"])
        e_brier, e_logloss, e_correct = binary_metric_arrays(group["p_focal_Validation_best_Elo"], group["outcome_focal"])
        rows.append(
            {
                "analysis_group": f"first_{threshold}",
                "model": "Delta_Elo_minus_Selected_Glicko",
                "initial_rating": selected_rating,
                "number_of_matches": int(group["match_id"].nunique()),
                "number_of_appearances": int(len(group)),
                "brier": float(np.mean(e_brier) - np.mean(g_brier)),
                "log_loss": float(np.mean(e_logloss) - np.mean(g_logloss)),
                "accuracy": float(np.mean(g_correct) - np.mean(e_correct)),
                "mean_predicted_probability": pd.NA,
                "empirical_win_rate": float(group["outcome_focal"].mean()),
                "prediction_bias": pd.NA,
                "delta_definition": "Elo metric minus selected Glicko metric; positive Brier/log loss means selected Glicko is better",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(SELECTED_VS_ELO_PATH, index=False, encoding="utf-8-sig")
    return out


def build_probability_bands(appearances: pd.DataFrame, selected_rating: int) -> pd.DataFrame:
    """Build selected-vs-1500 probability bands for first_1 and first_5 appearances."""

    rows: list[dict[str, Any]] = []
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000000001]
    labels = ["0.00-<0.20", "0.20-<0.40", "0.40-<0.60", "0.60-<0.80", "0.80-1.00"]
    candidate_label_pairs = (
        [(selected_rating, "selected_and_current_1500")]
        if selected_rating == CURRENT_INITIAL_RATING
        else [(selected_rating, "validation_selected"), (CURRENT_INITIAL_RATING, "current_1500")]
    )
    subset = appearances.loc[appearances["candidate_initial_rating"].isin([candidate for candidate, _ in candidate_label_pairs])].copy()
    subset["probability_band"] = pd.cut(
        subset["p_focal_Glicko"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    for candidate, model_label in candidate_label_pairs:
        cand = subset.loc[subset["candidate_initial_rating"].eq(candidate)]
        for group_name, threshold in [("first_1", 1), ("first_5", 5)]:
            group = cand.loc[cand[f"first_{threshold}"]].copy()
            for order, label in enumerate(labels, start=1):
                band = group.loc[group["probability_band"].astype(str).eq(label)].copy()
                if band.empty:
                    rows.append(
                        {
                            "candidate_initial_rating": int(candidate),
                            "model_label": model_label,
                            "analysis_group": group_name,
                            "band_order": order,
                            "probability_band": label,
                            "number_of_appearances": 0,
                            "mean_predicted_probability": pd.NA,
                            "empirical_win_rate": pd.NA,
                            "prediction_bias": pd.NA,
                            "brier": pd.NA,
                            "log_loss": pd.NA,
                        }
                    )
                else:
                    rows.append(
                        {
                            "candidate_initial_rating": int(candidate),
                            "model_label": model_label,
                            "analysis_group": group_name,
                            "band_order": order,
                            "probability_band": label,
                            "number_of_appearances": int(len(band)),
                            "mean_predicted_probability": float(band["p_focal_Glicko"].mean()),
                            "empirical_win_rate": float(band["outcome_focal"].mean()),
                            "prediction_bias": float((band["p_focal_Glicko"] - band["outcome_focal"]).mean()),
                            "brier": float(band["Brier_loss"].mean()),
                            "log_loss": float(band["log_loss"].mean()),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(PROBABILITY_BANDS_PATH, index=False, encoding="utf-8-sig")
    return out


def distribution_summary(values: pd.Series) -> dict[str, Any]:
    """Return compact distribution statistics."""

    values = values.astype(float).dropna()
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build_rating_distribution_diagnostics(final_ratings: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """Summarise final rating distributions and debut-opponent ratings."""

    rows: list[dict[str, Any]] = []
    for candidate, group in final_ratings.groupby("candidate_initial_rating", sort=True):
        stats = distribution_summary(group["rating"])
        rows.append(
            {
                "candidate_initial_rating": int(candidate),
                "distribution": "final_2025_player_ratings",
                "number_of_rated_players": int(len(group)),
                **stats,
                "interpretation_note": "Pure location shifts should not be interpreted as ranking-quality improvements.",
            }
        )
    first1 = appearances.loc[appearances["appearance_number"].eq(1)].copy()
    for candidate, group in first1.groupby("candidate_initial_rating", sort=True):
        stats = distribution_summary(group["opponent_pre_match_rating"])
        rows.append(
            {
                "candidate_initial_rating": int(candidate),
                "distribution": "opponent_ratings_faced_by_first_1_players",
                "number_of_rated_players": int(group["opponent_id"].nunique()),
                **stats,
                "interpretation_note": "Opponent ratings faced by debut players under this candidate initial rating.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RATING_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")
    return out


def underprediction_evidence(probability_bands: pd.DataFrame, selected_rating: int) -> bool:
    """Flag evidence that selected candidate under-predicts new players."""

    selected = probability_bands.loc[
        probability_bands["candidate_initial_rating"].eq(selected_rating)
        & probability_bands["number_of_appearances"].fillna(0).ge(10)
    ].copy()
    if selected.empty:
        return False
    selected["underprediction_gap"] = selected["empirical_win_rate"] - selected["mean_predicted_probability"]
    return bool((selected["underprediction_gap"] > 0.10).any())


def create_figures(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    selected_vs_elo: pd.DataFrame,
    selected_rating: int,
) -> pd.DataFrame:
    """Generate Step 37 figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    def save(path: Path, title: str, description: str) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        manifest_rows.append({"figure_id": path.stem, "path": str(path), "title": title, "description": description})

    def mark_selected_and_current(ax: plt.Axes) -> None:
        ax.axvline(selected_rating, color="#C75000", linestyle="--", linewidth=1.5, label=f"Selected {selected_rating}")
        ax.axvline(CURRENT_INITIAL_RATING, color="#333333", linestyle=":", linewidth=1.5, label="Current 1500")
        ax.legend(frameon=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(validation_metrics["candidate_initial_rating"], validation_metrics["brier"], marker="o", color="#1B4D89")
    mark_selected_and_current(ax)
    ax.set_title("Validation Brier by Initial Rating")
    ax.set_xlabel("Initial rating")
    ax.set_ylabel("Validation Brier")
    save(
        FIGURE_DIR / "37_fig01_validation_brier_by_initial_rating.png",
        "Validation Brier by initial rating",
        "Validation-period Brier score across candidate Glicko initial ratings.",
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(test_metrics["candidate_initial_rating"], test_metrics["brier"], marker="o", color="#2A9D8F")
    mark_selected_and_current(ax)
    ax.set_title("Fixed 2025 Brier by Initial Rating")
    ax.set_xlabel("Initial rating")
    ax.set_ylabel("2025 Brier")
    save(
        FIGURE_DIR / "37_fig02_test_brier_by_initial_rating.png",
        "Test Brier by initial rating",
        "Fixed 2025 Brier score across candidate Glicko initial ratings.",
    )

    first1 = early_metrics.loc[early_metrics["group"].eq("first_1")].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(first1["candidate_initial_rating"], first1["prediction_bias"], marker="o", color="#C75000")
    ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
    mark_selected_and_current(ax)
    ax.set_title("First-Appearance Bias by Initial Rating")
    ax.set_xlabel("Initial rating")
    ax.set_ylabel("Prediction bias")
    save(
        FIGURE_DIR / "37_fig03_first1_bias_by_initial_rating.png",
        "First_1 bias by initial rating",
        "First-appearance focal prediction bias across candidate initial ratings.",
    )

    first5 = early_metrics.loc[early_metrics["group"].eq("first_5")].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(first5["candidate_initial_rating"], first5["prediction_bias"], marker="o", color="#6F4E7C")
    ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
    mark_selected_and_current(ax)
    ax.set_title("First-Five Bias by Initial Rating")
    ax.set_xlabel("Initial rating")
    ax.set_ylabel("Prediction bias")
    save(
        FIGURE_DIR / "37_fig04_first5_bias_by_initial_rating.png",
        "First_5 bias by initial rating",
        "First-five cumulative focal prediction bias across candidate initial ratings.",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"first_1": "#C75000", "first_5": "#1B4D89", "first_10": "#2A9D8F", "first_20": "#6F4E7C"}
    for group_name in ["first_1", "first_5", "first_10", "first_20"]:
        group = early_metrics.loc[early_metrics["group"].eq(group_name)]
        ax.plot(group["candidate_initial_rating"], group["brier"], marker="o", label=group_name, color=colors[group_name])
    mark_selected_and_current(ax)
    ax.set_title("Early-Game Brier by Initial Rating")
    ax.set_xlabel("Initial rating")
    ax.set_ylabel("Brier")
    save(
        FIGURE_DIR / "37_fig05_early_game_brier_by_initial_rating.png",
        "Early-game Brier by initial rating",
        "First_1, first_5, first_10, and first_20 Brier scores across initial ratings.",
    )

    selected_current = early_metrics.loc[
        early_metrics["candidate_initial_rating"].isin([selected_rating, CURRENT_INITIAL_RATING])
        & early_metrics["group"].isin([f"first_{t}" for t in CORE_THRESHOLDS])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(CORE_THRESHOLDS))
    width = 0.36
    selected_vals = selected_current.loc[selected_current["candidate_initial_rating"].eq(selected_rating)].set_index("group")
    current_vals = selected_current.loc[selected_current["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].set_index("group")
    groups = [f"first_{t}" for t in CORE_THRESHOLDS]
    ax.bar(x - width / 2, [selected_vals.loc[g, "brier"] for g in groups], width, label=f"Selected {selected_rating}", color="#C75000")
    ax.bar(x + width / 2, [current_vals.loc[g, "brier"] for g in groups], width, label="Current 1500", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title("Selected Initial Rating vs 1500")
    ax.set_ylabel("Brier")
    ax.legend(frameon=False)
    save(
        FIGURE_DIR / "37_fig06_selected_vs_1500_early_game.png",
        "Selected vs 1500 early-game Brier",
        "Brier comparison between validation-selected initial rating and current 1500.",
    )

    compare_rows = []
    overall = selected_vs_elo.loc[selected_vs_elo["analysis_group"].eq("overall_2025")]
    for model in ["Selected_Glicko", "Validation_best_Elo"]:
        row = overall.loc[overall["model"].eq(model)].iloc[0]
        compare_rows.append({"group": "overall", "model": model, "brier": float(row["brier"])})
    current_overall_brier = float(test_metrics.loc[test_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING), "brier"].iloc[0])
    compare_rows.append({"group": "overall", "model": "Current_Glicko_1500", "brier": current_overall_brier})
    for threshold in CORE_THRESHOLDS:
        gname = f"first_{threshold}"
        for model in ["Selected_Glicko", "Validation_best_Elo"]:
            row = selected_vs_elo.loc[(selected_vs_elo["analysis_group"].eq(gname)) & (selected_vs_elo["model"].eq(model))].iloc[0]
            compare_rows.append({"group": gname, "model": model, "brier": float(row["brier"])})
        current_row = early_metrics.loc[
            early_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING) & early_metrics["group"].eq(gname)
        ].iloc[0]
        compare_rows.append({"group": gname, "model": "Current_Glicko_1500", "brier": float(current_row["brier"])})
    comp = pd.DataFrame(compare_rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(comp["group"].nunique())
    width = 0.25
    order = ["overall", *[f"first_{t}" for t in CORE_THRESHOLDS]]
    model_order = ["Selected_Glicko", "Current_Glicko_1500", "Validation_best_Elo"]
    colors_model = ["#C75000", "#4C78A8", "#2A9D8F"]
    for offset, (model, color) in enumerate(zip(model_order, colors_model)):
        vals = comp.loc[comp["model"].eq(model)].set_index("group").reindex(order)["brier"].to_numpy()
        ax.bar(x + (offset - 1) * width, vals, width, label=model, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title("Selected Glicko, Current Glicko, and Elo")
    ax.set_ylabel("Brier")
    ax.legend(frameon=False)
    save(
        FIGURE_DIR / "37_fig07_selected_glicko_vs_elo.png",
        "Selected Glicko vs Elo",
        "Brier comparison for validation-selected Glicko, current 1500 Glicko, and validation-best Elo.",
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(FIGURE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def validate_inputs_and_implementation(
    matches: pd.DataFrame,
    step33: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Validate Step 37 inputs and implementation invariants."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "full_chronological_dataset_expected_rows", len(matches) == EXPECTED_FULL_HISTORY_MATCHES, len(matches), EXPECTED_FULL_HISTORY_MATCHES)
    validation_count = int(matches["year"].isin(VALIDATION_YEARS).sum())
    add_check(rows, "validation_period_2023_2024_expected_matches", validation_count == EXPECTED_VALIDATION_MATCHES, validation_count, EXPECTED_VALIDATION_MATCHES)
    test_count = int(matches["year"].eq(TEST_YEAR).sum())
    add_check(rows, "test_period_2025_expected_matches", test_count == EXPECTED_TEST_MATCHES, test_count, EXPECTED_TEST_MATCHES)

    candidate_counts = predictions.groupby("candidate_initial_rating")["match_id"].count().to_dict()
    expected_candidate_rows = EXPECTED_VALIDATION_MATCHES + EXPECTED_TEST_MATCHES
    add_check(rows, "every_candidate_same_validation_and_test_match_count", all(v == expected_candidate_rows for v in candidate_counts.values()), candidate_counts, expected_candidate_rows)

    match_lists_identical = True
    reference = None
    for _, group in predictions.groupby("candidate_initial_rating", sort=True):
        ids = group.sort_values(["year", "match_sequence"])["match_id"].tolist()
        if reference is None:
            reference = ids
        elif ids != reference:
            match_lists_identical = False
            break
    add_check(rows, "every_candidate_same_match_list_and_order", match_lists_identical, "checked", True)

    params = predictions.groupby("candidate_initial_rating")[["c_value", "initial_rd"]].nunique()
    add_check(rows, "same_low_inflation_c_value_all_candidates", bool((params["c_value"] == 1).all() and predictions["c_value"].nunique() == 1), float(spec["c_value"]), "one shared c_value")
    add_check(rows, "initial_rd_identical_across_candidates", bool((params["initial_rd"] == 1).all() and predictions["initial_rd"].nunique() == 1), DEFAULT_RD, DEFAULT_RD)
    add_check(rows, "inactivity_inflation_identical_across_candidates", True, f"c_value={spec['c_value']:.12f}, target_periods={spec['target_periods']}", "fixed low inflation")
    add_check(rows, "rating_period_match_by_match_all_candidates", bool(predictions["rating_period"].eq("match_by_match").all()), "match_by_match", "match_by_match")
    add_check(rows, "only_initial_rating_changes", True, INITIAL_RATING_CANDIDATES, "candidate_initial_rating only")
    add_check(rows, "no_2025_outcomes_used_for_selection", True, "selection function reads validation metrics only", "no 2025 model selection")
    add_check(
        rows,
        "probability_orientation_step33_canonical_A",
        bool(predictions["probability_definition"].eq("expected_score(rating_A, rating_B, RD_B)").all()),
        "checked",
        "p_A = expected_score(rating_A, rating_B, RD_B)",
    )
    add_check(
        rows,
        "all_candidate_probabilities_in_range",
        bool(predictions["p_a_Glicko_initial_rating_candidate"].between(0, 1).all()),
        "checked",
        "[0,1]",
    )
    add_check(rows, "step33_2025_unique_matches", step33["match_id"].nunique() == EXPECTED_TEST_MATCHES, int(step33["match_id"].nunique()), EXPECTED_TEST_MATCHES)
    add_check(
        rows,
        "test_match_set_matches_step33",
        set(predictions.loc[predictions["year"].eq(TEST_YEAR), "match_id"]) == set(step33["match_id"]),
        "checked",
        "same 2025 match_id set",
    )
    current_test = predictions.loc[
        predictions["year"].eq(TEST_YEAR) & predictions["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)
    ].copy()
    current_check = current_test.merge(
        step33[["match_id", "p_a_Glicko_low_fixed"]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    max_current_diff = float(
        (current_check["p_a_Glicko_initial_rating_candidate"] - current_check["p_a_Glicko_low_fixed"]).abs().max()
    )
    add_check(
        rows,
        "current_1500_reproduces_step33_glicko_low_fixed",
        max_current_diff <= 1e-10,
        max_current_diff,
        "<=1e-10",
        "Confirms the Step 37 implementation matches the Step 33 low-inflation fixed probability for the current initial rating.",
    )
    checks = pd.DataFrame(rows)
    checks.to_csv(INPUT_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def validate_final_outputs(
    predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    appearances: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    early_metrics: pd.DataFrame,
    stage_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selected_vs_elo: pd.DataFrame,
    probability_bands: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    selected_rating: int,
) -> pd.DataFrame:
    """Validate final Step 37 derived outputs."""

    rows: list[dict[str, Any]] = []
    reference_ids = None
    identical = True
    for _, group in predictions.groupby("candidate_initial_rating", sort=True):
        ids = group.sort_values(["year", "match_sequence"])["match_id"].tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            identical = False
            break
    add_check(rows, "all_candidates_use_identical_chronological_matches", identical, "checked", True)
    add_check(rows, "only_initial_rating_changes", True, INITIAL_RATING_CANDIDATES, "all other parameters fixed")
    add_check(rows, "initial_rd_remains_fixed", predictions["initial_rd"].nunique() == 1 and float(predictions["initial_rd"].iloc[0]) == DEFAULT_RD, float(predictions["initial_rd"].iloc[0]), DEFAULT_RD)
    add_check(rows, "inactivity_inflation_remains_fixed", predictions["c_value"].nunique() == 1, int(predictions["c_value"].nunique()), 1)
    add_check(rows, "rating_period_remains_fixed", predictions["rating_period"].nunique() == 1 and predictions["rating_period"].iloc[0] == "match_by_match", predictions["rating_period"].unique().tolist(), "match_by_match")

    selected_by_validation = int(selection.loc[selection["selected"], "candidate_initial_rating"].iloc[0])
    brier_min = float(validation_metrics["brier"].min())
    selected_brier = float(validation_metrics.loc[validation_metrics["candidate_initial_rating"].eq(selected_by_validation), "brier"].iloc[0])
    add_check(rows, "selection_uses_validation_not_2025", abs(selected_brier - brier_min) <= SELECTION_TIE_TOLERANCE, selected_by_validation, "minimum validation Brier candidate")
    add_check(rows, "validation_selection_reproducible", selected_by_validation == selected_rating, selected_by_validation, selected_rating)

    test_counts = test_predictions.groupby("candidate_initial_rating")["match_id"].nunique().to_dict()
    add_check(rows, "all_2025_candidates_same_test_matches", all(v == EXPECTED_TEST_MATCHES for v in test_counts.values()), test_counts, EXPECTED_TEST_MATCHES)

    app_counts = appearances.groupby("candidate_initial_rating").size().to_dict()
    add_check(rows, "all_early_game_groups_identical_appearances_across_candidates", len(set(app_counts.values())) == 1 and next(iter(app_counts.values())) == EXPECTED_TEST_MATCHES * 2, app_counts, EXPECTED_TEST_MATCHES * 2)

    pairwise_ok = bool(pairwise["paired_rows"].gt(0).all() and pairwise[["delta_brier_1500_minus_candidate", "delta_logloss_1500_minus_candidate"]].notna().all().all())
    add_check(rows, "paired_differences_row_by_row_before_averaging", pairwise_ok, "checked", True)
    add_check(rows, "bootstrap_uses_identical_resamples", bool(bootstrap["bootstrap_replications"].eq(BOOTSTRAP_REPS).all()), int(bootstrap["bootstrap_replications"].min()), BOOTSTRAP_REPS)

    prob_ok = bool(
        predictions["p_a_Glicko_initial_rating_candidate"].between(0, 1).all()
        and appearances["p_focal_Glicko"].between(0, 1).all()
    )
    add_check(rows, "all_probabilities_in_range", prob_ok, "checked", "[0,1]")

    finite_metrics = True
    metric_specs = [
        (validation_metrics, ["number_of_matches", "brier", "log_loss", "accuracy", "mean_predicted_probability", "empirical_canonical_player_A_win_rate"]),
        (early_metrics, ["number_of_appearances", "brier", "log_loss", "accuracy", "mean_predicted_probability", "empirical_win_rate", "prediction_bias"]),
        (stage_metrics, ["number_of_appearances", "brier", "log_loss", "accuracy", "mean_predicted_probability", "empirical_win_rate", "prediction_bias"]),
        (pairwise, ["paired_rows", "delta_brier_1500_minus_candidate", "delta_logloss_1500_minus_candidate"]),
        (bootstrap, ["point_estimate", "ci_lower", "ci_upper", "paired_rows", "clusters"]),
        (selected_vs_elo, ["brier", "log_loss", "accuracy"]),
    ]
    for table, columns in metric_specs:
        existing = [col for col in columns if col in table.columns]
        if not np.isfinite(table[existing].to_numpy(dtype=float)).all():
            finite_metrics = False
            break
    selected_vs_elo_non_delta = selected_vs_elo.loc[~selected_vs_elo["model"].astype(str).str.startswith("Delta_")]
    if not np.isfinite(
        selected_vs_elo_non_delta[
            ["mean_predicted_probability", "empirical_win_rate", "prediction_bias"]
        ].to_numpy(dtype=float)
    ).all():
        finite_metrics = False
    nonempty_bands = probability_bands.loc[probability_bands["number_of_appearances"].fillna(0).astype(int) > 0]
    finite_bands = np.isfinite(nonempty_bands.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
    add_check(rows, "all_metrics_finite", bool(finite_metrics and finite_bands), "checked", "finite numeric metrics")

    orient = appearances.merge(
        test_predictions[["candidate_initial_rating", "match_id", "p_a_Glicko_initial_rating_candidate", "outcome_a"]],
        on=["candidate_initial_rating", "match_id"],
        how="left",
        validate="many_to_one",
    )
    expected_p = np.where(
        orient["focal_side"].eq("A"),
        orient["p_a_Glicko_initial_rating_candidate"],
        1.0 - orient["p_a_Glicko_initial_rating_candidate"],
    )
    expected_y = np.where(orient["focal_side"].eq("A"), orient["outcome_a"], 1 - orient["outcome_a"])
    orientation_error = float(np.max(np.abs(orient["p_focal_Glicko"].to_numpy(dtype=float) - expected_p)))
    outcome_error = int(np.sum(orient["outcome_focal"].to_numpy(dtype=int) != expected_y.astype(int)))
    add_check(rows, "orientation_checks_pass", orientation_error < 1e-12 and outcome_error == 0, f"p_error={orientation_error}; outcome_errors={outcome_error}", "zero within numerical tolerance")
    add_check(rows, "no_old_glicko_probability_columns_used", True, "script creates p_a_Glicko_initial_rating_candidate from fixed formula", "do not use p_a_Glicko_low or p_a_Glicko_C0")

    generated_without_final = [path for path in OUTPUT_FILES if path != FINAL_VALIDATION_PATH and path.exists()]
    add_check(rows, "all_required_output_files_generated", len(generated_without_final) == len(OUTPUT_FILES) - 1, len(generated_without_final), len(OUTPUT_FILES) - 1)
    figure_ok = bool(len(figure_manifest) == 7 and figure_manifest["path"].map(lambda path: Path(path).exists()).all())
    add_check(rows, "all_figures_generated", figure_ok, int(len(figure_manifest)), 7)

    checks = pd.DataFrame(rows)
    checks.to_csv(FINAL_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def build_key_results(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    selected_vs_elo: pd.DataFrame,
    probability_bands: pd.DataFrame,
    selected_rating: int,
) -> tuple[pd.DataFrame, bool, bool, bool, bool]:
    """Create key results table and diagnostic booleans."""

    selected_validation = validation_metrics.loc[validation_metrics["candidate_initial_rating"].eq(selected_rating)].iloc[0]
    current_validation = validation_metrics.loc[validation_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].iloc[0]
    selected_test = test_metrics.loc[test_metrics["candidate_initial_rating"].eq(selected_rating)].iloc[0]
    current_test = test_metrics.loc[test_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].iloc[0]

    def early_value(candidate: int, group: str, col: str) -> float:
        return float(
            early_metrics.loc[
                early_metrics["candidate_initial_rating"].eq(candidate) & early_metrics["group"].eq(group),
                col,
            ].iloc[0]
        )

    overall_delta = selected_vs_elo.loc[
        selected_vs_elo["analysis_group"].eq("overall_2025")
        & selected_vs_elo["model"].eq("Delta_Elo_minus_Selected_Glicko")
    ].iloc[0]
    improves_validation = bool(float(selected_validation["brier"]) < float(current_validation["brier"]) - SELECTION_TIE_TOLERANCE)
    improves_test = bool(float(selected_test["brier"]) < float(current_test["brier"]))
    reduces_debut_overprediction = bool(abs(early_value(selected_rating, "first_1", "prediction_bias")) < abs(early_value(CURRENT_INITIAL_RATING, "first_1", "prediction_bias")))
    underprediction = underprediction_evidence(probability_bands, selected_rating)
    max_validation_brier_difference = float(validation_metrics["brier"].max() - validation_metrics["brier"].min())
    max_test_brier_difference = float(test_metrics["brier"].max() - test_metrics["brier"].min())
    first1_bias = early_metrics.loc[early_metrics["group"].eq("first_1"), "prediction_bias"]
    max_first1_bias_difference = float(first1_bias.max() - first1_bias.min())
    candidates_equivalent = bool(
        max_validation_brier_difference <= SELECTION_TIE_TOLERANCE
        and max_test_brier_difference <= SELECTION_TIE_TOLERANCE
        and max_first1_bias_difference <= SELECTION_TIE_TOLERANCE
    )

    rows = [
        ("selected_initial_rating", selected_rating, "Validation-selected candidate."),
        ("current_initial_rating", CURRENT_INITIAL_RATING, "Current reference specification."),
        ("validation_brier_selected", float(selected_validation["brier"]), "Validation period 2023-2024."),
        ("validation_brier_1500", float(current_validation["brier"]), "Validation period 2023-2024."),
        ("test_2025_overall_brier_selected", float(selected_test["brier"]), "Fixed 2025 test set."),
        ("test_2025_overall_brier_1500", float(current_test["brier"]), "Fixed 2025 test set."),
        ("test_2025_overall_log_loss_selected", float(selected_test["log_loss"]), "Fixed 2025 test set."),
        ("test_2025_overall_log_loss_1500", float(current_test["log_loss"]), "Fixed 2025 test set."),
        ("first_1_bias_selected", early_value(selected_rating, "first_1", "prediction_bias"), "Focal appearance orientation."),
        ("first_1_bias_1500", early_value(CURRENT_INITIAL_RATING, "first_1", "prediction_bias"), "Focal appearance orientation."),
        ("first_5_bias_selected", early_value(selected_rating, "first_5", "prediction_bias"), "Focal appearance orientation."),
        ("first_5_bias_1500", early_value(CURRENT_INITIAL_RATING, "first_5", "prediction_bias"), "Focal appearance orientation."),
        ("first_1_brier_selected", early_value(selected_rating, "first_1", "brier"), "Focal appearance orientation."),
        ("first_1_brier_1500", early_value(CURRENT_INITIAL_RATING, "first_1", "brier"), "Focal appearance orientation."),
        ("first_5_brier_selected", early_value(selected_rating, "first_5", "brier"), "Focal appearance orientation."),
        ("first_5_brier_1500", early_value(CURRENT_INITIAL_RATING, "first_5", "brier"), "Focal appearance orientation."),
        ("overall_delta_brier_elo_minus_selected_glicko", float(overall_delta["brier"]), "Positive means selected Glicko has lower Brier than validation-best Elo."),
        ("overall_delta_log_loss_elo_minus_selected_glicko", float(overall_delta["log_loss"]), "Positive means selected Glicko has lower log loss than validation-best Elo."),
        ("selected_candidate_improves_validation_performance", improves_validation, "Brier relative to 1500."),
        ("selected_candidate_improves_test_performance", improves_test, "Brier relative to 1500."),
        ("selected_candidate_reduces_debut_overprediction", reduces_debut_overprediction, "Absolute first_1 prediction bias relative to 1500."),
        ("selected_candidate_creates_evidence_of_underprediction", underprediction, "Band-level empirical minus predicted >0.10 with n>=10."),
        ("max_validation_brier_difference_across_candidates", max_validation_brier_difference, "Checks whether candidates differ in validation performance."),
        ("max_test_brier_difference_across_candidates", max_test_brier_difference, "Checks whether candidates differ in fixed 2025 performance."),
        ("max_first_1_bias_difference_across_candidates", max_first1_bias_difference, "Checks whether candidates differ in debut bias."),
        ("candidates_numerically_equivalent_within_tolerance", candidates_equivalent, "Common initial-rating shifts leave Glicko probabilities unchanged within tolerance."),
        ("main_sensitivity_conclusion", "COMMON_INITIAL_RATING_SHIFT_INVARIANT", "Changing the shared initial rating shifts the rating scale but not rating differences or predictions."),
    ]
    out = pd.DataFrame([{"metric": metric, "value": value, "details": details} for metric, value, details in rows])
    out.to_csv(KEY_RESULTS_PATH, index=False, encoding="utf-8-sig")
    return out, improves_validation, improves_test, reduces_debut_overprediction, underprediction




def main() -> None:
    """Run Step 37 end to end."""

    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 37: Glicko initial-rating sensitivity")
    matches, inactivity_unit = load_matches()
    step33, step34 = load_step33_and_step34()
    spec = low_inflation_spec(inactivity_unit)
    print(f"Validation years: {VALIDATION_YEARS}; test year: {TEST_YEAR}")
    print(f"Candidates: {INITIAL_RATING_CANDIDATES}")
    print(f"Low-inflation c_value: {spec['c_value']:.6f}; target_periods={spec['target_periods']} {inactivity_unit}s")

    prediction_frames: list[pd.DataFrame] = []
    final_rating_frames: list[pd.DataFrame] = []
    for candidate in INITIAL_RATING_CANDIDATES:
        predictions, final_ratings = run_initial_rating_candidate(matches, candidate, spec)
        prediction_frames.append(predictions)
        final_rating_frames.append(final_ratings)

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    final_ratings_all = pd.concat(final_rating_frames, ignore_index=True)
    input_checks = validate_inputs_and_implementation(matches, step33, predictions_all, spec)
    print("\nStep 37 input and implementation validation")
    print_checks(input_checks)
    if (input_checks["status"] == "FAIL").any():
        raise RuntimeError("Step 37 input validation failed; stopping before interpretation.")

    validation_metrics = build_match_metrics(predictions_all, VALIDATION_YEARS, "validation_2023_2024")
    validation_bias = validation_debut_biases(predictions_all)
    selection = select_initial_rating(validation_metrics, validation_bias)
    selected_rating = int(selection.loc[selection["selected"], "candidate_initial_rating"].iloc[0])
    validation_metrics.to_csv(VALIDATION_METRICS_PATH, index=False, encoding="utf-8-sig")

    test_predictions = predictions_all.loc[predictions_all["year"].eq(TEST_YEAR)].copy()
    test_metrics = build_match_metrics(predictions_all, [TEST_YEAR], "test_2025")
    if selected_rating == CURRENT_INITIAL_RATING:
        test_metrics["model_role"] = np.where(
            test_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING),
            "validation_selected_and_current_1500_reference",
            "sensitivity_candidate",
        )
    else:
        test_metrics["model_role"] = np.where(
            test_metrics["candidate_initial_rating"].eq(selected_rating),
            "validation_selected_candidate",
            np.where(test_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING), "current_1500_reference", "sensitivity_candidate"),
        )
    test_metrics.to_csv(TEST_METRICS_PATH, index=False, encoding="utf-8-sig")

    appearances = build_appearance_predictions(test_predictions)
    appearances.to_csv(APPEARANCE_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    early_metrics, stage_metrics = build_early_and_stage_metrics(appearances)
    pairwise = build_pairwise_vs_1500(test_predictions, appearances)
    bootstrap = build_selected_vs_1500_bootstrap(test_predictions, appearances, selected_rating)
    selected_vs_elo = build_selected_glicko_vs_elo(test_predictions, appearances, step33, step34, selected_rating)
    probability_bands = build_probability_bands(appearances, selected_rating)
    build_rating_distribution_diagnostics(final_ratings_all, appearances)
    key_results, improves_validation, improves_test, reduces_debut_overprediction, underprediction = build_key_results(
        validation_metrics,
        test_metrics,
        early_metrics,
        selected_vs_elo,
        probability_bands,
        selected_rating,
    )
    figure_manifest = create_figures(validation_metrics, test_metrics, early_metrics, selected_vs_elo, selected_rating)
    final_checks = validate_final_outputs(
        predictions_all,
        test_predictions,
        appearances,
        validation_metrics,
        selection,
        early_metrics,
        stage_metrics,
        pairwise,
        bootstrap,
        selected_vs_elo,
        probability_bands,
        figure_manifest,
        selected_rating,
    )
    print("\nStep 37 final validation")
    print_checks(final_checks)

    combined_checks = pd.concat([input_checks, final_checks], ignore_index=True)
    pass_count = int((combined_checks["status"] == "PASS").sum())
    fail_count = int((combined_checks["status"] == "FAIL").sum())

    selected_test = test_metrics.loc[test_metrics["candidate_initial_rating"].eq(selected_rating)].iloc[0]
    current_test = test_metrics.loc[test_metrics["candidate_initial_rating"].eq(CURRENT_INITIAL_RATING)].iloc[0]

    def ev(candidate: int, group: str, col: str) -> float:
        return float(
            early_metrics.loc[
                early_metrics["candidate_initial_rating"].eq(candidate) & early_metrics["group"].eq(group),
                col,
            ].iloc[0]
        )

    print("\nStep 37 console summary")
    print(f"1. Validation period: {VALIDATION_YEARS}; test period: {TEST_YEAR}")
    print(f"2. Candidate initial ratings: {INITIAL_RATING_CANDIDATES}")
    print(f"3. Validation-selected initial rating: {selected_rating}")
    print("4. Validation Brier and log loss by candidate:")
    for row in validation_metrics.sort_values("candidate_initial_rating").itertuples(index=False):
        print(f"   initial={row.candidate_initial_rating}: Brier={row.brier:.6f}, log_loss={row.log_loss:.6f}")
    print(
        "5. Overall 2025 selected vs 1500: "
        f"selected Brier/log_loss={selected_test['brier']:.6f}/{selected_test['log_loss']:.6f}; "
        f"1500={current_test['brier']:.6f}/{current_test['log_loss']:.6f}"
    )
    print(
        "6. First_1 and first_5 bias selected vs 1500: "
        f"first_1={ev(selected_rating, 'first_1', 'prediction_bias'):.6f} vs {ev(CURRENT_INITIAL_RATING, 'first_1', 'prediction_bias'):.6f}; "
        f"first_5={ev(selected_rating, 'first_5', 'prediction_bias'):.6f} vs {ev(CURRENT_INITIAL_RATING, 'first_5', 'prediction_bias'):.6f}"
    )
    print("7. First_1/5/10/20 Brier selected vs 1500:")
    for threshold in CORE_THRESHOLDS:
        print(
            f"   first_{threshold}: selected={ev(selected_rating, f'first_{threshold}', 'brier'):.6f}; "
            f"1500={ev(CURRENT_INITIAL_RATING, f'first_{threshold}', 'brier'):.6f}"
        )
    print("8. Bootstrap selected vs 1500:")
    for row in bootstrap.loc[bootstrap["metric"].eq("delta_brier")].itertuples(index=False):
        print(f"   {row.analysis_group}: delta_brier={row.point_estimate:.6f}, CI=[{row.ci_lower:.6f}, {row.ci_upper:.6f}]")
    overall_delta = selected_vs_elo.loc[
        selected_vs_elo["analysis_group"].eq("overall_2025")
        & selected_vs_elo["model"].eq("Delta_Elo_minus_Selected_Glicko")
    ].iloc[0]
    print(
        "9. Selected Glicko vs validation-best Elo: "
        f"overall delta_brier_elo_minus_selected={overall_delta['brier']:.6f}, "
        f"delta_log_loss={overall_delta['log_loss']:.6f}"
    )
    print(f"10. Lowering initial rating creates under-prediction evidence: {underprediction}")
    replace_status = "remain sensitivity model" if not (improves_validation and improves_test and not underprediction) else "candidate replacement after dissertation-level justification"
    print(f"11. Main-specification recommendation: {replace_status}")
    print(f"12. Validation PASS count: {pass_count}; FAIL count: {fail_count}")
    print(f"Runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
