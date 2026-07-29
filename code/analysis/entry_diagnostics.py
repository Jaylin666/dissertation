"""Meeting 8 Step 42: prematch entry-scale and cross-file audit.

This supplementary diagnostic uses the exact Step 24 model-processing order to
identify every player's first recorded appearance, measures the contemporaneous
rating scale immediately before that appearance, reconciles the 2025 definition
with Steps 33 and 34, and audits a direct-focal probability sensitivity.

It does not create a new rating model, tune parameters, or replace the frozen
Meeting 7 probability convention.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLICKO_PIPELINE_PATH = PROJECT_ROOT / "code" / "pipelines" / "glicko_pipeline.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting8_technical"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP33_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "meeting6"
    / "33_orientation_corrected_per_match_scores_2025.csv"
)
STEP34_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "meeting7"
    / "34_early_game_appearance_dataset.csv"
)
STEP41_CLASSIFICATION_PATH = OUTPUT_DIR / "41_player_entry_classification.csv"
STEP41_YEARLY_SCALE_PATH = OUTPUT_DIR / "41_yearly_rating_scale_drift.csv"

STRICT_ENTRY_PATH = OUTPUT_DIR / "42_strict_player_entry_classification.csv"
STEP41_MISMATCH_PATH = OUTPUT_DIR / "42_step41_entry_classification_mismatches.csv"
PREMATCH_DIAGNOSTIC_PATH = (
    OUTPUT_DIR / "42_prematch_contemporaneous_scale_diagnostics.csv"
)
ENTRY_YEAR_SUMMARY_PATH = OUTPUT_DIR / "42_entry_year_scale_summary.csv"
ENTRY_COHORT_SUMMARY_PATH = OUTPUT_DIR / "42_entry_cohort_scale_summary.csv"
CROSSFILE_AUDIT_PATH = OUTPUT_DIR / "42_2025_crossfile_entry_audit.csv"
ORIENTATION_SENSITIVITY_PATH = (
    OUTPUT_DIR / "42_probability_orientation_sensitivity.csv"
)
BURN_IN_SENSITIVITY_PATH = OUTPUT_DIR / "42_burnin_sensitivity_audit.csv"
VALIDATION_PATH = OUTPUT_DIR / "42_validation_checks.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "42_figure_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "42_prematch_entry_scale_summary.md"

FIGURE_1_PATH = (
    FIGURE_DIR / "42_fig01_entry_anchor_vs_contemporaneous_scale.png"
)
FIGURE_2_PATH = FIGURE_DIR / "42_fig02_entry_anchor_vs_actual_opponent.png"
FIGURE_3_PATH = FIGURE_DIR / "42_fig03_historical_orientation_sensitivity.png"

MODEL_START_YEAR = 1985
POST_BURN_IN_START_YEAR = 1990
TEST_YEAR = 2025
INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
EPS = 1e-15

EXPECTED_FULL_HISTORY_ROWS = 456_382
EXPECTED_UNIQUE_PLAYERS = 5_143
EXPECTED_SYSTEM_START_PLAYERS = 314
EXPECTED_WITHIN_BURN_IN_PLAYERS = 456
EXPECTED_POST_BURN_IN_1990_2024_PLAYERS = 4_297
EXPECTED_TEST_YEAR_ENTRANTS = 76
EXPECTED_STEP33_ROWS = 11_379
EXPECTED_STEP34_ROWS = 22_758
EXPECTED_DEBUT_APPEARANCES = 76
EXPECTED_DEBUT_PLAYERS = 76
EXPECTED_DEBUT_MATCHES = 74
EXPECTED_EXACTLY_ONE_DEBUT_MATCHES = 72
EXPECTED_BOTH_DEBUT_MATCHES = 2


def configure_output_root(output_root: str | Path) -> Path:
    """Redirect Step 42 outputs and its Step 41 inputs to a validation root."""

    global OUTPUT_DIR, FIGURE_DIR
    global STEP41_CLASSIFICATION_PATH, STEP41_YEARLY_SCALE_PATH
    global STRICT_ENTRY_PATH, STEP41_MISMATCH_PATH, PREMATCH_DIAGNOSTIC_PATH
    global ENTRY_YEAR_SUMMARY_PATH, ENTRY_COHORT_SUMMARY_PATH
    global CROSSFILE_AUDIT_PATH, ORIENTATION_SENSITIVITY_PATH
    global BURN_IN_SENSITIVITY_PATH, VALIDATION_PATH, FIGURE_MANIFEST_PATH
    global SUMMARY_PATH, FIGURE_1_PATH, FIGURE_2_PATH, FIGURE_3_PATH

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    OUTPUT_DIR = root.resolve() / "meeting8_technical"
    FIGURE_DIR = OUTPUT_DIR / "figures"
    STEP41_CLASSIFICATION_PATH = OUTPUT_DIR / "41_player_entry_classification.csv"
    STEP41_YEARLY_SCALE_PATH = OUTPUT_DIR / "41_yearly_rating_scale_drift.csv"
    STRICT_ENTRY_PATH = OUTPUT_DIR / "42_strict_player_entry_classification.csv"
    STEP41_MISMATCH_PATH = OUTPUT_DIR / "42_step41_entry_classification_mismatches.csv"
    PREMATCH_DIAGNOSTIC_PATH = OUTPUT_DIR / "42_prematch_contemporaneous_scale_diagnostics.csv"
    ENTRY_YEAR_SUMMARY_PATH = OUTPUT_DIR / "42_entry_year_scale_summary.csv"
    ENTRY_COHORT_SUMMARY_PATH = OUTPUT_DIR / "42_entry_cohort_scale_summary.csv"
    CROSSFILE_AUDIT_PATH = OUTPUT_DIR / "42_2025_crossfile_entry_audit.csv"
    ORIENTATION_SENSITIVITY_PATH = OUTPUT_DIR / "42_probability_orientation_sensitivity.csv"
    BURN_IN_SENSITIVITY_PATH = OUTPUT_DIR / "42_burnin_sensitivity_audit.csv"
    VALIDATION_PATH = OUTPUT_DIR / "42_validation_checks.csv"
    FIGURE_MANIFEST_PATH = OUTPUT_DIR / "42_figure_manifest.csv"
    SUMMARY_PATH = OUTPUT_DIR / "42_prematch_entry_scale_summary.md"
    FIGURE_1_PATH = FIGURE_DIR / "42_fig01_entry_anchor_vs_contemporaneous_scale.png"
    FIGURE_2_PATH = FIGURE_DIR / "42_fig02_entry_anchor_vs_actual_opponent.png"
    FIGURE_3_PATH = FIGURE_DIR / "42_fig03_historical_orientation_sensitivity.png"
    return OUTPUT_DIR


def load_step24_module() -> Any:
    """Return the canonical validated Step 24 implementation."""

    from code.pipelines import glicko_pipeline

    return glicko_pipeline


def robust_bool_value(value: Any) -> bool:
    """Convert common boolean representations without relying on truthiness."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1"}:
            return True
        if normalised in {"false", "0"}:
            return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def robust_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    """Convert a complete series with explicit validation."""

    if series.isna().any():
        raise ValueError(f"{column_name} contains missing boolean values")
    try:
        return series.map(robust_bool_value).astype(bool)
    except ValueError as exc:
        raise ValueError(f"Could not convert {column_name} to boolean") from exc


def cohort_for_year(first_year: int) -> str:
    """Assign the mutually exclusive primary entry cohort."""

    if first_year == MODEL_START_YEAR:
        return "system_start_left_censored"
    if first_year < POST_BURN_IN_START_YEAR:
        return "within_5y_burn_in_recorded_entry"
    if first_year < TEST_YEAR:
        return "post_burn_in_recorded_entry"
    if first_year == TEST_YEAR:
        return "test_year_recorded_entry"
    raise ValueError(f"Unexpected first recorded year: {first_year}")


def evidence_status_for_year(year: int) -> str:
    """Return a cautious evidence label for one entry year."""

    if year == MODEL_START_YEAR:
        return "model-start left-censored descriptive context"
    if year < POST_BURN_IN_START_YEAR:
        return "burn-in descriptive context"
    if year < TEST_YEAR:
        return "in-sample descriptive mechanism evidence"
    return "held-out test evidence"


def safe_log_loss_scalar(probability: float, outcome: int) -> float:
    """Return binary log loss for one observation with safe clipping."""

    p = float(np.clip(float(probability), EPS, 1.0 - EPS))
    y = float(outcome)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    severity: str,
    detail: str,
) -> None:
    """Append a machine-readable validation check."""

    if severity not in {"error", "warning"}:
        raise ValueError(f"Unsupported severity: {severity}")
    rows.append(
        {
            "check_name": check_name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "detail": detail,
        }
    )


def require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when a frozen input lacks required columns."""

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_strict_appearance_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Build ordered winner and loser appearance rows from frozen match order."""

    carry = [
        "fcode",
        "match_sequence",
        "year",
        "event",
        "event_order_date",
        "inactivity_period_index",
    ]
    winner = matches[carry + ["winner", "loser"]].copy()
    winner = winner.rename(columns={"winner": "player_id", "loser": "opponent_id"})
    winner["outcome_focal"] = 1
    winner["appearance_role"] = "winner"

    loser = matches[carry + ["winner", "loser"]].copy()
    loser = loser.rename(columns={"loser": "player_id", "winner": "opponent_id"})
    loser["outcome_focal"] = 0
    loser["appearance_role"] = "loser"

    long = pd.concat([winner, loser], ignore_index=True)
    long = long.rename(columns={"fcode": "match_id"})
    long["fcode"] = long["match_id"]
    for column in [
        "match_id",
        "fcode",
        "match_sequence",
        "year",
        "player_id",
        "opponent_id",
    ]:
        long[column] = pd.to_numeric(long[column], errors="raise").astype(int)
    long["event_order_date"] = pd.to_datetime(
        long["event_order_date"], errors="coerce"
    )
    return long


def build_strict_player_classification(
    appearance_long: pd.DataFrame,
) -> pd.DataFrame:
    """Identify first recorded appearances by minimum frozen match_sequence."""

    first_indices = appearance_long.groupby("player_id", sort=False)[
        "match_sequence"
    ].idxmin()
    first = appearance_long.loc[first_indices].copy()
    counts = (
        appearance_long.groupby("player_id", sort=False)
        .size()
        .rename("recorded_matches_1985_2025")
    )
    first = first.merge(counts, on="player_id", how="left", validate="one_to_one")
    first = first.rename(
        columns={
            "match_id": "first_match_id",
            "match_sequence": "first_match_sequence",
            "year": "first_recorded_year",
            "event_order_date": "first_recorded_date",
            "event": "first_event_id",
            "opponent_id": "first_opponent_id",
            "outcome_focal": "first_outcome",
            "appearance_role": "first_appearance_role",
        }
    )
    first["entry_cohort"] = first["first_recorded_year"].map(cohort_for_year)
    first["is_model_start_player"] = first["first_recorded_year"].eq(
        MODEL_START_YEAR
    )
    first["is_within_primary_burn_in"] = first["first_recorded_year"].lt(
        POST_BURN_IN_START_YEAR
    )
    first["is_post_burn_in_recorded_entry"] = first["first_recorded_year"].ge(
        POST_BURN_IN_START_YEAR
    )
    first["is_test_year_recorded_entry"] = first["first_recorded_year"].eq(
        TEST_YEAR
    )

    columns = [
        "player_id",
        "first_match_id",
        "first_match_sequence",
        "first_recorded_year",
        "first_recorded_date",
        "first_event_id",
        "first_opponent_id",
        "first_outcome",
        "first_appearance_role",
        "recorded_matches_1985_2025",
        "entry_cohort",
        "is_model_start_player",
        "is_within_primary_burn_in",
        "is_post_burn_in_recorded_entry",
        "is_test_year_recorded_entry",
    ]
    return first[columns].sort_values(
        ["first_match_sequence", "player_id"]
    ).reset_index(drop=True)


def compare_with_step41(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare strict processing-order classification with existing Step 41."""

    mismatch_columns = [
        "player_id",
        "strict_first_recorded_year",
        "step41_first_recorded_year",
        "strict_first_match_id",
        "step41_first_match_id",
        "strict_entry_cohort",
        "step41_entry_cohort",
        "year_agrees",
        "match_id_agrees",
        "cohort_agrees",
        "mismatch_explanation",
    ]
    if not STEP41_CLASSIFICATION_PATH.exists():
        return pd.DataFrame(columns=mismatch_columns), {
            "step41_available": False,
            "matched_players": 0,
            "mismatch_rows": 0,
            "all_agree": False,
        }

    step41 = pd.read_csv(STEP41_CLASSIFICATION_PATH, low_memory=False)
    required = [
        "player_id",
        "first_recorded_year",
        "first_recorded_match_id",
        "entry_cohort",
    ]
    require_columns(step41, required, "Step 41 classification")
    left = strict[
        ["player_id", "first_recorded_year", "first_match_id", "entry_cohort"]
    ].rename(
        columns={
            "first_recorded_year": "strict_first_recorded_year",
            "first_match_id": "strict_first_match_id",
            "entry_cohort": "strict_entry_cohort",
        }
    )
    right = step41[required].rename(
        columns={
            "first_recorded_year": "step41_first_recorded_year",
            "first_recorded_match_id": "step41_first_match_id",
            "entry_cohort": "step41_entry_cohort",
        }
    )
    comparison = left.merge(
        right, on="player_id", how="outer", indicator=True, validate="one_to_one"
    )
    comparison["year_agrees"] = (
        comparison["strict_first_recorded_year"]
        == comparison["step41_first_recorded_year"]
    )
    comparison["match_id_agrees"] = (
        comparison["strict_first_match_id"] == comparison["step41_first_match_id"]
    )
    comparison["cohort_agrees"] = (
        comparison["strict_entry_cohort"] == comparison["step41_entry_cohort"]
    )
    comparison["mismatch_explanation"] = np.select(
        [
            comparison["_merge"].eq("left_only"),
            comparison["_merge"].eq("right_only"),
            ~comparison["match_id_agrees"],
            ~comparison["year_agrees"],
            ~comparison["cohort_agrees"],
        ],
        [
            "Player is present only in the strict Step 42 classification.",
            "Player is present only in the Step 41 classification.",
            "First match differs; inspect the independent Step 41 ordering logic.",
            "First recorded year differs as a consequence of first-match disagreement.",
            "Entry cohort differs as a consequence of year classification.",
        ],
        default="",
    )
    mismatches = comparison.loc[
        comparison["_merge"].ne("both")
        | ~comparison["year_agrees"]
        | ~comparison["match_id_agrees"]
        | ~comparison["cohort_agrees"]
    ].copy()
    mismatches = mismatches[mismatch_columns]
    return mismatches, {
        "step41_available": True,
        "matched_players": int(comparison["_merge"].eq("both").sum()),
        "mismatch_rows": int(len(mismatches)),
        "all_agree": bool(len(mismatches) == 0),
    }


def distribution_statistics(values: list[float]) -> dict[str, float]:
    """Return the required prematch scale statistics."""

    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def run_frozen_low_inflation_state(
    matches: pd.DataFrame,
    strict_classification: pd.DataFrame,
    step24: Any,
    low_variant: dict[str, Any],
) -> pd.DataFrame:
    """Run the frozen model and save only first-appearance diagnostics."""

    c_value = float(low_variant["c_value"])
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_period_index: dict[int, int] = {}
    games_before: defaultdict[int, int] = defaultdict(int)
    last_recorded_date: dict[int, pd.Timestamp] = {}

    first_year = strict_classification.set_index("player_id")[
        "first_recorded_year"
    ].astype(int).to_dict()
    first_sequence = strict_classification.set_index("player_id")[
        "first_match_sequence"
    ].astype(int).to_dict()

    rows: list[dict[str, Any]] = []
    simulation_columns = [
        "fcode",
        "match_sequence",
        "year",
        "event",
        "event_order_date",
        "inactivity_period_index",
        "winner",
        "loser",
    ]

    for match in matches[simulation_columns].itertuples(index=False):
        match_id = int(match.fcode)
        match_sequence = int(match.match_sequence)
        year = int(match.year)
        winner = int(match.winner)
        loser = int(match.loser)
        period_index = int(match.inactivity_period_index)
        current_date = pd.to_datetime(match.event_order_date, errors="coerce")

        winner_is_first = games_before[winner] == 0
        loser_is_first = games_before[loser] == 0

        if winner not in ratings:
            ratings[winner] = INITIAL_RATING
            rds[winner] = INITIAL_RD
        else:
            elapsed = period_index - last_period_index[winner]
            rds[winner] = step24.inflate_rd_for_inactivity(
                rds[winner], elapsed, c_value
            )

        if loser not in ratings:
            ratings[loser] = INITIAL_RATING
            rds[loser] = INITIAL_RD
        else:
            elapsed = period_index - last_period_index[loser]
            rds[loser] = step24.inflate_rd_for_inactivity(
                rds[loser], elapsed, c_value
            )

        winner_rating = float(ratings[winner])
        winner_rd = float(rds[winner])
        loser_rating = float(ratings[loser])
        loser_rd = float(rds[loser])

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        if player_a == winner:
            rating_a, rating_b, rd_b = winner_rating, loser_rating, loser_rd
        else:
            rating_a, rating_b, rd_b = loser_rating, winner_rating, winner_rd
        p_a_current = step24.expected_score(rating_a, rating_b, rd_b)

        if winner_is_first or loser_is_first:
            established_ids = [
                player for player in ratings if games_before[player] > 0
            ]
            established_values = [ratings[player] for player in established_ids]
            established_stats = distribution_statistics(established_values)
            contemporaneous_available = len(established_values) > 0

            if pd.notna(current_date):
                active_ids = []
                for player in established_ids:
                    prior_date = last_recorded_date.get(player)
                    if prior_date is None or pd.isna(prior_date):
                        continue
                    elapsed_days = (current_date - prior_date).days
                    if 0 <= elapsed_days <= 365:
                        active_ids.append(player)
                active_values = [ratings[player] for player in active_ids]
                active_stats = distribution_statistics(active_values)
                active_available = len(active_values) > 0
            else:
                active_ids = []
                active_stats = distribution_statistics([])
                active_available = False

            states = {
                winner: {
                    "opponent_id": loser,
                    "outcome": 1,
                    "rating": winner_rating,
                    "rd": winner_rd,
                    "opponent_rating": loser_rating,
                    "opponent_rd": loser_rd,
                    "opponent_games_before": games_before[loser],
                    "is_first": winner_is_first,
                },
                loser: {
                    "opponent_id": winner,
                    "outcome": 0,
                    "rating": loser_rating,
                    "rd": loser_rd,
                    "opponent_rating": winner_rating,
                    "opponent_rd": winner_rd,
                    "opponent_games_before": games_before[winner],
                    "is_first": loser_is_first,
                },
            }

            for focal, state in states.items():
                if not state["is_first"]:
                    continue
                focal_side = "A" if focal == player_a else "B"
                p_current = (
                    float(p_a_current) if focal == player_a else 1.0 - float(p_a_current)
                )
                p_direct = float(
                    step24.expected_score(
                        state["rating"],
                        state["opponent_rating"],
                        state["opponent_rd"],
                    )
                )
                outcome = int(state["outcome"])
                recorded_year = int(first_year[focal])
                if first_sequence[focal] != match_sequence:
                    raise RuntimeError(
                        f"Strict entry sequence mismatch for player {focal}: "
                        f"{first_sequence[focal]} != {match_sequence}"
                    )
                rows.append(
                    {
                        "match_id": match_id,
                        "match_sequence": match_sequence,
                        "year": year,
                        "event": match.event,
                        "event_order_date": current_date,
                        "player_id": focal,
                        "opponent_id": int(state["opponent_id"]),
                        "focal_side": focal_side,
                        "outcome_focal": outcome,
                        "first_recorded_year": recorded_year,
                        "entry_cohort": cohort_for_year(recorded_year),
                        "post_primary_burn_in": bool(
                            recorded_year >= POST_BURN_IN_START_YEAR
                        ),
                        "both_players_first_recorded": bool(
                            winner_is_first and loser_is_first
                        ),
                        "focal_rating_before": float(state["rating"]),
                        "focal_rd_before": float(state["rd"]),
                        "opponent_rating_before": float(state["opponent_rating"]),
                        "opponent_rd_before": float(state["opponent_rd"]),
                        "opponent_games_before": int(
                            state["opponent_games_before"]
                        ),
                        "p_focal_current_convention": p_current,
                        "p_focal_direct": p_direct,
                        "probability_difference_direct_minus_current": float(
                            p_direct - p_current
                        ),
                        "brier_current": float((p_current - outcome) ** 2),
                        "brier_direct": float((p_direct - outcome) ** 2),
                        "logloss_current": safe_log_loss_scalar(p_current, outcome),
                        "logloss_direct": safe_log_loss_scalar(p_direct, outcome),
                        "n_contemporaneous_established_players": int(
                            len(established_values)
                        ),
                        "contemporaneous_established_mean_rating": established_stats[
                            "mean"
                        ],
                        "contemporaneous_established_median_rating": established_stats[
                            "median"
                        ],
                        "contemporaneous_established_p10_rating": established_stats[
                            "p10"
                        ],
                        "contemporaneous_established_p90_rating": established_stats[
                            "p90"
                        ],
                        "initial_minus_contemporaneous_mean": float(
                            INITIAL_RATING - established_stats["mean"]
                        )
                        if contemporaneous_available
                        else float("nan"),
                        "initial_minus_contemporaneous_median": float(
                            INITIAL_RATING - established_stats["median"]
                        )
                        if contemporaneous_available
                        else float("nan"),
                        "contemporaneous_scale_available": bool(
                            contemporaneous_available
                        ),
                        "n_active_established_365d": int(len(active_ids)),
                        "active_established_365d_mean_rating": active_stats["mean"],
                        "active_established_365d_median_rating": active_stats[
                            "median"
                        ],
                        "initial_minus_active_established_365d_median": float(
                            INITIAL_RATING - active_stats["median"]
                        )
                        if active_available
                        else float("nan"),
                        "active_scale_available": bool(active_available),
                        "initial_minus_opponent_rating": float(
                            INITIAL_RATING - state["opponent_rating"]
                        ),
                        "focal_minus_opponent_rating": float(
                            state["rating"] - state["opponent_rating"]
                        ),
                        "primary_probability_convention": (
                            "canonical player-A probability; complement for player B"
                        ),
                        "direct_probability_role": (
                            "historical orientation sensitivity only"
                        ),
                        "low_inflation_c": c_value,
                    }
                )

        update = step24.update_two_players_single_game(
            winner_rating,
            winner_rd,
            loser_rating,
            loser_rd,
            1.0,
        )
        ratings[winner] = float(update.player1_rating_after)
        rds[winner] = float(update.player1_rd_after)
        ratings[loser] = float(update.player2_rating_after)
        rds[loser] = float(update.player2_rd_after)

        last_period_index[winner] = max(
            last_period_index.get(winner, period_index), period_index
        )
        last_period_index[loser] = max(
            last_period_index.get(loser, period_index), period_index
        )
        games_before[winner] += 1
        games_before[loser] += 1
        if pd.notna(current_date):
            last_recorded_date[winner] = current_date
            last_recorded_date[loser] = current_date

    return pd.DataFrame(rows).sort_values(
        ["match_sequence", "player_id"]
    ).reset_index(drop=True)


def summarise_entry_group(
    group: pd.DataFrame,
    group_label: str,
    evidence_status: str,
) -> dict[str, Any]:
    """Summarise probabilities and prematch scales for one year or cohort."""

    both_new_matches = group.loc[
        group["both_players_first_recorded"], "match_id"
    ].nunique()
    return {
        "group": group_label,
        "evidence_status": evidence_status,
        "n_first_appearances": int(len(group)),
        "n_unique_players": int(group["player_id"].nunique()),
        "n_unique_matches": int(group["match_id"].nunique()),
        "n_both_new_matches": int(both_new_matches),
        "mean_current_probability": float(
            group["p_focal_current_convention"].mean()
        ),
        "mean_direct_probability": float(group["p_focal_direct"].mean()),
        "empirical_win_rate": float(group["outcome_focal"].mean()),
        "current_prediction_bias": float(
            group["p_focal_current_convention"].mean()
            - group["outcome_focal"].mean()
        ),
        "direct_prediction_bias": float(
            group["p_focal_direct"].mean() - group["outcome_focal"].mean()
        ),
        "current_brier": float(group["brier_current"].mean()),
        "direct_brier": float(group["brier_direct"].mean()),
        "current_log_loss": float(group["logloss_current"].mean()),
        "direct_log_loss": float(group["logloss_direct"].mean()),
        "mean_opponent_rating": float(group["opponent_rating_before"].mean()),
        "median_opponent_rating": float(group["opponent_rating_before"].median()),
        "mean_initial_minus_opponent_rating": float(
            group["initial_minus_opponent_rating"].mean()
        ),
        "median_initial_minus_opponent_rating": float(
            group["initial_minus_opponent_rating"].median()
        ),
        "mean_contemporaneous_established_rating": float(
            group["contemporaneous_established_mean_rating"].mean()
        ),
        "median_contemporaneous_established_rating": float(
            group["contemporaneous_established_median_rating"].median()
        ),
        "mean_initial_minus_contemporaneous_median": float(
            group["initial_minus_contemporaneous_median"].mean()
        ),
        "median_initial_minus_contemporaneous_median": float(
            group["initial_minus_contemporaneous_median"].median()
        ),
        "mean_active_established_365d_median": float(
            group["active_established_365d_median_rating"].mean()
        ),
        "mean_initial_minus_active_established_365d_median": float(
            group["initial_minus_active_established_365d_median"].mean()
        ),
        "mean_probability_difference_direct_minus_current": float(
            group["probability_difference_direct_minus_current"].mean()
        ),
        "mean_contemporaneous_pool_size": float(
            group["n_contemporaneous_established_players"].mean()
        ),
        "share_contemporaneous_scale_available": float(
            group["contemporaneous_scale_available"].mean()
        ),
        "share_active_scale_available": float(
            group["active_scale_available"].mean()
        ),
    }


def build_entry_year_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarise strict first appearances by first recorded year."""

    rows = []
    for year, group in diagnostics.groupby("first_recorded_year", sort=True):
        year_int = int(year)
        row = summarise_entry_group(
            group,
            str(year_int),
            evidence_status_for_year(year_int),
        )
        row["year"] = year_int
        row["entry_cohort"] = cohort_for_year(year_int)
        rows.append(row)
    result = pd.DataFrame(rows)
    columns = ["year", "entry_cohort", "evidence_status"] + [
        column
        for column in result.columns
        if column not in {"year", "entry_cohort", "evidence_status", "group"}
    ]
    return result[columns]


def build_entry_cohort_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarise the four primary cohorts and all post-burn-in entries."""

    specifications = [
        (
            "system_start_left_censored",
            diagnostics["entry_cohort"].eq("system_start_left_censored"),
            "model-start left-censored descriptive context",
        ),
        (
            "within_5y_burn_in_recorded_entry",
            diagnostics["entry_cohort"].eq(
                "within_5y_burn_in_recorded_entry"
            ),
            "burn-in descriptive context",
        ),
        (
            "post_burn_in_recorded_entry",
            diagnostics["entry_cohort"].eq("post_burn_in_recorded_entry"),
            "in-sample descriptive mechanism evidence",
        ),
        (
            "test_year_recorded_entry",
            diagnostics["entry_cohort"].eq("test_year_recorded_entry"),
            "held-out test evidence",
        ),
        (
            "all_post_burn_in_recorded_entries_1990_2025",
            diagnostics["first_recorded_year"].ge(POST_BURN_IN_START_YEAR),
            "mixed historical descriptive and held-out evidence",
        ),
    ]
    rows = [
        summarise_entry_group(diagnostics.loc[mask], label, evidence)
        for label, mask, evidence in specifications
    ]
    return pd.DataFrame(rows)


def build_orientation_sensitivity(
    cohort_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact current-versus-direct cohort sensitivity table."""

    result = cohort_summary[
        [
            "group",
            "evidence_status",
            "n_first_appearances",
            "mean_current_probability",
            "mean_direct_probability",
            "empirical_win_rate",
            "current_prediction_bias",
            "direct_prediction_bias",
            "current_brier",
            "direct_brier",
            "current_log_loss",
            "direct_log_loss",
            "mean_probability_difference_direct_minus_current",
        ]
    ].copy()
    result["current_overprediction_direction"] = (
        result["current_prediction_bias"] > 0
    )
    result["direct_overprediction_direction"] = (
        result["direct_prediction_bias"] > 0
    )
    result["qualitative_direction_agrees"] = (
        result["current_overprediction_direction"]
        == result["direct_overprediction_direction"]
    )
    result["interpretation_limit"] = (
        "Direct-focal probabilities are sensitivity evidence only; the current "
        "Meeting 7 convention remains primary."
    )
    return result


def load_and_audit_step33() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load Step 33 and create its long 2025 debut table."""

    step33 = pd.read_csv(STEP33_PATH, low_memory=False)
    required = [
        "match_id",
        "match_sequence",
        "year",
        "player_a_id",
        "player_b_id",
        "a_is_debut",
        "b_is_debut",
    ]
    require_columns(step33, required, "Step 33")
    for column in [
        "match_id",
        "match_sequence",
        "year",
        "player_a_id",
        "player_b_id",
    ]:
        step33[column] = pd.to_numeric(step33[column], errors="raise").astype(int)
    step33["a_is_debut_bool"] = robust_bool_series(
        step33["a_is_debut"], "Step 33 a_is_debut"
    )
    step33["b_is_debut_bool"] = robust_bool_series(
        step33["b_is_debut"], "Step 33 b_is_debut"
    )

    a = step33.loc[
        step33["a_is_debut_bool"],
        ["match_id", "match_sequence", "year", "player_a_id", "player_b_id"],
    ].copy()
    a = a.rename(
        columns={"player_a_id": "player_id", "player_b_id": "opponent_id"}
    )
    a["focal_side"] = "A"

    b = step33.loc[
        step33["b_is_debut_bool"],
        ["match_id", "match_sequence", "year", "player_b_id", "player_a_id"],
    ].copy()
    b = b.rename(
        columns={"player_b_id": "player_id", "player_a_id": "opponent_id"}
    )
    b["focal_side"] = "B"
    debut = pd.concat([a, b], ignore_index=True).sort_values(
        ["match_sequence", "player_id"]
    )

    debut_count_per_match = (
        step33["a_is_debut_bool"].astype(int)
        + step33["b_is_debut_bool"].astype(int)
    )
    metrics = {
        "rows": int(len(step33)),
        "match_id_unique": bool(step33["match_id"].is_unique),
        "all_2025": bool(step33["year"].eq(TEST_YEAR).all()),
        "players_distinct": bool(
            step33["player_a_id"].ne(step33["player_b_id"]).all()
        ),
        "debut_appearance_rows": int(len(debut)),
        "unique_debut_players": int(debut["player_id"].nunique()),
        "unique_debut_matches": int(debut["match_id"].nunique()),
        "exactly_one_debut_matches": int(debut_count_per_match.eq(1).sum()),
        "both_debut_matches": int(debut_count_per_match.eq(2).sum()),
    }
    return step33, metrics, debut.reset_index(drop=True)


def load_and_audit_step34() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load Step 34 and validate its two equivalent first-appearance formulas."""

    step34 = pd.read_csv(STEP34_PATH, low_memory=False)
    required = [
        "match_id",
        "match_sequence",
        "year",
        "focal_side",
        "player_id",
        "opponent_id",
        "total_games_before",
        "debut_flag",
        "appearance_number",
        "first_1",
        "p_focal_Glicko_low_fixed",
    ]
    require_columns(step34, required, "Step 34")
    for column in [
        "match_id",
        "match_sequence",
        "year",
        "player_id",
        "opponent_id",
        "total_games_before",
        "appearance_number",
    ]:
        step34[column] = pd.to_numeric(step34[column], errors="raise").astype(int)
    step34["debut_flag_bool"] = robust_bool_series(
        step34["debut_flag"], "Step 34 debut_flag"
    )
    step34["first_1_bool"] = robust_bool_series(
        step34["first_1"], "Step 34 first_1"
    )

    by_flag = step34.loc[step34["debut_flag_bool"]].copy()
    by_appearance = step34.loc[step34["appearance_number"].eq(1)].copy()
    flag_keys = set(zip(by_flag["match_id"], by_flag["player_id"]))
    appearance_keys = set(
        zip(by_appearance["match_id"], by_appearance["player_id"])
    )
    per_match = step34.groupby("match_id").size()
    metrics = {
        "rows": int(len(step34)),
        "every_match_two_rows": bool(per_match.eq(2).all()),
        "debut_formula_correct": bool(
            step34["debut_flag_bool"].eq(step34["total_games_before"].eq(0)).all()
        ),
        "appearance_number_formula_correct": bool(
            step34["appearance_number"]
            .eq(step34["total_games_before"] + 1)
            .all()
        ),
        "first_definitions_identical": bool(flag_keys == appearance_keys),
        "debut_rows": int(len(by_flag)),
    }
    return step34, metrics, by_flag.reset_index(drop=True)


def build_crossfile_audit(
    diagnostics: pd.DataFrame,
    step33_debut: pd.DataFrame,
    step34_debut: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconcile full-history, Step 33, and Step 34 player-match definitions."""

    full = diagnostics.loc[diagnostics["year"].eq(TEST_YEAR)].copy()
    full = full[
        [
            "match_id",
            "player_id",
            "match_sequence",
            "opponent_id",
            "focal_side",
            "p_focal_current_convention",
        ]
    ].rename(
        columns={
            "match_sequence": "full_history_match_sequence",
            "opponent_id": "full_history_opponent_id",
            "focal_side": "full_history_focal_side",
        }
    )
    full["in_full_history"] = True

    s33 = step33_debut[
        [
            "match_id",
            "player_id",
            "match_sequence",
            "opponent_id",
            "focal_side",
        ]
    ].rename(
        columns={
            "match_sequence": "step33_match_sequence",
            "opponent_id": "step33_opponent_id",
            "focal_side": "step33_focal_side",
        }
    )
    s33["in_step33"] = True

    s34 = step34_debut[
        [
            "match_id",
            "player_id",
            "match_sequence",
            "opponent_id",
            "focal_side",
            "p_focal_Glicko_low_fixed",
        ]
    ].rename(
        columns={
            "match_sequence": "step34_match_sequence",
            "opponent_id": "step34_opponent_id",
            "focal_side": "step34_focal_side",
        }
    )
    s34["in_step34"] = True

    all_keys = pd.concat(
        [
            full[["match_id", "player_id"]],
            s33[["match_id", "player_id"]],
            s34[["match_id", "player_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    audit = all_keys.merge(
        full, on=["match_id", "player_id"], how="left", validate="one_to_one"
    )
    audit = audit.merge(
        s33, on=["match_id", "player_id"], how="left", validate="one_to_one"
    )
    audit = audit.merge(
        s34, on=["match_id", "player_id"], how="left", validate="one_to_one"
    )
    for column in ["in_full_history", "in_step33", "in_step34"]:
        audit[column] = audit[column].fillna(False).astype(bool)

    audit["match_sequence_agrees"] = (
        audit["full_history_match_sequence"].eq(audit["step33_match_sequence"])
        & audit["full_history_match_sequence"].eq(audit["step34_match_sequence"])
    )
    audit["opponent_id_agrees"] = (
        audit["full_history_opponent_id"].eq(audit["step33_opponent_id"])
        & audit["full_history_opponent_id"].eq(audit["step34_opponent_id"])
    )
    audit["focal_side_agrees"] = (
        audit["full_history_focal_side"].eq(audit["step33_focal_side"])
        & audit["full_history_focal_side"].eq(audit["step34_focal_side"])
    )
    audit["probability_difference_full_minus_step34"] = (
        audit["p_focal_current_convention"]
        - audit["p_focal_Glicko_low_fixed"]
    )
    audit["all_sources_agree"] = (
        audit["in_full_history"]
        & audit["in_step33"]
        & audit["in_step34"]
        & audit["match_sequence_agrees"]
        & audit["opponent_id_agrees"]
        & audit["focal_side_agrees"]
    )
    audit = audit.sort_values(
        ["full_history_match_sequence", "player_id"], na_position="last"
    ).reset_index(drop=True)

    full_keys = set(zip(full["match_id"], full["player_id"]))
    step33_keys = set(zip(s33["match_id"], s33["player_id"]))
    step34_keys = set(zip(s34["match_id"], s34["player_id"]))
    metrics = {
        "full_step33_keys_identical": bool(full_keys == step33_keys),
        "full_step34_keys_identical": bool(full_keys == step34_keys),
        "step33_step34_keys_identical": bool(step33_keys == step34_keys),
        "player_sets_identical": bool(
            set(full["player_id"])
            == set(s33["player_id"])
            == set(s34["player_id"])
        ),
        "match_sets_identical": bool(
            set(full["match_id"])
            == set(s33["match_id"])
            == set(s34["match_id"])
        ),
        "match_sequence_agrees": bool(audit["match_sequence_agrees"].all()),
        "opponent_id_agrees": bool(audit["opponent_id_agrees"].all()),
        "all_sources_agree": bool(audit["all_sources_agree"].all()),
        "max_abs_probability_difference": float(
            audit["probability_difference_full_minus_step34"].abs().max()
        ),
    }
    return audit, metrics


def build_burn_in_sensitivity(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Audit classification sensitivity without rerunning a rating model."""

    rows = []
    for burn_in_years in [1, 3, 5, 10]:
        post_start = MODEL_START_YEAR + burn_in_years
        retained = diagnostics.loc[
            diagnostics["first_recorded_year"].ge(post_start)
        ]
        rows.append(
            {
                "burn_in_years": burn_in_years,
                "burn_in_start_year": MODEL_START_YEAR,
                "burn_in_end_year": post_start - 1,
                "post_burn_in_start_year": post_start,
                "post_burn_in_first_appearances": int(len(retained)),
                "test_year_first_appearances_retained": int(
                    retained["first_recorded_year"].eq(TEST_YEAR).sum()
                ),
                "audit_role": (
                    "classification sensitivity only; the rating model is unchanged"
                ),
            }
        )
    return pd.DataFrame(rows)


def create_figures(
    year_summary: pd.DataFrame,
    cohort_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create restrained meeting-ready figures and a manifest."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    post_years = year_summary.loc[
        year_summary["year"].ge(POST_BURN_IN_START_YEAR)
    ].copy()

    plt.figure(figsize=(9.5, 5.3))
    plt.plot(
        post_years["year"],
        post_years["median_contemporaneous_established_rating"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Median prematch established-player scale",
    )
    plt.axhline(
        INITIAL_RATING,
        color="#C44E52",
        linestyle="--",
        label="Fixed new-player anchor (1500)",
    )
    plt.xlabel("First recorded year")
    plt.ylabel("Glicko rating")
    plt.title("New-player anchor relative to the prematch established-player scale")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_1_PATH, dpi=200)
    plt.close()

    plt.figure(figsize=(9.5, 5.3))
    plt.plot(
        post_years["year"],
        post_years["mean_opponent_rating"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Mean first-opponent prematch rating",
    )
    plt.plot(
        post_years["year"],
        post_years["median_opponent_rating"],
        linewidth=1.4,
        label="Median first-opponent prematch rating",
    )
    plt.axhline(
        INITIAL_RATING,
        color="#C44E52",
        linestyle="--",
        label="Fixed new-player anchor (1500)",
    )
    plt.xlabel("First recorded year")
    plt.ylabel("Glicko rating")
    plt.title("New-player anchor relative to first recorded opponents")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_2_PATH, dpi=200)
    plt.close()

    cohort_order = [
        "system_start_left_censored",
        "within_5y_burn_in_recorded_entry",
        "post_burn_in_recorded_entry",
        "test_year_recorded_entry",
    ]
    plot = cohort_summary.set_index("group").loc[cohort_order].reset_index()
    labels = ["1985 start", "1986-89 burn-in", "1990-2024 entry", "2025 test entry"]
    x = np.arange(len(plot))
    width = 0.36
    plt.figure(figsize=(9.5, 5.3))
    plt.bar(
        x - width / 2,
        plot["current_prediction_bias"],
        width,
        label="Current-convention bias",
    )
    plt.bar(
        x + width / 2,
        plot["direct_prediction_bias"],
        width,
        label="Direct-focal bias",
    )
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("Mean predicted probability minus win rate")
    plt.title("Historical first-appearance bias under two probability conventions")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_3_PATH, dpi=200)
    plt.close()

    return pd.DataFrame(
        [
            {
                "figure_id": "42_fig01",
                "path": str(FIGURE_1_PATH.relative_to(PROJECT_ROOT)),
                "description": (
                    "The fixed 1500 anchor and the median established-player scale "
                    "calculated immediately before each entrant's first recorded match."
                ),
                "source_table": ENTRY_YEAR_SUMMARY_PATH.name,
                "interpretation_limit": (
                    "Prematch descriptive scale alignment; not a causal estimate."
                ),
            },
            {
                "figure_id": "42_fig02",
                "path": str(FIGURE_2_PATH.relative_to(PROJECT_ROOT)),
                "description": (
                    "The fixed 1500 anchor compared with mean and median actual "
                    "first-opponent ratings."
                ),
                "source_table": ENTRY_YEAR_SUMMARY_PATH.name,
                "interpretation_limit": (
                    "First recorded appearance does not establish true career debut."
                ),
            },
            {
                "figure_id": "42_fig03",
                "path": str(FIGURE_3_PATH.relative_to(PROJECT_ROOT)),
                "description": (
                    "First-appearance prediction bias under the current and "
                    "direct-focal probability conventions."
                ),
                "source_table": ENTRY_COHORT_SUMMARY_PATH.name,
                "interpretation_limit": (
                    "Sensitivity only; it does not replace the formal Meeting 7 convention."
                ),
            },
        ]
    )


def build_validation_checks(
    matches: pd.DataFrame,
    strict: pd.DataFrame,
    diagnostics: pd.DataFrame,
    step41_comparison: dict[str, Any],
    step33: pd.DataFrame,
    step33_metrics: dict[str, Any],
    step34: pd.DataFrame,
    step34_metrics: dict[str, Any],
    crossfile: pd.DataFrame,
    crossfile_metrics: dict[str, Any],
    burn_in: pd.DataFrame,
    cohort_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create the required error-level and warning-level checks."""

    rows: list[dict[str, Any]] = []
    cohort_counts = strict["entry_cohort"].value_counts()
    test = diagnostics.loc[diagnostics["first_recorded_year"].eq(TEST_YEAR)]

    add_check(
        rows,
        "step24_file_exists",
        GLICKO_PIPELINE_PATH.exists(),
        GLICKO_PIPELINE_PATH.exists(),
        True,
        "error",
        "The validated canonical Glicko implementation must be present.",
    )
    add_check(rows, "step33_file_exists", STEP33_PATH.exists(), STEP33_PATH.exists(), True, "error", "Frozen Step 33 output must be present.")
    add_check(rows, "step34_file_exists", STEP34_PATH.exists(), STEP34_PATH.exists(), True, "error", "Frozen Step 34 output must be present.")
    add_check(rows, "full_history_rows", len(matches) == EXPECTED_FULL_HISTORY_ROWS, len(matches), EXPECTED_FULL_HISTORY_ROWS, "error", "Full-history row count.")
    add_check(rows, "full_history_fcode_unique", matches["fcode"].is_unique, matches["fcode"].nunique(), len(matches), "error", "match_id is defined as unique fcode.")
    add_check(rows, "winner_ids_not_missing", not matches["winner"].isna().any(), int(matches["winner"].isna().sum()), 0, "error", "Winner IDs must be complete.")
    add_check(rows, "loser_ids_not_missing", not matches["loser"].isna().any(), int(matches["loser"].isna().sum()), 0, "error", "Loser IDs must be complete.")
    add_check(rows, "winner_never_equals_loser", matches["winner"].ne(matches["loser"]).all(), int(matches["winner"].eq(matches["loser"]).sum()), 0, "error", "Self-matches are invalid.")
    add_check(rows, "history_year_range", int(matches["year"].min()) == MODEL_START_YEAR and int(matches["year"].max()) == TEST_YEAR, f"{int(matches['year'].min())}-{int(matches['year'].max())}", f"{MODEL_START_YEAR}-{TEST_YEAR}", "error", "Frozen history range.")
    add_check(rows, "exact_match_sequence_unique", matches["match_sequence"].is_unique, matches["match_sequence"].nunique(), len(matches), "error", "Frozen processing sequence must be one-to-one.")
    add_check(rows, "one_strict_entry_row_per_player", strict["player_id"].is_unique, strict["player_id"].nunique(), len(strict), "error", "Strict classification is one row per player.")
    add_check(rows, "unique_players", len(strict) == EXPECTED_UNIQUE_PLAYERS, len(strict), EXPECTED_UNIQUE_PLAYERS, "error", "Expected full-history player count.")
    add_check(rows, "system_start_players", int(cohort_counts.get("system_start_left_censored", 0)) == EXPECTED_SYSTEM_START_PLAYERS, int(cohort_counts.get("system_start_left_censored", 0)), EXPECTED_SYSTEM_START_PLAYERS, "error", "1985 model-start cohort.")
    add_check(rows, "within_burn_in_players", int(cohort_counts.get("within_5y_burn_in_recorded_entry", 0)) == EXPECTED_WITHIN_BURN_IN_PLAYERS, int(cohort_counts.get("within_5y_burn_in_recorded_entry", 0)), EXPECTED_WITHIN_BURN_IN_PLAYERS, "error", "1986-1989 recorded-entry cohort.")
    add_check(rows, "post_burn_in_1990_2024_players", int(cohort_counts.get("post_burn_in_recorded_entry", 0)) == EXPECTED_POST_BURN_IN_1990_2024_PLAYERS, int(cohort_counts.get("post_burn_in_recorded_entry", 0)), EXPECTED_POST_BURN_IN_1990_2024_PLAYERS, "error", "1990-2024 recorded-entry cohort.")
    add_check(rows, "test_year_entrants", int(cohort_counts.get("test_year_recorded_entry", 0)) == EXPECTED_TEST_YEAR_ENTRANTS, int(cohort_counts.get("test_year_recorded_entry", 0)), EXPECTED_TEST_YEAR_ENTRANTS, "error", "2025 held-out entry cohort.")
    add_check(rows, "cohort_counts_sum", int(cohort_counts.sum()) == EXPECTED_UNIQUE_PLAYERS, int(cohort_counts.sum()), EXPECTED_UNIQUE_PLAYERS, "error", "Cohorts are mutually exclusive and exhaustive.")
    add_check(rows, "one_first_appearance_diagnostic_per_player", len(diagnostics) == EXPECTED_UNIQUE_PLAYERS and diagnostics["player_id"].is_unique, f"rows={len(diagnostics)}, unique={diagnostics['player_id'].nunique()}", EXPECTED_UNIQUE_PLAYERS, "error", "One strict prematch diagnostic per player.")
    add_check(rows, "first_appearance_focal_ratings_1500", np.isclose(diagnostics["focal_rating_before"], INITIAL_RATING).all(), float(diagnostics["focal_rating_before"].mean()), INITIAL_RATING, "error", "Fixed rating anchor is unchanged.")
    add_check(rows, "first_appearance_focal_rds_350", np.isclose(diagnostics["focal_rd_before"], INITIAL_RD).all(), float(diagnostics["focal_rd_before"].mean()), INITIAL_RD, "error", "Fixed RD anchor is unchanged.")
    add_check(rows, "primary_probabilities_in_unit_interval", diagnostics["p_focal_current_convention"].between(0, 1).all(), f"{diagnostics['p_focal_current_convention'].min():.12f}-{diagnostics['p_focal_current_convention'].max():.12f}", "[0,1]", "error", "Meeting 7 convention probabilities.")
    add_check(rows, "direct_probabilities_in_unit_interval", diagnostics["p_focal_direct"].between(0, 1).all(), f"{diagnostics['p_focal_direct'].min():.12f}-{diagnostics['p_focal_direct'].max():.12f}", "[0,1]", "error", "Direct-focal sensitivity probabilities.")
    add_check(rows, "step33_rows", step33_metrics["rows"] == EXPECTED_STEP33_ROWS, step33_metrics["rows"], EXPECTED_STEP33_ROWS, "error", "Frozen Step 33 row count.")
    add_check(rows, "step33_match_id_unique", step33_metrics["match_id_unique"], step33["match_id"].nunique(), EXPECTED_STEP33_ROWS, "error", "Step 33 is one row per test match.")
    add_check(rows, "step33_all_2025", step33_metrics["all_2025"], sorted(step33["year"].unique().tolist()), [TEST_YEAR], "error", "Step 33 must contain only the held-out year.")
    add_check(rows, "step34_rows", step34_metrics["rows"] == EXPECTED_STEP34_ROWS, step34_metrics["rows"], EXPECTED_STEP34_ROWS, "error", "Frozen Step 34 appearance-row count.")
    add_check(rows, "step34_every_match_two_rows", step34_metrics["every_match_two_rows"], int(step34.groupby("match_id").size().ne(2).sum()), 0, "error", "Each match has one A and one B focal row.")
    add_check(rows, "step34_debut_formula_correct", step34_metrics["debut_formula_correct"], bool(step34_metrics["debut_formula_correct"]), True, "error", "debut_flag equals total_games_before == 0.")
    add_check(rows, "step34_appearance_number_formula_correct", step34_metrics["appearance_number_formula_correct"], bool(step34_metrics["appearance_number_formula_correct"]), True, "error", "appearance_number equals total_games_before + 1.")
    add_check(rows, "step33_debut_appearances", step33_metrics["debut_appearance_rows"] == EXPECTED_DEBUT_APPEARANCES, step33_metrics["debut_appearance_rows"], EXPECTED_DEBUT_APPEARANCES, "error", "Step 33 long debut rows.")
    add_check(rows, "step33_unique_debut_players", step33_metrics["unique_debut_players"] == EXPECTED_DEBUT_PLAYERS, step33_metrics["unique_debut_players"], EXPECTED_DEBUT_PLAYERS, "error", "Step 33 debut players.")
    add_check(rows, "step33_unique_debut_matches", step33_metrics["unique_debut_matches"] == EXPECTED_DEBUT_MATCHES, step33_metrics["unique_debut_matches"], EXPECTED_DEBUT_MATCHES, "error", "Step 33 debut matches.")
    add_check(rows, "exactly_one_debut_matches", step33_metrics["exactly_one_debut_matches"] == EXPECTED_EXACTLY_ONE_DEBUT_MATCHES, step33_metrics["exactly_one_debut_matches"], EXPECTED_EXACTLY_ONE_DEBUT_MATCHES, "error", "Matches with exactly one debuting player.")
    add_check(rows, "both_debut_matches", step33_metrics["both_debut_matches"] == EXPECTED_BOTH_DEBUT_MATCHES, step33_metrics["both_debut_matches"], EXPECTED_BOTH_DEBUT_MATCHES, "error", "Matches with two debuting players.")
    add_check(rows, "full_history_step33_keys_identical", crossfile_metrics["full_step33_keys_identical"], crossfile_metrics["full_step33_keys_identical"], True, "error", "2025 player-match debut keys.")
    add_check(rows, "full_history_step34_keys_identical", crossfile_metrics["full_step34_keys_identical"], crossfile_metrics["full_step34_keys_identical"], True, "error", "2025 player-match first-appearance keys.")
    add_check(rows, "step33_step34_keys_identical", crossfile_metrics["step33_step34_keys_identical"], crossfile_metrics["step33_step34_keys_identical"], True, "error", "Frozen downstream key agreement.")
    add_check(rows, "match_sequence_agrees_across_sources", crossfile_metrics["match_sequence_agrees"], int(crossfile["match_sequence_agrees"].sum()), len(crossfile), "error", "Full-history, Step 33, and Step 34 processing positions.")
    add_check(rows, "opponent_id_agrees_across_sources", crossfile_metrics["opponent_id_agrees"], int(crossfile["opponent_id_agrees"].sum()), len(crossfile), "error", "Opponent identity agreement.")
    add_check(rows, "all_2025_first_appearances_post_burn_in", test["post_primary_burn_in"].all(), int(test["post_primary_burn_in"].sum()), EXPECTED_DEBUT_APPEARANCES, "error", "Held-out first appearances are not model-start rows.")
    add_check(rows, "all_burn_in_definitions_retain_76_test_rows", burn_in["test_year_first_appearances_retained"].eq(EXPECTED_DEBUT_APPEARANCES).all(), burn_in["test_year_first_appearances_retained"].tolist(), [EXPECTED_DEBUT_APPEARANCES] * len(burn_in), "error", "Classification sensitivity only.")
    add_check(rows, "current_2025_probabilities_reproduce_step34", crossfile_metrics["max_abs_probability_difference"] < 1e-10, crossfile_metrics["max_abs_probability_difference"], "<1e-10", "error", "Frozen Meeting 7 probability convention.")
    add_check(rows, "current_2025_mean_probability", abs(test["p_focal_current_convention"].mean() - 0.743448) < 1e-6, float(test["p_focal_current_convention"].mean()), 0.743448, "error", "Rounded Meeting 7 headline.")
    add_check(rows, "current_2025_empirical_win_rate", abs(test["outcome_focal"].mean() - 0.407895) < 1e-6, float(test["outcome_focal"].mean()), 0.407895, "error", "Rounded Meeting 7 headline.")
    add_check(rows, "current_2025_brier", abs(test["brier_current"].mean() - 0.322316) < 1e-6, float(test["brier_current"].mean()), 0.322316, "error", "Rounded Meeting 7 headline.")
    add_check(rows, "current_2025_mean_opponent_rating", abs(test["opponent_rating_before"].mean() - 1180.755) < 1e-3, float(test["opponent_rating_before"].mean()), 1180.755, "error", "Rounded Meeting 7 mechanism result.")

    missing_dates = int(matches["event_order_date"].isna().sum())
    add_check(rows, "event_dates_complete", missing_dates == 0, missing_dates, 0, "warning", "Missing dates are retained; date-based active scales are unavailable for affected rows.")
    unavailable_primary = int((~diagnostics["contemporaneous_scale_available"]).sum())
    add_check(rows, "contemporaneous_scale_available_for_all_entries", unavailable_primary == 0, unavailable_primary, 0, "warning", "Unavailable values are expected at the beginning of 1985 before any player is established.")
    unavailable_active = int((~diagnostics["active_scale_available"]).sum())
    add_check(rows, "active_365d_scale_available_for_all_entries", unavailable_active == 0, unavailable_active, 0, "warning", "The secondary date-based scale can be unavailable for missing dates or an empty active pool.")
    annual_counts = diagnostics.groupby("first_recorded_year").size()
    small_years = int(annual_counts.lt(20).sum())
    add_check(rows, "annual_entry_cohorts_at_least_20", small_years == 0, small_years, 0, "warning", "Small annual cohorts should not be overinterpreted.")
    primary_groups = cohort_summary.loc[
        cohort_summary["group"].isin(
            [
                "system_start_left_censored",
                "within_5y_burn_in_recorded_entry",
                "post_burn_in_recorded_entry",
                "test_year_recorded_entry",
            ]
        )
    ]
    max_bias_difference = float(
        (
            primary_groups["direct_prediction_bias"]
            - primary_groups["current_prediction_bias"]
        )
        .abs()
        .max()
    )
    add_check(rows, "probability_convention_bias_difference_below_0_01", max_bias_difference <= 0.01, max_bias_difference, "<=0.01", "warning", "Material convention differences are reported as sensitivity evidence and do not replace the current convention.")
    add_check(rows, "strict_classification_agrees_with_step41", bool(step41_comparison["all_agree"]), step41_comparison["mismatch_rows"], 0, "warning", "Any mismatch is saved separately and does not silently replace Step 41.")
    return rows


def write_summary(
    dataset_path: Path,
    inactivity_unit: str,
    low_variant: dict[str, Any],
    strict: pd.DataFrame,
    step41_comparison: dict[str, Any],
    diagnostics: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    crossfile_metrics: dict[str, Any],
    step33_metrics: dict[str, Any],
    checks: pd.DataFrame,
) -> None:
    """Write the requested methodological and result summary."""

    test = diagnostics.loc[diagnostics["first_recorded_year"].eq(TEST_YEAR)]
    cohort = cohort_summary.set_index("group")
    post = cohort.loc["post_burn_in_recorded_entry"]
    test_cohort = cohort.loc["test_year_recorded_entry"]
    qualitative_persists = bool(
        post["current_prediction_bias"] > 0
        and post["direct_prediction_bias"] > 0
        and test_cohort["current_prediction_bias"] > 0
        and test_cohort["direct_prediction_bias"] > 0
    )
    error_checks = checks.loc[checks["severity"].eq("error")]
    failed_errors = error_checks.loc[~error_checks["passed"]]
    warning_rows = checks.loc[
        checks["severity"].eq("warning") & ~checks["passed"]
    ]

    if STEP41_YEARLY_SCALE_PATH.exists():
        step41_yearly = pd.read_csv(STEP41_YEARLY_SCALE_PATH, low_memory=False)
        step41_2025 = step41_yearly.loc[step41_yearly["year"].eq(TEST_YEAR)]
        step41_end_year_median = (
            float(step41_2025["median_rating_established_active"].iloc[0])
            if len(step41_2025)
            else float("nan")
        )
    else:
        step41_end_year_median = float("nan")

    lines = [
        "# Meeting 8 Step 42: Prematch Entry-Scale and Cross-File Audit",
        "",
        "## Purpose",
        "",
        "Step 42 supplements Step 41 by using the exact model-processing order returned by Step 24 and by measuring rating-scale alignment immediately before each player's first recorded match. It reuses the frozen low-inflation Glicko configuration and does not tune or create a rating model.",
        "",
        "## Definitions",
        "",
        "- **First recorded appearance:** the row with the minimum frozen `match_sequence` for a player in the 1985-2025 processed history.",
        "- **Model-start left censoring:** players first observed in 1985 may already have prior unrecorded experience.",
        "- **Post-burn-in recorded entry:** first observed from 1990 onward after the primary five-calendar-year burn-in.",
        "- **True career debut:** not observed and not claimed by this analysis.",
        "",
        "## Exact processing-order audit",
        "",
        f"- Strict unique players: {len(strict):,}",
        f"- Step 41 classification available: {step41_comparison['step41_available']}",
        f"- Step 41 mismatch rows: {step41_comparison['mismatch_rows']}",
        f"- Strict classification agrees with Step 41: {step41_comparison['all_agree']}",
        "",
        "The strict definition uses the sequence already returned by `step24.load_matches()` and does not perform a second chronological sort.",
        "",
        "## Cross-file reconciliation",
        "",
        f"- First-appearance rows: {step33_metrics['debut_appearance_rows']}",
        f"- Unique players: {step33_metrics['unique_debut_players']}",
        f"- Unique matches: {step33_metrics['unique_debut_matches']}",
        f"- Exactly-one-debut matches: {step33_metrics['exactly_one_debut_matches']}",
        f"- Both-debut matches: {step33_metrics['both_debut_matches']}",
        f"- Full history, Step 33, and Step 34 agree exactly: {crossfile_metrics['all_sources_agree']}",
        f"- Maximum absolute current-probability difference from Step 34: {crossfile_metrics['max_abs_probability_difference']:.3e}",
        "",
        "## Prematch contemporaneous scale",
        "",
        "For each entrant, the primary contemporaneous scale includes all players with at least one prior processed match immediately before the entrant's match. The current focal player is excluded; when both players are new, both are excluded.",
        "",
        f"- 2025 mean contemporaneous established-player rating: {test['contemporaneous_established_mean_rating'].mean():.3f}",
        f"- 2025 median contemporaneous established-player rating: {test['contemporaneous_established_median_rating'].median():.3f}",
        f"- 2025 mean initial-minus-contemporaneous-median gap: {test['initial_minus_contemporaneous_median'].mean():.3f}",
        f"- 2025 median initial-minus-contemporaneous-median gap: {test['initial_minus_contemporaneous_median'].median():.3f}",
        f"- 2025 mean active-established-365-day median: {test['active_established_365d_median_rating'].mean():.3f}",
        f"- 2025 mean initial-minus-active-365-day-median gap: {test['initial_minus_active_established_365d_median'].mean():.3f}",
        f"- 2025 mean actual opponent rating: {test['opponent_rating_before'].mean():.3f}",
        f"- 2025 mean initial-minus-opponent gap: {test['initial_minus_opponent_rating'].mean():.3f}",
        "",
        f"These are prematch entry-time quantities. They are distinct from the Step 41 end-of-year 2025 established-active median of {step41_end_year_median:.3f}; the end-of-year value must not be described as the contemporaneous entry-time scale.",
        "",
        "## Probability-orientation sensitivity",
        "",
        f"- 1990-2024 current-convention bias: {post['current_prediction_bias']:.6f}",
        f"- 1990-2024 direct-focal bias: {post['direct_prediction_bias']:.6f}",
        f"- 2025 current-convention mean probability: {test_cohort['mean_current_probability']:.6f}",
        f"- 2025 direct-focal mean probability: {test_cohort['mean_direct_probability']:.6f}",
        f"- 2025 empirical win rate: {test_cohort['empirical_win_rate']:.6f}",
        f"- Qualitative post-burn-in over-prediction direction persists: {qualitative_persists}",
        "",
        "The primary formal 2025 result remains the Meeting 7 current convention. Direct-focal probabilities are a historical orientation sensitivity only. Results from 1990-2024 are in-sample descriptive mechanism evidence, not an independent held-out test.",
        "",
        "## Interpretation",
        "",
        "The held-out 2025 first-appearance sample consists entirely of players who enter the recorded system after a long burn-in. The weakness is therefore not an artefact of the 1985 model start. Immediately before entry, the fixed 1500 anchor is high relative to both the contemporaneous established-player scale and the actual first opponent ratings. This supports a relative initialisation mismatch interpretation. However, first recorded appearance does not establish true career debut, and the 1990-2024 cohort results are descriptive historical mechanism evidence rather than an independent held-out test.",
        "",
        "Rating-scale alignment is one component of the mechanism. This diagnostic does not make a causal claim that rating-scale drift alone creates the prediction error.",
        "",
        "## Relationship to Step 41",
        "",
        "- Step 41 provides broad burn-in classification and end-of-year scale trends.",
        "- Step 42 provides strict processing-order classification, prematch scale alignment, and direct Step 33/34 reconciliation.",
        "- The two steps are complementary.",
        f"- Step 42 invalidates no Step 41 result because explicit classification mismatches found: {step41_comparison['mismatch_rows']}.",
        "",
        "## Dissertation use",
        "",
        "- Put operational entry definitions in Methodology.",
        "- Put the held-out 2025 first-appearance result in the Early-game Results section.",
        "- Put prematch scale alignment in the mechanism subsection.",
        "- Label historical cohort and direct-probability results as sensitivity or supporting evidence.",
        "- Treat the Step 41 end-of-year scale figure as descriptive rather than the primary initialisation diagnostic.",
        "",
        "## Validation",
        "",
        f"- Total checks: {len(checks)}",
        f"- Passed error checks: {int(error_checks['passed'].sum())}/{len(error_checks)}",
        f"- Failed error checks: {len(failed_errors)}",
        f"- Active warnings: {len(warning_rows)}",
        f"- Dataset: `{dataset_path.relative_to(PROJECT_ROOT)}`",
        f"- Inactivity unit: {inactivity_unit}",
        f"- Reused variant: `{low_variant['variant']}` with C={float(low_variant['c_value']):.12f}",
    ]
    if len(warning_rows):
        lines.extend(
            [
                "",
                "### Active warnings",
                "",
                *[
                    f"- {row.check_name}: {row.detail} Observed={row.observed}."
                    for row in warning_rows.itertuples(index=False)
                ],
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_checks(checks: pd.DataFrame) -> None:
    """Print every validation check with PASS or FAIL."""

    print("\nValidation checks")
    for row in checks.itertuples(index=False):
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{status} [{row.severity}] {row.check_name}: "
            f"observed={row.observed}; expected={row.expected}"
        )


def print_console_report(
    strict: pd.DataFrame,
    diagnostics: pd.DataFrame,
    step33_metrics: dict[str, Any],
    crossfile_metrics: dict[str, Any],
    checks: pd.DataFrame,
) -> None:
    """Print the requested final console report."""

    cohort_counts = strict["entry_cohort"].value_counts()
    test = diagnostics.loc[diagnostics["first_recorded_year"].eq(TEST_YEAR)]
    error_checks = checks.loc[checks["severity"].eq("error")]
    active_warnings = checks.loc[
        checks["severity"].eq("warning") & ~checks["passed"]
    ]

    print("\nMeeting 8 Step 42 complete")
    print(f"Strict unique players: {len(strict)}")
    print(
        "System-start players: "
        f"{int(cohort_counts.get('system_start_left_censored', 0))}"
    )
    print(
        "Within-burn-in players: "
        f"{int(cohort_counts.get('within_5y_burn_in_recorded_entry', 0))}"
    )
    print(
        "Post-burn-in 1990-2024 players: "
        f"{int(cohort_counts.get('post_burn_in_recorded_entry', 0))}"
    )
    print(
        "2025 entrants: "
        f"{int(cohort_counts.get('test_year_recorded_entry', 0))}"
    )
    print()
    print(f"2025 first appearances: {len(test)}")
    print(f"2025 unique matches: {test['match_id'].nunique()}")
    print(f"Exactly one debut: {step33_metrics['exactly_one_debut_matches']}")
    print(f"Both debut: {step33_metrics['both_debut_matches']}")
    print()
    print(
        "Step 33 / Step 34 / full-history agreement: "
        f"{crossfile_metrics['all_sources_agree']}"
    )
    print()
    print(
        "2025 mean contemporaneous established rating: "
        f"{test['contemporaneous_established_mean_rating'].mean():.6f}"
    )
    print(
        "2025 median contemporaneous established rating: "
        f"{test['contemporaneous_established_median_rating'].median():.6f}"
    )
    print(
        "2025 mean initial-minus-contemporaneous-median gap: "
        f"{test['initial_minus_contemporaneous_median'].mean():.6f}"
    )
    print(
        "2025 median initial-minus-contemporaneous-median gap: "
        f"{test['initial_minus_contemporaneous_median'].median():.6f}"
    )
    print()
    print(
        "2025 mean actual opponent rating: "
        f"{test['opponent_rating_before'].mean():.6f}"
    )
    print(
        "2025 mean initial-minus-opponent gap: "
        f"{test['initial_minus_opponent_rating'].mean():.6f}"
    )
    print()
    print(
        "Current-convention 2025 mean probability: "
        f"{test['p_focal_current_convention'].mean():.12f}"
    )
    print(
        "Direct-focal 2025 mean probability: "
        f"{test['p_focal_direct'].mean():.12f}"
    )
    print()
    print(
        "Validation error checks passed: "
        f"{int(error_checks['passed'].sum())} / {len(error_checks)}"
    )
    print(f"Warnings: {len(active_warnings)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


def main() -> None:
    """Run the complete Step 42 supplement."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    step24 = load_step24_module()
    matches, inactivity_unit, dataset_path = step24.load_matches()
    matches = matches.reset_index(drop=True)
    matches["match_sequence"] = np.arange(1, len(matches) + 1)
    variants = step24.build_variants(inactivity_unit)
    low_variant = next(
        variant for variant in variants if variant["variant"] == "low_inflation"
    )

    appearance_long = build_strict_appearance_long(matches)
    strict = build_strict_player_classification(appearance_long)
    step41_mismatches, step41_comparison = compare_with_step41(strict)
    diagnostics = run_frozen_low_inflation_state(
        matches, strict, step24, low_variant
    )
    year_summary = build_entry_year_summary(diagnostics)
    cohort_summary = build_entry_cohort_summary(diagnostics)
    orientation = build_orientation_sensitivity(cohort_summary)
    step33, step33_metrics, step33_debut = load_and_audit_step33()
    step34, step34_metrics, step34_debut = load_and_audit_step34()
    crossfile, crossfile_metrics = build_crossfile_audit(
        diagnostics, step33_debut, step34_debut
    )
    burn_in = build_burn_in_sensitivity(diagnostics)
    figure_manifest = create_figures(year_summary, cohort_summary)

    strict.to_csv(STRICT_ENTRY_PATH, index=False, encoding="utf-8")
    step41_mismatches.to_csv(STEP41_MISMATCH_PATH, index=False, encoding="utf-8")
    diagnostics.to_csv(PREMATCH_DIAGNOSTIC_PATH, index=False, encoding="utf-8")
    year_summary.to_csv(ENTRY_YEAR_SUMMARY_PATH, index=False, encoding="utf-8")
    cohort_summary.to_csv(
        ENTRY_COHORT_SUMMARY_PATH, index=False, encoding="utf-8"
    )
    crossfile.to_csv(CROSSFILE_AUDIT_PATH, index=False, encoding="utf-8")
    orientation.to_csv(
        ORIENTATION_SENSITIVITY_PATH, index=False, encoding="utf-8"
    )
    burn_in.to_csv(BURN_IN_SENSITIVITY_PATH, index=False, encoding="utf-8")
    figure_manifest.to_csv(
        FIGURE_MANIFEST_PATH, index=False, encoding="utf-8"
    )

    check_rows = build_validation_checks(
        matches,
        strict,
        diagnostics,
        step41_comparison,
        step33,
        step33_metrics,
        step34,
        step34_metrics,
        crossfile,
        crossfile_metrics,
        burn_in,
        cohort_summary,
    )
    checks = pd.DataFrame(check_rows)
    checks.to_csv(VALIDATION_PATH, index=False, encoding="utf-8")

    output_tables = [
        STRICT_ENTRY_PATH,
        STEP41_MISMATCH_PATH,
        PREMATCH_DIAGNOSTIC_PATH,
        ENTRY_YEAR_SUMMARY_PATH,
        ENTRY_COHORT_SUMMARY_PATH,
        CROSSFILE_AUDIT_PATH,
        ORIENTATION_SENSITIVITY_PATH,
        BURN_IN_SENSITIVITY_PATH,
        VALIDATION_PATH,
        FIGURE_MANIFEST_PATH,
    ]
    add_check(
        check_rows,
        "all_output_tables_created",
        all(path.exists() for path in output_tables),
        [path.name for path in output_tables if path.exists()],
        [path.name for path in output_tables],
        "error",
        "Every requested Step 42 CSV output exists.",
    )
    figure_paths = [FIGURE_1_PATH, FIGURE_2_PATH, FIGURE_3_PATH]
    figures_ok = all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
    add_check(
        check_rows,
        "all_figures_created_and_nonempty",
        figures_ok,
        {
            path.name: path.stat().st_size if path.exists() else 0
            for path in figure_paths
        },
        "all three figures exist and are non-empty",
        "error",
        "Meeting-ready figure files.",
    )
    checks = pd.DataFrame(check_rows)
    checks.to_csv(VALIDATION_PATH, index=False, encoding="utf-8")

    write_summary(
        dataset_path,
        inactivity_unit,
        low_variant,
        strict,
        step41_comparison,
        diagnostics,
        cohort_summary,
        crossfile_metrics,
        step33_metrics,
        checks,
    )

    print_checks(checks)
    print_console_report(
        strict, diagnostics, step33_metrics, crossfile_metrics, checks
    )

    failed_errors = checks.loc[
        checks["severity"].eq("error") & ~checks["passed"]
    ]
    if len(failed_errors):
        raise RuntimeError(
            "Step 42 error-level validation failures: "
            + ", ".join(failed_errors["check_name"].tolist())
        )


def run_prematch_entry_audit(output_root: str | Path | None = None) -> None:
    """Run the Step 42 prematch entry-scale audit."""

    if output_root is not None:
        configure_output_root(output_root)
    main()


def run_burnin_and_drift(output_root: str | Path | None = None) -> None:
    """Run the Step 41 burn-in and rating-scale diagnostic."""

    from code.analysis import rating_drift

    if output_root is not None:
        rating_drift.configure_output_root(output_root)
    rating_drift.main()


def run_all_entry_diagnostics(output_root: str | Path | None = None) -> None:
    """Run Steps 41 and 42 in their required dependency order."""

    run_burnin_and_drift(output_root)
    run_prematch_entry_audit(output_root)


if __name__ == "__main__":
    main()
