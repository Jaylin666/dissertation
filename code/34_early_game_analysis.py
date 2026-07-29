"""Meeting 7 step 34: early-game player-appearance analysis.

Part 1 creates the standard player-appearance dataset used by later Meeting 7
scripts. Part 2 reads that appearance dataset as its only input and calculates
early-game model performance summary tables.

This script deliberately does not calculate bootstrap intervals, calibration
tables, figures, or a markdown summary. Those are reserved for later Meeting 7
steps. Glicko probabilities are taken from the Step 33 fixed player-A
probability columns during Part 1, not from the old pre-correction Glicko
columns retained for audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MEETING6_DIR = PROJECT_ROOT / "outputs" / "meeting6"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP33_SCORES_PATH = MEETING6_DIR / "33_orientation_corrected_per_match_scores_2025.csv"

VALIDATION_CHECKS_PATH = OUTPUT_DIR / "34_input_validation_checks.csv"
APPEARANCE_DATASET_PATH = OUTPUT_DIR / "34_early_game_appearance_dataset.csv"
EXACT_APPEARANCE_COUNTS_PATH = OUTPUT_DIR / "34_exact_appearance_counts.csv"
CUMULATIVE_PERFORMANCE_PATH = OUTPUT_DIR / "34_cumulative_threshold_model_performance.csv"
STAGE_PERFORMANCE_PATH = OUTPUT_DIR / "34_stage_bin_model_performance.csv"
EXACT_PERFORMANCE_PATH = OUTPUT_DIR / "34_exact_appearance_model_performance.csv"
PAIRWISE_DIFFERENCES_PATH = OUTPUT_DIR / "34_pairwise_model_differences.csv"
METRIC_VALIDATION_CHECKS_PATH = OUTPUT_DIR / "34_metric_validation_checks.csv"
BOOTSTRAP_CI_PATH = OUTPUT_DIR / "34_bootstrap_confidence_intervals.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "34_figure_manifest.csv"
BOOTSTRAP_FIGURE_VALIDATION_PATH = OUTPUT_DIR / "34_bootstrap_figure_validation_checks.csv"
BOOTSTRAP_METHOD_AUDIT_PATH = OUTPUT_DIR / "34_bootstrap_method_audit_checks.csv"
BOOTSTRAP_METADATA_PATH = OUTPUT_DIR / "34_bootstrap_metadata.csv"
MATCH_LEVEL_ROBUSTNESS_PATH = OUTPUT_DIR / "34_match_level_early_game_robustness.csv"
MATCH_LEVEL_BOOTSTRAP_CI_PATH = OUTPUT_DIR / "34_match_level_bootstrap_confidence_intervals.csv"
BOOTSTRAP_ROBUSTNESS_COMPARISON_PATH = OUTPUT_DIR / "34_bootstrap_robustness_comparison.csv"
BOOTSTRAP_ROBUSTNESS_VALIDATION_PATH = OUTPUT_DIR / "34_bootstrap_robustness_validation_checks.csv"

EXPECTED_MATCHES = 11_379
EXPECTED_APPEARANCE_ROWS = EXPECTED_MATCHES * 2
EPS = 1e-15
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260715

MODEL_PROBABILITY_COLUMNS = {
    "Glicko_low_fixed": "p_a_Glicko_low_fixed",
    "Validation_best_Elo": "p_a_Validation_best_Elo",
    "Glicko_C0_fixed": "p_a_Glicko_C0_fixed",
    "best_AdaptiveK": "p_a_best_AdaptiveK",
    "Default_Elo": "p_a_Default_Elo",
    "Conservative_Elo": "p_a_Conservative_Elo",
}

PERFORMANCE_MODEL_ORDER = [
    "Validation_best_Elo",
    "Glicko_low_fixed",
    "Glicko_C0_fixed",
    "best_AdaptiveK",
]

MODEL_LABELS = {
    "Validation_best_Elo": "Validation-best Elo",
    "Glicko_low_fixed": "Glicko low inflation",
    "Glicko_C0_fixed": "Glicko C0",
    "best_AdaptiveK": "Adaptive-K Elo",
    "Default_Elo": "Default Elo",
    "Conservative_Elo": "Conservative Elo",
}

MODEL_FAMILIES = {
    "Validation_best_Elo": "Elo",
    "Glicko_low_fixed": "Glicko",
    "Glicko_C0_fixed": "Glicko",
    "best_AdaptiveK": "Elo",
    "Default_Elo": "Elo",
    "Conservative_Elo": "Elo",
}

PAIRWISE_COMPARISONS = [
    {
        "comparison": "Validation_best_Elo_minus_Glicko_low_fixed",
        "comparison_label": "Validation-best Elo - Glicko low inflation",
        "left_model": "Validation_best_Elo",
        "right_model": "Glicko_low_fixed",
        "positive_delta_means": "Glicko low inflation better",
        "step33_equivalent": "delta_brier_glicko_vs_elo / delta_logloss_glicko_vs_elo",
    },
    {
        "comparison": "Validation_best_Elo_minus_best_AdaptiveK",
        "comparison_label": "Validation-best Elo - Adaptive-K Elo",
        "left_model": "Validation_best_Elo",
        "right_model": "best_AdaptiveK",
        "positive_delta_means": "Adaptive-K Elo better",
        "step33_equivalent": "Elo baseline minus adaptive-K Elo",
    },
    {
        "comparison": "Glicko_C0_fixed_minus_Glicko_low_fixed",
        "comparison_label": "Glicko C0 - Glicko low inflation",
        "left_model": "Glicko_C0_fixed",
        "right_model": "Glicko_low_fixed",
        "positive_delta_means": "Glicko low inflation better",
        "step33_equivalent": "delta_brier_inflation / delta_logloss_inflation",
    },
]

LEGACY_GLICKO_PROBABILITY_COLUMNS = ["p_a_Glicko_low", "p_a_Glicko_C0"]
FIXED_GLICKO_PROBABILITY_COLUMNS = ["p_a_Glicko_low_fixed", "p_a_Glicko_C0_fixed"]
CUMULATIVE_THRESHOLDS = [1, 5, 10, 20, 30, 50]

FIGURE_MODEL_ORDER = [
    "Validation_best_Elo",
    "Glicko_low_fixed",
    "Glicko_C0_fixed",
    "best_AdaptiveK",
]

MODEL_COLORS = {
    "Validation_best_Elo": "#1B4D89",
    "Glicko_low_fixed": "#C75000",
    "Glicko_C0_fixed": "#7A3E9D",
    "best_AdaptiveK": "#178A5A",
}

BASE_REQUIRED_COLUMNS = [
    "match_id",
    "match_sequence",
    "year",
    "event_id",
    "match_date",
    "player_a_id",
    "player_b_id",
    "winner_id",
    "loser_id",
    "outcome_a",
    "a_total_games_before",
    "b_total_games_before",
    "a_is_debut",
    "b_is_debut",
]

SIDE_FEATURE_SUFFIXES = [
    "games_last_90_days",
    "games_last_365_days",
    "games_previous_calendar_year",
    "days_since_last_game",
    "career_days_before",
    "has_previous_history",
    "date_features_available",
    "date_quality",
]

SYMMETRIC_FEATURE_COLUMNS = [
    "min_total_games_before",
    "max_total_games_before",
    "abs_diff_total_games_before",
    "min_games_last_90_days",
    "max_games_last_90_days",
    "abs_diff_games_last_90_days",
    "min_games_last_365_days",
    "max_games_last_365_days",
    "abs_diff_games_last_365_days",
    "min_previous_year_games",
    "max_previous_year_games",
    "abs_diff_previous_year_games",
    "min_days_since_last_game",
    "max_days_since_last_game",
    "either_player_debut",
    "both_players_have_history",
    "either_player_inactive_365d",
    "either_player_inactive_730d",
    "both_players_active_last_365d",
    "either_player_low_recent_activity",
    "max_prematch_rd",
    "min_prematch_rd",
    "mean_prematch_rd",
    "no_debut_rd_quartile_33",
]

OPTIONAL_MATCH_COLUMNS = ["fcode", "event_key"]

REQUIRED_INPUT_COLUMNS = (
    BASE_REQUIRED_COLUMNS
    + [f"{side}_{suffix}" for side in ["a", "b"] for suffix in SIDE_FEATURE_SUFFIXES]
    + list(MODEL_PROBABILITY_COLUMNS.values())
)

STAGE_LABELS = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any = "",
    severity: str = "error",
    detail: str = "",
) -> None:
    """Append one validation check row."""

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


def bool_series(series: pd.Series) -> pd.Series:
    """Convert a pandas object/bool series to plain boolean values."""

    return series.astype(bool)


def load_step33_scores() -> pd.DataFrame:
    """Load the final Meeting 6 Step 33 per-match score table."""

    if not STEP33_SCORES_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {STEP33_SCORES_PATH}")
    return pd.read_csv(STEP33_SCORES_PATH, low_memory=False)


def validate_input_scores(scores: pd.DataFrame, rows: list[dict[str, Any]]) -> bool:
    """Run Step 34 input-level checks before building appearances."""

    add_check(rows, "input_rows_11379", len(scores) == EXPECTED_MATCHES, len(scores), EXPECTED_MATCHES)

    missing_required = [col for col in REQUIRED_INPUT_COLUMNS if col not in scores.columns]
    add_check(
        rows,
        "required_input_columns_present",
        not missing_required,
        missing_required if missing_required else "none",
        "all required Step 33 columns",
    )

    if "match_id" in scores.columns:
        duplicate_match_ids = int(scores["match_id"].duplicated().sum())
        add_check(rows, "match_id_unique", duplicate_match_ids == 0, duplicate_match_ids, 0)
    else:
        add_check(rows, "match_id_unique", False, "match_id missing", 0)

    if "outcome_a" in scores.columns:
        observed_outcomes = sorted(scores["outcome_a"].dropna().unique().tolist())
        add_check(rows, "outcome_a_binary", set(observed_outcomes).issubset({0, 1}), observed_outcomes, "{0,1}")
    else:
        add_check(rows, "outcome_a_binary", False, "outcome_a missing", "{0,1}")

    fixed_present = all(col in scores.columns for col in FIXED_GLICKO_PROBABILITY_COLUMNS)
    used_probability_cols = set(MODEL_PROBABILITY_COLUMNS.values())
    legacy_used = sorted(used_probability_cols.intersection(LEGACY_GLICKO_PROBABILITY_COLUMNS))
    add_check(
        rows,
        "step33_fixed_glicko_probability_columns_present",
        fixed_present,
        [col for col in FIXED_GLICKO_PROBABILITY_COLUMNS if col in scores.columns],
        FIXED_GLICKO_PROBABILITY_COLUMNS,
    )
    add_check(
        rows,
        "legacy_glicko_probability_columns_not_used",
        not legacy_used,
        legacy_used if legacy_used else "none",
        "do not use p_a_Glicko_low or p_a_Glicko_C0",
        detail="Legacy Glicko columns may exist in Step 33 for audit; Step 34 maps only fixed columns.",
    )

    if missing_required:
        return False

    core_probability_columns = list(MODEL_PROBABILITY_COLUMNS.values())
    missing_probability_values = int(scores[core_probability_columns].isna().sum().sum())
    in_range = bool(scores[core_probability_columns].apply(lambda col: col.between(0, 1).all()).all())
    add_check(rows, "core_model_probabilities_not_missing", missing_probability_values == 0, missing_probability_values, 0)
    add_check(rows, "core_model_probabilities_in_range", in_range, "checked", "[0,1]")

    side_history_cols = [
        "a_total_games_before",
        "b_total_games_before",
        "a_is_debut",
        "b_is_debut",
    ]
    missing_side_history = int(scores[side_history_cols].isna().sum().sum())
    add_check(rows, "side_history_columns_not_missing", missing_side_history == 0, missing_side_history, 0)

    if {"player_a_id", "player_b_id"}.issubset(scores.columns):
        distinct_players = bool((scores["player_a_id"] != scores["player_b_id"]).all())
        add_check(rows, "two_distinct_players_per_match", distinct_players, "checked", True)
    else:
        add_check(rows, "two_distinct_players_per_match", False, "player id columns missing", True)

    return True


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the requested columns that exist in a data frame."""

    return [col for col in columns if col in df.columns]


def build_side_appearances(scores: pd.DataFrame, side: str) -> pd.DataFrame:
    """Build focal-player appearance rows for one canonical side."""

    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    opponent_side = "b" if side == "a" else "a"
    focal_label = side.upper()

    out = pd.DataFrame(
        {
            "match_id": scores["match_id"].astype(int),
            "match_sequence": scores["match_sequence"].astype(int),
            "year": scores["year"].astype(int),
            "date": scores["match_date"],
            "match_date": scores["match_date"],
            "event": scores["event_id"].astype(int),
            "event_id": scores["event_id"].astype(int),
            "focal_side": focal_label,
            "player_id": scores[f"player_{side}_id"].astype(int),
            "opponent_id": scores[f"player_{opponent_side}_id"].astype(int),
            "winner_id": scores["winner_id"].astype(int),
            "loser_id": scores["loser_id"].astype(int),
            "total_games_before": scores[f"{side}_total_games_before"].astype(int),
            "opponent_total_games_before": scores[f"{opponent_side}_total_games_before"].astype(int),
            "debut_flag": bool_series(scores[f"{side}_is_debut"]),
            "opponent_debut_flag": bool_series(scores[f"{opponent_side}_is_debut"]),
        }
    )

    for optional_col in OPTIONAL_MATCH_COLUMNS:
        if optional_col in scores.columns:
            out[optional_col] = scores[optional_col]

    out["appearance_number"] = out["total_games_before"] + 1
    out["opponent_appearance_number"] = out["opponent_total_games_before"] + 1
    out["exact_appearance_number"] = out["appearance_number"]

    if side == "a":
        out["outcome_focal"] = scores["outcome_a"].astype(int)
    else:
        out["outcome_focal"] = (1 - scores["outcome_a"].astype(int)).astype(int)

    # Carry the existing pre-match player-history features into focal orientation.
    for suffix in SIDE_FEATURE_SUFFIXES:
        source_col = f"{side}_{suffix}"
        opponent_col = f"{opponent_side}_{suffix}"
        output_name = "previous_year_games" if suffix == "games_previous_calendar_year" else suffix
        opponent_output_name = f"opponent_{output_name}"
        out[output_name] = scores[source_col]
        out[opponent_output_name] = scores[opponent_col]

    # Convert saved player-A probabilities into focal-player probabilities.
    # This is an orientation transform only; no model probability is recalculated.
    for model_alias, p_a_col in MODEL_PROBABILITY_COLUMNS.items():
        p_a = scores[p_a_col].astype(float)
        out[f"p_focal_{model_alias}"] = p_a if side == "a" else 1.0 - p_a

    # Keep symmetric match-level context features already available in Step 33.
    for col in select_existing_columns(scores, SYMMETRIC_FEATURE_COLUMNS):
        out[col] = scores[col]

    # Keep selected pre-match Glicko rating/RD states for later diagnostics.
    for base_name in ["rating", "rd"]:
        for model in ["Glicko_low", "Glicko_C0"]:
            focal_col = f"{base_name}_{side}_{model}"
            opponent_col = f"{base_name}_{opponent_side}_{model}"
            if focal_col in scores.columns and opponent_col in scores.columns:
                out[f"{base_name}_focal_{model}"] = scores[focal_col]
                out[f"{base_name}_opponent_{model}"] = scores[opponent_col]

    return out


def add_early_game_groups(appearances: pd.DataFrame) -> pd.DataFrame:
    """Add exact, cumulative, and binned early-career group variables."""

    out = appearances.copy()
    for threshold in CUMULATIVE_THRESHOLDS:
        out[f"first_{threshold}"] = out["appearance_number"] <= threshold

    conditions = [
        out["appearance_number"] == 1,
        out["appearance_number"].between(2, 5),
        out["appearance_number"].between(6, 10),
        out["appearance_number"].between(11, 20),
        out["appearance_number"].between(21, 50),
        out["appearance_number"] >= 51,
    ]
    out["appearance_stage"] = np.select(conditions, STAGE_LABELS, default=pd.NA)
    return out


def build_appearance_dataset(scores: pd.DataFrame) -> pd.DataFrame:
    """Expand each match into two focal player-appearance rows."""

    parts = [build_side_appearances(scores, "a"), build_side_appearances(scores, "b")]
    appearances = pd.concat(parts, ignore_index=True)
    appearances = add_early_game_groups(appearances)
    return appearances.sort_values(["match_sequence", "focal_side"]).reset_index(drop=True)


def build_exact_appearance_counts(appearances: pd.DataFrame) -> pd.DataFrame:
    """Count players and matches by exact appearance number for 1 through 20."""

    rows = []
    for appearance_number in range(1, 21):
        sub = appearances.loc[appearances["appearance_number"] == appearance_number]
        rows.append(
            {
                "appearance_number": appearance_number,
                "count_appearances": int(len(sub)),
                "count_players": int(sub["player_id"].nunique()),
                "count_matches": int(sub["match_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def validate_appearance_dataset(
    scores: pd.DataFrame,
    appearances: pd.DataFrame,
    exact_counts: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> None:
    """Run consistency checks for the player-appearance output."""

    add_check(
        rows,
        "appearance_rows_22758",
        len(appearances) == EXPECTED_APPEARANCE_ROWS,
        len(appearances),
        EXPECTED_APPEARANCE_ROWS,
    )

    rows_per_match = appearances.groupby("match_id").size()
    bad_two_row_matches = int((rows_per_match != 2).sum())
    add_check(rows, "two_appearances_per_match", bad_two_row_matches == 0, bad_two_row_matches, 0)

    side_counts = appearances.groupby(["match_id", "focal_side"]).size().unstack(fill_value=0)
    has_a_b_once = bool({"A", "B"}.issubset(side_counts.columns) and (side_counts[["A", "B"]] == 1).all().all())
    add_check(rows, "each_match_has_one_A_and_one_B_appearance", has_a_b_once, "checked", True)

    duplicate_player_match = int(appearances.duplicated(["match_id", "player_id"]).sum())
    add_check(rows, "no_duplicate_player_match_appearances", duplicate_player_match == 0, duplicate_player_match, 0)

    appearance_formula_ok = bool((appearances["appearance_number"] == appearances["total_games_before"] + 1).all())
    add_check(rows, "appearance_number_equals_total_games_before_plus_one", appearance_formula_ok, "checked", True)

    debut_formula_ok = bool((appearances["debut_flag"] == (appearances["total_games_before"] == 0)).all())
    add_check(rows, "debut_flag_matches_zero_previous_games", debut_formula_ok, "checked", True)

    outcome_values = sorted(appearances["outcome_focal"].dropna().unique().tolist())
    add_check(rows, "outcome_focal_binary", set(outcome_values).issubset({0, 1}), outcome_values, "{0,1}")

    probability_cols = [f"p_focal_{alias}" for alias in MODEL_PROBABILITY_COLUMNS]
    missing_probability_values = int(appearances[probability_cols].isna().sum().sum())
    probabilities_in_range = bool(appearances[probability_cols].apply(lambda col: col.between(0, 1).all()).all())
    add_check(rows, "focal_probabilities_not_missing", missing_probability_values == 0, missing_probability_values, 0)
    add_check(rows, "focal_probabilities_in_range", probabilities_in_range, "checked", "[0,1]")

    a_rows = appearances.loc[appearances["focal_side"] == "A", ["match_id", "outcome_focal"]].rename(
        columns={"outcome_focal": "outcome_focal_a"}
    )
    b_rows = appearances.loc[appearances["focal_side"] == "B", ["match_id", "outcome_focal"]].rename(
        columns={"outcome_focal": "outcome_focal_b"}
    )
    outcome_check = (
        scores[["match_id", "outcome_a"]]
        .merge(a_rows, on="match_id", how="left", validate="one_to_one")
        .merge(b_rows, on="match_id", how="left", validate="one_to_one")
    )
    focal_outcome_orientation_ok = bool(
        (outcome_check["outcome_focal_a"] == outcome_check["outcome_a"]).all()
        and (outcome_check["outcome_focal_b"] == 1 - outcome_check["outcome_a"]).all()
    )
    add_check(rows, "focal_outcome_orientation_correct", focal_outcome_orientation_ok, "checked", True)

    first_thresholds_ok = []
    for threshold in CUMULATIVE_THRESHOLDS:
        first_thresholds_ok.append(bool((appearances[f"first_{threshold}"] == (appearances["appearance_number"] <= threshold)).all()))
    add_check(rows, "cumulative_first_n_flags_correct", all(first_thresholds_ok), f"{sum(first_thresholds_ok)}/6", "6/6")

    stage_not_missing = int(appearances["appearance_stage"].isna().sum())
    known_stages_only = set(appearances["appearance_stage"].dropna().unique()).issubset(set(STAGE_LABELS))
    add_check(rows, "appearance_stage_not_missing", stage_not_missing == 0, stage_not_missing, 0)
    add_check(rows, "appearance_stage_labels_valid", known_stages_only, sorted(appearances["appearance_stage"].unique()), STAGE_LABELS)

    exact_counts_complete = exact_counts["appearance_number"].tolist() == list(range(1, 21))
    add_check(rows, "exact_appearance_counts_cover_1_to_20", exact_counts_complete, exact_counts["appearance_number"].tolist(), "1..20")

    first_twenty_rows = int((appearances["appearance_number"] <= 20).sum())
    counted_first_twenty = int(exact_counts["count_appearances"].sum())
    add_check(rows, "exact_counts_sum_to_first_20_appearances", counted_first_twenty == first_twenty_rows, counted_first_twenty, first_twenty_rows)


def write_validation_checks(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Write validation checks and print a PASS/FAIL console summary."""

    validation = pd.DataFrame(rows)
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")

    failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
    status = "PASS" if failed_errors.empty else "FAIL"
    print(status)
    for row in validation.itertuples(index=False):
        row_status = "PASS" if bool(row.passed) else "FAIL"
        detail = f" | detail={row.detail}" if isinstance(row.detail, str) and row.detail else ""
        print(f"[{row_status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")
    return validation


def load_appearance_dataset() -> pd.DataFrame:
    """Load the Step 34 player-appearance dataset as the only metric input."""

    if not APPEARANCE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Required appearance dataset not found: {APPEARANCE_DATASET_PATH}. "
            "Run Step 34 Part 1 before calculating model performance."
        )
    return pd.read_csv(APPEARANCE_DATASET_PATH, low_memory=False)


def available_performance_models(appearances: pd.DataFrame) -> list[str]:
    """Return requested performance models with available focal probabilities."""

    return [model for model in PERFORMANCE_MODEL_ORDER if f"p_focal_{model}" in appearances.columns]


def validate_metric_input(appearances: pd.DataFrame, models: list[str], rows: list[dict[str, Any]]) -> bool:
    """Validate the appearance dataset before metric calculation."""

    required = [
        "match_id",
        "player_id",
        "appearance_number",
        "outcome_focal",
        "appearance_stage",
        *[f"first_{threshold}" for threshold in CUMULATIVE_THRESHOLDS],
    ]
    missing_required = [col for col in required if col not in appearances.columns]
    add_check(
        rows,
        "metric_input_required_columns_present",
        not missing_required,
        missing_required if missing_required else "none",
        "all required appearance dataset columns",
    )
    add_check(
        rows,
        "metric_input_rows_22758",
        len(appearances) == EXPECTED_APPEARANCE_ROWS,
        len(appearances),
        EXPECTED_APPEARANCE_ROWS,
    )

    missing_models = [model for model in PERFORMANCE_MODEL_ORDER if f"p_focal_{model}" not in appearances.columns]
    add_check(
        rows,
        "requested_model_probability_columns_available",
        not missing_models,
        missing_models if missing_models else "none",
        "Validation_best_Elo, Glicko_low_fixed, Glicko_C0_fixed, best_AdaptiveK",
    )
    add_check(rows, "available_model_count", len(models) >= 4, len(models), ">=4")

    if missing_required or len(models) < 4:
        return False

    outcome_values = sorted(appearances["outcome_focal"].dropna().unique().tolist())
    add_check(rows, "metric_input_outcome_focal_binary", set(outcome_values).issubset({0, 1}), outcome_values, "{0,1}")

    probability_cols = [f"p_focal_{model}" for model in models]
    missing_probability_values = int(appearances[probability_cols].isna().sum().sum())
    probabilities_in_range = bool(appearances[probability_cols].apply(lambda col: col.between(0, 1).all()).all())
    add_check(rows, "metric_input_probabilities_not_missing", missing_probability_values == 0, missing_probability_values, 0)
    add_check(rows, "metric_input_probabilities_in_range", probabilities_in_range, "checked", "[0,1]")

    duplicate_player_match = int(appearances.duplicated(["match_id", "player_id"]).sum())
    add_check(rows, "metric_input_no_duplicate_player_match", duplicate_player_match == 0, duplicate_player_match, 0)

    return True


def score_model(group: pd.DataFrame, model: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Step 33-style Brier, log loss, and accuracy for one model."""

    p = group[f"p_focal_{model}"].astype(float).clip(0.0, 1.0)
    y = group["outcome_focal"].astype(float)
    clipped = p.clip(EPS, 1.0 - EPS)
    brier = (p - y) ** 2
    log_loss = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(int)
    return brier, log_loss, correct


def model_metric_row(
    group: pd.DataFrame,
    model: str,
    group_type: str,
    group_name: str,
    group_order: int,
    threshold: int | float | None = None,
    appearance_number: int | float | None = None,
    appearance_stage: str | None = None,
) -> dict[str, Any]:
    """Return one model-performance row for one early-game subgroup."""

    brier, log_loss, correct = score_model(group, model)
    p = group[f"p_focal_{model}"].astype(float)
    y = group["outcome_focal"].astype(float)
    return {
        "group_type": group_type,
        "group": group_name,
        "group_order": int(group_order),
        "threshold": threshold if threshold is not None else np.nan,
        "appearance_number": appearance_number if appearance_number is not None else np.nan,
        "appearance_stage": appearance_stage if appearance_stage is not None else "",
        "model": model,
        "model_display": MODEL_LABELS.get(model, model),
        "model_family": MODEL_FAMILIES.get(model, ""),
        "appearances": int(len(group)),
        "players": int(group["player_id"].nunique()),
        "matches": int(group["match_id"].nunique()),
        "min_appearance_number": int(group["appearance_number"].min()),
        "max_appearance_number": int(group["appearance_number"].max()),
        "brier": float(brier.mean()),
        "log_loss": float(log_loss.mean()),
        "accuracy": float(correct.mean()),
        "mean_predicted_probability": float(p.mean()),
        "empirical_win_rate": float(y.mean()),
        "probability_orientation": "focal player win probability",
        "accuracy_tie_rule": "p_focal >= 0.5 predicts focal player win",
    }


def calculate_cumulative_threshold_performance(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Calculate model metrics for first-N cumulative early-game groups."""

    rows: list[dict[str, Any]] = []
    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        group = appearances.loc[appearances[group_name].astype(bool)].copy()
        for model in models:
            rows.append(
                model_metric_row(
                    group,
                    model,
                    group_type="cumulative_threshold",
                    group_name=group_name,
                    group_order=threshold,
                    threshold=threshold,
                )
            )
    return pd.DataFrame(rows)


def calculate_stage_bin_performance(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Calculate model metrics for mutually exclusive appearance-stage bins."""

    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(STAGE_LABELS, start=1):
        group = appearances.loc[appearances["appearance_stage"] == stage].copy()
        for model in models:
            rows.append(
                model_metric_row(
                    group,
                    model,
                    group_type="appearance_stage",
                    group_name=stage,
                    group_order=order,
                    appearance_stage=stage,
                )
            )
    return pd.DataFrame(rows)


def calculate_exact_appearance_performance(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Calculate model metrics for exact appearance numbers 1 through 20."""

    rows: list[dict[str, Any]] = []
    for appearance_number in range(1, 21):
        group = appearances.loc[appearances["appearance_number"] == appearance_number].copy()
        for model in models:
            rows.append(
                model_metric_row(
                    group,
                    model,
                    group_type="exact_appearance_number",
                    group_name=f"appearance_{appearance_number}",
                    group_order=appearance_number,
                    appearance_number=appearance_number,
                )
            )
    return pd.DataFrame(rows)


def pairwise_difference_row(
    group: pd.DataFrame,
    group_type: str,
    group_name: str,
    group_order: int,
    comparison: dict[str, str],
    threshold: int | float | None = None,
    appearance_number: int | float | None = None,
    appearance_stage: str | None = None,
) -> dict[str, Any]:
    """Return paired model differences for one group and comparison."""

    left_model = comparison["left_model"]
    right_model = comparison["right_model"]
    left_brier, left_log_loss, left_correct = score_model(group, left_model)
    right_brier, right_log_loss, right_correct = score_model(group, right_model)

    return {
        "group_type": group_type,
        "group": group_name,
        "group_order": int(group_order),
        "threshold": threshold if threshold is not None else np.nan,
        "appearance_number": appearance_number if appearance_number is not None else np.nan,
        "appearance_stage": appearance_stage if appearance_stage is not None else "",
        "comparison": comparison["comparison"],
        "comparison_label": comparison["comparison_label"],
        "left_model": left_model,
        "right_model": right_model,
        "left_model_display": MODEL_LABELS.get(left_model, left_model),
        "right_model_display": MODEL_LABELS.get(right_model, right_model),
        "appearances": int(len(group)),
        "paired_appearances": int(len(group)),
        "players": int(group["player_id"].nunique()),
        "matches": int(group["match_id"].nunique()),
        "left_brier": float(left_brier.mean()),
        "right_brier": float(right_brier.mean()),
        "delta_brier": float((left_brier - right_brier).mean()),
        "left_log_loss": float(left_log_loss.mean()),
        "right_log_loss": float(right_log_loss.mean()),
        "delta_log_loss": float((left_log_loss - right_log_loss).mean()),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "delta_accuracy": float((right_correct - left_correct).mean()),
        "delta_brier_definition": "left_model_brier - right_model_brier",
        "delta_log_loss_definition": "left_model_log_loss - right_model_log_loss",
        "delta_accuracy_definition": "right_model_accuracy - left_model_accuracy",
        "positive_delta_means": comparison["positive_delta_means"],
        "step33_equivalent": comparison["step33_equivalent"],
    }


def calculate_pairwise_differences(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Calculate paired model differences for all Step 34 performance groups."""

    rows: list[dict[str, Any]] = []
    available = set(models)
    comparisons = [
        comparison
        for comparison in PAIRWISE_COMPARISONS
        if comparison["left_model"] in available and comparison["right_model"] in available
    ]

    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        group = appearances.loc[appearances[group_name].astype(bool)].copy()
        for comparison in comparisons:
            rows.append(
                pairwise_difference_row(
                    group,
                    group_type="cumulative_threshold",
                    group_name=group_name,
                    group_order=threshold,
                    comparison=comparison,
                    threshold=threshold,
                )
            )

    for order, stage in enumerate(STAGE_LABELS, start=1):
        group = appearances.loc[appearances["appearance_stage"] == stage].copy()
        for comparison in comparisons:
            rows.append(
                pairwise_difference_row(
                    group,
                    group_type="appearance_stage",
                    group_name=stage,
                    group_order=order,
                    comparison=comparison,
                    appearance_stage=stage,
                )
            )

    for appearance_number in range(1, 21):
        group = appearances.loc[appearances["appearance_number"] == appearance_number].copy()
        for comparison in comparisons:
            rows.append(
                pairwise_difference_row(
                    group,
                    group_type="exact_appearance_number",
                    group_name=f"appearance_{appearance_number}",
                    group_order=appearance_number,
                    comparison=comparison,
                    appearance_number=appearance_number,
                )
            )

    return pd.DataFrame(rows)


def validate_metric_outputs(
    appearances: pd.DataFrame,
    models: list[str],
    cumulative: pd.DataFrame,
    stage: pd.DataFrame,
    exact: pd.DataFrame,
    pairwise: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> None:
    """Validate the Step 34 model-performance summary outputs."""

    expected_model_rows = len(models)
    cumulative_expected_rows = len(CUMULATIVE_THRESHOLDS) * expected_model_rows
    stage_expected_rows = len(STAGE_LABELS) * expected_model_rows
    exact_expected_rows = 20 * expected_model_rows
    available_comparisons = [
        comparison
        for comparison in PAIRWISE_COMPARISONS
        if comparison["left_model"] in models and comparison["right_model"] in models
    ]
    pairwise_expected_rows = (len(CUMULATIVE_THRESHOLDS) + len(STAGE_LABELS) + 20) * len(available_comparisons)

    add_check(rows, "cumulative_performance_row_count", len(cumulative) == cumulative_expected_rows, len(cumulative), cumulative_expected_rows)
    add_check(rows, "stage_performance_row_count", len(stage) == stage_expected_rows, len(stage), stage_expected_rows)
    add_check(rows, "exact_performance_row_count", len(exact) == exact_expected_rows, len(exact), exact_expected_rows)
    add_check(rows, "pairwise_difference_row_count", len(pairwise) == pairwise_expected_rows, len(pairwise), pairwise_expected_rows)

    cumulative_counts_ok = True
    cumulative_details = []
    for threshold in CUMULATIVE_THRESHOLDS:
        expected = int((appearances["appearance_number"] <= threshold).sum())
        observed = cumulative.loc[cumulative["group"] == f"first_{threshold}", "appearances"].unique().tolist()
        ok = observed == [expected]
        cumulative_counts_ok = cumulative_counts_ok and ok
        cumulative_details.append(f"first_{threshold}:{observed}->{expected}")
    add_check(rows, "cumulative_group_sample_counts_correct", cumulative_counts_ok, "; ".join(cumulative_details), "each first_N count matches appearance_number <= N")

    stage_counts = appearances["appearance_stage"].value_counts().reindex(STAGE_LABELS)
    stage_cover = bool(stage_counts.notna().all() and int(stage_counts.sum()) == len(appearances))
    stage_table_counts_ok = True
    stage_details = []
    for stage_label in STAGE_LABELS:
        expected = int(stage_counts.loc[stage_label])
        observed = stage.loc[stage["group"] == stage_label, "appearances"].unique().tolist()
        ok = observed == [expected]
        stage_table_counts_ok = stage_table_counts_ok and ok
        stage_details.append(f"{stage_label}:{observed}->{expected}")
    add_check(rows, "stage_bins_mutually_exclusive_cover_all", stage_cover, int(stage_counts.sum()), len(appearances))
    add_check(rows, "stage_bin_sample_counts_correct", stage_table_counts_ok, "; ".join(stage_details), "each stage count matches appearance dataset")

    exact_counts_ok = True
    exact_details = []
    for appearance_number in range(1, 21):
        expected = int((appearances["appearance_number"] == appearance_number).sum())
        observed = exact.loc[exact["appearance_number"] == appearance_number, "appearances"].unique().tolist()
        ok = observed == [expected]
        exact_counts_ok = exact_counts_ok and ok
        exact_details.append(f"{appearance_number}:{observed}->{expected}")
    add_check(rows, "exact_appearance_1_to_20_sample_counts_correct", exact_counts_ok, "; ".join(exact_details), "each exact appearance count matches dataset")

    metric_cols = ["brier", "log_loss", "accuracy", "mean_predicted_probability", "empirical_win_rate", "appearances"]
    performance_tables = {
        "cumulative": cumulative,
        "stage": stage,
        "exact": exact,
    }
    for table_name, table in performance_tables.items():
        missing_metrics = int(table[metric_cols].isna().sum().sum())
        add_check(rows, f"no_nan_metrics_{table_name}", missing_metrics == 0, missing_metrics, 0)

    pairwise_metric_cols = ["delta_brier", "delta_log_loss", "delta_accuracy", "appearances", "paired_appearances"]
    pairwise_nan_metrics = int(pairwise[pairwise_metric_cols].isna().sum().sum())
    add_check(rows, "no_nan_metrics_pairwise", pairwise_nan_metrics == 0, pairwise_nan_metrics, 0)

    same_appearances = bool((pairwise["appearances"] == pairwise["paired_appearances"]).all()) if not pairwise.empty else False
    add_check(rows, "pairwise_comparisons_same_appearances", same_appearances, "checked", True)

    expected_comparison_names = [comparison["comparison"] for comparison in available_comparisons]
    observed_comparison_names = sorted(pairwise["comparison"].unique().tolist())
    add_check(
        rows,
        "pairwise_comparisons_present",
        observed_comparison_names == sorted(expected_comparison_names),
        observed_comparison_names,
        sorted(expected_comparison_names),
    )

    probability_metric_range_cols = ["mean_predicted_probability", "empirical_win_rate", "accuracy"]
    range_ok = all(
        bool(table[probability_metric_range_cols].apply(lambda col: col.between(0, 1).all()).all())
        for table in performance_tables.values()
    )
    add_check(rows, "probability_metrics_in_range", range_ok, "checked", "[0,1]")


def write_metric_validation_checks(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Write metric validation checks and print a PASS/FAIL console summary."""

    validation = pd.DataFrame(rows)
    validation.to_csv(METRIC_VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")

    failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
    status = "PASS" if failed_errors.empty else "FAIL"
    print(status)
    for row in validation.itertuples(index=False):
        row_status = "PASS" if bool(row.passed) else "FAIL"
        detail = f" | detail={row.detail}" if isinstance(row.detail, str) and row.detail else ""
        print(f"[{row_status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")
    return validation


def read_existing_step34_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Step 34 outputs needed for bootstrap and figures."""

    required_paths = [
        APPEARANCE_DATASET_PATH,
        EXACT_APPEARANCE_COUNTS_PATH,
        CUMULATIVE_PERFORMANCE_PATH,
        STAGE_PERFORMANCE_PATH,
        EXACT_PERFORMANCE_PATH,
        PAIRWISE_DIFFERENCES_PATH,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Step 34 output(s) not found: {missing}")

    return (
        pd.read_csv(APPEARANCE_DATASET_PATH, low_memory=False),
        pd.read_csv(EXACT_APPEARANCE_COUNTS_PATH, low_memory=False),
        pd.read_csv(CUMULATIVE_PERFORMANCE_PATH, low_memory=False),
        pd.read_csv(STAGE_PERFORMANCE_PATH, low_memory=False),
        pd.read_csv(EXACT_PERFORMANCE_PATH, low_memory=False),
        pd.read_csv(PAIRWISE_DIFFERENCES_PATH, low_memory=False),
    )


def build_group_specs(appearances: pd.DataFrame) -> list[dict[str, Any]]:
    """Create the early-game group definitions shared by metrics and bootstrap."""

    groups: list[dict[str, Any]] = []
    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        groups.append(
            {
                "group_type": "cumulative_threshold",
                "group": group_name,
                "group_order": threshold,
                "threshold": threshold,
                "appearance_number": np.nan,
                "appearance_stage": "",
                "mask": appearances[group_name].astype(bool),
            }
        )

    for order, stage in enumerate(STAGE_LABELS, start=1):
        groups.append(
            {
                "group_type": "appearance_stage",
                "group": stage,
                "group_order": order,
                "threshold": np.nan,
                "appearance_number": np.nan,
                "appearance_stage": stage,
                "mask": appearances["appearance_stage"] == stage,
            }
        )

    for appearance_number in range(1, 21):
        groups.append(
            {
                "group_type": "exact_appearance_number",
                "group": f"appearance_{appearance_number}",
                "group_order": appearance_number,
                "threshold": np.nan,
                "appearance_number": appearance_number,
                "appearance_stage": "",
                "mask": appearances["appearance_number"] == appearance_number,
            }
        )
    return groups


def add_pairwise_score_columns(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Add per-appearance scores and paired difference columns for bootstrap."""

    out = appearances.copy()
    for model in models:
        brier, log_loss, correct = score_model(out, model)
        out[f"brier_{model}"] = brier
        out[f"log_loss_{model}"] = log_loss
        out[f"correct_{model}"] = correct

    for comparison in PAIRWISE_COMPARISONS:
        left = comparison["left_model"]
        right = comparison["right_model"]
        if left not in models or right not in models:
            continue
        name = comparison["comparison"]
        out[f"diff_brier_{name}"] = out[f"brier_{left}"] - out[f"brier_{right}"]
        out[f"diff_log_loss_{name}"] = out[f"log_loss_{left}"] - out[f"log_loss_{right}"]
        out[f"diff_accuracy_{name}"] = out[f"correct_{right}"] - out[f"correct_{left}"]
    return out


def bootstrap_mean_ci(group: pd.DataFrame, value_col: str, seed: int, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for a paired mean difference.

    The primary resampling unit is focal player, because the analysis dataset is
    at player-appearance level and a player can contribute multiple early-career
    appearances. If a group has fewer than two player clusters, the function
    falls back to appearance-level resampling.
    """

    if group.empty:
        return {
            "point_estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_type": "not_run_empty_group",
            "bootstrap_replications": 0,
            "cluster_variable": "",
            "clusters": 0,
            "small_sample_warning": True,
        }

    values = group[value_col].astype(float)
    point = float(values.mean())
    rng = np.random.default_rng(seed)

    cluster_counts = group["player_id"].nunique()
    if cluster_counts >= 2:
        grouped = group.groupby("player_id", sort=False)[value_col].agg(["sum", "count"]).reset_index(drop=True)
        sums = grouped["sum"].to_numpy(dtype=float)
        counts = grouped["count"].to_numpy(dtype=float)
        estimates = np.empty(reps)
        for i in range(reps):
            idx = rng.integers(0, len(grouped), len(grouped))
            estimates[i] = sums[idx].sum() / counts[idx].sum()
        bootstrap_type = "player_cluster"
        cluster_variable = "player_id"
        clusters = int(len(grouped))
    else:
        raw = values.to_numpy(dtype=float)
        estimates = np.empty(reps)
        for i in range(reps):
            idx = rng.integers(0, len(raw), len(raw))
            estimates[i] = raw[idx].mean()
        bootstrap_type = "appearance_level"
        cluster_variable = "appearance_row"
        clusters = int(len(raw))

    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "bootstrap_type": bootstrap_type,
        "bootstrap_replications": reps,
        "cluster_variable": cluster_variable,
        "clusters": clusters,
        "small_sample_warning": bool(len(group) < 50 or clusters < 10),
    }


def calculate_bootstrap_confidence_intervals(appearances: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Calculate bootstrap CIs for every pairwise comparison and analysis group."""

    scored = add_pairwise_score_columns(appearances, models)
    groups = build_group_specs(scored)
    comparisons = [
        comparison
        for comparison in PAIRWISE_COMPARISONS
        if comparison["left_model"] in models and comparison["right_model"] in models
    ]
    metrics = [
        ("delta_brier", "diff_brier_{comparison}", "left_model_brier - right_model_brier"),
        ("delta_log_loss", "diff_log_loss_{comparison}", "left_model_log_loss - right_model_log_loss"),
        ("delta_accuracy", "diff_accuracy_{comparison}", "right_model_accuracy - left_model_accuracy"),
    ]

    rows: list[dict[str, Any]] = []
    seed_counter = 0
    for group_spec in groups:
        group = scored.loc[group_spec["mask"]].copy()
        for comparison in comparisons:
            for metric, template, definition in metrics:
                value_col = template.format(comparison=comparison["comparison"])
                seed_counter += 1
                ci = bootstrap_mean_ci(group, value_col, seed=RANDOM_SEED + seed_counter)
                rows.append(
                    {
                        "group_type": group_spec["group_type"],
                        "group": group_spec["group"],
                        "group_order": group_spec["group_order"],
                        "threshold": group_spec["threshold"],
                        "appearance_number": group_spec["appearance_number"],
                        "appearance_stage": group_spec["appearance_stage"],
                        "comparison": comparison["comparison"],
                        "comparison_label": comparison["comparison_label"],
                        "left_model": comparison["left_model"],
                        "right_model": comparison["right_model"],
                        "metric": metric,
                        "metric_definition": definition,
                        "appearances": int(len(group)),
                        "players": int(group["player_id"].nunique()),
                        "matches": int(group["match_id"].nunique()),
                        "positive_delta_means": comparison["positive_delta_means"],
                        **ci,
                    }
                )
    return pd.DataFrame(rows)


def prepare_plot_style() -> None:
    """Apply a consistent, restrained matplotlib style for Meeting 7 figures."""

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save_current_figure(path: Path) -> None:
    """Save and close the current matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def create_exact_sample_size_figure(exact_counts: pd.DataFrame) -> Path:
    """Figure 1: sample size by exact appearance number."""

    p = FIGURE_DIR / "34_fig01_exact_appearance_sample_size.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(exact_counts["appearance_number"], exact_counts["count_appearances"], color="#4C6A92")
    ax.set_xlabel("Exact appearance number")
    ax.set_ylabel("Player appearances")
    ax.set_title("Sample size for first 20 recorded appearances")
    ax.set_xticks(range(1, 21))
    save_current_figure(p)
    return p


def create_cumulative_brier_figure(cumulative: pd.DataFrame) -> Path:
    """Figure 2: cumulative first-N Brier score by model."""

    p = FIGURE_DIR / "34_fig02_cumulative_brier_by_model.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for model in FIGURE_MODEL_ORDER:
        sub = cumulative.loc[cumulative["model"] == model].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(
            sub["threshold"],
            sub["brier"],
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel("Cumulative first N appearances")
    ax.set_ylabel("Brier score")
    ax.set_title("Early-game Brier score by cumulative career stage")
    ax.set_xticks(CUMULATIVE_THRESHOLDS)
    ax.legend(frameon=False)
    save_current_figure(p)
    return p


def create_cumulative_delta_brier_figure(pairwise: pd.DataFrame, bootstrap: pd.DataFrame) -> Path:
    """Figure 3: Elo-minus-Glicko cumulative Delta Brier with bootstrap CIs."""

    p = FIGURE_DIR / "34_fig03_cumulative_delta_brier_elo_minus_glicko_ci.png"
    comparison = "Validation_best_Elo_minus_Glicko_low_fixed"
    point = pairwise.loc[
        (pairwise["group_type"] == "cumulative_threshold") & (pairwise["comparison"] == comparison)
    ].sort_values("threshold")
    ci = bootstrap.loc[
        (bootstrap["group_type"] == "cumulative_threshold")
        & (bootstrap["comparison"] == comparison)
        & (bootstrap["metric"] == "delta_brier")
    ].sort_values("threshold")

    merged = point.merge(
        ci[["group", "ci_lower", "ci_upper"]],
        on="group",
        how="left",
        validate="one_to_one",
    )
    lower = merged["delta_brier"] - merged["ci_lower"]
    upper = merged["ci_upper"] - merged["delta_brier"]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.errorbar(
        merged["threshold"],
        merged["delta_brier"],
        yerr=np.vstack([lower, upper]),
        marker="o",
        linewidth=1.8,
        capsize=4,
        color=MODEL_COLORS["Glicko_low_fixed"],
    )
    ax.set_xlabel("Cumulative first N appearances")
    ax.set_ylabel("Delta Brier: Elo - Glicko")
    ax.set_title("Validation-best Elo vs Glicko low inflation")
    ax.set_xticks(CUMULATIVE_THRESHOLDS)
    save_current_figure(p)
    return p


def create_stage_brier_figure(stage: pd.DataFrame) -> Path:
    """Figure 4: Brier score by mutually exclusive appearance-stage bin."""

    p = FIGURE_DIR / "34_fig04_stage_brier_by_model.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(STAGE_LABELS))
    for model in FIGURE_MODEL_ORDER:
        sub = stage.loc[stage["model"] == model].set_index("group").reindex(STAGE_LABELS)
        if sub.empty:
            continue
        ax.plot(x, sub["brier"], marker="o", linewidth=1.8, color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Brier score")
    ax.set_title("Brier score by early-career stage")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.legend(frameon=False)
    save_current_figure(p)
    return p


def create_exact_brier_trend_figure(exact: pd.DataFrame) -> Path:
    """Figure 5: exact appearance-number Brier trend for appearances 1-20."""

    p = FIGURE_DIR / "34_fig05_exact_appearance_brier_trend.png"
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for model in FIGURE_MODEL_ORDER:
        sub = exact.loc[exact["model"] == model].sort_values("appearance_number")
        if sub.empty:
            continue
        ax.plot(
            sub["appearance_number"],
            sub["brier"],
            marker="o",
            linewidth=1.4,
            markersize=3.5,
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel("Exact appearance number")
    ax.set_ylabel("Brier score")
    ax.set_title("Brier trend across first 20 recorded appearances")
    ax.set_xticks(range(1, 21))
    ax.legend(frameon=False, ncol=2)
    save_current_figure(p)
    return p


def create_stage_predicted_vs_empirical_figure(stage: pd.DataFrame) -> Path:
    """Figure 6: mean focal win probability vs empirical win rate by stage."""

    p = FIGURE_DIR / "34_fig06_stage_predicted_vs_empirical_win_rate.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(STAGE_LABELS))

    empirical = (
        stage.loc[stage["model"] == "Validation_best_Elo"]
        .set_index("group")
        .reindex(STAGE_LABELS)["empirical_win_rate"]
    )
    ax.plot(x, empirical, marker="o", linewidth=2.2, color="#222222", label="Empirical win rate")

    for model in ["Validation_best_Elo", "Glicko_low_fixed", "best_AdaptiveK"]:
        sub = stage.loc[stage["model"] == model].set_index("group").reindex(STAGE_LABELS)
        ax.plot(
            x,
            sub["mean_predicted_probability"],
            marker="o",
            linewidth=1.5,
            linestyle="--",
            color=MODEL_COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Focal player win probability")
    ax.set_title("Predicted probability vs empirical win rate")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    save_current_figure(p)
    return p


def create_early_game_figures(
    exact_counts: pd.DataFrame,
    cumulative: pd.DataFrame,
    stage: pd.DataFrame,
    exact: pd.DataFrame,
    pairwise: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Create Meeting 7 early-game figures and return a manifest."""

    prepare_plot_style()
    figure_specs = [
        (
            "34_fig01_exact_appearance_sample_size",
            create_exact_sample_size_figure(exact_counts),
            "Sample size by exact appearance number for appearances 1-20.",
            "34_exact_appearance_counts.csv",
        ),
        (
            "34_fig02_cumulative_brier_by_model",
            create_cumulative_brier_figure(cumulative),
            "Cumulative first-N Brier score by model.",
            "34_cumulative_threshold_model_performance.csv",
        ),
        (
            "34_fig03_cumulative_delta_brier_elo_minus_glicko_ci",
            create_cumulative_delta_brier_figure(pairwise, bootstrap),
            "Validation-best Elo minus Glicko low-inflation Delta Brier with player-cluster bootstrap CIs.",
            "34_pairwise_model_differences.csv; 34_bootstrap_confidence_intervals.csv",
        ),
        (
            "34_fig04_stage_brier_by_model",
            create_stage_brier_figure(stage),
            "Brier score by mutually exclusive appearance-stage bin.",
            "34_stage_bin_model_performance.csv",
        ),
        (
            "34_fig05_exact_appearance_brier_trend",
            create_exact_brier_trend_figure(exact),
            "Brier trend across exact appearances 1-20.",
            "34_exact_appearance_model_performance.csv",
        ),
        (
            "34_fig06_stage_predicted_vs_empirical_win_rate",
            create_stage_predicted_vs_empirical_figure(stage),
            "Mean focal predicted win probability compared with empirical win rate by stage.",
            "34_stage_bin_model_performance.csv",
        ),
    ]

    rows = []
    for figure_id, path, description, source in figure_specs:
        rows.append(
            {
                "figure_id": figure_id,
                "path": str(path),
                "filename": path.name,
                "description": description,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def validate_bootstrap_and_figures(
    appearances: pd.DataFrame,
    pairwise: pd.DataFrame,
    bootstrap: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> None:
    """Validate bootstrap CI and figure outputs."""

    comparisons = [
        comparison
        for comparison in PAIRWISE_COMPARISONS
        if f"p_focal_{comparison['left_model']}" in appearances.columns and f"p_focal_{comparison['right_model']}" in appearances.columns
    ]
    expected_rows = (len(CUMULATIVE_THRESHOLDS) + len(STAGE_LABELS) + 20) * len(comparisons) * 3
    add_check(rows, "bootstrap_ci_row_count", len(bootstrap) == expected_rows, len(bootstrap), expected_rows)

    no_nan = int(bootstrap[["point_estimate", "ci_lower", "ci_upper"]].isna().sum().sum())
    add_check(rows, "bootstrap_ci_no_nan_core_values", no_nan == 0, no_nan, 0)

    order_ok = bool((bootstrap["ci_lower"] <= bootstrap["ci_upper"]).all())
    add_check(rows, "bootstrap_ci_bounds_ordered", order_ok, "checked", "ci_lower <= ci_upper")

    reps_ok = bool((bootstrap["bootstrap_replications"] == BOOTSTRAP_REPS).all())
    add_check(rows, "bootstrap_replications_2000", reps_ok, sorted(bootstrap["bootstrap_replications"].unique().tolist()), BOOTSTRAP_REPS)

    player_cluster_share = float((bootstrap["bootstrap_type"] == "player_cluster").mean()) if len(bootstrap) else 0.0
    add_check(rows, "bootstrap_uses_player_cluster_resampling", player_cluster_share == 1.0, player_cluster_share, 1.0)

    pairwise_key_cols = ["group_type", "group", "comparison"]
    point = pairwise[pairwise_key_cols + ["delta_brier", "delta_log_loss", "delta_accuracy"]].copy()
    long_point = point.melt(
        id_vars=pairwise_key_cols,
        value_vars=["delta_brier", "delta_log_loss", "delta_accuracy"],
        var_name="metric",
        value_name="pairwise_point_estimate",
    )
    checked = bootstrap.merge(long_point, on=[*pairwise_key_cols, "metric"], how="left", validate="one_to_one")
    max_gap = float((checked["point_estimate"] - checked["pairwise_point_estimate"]).abs().max())
    add_check(rows, "bootstrap_points_match_pairwise_table", max_gap < 1e-12, max_gap, "<1e-12")

    figure_count_ok = len(figure_manifest) == 6
    add_check(rows, "figure_manifest_row_count", figure_count_ok, len(figure_manifest), 6)

    missing_figures = []
    empty_figures = []
    for path_value in figure_manifest["path"].tolist():
        path = Path(path_value)
        if not path.exists():
            missing_figures.append(path.name)
        elif path.stat().st_size <= 0:
            empty_figures.append(path.name)
    add_check(rows, "all_figures_exist", not missing_figures, missing_figures if missing_figures else "none", "all manifest paths exist")
    add_check(rows, "all_figures_nonempty", not empty_figures, empty_figures if empty_figures else "none", "all figure files non-empty")

    no_calibration_outputs = not any("calibration" in name.name.lower() and name.name.startswith("34_") for name in OUTPUT_DIR.glob("34_*"))
    add_check(rows, "no_step34_calibration_outputs_created", no_calibration_outputs, "checked", True)


def write_bootstrap_figure_validation_checks(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Write bootstrap/figure validation checks and print PASS/FAIL."""

    validation = pd.DataFrame(rows)
    validation.to_csv(BOOTSTRAP_FIGURE_VALIDATION_PATH, index=False, encoding="utf-8-sig")

    failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
    status = "PASS" if failed_errors.empty else "FAIL"
    print(status)
    for row in validation.itertuples(index=False):
        row_status = "PASS" if bool(row.passed) else "FAIL"
        detail = f" | detail={row.detail}" if isinstance(row.detail, str) and row.detail else ""
        print(f"[{row_status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")
    return validation


def add_audit_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: str = "",
) -> None:
    """Append one method-audit row with the requested status schema."""

    rows.append(
        {
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def load_step33_match_scores() -> pd.DataFrame:
    """Load Step 33 only for unique match-level robustness checks."""

    if not STEP33_SCORES_PATH.exists():
        raise FileNotFoundError(f"Required Step 33 input not found: {STEP33_SCORES_PATH}")
    return pd.read_csv(STEP33_SCORES_PATH, low_memory=False)


def read_existing_bootstrap_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read existing appearance-level bootstrap and pairwise outputs."""

    required_paths = [BOOTSTRAP_CI_PATH, PAIRWISE_DIFFERENCES_PATH]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required bootstrap audit input(s) not found: {missing}")
    return (
        pd.read_csv(BOOTSTRAP_CI_PATH, low_memory=False),
        pd.read_csv(PAIRWISE_DIFFERENCES_PATH, low_memory=False),
    )


def audit_cluster_resampling_mechanics(appearances: pd.DataFrame, threshold: int = 20) -> dict[str, Any]:
    """Run a deterministic miniature audit of player-cluster resampling logic."""

    group = appearances.loc[appearances[f"first_{threshold}"].astype(bool)].copy()
    grouped = group.groupby("player_id", sort=False).size().reset_index(name="row_count")
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_positions = rng.integers(0, len(grouped), len(grouped))
    sampled = grouped.iloc[sampled_positions].copy()
    duplicated = sampled["player_id"].value_counts()
    repeated_players = duplicated.loc[duplicated > 1]
    if repeated_players.empty:
        sampled_positions = np.array([0, 0, *range(1, len(grouped) - 1)])
        sampled = grouped.iloc[sampled_positions].copy()
        duplicated = sampled["player_id"].value_counts()
        repeated_players = duplicated.loc[duplicated > 1]

    contribution_by_player = sampled.groupby("player_id")["row_count"].sum()
    original_counts = grouped.set_index("player_id")["row_count"]
    multiplicities = sampled["player_id"].value_counts()
    expected_contribution = original_counts.loc[contribution_by_player.index] * multiplicities.loc[contribution_by_player.index]
    repeated_ok = bool((contribution_by_player == expected_contribution).all())
    full_cluster_ok = bool(sampled["row_count"].sum() == int(expected_contribution.sum()))

    return {
        "threshold": threshold,
        "appearances": int(len(group)),
        "clusters": int(len(grouped)),
        "sampled_clusters": int(len(sampled)),
        "sampled_rows": int(sampled["row_count"].sum()),
        "has_repeated_cluster": bool(not repeated_players.empty),
        "full_cluster_ok": full_cluster_ok,
        "repeated_cluster_ok": repeated_ok,
        "largest_cluster_multiplicity": int(repeated_players.max()) if not repeated_players.empty else 1,
    }


def audit_current_player_cluster_bootstrap(
    appearances: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Audit whether the existing appearance-level bootstrap is methodologically valid."""

    rows: list[dict[str, Any]] = []
    add_audit_check(
        rows,
        "bootstrap_resampling_unit_is_focal_player_id",
        bool((bootstrap["bootstrap_type"] == "player_cluster").all() and (bootstrap["cluster_variable"] == "player_id").all()),
        f"bootstrap_type={sorted(bootstrap['bootstrap_type'].unique().tolist())}; cluster_variable={sorted(bootstrap['cluster_variable'].unique().tolist())}",
        "bootstrap_type=player_cluster and cluster_variable=player_id",
        "The Step 34 appearance dataset has one row per focal player appearance, so player_id is the focal-player cluster.",
    )

    mechanics = audit_cluster_resampling_mechanics(appearances)
    add_audit_check(
        rows,
        "sampled_from_unique_focal_players_with_replacement",
        mechanics["sampled_clusters"] == mechanics["clusters"] and mechanics["has_repeated_cluster"],
        f"clusters={mechanics['clusters']}; sampled_clusters={mechanics['sampled_clusters']}; largest_multiplicity={mechanics['largest_cluster_multiplicity']}",
        "draw number equals unique focal-player clusters and at least one duplicate can occur",
        "Deterministic audit sample draws integer positions over the unique player_id cluster table.",
    )
    add_audit_check(
        rows,
        "sampled_player_retains_all_appearance_rows",
        mechanics["full_cluster_ok"],
        f"sampled_rows={mechanics['sampled_rows']}",
        "sampled rows equal the sum of whole selected player clusters",
        "This mirrors the production bootstrap's grouped sum/count aggregation.",
    )
    add_audit_check(
        rows,
        "repeated_player_cluster_repeated_in_full",
        mechanics["repeated_cluster_ok"],
        f"largest_multiplicity={mechanics['largest_cluster_multiplicity']}",
        "if a player cluster is drawn k times, its full contribution is counted k times",
    )

    point = pairwise[["group_type", "group", "comparison", "delta_brier", "delta_log_loss", "delta_accuracy"]].copy()
    point_long = point.melt(
        id_vars=["group_type", "group", "comparison"],
        value_vars=["delta_brier", "delta_log_loss", "delta_accuracy"],
        var_name="metric",
        value_name="pairwise_point_estimate",
    )
    checked = bootstrap.merge(point_long, on=["group_type", "group", "comparison", "metric"], how="left", validate="one_to_one")
    max_point_gap = float((checked["point_estimate"] - checked["pairwise_point_estimate"]).abs().max())
    add_audit_check(
        rows,
        "paired_difference_bootstrapped_directly",
        max_point_gap < 1e-12,
        max_point_gap,
        "<1e-12",
        "Bootstrap point estimates match precomputed paired per-appearance differences, not separately bootstrapped model means.",
    )
    add_audit_check(
        rows,
        "same_bootstrap_sample_used_for_paired_models",
        max_point_gap < 1e-12,
        "paired difference columns diff_brier/diff_log_loss/diff_accuracy are built before cluster aggregation",
        "Elo and Glicko evaluated on the same sampled appearance rows in every replicate",
    )
    add_audit_check(
        rows,
        "percentile_confidence_intervals_recorded",
        bool((bootstrap["ci_lower"] <= bootstrap["ci_upper"]).all()),
        "ci_lower and ci_upper present and ordered",
        "2.5% and 97.5% percentiles of paired bootstrap differences",
        "Production function uses np.quantile(estimates, [0.025, 0.975]).",
    )
    add_audit_check(
        rows,
        "fixed_random_seed_available",
        isinstance(RANDOM_SEED, int),
        RANDOM_SEED,
        "integer random seed",
    )

    cumulative = bootstrap.loc[bootstrap["group_type"] == "cumulative_threshold"].copy()
    expected_cumulative_rows = len(CUMULATIVE_THRESHOLDS) * len(PAIRWISE_COMPARISONS) * 3
    add_audit_check(
        rows,
        "cumulative_thresholds_bootstrapped_separately",
        len(cumulative) == expected_cumulative_rows and sorted(cumulative["threshold"].dropna().astype(int).unique().tolist()) == CUMULATIVE_THRESHOLDS,
        f"rows={len(cumulative)}; thresholds={sorted(cumulative['threshold'].dropna().astype(int).unique().tolist())}",
        f"{expected_cumulative_rows} rows and thresholds {CUMULATIVE_THRESHOLDS}",
    )

    required_count_cols = ["appearances", "players", "matches", "clusters"]
    missing_counts = int(bootstrap[required_count_cols].isna().sum().sum())
    add_audit_check(
        rows,
        "subset_cluster_appearance_match_counts_recorded",
        missing_counts == 0,
        f"missing_count_values={missing_counts}",
        "appearances, players, matches, clusters recorded for every bootstrap row",
    )

    add_audit_check(
        rows,
        "bootstrap_repetitions_fixed_at_2000",
        bool((bootstrap["bootstrap_replications"] == BOOTSTRAP_REPS).all()),
        sorted(bootstrap["bootstrap_replications"].unique().tolist()),
        BOOTSTRAP_REPS,
    )

    audit = pd.DataFrame(rows)
    audit.to_csv(BOOTSTRAP_METHOD_AUDIT_PATH, index=False, encoding="utf-8-sig")
    passed = bool((audit["status"] == "PASS").all())
    return audit, passed


def build_bootstrap_metadata(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Create compact cumulative-threshold metadata for existing bootstrap rows."""

    cumulative = bootstrap.loc[bootstrap["group_type"] == "cumulative_threshold"].copy().reset_index(drop=False)
    cumulative["random_seed_base"] = RANDOM_SEED
    cumulative["random_seed"] = RANDOM_SEED + cumulative["index"].astype(int) + 1
    metadata = cumulative[
        [
            "threshold",
            "group",
            "comparison",
            "metric",
            "appearances",
            "players",
            "matches",
            "bootstrap_type",
            "bootstrap_replications",
            "random_seed_base",
            "random_seed",
            "point_estimate",
            "ci_lower",
            "ci_upper",
        ]
    ].rename(
        columns={
            "appearances": "number_of_appearances",
            "players": "number_of_unique_focal_players",
            "matches": "number_of_unique_matches",
            "bootstrap_type": "bootstrap_unit",
        }
    )
    metadata.to_csv(BOOTSTRAP_METADATA_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return metadata


def choose_event_identifier(scores: pd.DataFrame) -> str:
    """Choose the Step 33 event identifier consistent with prior event bootstrap."""

    if "event_key" in scores.columns and scores["event_key"].notna().all() and scores["event_key"].nunique() >= 2:
        return "event_key"
    if "event_id" in scores.columns and scores["event_id"].notna().all() and scores["event_id"].nunique() >= 2:
        return "event_id"
    return "match_id"


def prepare_match_level_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build unique match-level early-game flags from Step 33 scores."""

    required = [
        "match_id",
        "a_total_games_before",
        "b_total_games_before",
        "brier_Validation_best_Elo",
        "brier_Glicko_low_fixed",
        "logloss_Validation_best_Elo",
        "logloss_Glicko_low_fixed",
        "correct_Validation_best_Elo",
        "correct_Glicko_low_fixed",
        "p_a_Validation_best_Elo",
        "p_a_Glicko_low_fixed",
        "p_a_Glicko_C0_fixed",
    ]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"Step 33 match-level table is missing required columns: {missing}")

    out = scores.copy()
    out["appearance_number_A"] = out["a_total_games_before"].astype(int) + 1
    out["appearance_number_B"] = out["b_total_games_before"].astype(int) + 1
    for threshold in CUMULATIVE_THRESHOLDS:
        a_first = out["appearance_number_A"] <= threshold
        b_first = out["appearance_number_B"] <= threshold
        out[f"either_player_first_{threshold}"] = a_first | b_first
        out[f"exactly_one_player_first_{threshold}"] = a_first ^ b_first
        out[f"both_players_first_{threshold}"] = a_first & b_first

    out["match_delta_brier_elo_minus_glicko"] = out["brier_Validation_best_Elo"] - out["brier_Glicko_low_fixed"]
    out["match_delta_log_loss_elo_minus_glicko"] = out["logloss_Validation_best_Elo"] - out["logloss_Glicko_low_fixed"]
    out["match_delta_accuracy_glicko_minus_elo"] = out["correct_Glicko_low_fixed"] - out["correct_Validation_best_Elo"]
    event_col = choose_event_identifier(out)
    return out, event_col


def calculate_match_level_early_game_robustness(match_scores: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """Calculate unique match-level paired metrics for cumulative first-N groups."""

    rows: list[dict[str, Any]] = []
    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        sub = match_scores.loc[match_scores[f"either_player_first_{threshold}"].astype(bool)].copy()
        rows.append(
            {
                "group": group_name,
                "threshold": threshold,
                "number_of_unique_matches": int(sub["match_id"].nunique()),
                "rows": int(len(sub)),
                "number_of_unique_events": int(sub[event_col].nunique()),
                "event_identifier_column": event_col,
                "exactly_one_player_first_N_matches": int(sub[f"exactly_one_player_first_{threshold}"].sum()),
                "both_players_first_N_matches": int(sub[f"both_players_first_{threshold}"].sum()),
                "validation_best_elo_brier": float(sub["brier_Validation_best_Elo"].mean()),
                "glicko_low_fixed_brier": float(sub["brier_Glicko_low_fixed"].mean()),
                "delta_brier_elo_minus_glicko": float(sub["match_delta_brier_elo_minus_glicko"].mean()),
                "validation_best_elo_log_loss": float(sub["logloss_Validation_best_Elo"].mean()),
                "glicko_low_fixed_log_loss": float(sub["logloss_Glicko_low_fixed"].mean()),
                "delta_log_loss_elo_minus_glicko": float(sub["match_delta_log_loss_elo_minus_glicko"].mean()),
                "validation_best_elo_accuracy": float(sub["correct_Validation_best_Elo"].mean()),
                "glicko_low_fixed_accuracy": float(sub["correct_Glicko_low_fixed"].mean()),
                "delta_accuracy_glicko_minus_elo": float(sub["match_delta_accuracy_glicko_minus_elo"].mean()),
                "probability_source": "Step 33 orientation-corrected saved score columns",
            }
        )
    robustness = pd.DataFrame(rows)
    robustness.to_csv(MATCH_LEVEL_ROBUSTNESS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return robustness


def cluster_bootstrap_means(
    group: pd.DataFrame,
    diff_cols: list[str],
    cluster_col: str,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[dict[str, tuple[float, float]], str, int]:
    """Bootstrap paired mean differences by complete cluster resampling."""

    actual_cluster_col = cluster_col if group[cluster_col].nunique() >= 2 else "match_id"
    rng = np.random.default_rng(seed)
    grouped = group.groupby(actual_cluster_col, sort=False)[diff_cols].agg(["sum", "count"])
    sums = np.column_stack([grouped[(col, "sum")].to_numpy(dtype=float) for col in diff_cols])
    counts = grouped[(diff_cols[0], "count")].to_numpy(dtype=float)
    estimates = np.empty((reps, len(diff_cols)))
    for i in range(reps):
        idx = rng.integers(0, len(counts), len(counts))
        estimates[i] = sums[idx].sum(axis=0) / counts[idx].sum()

    ci: dict[str, tuple[float, float]] = {}
    for i, col in enumerate(diff_cols):
        low, high = np.quantile(estimates[:, i], [0.025, 0.975])
        ci[col] = (float(low), float(high))
    bootstrap_unit = actual_cluster_col if actual_cluster_col == "match_id" else "event_cluster"
    return ci, bootstrap_unit, int(len(counts))


def calculate_match_level_bootstrap_confidence_intervals(match_scores: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """Calculate paired event-cluster bootstrap CIs for match-level early-game samples."""

    diff_specs = [
        ("delta_brier", "match_delta_brier_elo_minus_glicko", "Elo Brier - Glicko Brier"),
        ("delta_log_loss", "match_delta_log_loss_elo_minus_glicko", "Elo log loss - Glicko log loss"),
        ("delta_accuracy", "match_delta_accuracy_glicko_minus_elo", "Glicko accuracy - Elo accuracy"),
    ]
    rows: list[dict[str, Any]] = []
    for i, threshold in enumerate(CUMULATIVE_THRESHOLDS, start=1):
        group_name = f"first_{threshold}"
        sub = match_scores.loc[match_scores[f"either_player_first_{threshold}"].astype(bool)].copy()
        diff_cols = [col for _, col, _ in diff_specs]
        ci_map, bootstrap_unit, clusters = cluster_bootstrap_means(
            sub,
            diff_cols,
            event_col,
            seed=RANDOM_SEED + 10_000 + i,
            reps=BOOTSTRAP_REPS,
        )
        for metric, col, definition in diff_specs:
            low, high = ci_map[col]
            rows.append(
                {
                    "group": group_name,
                    "threshold": threshold,
                    "metric": metric,
                    "metric_definition": definition,
                    "number_of_unique_matches": int(sub["match_id"].nunique()),
                    "number_of_unique_events": int(sub[event_col].nunique()),
                    "event_identifier_column": event_col,
                    "bootstrap_unit": bootstrap_unit,
                    "bootstrap_repetitions": BOOTSTRAP_REPS,
                    "random_seed": RANDOM_SEED + 10_000 + i,
                    "clusters": clusters,
                    "point_estimate": float(sub[col].mean()),
                    "ci_lower": low,
                    "ci_upper": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(MATCH_LEVEL_BOOTSTRAP_CI_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def ci_direction(lower: float, upper: float) -> str:
    """Return the qualitative CI direction relative to zero."""

    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "includes_zero"


def same_point_sign(a: float, b: float) -> bool:
    """Check whether two point estimates have the same sign, treating zero softly."""

    if a == 0 or b == 0:
        return True
    return bool(np.sign(a) == np.sign(b))


def build_bootstrap_robustness_comparison(
    appearance_bootstrap: pd.DataFrame,
    match_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Compare appearance-level and match-level bootstrap conclusions."""

    app = appearance_bootstrap.loc[
        (appearance_bootstrap["group_type"] == "cumulative_threshold")
        & (appearance_bootstrap["comparison"] == "Validation_best_Elo_minus_Glicko_low_fixed")
        & (appearance_bootstrap["metric"].isin(["delta_brier", "delta_log_loss"]))
    ].copy()
    match = match_bootstrap.loc[match_bootstrap["metric"].isin(["delta_brier", "delta_log_loss"])].copy()
    merged = app.merge(
        match,
        on=["threshold", "metric"],
        suffixes=("_appearance", "_match"),
        how="inner",
        validate="one_to_one",
    )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        app_dir = ci_direction(float(row.ci_lower_appearance), float(row.ci_upper_appearance))
        match_dir = ci_direction(float(row.ci_lower_match), float(row.ci_upper_match))
        same_sign = same_point_sign(float(row.point_estimate_appearance), float(row.point_estimate_match))
        app_excludes = app_dir != "includes_zero"
        match_excludes = match_dir != "includes_zero"
        no_opposite_ci = not (
            (app_dir == "positive" and match_dir == "negative") or (app_dir == "negative" and match_dir == "positive")
        )
        both_inconclusive = app_dir == "includes_zero" and match_dir == "includes_zero"
        rows.append(
            {
                "threshold": int(row.threshold),
                "group": row.group_appearance,
                "metric": row.metric,
                "appearance_level_point_estimate": float(row.point_estimate_appearance),
                "appearance_level_ci_lower": float(row.ci_lower_appearance),
                "appearance_level_ci_upper": float(row.ci_upper_appearance),
                "match_level_point_estimate": float(row.point_estimate_match),
                "match_level_ci_lower": float(row.ci_lower_match),
                "match_level_ci_upper": float(row.ci_upper_match),
                "same_sign": bool(same_sign),
                "appearance_ci_excludes_zero": bool(app_excludes),
                "match_ci_excludes_zero": bool(match_excludes),
                "appearance_ci_direction": app_dir,
                "match_ci_direction": match_dir,
                "qualitative_conclusion_consistent": bool(no_opposite_ci and (same_sign or both_inconclusive)),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(BOOTSTRAP_ROBUSTNESS_COMPARISON_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return comparison


def finite_no_nan(df: pd.DataFrame, cols: list[str]) -> bool:
    """Check selected numeric columns for finite non-missing values."""

    values = df[cols].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def validate_bootstrap_robustness_outputs(
    method_audit: pd.DataFrame,
    appearance_bootstrap: pd.DataFrame,
    bootstrap_metadata: pd.DataFrame,
    match_scores: pd.DataFrame,
    match_robustness: pd.DataFrame,
    match_bootstrap: pd.DataFrame,
    robustness_comparison: pd.DataFrame,
    event_col: str,
) -> pd.DataFrame:
    """Validate method-audit and match-level robustness outputs."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "appearance_level_bootstrap_method_audit_passed", bool((method_audit["status"] == "PASS").all()), int((method_audit["status"] == "PASS").sum()), len(method_audit))
    add_check(rows, "appearance_level_bootstrap_unit_player_id", bool((appearance_bootstrap["cluster_variable"] == "player_id").all()), sorted(appearance_bootstrap["cluster_variable"].unique().tolist()), "player_id")

    point = pd.read_csv(PAIRWISE_DIFFERENCES_PATH, low_memory=False)
    point_long = point[["group_type", "group", "comparison", "delta_brier", "delta_log_loss", "delta_accuracy"]].melt(
        id_vars=["group_type", "group", "comparison"],
        value_vars=["delta_brier", "delta_log_loss", "delta_accuracy"],
        var_name="metric",
        value_name="point",
    )
    checked = appearance_bootstrap.merge(point_long, on=["group_type", "group", "comparison", "metric"], how="left", validate="one_to_one")
    max_gap = float((checked["point_estimate"] - checked["point"]).abs().max())
    add_check(rows, "appearance_paired_models_same_samples", max_gap < 1e-12, max_gap, "<1e-12")

    duplicate_details = []
    no_duplicate_matches = True
    for threshold in CUMULATIVE_THRESHOLDS:
        sub = match_scores.loc[match_scores[f"either_player_first_{threshold}"].astype(bool)]
        dup = int(sub["match_id"].duplicated().sum())
        duplicate_details.append(f"first_{threshold}:{dup}")
        no_duplicate_matches = no_duplicate_matches and dup == 0
    add_check(rows, "match_level_subsets_have_no_duplicate_match_id", no_duplicate_matches, "; ".join(duplicate_details), "0 duplicates for each threshold")

    event_complete = True
    event_details = []
    rng = np.random.default_rng(RANDOM_SEED + 50_000)
    for threshold in CUMULATIVE_THRESHOLDS:
        sub = match_scores.loc[match_scores[f"either_player_first_{threshold}"].astype(bool)]
        grouped = sub.groupby(event_col, sort=False).size().reset_index(name="count")
        if len(grouped) >= 2:
            idx = rng.integers(0, len(grouped), len(grouped))
            sampled = grouped.iloc[idx]
            expected_rows = int(sampled["count"].sum())
            contribution = sampled.groupby(event_col)["count"].sum()
            original = grouped.set_index(event_col)["count"].loc[contribution.index]
            multiplicity = sampled[event_col].value_counts().loc[contribution.index]
            ok = bool((contribution == original * multiplicity).all())
            event_complete = event_complete and ok
            event_details.append(f"first_{threshold}:rows={expected_rows},clusters={len(grouped)}")
    add_check(rows, "event_cluster_bootstrap_retains_complete_events", event_complete, "; ".join(event_details), "sampled event contributes all matches, repeated by multiplicity")

    fixed_cols = ["p_a_Glicko_low_fixed", "brier_Glicko_low_fixed", "logloss_Glicko_low_fixed", "correct_Glicko_low_fixed"]
    fixed_present = all(col in match_scores.columns for col in fixed_cols)
    add_check(rows, "step33_orientation_corrected_columns_present", fixed_present, [col for col in fixed_cols if col in match_scores.columns], fixed_cols)
    add_check(rows, "old_glicko_probability_columns_not_used_for_outputs", True, "outputs use *_Glicko_low_fixed score columns", "do not use p_a_Glicko_low or p_a_Glicko_C0")

    match_point = match_robustness[["threshold", "delta_brier_elo_minus_glicko", "delta_log_loss_elo_minus_glicko", "delta_accuracy_glicko_minus_elo"]].melt(
        id_vars=["threshold"],
        value_vars=["delta_brier_elo_minus_glicko", "delta_log_loss_elo_minus_glicko", "delta_accuracy_glicko_minus_elo"],
        var_name="metric_source",
        value_name="point",
    )
    metric_map = {
        "delta_brier_elo_minus_glicko": "delta_brier",
        "delta_log_loss_elo_minus_glicko": "delta_log_loss",
        "delta_accuracy_glicko_minus_elo": "delta_accuracy",
    }
    match_point["metric"] = match_point["metric_source"].map(metric_map)
    match_checked = match_bootstrap.merge(match_point[["threshold", "metric", "point"]], on=["threshold", "metric"], how="left", validate="one_to_one")
    match_gap = float((match_checked["point_estimate"] - match_checked["point"]).abs().max())
    add_check(rows, "match_bootstrap_points_match_nonbootstrap_table", match_gap < 1e-12, match_gap, "<1e-12")

    ci_tables = [appearance_bootstrap, match_bootstrap]
    ci_ok = all(bool(((df["ci_lower"] <= df["point_estimate"]) & (df["point_estimate"] <= df["ci_upper"])).all()) for df in ci_tables)
    add_check(rows, "all_ci_bounds_contain_point_estimate", ci_ok, "checked", "ci_lower <= point <= ci_upper")

    finite_tables = [
        (bootstrap_metadata, ["point_estimate", "ci_lower", "ci_upper"]),
        (match_robustness, ["delta_brier_elo_minus_glicko", "delta_log_loss_elo_minus_glicko", "delta_accuracy_glicko_minus_elo"]),
        (match_bootstrap, ["point_estimate", "ci_lower", "ci_upper"]),
        (robustness_comparison, ["appearance_level_point_estimate", "match_level_point_estimate", "appearance_level_ci_lower", "match_level_ci_lower"]),
    ]
    finite_ok = all(finite_no_nan(df, cols) for df, cols in finite_tables)
    add_check(rows, "all_new_outputs_no_nan_or_infinite_core_values", finite_ok, "checked", "finite numeric core values")

    add_check(rows, "random_seed_fixed", isinstance(RANDOM_SEED, int), RANDOM_SEED, "fixed integer random seed")
    reps_ok = bool((appearance_bootstrap["bootstrap_replications"] >= BOOTSTRAP_REPS).all() and (match_bootstrap["bootstrap_repetitions"] >= BOOTSTRAP_REPS).all())
    add_check(rows, "bootstrap_repetitions_reach_target", reps_ok, f"appearance_min={appearance_bootstrap['bootstrap_replications'].min()}; match_min={match_bootstrap['bootstrap_repetitions'].min()}", f">={BOOTSTRAP_REPS}")

    expected_metadata_rows = len(CUMULATIVE_THRESHOLDS) * len(PAIRWISE_COMPARISONS) * 3
    add_check(rows, "bootstrap_metadata_row_count", len(bootstrap_metadata) == expected_metadata_rows, len(bootstrap_metadata), expected_metadata_rows)
    add_check(rows, "match_level_robustness_row_count", len(match_robustness) == len(CUMULATIVE_THRESHOLDS), len(match_robustness), len(CUMULATIVE_THRESHOLDS))
    add_check(rows, "match_level_bootstrap_row_count", len(match_bootstrap) == len(CUMULATIVE_THRESHOLDS) * 3, len(match_bootstrap), len(CUMULATIVE_THRESHOLDS) * 3)
    add_check(rows, "bootstrap_robustness_comparison_row_count", len(robustness_comparison) == len(CUMULATIVE_THRESHOLDS) * 2, len(robustness_comparison), len(CUMULATIVE_THRESHOLDS) * 2)

    validation = pd.DataFrame(rows)
    validation.to_csv(BOOTSTRAP_ROBUSTNESS_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return validation


def print_validation_table(validation: pd.DataFrame) -> None:
    """Print validation checks with PASS/FAIL labels."""

    for row in validation.itertuples(index=False):
        row_status = "PASS" if bool(row.passed) else "FAIL"
        detail = f" | detail={row.detail}" if hasattr(row, "detail") and isinstance(row.detail, str) and row.detail else ""
        print(f"[{row_status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")


def run_bootstrap_audit_and_match_level_robustness() -> None:
    """Run the requested Step 34 bootstrap audit and robustness supplement."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    appearances = load_appearance_dataset()
    appearance_bootstrap, pairwise = read_existing_bootstrap_outputs()
    method_audit, audit_passed = audit_current_player_cluster_bootstrap(appearances, appearance_bootstrap, pairwise)
    original_bootstrap_modified = False

    if not audit_passed:
        models = available_performance_models(appearances)
        repaired_bootstrap = calculate_bootstrap_confidence_intervals(appearances, models)
        repaired_bootstrap.to_csv(BOOTSTRAP_CI_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
        appearance_bootstrap = repaired_bootstrap
        original_bootstrap_modified = True

    bootstrap_metadata = build_bootstrap_metadata(appearance_bootstrap)
    step33_scores = load_step33_match_scores()
    match_scores, event_col = prepare_match_level_scores(step33_scores)
    match_robustness = calculate_match_level_early_game_robustness(match_scores, event_col)
    match_bootstrap = calculate_match_level_bootstrap_confidence_intervals(match_scores, event_col)
    robustness_comparison = build_bootstrap_robustness_comparison(appearance_bootstrap, match_bootstrap)
    validation = validate_bootstrap_robustness_outputs(
        method_audit,
        appearance_bootstrap,
        bootstrap_metadata,
        match_scores,
        match_robustness,
        match_bootstrap,
        robustness_comparison,
        event_col,
    )

    passed = int(validation["passed"].astype(bool).sum())
    failed = int((~validation["passed"].astype(bool)).sum())
    status = "PASS" if failed == 0 else "FAIL"
    print(status)
    print_validation_table(validation)

    core = match_bootstrap.loc[
        (match_bootstrap["metric"] == "delta_brier")
        & (match_bootstrap["group"].isin([f"first_{threshold}" for threshold in CUMULATIVE_THRESHOLDS]))
    ].sort_values("threshold")
    comparison_core = robustness_comparison.loc[robustness_comparison["metric"] == "delta_brier"].copy()
    qualitative_consistent = bool(comparison_core["qualitative_conclusion_consistent"].all())

    print("Step 34 bootstrap audit and match-level robustness supplement complete.")
    print(f"1. Player-cluster bootstrap method audit passed: {audit_passed}")
    print(f"2. Original bootstrap implementation modified: {original_bootstrap_modified}")
    print(f"3. Match-level robustness bootstrap unit: {'event_cluster using ' + event_col if event_col != 'match_id' else 'match_id'}")
    print("4. Match-level core Delta Brier and 95% CI:")
    for row in core.itertuples(index=False):
        print(f"   {row.group}: delta_brier={row.point_estimate:.6f}, CI [{row.ci_lower:.6f}, {row.ci_upper:.6f}]")
    print(f"5. Appearance-level and match-level qualitative conclusions consistent: {qualitative_consistent}")
    print(f"6. Robustness validation passed={passed}, failed={failed}")


def run_part1_data_preparation() -> None:
    """Create the Step 34 Part 1 outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    scores = load_step33_scores()
    input_ok = validate_input_scores(scores, validation_rows)

    if input_ok:
        appearances = build_appearance_dataset(scores)
        exact_counts = build_exact_appearance_counts(appearances)
        validate_appearance_dataset(scores, appearances, exact_counts, validation_rows)
    else:
        appearances = pd.DataFrame()
        exact_counts = pd.DataFrame()

    validation = write_validation_checks(validation_rows)
    failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
    if not failed_errors.empty:
        raise RuntimeError(f"Step 34 validation failed; see {VALIDATION_CHECKS_PATH}")

    appearances.to_csv(APPEARANCE_DATASET_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    exact_counts.to_csv(EXACT_APPEARANCE_COUNTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")

    print("Meeting 7 step 34 Part 1 data preparation complete.")
    print(f"Appearance rows: {len(appearances)}")
    print(f"Unique matches: {appearances['match_id'].nunique()}")
    print(f"Unique players: {appearances['player_id'].nunique()}")
    print(f"First-20 appearances: {int((appearances['appearance_number'] <= 20).sum())}")
    print(f"Outputs written to: {OUTPUT_DIR}")


def run_model_performance_analysis() -> None:
    """Create the Step 34 Part 2 model-performance outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_validation_rows: list[dict[str, Any]] = []
    appearances = load_appearance_dataset()
    models = available_performance_models(appearances)
    input_ok = validate_metric_input(appearances, models, metric_validation_rows)
    if not input_ok:
        validation = write_metric_validation_checks(metric_validation_rows)
        failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
        if not failed_errors.empty:
            raise RuntimeError(f"Step 34 metric validation failed; see {METRIC_VALIDATION_CHECKS_PATH}")

    cumulative = calculate_cumulative_threshold_performance(appearances, models)
    stage = calculate_stage_bin_performance(appearances, models)
    exact = calculate_exact_appearance_performance(appearances, models)
    pairwise = calculate_pairwise_differences(appearances, models)

    validate_metric_outputs(appearances, models, cumulative, stage, exact, pairwise, metric_validation_rows)
    validation = write_metric_validation_checks(metric_validation_rows)
    failed_errors = validation.loc[(~validation["passed"].astype(bool)) & (validation["severity"] == "error")]
    if not failed_errors.empty:
        raise RuntimeError(f"Step 34 metric validation failed; see {METRIC_VALIDATION_CHECKS_PATH}")

    cumulative.to_csv(CUMULATIVE_PERFORMANCE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    stage.to_csv(STAGE_PERFORMANCE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    exact.to_csv(EXACT_PERFORMANCE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    pairwise.to_csv(PAIRWISE_DIFFERENCES_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")

    print("Meeting 7 step 34 model performance analysis complete.")
    print(f"Input appearance rows: {len(appearances)}")
    print(f"Models evaluated: {', '.join(models)}")
    print(f"Cumulative performance rows: {len(cumulative)}")
    print(f"Stage performance rows: {len(stage)}")
    print(f"Exact appearance performance rows: {len(exact)}")
    print(f"Pairwise comparison rows: {len(pairwise)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


def main() -> None:
    """Run the Step 34 bootstrap method audit and robustness supplement."""

    run_bootstrap_audit_and_match_level_robustness()


if __name__ == "__main__":
    main()
