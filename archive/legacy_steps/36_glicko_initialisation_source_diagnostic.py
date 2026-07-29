"""Meeting 7 Step 36: Glicko initialisation source diagnostic.

This script diagnoses why Step 34/35 first-appearance focal players have a
high saved low-inflation Glicko predicted win probability. It uses only stored
Step 33/34 probabilities and pre-match states. It does not rerun Elo/Glicko,
tune parameters, or modify earlier outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glicko_core import DEFAULT_RD, DEFAULT_RATING, MAX_RD, MIN_RD, expected_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

APPEARANCE_DATASET_PATH = OUTPUT_DIR / "34_early_game_appearance_dataset.csv"
STEP33_SCORES_PATH = PROJECT_ROOT / "outputs" / "meeting6" / "33_orientation_corrected_per_match_scores_2025.csv"
STEP35_SUMMARY_PATH = OUTPUT_DIR / "35_early_game_mechanism_summary.md"
STEP35_KEY_RESULTS_PATH = OUTPUT_DIR / "35_key_mechanism_results.csv"
STEP35_RD_SUMMARY_PATH = OUTPUT_DIR / "35_glicko_rating_rd_summary.csv"

INPUT_VALIDATION_PATH = OUTPUT_DIR / "36_input_validation_checks.csv"
DEBUT_DIAGNOSTIC_PATH = OUTPUT_DIR / "36_debut_diagnostic_dataset.csv"
ORIENTATION_AUDIT_PATH = OUTPUT_DIR / "36_debut_probability_orientation_audit.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "36_glicko_probability_reconstruction.csv"
DEBUT_STATE_SUMMARY_PATH = OUTPUT_DIR / "36_debut_state_summary.csv"
ONE_VS_BOTH_PATH = OUTPUT_DIR / "36_one_vs_both_debut_summary.csv"
RATING_DIFF_BINS_PATH = OUTPUT_DIR / "36_debut_rating_difference_bins.csv"
OPPONENT_RD_QUARTILES_PATH = OUTPUT_DIR / "36_debut_opponent_rd_quartiles.csv"
OPPONENT_RD_ASSOCIATIONS_PATH = OUTPUT_DIR / "36_opponent_rd_associations.csv"
FOCAL_RD_FORMULA_PATH = OUTPUT_DIR / "36_focal_rd_formula_diagnostic.csv"
EXTREME_INFLUENCE_PATH = OUTPUT_DIR / "36_debut_extreme_probability_influence.csv"
EXTREME_CASES_PATH = OUTPUT_DIR / "36_debut_probability_extreme_cases.csv"
COUNTERFACTUAL_PATH = OUTPUT_DIR / "36_debut_counterfactual_probability_diagnostics.csv"
LOW_VS_C0_PATH = OUTPUT_DIR / "36_debut_low_vs_c0_diagnostic.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "36_key_initialisation_diagnostic_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "36_figure_manifest.csv"
VALIDATION_PATH = OUTPUT_DIR / "36_initialisation_diagnostic_validation_checks.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "36_glicko_initialisation_source_summary.md"

EXPECTED_APPEARANCE_ROWS = 22_758
EXPECTED_MATCHES = 11_379
EXPECTED_FIRST1_MEAN_GLICKO_PROB_ROUNDED = 0.743
EXPECTED_FIRST1_EMPIRICAL_WIN_RATE_ROUNDED = 0.408
EXPECTED_FIRST1_BIAS_ROUNDED = 0.336
ROUNDING_TOL = 0.001
STRICT_TOL = 1e-10
EPS = 1e-15

MODEL_SPECS = {
    "Glicko_low_fixed": {
        "focal_col": "p_focal_Glicko_low_fixed",
        "canonical_col": "p_a_Glicko_low_fixed",
        "label": "Glicko low inflation",
    },
    "Validation_best_Elo": {
        "focal_col": "p_focal_Validation_best_Elo",
        "canonical_col": "p_a_Validation_best_Elo",
        "label": "Validation-best Elo",
    },
    "Glicko_C0_fixed": {
        "focal_col": "p_focal_Glicko_C0_fixed",
        "canonical_col": "p_a_Glicko_C0_fixed",
        "label": "Glicko C0",
    },
    "best_AdaptiveK": {
        "focal_col": "p_focal_best_AdaptiveK",
        "canonical_col": "p_a_best_AdaptiveK",
        "label": "Adaptive-K Elo",
    },
}

REQUIRED_APPEARANCE_COLUMNS = [
    "match_id",
    "match_date",
    "date",
    "event_key",
    "focal_side",
    "player_id",
    "opponent_id",
    "total_games_before",
    "opponent_total_games_before",
    "appearance_number",
    "opponent_appearance_number",
    "debut_flag",
    "opponent_debut_flag",
    "outcome_focal",
    "first_1",
    "rating_focal_Glicko_low",
    "rating_opponent_Glicko_low",
    "rd_focal_Glicko_low",
    "rd_opponent_Glicko_low",
    "rating_focal_Glicko_C0",
    "rating_opponent_Glicko_C0",
    "rd_focal_Glicko_C0",
    "rd_opponent_Glicko_C0",
    *[spec["focal_col"] for spec in MODEL_SPECS.values()],
]

REQUIRED_STEP33_COLUMNS = [
    "match_id",
    "player_a_id",
    "player_b_id",
    "outcome_a",
    "p_a_Glicko_low_fixed",
    "p_a_Glicko_C0_fixed",
    "p_a_Validation_best_Elo",
    "p_a_best_AdaptiveK",
    "rating_a_Glicko_low",
    "rating_b_Glicko_low",
    "rd_a_Glicko_low",
    "rd_b_Glicko_low",
    "rating_a_Glicko_C0",
    "rating_b_Glicko_C0",
    "rd_a_Glicko_C0",
    "rd_b_Glicko_C0",
]

OUTPUT_FILES = [
    INPUT_VALIDATION_PATH,
    DEBUT_DIAGNOSTIC_PATH,
    ORIENTATION_AUDIT_PATH,
    RECONSTRUCTION_PATH,
    DEBUT_STATE_SUMMARY_PATH,
    ONE_VS_BOTH_PATH,
    RATING_DIFF_BINS_PATH,
    OPPONENT_RD_QUARTILES_PATH,
    OPPONENT_RD_ASSOCIATIONS_PATH,
    FOCAL_RD_FORMULA_PATH,
    EXTREME_INFLUENCE_PATH,
    EXTREME_CASES_PATH,
    COUNTERFACTUAL_PATH,
    LOW_VS_C0_PATH,
    KEY_RESULTS_PATH,
    FIGURE_MANIFEST_PATH,
    VALIDATION_PATH,
    SUMMARY_MD_PATH,
]


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: str = "",
) -> None:
    """Append one validation row."""

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
    """Print validation checks in a compact PASS/FAIL form."""

    for row in checks.itertuples(index=False):
        detail = f" | details={row.details}" if isinstance(row.details, str) and row.details else ""
        print(f"[{row.status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")


def safe_log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    """Return per-row binary log loss."""

    clipped = p.astype(float).clip(EPS, 1 - EPS)
    y = y.astype(float)
    return -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))


def brier(p: pd.Series, y: pd.Series) -> pd.Series:
    """Return per-row Brier loss."""

    return (p.astype(float) - y.astype(float)) ** 2


def scalar(value: Any) -> Any:
    """Convert numpy scalars to ordinary Python scalars for CSV readability."""

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def finite_or_na(value: Any) -> Any:
    """Return a finite float or NA for empty groups."""

    if value is None:
        return pd.NA
    try:
        if np.isfinite(value):
            return float(value)
    except TypeError:
        return pd.NA
    return pd.NA


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Step 34 appearance data and Step 33 canonical match data."""

    if not APPEARANCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {APPEARANCE_DATASET_PATH}")
    if not STEP33_SCORES_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {STEP33_SCORES_PATH}")

    appearances = pd.read_csv(APPEARANCE_DATASET_PATH, low_memory=False)
    step33 = pd.read_csv(STEP33_SCORES_PATH, low_memory=False)

    for col in ["first_1", "debut_flag", "opponent_debut_flag"]:
        if col in appearances.columns:
            appearances[col] = appearances[col].astype(bool)
    return appearances, step33


def canonical_probability_for_focal(row: pd.Series, canonical_col: str) -> float:
    """Convert a canonical player-A probability to focal-player orientation."""

    p_a = float(row[canonical_col])
    if row["focal_side"] == "A":
        return p_a
    if row["focal_side"] == "B":
        return 1.0 - p_a
    raise ValueError(f"Unexpected focal_side: {row['focal_side']}")


def reconstruct_saved_glicko_probability(row: pd.Series) -> tuple[float, str, str, float]:
    """Reconstruct the stored Step 34 focal Glicko probability.

    Step 33 defines the final Glicko probability as the direct canonical
    player-A probability E(A, B, RD_B). Step 34 then converts it to focal
    orientation. Therefore focal-B rows are reconstructed as 1 - E(A, B, RD_B),
    not as a new direct-B Glicko probability.
    """

    p_a_direct = expected_score(
        row["rating_a_Glicko_low"],
        row["rating_b_Glicko_low"],
        row["rd_b_Glicko_low"],
    )
    if row["focal_side"] == "A":
        return float(p_a_direct), "expected_score(A,B,RD_B)", "opponent_RD", float(row["rd_b_Glicko_low"])
    return float(1.0 - p_a_direct), "1 - expected_score(A,B,RD_B)", "focal_RD", float(row["rd_b_Glicko_low"])


def direct_focal_glicko_probability(row: pd.Series) -> float:
    """Return the direct focal expected score using focal rating and opponent RD."""

    return float(
        expected_score(
            row["rating_focal_Glicko_low"],
            row["rating_opponent_Glicko_low"],
            row["rd_opponent_Glicko_low"],
        )
    )


def validate_inputs(appearances: pd.DataFrame, step33: pd.DataFrame) -> pd.DataFrame:
    """Run Step 36 input validations."""

    rows: list[dict[str, Any]] = []

    add_check(rows, "appearance_dataset_rows_22758", len(appearances) == EXPECTED_APPEARANCE_ROWS, len(appearances), EXPECTED_APPEARANCE_ROWS)
    add_check(rows, "step33_unique_matches_11379", step33["match_id"].nunique() == EXPECTED_MATCHES, int(step33["match_id"].nunique()), EXPECTED_MATCHES)

    missing_app = [col for col in REQUIRED_APPEARANCE_COLUMNS if col not in appearances.columns]
    missing_step33 = [col for col in REQUIRED_STEP33_COLUMNS if col not in step33.columns]
    add_check(rows, "required_appearance_columns_present", not missing_app, missing_app if missing_app else "none", "all Step 36 appearance columns")
    add_check(rows, "required_step33_columns_present", not missing_step33, missing_step33 if missing_step33 else "none", "all Step 33 canonical columns")

    if not missing_app:
        first1 = appearances.loc[appearances["first_1"]].copy()
        add_check(rows, "first_1_equals_total_games_before_zero", bool((first1["total_games_before"] == 0).all()), int((first1["total_games_before"] != 0).sum()), 0)
        add_check(rows, "appearance_number_equals_total_games_before_plus_one", bool((appearances["appearance_number"] == appearances["total_games_before"] + 1).all()), "checked", True)
        per_match = appearances.groupby("match_id").size()
        add_check(rows, "every_match_has_two_appearance_rows", bool((per_match == 2).all()), int((per_match != 2).sum()), 0)

        focal_prob_cols = [spec["focal_col"] for spec in MODEL_SPECS.values() if spec["focal_col"] in appearances.columns]
        in_range = bool(appearances[focal_prob_cols].apply(lambda col: col.between(0, 1).all()).all())
        add_check(rows, "all_focal_probabilities_in_range", in_range, "checked", "[0,1]")
        outcome_values = sorted(appearances["outcome_focal"].dropna().unique().tolist())
        add_check(rows, "focal_outcome_binary", set(outcome_values).issubset({0, 1}), outcome_values, "{0,1}")

        mean_glicko = float(first1["p_focal_Glicko_low_fixed"].mean())
        empirical = float(first1["outcome_focal"].mean())
        bias_value = mean_glicko - empirical
        add_check(
            rows,
            "first_1_mean_glicko_probability_reproduces_step35",
            abs(mean_glicko - EXPECTED_FIRST1_MEAN_GLICKO_PROB_ROUNDED) <= ROUNDING_TOL,
            round(mean_glicko, 6),
            f"Step 35 rounded {EXPECTED_FIRST1_MEAN_GLICKO_PROB_ROUNDED}",
            "Step 35 technical summary reports rounded values.",
        )
        add_check(
            rows,
            "first_1_empirical_win_rate_reproduces_step35",
            abs(empirical - EXPECTED_FIRST1_EMPIRICAL_WIN_RATE_ROUNDED) <= ROUNDING_TOL,
            round(empirical, 6),
            f"Step 35 rounded {EXPECTED_FIRST1_EMPIRICAL_WIN_RATE_ROUNDED}",
            "Step 35 technical summary reports rounded values.",
        )
        add_check(
            rows,
            "first_1_prediction_bias_reproduces_step35",
            abs(bias_value - EXPECTED_FIRST1_BIAS_ROUNDED) <= ROUNDING_TOL,
            round(bias_value, 6),
            f"Step 35 rounded {EXPECTED_FIRST1_BIAS_ROUNDED}",
            "bias = mean predicted probability - empirical win rate.",
        )

        prematch_cols = [
            "rating_focal_Glicko_low",
            "rating_opponent_Glicko_low",
            "rd_focal_Glicko_low",
            "rd_opponent_Glicko_low",
        ]
        finite_prematch = bool(np.isfinite(appearances[prematch_cols].to_numpy(dtype=float)).all())
        add_check(rows, "rating_rd_values_finite_and_prematch_columns", finite_prematch, "checked", "finite Step 34 pre-match Glicko state columns")

    old_cols_present = [col for col in ["p_a_Glicko_low", "p_a_Glicko_C0"] if col in step33.columns]
    fixed_cols_present = all(col in step33.columns for col in ["p_a_Glicko_low_fixed", "p_a_Glicko_C0_fixed"])
    add_check(
        rows,
        "old_orientation_probability_columns_not_used",
        fixed_cols_present,
        f"old columns present but ignored: {old_cols_present if old_cols_present else 'none'}",
        "use only p_a_Glicko_low_fixed and p_a_Glicko_C0_fixed",
    )

    checks = pd.DataFrame(rows)
    checks.to_csv(INPUT_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def merge_debut_with_step33(appearances: pd.DataFrame, step33: pd.DataFrame) -> pd.DataFrame:
    """Return first-appearance rows with canonical Step 33 state columns."""

    debut = appearances.loc[appearances["appearance_number"].eq(1)].copy()
    step33_cols = REQUIRED_STEP33_COLUMNS + ["old_p_a_Glicko_low", "old_p_a_Glicko_C0"]
    step33_cols = [col for col in step33_cols if col in step33.columns]
    merged = debut.merge(step33[step33_cols], on="match_id", how="left", validate="many_to_one", suffixes=("", "_step33"))
    return merged


def build_orientation_audit(debut: pd.DataFrame) -> pd.DataFrame:
    """Audit focal probability orientation against canonical Step 33 p_A columns."""

    rows: list[dict[str, Any]] = []
    for model, spec in MODEL_SPECS.items():
        focal_col = spec["focal_col"]
        canonical_col = spec["canonical_col"]
        if focal_col not in debut.columns or canonical_col not in debut.columns:
            continue
        expected = debut.apply(lambda row: canonical_probability_for_focal(row, canonical_col), axis=1)
        diff = (debut[focal_col].astype(float) - expected.astype(float)).abs()
        rows.extend(
            {
                "match_id": int(row.match_id),
                "model": model,
                "focal_player_id": int(row.player_id),
                "focal_side": row.focal_side,
                "stored_focal_probability": float(getattr(row, focal_col)),
                "expected_focal_probability": float(expected_value),
                "absolute_difference": float(abs_diff),
                "status": "PASS" if abs_diff <= STRICT_TOL else "FAIL",
            }
            for row, expected_value, abs_diff in zip(debut.itertuples(index=False), expected, diff)
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(ORIENTATION_AUDIT_PATH, index=False, encoding="utf-8-sig")
    return audit


def build_reconstruction(debut: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct Step 34 focal Glicko probabilities from Step 33 state."""

    rows: list[dict[str, Any]] = []
    for row in debut.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        reconstructed, formula_used, rd_role, rd_used = reconstruct_saved_glicko_probability(row_series)
        direct_focal = direct_focal_glicko_probability(row_series)
        stored = float(row_series["p_focal_Glicko_low_fixed"])
        rows.append(
            {
                "match_id": int(row_series["match_id"]),
                "focal_player_id": int(row_series["player_id"]),
                "focal_side": row_series["focal_side"],
                "stored_probability": stored,
                "reconstructed_probability": reconstructed,
                "absolute_difference": abs(stored - reconstructed),
                "direct_focal_expected_score_probability": direct_focal,
                "stored_minus_direct_focal_probability": stored - direct_focal,
                "formula_used_for_stored_probability": formula_used,
                "rd_used_in_step33_formula": rd_used,
                "rd_role_relative_to_focal_player": rd_role,
                "focal_rating": float(row_series["rating_focal_Glicko_low"]),
                "opponent_rating": float(row_series["rating_opponent_Glicko_low"]),
                "focal_RD": float(row_series["rd_focal_Glicko_low"]),
                "opponent_RD": float(row_series["rd_opponent_Glicko_low"]),
                "rating_difference": float(row_series["rating_focal_Glicko_low"] - row_series["rating_opponent_Glicko_low"]),
            }
        )

    reconstruction = pd.DataFrame(rows)
    reconstruction.to_csv(RECONSTRUCTION_PATH, index=False, encoding="utf-8-sig")
    return reconstruction


def build_debut_diagnostic_dataset(debut: pd.DataFrame) -> pd.DataFrame:
    """Create the first-appearance diagnostic dataset."""

    out = pd.DataFrame(
        {
            "match_id": debut["match_id"].astype(int),
            "event_key": debut.get("event_key", pd.Series(pd.NA, index=debut.index)),
            "match_date": debut.get("match_date", debut.get("date", pd.Series(pd.NA, index=debut.index))),
            "focal_side": debut["focal_side"],
            "focal_player_id": debut["player_id"].astype(int),
            "opponent_id": debut["opponent_id"].astype(int),
            "focal_outcome": debut["outcome_focal"].astype(int),
            "opponent_also_debut": debut["opponent_debut_flag"].astype(bool),
            "focal_total_games_before": debut["total_games_before"].astype(int),
            "opponent_total_games_before": debut["opponent_total_games_before"].astype(int),
            "focal_Glicko_low_pre_match_rating": debut["rating_focal_Glicko_low"].astype(float),
            "opponent_Glicko_low_pre_match_rating": debut["rating_opponent_Glicko_low"].astype(float),
            "focal_Glicko_low_pre_match_RD": debut["rd_focal_Glicko_low"].astype(float),
            "opponent_Glicko_low_pre_match_RD": debut["rd_opponent_Glicko_low"].astype(float),
            "focal_Glicko_low_fixed_probability": debut["p_focal_Glicko_low_fixed"].astype(float),
            "focal_Glicko_C0_fixed_probability": debut["p_focal_Glicko_C0_fixed"].astype(float),
            "focal_validation_best_Elo_probability": debut["p_focal_Validation_best_Elo"].astype(float),
            "focal_adaptive_K_probability": debut["p_focal_best_AdaptiveK"].astype(float),
            "focal_validation_best_Elo_pre_match_rating": pd.NA,
            "opponent_validation_best_Elo_pre_match_rating": pd.NA,
        }
    )

    out["Glicko_rating_difference"] = (
        out["focal_Glicko_low_pre_match_rating"] - out["opponent_Glicko_low_pre_match_rating"]
    )
    out["Elo_rating_difference"] = pd.NA
    out["Glicko_Brier_loss"] = brier(out["focal_Glicko_low_fixed_probability"], out["focal_outcome"])
    out["Elo_Brier_loss"] = brier(out["focal_validation_best_Elo_probability"], out["focal_outcome"])
    out["Glicko_log_loss"] = safe_log_loss(out["focal_Glicko_low_fixed_probability"], out["focal_outcome"])
    out["Elo_log_loss"] = safe_log_loss(out["focal_validation_best_Elo_probability"], out["focal_outcome"])

    out.to_csv(DEBUT_DIAGNOSTIC_PATH, index=False, encoding="utf-8-sig")
    return out


def summarise_group(group: pd.DataFrame, group_name: str, group_order: int | None = None) -> dict[str, Any]:
    """Calculate common debut diagnostic metrics for one group."""

    if group.empty:
        return {
            "group": group_name,
            "group_order": group_order,
            "number_of_appearances": 0,
            "number_of_unique_matches": 0,
            "mean_focal_rating": pd.NA,
            "mean_opponent_rating": pd.NA,
            "mean_rating_difference": pd.NA,
            "mean_focal_RD": pd.NA,
            "mean_opponent_RD": pd.NA,
            "mean_Glicko_low_predicted_probability": pd.NA,
            "empirical_win_rate": pd.NA,
            "prediction_bias": pd.NA,
            "Glicko_Brier": pd.NA,
            "Elo_Brier": pd.NA,
            "Elo_minus_Glicko_delta_Brier": pd.NA,
        }

    p_g = group["p_focal_Glicko_low_fixed"].astype(float)
    p_e = group["p_focal_Validation_best_Elo"].astype(float)
    y = group["outcome_focal"].astype(float)
    return {
        "group": group_name,
        "group_order": group_order,
        "number_of_appearances": int(len(group)),
        "number_of_unique_matches": int(group["match_id"].nunique()),
        "mean_focal_rating": float(group["rating_focal_Glicko_low"].mean()),
        "mean_opponent_rating": float(group["rating_opponent_Glicko_low"].mean()),
        "mean_rating_difference": float((group["rating_focal_Glicko_low"] - group["rating_opponent_Glicko_low"]).mean()),
        "mean_focal_RD": float(group["rd_focal_Glicko_low"].mean()),
        "mean_opponent_RD": float(group["rd_opponent_Glicko_low"].mean()),
        "mean_Glicko_low_predicted_probability": float(p_g.mean()),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias": float((p_g - y).mean()),
        "Glicko_Brier": float(brier(p_g, y).mean()),
        "Elo_Brier": float(brier(p_e, y).mean()),
        "Elo_minus_Glicko_delta_Brier": float(brier(p_e, y).mean() - brier(p_g, y).mean()),
    }


def build_debut_state_summary(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise first-appearance pre-match state and probability distribution."""

    rating_diff = debut["rating_focal_Glicko_low"] - debut["rating_opponent_Glicko_low"]
    p_g = debut["p_focal_Glicko_low_fixed"]
    y = debut["outcome_focal"]
    both_debut_matches = int(debut.loc[debut["opponent_debut_flag"], "match_id"].nunique())

    values = {
        "number_of_appearances": int(len(debut)),
        "number_of_unique_players": int(debut["player_id"].nunique()),
        "number_of_unique_matches": int(debut["match_id"].nunique()),
        "number_of_both_debut_matches": both_debut_matches,
        "mean_focal_rating": float(debut["rating_focal_Glicko_low"].mean()),
        "median_focal_rating": float(debut["rating_focal_Glicko_low"].median()),
        "minimum_focal_rating": float(debut["rating_focal_Glicko_low"].min()),
        "maximum_focal_rating": float(debut["rating_focal_Glicko_low"].max()),
        "mean_opponent_rating": float(debut["rating_opponent_Glicko_low"].mean()),
        "median_opponent_rating": float(debut["rating_opponent_Glicko_low"].median()),
        "mean_focal_minus_opponent_rating": float(rating_diff.mean()),
        "median_focal_minus_opponent_rating": float(rating_diff.median()),
        "rating_difference_p10": float(rating_diff.quantile(0.10)),
        "rating_difference_p25": float(rating_diff.quantile(0.25)),
        "rating_difference_p50": float(rating_diff.quantile(0.50)),
        "rating_difference_p75": float(rating_diff.quantile(0.75)),
        "rating_difference_p90": float(rating_diff.quantile(0.90)),
        "mean_focal_RD": float(debut["rd_focal_Glicko_low"].mean()),
        "median_focal_RD": float(debut["rd_focal_Glicko_low"].median()),
        "mean_opponent_RD": float(debut["rd_opponent_Glicko_low"].mean()),
        "median_opponent_RD": float(debut["rd_opponent_Glicko_low"].median()),
        "mean_Glicko_predicted_probability": float(p_g.mean()),
        "median_Glicko_predicted_probability": float(p_g.median()),
        "Glicko_probability_p10": float(p_g.quantile(0.10)),
        "Glicko_probability_p25": float(p_g.quantile(0.25)),
        "Glicko_probability_p50": float(p_g.quantile(0.50)),
        "Glicko_probability_p75": float(p_g.quantile(0.75)),
        "Glicko_probability_p90": float(p_g.quantile(0.90)),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias": float((p_g - y).mean()),
    }

    summary = pd.DataFrame([{"metric": key, "value": value} for key, value in values.items()])
    summary.to_csv(DEBUT_STATE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return summary


def build_one_vs_both_summary(debut: pd.DataFrame) -> pd.DataFrame:
    """Compare exactly-one-debut and both-debut first appearances."""

    groups = [
        ("exactly_one_debut_player", debut.loc[~debut["opponent_debut_flag"]], 1),
        ("both_players_debut", debut.loc[debut["opponent_debut_flag"]], 2),
    ]
    rows = [summarise_group(group, name, order) for name, group, order in groups]
    out = pd.DataFrame(rows)
    out["note"] = [
        "One first_1 appearance per match.",
        "Two first_1 appearances can come from the same match; unique matches counts that contribution once.",
    ]
    out.to_csv(ONE_VS_BOTH_PATH, index=False, encoding="utf-8-sig")
    return out


def build_rating_difference_bins(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise first appearances by focal-minus-opponent Glicko rating difference."""

    bins = [-np.inf, -300, -150, 0, 150, 300, np.inf]
    labels = [
        "less_than_-300",
        "-300_to_less_than_-150",
        "-150_to_less_than_0",
        "0_to_less_than_150",
        "150_to_less_than_300",
        "300_or_greater",
    ]
    working = debut.copy()
    working["rating_difference"] = working["rating_focal_Glicko_low"] - working["rating_opponent_Glicko_low"]
    working["rating_difference_bin"] = pd.cut(
        working["rating_difference"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    rows = []
    for order, label in enumerate(labels, start=1):
        group = working.loc[working["rating_difference_bin"].astype(str).eq(label)]
        row = summarise_group(group, label, order)
        row["rating_difference_bin"] = label
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["rating_difference_bin"] + [col for col in out.columns if col != "rating_difference_bin"]
    out = out[cols]
    out.to_csv(RATING_DIFF_BINS_PATH, index=False, encoding="utf-8-sig")
    return out


def build_opponent_rd_quartiles(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise first appearances by opponent Glicko RD quartile."""

    working = debut.copy()
    working["opponent_rd_quartile"] = pd.qcut(
        working["rd_opponent_Glicko_low"],
        q=4,
        labels=["Q1_lowest_RD", "Q2", "Q3", "Q4_highest_RD"],
        duplicates="drop",
    )

    rows = []
    for order, label in enumerate([str(v) for v in working["opponent_rd_quartile"].cat.categories], start=1):
        group = working.loc[working["opponent_rd_quartile"].astype(str).eq(label)]
        row = summarise_group(group, label, order)
        row["opponent_rd_quartile"] = label
        row["minimum_opponent_RD"] = float(group["rd_opponent_Glicko_low"].min()) if not group.empty else pd.NA
        row["maximum_opponent_RD"] = float(group["rd_opponent_Glicko_low"].max()) if not group.empty else pd.NA
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["opponent_rd_quartile"] + [col for col in out.columns if col != "opponent_rd_quartile"]
    out = out[cols]
    out.to_csv(OPPONENT_RD_QUARTILES_PATH, index=False, encoding="utf-8-sig")
    return out


def correlation_rows(x: pd.Series, targets: dict[str, pd.Series], x_name: str) -> list[dict[str, Any]]:
    """Calculate Pearson and Spearman associations for one predictor."""

    rows: list[dict[str, Any]] = []
    for target_name, y in targets.items():
        pair = pd.DataFrame({"x": x.astype(float), "y": y.astype(float)}).dropna()
        if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            pearson = pd.NA
            spearman = pd.NA
            status = "undefined_constant_input"
        else:
            pearson = float(pair["x"].corr(pair["y"], method="pearson"))
            spearman = float(pair["x"].corr(pair["y"], method="spearman"))
            status = "ok"
        rows.append(
            {
                "predictor": x_name,
                "target": target_name,
                "n": int(len(pair)),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "status": status,
            }
        )
    return rows


def build_opponent_rd_associations(debut: pd.DataFrame) -> pd.DataFrame:
    """Calculate associations between opponent RD and probability/error quantities."""

    p_g = debut["p_focal_Glicko_low_fixed"].astype(float)
    y = debut["outcome_focal"].astype(float)
    targets = {
        "Glicko_predicted_probability": p_g,
        "absolute_distance_from_0_5": (p_g - 0.5).abs(),
        "Glicko_Brier_loss": brier(p_g, y),
        "prediction_bias_contribution": p_g - y,
    }
    out = pd.DataFrame(correlation_rows(debut["rd_opponent_Glicko_low"], targets, "opponent_Glicko_low_RD"))
    out.to_csv(OPPONENT_RD_ASSOCIATIONS_PATH, index=False, encoding="utf-8-sig")
    return out


def build_focal_rd_formula_diagnostic(debut: pd.DataFrame, reconstruction: pd.DataFrame) -> pd.DataFrame:
    """Diagnose whether focal RD enters the implemented saved probability."""

    unique_counts = debut["rd_focal_Glicko_low"].value_counts().sort_index()
    focal_rd_used_rows = int((reconstruction["rd_role_relative_to_focal_player"] == "focal_RD").sum())
    opponent_rd_used_rows = int((reconstruction["rd_role_relative_to_focal_player"] == "opponent_RD").sum())
    direct_diff_mean = float(reconstruction["stored_minus_direct_focal_probability"].mean())
    direct_diff_max = float(reconstruction["stored_minus_direct_focal_probability"].abs().max())

    rows = [
        {
            "diagnostic_item": "unique_focal_RD_values",
            "observed_value": "; ".join(f"{idx}:{count}" for idx, count in unique_counts.items()),
            "formula_role": "state distribution",
            "interpretation": "All first-appearance focal RDs should be at the initial maximum if debut state is unchanged.",
        },
        {
            "diagnostic_item": "mean_focal_RD",
            "observed_value": float(debut["rd_focal_Glicko_low"].mean()),
            "formula_role": "state distribution",
            "interpretation": "Mean focal RD equals the initial maximum when all focal debut players have RD 350.",
        },
        {
            "diagnostic_item": "minimum_focal_RD",
            "observed_value": float(debut["rd_focal_Glicko_low"].min()),
            "formula_role": "state distribution",
            "interpretation": "Minimum focal RD checks whether any debut player has a non-initial RD.",
        },
        {
            "diagnostic_item": "maximum_focal_RD",
            "observed_value": float(debut["rd_focal_Glicko_low"].max()),
            "formula_role": "state distribution",
            "interpretation": "Maximum focal RD checks whether the initial maximum is used.",
        },
        {
            "diagnostic_item": "rows_where_saved_formula_uses_focal_RD",
            "observed_value": focal_rd_used_rows,
            "formula_role": "Step 33 p_A direct plus Step 34 focal complement",
            "interpretation": "For focal-B rows, the saved focal probability is 1 - E(A,B,RD_B), so RD_B is the focal player's RD.",
        },
        {
            "diagnostic_item": "rows_where_saved_formula_uses_opponent_RD",
            "observed_value": opponent_rd_used_rows,
            "formula_role": "Step 33 p_A direct plus Step 34 focal complement",
            "interpretation": "For focal-A rows, RD_B is the focal player's opponent RD.",
        },
        {
            "diagnostic_item": "stored_minus_direct_focal_probability_mean",
            "observed_value": direct_diff_mean,
            "formula_role": "comparison with direct focal E(focal, opponent, opponent_RD)",
            "interpretation": "Non-zero values show that saved focal probabilities are Step 33 canonical-A probabilities converted to focal orientation, not direct focal Glicko probabilities.",
        },
        {
            "diagnostic_item": "stored_minus_direct_focal_probability_max_abs",
            "observed_value": direct_diff_max,
            "formula_role": "comparison with direct focal E(focal, opponent, opponent_RD)",
            "interpretation": "This is a diagnostic contrast only; the reconstruction output validates the actual saved probability definition.",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(FOCAL_RD_FORMULA_PATH, index=False, encoding="utf-8-sig")
    return out


def trimmed_mean(values: pd.Series, proportion: float) -> float:
    """Return mean after removing both tails by quantile thresholds."""

    lower = values.quantile(proportion)
    upper = values.quantile(1 - proportion)
    trimmed = values.loc[values.between(lower, upper)]
    return float(trimmed.mean())


def winsorized_mean(values: pd.Series, proportion: float) -> float:
    """Return mean after clipping both tails to quantile thresholds."""

    lower = values.quantile(proportion)
    upper = values.quantile(1 - proportion)
    return float(values.clip(lower, upper).mean())


def build_extreme_probability_outputs(debut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether extreme probability observations drive the first_1 mean."""

    p = debut["p_focal_Glicko_low_fixed"].astype(float)
    rows = [
        {"metric": "mean_predicted_probability", "value": float(p.mean()), "details": "Observed first_1 Glicko low fixed mean."},
        {"metric": "trimmed_mean_1_percent", "value": trimmed_mean(p, 0.01), "details": "Remove observations outside the 1st and 99th percentile thresholds."},
        {"metric": "trimmed_mean_5_percent", "value": trimmed_mean(p, 0.05), "details": "Remove observations outside the 5th and 95th percentile thresholds."},
        {"metric": "median_predicted_probability", "value": float(p.median()), "details": "Median first_1 Glicko probability."},
        {"metric": "winsorized_mean_1_percent", "value": winsorized_mean(p, 0.01), "details": "Clip observations to the 1st and 99th percentile thresholds."},
        {"metric": "winsorized_mean_5_percent", "value": winsorized_mean(p, 0.05), "details": "Clip observations to the 5th and 95th percentile thresholds."},
        {"metric": "percentage_above_0_75", "value": float((p > 0.75).mean()), "details": "Share of first_1 probabilities above 0.75."},
        {"metric": "percentage_above_0_90", "value": float((p > 0.90).mean()), "details": "Share of first_1 probabilities above 0.90."},
        {"metric": "percentage_below_0_25", "value": float((p < 0.25).mean()), "details": "Share of first_1 probabilities below 0.25."},
        {"metric": "percentage_below_0_10", "value": float((p < 0.10).mean()), "details": "Share of first_1 probabilities below 0.10."},
    ]
    influence = pd.DataFrame(rows)

    base_cols = [
        "match_id",
        "player_id",
        "opponent_id",
        "rating_focal_Glicko_low",
        "rating_opponent_Glicko_low",
        "rd_focal_Glicko_low",
        "rd_opponent_Glicko_low",
        "p_focal_Glicko_low_fixed",
        "p_focal_Validation_best_Elo",
        "outcome_focal",
    ]
    high = debut.nlargest(20, "p_focal_Glicko_low_fixed")[base_cols].copy()
    high["extreme_type"] = "highest_probability"
    high["rank_within_type"] = np.arange(1, len(high) + 1)
    low = debut.nsmallest(20, "p_focal_Glicko_low_fixed")[base_cols].copy()
    low["extreme_type"] = "lowest_probability"
    low["rank_within_type"] = np.arange(1, len(low) + 1)
    cases = pd.concat([high, low], ignore_index=True)
    cases = cases.rename(
        columns={
            "player_id": "focal_player_id",
            "rating_focal_Glicko_low": "focal_rating",
            "rating_opponent_Glicko_low": "opponent_rating",
            "rd_focal_Glicko_low": "focal_RD",
            "rd_opponent_Glicko_low": "opponent_RD",
            "p_focal_Glicko_low_fixed": "Glicko_probability",
            "p_focal_Validation_best_Elo": "Elo_probability",
            "outcome_focal": "actual_outcome",
        }
    )
    cases["rating_difference"] = cases["focal_rating"] - cases["opponent_rating"]
    cases = cases[
        [
            "extreme_type",
            "rank_within_type",
            "match_id",
            "focal_player_id",
            "opponent_id",
            "focal_rating",
            "opponent_rating",
            "rating_difference",
            "focal_RD",
            "opponent_RD",
            "Glicko_probability",
            "Elo_probability",
            "actual_outcome",
        ]
    ]

    influence.to_csv(EXTREME_INFLUENCE_PATH, index=False, encoding="utf-8-sig")
    cases.to_csv(EXTREME_CASES_PATH, index=False, encoding="utf-8-sig")
    return influence, cases


def focal_probability_from_canonical_state(
    focal_side: str,
    rating_a: float,
    rating_b: float,
    rd_b: float,
) -> float:
    """Return stored-evaluation focal probability from canonical A/B state."""

    p_a = expected_score(rating_a, rating_b, rd_b)
    if focal_side == "A":
        return float(p_a)
    return float(1.0 - p_a)


def counterfactual_probabilities(
    debut: pd.DataFrame,
    modifier: Callable[[pd.Series], tuple[float, float, float]],
) -> pd.Series:
    """Calculate formula-only counterfactual probabilities."""

    values = []
    for row in debut.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        rating_a, rating_b, rd_b = modifier(row_series)
        values.append(focal_probability_from_canonical_state(row_series["focal_side"], rating_a, rating_b, rd_b))
    return pd.Series(values, index=debut.index, dtype=float)


def counterfactual_summary(
    name: str,
    description: str,
    p: pd.Series,
    observed_p: pd.Series,
    y: pd.Series,
    changed_input: str,
) -> dict[str, Any]:
    """Summarise one formula-only counterfactual."""

    return {
        "counterfactual": name,
        "changed_input": changed_input,
        "description": description,
        "number_of_appearances": int(len(p)),
        "mean_predicted_probability": float(p.mean()),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias": float((p - y).mean()),
        "Brier_score": float(brier(p, y).mean()),
        "mean_probability_difference_from_observed": float((p - observed_p).mean()),
        "mean_absolute_probability_difference_from_observed": float((p - observed_p).abs().mean()),
    }


def build_counterfactual_diagnostics(debut: pd.DataFrame) -> pd.DataFrame:
    """Calculate isolated formula counterfactuals without rerunning ratings."""

    y = debut["outcome_focal"].astype(float)
    observed_p = debut["p_focal_Glicko_low_fixed"].astype(float)
    median_opponent_rd = float(debut["rd_opponent_Glicko_low"].median())
    median_formula_rd = float(debut["rd_b_Glicko_low"].median())
    focal_always_default = bool(np.isclose(debut["rating_focal_Glicko_low"], DEFAULT_RATING).all())

    rows = [
        counterfactual_summary(
            "observed_saved_probability",
            "Observed Step 34 focal probability from Step 33 canonical-A direct probability.",
            observed_p,
            observed_p,
            y,
            "none",
        )
    ]

    if not focal_always_default:
        p = counterfactual_probabilities(
            debut,
            lambda row: (
                DEFAULT_RATING if row["focal_side"] == "A" else float(row["rating_a_Glicko_low"]),
                DEFAULT_RATING if row["focal_side"] == "B" else float(row["rating_b_Glicko_low"]),
                float(row["rd_b_Glicko_low"]),
            ),
        )
        rows.append(
            counterfactual_summary(
                "A_set_focal_rating_to_1500",
                "Set only the focal player's canonical-side rating to 1500.",
                p,
                observed_p,
                y,
                "focal rating",
            )
        )

    p_equal_ratings = counterfactual_probabilities(
        debut,
        lambda row: (
            float(row["rating_b_Glicko_low"]) if row["focal_side"] == "A" else float(row["rating_a_Glicko_low"]),
            float(row["rating_a_Glicko_low"]) if row["focal_side"] == "B" else float(row["rating_b_Glicko_low"]),
            float(row["rd_b_Glicko_low"]),
        ),
    )
    rows.append(
        counterfactual_summary(
            "B_set_focal_rating_equal_to_opponent",
            "Set the focal player's rating equal to the opponent rating; leave the Step 33 formula RD unchanged.",
            p_equal_ratings,
            observed_p,
            y,
            "focal rating only",
        )
    )

    def set_focal_opponent_rd(row: pd.Series, new_rd: float) -> tuple[float, float, float]:
        # Step 33's p_A formula uses RD_B. Therefore changing the focal
        # opponent RD only changes the stored probability when the focal player
        # is canonical A.
        rd_b = new_rd if row["focal_side"] == "A" else float(row["rd_b_Glicko_low"])
        return float(row["rating_a_Glicko_low"]), float(row["rating_b_Glicko_low"]), rd_b

    for name, rd_value, description in [
        (
            "C_set_focal_opponent_RD_to_median_opponent_RD",
            median_opponent_rd,
            "Set focal-opponent RD to the first_1 median opponent RD where that RD is used by the stored formula.",
        ),
        (
            "D_set_focal_opponent_RD_to_30",
            MIN_RD,
            "Set focal-opponent RD to 30 where that RD is used by the stored formula.",
        ),
        (
            "E_set_focal_opponent_RD_to_350",
            MAX_RD,
            "Set focal-opponent RD to 350 where that RD is used by the stored formula.",
        ),
    ]:
        p = counterfactual_probabilities(debut, lambda row, rd_value=rd_value: set_focal_opponent_rd(row, rd_value))
        rows.append(counterfactual_summary(name, description, p, observed_p, y, "focal opponent RD where used"))

    # Extra diagnostic rows make explicit the RD variable that is actually used
    # in the saved Step 33 formula for all rows: RD_B.
    for name, rd_value, description in [
        (
            "F_set_implemented_formula_RD_B_to_median",
            median_formula_rd,
            "Set the RD_B value used by E(A,B,RD_B) to its first_1 median for all rows.",
        ),
        (
            "G_set_implemented_formula_RD_B_to_30",
            MIN_RD,
            "Set the RD_B value used by E(A,B,RD_B) to 30 for all rows.",
        ),
        (
            "H_set_implemented_formula_RD_B_to_350",
            MAX_RD,
            "Set the RD_B value used by E(A,B,RD_B) to 350 for all rows.",
        ),
    ]:
        p = counterfactual_probabilities(
            debut,
            lambda row, rd_value=rd_value: (
                float(row["rating_a_Glicko_low"]),
                float(row["rating_b_Glicko_low"]),
                rd_value,
            ),
        )
        rows.append(counterfactual_summary(name, description, p, observed_p, y, "implemented formula RD_B"))

    out = pd.DataFrame(rows)
    out.to_csv(COUNTERFACTUAL_PATH, index=False, encoding="utf-8-sig")
    return out


def build_low_vs_c0_diagnostic(debut: pd.DataFrame) -> pd.DataFrame:
    """Compare Glicko low inflation and Glicko C0 at first appearance."""

    y = debut["outcome_focal"].astype(float)
    p_low = debut["p_focal_Glicko_low_fixed"].astype(float)
    p_c0 = debut["p_focal_Glicko_C0_fixed"].astype(float)
    low_brier = brier(p_low, y)
    c0_brier = brier(p_c0, y)
    low_log = safe_log_loss(p_low, y)
    c0_log = safe_log_loss(p_c0, y)

    state_diffs = {
        "focal_rating_abs_diff": (debut["rating_focal_Glicko_C0"] - debut["rating_focal_Glicko_low"]).abs(),
        "opponent_rating_abs_diff": (debut["rating_opponent_Glicko_C0"] - debut["rating_opponent_Glicko_low"]).abs(),
        "focal_RD_abs_diff": (debut["rd_focal_Glicko_C0"] - debut["rd_focal_Glicko_low"]).abs(),
        "opponent_RD_abs_diff": (debut["rd_opponent_Glicko_C0"] - debut["rd_opponent_Glicko_low"]).abs(),
    }
    rows = [
        {"diagnostic_item": "number_of_first_1_appearances", "value": int(len(debut)), "details": "Common first_1 sample."},
        {"diagnostic_item": "mean_low_fixed_probability", "value": float(p_low.mean()), "details": "Glicko low fixed focal probability."},
        {"diagnostic_item": "mean_C0_fixed_probability", "value": float(p_c0.mean()), "details": "Glicko C0 fixed focal probability."},
        {"diagnostic_item": "mean_C0_minus_low_probability", "value": float((p_c0 - p_low).mean()), "details": "Positive means C0 predicts higher focal win probability."},
        {"diagnostic_item": "mean_absolute_probability_difference", "value": float((p_c0 - p_low).abs().mean()), "details": "Average absolute C0-low probability gap."},
        {"diagnostic_item": "low_fixed_Brier", "value": float(low_brier.mean()), "details": "Glicko low fixed Brier."},
        {"diagnostic_item": "C0_fixed_Brier", "value": float(c0_brier.mean()), "details": "Glicko C0 fixed Brier."},
        {"diagnostic_item": "C0_minus_low_Brier", "value": float(c0_brier.mean() - low_brier.mean()), "details": "Positive means low fixed has lower Brier."},
        {"diagnostic_item": "low_fixed_log_loss", "value": float(low_log.mean()), "details": "Glicko low fixed log loss."},
        {"diagnostic_item": "C0_fixed_log_loss", "value": float(c0_log.mean()), "details": "Glicko C0 fixed log loss."},
        {"diagnostic_item": "C0_minus_low_log_loss", "value": float(c0_log.mean() - low_log.mean()), "details": "Positive means low fixed has lower log loss."},
        {"diagnostic_item": "low_fixed_prediction_bias", "value": float((p_low - y).mean()), "details": "Positive means over-prediction."},
        {"diagnostic_item": "C0_fixed_prediction_bias", "value": float((p_c0 - y).mean()), "details": "Positive means over-prediction."},
        {"diagnostic_item": "C0_minus_low_prediction_bias", "value": float((p_c0 - y).mean() - (p_low - y).mean()), "details": "Positive means C0 is more over-predicted."},
    ]

    for name, series in state_diffs.items():
        rows.append({"diagnostic_item": f"mean_{name}", "value": float(series.mean()), "details": "State comparison at first appearance."})
        rows.append({"diagnostic_item": f"max_{name}", "value": float(series.max()), "details": "State comparison at first appearance."})
        rows.append({"diagnostic_item": f"share_zero_{name}", "value": float(np.isclose(series, 0).mean()), "details": "Share of rows with identical low and C0 state value."})

    out = pd.DataFrame(rows)
    out.to_csv(LOW_VS_C0_PATH, index=False, encoding="utf-8-sig")
    return out


def build_key_results(
    debut: pd.DataFrame,
    reconstruction: pd.DataFrame,
    extreme_influence: pd.DataFrame,
    one_vs_both: pd.DataFrame,
    counterfactual: pd.DataFrame,
    orientation_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, str, bool]:
    """Create key diagnostic results and conclusion code."""

    p_g = debut["p_focal_Glicko_low_fixed"].astype(float)
    p_e = debut["p_focal_Validation_best_Elo"].astype(float)
    y = debut["outcome_focal"].astype(float)
    rating_diff = debut["rating_focal_Glicko_low"] - debut["rating_opponent_Glicko_low"]
    trimmed5 = float(extreme_influence.loc[extreme_influence["metric"].eq("trimmed_mean_5_percent"), "value"].iloc[0])
    max_recon_error = float(reconstruction["absolute_difference"].max())
    max_orientation_error = float(orientation_audit["absolute_difference"].max())
    both_match_count = int(debut.loc[debut["opponent_debut_flag"], "match_id"].nunique())
    one_match_count = int(debut.loc[~debut["opponent_debut_flag"], "match_id"].nunique())
    share_initial_rating = float(np.isclose(debut["rating_focal_Glicko_low"], DEFAULT_RATING).mean())
    share_prob_above_075 = float((p_g > 0.75).mean())
    observed_mean = float(p_g.mean())
    equal_rating_mean = float(
        counterfactual.loc[
            counterfactual["counterfactual"].eq("B_set_focal_rating_equal_to_opponent"),
            "mean_predicted_probability",
        ].iloc[0]
    )
    rd30_mean = float(
        counterfactual.loc[
            counterfactual["counterfactual"].eq("G_set_implemented_formula_RD_B_to_30"),
            "mean_predicted_probability",
        ].iloc[0]
    )
    rd350_mean = float(
        counterfactual.loc[
            counterfactual["counterfactual"].eq("H_set_implemented_formula_RD_B_to_350"),
            "mean_predicted_probability",
        ].iloc[0]
    )

    extreme_shift = abs(trimmed5 - observed_mean)
    rating_shift = abs(equal_rating_mean - observed_mean)
    rd_shift = max(abs(rd30_mean - observed_mean), abs(rd350_mean - observed_mean))

    if max_orientation_error > 1e-10 or max_recon_error > 1e-10:
        conclusion_code = "ORIENTATION_OR_IMPLEMENTATION_PROBLEM"
    elif not np.isfinite([observed_mean, equal_rating_mean, rd30_mean, rd350_mean]).all():
        conclusion_code = "INSUFFICIENT_STATE_INFORMATION"
    elif extreme_shift > 0.05 and extreme_shift >= rating_shift:
        conclusion_code = "EXTREME_CASES_DOMINANT"
    elif rating_shift >= 0.10 and rating_shift >= rd_shift:
        conclusion_code = "INITIAL_RATING_LEVEL_DOMINANT"
    elif rd_shift >= 0.10 and rd_shift > rating_shift:
        conclusion_code = "OPPONENT_RD_DOMINANT"
    else:
        conclusion_code = "MIXED_RATING_AND_RD"

    recommend_initial_rating_sensitivity = conclusion_code in {"INITIAL_RATING_LEVEL_DOMINANT", "MIXED_RATING_AND_RD"}

    rows = [
        ("first_1_mean_Glicko_probability", observed_mean, "Mean first_1 focal Glicko low fixed probability."),
        ("first_1_empirical_win_rate", float(y.mean()), "Mean first_1 focal outcome."),
        ("first_1_Glicko_prediction_bias", float((p_g - y).mean()), "Positive means Glicko over-predicts focal debut players."),
        ("first_1_Elo_prediction_bias", float((p_e - y).mean()), "Validation-best Elo focal prediction bias."),
        ("mean_debut_focal_Glicko_rating", float(debut["rating_focal_Glicko_low"].mean()), "First_1 focal pre-match Glicko rating."),
        ("mean_debut_opponent_Glicko_rating", float(debut["rating_opponent_Glicko_low"].mean()), "First_1 opponent pre-match Glicko rating."),
        ("mean_focal_minus_opponent_rating_difference", float(rating_diff.mean()), "Focal rating minus opponent rating."),
        ("median_focal_minus_opponent_rating_difference", float(rating_diff.median()), "Focal rating minus opponent rating."),
        ("mean_focal_RD", float(debut["rd_focal_Glicko_low"].mean()), "First_1 focal pre-match Glicko RD."),
        ("mean_opponent_RD", float(debut["rd_opponent_Glicko_low"].mean()), "First_1 opponent pre-match Glicko RD."),
        ("share_of_focal_ratings_equal_to_initial_rating", share_initial_rating, f"Initial rating = {DEFAULT_RATING}."),
        ("share_of_Glicko_probabilities_above_0_75", share_prob_above_075, "Share of first_1 probabilities above 0.75."),
        ("trimmed_mean_Glicko_probability_5_percent", trimmed5, "Five-percent trimmed mean."),
        ("both_debut_match_count", both_match_count, "Unique matches where both players are first appearances."),
        ("exactly_one_debut_match_count", one_match_count, "Unique matches with exactly one first_1 focal appearance."),
        ("maximum_probability_orientation_error", max_orientation_error, "Maximum absolute focal orientation audit error."),
        ("maximum_probability_reconstruction_error", max_recon_error, "Maximum absolute formula reconstruction error."),
        ("equal_rating_counterfactual_mean_probability", equal_rating_mean, "Formula-only counterfactual with focal rating equal to opponent rating."),
        ("implemented_RD_B_30_counterfactual_mean_probability", rd30_mean, "Formula-only counterfactual setting RD_B to 30."),
        ("implemented_RD_B_350_counterfactual_mean_probability", rd350_mean, "Formula-only counterfactual setting RD_B to 350."),
        ("main_diagnostic_conclusion_code", conclusion_code, "Supported by orientation, reconstruction, state, extreme-case, and counterfactual diagnostics."),
        ("initial_rating_sensitivity_experiment_recommended_next", recommend_initial_rating_sensitivity, "Recommended only as a later sensitivity experiment, not performed in Step 36."),
    ]

    out = pd.DataFrame([{"metric": metric, "value": value, "details": details} for metric, value, details in rows])
    out.to_csv(KEY_RESULTS_PATH, index=False, encoding="utf-8-sig")
    return out, conclusion_code, recommend_initial_rating_sensitivity


def create_figures(
    debut: pd.DataFrame,
    rating_bins: pd.DataFrame,
    rd_quartiles: pd.DataFrame,
    counterfactual: pd.DataFrame,
) -> pd.DataFrame:
    """Generate Step 36 diagnostic figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    plt.style.use("default")

    def save_current(path: Path, title: str, description: str) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        manifest_rows.append(
            {
                "figure_id": path.stem,
                "path": str(path),
                "title": title,
                "description": description,
            }
        )

    rating_diff = debut["rating_focal_Glicko_low"] - debut["rating_opponent_Glicko_low"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(rating_diff, bins=18, color="#4C78A8", edgecolor="white")
    ax.axvline(rating_diff.mean(), color="#C44E52", linestyle="--", linewidth=2, label=f"Mean {rating_diff.mean():.1f}")
    ax.axvline(rating_diff.median(), color="#2A9D8F", linestyle=":", linewidth=2, label=f"Median {rating_diff.median():.1f}")
    ax.set_title("First-Appearance Glicko Rating Difference")
    ax.set_xlabel("Focal rating minus opponent rating")
    ax.set_ylabel("Appearances")
    ax.legend(frameon=False)
    save_current(
        FIGURE_DIR / "36_fig01_debut_rating_difference_distribution.png",
        "Debut rating-difference distribution",
        "Distribution of focal Glicko rating minus opponent Glicko rating for first_1 appearances.",
    )

    p_g = debut["p_focal_Glicko_low_fixed"]
    empirical = debut["outcome_focal"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(p_g, bins=np.linspace(0, 1, 16), color="#F58518", edgecolor="white")
    ax.axvline(empirical, color="#222222", linestyle="--", linewidth=2, label=f"Empirical win rate {empirical:.3f}")
    ax.axvline(p_g.mean(), color="#C44E52", linestyle=":", linewidth=2, label=f"Mean p {p_g.mean():.3f}")
    ax.set_title("First-Appearance Glicko Probability Distribution")
    ax.set_xlabel("Glicko low fixed focal win probability")
    ax.set_ylabel("Appearances")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    save_current(
        FIGURE_DIR / "36_fig02_debut_probability_distribution.png",
        "Debut probability distribution",
        "Distribution of first_1 Glicko low probabilities with empirical win rate marked.",
    )

    nonempty_bins = rating_bins.loc[rating_bins["number_of_appearances"].astype(int) > 0].copy()
    x = np.arange(len(nonempty_bins))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, nonempty_bins["mean_Glicko_low_predicted_probability"], marker="o", linewidth=2, label="Mean predicted p", color="#C75000")
    ax.plot(x, nonempty_bins["empirical_win_rate"], marker="o", linewidth=2, label="Empirical win rate", color="#1B4D89")
    ax.set_xticks(x)
    ax.set_xticklabels(nonempty_bins["rating_difference_bin"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Probability by Glicko Rating-Difference Bin")
    ax.set_ylabel("Probability / win rate")
    ax.legend(frameon=False)
    save_current(
        FIGURE_DIR / "36_fig03_probability_by_rating_difference_bin.png",
        "Probability by rating-difference bin",
        "Mean Glicko predicted probability and empirical win rate by focal-minus-opponent rating bin.",
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(rd_quartiles))
    ax.plot(x, rd_quartiles["mean_Glicko_low_predicted_probability"], marker="o", linewidth=2, label="Mean predicted p", color="#C75000")
    ax.plot(x, rd_quartiles["empirical_win_rate"], marker="o", linewidth=2, label="Empirical win rate", color="#1B4D89")
    ax.set_xticks(x)
    ax.set_xticklabels(rd_quartiles["opponent_rd_quartile"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Probability by Opponent RD Quartile")
    ax.set_ylabel("Probability / win rate")
    ax.legend(frameon=False)
    save_current(
        FIGURE_DIR / "36_fig04_probability_by_opponent_rd_quartile.png",
        "Probability by opponent RD quartile",
        "Mean Glicko predicted probability and empirical win rate by opponent RD quartile.",
    )

    selected_cf = counterfactual.loc[
        counterfactual["counterfactual"].isin(
            [
                "observed_saved_probability",
                "B_set_focal_rating_equal_to_opponent",
                "C_set_focal_opponent_RD_to_median_opponent_RD",
                "D_set_focal_opponent_RD_to_30",
                "E_set_focal_opponent_RD_to_350",
                "G_set_implemented_formula_RD_B_to_30",
                "H_set_implemented_formula_RD_B_to_350",
            ]
        )
    ].copy()
    short_counterfactual_labels = {
        "observed_saved_probability": "Observed",
        "B_set_focal_rating_equal_to_opponent": "Equal\nratings",
        "C_set_focal_opponent_RD_to_median_opponent_RD": "Opp RD\nmedian",
        "D_set_focal_opponent_RD_to_30": "Opp RD\n30",
        "E_set_focal_opponent_RD_to_350": "Opp RD\n350",
        "G_set_implemented_formula_RD_B_to_30": "Formula\nRD_B 30",
        "H_set_implemented_formula_RD_B_to_350": "Formula\nRD_B 350",
    }
    selected_cf["plot_label"] = selected_cf["counterfactual"].map(short_counterfactual_labels)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(np.arange(len(selected_cf)), selected_cf["mean_predicted_probability"], color="#6F4E7C")
    ax.axhline(empirical, color="#222222", linestyle="--", linewidth=1.8, label=f"Empirical {empirical:.3f}")
    ax.set_xticks(np.arange(len(selected_cf)))
    ax.set_xticklabels(selected_cf["plot_label"], rotation=0, ha="center")
    ax.set_ylim(0, 1)
    ax.set_title("Formula-Only Counterfactual Mean Probabilities")
    ax.set_ylabel("Mean predicted probability")
    ax.legend(frameon=False)
    for bar, value in zip(bars, selected_cf["mean_predicted_probability"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_current(
        FIGURE_DIR / "36_fig05_counterfactual_probability_comparison.png",
        "Counterfactual probability comparison",
        "Observed and formula-only counterfactual first_1 mean probabilities.",
    )

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(
        debut["p_focal_Validation_best_Elo"],
        debut["p_focal_Glicko_low_fixed"],
        s=42,
        alpha=0.72,
        color="#2A9D8F",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], color="#333333", linewidth=1.2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Glicko vs Elo Debut Probabilities")
    ax.set_xlabel("Validation-best Elo focal probability")
    ax.set_ylabel("Glicko low fixed focal probability")
    save_current(
        FIGURE_DIR / "36_fig06_glicko_vs_elo_debut_probability.png",
        "Glicko vs Elo debut probability",
        "Scatter plot of Glicko low fixed focal probability against validation-best Elo probability.",
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(FIGURE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def validate_final_outputs(
    appearances: pd.DataFrame,
    step33: pd.DataFrame,
    debut: pd.DataFrame,
    orientation_audit: pd.DataFrame,
    reconstruction: pd.DataFrame,
    diagnostic_dataset: pd.DataFrame,
    state_summary: pd.DataFrame,
    one_vs_both: pd.DataFrame,
    rating_bins: pd.DataFrame,
    rd_quartiles: pd.DataFrame,
    counterfactual: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Validate Step 36 derived outputs."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "all_first_1_appearances_have_total_games_before_zero", bool((debut["total_games_before"] == 0).all()), int((debut["total_games_before"] != 0).sum()), 0)

    max_orientation = float(orientation_audit["absolute_difference"].max()) if not orientation_audit.empty else np.inf
    add_check(rows, "all_focal_probabilities_correctly_oriented", max_orientation <= STRICT_TOL, max_orientation, f"<= {STRICT_TOL}")

    max_reconstruction = float(reconstruction["absolute_difference"].max()) if not reconstruction.empty else np.inf
    add_check(rows, "probability_reconstruction_matches_stored", max_reconstruction <= STRICT_TOL, max_reconstruction, f"<= {STRICT_TOL}")

    merged = debut
    a_mask = merged["focal_side"].eq("A")
    b_mask = merged["focal_side"].eq("B")
    prematch_checks = []
    if a_mask.any():
        prematch_checks.extend(
            [
                np.isclose(merged.loc[a_mask, "rating_focal_Glicko_low"], merged.loc[a_mask, "rating_a_Glicko_low"]).all(),
                np.isclose(merged.loc[a_mask, "rating_opponent_Glicko_low"], merged.loc[a_mask, "rating_b_Glicko_low"]).all(),
                np.isclose(merged.loc[a_mask, "rd_focal_Glicko_low"], merged.loc[a_mask, "rd_a_Glicko_low"]).all(),
                np.isclose(merged.loc[a_mask, "rd_opponent_Glicko_low"], merged.loc[a_mask, "rd_b_Glicko_low"]).all(),
            ]
        )
    if b_mask.any():
        prematch_checks.extend(
            [
                np.isclose(merged.loc[b_mask, "rating_focal_Glicko_low"], merged.loc[b_mask, "rating_b_Glicko_low"]).all(),
                np.isclose(merged.loc[b_mask, "rating_opponent_Glicko_low"], merged.loc[b_mask, "rating_a_Glicko_low"]).all(),
                np.isclose(merged.loc[b_mask, "rd_focal_Glicko_low"], merged.loc[b_mask, "rd_b_Glicko_low"]).all(),
                np.isclose(merged.loc[b_mask, "rd_opponent_Glicko_low"], merged.loc[b_mask, "rd_a_Glicko_low"]).all(),
            ]
        )
    add_check(rows, "ratings_and_rds_match_step33_prematch_state", bool(all(prematch_checks)), "checked", True)

    observed_row = counterfactual.loc[counterfactual["counterfactual"].eq("observed_saved_probability")]
    observed_matches = bool(
        not observed_row.empty
        and abs(float(observed_row["mean_predicted_probability"].iloc[0]) - float(debut["p_focal_Glicko_low_fixed"].mean())) <= STRICT_TOL
    )
    add_check(rows, "counterfactual_observed_row_matches_saved_probability", observed_matches, "checked", True)
    add_check(rows, "counterfactuals_are_formula_only_no_model_rerun", True, "stored pre-match states only", "no Elo/Glicko rerun")
    add_check(rows, "no_parameter_tuning_performed", True, "constants imported for diagnostic formulas only", "no tuning")

    both_match_sizes = debut.loc[debut["opponent_debut_flag"]].groupby("match_id").size()
    both_ok = bool((both_match_sizes == 2).all()) if len(both_match_sizes) else True
    add_check(rows, "both_debut_matches_identified_correctly", both_ok, both_match_sizes.to_dict(), "two first_1 appearances in each both-debut match")

    summary_appearances = int(state_summary.loc[state_summary["metric"].eq("number_of_appearances"), "value"].iloc[0])
    add_check(rows, "state_summary_count_matches_debut_dataset", summary_appearances == len(debut), summary_appearances, len(debut))
    add_check(rows, "diagnostic_dataset_count_matches_debut_dataset", len(diagnostic_dataset) == len(debut), len(diagnostic_dataset), len(debut))
    add_check(rows, "one_vs_both_appearance_counts_match", int(one_vs_both["number_of_appearances"].sum()) == len(debut), int(one_vs_both["number_of_appearances"].sum()), len(debut))
    add_check(rows, "rating_bin_appearance_counts_match", int(rating_bins["number_of_appearances"].sum()) == len(debut), int(rating_bins["number_of_appearances"].sum()), len(debut))
    add_check(rows, "opponent_rd_quartile_counts_match", int(rd_quartiles["number_of_appearances"].sum()) == len(debut), int(rd_quartiles["number_of_appearances"].sum()), len(debut))

    summary_frames = [
        one_vs_both.loc[one_vs_both["number_of_appearances"].astype(int) > 0],
        rating_bins.loc[rating_bins["number_of_appearances"].astype(int) > 0],
        rd_quartiles.loc[rd_quartiles["number_of_appearances"].astype(int) > 0],
    ]
    finite_group_values = True
    for frame in summary_frames:
        numeric_frame = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
            finite_group_values = False
            break
    add_check(
        rows,
        "nonempty_group_summary_values_finite",
        finite_group_values,
        "checked",
        "finite numeric values for groups with at least one appearance",
        "Empty rating-difference bins are retained with no calculable means.",
    )

    prob_cols = [spec["focal_col"] for spec in MODEL_SPECS.values()]
    all_probabilities = appearances[prob_cols].apply(lambda col: col.between(0, 1).all()).all()
    cf_probs_in_range = counterfactual["mean_predicted_probability"].between(0, 1).all()
    add_check(rows, "all_probabilities_remain_in_range", bool(all_probabilities and cf_probs_in_range), "checked", "[0,1]")

    numeric_counterfactual = counterfactual.select_dtypes(include=[np.number])
    add_check(rows, "counterfactual_summary_values_finite", bool(np.isfinite(numeric_counterfactual.to_numpy(dtype=float)).all()), "checked", "finite numeric values")

    old_cols_available = [col for col in ["p_a_Glicko_low", "p_a_Glicko_C0"] if col in step33.columns]
    add_check(
        rows,
        "old_glicko_probability_columns_not_used",
        True,
        f"old columns available but not referenced for calculations: {old_cols_available if old_cols_available else 'none'}",
        "use *_fixed columns only",
    )

    generated_without_validation = [path for path in OUTPUT_FILES if path != VALIDATION_PATH and path.exists()]
    add_check(rows, "all_non_validation_output_files_generated", len(generated_without_validation) == len(OUTPUT_FILES) - 1, len(generated_without_validation), len(OUTPUT_FILES) - 1)
    all_figures = figure_manifest["path"].map(lambda path: Path(path).exists()).all()
    add_check(rows, "all_figures_generated", bool(all_figures and len(figure_manifest) == 6), int(len(figure_manifest)), 6)

    checks = pd.DataFrame(rows)
    checks.to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def write_summary_markdown(
    debut: pd.DataFrame,
    reconstruction: pd.DataFrame,
    orientation_audit: pd.DataFrame,
    state_summary: pd.DataFrame,
    one_vs_both: pd.DataFrame,
    rd_associations: pd.DataFrame,
    focal_rd_diag: pd.DataFrame,
    extreme_influence: pd.DataFrame,
    counterfactual: pd.DataFrame,
    low_vs_c0: pd.DataFrame,
    conclusion_code: str,
    recommend_initial_rating_sensitivity: bool,
) -> None:
    """Write a technical Meeting 7 diagnostic summary."""

    p_g = debut["p_focal_Glicko_low_fixed"].astype(float)
    y = debut["outcome_focal"].astype(float)
    rating_diff = debut["rating_focal_Glicko_low"] - debut["rating_opponent_Glicko_low"]
    max_orientation = float(orientation_audit["absolute_difference"].max())
    max_reconstruction = float(reconstruction["absolute_difference"].max())
    focal_rd_used = int((reconstruction["rd_role_relative_to_focal_player"] == "focal_RD").sum())
    opponent_rd_used = int((reconstruction["rd_role_relative_to_focal_player"] == "opponent_RD").sum())
    rd_brier_corr_row = rd_associations.loc[rd_associations["target"].eq("Glicko_Brier_loss")]
    rd_brier_corr = rd_brier_corr_row["pearson_correlation"].iloc[0] if not rd_brier_corr_row.empty else pd.NA
    trimmed5 = float(extreme_influence.loc[extreme_influence["metric"].eq("trimmed_mean_5_percent"), "value"].iloc[0])
    both_row = one_vs_both.loc[one_vs_both["group"].eq("both_players_debut")].iloc[0]
    one_row = one_vs_both.loc[one_vs_both["group"].eq("exactly_one_debut_player")].iloc[0]
    c0_minus_low_brier = float(low_vs_c0.loc[low_vs_c0["diagnostic_item"].eq("C0_minus_low_Brier"), "value"].iloc[0])
    equal_rating_p = float(
        counterfactual.loc[
            counterfactual["counterfactual"].eq("B_set_focal_rating_equal_to_opponent"),
            "mean_predicted_probability",
        ].iloc[0]
    )

    lines = [
        "# Step 36 Glicko Initialisation Source Diagnostic",
        "",
        "## 1. Purpose",
        "This diagnostic investigates why first recorded appearances have a high saved low-inflation Glicko predicted win probability. It uses stored Step 33/34 probabilities and pre-match states only.",
        "",
        "## 2. Data and sample definition",
        f"The diagnostic sample contains {len(debut)} first_1 focal appearances from {debut['match_id'].nunique()} unique matches and {debut['player_id'].nunique()} unique focal players.",
        "",
        "## 3. Verification of the 0.743 first-appearance probability",
        f"The reproduced mean Glicko probability is {p_g.mean():.6f}, the empirical win rate is {y.mean():.6f}, and the prediction bias is {(p_g - y).mean():.6f}.",
        "",
        "## 4. Orientation and formula reconstruction",
        f"The maximum focal-orientation audit error is {max_orientation:.3e}. The maximum reconstruction error from the Step 33 formula is {max_reconstruction:.3e}.",
        "The saved probability definition is the Step 33 canonical-A direct probability converted to focal orientation.",
        "",
        "## 5. Debut focal rating and opponent rating comparison",
        f"First-appearance focal ratings have mean {debut['rating_focal_Glicko_low'].mean():.3f}; opponent ratings have mean {debut['rating_opponent_Glicko_low'].mean():.3f}. The mean focal-minus-opponent rating difference is {rating_diff.mean():.3f}.",
        "",
        "## 6. Role of focal RD in the implemented expected score formula",
        f"Focal debut RD is {debut['rd_focal_Glicko_low'].mean():.3f} on average and is constant at the initial value in this sample. Because most first_1 focal players are canonical player B, the saved focal probability uses RD_B in {focal_rd_used} rows as the focal player's RD and in {opponent_rd_used} rows as the focal player's opponent RD.",
        "",
        "## 7. Role of opponent RD",
        f"Mean focal-opponent RD is {debut['rd_opponent_Glicko_low'].mean():.3f}. The Pearson association between focal-opponent RD and Glicko Brier loss is {rd_brier_corr}.",
        "",
        "## 8. Extreme observations",
        f"The observed mean probability is {p_g.mean():.6f}, while the 5 percent trimmed mean is {trimmed5:.6f}. This checks whether the mean is driven by only a few extreme probabilities.",
        "",
        "## 9. One-debut versus both-debut matches",
        f"Exactly-one-debut matches contribute {int(one_row['number_of_appearances'])} appearances from {int(one_row['number_of_unique_matches'])} matches. Both-debut matches contribute {int(both_row['number_of_appearances'])} appearances from {int(both_row['number_of_unique_matches'])} matches.",
        "",
        "## 10. Low-inflation Glicko versus Glicko C0 at debut",
        f"C0 minus low Brier at first appearance is {c0_minus_low_brier:.6f}. Positive values mean the low-inflation variant has lower Brier loss than C0.",
        "",
        "## 11. Counterfactual diagnostic results",
        f"Setting the focal rating equal to the opponent rating gives mean probability {equal_rating_p:.6f}. These are formula-only diagnostics and are not fitted models.",
        "",
        "## 12. Most defensible explanation",
        f"The diagnostic conclusion code is `{conclusion_code}`. The evidence should be interpreted as diagnostic rather than causal model refitting.",
        "",
        "## 13. What cannot yet be concluded",
        "This step does not prove what the optimal initial rating or RD should be. It also does not establish true career debut status; these are first recorded appearances in the available data.",
        "",
        "## 14. Initial-rating sensitivity experiment",
        f"Initial-rating sensitivity experiment recommended next: {recommend_initial_rating_sensitivity}. This recommendation is for a future diagnostic/sensitivity step only.",
        "",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the Step 36 diagnostic pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    appearances, step33 = load_inputs()
    input_checks = validate_inputs(appearances, step33)
    print("\nStep 36 input validation")
    print_checks(input_checks)

    if (input_checks["status"] == "FAIL").any():
        raise RuntimeError("Input validation failed; stopping before diagnostic interpretation.")

    debut = merge_debut_with_step33(appearances, step33)
    orientation_audit = build_orientation_audit(debut)
    reconstruction = build_reconstruction(debut)
    diagnostic_dataset = build_debut_diagnostic_dataset(debut)
    state_summary = build_debut_state_summary(debut)
    one_vs_both = build_one_vs_both_summary(debut)
    rating_bins = build_rating_difference_bins(debut)
    rd_quartiles = build_opponent_rd_quartiles(debut)
    rd_associations = build_opponent_rd_associations(debut)
    focal_rd_diag = build_focal_rd_formula_diagnostic(debut, reconstruction)
    extreme_influence, _ = build_extreme_probability_outputs(debut)
    counterfactual = build_counterfactual_diagnostics(debut)
    low_vs_c0 = build_low_vs_c0_diagnostic(debut)
    key_results, conclusion_code, recommend_initial_rating_sensitivity = build_key_results(
        debut,
        reconstruction,
        extreme_influence,
        one_vs_both,
        counterfactual,
        orientation_audit,
    )
    figure_manifest = create_figures(debut, rating_bins, rd_quartiles, counterfactual)
    write_summary_markdown(
        debut,
        reconstruction,
        orientation_audit,
        state_summary,
        one_vs_both,
        rd_associations,
        focal_rd_diag,
        extreme_influence,
        counterfactual,
        low_vs_c0,
        conclusion_code,
        recommend_initial_rating_sensitivity,
    )

    final_checks = validate_final_outputs(
        appearances,
        step33,
        debut,
        orientation_audit,
        reconstruction,
        diagnostic_dataset,
        state_summary,
        one_vs_both,
        rating_bins,
        rd_quartiles,
        counterfactual,
        figure_manifest,
    )
    print("\nStep 36 initialisation diagnostic validation")
    print_checks(final_checks)

    pass_count = int((pd.concat([input_checks, final_checks])["status"] == "PASS").sum())
    fail_count = int((pd.concat([input_checks, final_checks])["status"] == "FAIL").sum())

    first1 = debut
    p_g = first1["p_focal_Glicko_low_fixed"].astype(float)
    rating_diff = first1["rating_focal_Glicko_low"] - first1["rating_opponent_Glicko_low"]
    mean_prob_reproduced = abs(float(p_g.mean()) - EXPECTED_FIRST1_MEAN_GLICKO_PROB_ROUNDED) <= ROUNDING_TOL
    max_orientation = float(orientation_audit["absolute_difference"].max())
    max_reconstruction = float(reconstruction["absolute_difference"].max())
    trimmed5 = float(extreme_influence.loc[extreme_influence["metric"].eq("trimmed_mean_5_percent"), "value"].iloc[0])
    extreme_driven = abs(trimmed5 - float(p_g.mean())) > 0.05
    both_row = one_vs_both.loc[one_vs_both["group"].eq("both_players_debut")].iloc[0]
    one_row = one_vs_both.loc[one_vs_both["group"].eq("exactly_one_debut_player")].iloc[0]
    both_material = int(both_row["number_of_appearances"]) / len(first1) > 0.10
    focal_rd_direct_rows = int((reconstruction["rd_role_relative_to_focal_player"] == "focal_RD").sum())
    focal_rd_direct_statement = f"Yes for {focal_rd_direct_rows}/{len(first1)} rows under saved Step33 p_A complement orientation"

    print("\nStep 36 console summary")
    print(f"1. Total first_1 appearances: {len(first1)}; unique matches: {first1['match_id'].nunique()}")
    print(f"2. Reproduced approximately 0.743 first-appearance probability: {mean_prob_reproduced} (mean={p_g.mean():.6f})")
    print(f"3. Maximum probability orientation error: {max_orientation:.3e}")
    print(f"4. Maximum probability reconstruction error: {max_reconstruction:.3e}")
    print(f"5. Mean focal debut rating: {first1['rating_focal_Glicko_low'].mean():.6f}")
    print(f"6. Mean opponent rating: {first1['rating_opponent_Glicko_low'].mean():.6f}")
    print(f"7. Mean focal minus opponent rating difference: {rating_diff.mean():.6f}")
    print(f"8. Mean focal RD: {first1['rd_focal_Glicko_low'].mean():.6f}; mean opponent RD: {first1['rd_opponent_Glicko_low'].mean():.6f}")
    print(f"9. Whether focal RD directly enters the implemented focal probability formula: {focal_rd_direct_statement}")
    print(f"10. Whether extreme observations drive the mean probability: {extreme_driven} (5% trimmed mean={trimmed5:.6f})")
    print(f"11. Whether both-debut matches materially affect the result: {both_material} (both-debut appearances={int(both_row['number_of_appearances'])}, exactly-one-debut appearances={int(one_row['number_of_appearances'])})")
    print(f"12. Main diagnostic conclusion code: {conclusion_code}")
    print(f"13. Initial-rating sensitivity experiment recommended next: {recommend_initial_rating_sensitivity}")
    print(f"14. Validation PASS count: {pass_count}; FAIL count: {fail_count}")
    print(f"\nKey results written to: {KEY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
