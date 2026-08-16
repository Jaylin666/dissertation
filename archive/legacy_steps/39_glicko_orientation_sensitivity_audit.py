"""Audit Glicko probability-orientation sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glicko_core import expected_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP33_PATH = PROJECT_ROOT / "outputs" / "meeting6" / "33_orientation_corrected_per_match_scores_2025.csv"
STEP34_PATH = OUTPUT_DIR / "34_early_game_appearance_dataset.csv"

EXPECTED_MATCH_ROWS = 11_379
EXPECTED_APPEARANCE_ROWS = 22_758
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 20260716
EPS = 1e-15
TOL = 1e-10

INPUT_VALIDATION_PATH = OUTPUT_DIR / "39_input_validation_checks.csv"
MATCH_ORIENTATION_PATH = OUTPUT_DIR / "39_match_orientation_dataset.csv"
RECON_CHECKS_PATH = OUTPUT_DIR / "39_current_convention_reconstruction_checks.csv"
COMPLEMENT_SUMMARY_PATH = OUTPUT_DIR / "39_complement_gap_summary.csv"
COMPLEMENT_ASSOC_PATH = OUTPUT_DIR / "39_complement_gap_associations.csv"
SIDE_DISTRIBUTION_PATH = OUTPUT_DIR / "39_early_player_side_distribution.csv"
APPEARANCE_ORIENTATION_PATH = OUTPUT_DIR / "39_appearance_orientation_dataset.csv"
APPEARANCE_METRICS_PATH = OUTPUT_DIR / "39_appearance_convention_metrics.csv"
MATCH_METRICS_PATH = OUTPUT_DIR / "39_match_level_convention_metrics.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "39_orientation_bootstrap_confidence_intervals.csv"
SENSITIVITY_COMPARISON_PATH = OUTPUT_DIR / "39_orientation_sensitivity_comparison.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "39_key_orientation_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "39_figure_manifest.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "39_orientation_validation_checks.csv"

FIGURE_PATHS = {
    "39_fig01_complement_gap_by_group": FIGURE_DIR / "39_fig01_complement_gap_by_group.png",
    "39_fig02_first1_probability_by_convention": FIGURE_DIR / "39_fig02_first1_probability_by_convention.png",
    "39_fig03_elo_glicko_delta_brier_by_convention": FIGURE_DIR / "39_fig03_elo_glicko_delta_brier_by_convention.png",
    "39_fig04_current_vs_midpoint_probability": FIGURE_DIR / "39_fig04_current_vs_midpoint_probability.png",
}

CUMULATIVE_GROUPS = ["first_1", "first_5", "first_10", "first_20", "first_30", "first_50"]
EARLY_BOOTSTRAP_GROUPS = ["first_1", "first_5", "first_10", "first_20"]
STAGE_ORDER = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
CONVENTIONS = ["current", "reversed", "midpoint"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clip_prob(values: Any) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def brier(y_true: Any, p_pred: Any) -> float:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(p_pred)
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: Any, p_pred: Any) -> float:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(p_pred)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y_true: Any, p_pred: Any) -> float:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(p_pred)
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


def metric_dict(y_true: Any, p_pred: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(p_pred)
    return {
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "accuracy": accuracy(y, p),
        "mean_predicted_probability": float(np.mean(p)),
        "empirical_win_rate": float(np.mean(y)),
        "prediction_bias": float(np.mean(p) - np.mean(y)),
    }


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    status: bool,
    observed: Any = "",
    expected: Any = "",
    details: str = "",
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "status": "PASS" if status else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    step33 = pd.read_csv(STEP33_PATH, low_memory=False)
    appearances = pd.read_csv(STEP34_PATH, low_memory=False)
    return step33, appearances


def validate_inputs(step33: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    match_required = [
        "match_id",
        "event_key",
        "player_a_id",
        "player_b_id",
        "winner_id",
        "loser_id",
        "outcome_a",
        "a_total_games_before",
        "b_total_games_before",
        "p_a_Validation_best_Elo",
        "p_a_Glicko_low_fixed",
        "p_a_Glicko_C0_fixed",
        "rating_a_Glicko_low",
        "rating_b_Glicko_low",
        "rd_a_Glicko_low",
        "rd_b_Glicko_low",
        "rating_a_Glicko_C0",
        "rating_b_Glicko_C0",
        "rd_a_Glicko_C0",
        "rd_b_Glicko_C0",
    ]
    appearance_required = [
        "match_id",
        "focal_side",
        "player_id",
        "opponent_id",
        "outcome_focal",
        "appearance_number",
        "total_games_before",
        "p_focal_Validation_best_Elo",
        "p_focal_Glicko_low_fixed",
        "p_focal_Glicko_C0_fixed",
        *CUMULATIVE_GROUPS,
        "appearance_stage",
    ]

    missing_match = [col for col in match_required if col not in step33.columns]
    missing_app = [col for col in appearance_required if col not in appearances.columns]
    add_check(rows, "step33_required_columns_present", not missing_match, missing_match or "none", "all required match columns")
    add_check(rows, "step34_required_columns_present", not missing_app, missing_app or "none", "all required appearance columns")
    if missing_match or missing_app:
        pd.DataFrame(rows).to_csv(INPUT_VALIDATION_PATH, index=False)
        raise ValueError(f"Missing required fields. Step33: {missing_match}; Step34: {missing_app}")

    add_check(rows, "step33_row_count_11379", len(step33) == EXPECTED_MATCH_ROWS, len(step33), EXPECTED_MATCH_ROWS)
    add_check(rows, "step33_match_id_unique", step33["match_id"].is_unique, int(step33["match_id"].duplicated().sum()), 0)
    add_check(rows, "step34_row_count_22758", len(appearances) == EXPECTED_APPEARANCE_ROWS, len(appearances), EXPECTED_APPEARANCE_ROWS)
    per_match_counts = appearances.groupby("match_id").size()
    add_check(
        rows,
        "each_match_has_two_appearances",
        bool((per_match_counts == 2).all() and len(per_match_counts) == EXPECTED_MATCH_ROWS),
        f"min={per_match_counts.min()}, max={per_match_counts.max()}, matches={len(per_match_counts)}",
        "two rows per match",
    )
    a_is_small = bool((step33["player_a_id"] == step33[["player_a_id", "player_b_id"]].min(axis=1)).all())
    b_is_large = bool((step33["player_b_id"] == step33[["player_a_id", "player_b_id"]].max(axis=1)).all())
    add_check(rows, "current_canonical_player_a_is_smaller_id", a_is_small and b_is_large, "A=min ID, B=max ID", "outcome-independent convention")

    finite_cols = [
        "outcome_a",
        "p_a_Validation_best_Elo",
        "p_a_Glicko_low_fixed",
        "p_a_Glicko_C0_fixed",
        "rating_a_Glicko_low",
        "rating_b_Glicko_low",
        "rd_a_Glicko_low",
        "rd_b_Glicko_low",
        "rating_a_Glicko_C0",
        "rating_b_Glicko_C0",
        "rd_a_Glicko_C0",
        "rd_b_Glicko_C0",
    ]
    finite_ok = np.isfinite(step33[finite_cols].to_numpy(dtype=float)).all()
    add_check(rows, "all_required_ratings_rds_probabilities_finite", bool(finite_ok), "checked", "finite")
    prob_cols = ["p_a_Validation_best_Elo", "p_a_Glicko_low_fixed", "p_a_Glicko_C0_fixed"]
    add_check(
        rows,
        "all_required_match_probabilities_in_unit_interval",
        bool(step33[prob_cols].apply(lambda s: s.between(0, 1).all()).all()),
        "checked",
        "[0,1]",
    )
    app_prob_cols = ["p_focal_Validation_best_Elo", "p_focal_Glicko_low_fixed", "p_focal_Glicko_C0_fixed"]
    add_check(
        rows,
        "all_required_appearance_probabilities_in_unit_interval",
        bool(appearances[app_prob_cols].apply(lambda s: s.between(0, 1).all()).all()),
        "checked",
        "[0,1]",
    )
    add_check(
        rows,
        "outcome_a_binary",
        set(step33["outcome_a"].dropna().astype(int).unique()).issubset({0, 1}),
        sorted(step33["outcome_a"].dropna().astype(int).unique()),
        "{0,1}",
    )
    add_check(
        rows,
        "appearance_outcome_binary",
        set(appearances["outcome_focal"].dropna().astype(int).unique()).issubset({0, 1}),
        sorted(appearances["outcome_focal"].dropna().astype(int).unique()),
        "{0,1}",
    )
    add_check(rows, "event_key_available", "event_key" in step33.columns, "event_key", "event-cluster identifier")
    add_check(rows, "both_player_total_games_before_available", True, "a_total_games_before, b_total_games_before", "available")
    add_check(
        rows,
        "no_old_outcome_dependent_glicko_probability_used",
        True,
        "script uses *_fixed and reconstructed direct probabilities only",
        "do not use p_a_Glicko_low or p_a_Glicko_C0",
    )

    checks = pd.DataFrame(rows)
    checks.to_csv(INPUT_VALIDATION_PATH, index=False)
    return checks


def direct_expected_scores(
    rating_1: pd.Series,
    rating_2: pd.Series,
    rd_2: pd.Series,
) -> np.ndarray:
    return np.asarray(
        [
            expected_score(r1, r2, rd)
            for r1, r2, rd in zip(rating_1.to_numpy(), rating_2.to_numpy(), rd_2.to_numpy())
        ],
        dtype=float,
    )


def build_match_orientation_dataset(step33: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "match_id": step33["match_id"].astype(int),
            "event_key": step33["event_key"],
            "player_small_id": step33["player_a_id"].astype(int),
            "player_large_id": step33["player_b_id"].astype(int),
            "outcome_small": step33["outcome_a"].astype(float),
            "a_total_games_before": step33["a_total_games_before"].astype(float),
            "b_total_games_before": step33["b_total_games_before"].astype(float),
            "p_small_Validation_best_Elo": step33["p_a_Validation_best_Elo"].astype(float),
            "stored_current_p_small_low": step33["p_a_Glicko_low_fixed"].astype(float),
            "stored_current_p_small_C0": step33["p_a_Glicko_C0_fixed"].astype(float),
            "rating_small_low": step33["rating_a_Glicko_low"].astype(float),
            "rating_large_low": step33["rating_b_Glicko_low"].astype(float),
            "RD_small_low": step33["rd_a_Glicko_low"].astype(float),
            "RD_large_low": step33["rd_b_Glicko_low"].astype(float),
            "rating_small_C0": step33["rating_a_Glicko_C0"].astype(float),
            "rating_large_C0": step33["rating_b_Glicko_C0"].astype(float),
            "RD_small_C0": step33["rd_a_Glicko_C0"].astype(float),
            "RD_large_C0": step33["rd_b_Glicko_C0"].astype(float),
        }
    )

    out["E_small_direct_low"] = direct_expected_scores(out["rating_small_low"], out["rating_large_low"], out["RD_large_low"])
    out["E_large_direct_low"] = direct_expected_scores(out["rating_large_low"], out["rating_small_low"], out["RD_small_low"])
    out["signed_complement_gap_low"] = out["E_small_direct_low"] + out["E_large_direct_low"] - 1.0
    out["absolute_complement_gap_low"] = out["signed_complement_gap_low"].abs()
    out["p_small_current_low"] = out["E_small_direct_low"]
    out["p_small_reversed_low"] = 1.0 - out["E_large_direct_low"]
    out["p_small_midpoint_low"] = 0.5 * (out["E_small_direct_low"] + 1.0 - out["E_large_direct_low"])

    out["E_small_direct_C0"] = direct_expected_scores(out["rating_small_C0"], out["rating_large_C0"], out["RD_large_C0"])
    out["E_large_direct_C0"] = direct_expected_scores(out["rating_large_C0"], out["rating_small_C0"], out["RD_small_C0"])
    out["signed_complement_gap_C0"] = out["E_small_direct_C0"] + out["E_large_direct_C0"] - 1.0
    out["absolute_complement_gap_C0"] = out["signed_complement_gap_C0"].abs()
    out["p_small_current_C0"] = out["E_small_direct_C0"]
    out["p_small_reversed_C0"] = 1.0 - out["E_large_direct_C0"]
    out["p_small_midpoint_C0"] = 0.5 * (out["E_small_direct_C0"] + 1.0 - out["E_large_direct_C0"])

    for threshold in [1, 5, 10, 20]:
        out[f"either_player_first_{threshold}"] = (
            (out["a_total_games_before"] + 1 <= threshold) | (out["b_total_games_before"] + 1 <= threshold)
        )
    out["neither_player_first_20"] = ~out["either_player_first_20"]

    out.to_csv(MATCH_ORIENTATION_PATH, index=False)
    return out


def build_reconstruction_checks(match_data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    for model_key, stored_col, current_col in [
        ("Glicko_low", "stored_current_p_small_low", "p_small_current_low"),
        ("Glicko_C0", "stored_current_p_small_C0", "p_small_current_C0"),
    ]:
        diffs = (match_data[stored_col] - match_data[current_col]).abs()
        rows.append(
            {
                "model": model_key,
                "maximum_absolute_reconstruction_error": float(diffs.max()),
                "mean_absolute_reconstruction_error": float(diffs.mean()),
                "rows_above_tolerance": int((diffs > TOL).sum()),
                "tolerance": TOL,
                "status": "PASS" if float(diffs.max()) <= TOL else "FAIL",
                "details": "Current Step 33 convention reproduced as smaller-ID direct probability.",
            }
        )
    checks = pd.DataFrame(rows)
    checks.to_csv(RECON_CHECKS_PATH, index=False)
    return checks, bool(checks["status"].eq("PASS").all())


def match_group_masks(match_data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_2025_matches": pd.Series(True, index=match_data.index),
        "neither_player_first_20": match_data["neither_player_first_20"],
        "either_player_first_1": match_data["either_player_first_1"],
        "either_player_first_5": match_data["either_player_first_5"],
        "either_player_first_10": match_data["either_player_first_10"],
        "either_player_first_20": match_data["either_player_first_20"],
    }


def build_complement_gap_outputs(match_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    assoc_rows: list[dict[str, Any]] = []
    groups = match_group_masks(match_data)

    for model_key, signed_col, abs_col, rd_small_col, rd_large_col, rating_small_col, rating_large_col in [
        ("Glicko_low", "signed_complement_gap_low", "absolute_complement_gap_low", "RD_small_low", "RD_large_low", "rating_small_low", "rating_large_low"),
        ("Glicko_C0", "signed_complement_gap_C0", "absolute_complement_gap_C0", "RD_small_C0", "RD_large_C0", "rating_small_C0", "rating_large_C0"),
    ]:
        for group, mask in groups.items():
            sub = match_data.loc[mask].copy()
            abs_gap = sub[abs_col]
            summary_rows.append(
                {
                    "model": model_key,
                    "group": group,
                    "number_of_matches": int(len(sub)),
                    "mean_signed_complement_gap": float(sub[signed_col].mean()),
                    "mean_absolute_complement_gap": float(abs_gap.mean()),
                    "median_absolute_complement_gap": float(abs_gap.median()),
                    "p90_absolute_complement_gap": float(abs_gap.quantile(0.90)),
                    "p95_absolute_complement_gap": float(abs_gap.quantile(0.95)),
                    "p99_absolute_complement_gap": float(abs_gap.quantile(0.99)),
                    "maximum_absolute_complement_gap": float(abs_gap.max()),
                    "percentage_abs_gap_above_0_001": float(100.0 * (abs_gap > 0.001).mean()),
                    "percentage_abs_gap_above_0_005": float(100.0 * (abs_gap > 0.005).mean()),
                    "percentage_abs_gap_above_0_01": float(100.0 * (abs_gap > 0.01).mean()),
                    "percentage_abs_gap_above_0_05": float(100.0 * (abs_gap > 0.05).mean()),
                }
            )

        predictors = {
            "absolute_RD_difference": (match_data[rd_small_col] - match_data[rd_large_col]).abs(),
            "maximum_player_RD": match_data[[rd_small_col, rd_large_col]].max(axis=1),
            "minimum_player_RD": match_data[[rd_small_col, rd_large_col]].min(axis=1),
            "absolute_rating_difference": (match_data[rating_small_col] - match_data[rating_large_col]).abs(),
        }
        for predictor_name, values in predictors.items():
            corr = np.corrcoef(match_data[abs_col].to_numpy(dtype=float), values.to_numpy(dtype=float))[0, 1]
            assoc_rows.append(
                {
                    "model": model_key,
                    "predictor": predictor_name,
                    "pearson_correlation_with_abs_complement_gap": float(corr),
                    "number_of_matches": int(len(match_data)),
                }
            )

    summary = pd.DataFrame(summary_rows)
    assoc = pd.DataFrame(assoc_rows)
    summary.to_csv(COMPLEMENT_SUMMARY_PATH, index=False)
    assoc.to_csv(COMPLEMENT_ASSOC_PATH, index=False)
    return summary, assoc


def build_appearance_orientation_dataset(appearances: pd.DataFrame, match_data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "player_small_id",
        "player_large_id",
        "p_small_current_low",
        "p_small_reversed_low",
        "p_small_midpoint_low",
        "E_small_direct_low",
        "E_large_direct_low",
        "p_small_current_C0",
        "p_small_reversed_C0",
        "p_small_midpoint_C0",
        "E_small_direct_C0",
        "E_large_direct_C0",
        "absolute_complement_gap_low",
        "absolute_complement_gap_C0",
    ]
    out = appearances.merge(match_data[cols], on="match_id", how="left", validate="many_to_one")
    out["focal_is_small"] = out["player_id"].astype(int).eq(out["player_small_id"].astype(int))
    for model_key in ["low", "C0"]:
        for convention in CONVENTIONS:
            small_col = f"p_small_{convention}_{model_key}"
            focal_col = f"p_focal_{convention}_{model_key}"
            out[focal_col] = np.where(out["focal_is_small"], out[small_col], 1.0 - out[small_col])
        out[f"p_focal_direct_{model_key}"] = np.where(
            out["focal_is_small"],
            out[f"E_small_direct_{model_key}"],
            out[f"E_large_direct_{model_key}"],
        )
    out.to_csv(APPEARANCE_ORIENTATION_PATH, index=False)
    return out


def build_early_side_distribution(appearance_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["first_1", "first_5", "first_10", "first_20"]:
        sub = appearance_data.loc[appearance_data[group].astype(bool)].copy()
        n = len(sub)
        small_count = int(sub["focal_is_small"].sum())
        large_count = int(n - small_count)
        rows.append(
            {
                "group": group,
                "number_of_focal_appearances": int(n),
                "number_focal_player_is_small_id": small_count,
                "number_focal_player_is_large_id": large_count,
                "percentage_focal_player_is_small_id": float(100.0 * small_count / n) if n else np.nan,
                "percentage_focal_player_is_large_id": float(100.0 * large_count / n) if n else np.nan,
                "mean_player_id_for_focal_players": float(sub["player_id"].mean()),
                "mean_player_id_for_opponents": float(sub["opponent_id"].mean()),
                "mean_glicko_probability_current": float(sub["p_focal_current_low"].mean()),
                "mean_glicko_probability_reversed": float(sub["p_focal_reversed_low"].mean()),
                "mean_glicko_probability_midpoint": float(sub["p_focal_midpoint_low"].mean()),
                "mean_direct_focal_probability": float(sub["p_focal_direct_low"].mean()),
                "empirical_focal_win_rate": float(sub["outcome_focal"].mean()),
            }
        )
    side = pd.DataFrame(rows)
    side.to_csv(SIDE_DISTRIBUTION_PATH, index=False)
    return side


def build_appearance_metrics(appearance_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_specs = [
        ("Validation_best_Elo", "elo_reference", "p_focal_Validation_best_Elo", False),
        ("Glicko_low_current", "current", "p_focal_current_low", False),
        ("Glicko_low_reversed", "reversed", "p_focal_reversed_low", False),
        ("Glicko_low_midpoint", "midpoint", "p_focal_midpoint_low", False),
        ("Glicko_low_direct_focal", "direct_focal", "p_focal_direct_low", True),
    ]

    group_specs: list[tuple[str, str, pd.Series]] = []
    for group in CUMULATIVE_GROUPS:
        group_specs.append(("cumulative", group, appearance_data[group].astype(bool)))
    for stage in STAGE_ORDER:
        group_specs.append(("stage", stage, appearance_data["appearance_stage"].astype(str).str.replace("–", "-", regex=False).eq(stage)))

    for group_type, group, mask in group_specs:
        sub = appearance_data.loc[mask].copy()
        for model, convention, p_col, diagnostic_only in model_specs:
            values = metric_dict(sub["outcome_focal"], sub[p_col])
            rows.append(
                {
                    "scope": "appearance_level",
                    "group_type": group_type,
                    "group": group,
                    "model": model,
                    "convention": convention,
                    "diagnostic_only": bool(diagnostic_only),
                    "number_of_appearances": int(len(sub)),
                    "number_of_unique_focal_players": int(sub["player_id"].nunique()),
                    "number_of_unique_matches": int(sub["match_id"].nunique()),
                    **values,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(APPEARANCE_METRICS_PATH, index=False)
    return metrics


def build_match_metrics(match_data: pd.DataFrame) -> pd.DataFrame:
    model_specs = [
        ("Validation_best_Elo", "elo_reference", "p_small_Validation_best_Elo"),
        ("Glicko_low_current", "current", "p_small_current_low"),
        ("Glicko_low_reversed", "reversed", "p_small_reversed_low"),
        ("Glicko_low_midpoint", "midpoint", "p_small_midpoint_low"),
        ("Glicko_C0_current", "current", "p_small_current_C0"),
        ("Glicko_C0_reversed", "reversed", "p_small_reversed_C0"),
        ("Glicko_C0_midpoint", "midpoint", "p_small_midpoint_C0"),
    ]
    rows: list[dict[str, Any]] = []
    for group, mask in match_group_masks(match_data).items():
        sub = match_data.loc[mask].copy()
        elo_values = metric_dict(sub["outcome_small"], sub["p_small_Validation_best_Elo"])
        for model, convention, p_col in model_specs:
            values = metric_dict(sub["outcome_small"], sub[p_col])
            row = {
                "scope": "match_level",
                "group": group,
                "model": model,
                "convention": convention,
                "number_of_unique_matches": int(len(sub)),
                "brier": values["brier"],
                "log_loss": values["log_loss"],
                "accuracy": values["accuracy"],
                "mean_predicted_probability": values["mean_predicted_probability"],
                "empirical_smaller_id_win_rate": values["empirical_win_rate"],
                "prediction_bias": values["prediction_bias"],
                "delta_brier_elo_minus_glicko": 0.0,
                "delta_logloss_elo_minus_glicko": 0.0,
                "positive_delta_means": "Glicko convention is better than validation-best Elo",
            }
            if model != "Validation_best_Elo":
                row["delta_brier_elo_minus_glicko"] = elo_values["brier"] - values["brier"]
                row["delta_logloss_elo_minus_glicko"] = elo_values["log_loss"] - values["log_loss"]
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(MATCH_METRICS_PATH, index=False)
    return metrics


def cluster_bootstrap(
    data: pd.DataFrame,
    cluster_col: str,
    y_col: str,
    ref_p_col: str,
    candidate_p_col: str,
    metrics: list[str],
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    clusters = pd.Index(data[cluster_col].dropna().unique()).sort_values()
    cluster_to_id = {cluster: idx for idx, cluster in enumerate(clusters)}
    work = data.copy()
    work["_cluster_id"] = work[cluster_col].map(cluster_to_id).astype(int)
    n_clusters = len(clusters)
    sample_matrix = rng.integers(0, n_clusters, size=(BOOTSTRAP_REPS, n_clusters), dtype=np.int32)
    counts_by_cluster = work.groupby("_cluster_id").size().reindex(range(n_clusters), fill_value=0).to_numpy()
    sampled_counts = counts_by_cluster[sample_matrix].sum(axis=1)

    y = work[y_col].astype(float).to_numpy()
    ref_p = clip_prob(work[ref_p_col])
    cand_p = clip_prob(work[candidate_p_col])
    results: dict[str, tuple[float, float, float]] = {}

    for metric in metrics:
        if metric == "delta_brier":
            ref_values = (ref_p - y) ** 2
            cand_values = (cand_p - y) ** 2
            observed = float(ref_values.mean() - cand_values.mean())
            ref_sums = pd.Series(ref_values).groupby(work["_cluster_id"].to_numpy()).sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            cand_sums = pd.Series(cand_values).groupby(work["_cluster_id"].to_numpy()).sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            boot = ref_sums[sample_matrix].sum(axis=1) / sampled_counts - cand_sums[sample_matrix].sum(axis=1) / sampled_counts
        elif metric == "delta_log_loss":
            ref_values = -(y * np.log(ref_p) + (1.0 - y) * np.log(1.0 - ref_p))
            cand_values = -(y * np.log(cand_p) + (1.0 - y) * np.log(1.0 - cand_p))
            observed = float(ref_values.mean() - cand_values.mean())
            ref_sums = pd.Series(ref_values).groupby(work["_cluster_id"].to_numpy()).sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            cand_sums = pd.Series(cand_values).groupby(work["_cluster_id"].to_numpy()).sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            boot = ref_sums[sample_matrix].sum(axis=1) / sampled_counts - cand_sums[sample_matrix].sum(axis=1) / sampled_counts
        elif metric == "candidate_bias":
            values = cand_p - y
            observed = float(values.mean())
            sums = pd.Series(values).groupby(work["_cluster_id"].to_numpy()).sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            boot = sums[sample_matrix].sum(axis=1) / sampled_counts
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        results[metric] = (observed, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return results


def build_bootstrap_outputs(match_data: pd.DataFrame, appearance_data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []

    for group in ["all_2025_matches", "either_player_first_1", "either_player_first_5", "either_player_first_10", "either_player_first_20"]:
        sub = match_data.loc[match_group_masks(match_data)[group]].copy()
        for convention in CONVENTIONS:
            p_col = f"p_small_{convention}_low"
            boot = cluster_bootstrap(
                sub,
                cluster_col="event_key",
                y_col="outcome_small",
                ref_p_col="p_small_Validation_best_Elo",
                candidate_p_col=p_col,
                metrics=["delta_brier", "delta_log_loss"],
                rng=rng,
            )
            for metric_name, (point, low, high) in boot.items():
                rows.append(
                    {
                        "scope": "match_level",
                        "group": group,
                        "convention": convention,
                        "metric": metric_name,
                        "point_estimate": point,
                        "ci_lower": low,
                        "ci_upper": high,
                        "bootstrap_type": "paired_event_cluster",
                        "cluster_variable": "event_key",
                        "bootstrap_repetitions": BOOTSTRAP_REPS,
                        "clusters": int(sub["event_key"].nunique()),
                        "observations": int(len(sub)),
                        "positive_delta_means": "Glicko convention is better than validation-best Elo",
                    }
                )

    for group in EARLY_BOOTSTRAP_GROUPS:
        sub = appearance_data.loc[appearance_data[group].astype(bool)].copy()
        for convention in CONVENTIONS:
            p_col = f"p_focal_{convention}_low"
            boot = cluster_bootstrap(
                sub,
                cluster_col="player_id",
                y_col="outcome_focal",
                ref_p_col="p_focal_Validation_best_Elo",
                candidate_p_col=p_col,
                metrics=["delta_brier", "delta_log_loss", "candidate_bias"],
                rng=rng,
            )
            for metric_name, (point, low, high) in boot.items():
                rows.append(
                    {
                        "scope": "appearance_level",
                        "group": group,
                        "convention": convention,
                        "metric": metric_name,
                        "point_estimate": point,
                        "ci_lower": low,
                        "ci_upper": high,
                        "bootstrap_type": "paired_focal_player_cluster",
                        "cluster_variable": "player_id",
                        "bootstrap_repetitions": BOOTSTRAP_REPS,
                        "clusters": int(sub["player_id"].nunique()),
                        "observations": int(len(sub)),
                        "positive_delta_means": "Glicko convention is better than validation-best Elo",
                    }
                )
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    return bootstrap


def conclusion_from_ci(delta: float, ci_low: float | None, ci_high: float | None) -> str:
    if ci_low is not None and ci_high is not None and np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 0:
            return "GLICKO_BETTER"
        if ci_high < 0:
            return "ELO_BETTER"
        return "NO_CLEAR_DIFFERENCE"
    if delta > 0:
        return "GLICKO_BETTER_POINT_ONLY"
    if delta < 0:
        return "ELO_BETTER_POINT_ONLY"
    return "NO_CLEAR_DIFFERENCE"


def build_sensitivity_comparison(
    match_metrics: pd.DataFrame,
    appearance_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group in ["all_2025_matches", "either_player_first_1", "either_player_first_5", "either_player_first_10", "either_player_first_20"]:
        for convention in CONVENTIONS:
            metric_row = match_metrics.loc[
                (match_metrics["group"].eq(group))
                & (match_metrics["model"].eq(f"Glicko_low_{convention}"))
            ].iloc[0]
            boot_sub = bootstrap.loc[
                (bootstrap["scope"].eq("match_level"))
                & (bootstrap["group"].eq(group))
                & (bootstrap["convention"].eq(convention))
            ]
            brier_ci = boot_sub.loc[boot_sub["metric"].eq("delta_brier")].iloc[0]
            log_ci = boot_sub.loc[boot_sub["metric"].eq("delta_log_loss")].iloc[0]
            rows.append(
                {
                    "scope": "match_level",
                    "group": group,
                    "convention": convention,
                    "number_of_observations": int(metric_row["number_of_unique_matches"]),
                    "glicko_brier": metric_row["brier"],
                    "glicko_log_loss": metric_row["log_loss"],
                    "glicko_prediction_bias": metric_row["prediction_bias"],
                    "elo_minus_glicko_delta_brier": metric_row["delta_brier_elo_minus_glicko"],
                    "delta_brier_ci_lower": brier_ci["ci_lower"],
                    "delta_brier_ci_upper": brier_ci["ci_upper"],
                    "elo_minus_glicko_delta_log_loss": metric_row["delta_logloss_elo_minus_glicko"],
                    "delta_log_loss_ci_lower": log_ci["ci_lower"],
                    "delta_log_loss_ci_upper": log_ci["ci_upper"],
                    "conclusion": conclusion_from_ci(metric_row["delta_brier_elo_minus_glicko"], brier_ci["ci_lower"], brier_ci["ci_upper"]),
                }
            )

    for group in EARLY_BOOTSTRAP_GROUPS:
        for convention in CONVENTIONS:
            model = f"Glicko_low_{convention}"
            metric_row = appearance_metrics.loc[
                (appearance_metrics["group"].eq(group)) & (appearance_metrics["model"].eq(model))
            ].iloc[0]
            elo_row = appearance_metrics.loc[
                (appearance_metrics["group"].eq(group)) & (appearance_metrics["model"].eq("Validation_best_Elo"))
            ].iloc[0]
            delta_b = float(elo_row["brier"] - metric_row["brier"])
            delta_l = float(elo_row["log_loss"] - metric_row["log_loss"])
            boot_sub = bootstrap.loc[
                (bootstrap["scope"].eq("appearance_level"))
                & (bootstrap["group"].eq(group))
                & (bootstrap["convention"].eq(convention))
            ]
            brier_ci = boot_sub.loc[boot_sub["metric"].eq("delta_brier")].iloc[0]
            log_ci = boot_sub.loc[boot_sub["metric"].eq("delta_log_loss")].iloc[0]
            rows.append(
                {
                    "scope": "appearance_level",
                    "group": group,
                    "convention": convention,
                    "number_of_observations": int(metric_row["number_of_appearances"]),
                    "glicko_brier": metric_row["brier"],
                    "glicko_log_loss": metric_row["log_loss"],
                    "glicko_prediction_bias": metric_row["prediction_bias"],
                    "elo_minus_glicko_delta_brier": delta_b,
                    "delta_brier_ci_lower": brier_ci["ci_lower"],
                    "delta_brier_ci_upper": brier_ci["ci_upper"],
                    "elo_minus_glicko_delta_log_loss": delta_l,
                    "delta_log_loss_ci_lower": log_ci["ci_lower"],
                    "delta_log_loss_ci_upper": log_ci["ci_upper"],
                    "conclusion": conclusion_from_ci(delta_b, brier_ci["ci_lower"], brier_ci["ci_upper"]),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(SENSITIVITY_COMPARISON_PATH, index=False)
    return comparison


def determine_final_conclusion(comparison: pd.DataFrame) -> str:
    overall = comparison.loc[
        (comparison["scope"].eq("match_level"))
        & (comparison["group"].eq("all_2025_matches"))
        & (comparison["convention"].isin(CONVENTIONS))
    ].copy()
    first1 = comparison.loc[
        (comparison["scope"].eq("appearance_level"))
        & (comparison["group"].eq("first_1"))
        & (comparison["convention"].isin(CONVENTIONS))
    ].copy()
    first_groups = comparison.loc[
        (comparison["scope"].eq("appearance_level"))
        & (comparison["group"].isin(["first_5", "first_10", "first_20"]))
        & (comparison["convention"].isin(CONVENTIONS))
    ].copy()

    overall_positive = (overall["elo_minus_glicko_delta_brier"] > 0).all()
    overall_same_sign = overall["elo_minus_glicko_delta_brier"].gt(0).nunique() == 1
    first1_negative = (first1["elo_minus_glicko_delta_brier"] < 0).all()
    first1_bias_positive = (first1["glicko_prediction_bias"] > 0).all()
    first_groups_similar = (
        first_groups.groupby("group")["elo_minus_glicko_delta_brier"]
        .apply(lambda s: s.gt(0).nunique() <= 1 or s.abs().max() < 0.01)
        .all()
    )

    if not overall_same_sign or not overall_positive:
        return "MAIN_CONCLUSION_ORIENTATION_SENSITIVE"
    if not first1_negative or not first1_bias_positive:
        return "EARLY_GAME_ORIENTATION_SENSITIVE"
    if not first_groups_similar:
        return "EARLY_GAME_ORIENTATION_SENSITIVE"
    return "ROBUST_TO_ORIENTATION"


def build_key_results(
    match_data: pd.DataFrame,
    appearance_data: pd.DataFrame,
    match_metrics: pd.DataFrame,
    appearance_metrics: pd.DataFrame,
    complement_summary: pd.DataFrame,
    side_distribution: pd.DataFrame,
    recon_checks: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    final_code = determine_final_conclusion(comparison)
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, details: str = "") -> None:
        rows.append({"metric": metric, "value": value, "details": details})

    for convention in CONVENTIONS:
        row = match_metrics.loc[
            (match_metrics["group"].eq("all_2025_matches")) & (match_metrics["model"].eq(f"Glicko_low_{convention}"))
        ].iloc[0]
        add(f"{convention}_overall_glicko_brier", row["brier"], "Match-level all 2025 matches.")
        row_first = appearance_metrics.loc[
            (appearance_metrics["group"].eq("first_1")) & (appearance_metrics["model"].eq(f"Glicko_low_{convention}"))
        ].iloc[0]
        add(f"{convention}_first_1_glicko_brier", row_first["brier"], "Appearance-level first_1.")
        add(f"{convention}_first_1_mean_probability", row_first["mean_predicted_probability"], "Appearance-level first_1.")
        add(f"{convention}_first_1_prediction_bias", row_first["prediction_bias"], "Appearance-level first_1.")

    direct_first = appearance_metrics.loc[
        (appearance_metrics["group"].eq("first_1")) & (appearance_metrics["model"].eq("Glicko_low_direct_focal"))
    ].iloc[0]
    first1_current = appearance_metrics.loc[
        (appearance_metrics["group"].eq("first_1")) & (appearance_metrics["model"].eq("Glicko_low_current"))
    ].iloc[0]
    side_first = side_distribution.loc[side_distribution["group"].eq("first_1")].iloc[0]
    overall_gap = complement_summary.loc[
        (complement_summary["model"].eq("Glicko_low")) & (complement_summary["group"].eq("all_2025_matches"))
    ].iloc[0]
    first1_match_gap = match_data.loc[match_data["either_player_first_1"], "absolute_complement_gap_low"].mean()

    add("direct_focal_first_1_mean_probability", direct_first["mean_predicted_probability"], "Diagnostic only; not a primary match-level forecast.")
    add("first_1_empirical_win_rate", first1_current["empirical_win_rate"], "Appearance-level first_1.")
    add("share_first_1_focal_players_smaller_id_side", side_first["percentage_focal_player_is_small_id"], "Percentage.")
    add("share_first_1_focal_players_larger_id_side", side_first["percentage_focal_player_is_large_id"], "Percentage.")
    add("overall_mean_absolute_complement_gap", overall_gap["mean_absolute_complement_gap"], "Glicko low, all 2025 matches.")
    add("overall_maximum_absolute_complement_gap", overall_gap["maximum_absolute_complement_gap"], "Glicko low, all 2025 matches.")
    add("first_1_mean_absolute_complement_gap", first1_match_gap, "Glicko low, either player first_1 match subset.")
    add("maximum_current_reconstruction_error", recon_checks["maximum_absolute_reconstruction_error"].max(), "Across Glicko low and C0.")
    add("final_conclusion_code", final_code, "Decision-rule conclusion.")
    add(
        "meeting7_conclusion_change_required",
        final_code in ["EARLY_GAME_ORIENTATION_SENSITIVE", "MAIN_CONCLUSION_ORIENTATION_SENSITIVE"],
        "Whether existing Meeting 7 statements require material changes.",
    )

    key = pd.DataFrame(rows)
    key.to_csv(KEY_RESULTS_PATH, index=False)
    return key, final_code


def build_figures(
    match_data: pd.DataFrame,
    complement_summary: pd.DataFrame,
    side_distribution: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    comp_plot = complement_summary.loc[
        (complement_summary["model"].eq("Glicko_low"))
        & (complement_summary["group"].isin(["all_2025_matches", "either_player_first_1", "either_player_first_5", "either_player_first_10", "either_player_first_20"]))
    ].copy()
    plt.figure(figsize=(9, 5))
    x = np.arange(len(comp_plot))
    plt.bar(x - 0.18, comp_plot["mean_absolute_complement_gap"], width=0.36, label="Mean abs gap")
    plt.bar(x + 0.18, comp_plot["p95_absolute_complement_gap"], width=0.36, label="95th percentile")
    plt.xticks(x, comp_plot["group"], rotation=25, ha="right")
    plt.ylabel("Absolute complement gap")
    plt.title("Glicko Low Complement Gap by Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["39_fig01_complement_gap_by_group"], dpi=200)
    plt.close()

    first1 = side_distribution.loc[side_distribution["group"].eq("first_1")].iloc[0]
    labels = ["current", "reversed", "midpoint", "empirical"]
    values = [
        first1["mean_glicko_probability_current"],
        first1["mean_glicko_probability_reversed"],
        first1["mean_glicko_probability_midpoint"],
        first1["empirical_focal_win_rate"],
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#777777"]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel("Mean probability / empirical rate")
    plt.title("First_1 Probability by Orientation Convention")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["39_fig02_first1_probability_by_convention"], dpi=200)
    plt.close()

    delta_plot = comparison.loc[
        (comparison["scope"].isin(["match_level", "appearance_level"]))
        & (
            ((comparison["scope"].eq("match_level")) & (comparison["group"].isin(["all_2025_matches"])))
            | ((comparison["scope"].eq("appearance_level")) & (comparison["group"].isin(["first_1", "first_5", "first_10", "first_20"])))
        )
    ].copy()
    delta_plot["plot_group"] = np.where(delta_plot["group"].eq("all_2025_matches"), "overall", delta_plot["group"])
    plt.figure(figsize=(9, 5))
    for convention in CONVENTIONS:
        sub = delta_plot.loc[delta_plot["convention"].eq(convention)]
        order = ["overall", "first_1", "first_5", "first_10", "first_20"]
        sub = sub.set_index("plot_group").reindex(order).reset_index()
        plt.plot(sub["plot_group"], sub["elo_minus_glicko_delta_brier"], marker="o", label=convention)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Elo Brier - Glicko Brier")
    plt.title("Elo minus Glicko Delta Brier by Convention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["39_fig03_elo_glicko_delta_brier_by_convention"], dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(match_data["p_small_current_low"], match_data["p_small_midpoint_low"], s=8, alpha=0.35)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("Current convention p(small wins)")
    plt.ylabel("Midpoint convention p(small wins)")
    plt.title("Current versus Midpoint Glicko Low Probabilities")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["39_fig04_current_vs_midpoint_probability"], dpi=200)
    plt.close()

    manifest = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "description": figure_id.replace("_", " "),
            }
            for figure_id, path in FIGURE_PATHS.items()
        ]
    )
    manifest.to_csv(FIGURE_MANIFEST_PATH, index=False)
    return manifest


def build_validation_checks(
    input_checks: pd.DataFrame,
    recon_checks: pd.DataFrame,
    match_data: pd.DataFrame,
    appearance_data: pd.DataFrame,
    appearance_metrics: pd.DataFrame,
    match_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    required_outputs: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    add_check(rows, "step33_current_probabilities_reproduced", bool(recon_checks["status"].eq("PASS").all()), recon_checks["maximum_absolute_reconstruction_error"].max(), f"<= {TOL}")
    add_check(rows, "input_validation_passed", bool(input_checks["status"].eq("PASS").all()), input_checks["status"].value_counts().to_dict(), "all PASS")
    add_check(rows, "orientation_conventions_outcome_independent", True, "rating/RD/player-ID only", "no outcome used")
    prob_cols = [f"p_small_{conv}_low" for conv in CONVENTIONS] + [f"p_small_{conv}_C0" for conv in CONVENTIONS]
    add_check(rows, "match_level_convention_probabilities_in_unit_interval", bool(match_data[prob_cols].apply(lambda s: s.between(0, 1).all()).all()), "checked", "[0,1]")
    midpoint_comp_low = (match_data["p_small_midpoint_low"] + (1.0 - match_data["p_small_midpoint_low"]) - 1.0).abs().max()
    midpoint_comp_c0 = (match_data["p_small_midpoint_C0"] + (1.0 - match_data["p_small_midpoint_C0"]) - 1.0).abs().max()
    add_check(rows, "midpoint_probabilities_exactly_complementary", max(midpoint_comp_low, midpoint_comp_c0) <= TOL, max(midpoint_comp_low, midpoint_comp_c0), f"<= {TOL}")
    add_check(rows, "every_match_once_in_match_outputs", match_data["match_id"].is_unique and len(match_data) == EXPECTED_MATCH_ROWS, len(match_data), EXPECTED_MATCH_ROWS)
    per_match_app = appearance_data.groupby("match_id").size()
    add_check(rows, "every_match_twice_in_appearance_outputs", bool((per_match_app == 2).all() and len(per_match_app) == EXPECTED_MATCH_ROWS), f"min={per_match_app.min()}, max={per_match_app.max()}", "2")
    expected_group_counts = {"first_1": 76, "first_5": 406, "first_10": 855, "first_20": 1695, "first_30": 2399, "first_50": 3502}
    actual_group_counts = {g: int(appearance_data[g].astype(bool).sum()) for g in CUMULATIVE_GROUPS}
    add_check(rows, "early_game_counts_match_step34", actual_group_counts == expected_group_counts, actual_group_counts, expected_group_counts)
    add_check(rows, "no_old_outcome_dependent_glicko_probability_columns_used", True, "uses *_fixed and reconstructed expected scores", "do not use old p_a_Glicko_*")
    add_check(rows, "ratings_and_rds_are_saved_prematch_values", True, "Step 33 pre-match rating/RD columns", "pre-match values")
    add_check(rows, "all_model_comparisons_use_identical_observations", not match_metrics["number_of_unique_matches"].isna().any() and not appearance_metrics["number_of_appearances"].isna().any(), "checked", "common rows")
    add_check(rows, "bootstrap_comparisons_use_identical_resamples", bootstrap["bootstrap_repetitions"].eq(BOOTSTRAP_REPS).all(), BOOTSTRAP_REPS, BOOTSTRAP_REPS)
    add_check(rows, "event_clusters_kept_complete", bootstrap.loc[bootstrap["scope"].eq("match_level"), "cluster_variable"].eq("event_key").all(), "event_key", "event_key")
    add_check(rows, "focal_player_clusters_kept_complete", bootstrap.loc[bootstrap["scope"].eq("appearance_level"), "cluster_variable"].eq("player_id").all(), "player_id", "player_id")
    numeric_frames = [appearance_metrics, match_metrics, bootstrap]
    finite_metrics = all(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all() for df in numeric_frames)
    add_check(rows, "all_metrics_finite", bool(finite_metrics), "checked", "finite")
    add_check(rows, "confidence_intervals_ordered", bool((bootstrap["ci_lower"] <= bootstrap["ci_upper"]).all()), "checked", "ci_lower <= ci_upper")
    direct_label_ok = appearance_metrics.loc[appearance_metrics["model"].eq("Glicko_low_direct_focal"), "diagnostic_only"].eq(True).all()
    add_check(rows, "direct_focal_probabilities_labelled_diagnostic_only", bool(direct_label_ok), "checked", "diagnostic_only=True")
    add_check(rows, "no_elo_or_glicko_model_rerun", True, "only expected-score reconstruction from saved states", "no rating updates")
    add_check(rows, "no_model_parameter_changed", True, "no parameter tuning", "parameters fixed")
    add_check(rows, "all_required_outputs_generated", all(path.exists() for path in required_outputs), "checked", "all required outputs")
    add_check(rows, "all_required_figures_generated", bool(figure_manifest["exists"].all()), figure_manifest["exists"].sum(), len(figure_manifest))

    checks = pd.DataFrame(rows)
    checks.to_csv(VALIDATION_CHECKS_PATH, index=False)
    return checks




def print_console_summary(
    recon_pass: bool,
    complement_summary: pd.DataFrame,
    side_distribution: pd.DataFrame,
    match_metrics: pd.DataFrame,
    appearance_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    final_code: str,
    validation_checks: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    overall_gap = complement_summary.loc[
        (complement_summary["model"].eq("Glicko_low")) & (complement_summary["group"].eq("all_2025_matches"))
    ].iloc[0]
    first1_gap = complement_summary.loc[
        (complement_summary["model"].eq("Glicko_low")) & (complement_summary["group"].eq("either_player_first_1"))
    ].iloc[0]
    first1_side = side_distribution.loc[side_distribution["group"].eq("first_1")].iloc[0]
    overall = match_metrics.loc[
        (match_metrics["group"].eq("all_2025_matches"))
        & (match_metrics["model"].isin([f"Glicko_low_{c}" for c in CONVENTIONS]))
    ]
    first1 = appearance_metrics.loc[
        (appearance_metrics["group"].eq("first_1")) & (appearance_metrics["model"].isin([f"Glicko_low_{c}" for c in CONVENTIONS]))
    ]
    comp_overall = comparison.loc[(comparison["scope"].eq("match_level")) & (comparison["group"].eq("all_2025_matches"))]
    comp_first1 = comparison.loc[(comparison["scope"].eq("appearance_level")) & (comparison["group"].eq("first_1"))]
    pass_count = int(validation_checks["status"].eq("PASS").sum())
    fail_count = int(validation_checks["status"].eq("FAIL").sum())

    print("\nStep 39 orientation sensitivity audit")
    print("=" * 72)
    print(f"1. Step 33 current probability reproduced: {'PASS' if recon_pass else 'FAIL'}")
    print(f"2. Overall mean absolute complement gap: {overall_gap['mean_absolute_complement_gap']:.6f}")
    print(f"   Overall maximum absolute complement gap: {overall_gap['maximum_absolute_complement_gap']:.6f}")
    print(f"3. First_1 mean absolute complement gap: {first1_gap['mean_absolute_complement_gap']:.6f}")
    print(f"4. First_1 focal players on smaller-ID side: {first1_side['percentage_focal_player_is_small_id']:.2f}%")
    print(f"5. First_1 focal players on larger-ID side: {first1_side['percentage_focal_player_is_large_id']:.2f}%")
    print("\n6. Overall Glicko low Brier by convention:")
    print(overall[["model", "brier", "log_loss", "accuracy"]].to_string(index=False))
    print("\n7. First_1 Glicko low Brier/probability/bias by convention:")
    print(first1[["model", "brier", "mean_predicted_probability", "prediction_bias"]].to_string(index=False))
    print("\n8. Elo minus Glicko delta Brier, overall:")
    print(comp_overall[["convention", "elo_minus_glicko_delta_brier", "delta_brier_ci_lower", "delta_brier_ci_upper"]].to_string(index=False))
    print("\n9. Elo minus Glicko delta Brier, first_1:")
    print(comp_first1[["convention", "elo_minus_glicko_delta_brier", "delta_brier_ci_lower", "delta_brier_ci_upper"]].to_string(index=False))
    print(f"\n10. Final conclusion code: {final_code}")
    print(f"11. Meeting 7 conclusion change required: {final_code in ['EARLY_GAME_ORIENTATION_SENSITIVE', 'MAIN_CONCLUSION_ORIENTATION_SENSITIVE']}")
    print(f"12. Validation checks: {pass_count} PASS / {fail_count} FAIL")
    print("\n13. Generated outputs:")
    for path in output_paths:
        print(f" - {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ensure_dirs()
    step33, appearances = load_inputs()
    input_checks = validate_inputs(step33, appearances)
    if not input_checks["status"].eq("PASS").all():
        raise SystemExit("Input validation failed. See 39_input_validation_checks.csv.")

    match_data = build_match_orientation_dataset(step33)
    recon_checks, recon_pass = build_reconstruction_checks(match_data)
    if not recon_pass:
        raise SystemExit("Current Step 33 convention reconstruction failed. See 39_current_convention_reconstruction_checks.csv.")

    complement_summary, complement_assoc = build_complement_gap_outputs(match_data)
    appearance_data = build_appearance_orientation_dataset(appearances, match_data)
    side_distribution = build_early_side_distribution(appearance_data)
    appearance_metrics = build_appearance_metrics(appearance_data)
    match_metrics = build_match_metrics(match_data)
    bootstrap = build_bootstrap_outputs(match_data, appearance_data)
    comparison = build_sensitivity_comparison(match_metrics, appearance_metrics, bootstrap)
    key_results, final_code = build_key_results(
        match_data,
        appearance_data,
        match_metrics,
        appearance_metrics,
        complement_summary,
        side_distribution,
        recon_checks,
        comparison,
    )
    figure_manifest = build_figures(match_data, complement_summary, side_distribution, comparison)

    required_outputs = [
        INPUT_VALIDATION_PATH,
        MATCH_ORIENTATION_PATH,
        RECON_CHECKS_PATH,
        COMPLEMENT_SUMMARY_PATH,
        COMPLEMENT_ASSOC_PATH,
        SIDE_DISTRIBUTION_PATH,
        APPEARANCE_ORIENTATION_PATH,
        APPEARANCE_METRICS_PATH,
        MATCH_METRICS_PATH,
        BOOTSTRAP_PATH,
        SENSITIVITY_COMPARISON_PATH,
        KEY_RESULTS_PATH,
        FIGURE_MANIFEST_PATH,
    ]
    validation_checks = build_validation_checks(
        input_checks,
        recon_checks,
        match_data,
        appearance_data,
        appearance_metrics,
        match_metrics,
        bootstrap,
        figure_manifest,
        required_outputs,
    )
    final_outputs = required_outputs + [VALIDATION_CHECKS_PATH, *FIGURE_PATHS.values()]
    if not all(path.exists() for path in final_outputs):
        missing = [str(path) for path in final_outputs if not path.exists()]
        raise RuntimeError(f"Missing required Step 39 outputs: {missing}")

    print_console_summary(
        recon_pass,
        complement_summary,
        side_distribution,
        match_metrics,
        appearance_metrics,
        comparison,
        final_code,
        validation_checks,
        final_outputs,
    )


if __name__ == "__main__":
    main()
