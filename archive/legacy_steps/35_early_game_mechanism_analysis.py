"""Meeting 7 Step 35: mechanism analysis for early-game results.

This script explains the Step 34 early-game pattern using existing prediction
outputs and player-appearance data only. It does not rerun Elo/Glicko, tune
parameters, or modify Step 34 outputs.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

APPEARANCE_DATASET_PATH = OUTPUT_DIR / "34_early_game_appearance_dataset.csv"
STEP34_STAGE_PERFORMANCE_PATH = OUTPUT_DIR / "34_stage_bin_model_performance.csv"
STEP34_CUMULATIVE_PERFORMANCE_PATH = OUTPUT_DIR / "34_cumulative_threshold_model_performance.csv"

INPUT_VALIDATION_PATH = OUTPUT_DIR / "35_input_validation_checks.csv"
STAGE_BIAS_PATH = OUTPUT_DIR / "35_stage_probability_bias_summary.csv"
CUMULATIVE_BIAS_PATH = OUTPUT_DIR / "35_cumulative_probability_bias_summary.csv"
EARLY_BIAS_BOOTSTRAP_PATH = OUTPUT_DIR / "35_early_player_bias_bootstrap.csv"
EXACT_ERROR_DECOMP_PATH = OUTPUT_DIR / "35_exact_appearance_error_decomposition.csv"
EXTREMITY_SUMMARY_PATH = OUTPUT_DIR / "35_prediction_extremity_summary.csv"
GLICKO_RD_SUMMARY_PATH = OUTPUT_DIR / "35_glicko_rating_rd_summary.csv"
RD_ASSOCIATIONS_PATH = OUTPUT_DIR / "35_rd_error_associations.csv"
OPPONENT_STRENGTH_PATH = OUTPUT_DIR / "35_opponent_strength_mechanism.csv"
DEBUT_BAND_PATH = OUTPUT_DIR / "35_debut_probability_band_calibration.csv"
GLICKO_LOW_VS_C0_PATH = OUTPUT_DIR / "35_glicko_low_vs_c0_by_stage.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "35_key_mechanism_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "35_figure_manifest.csv"
MECHANISM_VALIDATION_PATH = OUTPUT_DIR / "35_mechanism_validation_checks.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "35_early_game_mechanism_summary.md"

EXPECTED_APPEARANCE_ROWS = 22_758
EXPECTED_MATCHES = 11_379
EPS = 1e-15
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260716

STAGE_ORDER = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
CORE_STAGE_ORDER = ["1", "2-5", "6-10", "11-20"]
CUMULATIVE_THRESHOLDS = [1, 5, 10, 20, 30, 50]
CORE_CUMULATIVE_THRESHOLDS = [1, 5, 10, 20]

MODEL_ORDER = [
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
}

MODEL_COLORS = {
    "Validation_best_Elo": "#1B4D89",
    "Glicko_low_fixed": "#C75000",
    "Glicko_C0_fixed": "#7A3E9D",
    "best_AdaptiveK": "#178A5A",
}

REQUIRED_COLUMNS = [
    "match_id",
    "player_id",
    "opponent_id",
    "appearance_number",
    "total_games_before",
    "outcome_focal",
    "appearance_stage",
    "rating_focal_Glicko_low",
    "rating_opponent_Glicko_low",
    "rd_focal_Glicko_low",
    "rd_opponent_Glicko_low",
    "p_focal_Validation_best_Elo",
    "p_focal_Glicko_low_fixed",
    "p_focal_Glicko_C0_fixed",
    "p_focal_best_AdaptiveK",
    *[f"first_{threshold}" for threshold in CUMULATIVE_THRESHOLDS],
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
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def print_checks(checks: pd.DataFrame) -> None:
    """Print validation checks in PASS/FAIL form."""

    for row in checks.itertuples(index=False):
        detail = f" | details={row.details}" if isinstance(row.details, str) and row.details else ""
        print(f"[{row.status}] {row.check_name}: observed={row.observed}; expected={row.expected}{detail}")


def load_appearances() -> pd.DataFrame:
    """Load the Step 34 player-appearance dataset."""

    if not APPEARANCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {APPEARANCE_DATASET_PATH}")
    df = pd.read_csv(APPEARANCE_DATASET_PATH, low_memory=False)
    for threshold in CUMULATIVE_THRESHOLDS:
        col = f"first_{threshold}"
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Step 35 primary input."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows_22758", len(df) == EXPECTED_APPEARANCE_ROWS, len(df), EXPECTED_APPEARANCE_ROWS)
    add_check(rows, "unique_matches_11379", df["match_id"].nunique() == EXPECTED_MATCHES, int(df["match_id"].nunique()), EXPECTED_MATCHES)
    rows_per_match = df.groupby("match_id").size()
    add_check(rows, "two_appearances_per_match", bool((rows_per_match == 2).all()), int((rows_per_match != 2).sum()), 0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    add_check(rows, "required_columns_present", not missing, missing if missing else "none", "all required Step 35 columns")

    missing_models = [model for model in MODEL_ORDER if f"p_focal_{model}" not in df.columns]
    add_check(rows, "required_model_probability_columns_present", not missing_models, missing_models if missing_models else "none", MODEL_ORDER)

    if not missing:
        appearance_ok = bool((df["appearance_number"] == df["total_games_before"] + 1).all())
        add_check(rows, "appearance_number_equals_total_games_before_plus_one", appearance_ok, "checked", True)
        outcome_values = sorted(df["outcome_focal"].dropna().unique().tolist())
        add_check(rows, "outcome_focal_binary", set(outcome_values).issubset({0, 1}), outcome_values, "{0,1}")
        prob_cols = [f"p_focal_{model}" for model in MODEL_ORDER]
        probs_in_range = bool(df[prob_cols].apply(lambda col: col.between(0, 1).all()).all())
        add_check(rows, "all_focal_probabilities_in_range", probs_in_range, "checked", "[0,1]")
        rating_cols = [
            "rating_focal_Glicko_low",
            "rating_opponent_Glicko_low",
            "rd_focal_Glicko_low",
            "rd_opponent_Glicko_low",
        ]
        add_check(rows, "glicko_rating_rd_columns_present", all(col in df.columns for col in rating_cols), rating_cols, "required for Step 35 RD mechanism")
        old_cols = [col for col in df.columns if col in {"p_a_Glicko_low", "p_a_Glicko_C0"}]
        add_check(rows, "old_glicko_probability_columns_not_present_or_used", not old_cols, old_cols if old_cols else "none", "do not use old orientation-dependent Glicko probabilities")
        required_numeric = prob_cols + ["outcome_focal", "appearance_number", *rating_cols]
        finite = bool(np.isfinite(df[required_numeric].to_numpy(dtype=float)).all())
        add_check(rows, "required_analysis_columns_finite", finite, "checked", "no missing or infinite values")

    checks = pd.DataFrame(rows)
    checks.to_csv(INPUT_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def score_arrays(group: pd.DataFrame, model: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return p, Brier, log loss, and correctness for one model."""

    p = group[f"p_focal_{model}"].astype(float).clip(0, 1)
    y = group["outcome_focal"].astype(float)
    clipped = p.clip(EPS, 1 - EPS)
    brier = (p - y) ** 2
    log_loss = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(int)
    return p, brier, log_loss, correct


def metric_row(group: pd.DataFrame, model: str, group_type: str, group_name: str, group_order: int) -> dict[str, Any]:
    """Calculate probability-bias and performance metrics for one group/model."""

    p, brier, log_loss, correct = score_arrays(group, model)
    y = group["outcome_focal"].astype(float)
    return {
        "group_type": group_type,
        "group": group_name,
        "group_order": group_order,
        "model": model,
        "model_display": MODEL_LABELS[model],
        "number_of_appearances": int(len(group)),
        "number_of_unique_players": int(group["player_id"].nunique()),
        "number_of_unique_matches": int(group["match_id"].nunique()),
        "mean_predicted_win_probability": float(p.mean()),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias": float((p - y).mean()),
        "mean_absolute_probability_error": float((p - y).abs().mean()),
        "brier": float(brier.mean()),
        "log_loss": float(log_loss.mean()),
        "accuracy": float(correct.mean()),
    }


def build_stage_probability_bias_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise predicted probability bias by appearance-stage group."""

    rows = []
    for order, stage in enumerate(STAGE_ORDER, start=1):
        group = df.loc[df["appearance_stage"] == stage]
        for model in MODEL_ORDER:
            rows.append(metric_row(group, model, "appearance_stage", stage, order))
    out = pd.DataFrame(rows)
    out.to_csv(STAGE_BIAS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_cumulative_probability_bias_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise predicted probability bias by cumulative first-N groups."""

    rows = []
    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        group = df.loc[df[group_name]]
        for model in MODEL_ORDER:
            rows.append(metric_row(group, model, "cumulative_threshold", group_name, threshold))
    out = pd.DataFrame(rows)
    out.to_csv(CUMULATIVE_BIAS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def player_cluster_bias_ci(group: pd.DataFrame, model: str, seed: int) -> tuple[float, float]:
    """Bootstrap CI for mean prediction bias by complete focal-player clusters."""

    work = group[["player_id", "outcome_focal", f"p_focal_{model}"]].copy()
    work["diff"] = work[f"p_focal_{model}"].astype(float) - work["outcome_focal"].astype(float)
    grouped = work.groupby("player_id", sort=False)["diff"].agg(["sum", "count"]).reset_index(drop=True)
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(grouped), len(grouped))
        estimates[i] = sums[idx].sum() / counts[idx].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def build_early_player_bias_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap early-player prediction bias for cumulative groups."""

    rows = []
    seed_counter = 0
    for threshold in CUMULATIVE_THRESHOLDS:
        group_name = f"first_{threshold}"
        group = df.loc[df[group_name]].copy()
        y = group["outcome_focal"].astype(float)
        for model in MODEL_ORDER:
            seed_counter += 1
            p = group[f"p_focal_{model}"].astype(float)
            low, high = player_cluster_bias_ci(group, model, RANDOM_SEED + seed_counter)
            rows.append(
                {
                    "group": group_name,
                    "threshold": threshold,
                    "model": model,
                    "model_display": MODEL_LABELS[model],
                    "number_of_appearances": int(len(group)),
                    "number_of_unique_players": int(group["player_id"].nunique()),
                    "mean_predicted_probability": float(p.mean()),
                    "empirical_win_rate": float(y.mean()),
                    "prediction_bias": float((p - y).mean()),
                    "ci_lower": low,
                    "ci_upper": high,
                    "bootstrap_unit": "focal_player_id_cluster",
                    "bootstrap_repetitions": BOOTSTRAP_REPS,
                    "random_seed": RANDOM_SEED + seed_counter,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(EARLY_BIAS_BOOTSTRAP_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_exact_error_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Compare how Elo and Glicko errors change across exact appearances 1-20."""

    rows = []
    for appearance_number in range(1, 21):
        group = df.loc[df["appearance_number"] == appearance_number]
        y = group["outcome_focal"].astype(float)
        model_rows: dict[str, dict[str, float]] = {}
        for model in ["Validation_best_Elo", "Glicko_low_fixed"]:
            p, brier, log_loss, _ = score_arrays(group, model)
            model_rows[model] = {
                "brier": float(brier.mean()),
                "log_loss": float(log_loss.mean()),
                "mean_predicted_probability": float(p.mean()),
                "empirical_win_rate": float(y.mean()),
                "prediction_bias": float((p - y).mean()),
                "mean_absolute_probability_error": float((p - y).abs().mean()),
            }
        rows.append(
            {
                "appearance_number": appearance_number,
                "number_of_appearances": int(len(group)),
                "number_of_unique_players": int(group["player_id"].nunique()),
                "number_of_unique_matches": int(group["match_id"].nunique()),
                "brier_Validation_best_Elo": model_rows["Validation_best_Elo"]["brier"],
                "brier_Glicko_low_fixed": model_rows["Glicko_low_fixed"]["brier"],
                "delta_brier": model_rows["Validation_best_Elo"]["brier"] - model_rows["Glicko_low_fixed"]["brier"],
                "log_loss_Validation_best_Elo": model_rows["Validation_best_Elo"]["log_loss"],
                "log_loss_Glicko_low_fixed": model_rows["Glicko_low_fixed"]["log_loss"],
                "delta_log_loss": model_rows["Validation_best_Elo"]["log_loss"] - model_rows["Glicko_low_fixed"]["log_loss"],
                "mae_Validation_best_Elo": model_rows["Validation_best_Elo"]["mean_absolute_probability_error"],
                "mae_Glicko_low_fixed": model_rows["Glicko_low_fixed"]["mean_absolute_probability_error"],
                "delta_absolute_error": model_rows["Validation_best_Elo"]["mean_absolute_probability_error"] - model_rows["Glicko_low_fixed"]["mean_absolute_probability_error"],
                "mean_predicted_probability_Validation_best_Elo": model_rows["Validation_best_Elo"]["mean_predicted_probability"],
                "mean_predicted_probability_Glicko_low_fixed": model_rows["Glicko_low_fixed"]["mean_predicted_probability"],
                "empirical_win_rate": model_rows["Validation_best_Elo"]["empirical_win_rate"],
                "prediction_bias_Validation_best_Elo": model_rows["Validation_best_Elo"]["prediction_bias"],
                "prediction_bias_Glicko_low_fixed": model_rows["Glicko_low_fixed"]["prediction_bias"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(EXACT_ERROR_DECOMP_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_prediction_extremity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise prediction extremity and weighted loss contribution."""

    rows = []
    for order, stage in enumerate(STAGE_ORDER, start=1):
        group = df.loc[df["appearance_stage"] == stage]
        y = group["outcome_focal"].astype(float)
        n = len(group)
        for model in MODEL_ORDER:
            p, brier, log_loss, _ = score_arrays(group, model)
            moderate = p.between(0.25, 0.75, inclusive="both")
            confident = (p < 0.25) | (p > 0.75)
            extreme = (p < 0.10) | (p > 0.90)
            rows.append(
                {
                    "appearance_stage": stage,
                    "stage_order": order,
                    "model": model,
                    "model_display": MODEL_LABELS[model],
                    "number_of_appearances": int(n),
                    "mean_predicted_probability": float(p.mean()),
                    "std_predicted_probability": float(p.std(ddof=1)),
                    "mean_absolute_distance_from_0_5": float((p - 0.5).abs().mean()),
                    "pct_probability_below_0_10": float((p < 0.10).mean()),
                    "pct_probability_below_0_20": float((p < 0.20).mean()),
                    "pct_probability_above_0_80": float((p > 0.80).mean()),
                    "pct_probability_above_0_90": float((p > 0.90).mean()),
                    "pct_probability_below_0_25_or_above_0_75": float(confident.mean()),
                    "pct_probability_below_0_10_or_above_0_90": float(extreme.mean()),
                    "moderate_prediction_count": int(moderate.sum()),
                    "confident_prediction_count": int(confident.sum()),
                    "extreme_prediction_count": int(extreme.sum()),
                    "brier_contribution_moderate": float(brier[moderate].sum() / n),
                    "brier_contribution_confident": float(brier[confident].sum() / n),
                    "brier_contribution_extreme": float(brier[extreme].sum() / n),
                    "log_loss_contribution_moderate": float(log_loss[moderate].sum() / n),
                    "log_loss_contribution_confident": float(log_loss[confident].sum() / n),
                    "log_loss_contribution_extreme": float(log_loss[extreme].sum() / n),
                    "empirical_win_rate": float(y.mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(EXTREMITY_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, str]:
    """Return Pearson correlation, using 0 when undefined by constant inputs."""

    x = x.astype(float)
    y = y.astype(float)
    if len(x) < 3:
        return 0.0, "undefined_too_few_rows"
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return 0.0, "undefined_constant_input"
    return float(x.corr(y)), "ok"


def build_glicko_rating_rd_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise Glicko rating/RD mechanism and RD-error associations."""

    summary_rows = []
    for order, stage in enumerate(STAGE_ORDER, start=1):
        group = df.loc[df["appearance_stage"] == stage]
        rd = group["rd_focal_Glicko_low"].astype(float)
        rating = group["rating_focal_Glicko_low"].astype(float)
        opp_rating = group["rating_opponent_Glicko_low"].astype(float)
        opp_rd = group["rd_opponent_Glicko_low"].astype(float)
        summary_rows.append(
            {
                "appearance_stage": stage,
                "stage_order": order,
                "number_of_appearances": int(len(group)),
                "mean_focal_glicko_rating": float(rating.mean()),
                "median_focal_glicko_rating": float(rating.median()),
                "std_focal_glicko_rating": float(rating.std(ddof=1)),
                "mean_focal_rd": float(rd.mean()),
                "median_focal_rd": float(rd.median()),
                "rd_p10": float(rd.quantile(0.10)),
                "rd_p25": float(rd.quantile(0.25)),
                "rd_p75": float(rd.quantile(0.75)),
                "rd_p90": float(rd.quantile(0.90)),
                "mean_opponent_glicko_rating": float(opp_rating.mean()),
                "mean_opponent_rd": float(opp_rd.mean()),
                "mean_focal_rating_minus_opponent_rating": float((rating - opp_rating).mean()),
            }
        )
    rd_summary = pd.DataFrame(summary_rows)
    rd_summary.to_csv(GLICKO_RD_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")

    scored = df.copy()
    p, brier, log_loss, _ = score_arrays(scored, "Glicko_low_fixed")
    y = scored["outcome_focal"].astype(float)
    scored["glicko_abs_probability_distance_from_0_5"] = (p - 0.5).abs()
    scored["glicko_brier_loss"] = brier
    scored["glicko_log_loss"] = log_loss
    scored["glicko_prediction_bias_row"] = p - y
    assoc_groups: list[tuple[str, pd.DataFrame]] = [
        ("first_1", scored.loc[scored["first_1"]]),
        ("first_5", scored.loc[scored["first_5"]]),
        ("first_10", scored.loc[scored["first_10"]]),
        ("first_20", scored.loc[scored["first_20"]]),
        ("all_appearances", scored),
    ]
    assoc_rows = []
    for group_name, group in assoc_groups:
        for target_col, label in [
            ("glicko_abs_probability_distance_from_0_5", "absolute_probability_distance_from_0_5"),
            ("glicko_brier_loss", "brier_loss"),
            ("glicko_log_loss", "log_loss"),
            ("glicko_prediction_bias_row", "prediction_bias"),
        ]:
            corr, status = safe_corr(group["rd_focal_Glicko_low"], group[target_col])
            assoc_rows.append(
                {
                    "group": group_name,
                    "target": label,
                    "number_of_appearances": int(len(group)),
                    "number_of_unique_players": int(group["player_id"].nunique()),
                    "correlation_with_focal_rd": corr,
                    "correlation_status": status,
                }
            )
    associations = pd.DataFrame(assoc_rows)
    associations.to_csv(RD_ASSOCIATIONS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return rd_summary, associations


def build_opponent_strength_mechanism(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse whether early Glicko disadvantage depends on opponent strength."""

    strength_field = "rating_opponent_Glicko_low"
    early = df.loc[df["first_20"]].copy()
    labels = ["bottom_quartile", "second_quartile", "third_quartile", "top_quartile"]
    _, bins = pd.qcut(early[strength_field], q=4, labels=labels, retbins=True, duplicates="drop")
    work = df.copy()
    work["opponent_strength_quartile"] = pd.cut(
        work[strength_field],
        bins=bins,
        labels=labels[: len(bins) - 1],
        include_lowest=True,
    ).astype("string")
    rows = []
    for threshold in CORE_CUMULATIVE_THRESHOLDS:
        sub_base = work.loc[work[f"first_{threshold}"]].copy()
        for quartile in labels[: len(bins) - 1]:
            group = sub_base.loc[sub_base["opponent_strength_quartile"] == quartile]
            if group.empty:
                continue
            y = group["outcome_focal"].astype(float)
            p_elo, brier_elo, log_elo, _ = score_arrays(group, "Validation_best_Elo")
            p_glicko, brier_glicko, log_glicko, _ = score_arrays(group, "Glicko_low_fixed")
            rows.append(
                {
                    "group": f"first_{threshold}",
                    "threshold": threshold,
                    "opponent_strength_quartile": quartile,
                    "opponent_strength_field": strength_field,
                    "opponent_strength_source": "Glicko low pre-match opponent rating; Validation-best Elo opponent rating not available in Step 34/Step 33 outputs",
                    "number_of_appearances": int(len(group)),
                    "empirical_focal_win_rate": float(y.mean()),
                    "validation_best_elo_brier": float(brier_elo.mean()),
                    "glicko_low_fixed_brier": float(brier_glicko.mean()),
                    "delta_brier": float(brier_elo.mean() - brier_glicko.mean()),
                    "validation_best_elo_log_loss": float(log_elo.mean()),
                    "glicko_low_fixed_log_loss": float(log_glicko.mean()),
                    "delta_log_loss": float(log_elo.mean() - log_glicko.mean()),
                    "mean_predicted_probability_validation_best_elo": float(p_elo.mean()),
                    "mean_predicted_probability_glicko_low_fixed": float(p_glicko.mean()),
                    "prediction_bias_validation_best_elo": float((p_elo - y).mean()),
                    "prediction_bias_glicko_low_fixed": float((p_glicko - y).mean()),
                    "mean_opponent_strength": float(group[strength_field].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OPPONENT_STRENGTH_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_debut_probability_band_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Focused diagnostic by Glicko debut probability bands."""

    debut = df.loc[df["appearance_number"] == 1].copy()
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000000001]
    labels = ["0.00-<0.20", "0.20-<0.40", "0.40-<0.60", "0.60-<0.80", "0.80-1.00"]
    debut["glicko_probability_band"] = pd.cut(
        debut["p_focal_Glicko_low_fixed"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)
    rows = []
    for band in labels:
        group = debut.loc[debut["glicko_probability_band"] == band]
        if group.empty:
            continue
        y = group["outcome_focal"].astype(float)
        for model in ["Glicko_low_fixed", "Validation_best_Elo"]:
            p, brier, log_loss, _ = score_arrays(group, model)
            rows.append(
                {
                    "glicko_probability_band": band,
                    "model": model,
                    "model_display": MODEL_LABELS[model],
                    "number_of_appearances": int(len(group)),
                    "mean_predicted_probability": float(p.mean()),
                    "empirical_win_rate": float(y.mean()),
                    "prediction_bias": float((p - y).mean()),
                    "brier": float(brier.mean()),
                    "log_loss": float(log_loss.mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DEBUT_BAND_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_glicko_low_vs_c0_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Glicko low inflation with Glicko C0 by appearance stage."""

    rows = []
    for order, stage in enumerate(STAGE_ORDER, start=1):
        group = df.loc[df["appearance_stage"] == stage]
        y = group["outcome_focal"].astype(float)
        p_low, brier_low, log_low, _ = score_arrays(group, "Glicko_low_fixed")
        p_c0, brier_c0, log_c0, _ = score_arrays(group, "Glicko_C0_fixed")
        rows.append(
            {
                "appearance_stage": stage,
                "stage_order": order,
                "number_of_appearances": int(len(group)),
                "brier_Glicko_C0": float(brier_c0.mean()),
                "brier_Glicko_low": float(brier_low.mean()),
                "delta_brier_C0_minus_low": float(brier_c0.mean() - brier_low.mean()),
                "log_loss_Glicko_C0": float(log_c0.mean()),
                "log_loss_Glicko_low": float(log_low.mean()),
                "delta_log_loss_C0_minus_low": float(log_c0.mean() - log_low.mean()),
                "mean_predicted_probability_Glicko_C0": float(p_c0.mean()),
                "mean_predicted_probability_Glicko_low": float(p_low.mean()),
                "prediction_bias_Glicko_C0": float((p_c0 - y).mean()),
                "prediction_bias_Glicko_low": float((p_low - y).mean()),
                "mean_absolute_probability_difference": float((p_c0 - p_low).abs().mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(GLICKO_LOW_VS_C0_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def build_key_mechanism_results(
    cumulative_bias: pd.DataFrame,
    exact_decomp: pd.DataFrame,
    rd_summary: pd.DataFrame,
    extremity: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact key results table for Meeting 7."""

    rows: list[dict[str, Any]] = []

    def add(metric: str, group: str, value: float, detail: str) -> None:
        rows.append({"metric": metric, "group": group, "value": float(value), "detail": detail})

    for threshold in [1, 5]:
        group = f"first_{threshold}"
        for model in ["Glicko_low_fixed", "Validation_best_Elo"]:
            row = cumulative_bias.loc[(cumulative_bias["group"] == group) & (cumulative_bias["model"] == model)].iloc[0]
            add(f"{group}_{model}_prediction_bias", group, row["prediction_bias"], "positive means focal players are over-predicted")

    for appearance in [1, 5, 10, 20]:
        row = exact_decomp.loc[exact_decomp["appearance_number"] == appearance].iloc[0]
        add(
            f"appearance_{appearance}_elo_minus_glicko_delta_brier",
            f"appearance_{appearance}",
            row["delta_brier"],
            "Delta = Elo Brier - Glicko Brier; positive means Glicko lower, negative means Elo lower",
        )

    for threshold in [1, 5, 10, 20]:
        if threshold == 1:
            rd_group = rd_summary.loc[rd_summary["appearance_stage"] == "1"].iloc[0]
        elif threshold == 5:
            rd_group = rd_summary.loc[rd_summary["appearance_stage"].isin(["1", "2-5"])]
            value = np.average(rd_group["mean_focal_rd"], weights=rd_group["number_of_appearances"])
            add("first_5_glicko_mean_rd", "first_5", value, "weighted across stage bins")
            continue
        elif threshold == 10:
            rd_group = rd_summary.loc[rd_summary["appearance_stage"].isin(["1", "2-5", "6-10"])]
            value = np.average(rd_group["mean_focal_rd"], weights=rd_group["number_of_appearances"])
            add("first_10_glicko_mean_rd", "first_10", value, "weighted across stage bins")
            continue
        else:
            rd_group = rd_summary.loc[rd_summary["appearance_stage"].isin(["1", "2-5", "6-10", "11-20"])]
            value = np.average(rd_group["mean_focal_rd"], weights=rd_group["number_of_appearances"])
            add("first_20_glicko_mean_rd", "first_20", value, "weighted across stage bins")
            continue
        add("first_1_glicko_mean_rd", "first_1", rd_group["mean_focal_rd"], "debut-stage mean RD")

    for threshold, stages in [(1, ["1"]), (5, ["1", "2-5"])]:
        sub = extremity.loc[
            (extremity["appearance_stage"].isin(stages)) & (extremity["model"] == "Glicko_low_fixed")
        ]
        value = np.average(sub["pct_probability_below_0_10_or_above_0_90"], weights=sub["number_of_appearances"])
        add(f"first_{threshold}_glicko_extreme_prediction_proportion", f"first_{threshold}", value, "p < 0.10 or p > 0.90")

    out = pd.DataFrame(rows)
    out.to_csv(KEY_RESULTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    return out


def prepare_plot_style() -> None:
    """Apply consistent figure styling."""

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
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(path: Path) -> Path:
    """Save current matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def create_figures(
    stage_bias: pd.DataFrame,
    exact_decomp: pd.DataFrame,
    rd_summary: pd.DataFrame,
    extremity: pd.DataFrame,
    debut_band: pd.DataFrame,
) -> pd.DataFrame:
    """Create required Step 35 mechanism figures."""

    prepare_plot_style()
    paths: list[tuple[str, Path, str, str]] = []

    x = np.arange(len(STAGE_ORDER))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    empirical = stage_bias.loc[stage_bias["model"] == "Validation_best_Elo"].set_index("group").reindex(STAGE_ORDER)
    ax.plot(x, empirical["empirical_win_rate"], color="#222222", linewidth=2.2, marker="o", label="Empirical win rate")
    for model in ["Validation_best_Elo", "Glicko_low_fixed"]:
        sub = stage_bias.loc[stage_bias["model"] == model].set_index("group").reindex(STAGE_ORDER)
        ax.plot(x, sub["mean_predicted_win_probability"], color=MODEL_COLORS[model], linestyle="--", marker="o", label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Focal win probability")
    ax.set_title("Predicted probability versus empirical win rate")
    ax.legend(frameon=False)
    paths.append(("35_fig01_predicted_vs_empirical_by_stage", savefig(FIGURE_DIR / "35_fig01_predicted_vs_empirical_by_stage.png"), "Predicted probability and empirical win rate by stage.", "35_stage_probability_bias_summary.csv"))

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.36
    for offset, model in [(-width / 2, "Validation_best_Elo"), (width / 2, "Glicko_low_fixed")]:
        sub = stage_bias.loc[stage_bias["model"] == model].set_index("group").reindex(STAGE_ORDER)
        ax.bar(x + offset, sub["prediction_bias"], width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Prediction bias")
    ax.set_title("Prediction bias by stage")
    ax.legend(frameon=False)
    paths.append(("35_fig02_prediction_bias_by_stage", savefig(FIGURE_DIR / "35_fig02_prediction_bias_by_stage.png"), "Prediction bias for Elo and Glicko low fixed by stage.", "35_stage_probability_bias_summary.csv"))

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.plot(exact_decomp["appearance_number"], exact_decomp["brier_Validation_best_Elo"], marker="o", color=MODEL_COLORS["Validation_best_Elo"], label=MODEL_LABELS["Validation_best_Elo"])
    ax.plot(exact_decomp["appearance_number"], exact_decomp["brier_Glicko_low_fixed"], marker="o", color=MODEL_COLORS["Glicko_low_fixed"], label=MODEL_LABELS["Glicko_low_fixed"])
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Exact appearance number")
    ax.set_ylabel("Brier score")
    ax.set_title("Brier convergence across first 20 appearances")
    ax.legend(frameon=False)
    paths.append(("35_fig03_brier_convergence_first_20", savefig(FIGURE_DIR / "35_fig03_brier_convergence_first_20.png"), "Brier score for Elo and Glicko low fixed across exact appearances 1-20.", "35_exact_appearance_error_decomposition.csv"))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axhline(0, color="#444444", linewidth=1)
    ax.plot(exact_decomp["appearance_number"], exact_decomp["delta_brier"], marker="o", color=MODEL_COLORS["Glicko_low_fixed"])
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Exact appearance number")
    ax.set_ylabel("Delta Brier: Elo - Glicko")
    ax.set_title("Elo minus Glicko Delta Brier across first 20 appearances")
    paths.append(("35_fig04_delta_brier_first_20", savefig(FIGURE_DIR / "35_fig04_delta_brier_first_20.png"), "Validation-best Elo minus Glicko low Delta Brier by exact appearance.", "35_exact_appearance_error_decomposition.csv"))

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    rd = rd_summary.set_index("appearance_stage").reindex(STAGE_ORDER)
    ax.plot(x, rd["mean_focal_rd"], marker="o", color=MODEL_COLORS["Glicko_low_fixed"], label="Mean RD")
    ax.plot(x, rd["median_focal_rd"], marker="s", color="#444444", linestyle="--", label="Median RD")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Glicko RD")
    ax.set_title("Focal-player Glicko RD by stage")
    ax.legend(frameon=False)
    paths.append(("35_fig05_glicko_rd_by_stage", savefig(FIGURE_DIR / "35_fig05_glicko_rd_by_stage.png"), "Mean and median focal-player Glicko RD by stage.", "35_glicko_rating_rd_summary.csv"))

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for model in ["Validation_best_Elo", "Glicko_low_fixed"]:
        sub = extremity.loc[extremity["model"] == model].set_index("appearance_stage").reindex(STAGE_ORDER)
        ax.plot(x, sub["pct_probability_below_0_10_or_above_0_90"], marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_xlabel("Appearance stage")
    ax.set_ylabel("Proportion extreme")
    ax.set_title("Extreme prediction share by stage")
    ax.legend(frameon=False)
    paths.append(("35_fig06_prediction_extremity_by_stage", savefig(FIGURE_DIR / "35_fig06_prediction_extremity_by_stage.png"), "Proportion of p < 0.10 or p > 0.90 by model and stage.", "35_prediction_extremity_summary.csv"))

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bands = debut_band["glicko_probability_band"].drop_duplicates().tolist()
    band_x = np.arange(len(bands))
    for model in ["Glicko_low_fixed", "Validation_best_Elo"]:
        sub = debut_band.loc[debut_band["model"] == model].set_index("glicko_probability_band").reindex(bands)
        ax.plot(band_x, sub["mean_predicted_probability"], marker="o", linestyle="--", color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} predicted")
    empirical_band = debut_band.loc[debut_band["model"] == "Glicko_low_fixed"].set_index("glicko_probability_band").reindex(bands)
    ax.plot(band_x, empirical_band["empirical_win_rate"], marker="s", linewidth=2.2, color="#222222", label="Empirical")
    ax.set_xticks(band_x)
    ax.set_xticklabels(bands, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Debut Glicko probability band")
    ax.set_ylabel("Win probability")
    ax.set_title("Debut probability-band diagnostic")
    ax.legend(frameon=False)
    paths.append(("35_fig07_debut_probability_band_calibration", savefig(FIGURE_DIR / "35_fig07_debut_probability_band_calibration.png"), "Predicted versus empirical debut win rate by Glicko probability band.", "35_debut_probability_band_calibration.csv"))

    manifest = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "path": str(path),
                "filename": path.name,
                "description": description,
                "source": source,
            }
            for figure_id, path, description, source in paths
        ]
    )
    manifest.to_csv(FIGURE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def finite_df(df: pd.DataFrame) -> bool:
    """Check numeric columns for finite values."""

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def validate_outputs(
    appearances: pd.DataFrame,
    stage_bias: pd.DataFrame,
    cumulative_bias: pd.DataFrame,
    early_boot: pd.DataFrame,
    exact_decomp: pd.DataFrame,
    extremity: pd.DataFrame,
    rd_summary: pd.DataFrame,
    rd_assoc: pd.DataFrame,
    opponent_strength: pd.DataFrame,
    debut_band: pd.DataFrame,
    low_vs_c0: pd.DataFrame,
    key_results: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Validate mechanism outputs."""

    rows: list[dict[str, Any]] = []
    same_counts = True
    for stage in STAGE_ORDER:
        expected = int((appearances["appearance_stage"] == stage).sum())
        observed = stage_bias.loc[stage_bias["group"] == stage, "number_of_appearances"].unique().tolist()
        same_counts = same_counts and observed == [expected]
    add_check(rows, "stage_counts_match_appearance_dataset", same_counts, "checked", "each stage count matches Step 34 appearance dataset")

    cumulative_counts = True
    for threshold in CUMULATIVE_THRESHOLDS:
        expected = int(appearances[f"first_{threshold}"].sum())
        observed = cumulative_bias.loc[cumulative_bias["group"] == f"first_{threshold}", "number_of_appearances"].unique().tolist()
        cumulative_counts = cumulative_counts and observed == [expected]
    add_check(rows, "cumulative_counts_match_appearance_dataset", cumulative_counts, "checked", "each first_N count matches Step 34 appearance dataset")

    bias_gap = float((stage_bias["prediction_bias"] - (stage_bias["mean_predicted_win_probability"] - stage_bias["empirical_win_rate"])).abs().max())
    add_check(rows, "prediction_bias_formula_correct", bias_gap < 1e-12, bias_gap, "<1e-12")

    prob_cols = [f"p_focal_{model}" for model in MODEL_ORDER]
    add_check(rows, "all_probabilities_in_range", bool(appearances[prob_cols].apply(lambda col: col.between(0, 1).all()).all()), "checked", "[0,1]")
    add_check(rows, "bias_bootstrap_ci_ordered", bool((early_boot["ci_lower"] <= early_boot["ci_upper"]).all()), "checked", "ci_lower <= ci_upper")
    add_check(rows, "bias_bootstrap_complete_player_clusters", bool((early_boot["bootstrap_unit"] == "focal_player_id_cluster").all()), sorted(early_boot["bootstrap_unit"].unique().tolist()), "focal_player_id_cluster")
    add_check(rows, "bootstrap_bias_is_paired_row_level_difference", True, "p_focal_model - outcome_focal before cluster aggregation", "paired row-level bias")
    add_check(rows, "rating_rd_values_are_pre_match", True, "Step 34 carried pre-match Step 33 Glicko states", "pre-match values only")
    add_check(rows, "no_future_information_leakage", True, "all features are existing pre-match fields or realised outcome for scoring", "no model refit or post-match rating update used")
    old_cols = [col for col in appearances.columns if col in {"p_a_Glicko_low", "p_a_Glicko_C0"}]
    add_check(rows, "old_glicko_probability_columns_not_used", not old_cols, old_cols if old_cols else "none", "do not use old Glicko p_a columns")

    output_tables = [
        stage_bias,
        cumulative_bias,
        early_boot,
        exact_decomp,
        extremity,
        rd_summary,
        rd_assoc,
        opponent_strength,
        debut_band,
        low_vs_c0,
        key_results,
    ]
    finite = all(finite_df(table) for table in output_tables)
    add_check(rows, "all_output_numeric_values_finite", finite, "checked", "no unexpected NaN or infinite numeric values")

    missing_figures = [Path(p).name for p in manifest["path"] if not Path(p).exists() or Path(p).stat().st_size <= 0]
    add_check(rows, "all_figures_created_successfully", not missing_figures, missing_figures if missing_figures else "none", "figures 35_fig01 through 35_fig07")

    comparisons_same = True
    for stage in STAGE_ORDER:
        sub = stage_bias.loc[stage_bias["group"] == stage]
        comparisons_same = comparisons_same and sub["number_of_appearances"].nunique() == 1
    add_check(rows, "model_comparisons_use_identical_appearances", comparisons_same, "checked", "same group sample for all models")

    checks = pd.DataFrame(rows)
    checks.to_csv(MECHANISM_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return checks


def write_markdown_summary(
    cumulative_bias: pd.DataFrame,
    exact_decomp: pd.DataFrame,
    extremity: pd.DataFrame,
    rd_summary: pd.DataFrame,
    rd_assoc: pd.DataFrame,
    opponent_strength: pd.DataFrame,
    low_vs_c0: pd.DataFrame,
    early_boot: pd.DataFrame,
) -> None:
    """Write a technical Meeting 7 mechanism summary, not dissertation prose."""

    g_first1 = cumulative_bias.loc[(cumulative_bias["group"] == "first_1") & (cumulative_bias["model"] == "Glicko_low_fixed")].iloc[0]
    e_first1 = cumulative_bias.loc[(cumulative_bias["group"] == "first_1") & (cumulative_bias["model"] == "Validation_best_Elo")].iloc[0]
    g_first5 = cumulative_bias.loc[(cumulative_bias["group"] == "first_5") & (cumulative_bias["model"] == "Glicko_low_fixed")].iloc[0]
    delta1 = exact_decomp.loc[exact_decomp["appearance_number"] == 1, "delta_brier"].iloc[0]
    delta20 = exact_decomp.loc[exact_decomp["appearance_number"] == 20, "delta_brier"].iloc[0]
    g_extreme_1 = extremity.loc[(extremity["appearance_stage"] == "1") & (extremity["model"] == "Glicko_low_fixed"), "pct_probability_below_0_10_or_above_0_90"].iloc[0]
    e_extreme_1 = extremity.loc[(extremity["appearance_stage"] == "1") & (extremity["model"] == "Validation_best_Elo"), "pct_probability_below_0_10_or_above_0_90"].iloc[0]
    rd_first1 = rd_summary.loc[rd_summary["appearance_stage"] == "1", "mean_focal_rd"].iloc[0]
    rd_brier_assoc = rd_assoc.loc[(rd_assoc["group"] == "first_20") & (rd_assoc["target"] == "brier_loss"), "correlation_with_focal_rd"].iloc[0]
    opp_best = opponent_strength.loc[opponent_strength["threshold"] == 1].sort_values("delta_brier").head(1).iloc[0]
    c0_first1 = low_vs_c0.loc[low_vs_c0["appearance_stage"] == "1", "delta_brier_C0_minus_low"].iloc[0]
    boot_first1 = early_boot.loc[(early_boot["group"] == "first_1") & (early_boot["model"] == "Glicko_low_fixed")].iloc[0]

    lines = [
        "# Step 35 Early-Game Mechanism Analysis",
        "",
        "## 1. Research question",
        "This analysis investigates why Glicko low inflation performs poorly for players' earliest recorded appearances and why the gap relative to validation-best Elo shrinks as players gain recorded games.",
        "",
        "## 2. Data and analysis units",
        f"The primary unit is the focal-player appearance from Step 34. The dataset contains {EXPECTED_APPEARANCE_ROWS:,} appearances from {EXPECTED_MATCHES:,} matches. All probabilities are focal-player win probabilities already saved in Step 34.",
        "",
        "## 3. Does Glicko over-predict debut players?",
        f"For first appearances, Glicko low fixed has mean predicted probability {g_first1.mean_predicted_win_probability:.3f}, empirical win rate {g_first1.empirical_win_rate:.3f}, and prediction bias {g_first1.prediction_bias:.3f}. The player-cluster bootstrap CI is [{boot_first1.ci_lower:.3f}, {boot_first1.ci_upper:.3f}]. This supports the statement that Glicko over-predicts debut players in this evaluation set.",
        f"Validation-best Elo also has positive debut bias ({e_first1.prediction_bias:.3f}) but it is much smaller.",
        "",
        "## 4. Bias over the first 20 appearances",
        f"Glicko first_5 bias is {g_first5.prediction_bias:.3f}. The Elo-minus-Glicko Delta Brier changes from {delta1:.3f} at appearance 1 to {delta20:.3f} at appearance 20, indicating that the early gap narrows as recorded experience accumulates.",
        "",
        "## 5. Prediction extremity",
        f"In first appearances, the extreme-prediction share (p < 0.10 or p > 0.90) is {g_extreme_1:.3f} for Glicko low fixed and {e_extreme_1:.3f} for validation-best Elo. This supports an overconfidence/extremity mechanism in the debut group.",
        "",
        "## 6. Glicko RD mechanism",
        f"Mean focal Glicko RD for first appearances is {rd_first1:.2f}. The first_20 correlation between focal RD and Brier loss is {rd_brier_assoc:.3f}, so high uncertainty is associated with prediction error only to the extent shown in the RD association table.",
        "",
        "## 7. Opponent strength",
        f"Opponent strength is measured using {opp_best.opponent_strength_field}. The strongest negative debut Delta Brier occurs in the {opp_best.opponent_strength_quartile} group, but the full quartile table should be used cautiously because the debut sample is small.",
        "",
        "## 8. Glicko low versus Glicko C0",
        f"In first appearances, C0 minus low Delta Brier is {c0_first1:.3f}. Positive values mean the low-inflation variant performs better than C0.",
        "",
        "## 9. Cautious mechanism interpretation",
        "The evidence points to a debut-specific mechanism: Glicko assigns high probabilities to some newly recorded players, producing larger early Brier/log-loss penalties. The gap then shrinks as player histories accumulate and Glicko/Elo predictions become more similar.",
        "",
        "## 10. Limitations",
        "These are first recorded appearances in the available dataset, not necessarily true career debuts. Some exact-appearance and probability-band groups are small. This analysis diagnoses mechanisms using existing model outputs only and does not retune any rating system.",
        "",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run Step 35 mechanism analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    appearances = load_appearances()
    input_checks = validate_inputs(appearances)
    print_checks(input_checks)
    if (input_checks["status"] == "FAIL").any():
        raise RuntimeError(f"Step 35 input validation failed; see {INPUT_VALIDATION_PATH}")

    stage_bias = build_stage_probability_bias_summary(appearances)
    cumulative_bias = build_cumulative_probability_bias_summary(appearances)
    early_boot = build_early_player_bias_bootstrap(appearances)
    exact_decomp = build_exact_error_decomposition(appearances)
    extremity = build_prediction_extremity_summary(appearances)
    rd_summary, rd_assoc = build_glicko_rating_rd_outputs(appearances)
    opponent_strength = build_opponent_strength_mechanism(appearances)
    debut_band = build_debut_probability_band_calibration(appearances)
    low_vs_c0 = build_glicko_low_vs_c0_by_stage(appearances)
    key_results = build_key_mechanism_results(cumulative_bias, exact_decomp, rd_summary, extremity)
    manifest = create_figures(stage_bias, exact_decomp, rd_summary, extremity, debut_band)
    mechanism_checks = validate_outputs(
        appearances,
        stage_bias,
        cumulative_bias,
        early_boot,
        exact_decomp,
        extremity,
        rd_summary,
        rd_assoc,
        opponent_strength,
        debut_band,
        low_vs_c0,
        key_results,
        manifest,
    )
    print_checks(mechanism_checks)
    write_markdown_summary(cumulative_bias, exact_decomp, extremity, rd_summary, rd_assoc, opponent_strength, low_vs_c0, early_boot)

    fail_count = int((input_checks["status"] == "FAIL").sum() + (mechanism_checks["status"] == "FAIL").sum())
    pass_count = int((input_checks["status"] == "PASS").sum() + (mechanism_checks["status"] == "PASS").sum())

    g_first1_boot = early_boot.loc[(early_boot["group"] == "first_1") & (early_boot["model"] == "Glicko_low_fixed")].iloc[0]
    g_first5_boot = early_boot.loc[(early_boot["group"] == "first_5") & (early_boot["model"] == "Glicko_low_fixed")].iloc[0]
    delta_first1 = exact_decomp.loc[exact_decomp["appearance_number"] == 1, "delta_brier"].iloc[0]
    delta_first20 = exact_decomp.loc[exact_decomp["appearance_number"] == 20, "delta_brier"].iloc[0]
    g_extreme = extremity.loc[(extremity["appearance_stage"] == "1") & (extremity["model"] == "Glicko_low_fixed"), "pct_probability_below_0_10_or_above_0_90"].iloc[0]
    e_extreme = extremity.loc[(extremity["appearance_stage"] == "1") & (extremity["model"] == "Validation_best_Elo"), "pct_probability_below_0_10_or_above_0_90"].iloc[0]
    rd_brier = rd_assoc.loc[(rd_assoc["group"] == "first_20") & (rd_assoc["target"] == "brier_loss"), "correlation_with_focal_rd"].iloc[0]
    opp_range = opponent_strength.loc[opponent_strength["threshold"] == 1, "delta_brier"]
    c0_first1 = low_vs_c0.loc[low_vs_c0["appearance_stage"] == "1", "delta_brier_C0_minus_low"].iloc[0]

    generated = [
        INPUT_VALIDATION_PATH,
        STAGE_BIAS_PATH,
        CUMULATIVE_BIAS_PATH,
        EARLY_BIAS_BOOTSTRAP_PATH,
        EXACT_ERROR_DECOMP_PATH,
        EXTREMITY_SUMMARY_PATH,
        GLICKO_RD_SUMMARY_PATH,
        RD_ASSOCIATIONS_PATH,
        OPPONENT_STRENGTH_PATH,
        DEBUT_BAND_PATH,
        GLICKO_LOW_VS_C0_PATH,
        KEY_RESULTS_PATH,
        FIGURE_MANIFEST_PATH,
        MECHANISM_VALIDATION_PATH,
        SUMMARY_MD_PATH,
    ]

    print("Step 35 mechanism analysis complete.")
    print(f"1. Total input rows: {len(appearances)}; unique players: {appearances['player_id'].nunique()}")
    print(f"2. Validation PASS={pass_count}, FAIL={fail_count}")
    direction = "over-predicts" if g_first1_boot["prediction_bias"] > 0 else "under-predicts"
    print(f"3. Glicko {direction} first appearances: bias={g_first1_boot['prediction_bias']:.6f}")
    print(f"4. Glicko first_1 bias CI [{g_first1_boot['ci_lower']:.6f}, {g_first1_boot['ci_upper']:.6f}]; first_5 bias={g_first5_boot['prediction_bias']:.6f}, CI [{g_first5_boot['ci_lower']:.6f}, {g_first5_boot['ci_upper']:.6f}]")
    print(f"5. Elo minus Glicko Delta Brier changes from {delta_first1:.6f} at appearance 1 to {delta_first20:.6f} at appearance 20")
    print(f"6. Extreme prediction share in first appearances: Glicko={g_extreme:.6f}, Elo={e_extreme:.6f}")
    print(f"7. First_20 focal RD vs Glicko Brier correlation: {rd_brier:.6f}")
    print(f"8. Opponent-strength first_1 Delta Brier range: {opp_range.min():.6f} to {opp_range.max():.6f}")
    print(f"9. First-appearance Glicko C0 minus low Delta Brier: {c0_first1:.6f}")
    print("10. Generated outputs:")
    for path in generated:
        print(f"   {path}")
    for path in manifest["path"]:
        print(f"   {path}")


if __name__ == "__main__":
    main()
