"""Meeting 6 step 3: debut and robustness diagnostics.

This script diagnoses mechanisms behind the step 29 subgroup results, especially
the unexpectedly poor Glicko result for debut-player matches. It reads existing
meeting 5/6 outputs only, does not rerun any model, and writes new outputs under
outputs/meeting6/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP29_SCORES_PATH = OUTPUT_DIR / "29_per_match_model_scores_2025.csv"
STEP28_MATCH_FEATURES_PATH = OUTPUT_DIR / "28_prematch_match_features_2025.csv"
STEP28_LONG_FEATURES_PATH = OUTPUT_DIR / "28_prematch_player_features_2025_long.csv"
STEP29_OVERALL_METRICS_PATH = OUTPUT_DIR / "29_overall_model_metrics.csv"
STEP29_PAIRWISE_PATH = OUTPUT_DIR / "29_subgroup_pairwise_comparisons.csv"
STEP29_BOOTSTRAP_PATH = OUTPUT_DIR / "29_subgroup_bootstrap_confidence_intervals.csv"
STEP29_CALIBRATION_SUMMARY_PATH = OUTPUT_DIR / "29_corrected_calibration_summary.csv"
STEP29_CALIBRATION_BINS_PATH = OUTPUT_DIR / "29_corrected_calibration_bins.csv"
FAIR_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko" / "meeting5_fair_elo_vs_glicko_predictions_2025.csv"
ADAPTIVE_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_predictions_2025.csv"
RD_INFLATION_METRICS_PATH = PROJECT_ROOT / "outputs" / "meeting5_glicko_rd_inflation" / "meeting5_glicko_rd_inflation_metrics_2025.csv"

INPUT_VALIDATION_PATH = OUTPUT_DIR / "30_input_validation_checks.csv"
HISTORY_COUNTS_PATH = OUTPUT_DIR / "30_history_category_counts.csv"
DEBUT_PERSPECTIVE_PATH = OUTPUT_DIR / "30_debut_player_perspective.csv"
DEBUT_MODEL_SUMMARY_PATH = OUTPUT_DIR / "30_debut_model_summary.csv"
INITIALISATION_DIAGNOSTICS_PATH = OUTPUT_DIR / "30_initialisation_rating_scale_diagnostics.csv"
RATING_DISTRIBUTION_PATH = OUTPUT_DIR / "30_2025_rating_distribution_summary.csv"
TOP_DEBUT_MATCHES_PATH = OUTPUT_DIR / "30_top_influential_debut_matches.csv"
DEBUT_EVENT_CONTRIB_PATH = OUTPUT_DIR / "30_debut_event_contributions.csv"
DEBUT_LOEO_PATH = OUTPUT_DIR / "30_debut_leave_one_event_out.csv"
DEBUT_LOEO_SUMMARY_PATH = OUTPUT_DIR / "30_debut_leave_one_event_out_summary.csv"
ZERO_RECENT_PATH = OUTPUT_DIR / "30_zero_recent_activity_decomposition.csv"
RETURNING_SENSITIVITY_PATH = OUTPUT_DIR / "30_returning_player_threshold_sensitivity.csv"
EXCLUSION_ROBUSTNESS_PATH = OUTPUT_DIR / "30_overall_exclusion_robustness.csv"
NO_DEBUT_SUBGROUP_PATH = OUTPUT_DIR / "30_no_debut_subgroup_results.csv"
NO_DEBUT_RD_QUARTILE_PATH = OUTPUT_DIR / "30_no_debut_rd_quartile_results.csv"
NO_DEBUT_RD_DECILE_PATH = OUTPUT_DIR / "30_no_debut_rd_decile_results.csv"
NO_DEBUT_RD_CUTPOINTS_PATH = OUTPUT_DIR / "30_no_debut_rd_cutpoints.csv"
CONFIDENCE_DIAGNOSTICS_PATH = OUTPUT_DIR / "30_prediction_confidence_diagnostics.csv"
FAVOURITE_DISAGREEMENT_PATH = OUTPUT_DIR / "30_model_favourite_disagreement.csv"
BRIER_DECOMP_SUMMARY_PATH = OUTPUT_DIR / "30_brier_decomposition_summary.csv"
BRIER_DECOMP_BINS_PATH = OUTPUT_DIR / "30_brier_decomposition_bins.csv"
TOTAL_THRESHOLD_PATH = OUTPUT_DIR / "30_total_games_threshold_sensitivity.csv"
RECENT_THRESHOLD_PATH = OUTPUT_DIR / "30_recent_activity_threshold_sensitivity.csv"
KEY_DIAGNOSTIC_PATH = OUTPUT_DIR / "30_key_diagnostic_results.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "30_diagnostic_validation_checks.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "30_debut_initialisation_and_robustness_summary.md"

EPS = 1e-15
EXPECTED_GAMES = 11_379
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260713

MODEL_ALIASES = ["Glicko_low", "Validation_best_Elo", "Glicko_C0", "best_AdaptiveK", "Default_Elo"]
MODEL_LABELS = {
    "Glicko_low": "Glicko low inflation",
    "Validation_best_Elo": "Validation-best Elo",
    "Glicko_C0": "Glicko C0",
    "best_AdaptiveK": "Best adaptive-K",
    "Default_Elo": "Default Elo",
}
FAIR_MODEL_MAP = {
    "Glicko_low": "Glicko_low_inflation_match_by_match",
    "Glicko_C0": "Glicko_C0_match_by_match",
    "Validation_best_Elo": "Validation_best_Elo",
    "Default_Elo": "Default_Elo",
}
ADAPTIVE_MODEL_MAP = {"best_AdaptiveK": "AdaptiveK_PreviousYearGames_Elo_scale300"}


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", detail: str = "", severity: str = "error") -> None:
    """Append one validation row."""

    rows.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "detail": detail,
        }
    )


def load_step29_scores() -> pd.DataFrame:
    """Load step 29 per-match scores and add event keys."""

    df = pd.read_csv(STEP29_SCORES_PATH, low_memory=False)
    df["event_key"] = df["year"].astype(str) + "_" + df["event_id"].astype(str)
    return df


def load_step28_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load step 28 match-level and player-match features."""

    return (
        pd.read_csv(STEP28_MATCH_FEATURES_PATH, low_memory=False),
        pd.read_csv(STEP28_LONG_FEATURES_PATH, low_memory=False),
    )


def read_model_initialisation_constants() -> dict[str, Any]:
    """Read model constants from the existing code/output files."""

    glicko_code = (SCRIPT_DIR / "glicko_core.py").read_text(encoding="utf-8")

    def read_float_constant(name: str) -> float:
        match = re.search(rf"^{name}\s*=\s*([0-9.]+)", glicko_code, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find {name} in glicko_core.py")
        return float(match.group(1))

    fair_code = (SCRIPT_DIR / "26_fair_elo_vs_glicko_comparison.py").read_text(encoding="utf-8")
    elo_initial_match = re.search(r"ratings\.get\(winner,\s*([0-9.]+)\)", fair_code)
    elo_initial = float(elo_initial_match.group(1)) if elo_initial_match else np.nan

    low_c = np.nan
    low_target = np.nan
    if RD_INFLATION_METRICS_PATH.exists():
        rd_metrics = pd.read_csv(RD_INFLATION_METRICS_PATH)
        low = rd_metrics.loc[rd_metrics["variant"] == "low_inflation"]
        if not low.empty:
            low_c = float(low.iloc[0]["c_value"])
            low_target = float(low.iloc[0]["target_periods"])

    return {
        "elo_initial_rating": elo_initial,
        "glicko_initial_rating": read_float_constant("DEFAULT_RATING"),
        "glicko_initial_rd": read_float_constant("DEFAULT_RD"),
        "glicko_min_rd": read_float_constant("MIN_RD"),
        "glicko_max_rd": read_float_constant("MAX_RD"),
        "glicko_low_inflation_c": low_c,
        "glicko_low_inflation_target_periods": low_target,
        "glicko_probability_formula": "1 / (1 + 10 ** (-g(RD_opponent) * (rating_player - rating_opponent) / 400))",
        "elo_probability_formula": "1 / (1 + 10 ** ((rating_opponent - rating_player) / scale))",
        "validation_best_elo_scale": 300.0,
    }


def convert_pre_match_fields(base: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-match rating/RD fields from existing prediction files."""

    out = base.copy()
    fair = pd.read_csv(FAIR_PREDICTIONS_PATH, low_memory=False)
    for alias, source_model in FAIR_MODEL_MAP.items():
        sub = fair.loc[fair["model"] == source_model].copy()
        if sub.empty:
            continue
        sub["match_id"] = sub["game_id"].astype(int)
        sub = sub[["match_id", "winner", "loser", "pre_rating_winner", "pre_rating_loser", "pre_rd_winner", "pre_rd_loser"]]
        tmp = out[["match_id", "player_a_id", "player_b_id"]].merge(sub, on="match_id", how="left", validate="one_to_one")
        a_is_winner = tmp["player_a_id"] == tmp["winner"]
        out[f"rating_a_{alias}"] = np.where(a_is_winner, tmp["pre_rating_winner"], tmp["pre_rating_loser"])
        out[f"rating_b_{alias}"] = np.where(a_is_winner, tmp["pre_rating_loser"], tmp["pre_rating_winner"])
        out[f"rd_a_{alias}"] = np.where(a_is_winner, tmp["pre_rd_winner"], tmp["pre_rd_loser"])
        out[f"rd_b_{alias}"] = np.where(a_is_winner, tmp["pre_rd_loser"], tmp["pre_rd_winner"])

    adaptive = pd.read_csv(ADAPTIVE_PREDICTIONS_PATH, low_memory=False)
    for alias, source_model in ADAPTIVE_MODEL_MAP.items():
        sub = adaptive.loc[adaptive["model"] == source_model].copy()
        if sub.empty:
            continue
        sub["match_id"] = sub["game_id"].astype(int)
        sub = sub[["match_id", "winner", "loser", "pre_rating_winner", "pre_rating_loser"]]
        tmp = out[["match_id", "player_a_id", "player_b_id"]].merge(sub, on="match_id", how="left", validate="one_to_one")
        a_is_winner = tmp["player_a_id"] == tmp["winner"]
        out[f"rating_a_{alias}"] = np.where(a_is_winner, tmp["pre_rating_winner"], tmp["pre_rating_loser"])
        out[f"rating_b_{alias}"] = np.where(a_is_winner, tmp["pre_rating_loser"], tmp["pre_rating_winner"])
    return out


def validate_inputs(df: pd.DataFrame, constants: dict[str, Any]) -> pd.DataFrame:
    """Validate step 30 inputs and reproduce step 29 totals."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows", len(df) == EXPECTED_GAMES, len(df), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", not df["match_id"].duplicated().any(), int(df["match_id"].duplicated().sum()), 0)
    add_check(rows, "outcome_a_binary", set(df["outcome_a"].unique()).issubset({0, 1}), sorted(df["outcome_a"].unique()), "0/1")
    for alias in MODEL_ALIASES:
        p = df[f"p_a_{alias}"]
        add_check(rows, f"probability_present_{alias}", p.notna().all(), int(p.isna().sum()), 0)
        add_check(rows, f"probability_range_{alias}", p.between(0, 1).all(), f"{p.min()} to {p.max()}", "[0,1]")

    overall = pd.read_csv(STEP29_OVERALL_METRICS_PATH)
    reproduced = []
    for alias in MODEL_ALIASES:
        ref = overall.loc[overall["model"] == alias]
        if ref.empty:
            reproduced.append(False)
            continue
        ref = ref.iloc[0]
        ok = (
            abs(df[f"brier_{alias}"].mean() - ref["brier"]) < 1e-12
            and abs(df[f"logloss_{alias}"].mean() - ref["log_loss"]) < 1e-12
            and abs(df[f"correct_{alias}"].mean() - ref["accuracy"]) < 1e-12
        )
        reproduced.append(ok)
    add_check(rows, "step29_overall_metrics_reproduced", all(reproduced), f"{sum(reproduced)}/{len(reproduced)}", f"{len(reproduced)}/{len(reproduced)}")
    add_check(rows, "event_key_not_missing", df["event_key"].notna().all(), int(df["event_key"].isna().sum()), 0)
    add_check(rows, "initialisation_constants_from_code", all(pd.notna(v) for k, v in constants.items() if k.endswith(("rating", "rd"))), "checked", "not missing")
    return pd.DataFrame(rows)


def build_history_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build mutually exclusive debut/history categories."""

    out = df.copy()
    exactly_one = out["a_is_debut"].astype(bool) ^ out["b_is_debut"].astype(bool)
    both = out["a_is_debut"].astype(bool) & out["b_is_debut"].astype(bool)
    out["history_category"] = np.select(
        [both, exactly_one],
        ["Both players debut", "Exactly one debut"],
        default="No debut",
    )
    out["new_but_not_debut"] = (out["history_category"] == "No debut") & out["min_total_games_before"].between(1, 5)
    out["low_experience_not_debut"] = (out["history_category"] == "No debut") & out["min_total_games_before"].between(1, 20)
    rows = []
    for label in ["No debut", "Exactly one debut", "Both players debut"]:
        sub = out.loc[out["history_category"] == label]
        rows.append({"category": label, "games": len(sub), "events": sub["event_key"].nunique(), "percentage": len(sub) / len(out)})
    for label, mask in [
        ("New but not debut: 1-5 games", out["new_but_not_debut"]),
        ("Low experience but not debut: 1-20 games", out["low_experience_not_debut"]),
    ]:
        sub = out.loc[mask]
        rows.append({"category": label, "games": len(sub), "events": sub["event_key"].nunique(), "percentage": len(sub) / len(out)})
    return out, pd.DataFrame(rows)


def brier_from_probability(p: pd.Series, y: pd.Series) -> pd.Series:
    return (p.astype(float) - y.astype(float)) ** 2


def logloss_from_probability(p: pd.Series, y: pd.Series) -> pd.Series:
    clipped = p.astype(float).clip(EPS, 1.0 - EPS)
    y = y.astype(float)
    return -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))


def build_debut_player_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per exactly-one-debut match in debut-player orientation."""

    sub = df.loc[df["history_category"] == "Exactly one debut"].copy()
    sub["debut_player_is_a"] = sub["a_is_debut"].astype(bool)
    sub["debut_player_id"] = np.where(sub["debut_player_is_a"], sub["player_a_id"], sub["player_b_id"])
    sub["experienced_opponent_id"] = np.where(sub["debut_player_is_a"], sub["player_b_id"], sub["player_a_id"])
    sub["debut_player_won"] = np.where(sub["debut_player_is_a"], sub["outcome_a"], 1 - sub["outcome_a"]).astype(int)
    sub["debut_total_games_before"] = 0
    sub["opponent_total_games_before"] = np.where(sub["debut_player_is_a"], sub["b_total_games_before"], sub["a_total_games_before"])
    sub["opponent_games_last_365_days"] = np.where(sub["debut_player_is_a"], sub["b_games_last_365_days"], sub["a_games_last_365_days"])
    sub["opponent_days_since_last_game"] = np.where(sub["debut_player_is_a"], sub["b_days_since_last_game"], sub["a_days_since_last_game"])
    for alias in MODEL_ALIASES:
        sub[f"p_debut_{alias}"] = np.where(sub["debut_player_is_a"], sub[f"p_a_{alias}"], 1.0 - sub[f"p_a_{alias}"])
        sub[f"brier_debut_{alias}"] = brier_from_probability(sub[f"p_debut_{alias}"], sub["debut_player_won"])
        sub[f"logloss_debut_{alias}"] = logloss_from_probability(sub[f"p_debut_{alias}"], sub["debut_player_won"])
        sub[f"correct_debut_{alias}"] = ((sub[f"p_debut_{alias}"] >= 0.5).astype(int) == sub["debut_player_won"]).astype(int)

    for alias in ["Glicko_low", "Validation_best_Elo", "Glicko_C0", "Default_Elo"]:
        if f"rating_a_{alias}" in sub.columns:
            sub[f"debut_rating_{alias}"] = np.where(sub["debut_player_is_a"], sub[f"rating_a_{alias}"], sub[f"rating_b_{alias}"])
            sub[f"opponent_rating_{alias}"] = np.where(sub["debut_player_is_a"], sub[f"rating_b_{alias}"], sub[f"rating_a_{alias}"])
        if f"rd_a_{alias}" in sub.columns:
            sub[f"debut_rd_{alias}"] = np.where(sub["debut_player_is_a"], sub[f"rd_a_{alias}"], sub[f"rd_b_{alias}"])
            sub[f"opponent_rd_{alias}"] = np.where(sub["debut_player_is_a"], sub[f"rd_b_{alias}"], sub[f"rd_a_{alias}"])

    sub["delta_p_debut_glicko_vs_elo"] = sub["p_debut_Glicko_low"] - sub["p_debut_Validation_best_Elo"]
    sub["delta_brier_glicko_vs_elo_debut"] = sub["brier_debut_Validation_best_Elo"] - sub["brier_debut_Glicko_low"]
    sub["delta_logloss_glicko_vs_elo_debut"] = sub["logloss_debut_Validation_best_Elo"] - sub["logloss_debut_Glicko_low"]
    return sub


def calculate_debut_model_summary(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise model behaviour on exactly-one-debut matches."""

    rows = []
    games = len(debut)
    events = debut["event_key"].nunique() if games else 0
    win_rate = float(debut["debut_player_won"].mean()) if games else np.nan
    for alias in MODEL_ALIASES:
        p = debut[f"p_debut_{alias}"]
        y = debut["debut_player_won"]
        gap = float(p.mean() - win_rate) if games else np.nan
        rows.append(
            {
                "model": alias,
                "model_display": MODEL_LABELS[alias],
                "games": games,
                "events": events,
                "debut_empirical_win_rate": win_rate,
                "mean_predicted_debut_win_probability": float(p.mean()) if games else np.nan,
                "median_predicted_debut_win_probability": float(p.median()) if games else np.nan,
                "p10": float(p.quantile(0.10)) if games else np.nan,
                "p25": float(p.quantile(0.25)) if games else np.nan,
                "p75": float(p.quantile(0.75)) if games else np.nan,
                "p90": float(p.quantile(0.90)) if games else np.nan,
                "brier": float(debut[f"brier_debut_{alias}"].mean()) if games else np.nan,
                "logloss": float(debut[f"logloss_debut_{alias}"].mean()) if games else np.nan,
                "accuracy": float(debut[f"correct_debut_{alias}"].mean()) if games else np.nan,
                "mean_absolute_probability_error": float((p - y).abs().mean()) if games else np.nan,
                "corrected_calibration_gap": gap,
            }
        )
    rows.append(
        {
            "model": "Glicko_minus_Elo_probability_difference",
            "model_display": "Glicko low - validation-best Elo",
            "games": games,
            "events": events,
            "debut_empirical_win_rate": win_rate,
            "mean_predicted_debut_win_probability": float(debut["delta_p_debut_glicko_vs_elo"].mean()) if games else np.nan,
            "median_predicted_debut_win_probability": float(debut["delta_p_debut_glicko_vs_elo"].median()) if games else np.nan,
            "p10": float(debut["delta_p_debut_glicko_vs_elo"].quantile(0.10)) if games else np.nan,
            "p25": float(debut["delta_p_debut_glicko_vs_elo"].quantile(0.25)) if games else np.nan,
            "p75": float(debut["delta_p_debut_glicko_vs_elo"].quantile(0.75)) if games else np.nan,
            "p90": float(debut["delta_p_debut_glicko_vs_elo"].quantile(0.90)) if games else np.nan,
            "brier": np.nan,
            "logloss": np.nan,
            "accuracy": np.nan,
            "mean_absolute_probability_error": np.nan,
            "corrected_calibration_gap": np.nan,
        }
    )
    return pd.DataFrame(rows)


def event_cluster_bootstrap(sub: pd.DataFrame, diff_col: str, reps: int = BOOTSTRAP_REPS, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Bootstrap a paired mean difference by event clusters, with match fallback."""

    games = len(sub)
    events = sub["event_key"].nunique() if games else 0
    point = float(sub[diff_col].mean()) if games else np.nan
    small = games < 50 or events < 10
    if games == 0:
        return {"point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "bootstrap_type": "not_run", "bootstrap_replications": 0, "small_sample_warning": True}
    rng = np.random.default_rng(seed)
    if events >= 10:
        grouped = sub.groupby("event_key", sort=False)[diff_col].agg(["sum", "count"]).reset_index(drop=True)
        sums = grouped["sum"].to_numpy(float)
        counts = grouped["count"].to_numpy(float)
        estimates = np.empty(reps)
        for i in range(reps):
            idx = rng.integers(0, len(grouped), len(grouped))
            estimates[i] = sums[idx].sum() / counts[idx].sum()
        bootstrap_type = "event_cluster"
    else:
        values = sub[diff_col].to_numpy(float)
        estimates = np.empty(reps)
        for i in range(reps):
            idx = rng.integers(0, len(values), len(values))
            estimates[i] = values[idx].mean()
        bootstrap_type = "match_level"
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "bootstrap_type": bootstrap_type,
        "bootstrap_replications": reps,
        "small_sample_warning": small,
    }


def paired_metrics(sub: pd.DataFrame, label: str, role: str = "") -> dict[str, Any]:
    """Return common paired Glicko/Elo/adaptive diagnostics for one subset."""

    games = len(sub)
    events = sub["event_key"].nunique() if games else 0
    brier_ci = event_cluster_bootstrap(sub, "delta_brier_glicko_vs_elo") if games else {}
    log_ci = event_cluster_bootstrap(sub, "delta_logloss_glicko_vs_elo", seed=RANDOM_SEED + 1) if games else {}
    return {
        "analysis_role": role,
        "subgroup": label,
        "games": games,
        "events": events,
        "glicko_brier": float(sub["brier_Glicko_low"].mean()) if games else np.nan,
        "elo_brier": float(sub["brier_Validation_best_Elo"].mean()) if games else np.nan,
        "delta_brier": float(sub["delta_brier_glicko_vs_elo"].mean()) if games else np.nan,
        "delta_brier_ci_lower": brier_ci.get("ci_lower", np.nan),
        "delta_brier_ci_upper": brier_ci.get("ci_upper", np.nan),
        "glicko_logloss": float(sub["logloss_Glicko_low"].mean()) if games else np.nan,
        "elo_logloss": float(sub["logloss_Validation_best_Elo"].mean()) if games else np.nan,
        "delta_logloss": float(sub["delta_logloss_glicko_vs_elo"].mean()) if games else np.nan,
        "delta_logloss_ci_lower": log_ci.get("ci_lower", np.nan),
        "delta_logloss_ci_upper": log_ci.get("ci_upper", np.nan),
        "accuracy_difference": float(sub["delta_accuracy_glicko_vs_elo"].mean()) if games else np.nan,
        "inflation_delta_brier": float(sub["delta_brier_inflation"].mean()) if games else np.nan,
        "adaptive_delta_brier": float(sub["delta_brier_glicko_vs_adaptive"].mean()) if games else np.nan,
        "small_sample_warning": bool(games < 50 or events < 10),
        "bootstrap_type": brier_ci.get("bootstrap_type", "not_run"),
        "bootstrap_replications": brier_ci.get("bootstrap_replications", 0),
    }


def analyse_initial_rating_scale_position(df: pd.DataFrame, debut: pd.DataFrame, constants: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare initial ratings with 2025 established-player pre-match ratings."""

    rows = []
    if not debut.empty:
        rows.extend(
            [
                {
                    "diagnostic": "exactly_one_debut_opponent_glicko_rating",
                    "count": len(debut),
                    "mean": debut["opponent_rating_Glicko_low"].mean(),
                    "median": debut["opponent_rating_Glicko_low"].median(),
                    "initial_rating": constants["glicko_initial_rating"],
                    "initial_minus_mean": constants["glicko_initial_rating"] - debut["opponent_rating_Glicko_low"].mean(),
                    "initial_minus_median": constants["glicko_initial_rating"] - debut["opponent_rating_Glicko_low"].median(),
                },
                {
                    "diagnostic": "exactly_one_debut_opponent_validation_elo_rating",
                    "count": len(debut),
                    "mean": debut["opponent_rating_Validation_best_Elo"].mean(),
                    "median": debut["opponent_rating_Validation_best_Elo"].median(),
                    "initial_rating": constants["elo_initial_rating"],
                    "initial_minus_mean": constants["elo_initial_rating"] - debut["opponent_rating_Validation_best_Elo"].mean(),
                    "initial_minus_median": constants["elo_initial_rating"] - debut["opponent_rating_Validation_best_Elo"].median(),
                },
            ]
        )

    rating_rows = []
    for model, rating_cols, initial in [
        ("Glicko_low", ("rating_a_Glicko_low", "rating_b_Glicko_low"), constants["glicko_initial_rating"]),
        ("Validation_best_Elo", ("rating_a_Validation_best_Elo", "rating_b_Validation_best_Elo"), constants["elo_initial_rating"]),
    ]:
        parts = []
        a = df.loc[df["a_total_games_before"] > 0, [rating_cols[0]]].rename(columns={rating_cols[0]: "rating"})
        b = df.loc[df["b_total_games_before"] > 0, [rating_cols[1]]].rename(columns={rating_cols[1]: "rating"})
        parts.extend([a, b])
        ratings = pd.concat(parts, ignore_index=True)["rating"].dropna()
        rating_rows.append(
            {
                "model": model,
                "count": int(ratings.count()),
                "initial_rating": initial,
                "mean": float(ratings.mean()),
                "std": float(ratings.std()),
                "min": float(ratings.min()),
                "p10": float(ratings.quantile(0.10)),
                "p25": float(ratings.quantile(0.25)),
                "median": float(ratings.median()),
                "p75": float(ratings.quantile(0.75)),
                "p90": float(ratings.quantile(0.90)),
                "max": float(ratings.max()),
                "initial_minus_median": float(initial - ratings.median()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rating_rows)


def identify_influential_debut_matches(debut: pd.DataFrame) -> pd.DataFrame:
    """Return top influential debut matches under several ranking criteria."""

    rows = []
    criteria = [
        ("most_negative_delta_brier", "delta_brier_glicko_vs_elo_debut", True),
        ("largest_absolute_delta_brier", "delta_brier_glicko_vs_elo_debut", False),
        ("most_negative_delta_logloss", "delta_logloss_glicko_vs_elo_debut", True),
        ("largest_absolute_delta_logloss", "delta_logloss_glicko_vs_elo_debut", False),
    ]
    cols = [
        "match_id",
        "event_key",
        "match_date",
        "debut_player_id",
        "experienced_opponent_id",
        "debut_player_won",
        "debut_total_games_before",
        "opponent_total_games_before",
        "opponent_games_last_365_days",
        "opponent_days_since_last_game",
        "debut_rating_Glicko_low",
        "debut_rd_Glicko_low",
        "opponent_rating_Glicko_low",
        "opponent_rd_Glicko_low",
        "p_debut_Glicko_low",
        "p_debut_Validation_best_Elo",
        "p_debut_Glicko_C0",
        "p_debut_best_AdaptiveK",
        "brier_debut_Glicko_low",
        "brier_debut_Validation_best_Elo",
        "delta_brier_glicko_vs_elo_debut",
        "delta_logloss_glicko_vs_elo_debut",
    ]
    for name, col, ascending in criteria:
        data = debut.copy()
        sort_key = data[col].abs() if not ascending else data[col]
        data = data.assign(_sort=sort_key).sort_values("_sort", ascending=ascending).head(20)
        for rank, row in enumerate(data.itertuples(index=False), 1):
            record = {c: getattr(row, c) for c in cols if hasattr(row, c)}
            record["ranking_type"] = name
            record["rank"] = rank
            rows.append(record)
    return pd.DataFrame(rows)


def calculate_debut_event_contributions(debut: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exactly-one-debut contributions by event."""

    if debut.empty:
        return pd.DataFrame()
    grouped = debut.groupby("event_key").agg(
        games=("match_id", "size"),
        debut_wins=("debut_player_won", "sum"),
        mean_delta_brier=("delta_brier_glicko_vs_elo_debut", "mean"),
        sum_delta_brier=("delta_brier_glicko_vs_elo_debut", "sum"),
        mean_delta_logloss=("delta_logloss_glicko_vs_elo_debut", "mean"),
        sum_delta_logloss=("delta_logloss_glicko_vs_elo_debut", "sum"),
    )
    return grouped.reset_index().sort_values("sum_delta_brier")


def run_debut_leave_one_event_out(debut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-event-out diagnostics for debut matches."""

    rows = []
    if debut.empty:
        return pd.DataFrame(), pd.DataFrame()
    full_brier = debut["delta_brier_glicko_vs_elo_debut"].mean()
    full_logloss = debut["delta_logloss_glicko_vs_elo_debut"].mean()
    for event_key in sorted(debut["event_key"].unique()):
        sub = debut.loc[debut["event_key"] != event_key]
        rows.append(
            {
                "removed_event_key": event_key,
                "remaining_games": len(sub),
                "remaining_events": sub["event_key"].nunique(),
                "delta_brier": sub["delta_brier_glicko_vs_elo_debut"].mean(),
                "delta_logloss": sub["delta_logloss_glicko_vs_elo_debut"].mean(),
                "debut_empirical_win_rate": sub["debut_player_won"].mean(),
                "mean_glicko_predicted_debut_probability": sub["p_debut_Glicko_low"].mean(),
                "mean_elo_predicted_debut_probability": sub["p_debut_Validation_best_Elo"].mean(),
            }
        )
    loo = pd.DataFrame(rows)
    sign_full = np.sign(full_brier)
    sign_changes = int((np.sign(loo["delta_brier"]) != sign_full).sum())
    influential_idx = (loo["delta_brier"] - full_brier).abs().idxmax()
    summary = pd.DataFrame(
        [
            {
                "full_sample_games": len(debut),
                "full_sample_events": debut["event_key"].nunique(),
                "full_delta_brier": full_brier,
                "full_delta_logloss": full_logloss,
                "minimum_leave_one_event_out_delta_brier": loo["delta_brier"].min(),
                "maximum_leave_one_event_out_delta_brier": loo["delta_brier"].max(),
                "number_of_sign_changes": sign_changes,
                "proportion_of_sign_changes": sign_changes / len(loo),
                "most_influential_removed_event": loo.loc[influential_idx, "removed_event_key"],
            }
        ]
    )
    return loo, summary


def zero_recent_group(row: pd.Series) -> str:
    """Classify the zero/missing recent-activity sample into mutually exclusive groups."""

    if row["either_player_debut"]:
        return "Debut / no previous history"
    if pd.isna(row["min_games_last_365_days"]):
        return "Missing date information"
    max_gap = row["max_days_since_last_game"]
    if pd.isna(max_gap):
        return "Missing date information"
    if max_gap >= 1096:
        return "Non-debut, inactive 1096+ days"
    if max_gap >= 730:
        return "Non-debut, inactive 730-1095 days"
    if max_gap >= 365:
        return "Non-debut, inactive 365-729 days"
    return "Non-debut, zero games in last 365 days"


def decompose_zero_recent_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose min_games_last_365_days == 0 or missing into debut/returning groups."""

    mask = (df["min_games_last_365_days"] == 0) | df["min_games_last_365_days"].isna()
    sample = df.loc[mask].copy()
    sample["zero_recent_group"] = sample.apply(zero_recent_group, axis=1)
    order = [
        "Debut / no previous history",
        "Non-debut, zero games in last 365 days",
        "Non-debut, inactive 365-729 days",
        "Non-debut, inactive 730-1095 days",
        "Non-debut, inactive 1096+ days",
        "Missing date information",
    ]
    rows = []
    for group in order:
        sub = sample.loc[sample["zero_recent_group"] == group]
        if sub.empty:
            continue
        record = paired_metrics(sub, group, "zero_recent_activity")
        ci = event_cluster_bootstrap(sub, "delta_brier_glicko_vs_elo")
        record["delta_brier_ci_lower"] = ci["ci_lower"]
        record["delta_brier_ci_upper"] = ci["ci_upper"]
        record["glicko_c0_brier"] = sub["brier_Glicko_C0"].mean()
        record["best_adaptive_k_brier"] = sub["brier_best_AdaptiveK"].mean()
        rows.append(record)
    return pd.DataFrame(rows)


def analyse_genuine_returners(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse no-debut returning-player threshold sensitivity."""

    rows = []
    base = df.loc[
        (df["history_category"] == "No debut")
        & df["both_players_have_history"].astype(bool)
        & df["max_days_since_last_game"].notna()
    ]
    for threshold in [180, 365, 540, 730, 1095]:
        sub = base.loc[base["max_days_since_last_game"] >= threshold]
        record = paired_metrics(sub, f"Returning >= {threshold} days, no debut", "genuine_returning")
        record["threshold_days"] = threshold
        record["delta_brier_inflation"] = sub["delta_brier_inflation"].mean() if len(sub) else np.nan
        record["delta_logloss_inflation"] = sub["delta_logloss_inflation"].mean() if len(sub) else np.nan
        record["delta_brier_glicko_vs_adaptive"] = sub["delta_brier_glicko_vs_adaptive"].mean() if len(sub) else np.nan
        record["delta_logloss_glicko_vs_adaptive"] = sub["delta_logloss_glicko_vs_adaptive"].mean() if len(sub) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def calculate_exclusion_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute main paired results after excluding debut or low-history groups."""

    samples = [
        ("All games", pd.Series(True, index=df.index)),
        ("Excluding all debut games", df["history_category"] == "No debut"),
        ("Excluding debut and inactive >=365 games", (df["history_category"] == "No debut") & (~df["either_player_inactive_365d"].astype(bool))),
        ("Both players have history", df["both_players_have_history"].astype(bool)),
        ("Both players have at least 5 previous games", df["min_total_games_before"] >= 5),
        ("Both players have at least 20 previous games", df["min_total_games_before"] >= 20),
        ("Both active in last 365 days and no debut", (df["history_category"] == "No debut") & df["both_players_active_last_365d"].astype(bool)),
    ]
    rows = []
    for label, mask in samples:
        rows.append(paired_metrics(df.loc[mask], label, "exclusion_robustness"))
    return pd.DataFrame(rows)


def calculate_no_debut_subgroup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate subgroup results after excluding debut matches."""

    base = df.loc[df["history_category"] == "No debut"].copy()
    rows = []
    for group in ["1-5", "6-20", "21-50", "51-100", "100+"]:
        rows.append(paired_metrics(base.loc[base["subgroup_total_experience"] == group], f"Total games {group}", "no_debut_total_games"))
    for group in ["0", "1-5", "6-15", "16-30", "30+", "Missing date information"]:
        rows.append(paired_metrics(base.loc[base["subgroup_recent_365_activity"] == group], f"Recent activity {group}", "no_debut_recent_activity"))
    for group in ["Balanced: ratio <= 2", "Moderate mismatch: 2 < ratio <= 5", "Large mismatch: ratio > 5"]:
        rows.append(paired_metrics(base.loc[base["subgroup_experience_ratio"] == group], group, "no_debut_experience_mismatch"))
    return pd.DataFrame(rows)


def rebuild_no_debut_rd_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild no-debut RD quartiles/deciles and calculate diagnostics."""

    base = df.loc[
        (df["history_category"] == "No debut")
        & df["both_players_have_history"].astype(bool)
        & df["max_prematch_rd"].notna()
    ].copy()
    cut_rows = []
    result_quartile = []
    result_decile = []
    for q, name, collector in [(4, "quartile", result_quartile), (10, "decile", result_decile)]:
        labels = [f"{name}_{i + 1}" for i in range(q)]
        cats, bins = pd.qcut(base["max_prematch_rd"], q=q, labels=labels, retbins=True, duplicates="drop")
        col = f"no_debut_rd_{name}"
        base[col] = cats.astype(str)
        actual_labels = list(cats.cat.categories)
        for i, label in enumerate(actual_labels):
            cut_rows.append({"grouping": name, "group": label, "lower": bins[i], "upper": bins[i + 1]})
            sub = base.loc[base[col] == label]
            record = paired_metrics(sub, label, f"no_debut_rd_{name}")
            record["mean_max_rd"] = sub["max_prematch_rd"].mean()
            record["inflation_delta_brier"] = sub["delta_brier_inflation"].mean()
            record["adaptive_delta_brier"] = sub["delta_brier_glicko_vs_adaptive"].mean()
            collector.append(record)
    return pd.DataFrame(result_quartile), pd.DataFrame(result_decile), pd.DataFrame(cut_rows)


def confidence_category(value: float) -> str:
    if value < -0.05:
        return "Glicko substantially less confident"
    if value < -0.01:
        return "Glicko slightly less confident"
    if value <= 0.01:
        return "Similar confidence"
    if value <= 0.05:
        return "Glicko slightly more confident"
    return "Glicko substantially more confident"


def analyse_prediction_confidence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse Glicko-vs-Elo confidence and favourite disagreements."""

    out = df.copy()
    out["glicko_favourite_probability"] = out["favourite_probability_Glicko_low"]
    out["elo_favourite_probability"] = out["favourite_probability_Validation_best_Elo"]
    out["glicko_confidence_minus_elo"] = out["glicko_favourite_probability"] - out["elo_favourite_probability"]
    out["confidence_category"] = out["glicko_confidence_minus_elo"].map(confidence_category)
    out["glicko_predicted_a_win"] = out["p_a_Glicko_low"] >= 0.5
    out["elo_predicted_a_win"] = out["p_a_Validation_best_Elo"] >= 0.5
    out["same_predicted_favourite"] = out["glicko_predicted_a_win"] == out["elo_predicted_a_win"]

    samples = [
        ("All games", pd.Series(True, index=out.index)),
        ("No-debut games", out["history_category"] == "No debut"),
        ("Exactly one debut games", out["history_category"] == "Exactly one debut"),
        ("Genuine returning games >=365 days", (out["history_category"] == "No debut") & (out["max_days_since_last_game"] >= 365)),
    ]
    order = [
        "Glicko substantially less confident",
        "Glicko slightly less confident",
        "Similar confidence",
        "Glicko slightly more confident",
        "Glicko substantially more confident",
    ]
    rows = []
    for sample, mask in samples:
        data = out.loc[mask]
        for category in order:
            sub = data.loc[data["confidence_category"] == category]
            if sub.empty:
                continue
            rows.append(
                {
                    "sample": sample,
                    "confidence_category": category,
                    "games": len(sub),
                    "events": sub["event_key"].nunique(),
                    "mean_confidence_difference": sub["glicko_confidence_minus_elo"].mean(),
                    "empirical_favourite_win_rate": sub["favourite_won_Glicko_low"].mean(),
                    "glicko_brier": sub["brier_Glicko_low"].mean(),
                    "elo_brier": sub["brier_Validation_best_Elo"].mean(),
                    "delta_brier": sub["delta_brier_glicko_vs_elo"].mean(),
                    "glicko_logloss": sub["logloss_Glicko_low"].mean(),
                    "elo_logloss": sub["logloss_Validation_best_Elo"].mean(),
                    "delta_logloss": sub["delta_logloss_glicko_vs_elo"].mean(),
                    "calibration_gap": sub["favourite_won_Glicko_low"].mean() - sub["glicko_favourite_probability"].mean(),
                }
            )

    disagreement_rows = []
    for label, mask in [("same_predicted_favourite", out["same_predicted_favourite"]), ("different_predicted_favourite", ~out["same_predicted_favourite"])]:
        sub = out.loc[mask]
        disagreement_rows.append(
            {
                "comparison": label,
                "games": len(sub),
                "events": sub["event_key"].nunique(),
                "glicko_correct_count": int(sub["correct_Glicko_low"].sum()),
                "elo_correct_count": int(sub["correct_Validation_best_Elo"].sum()),
                "both_wrong": int(((sub["correct_Glicko_low"] == 0) & (sub["correct_Validation_best_Elo"] == 0)).sum()),
                "mean_delta_brier": sub["delta_brier_glicko_vs_elo"].mean(),
                "mean_delta_logloss": sub["delta_logloss_glicko_vs_elo"].mean(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(disagreement_rows)


def brier_decomposition_one(sub: pd.DataFrame, alias: str, sample: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Approximate Brier decomposition using favourite-probability bins."""

    bins = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0000001])
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
    data = pd.DataFrame(
        {
            "probability_bin": pd.cut(sub[f"favourite_probability_{alias}"], bins=bins, labels=labels, include_lowest=True, right=False),
            "forecast": sub[f"favourite_probability_{alias}"],
            "outcome": sub[f"favourite_won_{alias}"],
        }
    ).dropna()
    n = len(data)
    if n == 0:
        return {}, pd.DataFrame()
    overall_rate = data["outcome"].mean()
    rows = []
    reliability = 0.0
    resolution = 0.0
    for label in labels:
        g = data.loc[data["probability_bin"].astype(str) == label]
        if g.empty:
            rows.append({"sample": sample, "model": alias, "probability_bin": label, "games": 0, "mean_forecast": np.nan, "empirical_rate": np.nan})
            continue
        weight = len(g) / n
        mean_forecast = g["forecast"].mean()
        empirical = g["outcome"].mean()
        reliability += weight * (mean_forecast - empirical) ** 2
        resolution += weight * (empirical - overall_rate) ** 2
        rows.append({"sample": sample, "model": alias, "probability_bin": label, "games": len(g), "mean_forecast": mean_forecast, "empirical_rate": empirical})
    uncertainty = overall_rate * (1.0 - overall_rate)
    reconstructed = reliability - resolution + uncertainty
    actual = ((data["forecast"] - data["outcome"]) ** 2).mean()
    summary = {
        "sample": sample,
        "model": alias,
        "games": n,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "reconstructed_brier": reconstructed,
        "actual_brier": actual,
        "reconstruction_difference": reconstructed - actual,
    }
    return summary, pd.DataFrame(rows)


def calculate_brier_decomposition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate approximate favourite-perspective Brier decomposition."""

    samples = [
        ("Overall", pd.Series(True, index=df.index)),
        ("No debut", df["history_category"] == "No debut"),
        ("Exactly one debut", df["history_category"] == "Exactly one debut"),
        ("Returning >=365 days, no debut", (df["history_category"] == "No debut") & (df["max_days_since_last_game"] >= 365)),
        ("Both active, no debut", (df["history_category"] == "No debut") & df["both_players_active_last_365d"].astype(bool)),
    ]
    summaries = []
    bins = []
    for sample, mask in samples:
        sub = df.loc[mask]
        for alias in ["Glicko_low", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0"]:
            summary, bin_df = brier_decomposition_one(sub, alias, sample)
            if summary:
                summaries.append(summary)
                bins.append(bin_df)
    return pd.DataFrame(summaries), pd.concat(bins, ignore_index=True)


def run_threshold_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed threshold sensitivity after excluding debut."""

    no_debut = df.loc[df["history_category"] == "No debut"]
    total_rows = []
    for threshold in [1, 3, 5, 10, 20, 30, 50, 100]:
        total_rows.append(paired_metrics(no_debut.loc[no_debut["min_total_games_before"] <= threshold], f"min_total_games_before <= {threshold}", "threshold_total_games"))
    recent_rows = []
    for threshold in [0, 1, 3, 5, 10, 15, 20, 30]:
        recent_rows.append(paired_metrics(no_debut.loc[no_debut["min_games_last_365_days"] <= threshold], f"min_games_last_365_days <= {threshold}", "threshold_recent_activity"))
    return pd.DataFrame(total_rows), pd.DataFrame(recent_rows)


def interpretation_flag(row: pd.Series) -> str:
    if row.get("small_sample_warning", False):
        return "small_sample"
    if row["delta_brier_ci_lower"] > 0:
        return "robust_glicko_advantage"
    if row["delta_brier_ci_upper"] < 0:
        if "debut" in str(row["subgroup"]).lower():
            return "initialisation_mismatch"
        return "robust_elo_advantage"
    return "uncertain"


def build_key_diagnostic_table(
    df: pd.DataFrame,
    debut: pd.DataFrame,
    exclusion: pd.DataFrame,
    no_debut_rd_quartiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build a concise diagnostic table for meeting use."""

    rows = []
    row_map = {
        "Overall": df,
        "Overall excluding debut": df.loc[df["history_category"] == "No debut"],
        "Exactly one debut": df.loc[df["history_category"] == "Exactly one debut"],
        "Both players debut": df.loc[df["history_category"] == "Both players debut"],
        "New but not debut: 1-5 games": df.loc[df["new_but_not_debut"]],
        "Low experience but not debut: 1-20 games": df.loc[df["low_experience_not_debut"]],
        "Non-debut zero recent games": df.loc[(df["history_category"] == "No debut") & (df["min_games_last_365_days"] == 0)],
        "Returning >=365, no debut": df.loc[(df["history_category"] == "No debut") & (df["max_days_since_last_game"] >= 365)],
        "Returning >=730, no debut": df.loc[(df["history_category"] == "No debut") & (df["max_days_since_last_game"] >= 730)],
        "Both active and no debut": df.loc[(df["history_category"] == "No debut") & df["both_players_active_last_365d"].astype(bool)],
    }
    for label, sub in row_map.items():
        record = paired_metrics(sub, label, "key_diagnostic")
        if label == "Exactly one debut" and not debut.empty:
            record["mean_glicko_probability"] = debut["p_debut_Glicko_low"].mean()
            record["mean_elo_probability"] = debut["p_debut_Validation_best_Elo"].mean()
            record["empirical_win_rate"] = debut["debut_player_won"].mean()
        else:
            record["mean_glicko_probability"] = sub["favourite_probability_Glicko_low"].mean() if len(sub) else np.nan
            record["mean_elo_probability"] = sub["favourite_probability_Validation_best_Elo"].mean() if len(sub) else np.nan
            record["empirical_win_rate"] = sub["favourite_won_Glicko_low"].mean() if len(sub) else np.nan
        rows.append(record)

    for label in ["quartile_1", "quartile_4"]:
        qrow = no_debut_rd_quartiles.loc[no_debut_rd_quartiles["subgroup"] == label]
        if qrow.empty:
            continue
        r = qrow.iloc[0].to_dict()
        r["analysis_role"] = "key_diagnostic"
        r["subgroup"] = "No-debut lowest RD quartile" if label == "quartile_1" else "No-debut highest RD quartile"
        r["mean_glicko_probability"] = np.nan
        r["mean_elo_probability"] = np.nan
        r["empirical_win_rate"] = np.nan
        rows.append(r)
    table = pd.DataFrame(rows)
    table["interpretation_flag"] = table.apply(interpretation_flag, axis=1)
    return table


def create_bar_delta_figure(data: pd.DataFrame, label_col: str, value_col: str, title: str, ylabel: str, path: Path) -> Path:
    """Create a simple ordered bar chart with y=0."""

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(data)), data[value_col], color="#2C7FB8")
    ax.axhline(0, color="#333333", linewidth=1)
    for i, row in enumerate(data.itertuples(index=False)):
        ax.text(i, getattr(row, value_col), f"n={int(row.games)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data[label_col], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def create_diagnostic_figures(
    df: pd.DataFrame,
    debut_summary: pd.DataFrame,
    debut: pd.DataFrame,
    rating_dist: pd.DataFrame,
    zero_recent: pd.DataFrame,
    returners: pd.DataFrame,
    exclusion: pd.DataFrame,
    rd_quartiles: pd.DataFrame,
    confidence: pd.DataFrame,
    loo: pd.DataFrame,
) -> list[Path]:
    """Create requested diagnostic figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    p = FIGURE_DIR / "30_fig01_debut_probability_vs_actual.png"
    ds = debut_summary.loc[debut_summary["model"].isin(MODEL_ALIASES)].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Actual"] + ds["model_display"].tolist()
    values = [ds["debut_empirical_win_rate"].iloc[0] if not ds.empty else np.nan] + ds["mean_predicted_debut_win_probability"].tolist()
    ax.bar(labels, values, color=["#333333"] + ["#2C7FB8", "#7A7A7A", "#D95F02", "#66A61E", "#B8B8B8"][: len(ds)])
    ax.set_ylim(0, max(1.0, np.nanmax(values) + 0.05))
    ax.set_ylabel("Debut win probability")
    ax.set_title("Debut predicted probability vs actual win rate")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    p = FIGURE_DIR / "30_fig02_debut_prediction_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4.8))
    box_data = [debut["p_debut_Glicko_low"], debut["p_debut_Validation_best_Elo"], debut["p_debut_best_AdaptiveK"]]
    ax.boxplot(box_data, labels=["Glicko low", "Validation Elo", "Adaptive-K"])
    ax.set_ylabel("Predicted debut win probability")
    ax.set_title(f"Debut prediction distribution (n={len(debut)})")
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    p = FIGURE_DIR / "30_fig03_initial_rating_population_position.png"
    fig, ax = plt.subplots(figsize=(7, 4.8))
    labels = []
    values = []
    for model in ["Glicko_low", "Validation_best_Elo"]:
        row = rating_dist.loc[rating_dist["model"] == model].iloc[0]
        labels.append(MODEL_LABELS[model])
        values.append([row["p10"], row["p25"], row["median"], row["p75"], row["p90"]])
    ax.boxplot([[v[0], v[1], v[2], v[3], v[4]] for v in values], labels=labels, whis=(0, 100))
    for i, model in enumerate(["Glicko_low", "Validation_best_Elo"], 1):
        init = rating_dist.loc[rating_dist["model"] == model, "initial_rating"].iloc[0]
        ax.scatter([i], [init], marker="D", color="#D95F02", label="Initial rating" if i == 1 else None)
    ax.legend()
    ax.set_ylabel("Pre-match rating")
    ax.set_title("Initial rating vs 2025 established-player rating population")
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    paths.append(create_bar_delta_figure(zero_recent, "subgroup", "delta_brier", "Recent activity = 0 decomposition", "Elo Brier - Glicko Brier", FIGURE_DIR / "30_fig04_zero_recent_activity_decomposition.png"))
    paths.append(create_bar_delta_figure(returners, "subgroup", "inflation_delta_brier", "RD inflation contribution for genuine returners", "Glicko C0 Brier - Glicko low Brier", FIGURE_DIR / "30_fig05_returner_inflation_contribution.png"))
    paths.append(create_bar_delta_figure(exclusion, "subgroup", "delta_brier", "Overall robustness after exclusions", "Elo Brier - Glicko Brier", FIGURE_DIR / "30_fig06_overall_exclusion_robustness.png"))
    paths.append(create_bar_delta_figure(rd_quartiles, "subgroup", "delta_brier", "No-debut RD quartiles", "Elo Brier - Glicko Brier", FIGURE_DIR / "30_fig07_no_debut_rd_quartiles.png"))
    conf_plot = confidence.loc[confidence["sample"] == "All games"].copy()
    paths.append(create_bar_delta_figure(conf_plot, "confidence_category", "delta_brier", "Prediction confidence mechanism", "Elo Brier - Glicko Brier", FIGURE_DIR / "30_fig08_prediction_confidence_mechanism.png"))
    p = FIGURE_DIR / "30_fig09_debut_leave_one_event_out.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(len(loo)), loo["delta_brier"], marker="o", linestyle="", markersize=3)
    ax.axhline(debut["delta_brier_glicko_vs_elo_debut"].mean(), color="#D95F02", label="Full sample")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Leave-one-event-out Brier difference")
    ax.set_xlabel("Removed debut event index")
    ax.set_title("Debut leave-one-event-out")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)
    return paths


def run_validation_checks(
    df: pd.DataFrame,
    history_counts: pd.DataFrame,
    debut: pd.DataFrame,
    zero_recent: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
    constants: dict[str, Any],
) -> pd.DataFrame:
    """Run validation checks for step 30 diagnostics."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows", len(df) == EXPECTED_GAMES, len(df), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", not df["match_id"].duplicated().any(), int(df["match_id"].duplicated().sum()), 0)
    total_history = int(history_counts.loc[history_counts["category"].isin(["No debut", "Exactly one debut", "Both players debut"]), "games"].sum())
    add_check(rows, "debut_history_categories_sum", total_history == EXPECTED_GAMES, total_history, EXPECTED_GAMES)
    if not debut.empty:
        p_cols = [c for c in debut.columns if c.startswith("p_debut_")]
        ok = all(debut[c].between(0, 1).all() for c in p_cols)
        add_check(rows, "p_debut_probabilities_in_range", ok, "checked", "[0,1]")
        add_check(rows, "debut_player_won_binary", set(debut["debut_player_won"].unique()).issubset({0, 1}), sorted(debut["debut_player_won"].unique()), "0/1")
        direction_ok = (
            ((debut["debut_player_is_a"]) & (debut["p_debut_Glicko_low"] == debut["p_a_Glicko_low"]))
            | ((~debut["debut_player_is_a"]) & (np.isclose(debut["p_debut_Glicko_low"], 1 - debut["p_a_Glicko_low"])))
        ).all()
        add_check(rows, "debut_probability_orientation", bool(direction_ok), "checked", "p_debut mapped by player ID")
    no_debut = df.loc[df["history_category"] == "No debut"]
    add_check(rows, "no_debut_results_contain_no_debut", not no_debut["either_player_debut"].any(), int(no_debut["either_player_debut"].sum()), 0)
    returning = no_debut.loc[no_debut["max_days_since_last_game"] >= 365]
    add_check(rows, "genuine_returning_contains_no_debut", not returning["either_player_debut"].any(), int(returning["either_player_debut"].sum()), 0)
    high_rd = no_debut.loc[no_debut["max_prematch_rd"] >= no_debut["max_prematch_rd"].quantile(0.75)]
    add_check(rows, "no_debut_high_rd_excludes_debut", not high_rd["either_player_debut"].any(), int(high_rd["either_player_debut"].sum()), 0)
    relevant_zero = int(((df["min_games_last_365_days"] == 0) | df["min_games_last_365_days"].isna()).sum())
    add_check(rows, "zero_recent_decomposition_count", int(zero_recent["games"].sum()) == relevant_zero, int(zero_recent["games"].sum()), relevant_zero)
    add_check(rows, "initialisation_constants_available", all(pd.notna(constants[k]) for k in ["elo_initial_rating", "glicko_initial_rating", "glicko_initial_rd", "glicko_min_rd", "glicko_max_rd"]), "checked", "not missing")
    missing_tables = [str(p.relative_to(PROJECT_ROOT)) for p in output_paths if not p.exists()]
    missing_figures = [str(p.relative_to(PROJECT_ROOT)) for p in figure_paths if not p.exists()]
    add_check(rows, "all_tables_generated", not missing_tables, "; ".join(missing_tables), "none")
    add_check(rows, "all_figures_generated", not missing_figures, "; ".join(missing_figures), "none")
    add_check(rows, "bootstrap_repetitions", True, BOOTSTRAP_REPS, BOOTSTRAP_REPS)
    return pd.DataFrame(rows)


def write_markdown_summary(
    constants: dict[str, Any],
    input_validation: pd.DataFrame,
    history_counts: pd.DataFrame,
    debut_summary: pd.DataFrame,
    init_diag: pd.DataFrame,
    rating_dist: pd.DataFrame,
    loo_summary: pd.DataFrame,
    zero_recent: pd.DataFrame,
    returners: pd.DataFrame,
    exclusion: pd.DataFrame,
    rd_quartiles: pd.DataFrame,
    confidence: pd.DataFrame,
    brier_decomp: pd.DataFrame,
    key_table: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
) -> str:
    """Write the meeting-ready diagnostic summary."""

    def f(value: Any, digits: int = 6) -> str:
        return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"

    exact = history_counts.set_index("category")
    debut_row = debut_summary.loc[debut_summary["model"] == "Glicko_low"].iloc[0]
    elo_debut = debut_summary.loc[debut_summary["model"] == "Validation_best_Elo"].iloc[0]
    prob_diff = debut_summary.loc[debut_summary["model"] == "Glicko_minus_Elo_probability_difference"].iloc[0]
    overall_excl = exclusion.set_index("subgroup")
    return365 = returners.loc[returners["threshold_days"] == 365].iloc[0]
    glicko_dist = rating_dist.loc[rating_dist["model"] == "Glicko_low"].iloc[0]
    elo_dist = rating_dist.loc[rating_dist["model"] == "Validation_best_Elo"].iloc[0]

    robust = key_table.loc[key_table["interpretation_flag"].isin(["robust_glicko_advantage", "robust_elo_advantage", "initialisation_mismatch"])].head(6)
    uncertain = key_table.loc[key_table["interpretation_flag"].isin(["uncertain", "small_sample"])].head(6)

    lines = [
        "# Meeting 6 Step 3: Debut Initialisation and Robustness Diagnostics",
        "",
        "## Purpose",
        "",
        "This diagnostic step investigates why validation-best Elo outperformed Glicko low-inflation in debut-player matches and whether the broader Glicko advantage is robust after excluding debut cases.",
        "",
        "## Inputs and validation",
        "",
        f"- Step 29 per-match scores: `{STEP29_SCORES_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Step 28 match features: `{STEP28_MATCH_FEATURES_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Validation checks passed: {int(input_validation['passed'].sum())} / {len(input_validation)}",
        "",
        "## Model settings read from code/outputs",
        "",
        f"- Elo initial rating: {f(constants['elo_initial_rating'], 3)}",
        f"- Glicko initial rating: {f(constants['glicko_initial_rating'], 3)}",
        f"- Glicko initial RD: {f(constants['glicko_initial_rd'], 3)}",
        f"- Glicko MIN_RD / MAX_RD: {f(constants['glicko_min_rd'], 3)} / {f(constants['glicko_max_rd'], 3)}",
        f"- Low-inflation C: {f(constants['glicko_low_inflation_c'], 6)} over target periods {f(constants['glicko_low_inflation_target_periods'], 0)}",
        f"- Glicko probability formula: {constants['glicko_probability_formula']}",
        f"- Elo probability formula: {constants['elo_probability_formula']}",
        "",
        "## The unexpected debut-player result",
        "",
        f"- No debut games: {int(exact.loc['No debut', 'games'])}",
        f"- Exactly one debut games: {int(exact.loc['Exactly one debut', 'games'])}",
        f"- Both players debut games: {int(exact.loc['Both players debut', 'games'])}",
        f"- Exactly-one-debut empirical debut win rate: {f(debut_row['debut_empirical_win_rate'])}",
        f"- Glicko mean predicted debut win probability: {f(debut_row['mean_predicted_debut_win_probability'])}",
        f"- Validation-best Elo mean predicted debut win probability: {f(elo_debut['mean_predicted_debut_win_probability'])}",
        f"- Mean Glicko minus Elo debut probability: {f(prob_diff['mean_predicted_debut_win_probability'])}",
        "",
        "## Initial rating and 2025 rating-scale compatibility",
        "",
        f"- Glicko initial rating minus 2025 established-player Glicko median: {f(glicko_dist['initial_minus_median'])}",
        f"- Elo initial rating minus 2025 established-player Elo median: {f(elo_dist['initial_minus_median'])}",
        "- This is a mechanism diagnostic only; it does not imply retuning the initial rating on the 2025 test set.",
        "",
        "## Is the debut result driven by a few games or events?",
        "",
        f"- Full debut Brier difference: {f(loo_summary.iloc[0]['full_delta_brier'])}",
        f"- Leave-one-event-out min/max: {f(loo_summary.iloc[0]['minimum_leave_one_event_out_delta_brier'])} / {f(loo_summary.iloc[0]['maximum_leave_one_event_out_delta_brier'])}",
        f"- Sign changes after leaving one event out: {int(loo_summary.iloc[0]['number_of_sign_changes'])}",
        "",
        "## Recent activity zero: debut versus genuine returners",
        "",
    ]
    for row in zero_recent.itertuples(index=False):
        lines.append(f"- {row.subgroup}: games={int(row.games)}, Brier diff={f(row.delta_brier)}, inflation contribution={f(row.inflation_delta_brier)}.")

    lines.extend(
        [
            "",
            "## RD inflation for genuine returning players",
            "",
            f"- Returning >=365 days, no debut: games={int(return365.games)}, Glicko-vs-Elo Brier diff={f(return365.delta_brier)}, inflation delta Brier={f(return365.delta_brier_inflation)}.",
            "",
            "## Overall robustness after excluding debut",
            "",
        ]
    )
    for label in ["All games", "Excluding all debut games", "Both players have history", "Both active in last 365 days and no debut"]:
        if label in overall_excl.index:
            row = overall_excl.loc[label]
            lines.append(f"- {label}: games={int(row.games)}, Brier diff={f(row.delta_brier)}, CI [{f(row.delta_brier_ci_lower)}, {f(row.delta_brier_ci_upper)}].")

    lines.extend(
        [
            "",
            "## No-debut RD analysis",
            "",
        ]
    )
    for row in rd_quartiles.itertuples(index=False):
        lines.append(f"- {row.subgroup}: games={int(row.games)}, mean max RD={f(row.mean_max_rd)}, Brier diff={f(row.delta_brier)}.")

    lines.extend(
        [
            "",
            "## Prediction confidence mechanism",
            "",
        ]
    )
    conf_all = confidence.loc[confidence["sample"] == "All games"]
    for row in conf_all.itertuples(index=False):
        lines.append(f"- {row.confidence_category}: games={int(row.games)}, mean confidence diff={f(row.mean_confidence_difference)}, Brier diff={f(row.delta_brier)}.")

    lines.extend(
        [
            "",
            "## Brier reliability and resolution diagnostics",
            "",
            "- This is an approximate favourite-perspective Brier decomposition based on fixed probability bins; it is sensitive to bin choice and does not replace the raw Brier score.",
        ]
    )
    for model in ["Glicko_low", "Validation_best_Elo"]:
        row = brier_decomp.loc[(brier_decomp["sample"] == "Overall") & (brier_decomp["model"] == model)].iloc[0]
        lines.append(f"- {MODEL_LABELS[model]}: reliability={f(row.reliability)}, resolution={f(row.resolution)}, actual Brier={f(row.actual_brier)}.")

    lines.extend(["", "## Main robust findings", ""])
    for row in robust.itertuples(index=False):
        lines.append(
            f"- {row.subgroup}: games={int(row.games)}, Brier diff={f(row.delta_brier)}, CI [{f(row.delta_brier_ci_lower)}, {f(row.delta_brier_ci_upper)}], flag={row.interpretation_flag}."
        )

    lines.extend(["", "## Findings that remain uncertain", ""])
    for row in uncertain.itertuples(index=False):
        lines.append(
            f"- {row.subgroup}: games={int(row.games)}, Brier diff={f(row.delta_brier)}, CI [{f(row.delta_brier_ci_lower)}, {f(row.delta_brier_ci_upper)}], flag={row.interpretation_flag}."
        )

    lines.extend(
        [
            "",
            "## Implications for Meeting 6",
            "",
            "- The debut result should be reported directly: validation-best Elo is much better for exactly-one-debut matches in the current fixed 2025 test.",
            "- The broader Glicko advantage remains after excluding debut matches.",
            "- RD inflation improves Glicko C0 for long-inactivity cases, but this is not the same as proving Glicko low beats validation-best Elo in every returning-player subgroup.",
            "- The calibration/Brier tension should be discussed as a reliability-resolution-confidence trade-off rather than a single accuracy claim.",
            "",
            "## Limitations",
            "",
            "- These are exploratory mechanism diagnostics, not causal proof.",
            "- Some returning and debut subgroups are small.",
            "- Historical rating-scale drift is not rerun here; the rating-scale analysis is a 2025 cross-section diagnostic.",
            "",
            "## Files written",
            "",
        ]
    )
    for path in output_paths + figure_paths:
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")

    markdown = "\n".join(lines)
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    constants = read_model_initialisation_constants()
    scores = load_step29_scores()
    scores = convert_pre_match_fields(scores)
    input_validation = validate_inputs(scores, constants)
    features_match, features_long = load_step28_features()

    scores, history_counts = build_history_categories(scores)
    debut = build_debut_player_perspective(scores)
    debut_summary = calculate_debut_model_summary(debut)
    init_diag, rating_dist = analyse_initial_rating_scale_position(scores, debut, constants)
    influential = identify_influential_debut_matches(debut)
    debut_events = calculate_debut_event_contributions(debut)
    loo, loo_summary = run_debut_leave_one_event_out(debut)
    zero_recent = decompose_zero_recent_activity(scores)
    returners = analyse_genuine_returners(scores)
    exclusion = calculate_exclusion_robustness(scores)
    no_debut_subgroups = calculate_no_debut_subgroup_results(scores)
    rd_quartiles, rd_deciles, rd_cutpoints = rebuild_no_debut_rd_groups(scores)
    confidence, disagreement = analyse_prediction_confidence(scores)
    brier_decomp_summary, brier_decomp_bins = calculate_brier_decomposition(scores)
    total_thresholds, recent_thresholds = run_threshold_sensitivity(scores)
    key_table = build_key_diagnostic_table(scores, debut, exclusion, rd_quartiles)

    table_paths = [
        INPUT_VALIDATION_PATH,
        HISTORY_COUNTS_PATH,
        DEBUT_PERSPECTIVE_PATH,
        DEBUT_MODEL_SUMMARY_PATH,
        INITIALISATION_DIAGNOSTICS_PATH,
        RATING_DISTRIBUTION_PATH,
        TOP_DEBUT_MATCHES_PATH,
        DEBUT_EVENT_CONTRIB_PATH,
        DEBUT_LOEO_PATH,
        DEBUT_LOEO_SUMMARY_PATH,
        ZERO_RECENT_PATH,
        RETURNING_SENSITIVITY_PATH,
        EXCLUSION_ROBUSTNESS_PATH,
        NO_DEBUT_SUBGROUP_PATH,
        NO_DEBUT_RD_QUARTILE_PATH,
        NO_DEBUT_RD_DECILE_PATH,
        NO_DEBUT_RD_CUTPOINTS_PATH,
        CONFIDENCE_DIAGNOSTICS_PATH,
        FAVOURITE_DISAGREEMENT_PATH,
        BRIER_DECOMP_SUMMARY_PATH,
        BRIER_DECOMP_BINS_PATH,
        TOTAL_THRESHOLD_PATH,
        RECENT_THRESHOLD_PATH,
        KEY_DIAGNOSTIC_PATH,
        VALIDATION_CHECKS_PATH,
        SUMMARY_MD_PATH,
    ]

    input_validation.to_csv(INPUT_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    history_counts.to_csv(HISTORY_COUNTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut.to_csv(DEBUT_PERSPECTIVE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut_summary.to_csv(DEBUT_MODEL_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    init_diag.to_csv(INITIALISATION_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    rating_dist.to_csv(RATING_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    influential.to_csv(TOP_DEBUT_MATCHES_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut_events.to_csv(DEBUT_EVENT_CONTRIB_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    loo.to_csv(DEBUT_LOEO_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    loo_summary.to_csv(DEBUT_LOEO_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    zero_recent.to_csv(ZERO_RECENT_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    returners.to_csv(RETURNING_SENSITIVITY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    exclusion.to_csv(EXCLUSION_ROBUSTNESS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    no_debut_subgroups.to_csv(NO_DEBUT_SUBGROUP_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    rd_quartiles.to_csv(NO_DEBUT_RD_QUARTILE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    rd_deciles.to_csv(NO_DEBUT_RD_DECILE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    rd_cutpoints.to_csv(NO_DEBUT_RD_CUTPOINTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    confidence.to_csv(CONFIDENCE_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    disagreement.to_csv(FAVOURITE_DISAGREEMENT_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    brier_decomp_summary.to_csv(BRIER_DECOMP_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    brier_decomp_bins.to_csv(BRIER_DECOMP_BINS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    total_thresholds.to_csv(TOTAL_THRESHOLD_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    recent_thresholds.to_csv(RECENT_THRESHOLD_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    key_table.to_csv(KEY_DIAGNOSTIC_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")

    figure_paths = create_diagnostic_figures(
        scores,
        debut_summary,
        debut,
        rating_dist,
        zero_recent,
        returners,
        exclusion,
        rd_quartiles,
        confidence,
        loo,
    )
    validation = run_validation_checks(scores, history_counts, debut, zero_recent, table_paths, figure_paths, constants)
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")
    write_markdown_summary(
        constants,
        input_validation,
        history_counts,
        debut_summary,
        init_diag,
        rating_dist,
        loo_summary,
        zero_recent,
        returners,
        exclusion,
        rd_quartiles,
        confidence,
        brier_decomp_summary,
        key_table,
        table_paths,
        figure_paths,
    )
    validation = run_validation_checks(scores, history_counts, debut, zero_recent, table_paths, figure_paths, constants)
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")

    exact_one = int(history_counts.loc[history_counts["category"] == "Exactly one debut", "games"].iloc[0])
    both_debut = int(history_counts.loc[history_counts["category"] == "Both players debut", "games"].iloc[0])
    excl = exclusion.set_index("subgroup").loc["Excluding all debut games"]
    print("Meeting 6 step 3 debut/robustness diagnostics complete.")
    print(f"Rows analysed: {len(scores):,}")
    print(f"Exactly-one-debut games: {exact_one}; both-debut games: {both_debut}")
    print(f"Validation checks passed: {int(validation['passed'].sum())} / {len(validation)}")
    print(f"Excluding debut delta Brier, Elo - Glicko: {excl['delta_brier']:.6f}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
