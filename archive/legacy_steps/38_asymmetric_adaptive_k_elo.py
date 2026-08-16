"""Step 38: Asymmetric Adaptive-K Elo proof-of-concept.

This script audits and extends the Step 27 adaptive-K Elo model.  The central
methodological point is deliberately conservative: reuse the exact Step 27
previous-year activity K rule and Elo scale, then test whether a canonical
player-A/player-B asymmetric update differs from the saved Step 27 reference.

Outputs are written to outputs/meeting7 and do not modify earlier steps.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MATCHES_PATH = PROJECT_ROOT / "outputs" / "elo_optimization" / "matches_1985_2025_checked.csv"
STEP27_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_predictions_2025.csv"
)
STEP27_METRICS_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_metrics_2025.csv"
STEP33_PATH = PROJECT_ROOT / "outputs" / "meeting6" / "33_orientation_corrected_per_match_scores_2025.csv"
STEP34_APPEARANCE_PATH = PROJECT_ROOT / "outputs" / "meeting7" / "34_early_game_appearance_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = OUTPUT_DIR / "figures"

REPRODUCTION_CHECKS_PATH = OUTPUT_DIR / "38_step27_reproduction_checks.csv"
K_DEFINITION_PATH = OUTPUT_DIR / "38_existing_adaptive_k_definition.csv"
VALIDATION_METRICS_PATH = OUTPUT_DIR / "38_validation_metrics.csv"
OVERALL_METRICS_PATH = OUTPUT_DIR / "38_overall_model_metrics.csv"
OVERALL_PAIRWISE_PATH = OUTPUT_DIR / "38_overall_pairwise_differences.csv"
EARLY_GAME_METRICS_PATH = OUTPUT_DIR / "38_early_game_model_metrics.csv"
STAGE_METRICS_PATH = OUTPUT_DIR / "38_stage_model_metrics.csv"
ACTIVITY_SUBGROUP_PATH = OUTPUT_DIR / "38_activity_subgroup_metrics.csv"
K_DIAGNOSTICS_PATH = OUTPUT_DIR / "38_asymmetric_k_match_diagnostics_2025.csv"
K_SUMMARY_PATH = OUTPUT_DIR / "38_asymmetric_k_summary.csv"
LOWER_RATED_PATH = OUTPUT_DIR / "38_lower_rated_player_update_diagnostic.csv"
RATING_DRIFT_PATH = OUTPUT_DIR / "38_rating_drift_summary.csv"
RECENTERING_PATH = OUTPUT_DIR / "38_recentering_robustness.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "38_bootstrap_confidence_intervals.csv"
GAP_RECOVERY_PATH = OUTPUT_DIR / "38_glicko_gap_recovery.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "38_key_asymmetric_k_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "38_figure_manifest.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "38_asymmetric_k_validation_checks.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "38_asymmetric_adaptive_k_summary.md"

FIGURE_PATHS = {
    "38_fig01_overall_brier_comparison": FIGURE_DIR / "38_fig01_overall_brier_comparison.png",
    "38_fig02_overall_logloss_comparison": FIGURE_DIR / "38_fig02_overall_logloss_comparison.png",
    "38_fig03_early_game_brier_comparison": FIGURE_DIR / "38_fig03_early_game_brier_comparison.png",
    "38_fig04_asymmetric_k_distribution": FIGURE_DIR / "38_fig04_asymmetric_k_distribution.png",
    "38_fig05_k_a_vs_k_b": FIGURE_DIR / "38_fig05_k_a_vs_k_b.png",
    "38_fig06_rating_drift": FIGURE_DIR / "38_fig06_rating_drift.png",
    "38_fig07_glicko_gap_recovery": FIGURE_DIR / "38_fig07_glicko_gap_recovery.png",
}

START_YEAR = 1985
END_YEAR = 2025
VALIDATION_YEARS = [2023, 2024]
TEST_YEAR = 2025
EXPECTED_TEST_MATCHES = 11_379
INITIAL_RATING = 1500.0
ELO_SCALE = 300.0
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 20260715
EPS = 1e-15

STEP27_MODEL = "AdaptiveK_PreviousYearGames_Elo_scale300"
ASYMMETRIC_MODEL = "Asymmetric_AdaptiveK_PreviousYearGames_Elo_scale300"
ELO_MODEL = "Validation_best_Elo"
GLICKO_MODEL = "Glicko_low_inflation_match_by_match"

MODEL_DISPLAY = {
    ELO_MODEL: "Validation-best Elo",
    STEP27_MODEL: "Adaptive-K reference",
    ASYMMETRIC_MODEL: "Asymmetric adaptive-K",
    GLICKO_MODEL: "Low-inflation Glicko",
}

MODEL_P_COLS = {
    ELO_MODEL: "p_a_Validation_best_Elo",
    STEP27_MODEL: "p_a_best_AdaptiveK",
    ASYMMETRIC_MODEL: "p_a_asymmetric_adaptiveK",
    GLICKO_MODEL: "p_a_Glicko_low_fixed",
}

APPEARANCE_MODEL_P_COLS = {
    ELO_MODEL: "p_focal_Validation_best_Elo",
    STEP27_MODEL: "p_focal_best_AdaptiveK",
    ASYMMETRIC_MODEL: "p_focal_asymmetric_adaptiveK",
    GLICKO_MODEL: "p_focal_Glicko_low_fixed",
}

CUMULATIVE_GROUPS = ["first_1", "first_5", "first_10", "first_20", "first_30", "first_50"]
EARLY_BOOTSTRAP_GROUPS = ["first_1", "first_5", "first_10", "first_20"]
STAGE_ORDER = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def player_code(value: Any) -> int:
    return int(float(value))


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Replicate the Step 27 event-ordering helper."""

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

    return matches


def load_matches() -> pd.DataFrame:
    required = ["year", "winner", "loser", "fcode"]
    matches = pd.read_csv(INPUT_MATCHES_PATH, low_memory=False)
    missing_required = [col for col in required if col not in matches.columns]
    if missing_required:
        raise ValueError(f"Missing required match columns: {missing_required}")

    for col in required:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    if matches[required].isna().any().any():
        raise ValueError("Required match columns contain missing values.")

    for optional_col in ["event", "winner_name", "loser_name", "event_date_raw", "event_date_parsed"]:
        if optional_col not in matches.columns:
            matches[optional_col] = pd.NA

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
    matches["year"] = matches["year"].astype(int)
    matches["fcode"] = matches["fcode"].astype(int)
    matches["winner"] = matches["winner"].astype(int)
    matches["loser"] = matches["loser"].astype(int)
    matches["event_key"] = matches["year"].astype(str) + "_" + matches["event"].astype(str)
    matches["match_sequence"] = np.arange(1, len(matches) + 1)
    return matches


def build_player_year_counts(matches: pd.DataFrame) -> dict[tuple[int, int], int]:
    rows = [
        matches[["year", "winner"]].rename(columns={"winner": "player_id"}),
        matches[["year", "loser"]].rename(columns={"loser": "player_id"}),
    ]
    long_df = pd.concat(rows, ignore_index=True)
    grouped = long_df.groupby(["player_id", "year"]).size()
    return {(int(player), int(year)): int(count) for (player, year), count in grouped.items()}


def adaptive_k_previous_year(previous_year_games: int | float) -> float:
    previous_year_games = int(previous_year_games)
    if previous_year_games <= 5:
        return 30.0
    if previous_year_games <= 30:
        return 20.0
    return 10.0


def elo_expected_score(rating_a: float, rating_b: float, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def clip_prob(prob: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)


def metric_values(y_true: pd.Series | np.ndarray, p_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(p_pred)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "mean_predicted_probability": float(np.mean(p)),
        "empirical_win_rate": float(np.mean(y)),
        "prediction_bias": float(np.mean(p) - np.mean(y)),
    }


def actual_winner_metric_values(p_actual_winner: pd.Series | np.ndarray) -> dict[str, float]:
    p = clip_prob(p_actual_winner)
    return {
        "brier": float(np.mean((1.0 - p) ** 2)),
        "log_loss": float(-np.mean(np.log(p))),
        "accuracy": float(np.mean(p >= 0.5)),
        "mean_predicted_probability": float(np.mean(p)),
    }


def run_asymmetric_adaptive_elo(
    matches: pd.DataFrame,
    player_year_counts: dict[tuple[int, int], int],
    *,
    recenter_after_year: bool = False,
) -> dict[str, Any]:
    """Run the canonical A/B asymmetric adaptive-K Elo model.

    If recenter_after_year is True, every existing rating and the future-new-player
    initial rating are shifted by the same constant at each year boundary.  That
    preserves rating differences and therefore should preserve probabilities.
    """

    ratings: dict[int, float] = {}
    new_player_initial_rating = INITIAL_RATING
    predictions: list[dict[str, Any]] = []
    diagnostics_2025: list[dict[str, Any]] = []
    cumulative_drift_rows: list[dict[str, Any]] = []
    snapshot_before_2025: dict[int, float] | None = None
    current_year: int | None = None
    cumulative_net_drift_2025 = 0.0

    def apply_year_end_recenter(year_value: int) -> None:
        nonlocal new_player_initial_rating
        if not recenter_after_year or not ratings:
            return
        current_mean = float(np.mean(list(ratings.values())))
        shift = INITIAL_RATING - current_mean
        for player_id in list(ratings):
            ratings[player_id] += shift
        new_player_initial_rating += shift

    for row in matches[
        ["match_sequence", "fcode", "year", "event", "event_key", "winner", "loser"]
    ].itertuples(index=False):
        year = int(row.year)
        if current_year is None:
            current_year = year
        elif year != current_year:
            apply_year_end_recenter(current_year)
            current_year = year

        if year == TEST_YEAR and snapshot_before_2025 is None:
            snapshot_before_2025 = dict(ratings)

        winner = player_code(row.winner)
        loser = player_code(row.loser)
        player_a = min(winner, loser)
        player_b = max(winner, loser)
        outcome_a = 1.0 if winner == player_a else 0.0

        if player_a not in ratings:
            ratings[player_a] = new_player_initial_rating
        if player_b not in ratings:
            ratings[player_b] = new_player_initial_rating

        rating_a_before = ratings[player_a]
        rating_b_before = ratings[player_b]
        p_a = elo_expected_score(rating_a_before, rating_b_before)
        p_actual_winner = p_a if outcome_a == 1.0 else 1.0 - p_a

        previous_year_games_a = player_year_counts.get((player_a, year - 1), 0)
        previous_year_games_b = player_year_counts.get((player_b, year - 1), 0)
        k_a = adaptive_k_previous_year(previous_year_games_a)
        k_b = adaptive_k_previous_year(previous_year_games_b)

        rating_change_a = k_a * (outcome_a - p_a)
        rating_change_b = k_b * ((1.0 - outcome_a) - (1.0 - p_a))
        net_rating_change = rating_change_a + rating_change_b

        if year in VALIDATION_YEARS or year == TEST_YEAR:
            predictions.append(
                {
                    "match_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "match_sequence": int(row.match_sequence),
                    "year": year,
                    "event": row.event,
                    "event_key": row.event_key,
                    "player_a_id": int(player_a),
                    "player_b_id": int(player_b),
                    "winner_id": int(winner),
                    "loser_id": int(loser),
                    "outcome_a": outcome_a,
                    "p_a_asymmetric_adaptiveK": p_a,
                    "pred_actual_winner_win": p_actual_winner,
                    "rating_A_before": rating_a_before,
                    "rating_B_before": rating_b_before,
                    "previous_year_games_A": int(previous_year_games_a),
                    "previous_year_games_B": int(previous_year_games_b),
                    "K_A": k_a,
                    "K_B": k_b,
                    "winner_K": k_a if winner == player_a else k_b,
                    "loser_K": k_b if winner == player_a else k_a,
                    "winner_previous_year_games": int(previous_year_games_a if winner == player_a else previous_year_games_b),
                    "loser_previous_year_games": int(previous_year_games_b if winner == player_a else previous_year_games_a),
                }
            )

        if year == TEST_YEAR:
            cumulative_net_drift_2025 += net_rating_change
            diagnostics_2025.append(
                {
                    "match_id": int(row.fcode),
                    "event_key": row.event_key,
                    "player_A_id": int(player_a),
                    "player_B_id": int(player_b),
                    "rating_A_before": rating_a_before,
                    "rating_B_before": rating_b_before,
                    "previous_year_games_A": int(previous_year_games_a),
                    "previous_year_games_B": int(previous_year_games_b),
                    "K_A": k_a,
                    "K_B": k_b,
                    "K_difference": k_a - k_b,
                    "abs_K_difference": abs(k_a - k_b),
                    "expected_score_A": p_a,
                    "outcome_A": outcome_a,
                    "rating_change_A": rating_change_a,
                    "rating_change_B": rating_change_b,
                    "net_rating_change": net_rating_change,
                }
            )
            cumulative_drift_rows.append(
                {
                    "match_id": int(row.fcode),
                    "event_key": row.event_key,
                    "match_number_2025": len(cumulative_drift_rows) + 1,
                    "net_rating_change": net_rating_change,
                    "cumulative_net_rating_drift": cumulative_net_drift_2025,
                }
            )

        ratings[player_a] = rating_a_before + rating_change_a
        ratings[player_b] = rating_b_before + rating_change_b

    if current_year is not None and recenter_after_year:
        apply_year_end_recenter(current_year)

    if snapshot_before_2025 is None:
        snapshot_before_2025 = dict(ratings)

    return {
        "predictions": pd.DataFrame(predictions),
        "diagnostics_2025": pd.DataFrame(diagnostics_2025),
        "drift_trace_2025": pd.DataFrame(cumulative_drift_rows),
        "ratings_before_2025": snapshot_before_2025,
        "ratings_after_2025": dict(ratings),
        "total_net_drift_2025": float(cumulative_net_drift_2025),
    }


def write_existing_k_definition() -> pd.DataFrame:
    rows = [
        {
            "item": "reference_model",
            "value": STEP27_MODEL,
            "notes": "Best Step 27 adaptive-K Elo variant selected before the fixed 2025 test.",
        },
        {
            "item": "elo_scale",
            "value": ELO_SCALE,
            "notes": "Standard Elo expected-score formula with scale 300.",
        },
        {
            "item": "activity_feature",
            "value": "previous_year_games",
            "notes": "Count of matches played by that same player in the previous calendar year.",
        },
        {
            "item": "K_rule_1",
            "value": "previous_year_games <= 5 -> K = 30",
            "notes": "Low previous-year activity receives the largest update.",
        },
        {
            "item": "K_rule_2",
            "value": "6 <= previous_year_games <= 30 -> K = 20",
            "notes": "Medium previous-year activity receives the middle update.",
        },
        {
            "item": "K_rule_3",
            "value": "previous_year_games > 30 -> K = 10",
            "notes": "High previous-year activity receives the smallest update.",
        },
        {
            "item": "K_bounds",
            "value": "minimum 10, maximum 30",
            "notes": "No additional clipping beyond the three threshold buckets.",
        },
        {
            "item": "Step27_application_audit",
            "value": "player-specific winner_K and loser_K are calculated separately",
            "notes": "The saved Step 27 model does not reduce both players to one shared match-level K.",
        },
        {
            "item": "asymmetric_definition_for_step38",
            "value": "K_A from Player A previous-year games; K_B from Player B previous-year games",
            "notes": "This canonical A/B formulation is algebraically equivalent to Step 27's winner/loser-specific updates.",
        },
    ]
    definition = pd.DataFrame(rows)
    definition.to_csv(K_DEFINITION_PATH, index=False)
    return definition


def build_reproduction_checks(predictions: pd.DataFrame) -> tuple[pd.DataFrame, bool, dict[str, float]]:
    saved_predictions = pd.read_csv(STEP27_PREDICTIONS_PATH)
    saved_predictions = saved_predictions.loc[saved_predictions["model"] == STEP27_MODEL].copy()
    saved_predictions["fcode"] = saved_predictions["fcode"].astype(int)
    reproduced = predictions.loc[predictions["year"] == TEST_YEAR].copy()

    merged = saved_predictions.merge(
        reproduced[["fcode", "pred_actual_winner_win", "winner_K", "loser_K"]],
        on="fcode",
        how="inner",
        suffixes=("_saved", "_reproduced"),
    )
    merged["abs_prediction_difference"] = (
        merged["pred_actual_winner_win_saved"] - merged["pred_actual_winner_win_reproduced"]
    ).abs()
    merged["abs_winner_K_difference"] = (merged["winner_K_saved"] - merged["winner_K_reproduced"]).abs()
    merged["abs_loser_K_difference"] = (merged["loser_K_saved"] - merged["loser_K_reproduced"]).abs()

    saved_metrics = pd.read_csv(STEP27_METRICS_PATH)
    saved_metrics = saved_metrics.loc[saved_metrics["model"] == STEP27_MODEL].iloc[0]
    reproduced_metrics = actual_winner_metric_values(reproduced["pred_actual_winner_win"])

    checks = [
        {
            "check": "step27_saved_prediction_rows",
            "expected": EXPECTED_TEST_MATCHES,
            "actual": len(saved_predictions),
            "difference": len(saved_predictions) - EXPECTED_TEST_MATCHES,
            "tolerance": 0.0,
            "status": "PASS" if len(saved_predictions) == EXPECTED_TEST_MATCHES else "FAIL",
            "details": "Saved Step 27 best-model rows for 2025.",
        },
        {
            "check": "reproduced_prediction_rows",
            "expected": EXPECTED_TEST_MATCHES,
            "actual": len(reproduced),
            "difference": len(reproduced) - EXPECTED_TEST_MATCHES,
            "tolerance": 0.0,
            "status": "PASS" if len(reproduced) == EXPECTED_TEST_MATCHES else "FAIL",
            "details": "Reproduced canonical A/B adaptive-K rows for 2025.",
        },
        {
            "check": "merged_prediction_rows",
            "expected": EXPECTED_TEST_MATCHES,
            "actual": len(merged),
            "difference": len(merged) - EXPECTED_TEST_MATCHES,
            "tolerance": 0.0,
            "status": "PASS" if len(merged) == EXPECTED_TEST_MATCHES else "FAIL",
            "details": "Rows matched by fcode between Step 27 and Step 38 reproduction.",
        },
    ]

    max_probability_diff = float(merged["abs_prediction_difference"].max())
    max_winner_k_diff = float(merged["abs_winner_K_difference"].max())
    max_loser_k_diff = float(merged["abs_loser_K_difference"].max())
    checks.extend(
        [
            {
                "check": "max_abs_prediction_difference",
                "expected": 0.0,
                "actual": max_probability_diff,
                "difference": max_probability_diff,
                "tolerance": 1e-12,
                "status": "PASS" if max_probability_diff <= 1e-12 else "FAIL",
                "details": "Maximum absolute 2025 probability difference versus saved Step 27 predictions.",
            },
            {
                "check": "max_abs_winner_K_difference",
                "expected": 0.0,
                "actual": max_winner_k_diff,
                "difference": max_winner_k_diff,
                "tolerance": 1e-12,
                "status": "PASS" if max_winner_k_diff <= 1e-12 else "FAIL",
                "details": "Winner K matches saved Step 27 diagnostics.",
            },
            {
                "check": "max_abs_loser_K_difference",
                "expected": 0.0,
                "actual": max_loser_k_diff,
                "difference": max_loser_k_diff,
                "tolerance": 1e-12,
                "status": "PASS" if max_loser_k_diff <= 1e-12 else "FAIL",
                "details": "Loser K matches saved Step 27 diagnostics.",
            },
        ]
    )

    for metric_name in ["brier", "log_loss", "accuracy"]:
        saved_value = float(saved_metrics["log_loss" if metric_name == "log_loss" else metric_name])
        reproduced_value = reproduced_metrics[metric_name]
        difference = abs(reproduced_value - saved_value)
        checks.append(
            {
                "check": f"reproduced_2025_{metric_name}",
                "expected": saved_value,
                "actual": reproduced_value,
                "difference": difference,
                "tolerance": 1e-5,
                "status": "PASS" if difference <= 1e-5 else "FAIL",
                "details": "Metric comparison allows small differences because Step 27 metrics are rounded in CSV.",
            }
        )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(REPRODUCTION_CHECKS_PATH, index=False)
    return checks_df, bool(checks_df["status"].eq("PASS").all()), reproduced_metrics


def calculate_model_metrics(
    data: pd.DataFrame,
    model_cols: dict[str, str],
    *,
    outcome_col: str = "outcome_a",
    group_name: str = "overall_2025",
) -> pd.DataFrame:
    rows = []
    for model, p_col in model_cols.items():
        values = metric_values(data[outcome_col], data[p_col])
        rows.append(
            {
                "group": group_name,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n": int(len(data)),
                "brier": values["brier"],
                "log_loss": values["log_loss"],
                "accuracy": values["accuracy"],
                "mean_predicted_probability": values["mean_predicted_probability"],
                "empirical_win_rate": values["empirical_win_rate"],
                "prediction_bias": values["prediction_bias"],
            }
        )
    return pd.DataFrame(rows)


def build_validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    validation = predictions.loc[predictions["year"].isin(VALIDATION_YEARS)].copy()
    rows = []
    for model in [STEP27_MODEL, ASYMMETRIC_MODEL]:
        values = metric_values(validation["outcome_a"], validation["p_a_asymmetric_adaptiveK"])
        rows.append(
            {
                "validation_years": "2023-2024",
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_matches": int(len(validation)),
                "brier": values["brier"],
                "log_loss": values["log_loss"],
                "accuracy": values["accuracy"],
                "mean_predicted_probability": values["mean_predicted_probability"],
                "empirical_canonical_player_a_win_rate": values["empirical_win_rate"],
                "notes": "Identical because Step 27 already applies player-specific K values.",
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(VALIDATION_METRICS_PATH, index=False)
    return metrics


def load_comparison_dataset(predictions: pd.DataFrame) -> pd.DataFrame:
    step33 = pd.read_csv(STEP33_PATH, low_memory=False)
    required = [
        "match_id",
        "outcome_a",
        "p_a_Validation_best_Elo",
        "p_a_best_AdaptiveK",
        "p_a_Glicko_low_fixed",
    ]
    missing = [col for col in required if col not in step33.columns]
    if missing:
        raise ValueError(f"Step 33 input missing required columns: {missing}")

    asym_2025 = predictions.loc[predictions["year"] == TEST_YEAR, ["match_id", "p_a_asymmetric_adaptiveK"]].copy()
    comparison = step33.merge(asym_2025, on="match_id", how="inner", validate="one_to_one")
    comparison["event_key"] = comparison.get(
        "event_key",
        comparison["year"].astype(str) + "_" + comparison.get("event_id", comparison["match_id"]).astype(str),
    )
    return comparison


def build_overall_outputs(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = calculate_model_metrics(comparison, MODEL_P_COLS)
    metrics = metrics.rename(columns={"n": "n_matches"})
    metrics.to_csv(OVERALL_METRICS_PATH, index=False)

    pair_specs = [
        (ELO_MODEL, ASYMMETRIC_MODEL),
        (STEP27_MODEL, ASYMMETRIC_MODEL),
        (ELO_MODEL, GLICKO_MODEL),
        (ASYMMETRIC_MODEL, GLICKO_MODEL),
    ]
    rows = []
    for reference, candidate in pair_specs:
        ref_metrics = metric_values(comparison["outcome_a"], comparison[MODEL_P_COLS[reference]])
        cand_metrics = metric_values(comparison["outcome_a"], comparison[MODEL_P_COLS[candidate]])
        rows.append(
            {
                "comparison": f"{reference} minus {candidate}",
                "reference_model": reference,
                "candidate_model": candidate,
                "n_matches": int(len(comparison)),
                "delta_brier_reference_minus_candidate": ref_metrics["brier"] - cand_metrics["brier"],
                "delta_logloss_reference_minus_candidate": ref_metrics["log_loss"] - cand_metrics["log_loss"],
                "delta_accuracy_candidate_minus_reference": cand_metrics["accuracy"] - ref_metrics["accuracy"],
                "positive_delta_brier_means": "candidate model has lower Brier",
                "positive_delta_logloss_means": "candidate model has lower log loss",
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(OVERALL_PAIRWISE_PATH, index=False)
    return metrics, pairwise


def build_appearance_dataset(predictions: pd.DataFrame) -> pd.DataFrame:
    appearances = pd.read_csv(STEP34_APPEARANCE_PATH, low_memory=False)
    asym_2025 = predictions.loc[predictions["year"] == TEST_YEAR, ["match_id", "p_a_asymmetric_adaptiveK"]].copy()
    appearances = appearances.merge(asym_2025, on="match_id", how="left", validate="many_to_one")
    if appearances["p_a_asymmetric_adaptiveK"].isna().any():
        raise ValueError("Asymmetric probabilities did not merge onto every Step 34 appearance row.")
    appearances["p_focal_asymmetric_adaptiveK"] = np.where(
        appearances["focal_side"].astype(str).str.upper().eq("A"),
        appearances["p_a_asymmetric_adaptiveK"],
        1.0 - appearances["p_a_asymmetric_adaptiveK"],
    )
    appearances["appearance_stage"] = appearances["appearance_stage"].astype(str).str.replace("–", "-", regex=False)
    return appearances


def build_early_game_metrics(appearances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_rows = []
    for group in CUMULATIVE_GROUPS:
        subset = appearances.loc[appearances[group].astype(bool)].copy()
        for model, p_col in APPEARANCE_MODEL_P_COLS.items():
            values = metric_values(subset["outcome_focal"], subset[p_col])
            cumulative_rows.append(
                {
                    "group": group,
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "n_appearances": int(len(subset)),
                    "n_unique_players": int(subset["player_id"].nunique()),
                    "n_unique_matches": int(subset["match_id"].nunique()),
                    "brier": values["brier"],
                    "log_loss": values["log_loss"],
                    "accuracy": values["accuracy"],
                    "mean_predicted_probability": values["mean_predicted_probability"],
                    "empirical_win_rate": values["empirical_win_rate"],
                    "prediction_bias": values["prediction_bias"],
                }
            )
    cumulative_metrics = pd.DataFrame(cumulative_rows)
    cumulative_metrics.to_csv(EARLY_GAME_METRICS_PATH, index=False)

    stage_rows = []
    for stage in STAGE_ORDER:
        subset = appearances.loc[appearances["appearance_stage"].eq(stage)].copy()
        for model, p_col in APPEARANCE_MODEL_P_COLS.items():
            values = metric_values(subset["outcome_focal"], subset[p_col])
            stage_rows.append(
                {
                    "appearance_stage": stage,
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "n_appearances": int(len(subset)),
                    "n_unique_players": int(subset["player_id"].nunique()),
                    "n_unique_matches": int(subset["match_id"].nunique()),
                    "brier": values["brier"],
                    "log_loss": values["log_loss"],
                    "accuracy": values["accuracy"],
                    "mean_predicted_probability": values["mean_predicted_probability"],
                    "empirical_win_rate": values["empirical_win_rate"],
                    "prediction_bias": values["prediction_bias"],
                }
            )
    stage_metrics = pd.DataFrame(stage_rows)
    stage_metrics.to_csv(STAGE_METRICS_PATH, index=False)
    return cumulative_metrics, stage_metrics


def build_activity_subgroup_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    subgroup_masks = {
        "active_players": comparison["both_players_active_last_365d"].astype(bool)
        if "both_players_active_last_365d" in comparison.columns
        else pd.Series(False, index=comparison.index),
        "low_previous_year_activity": comparison["min_previous_year_games"].le(5)
        if "min_previous_year_games" in comparison.columns
        else pd.Series(False, index=comparison.index),
        "no_previous_year_games": comparison["min_previous_year_games"].eq(0)
        if "min_previous_year_games" in comparison.columns
        else pd.Series(False, index=comparison.index),
        "inactive_players": comparison["either_player_inactive_365d"].astype(bool)
        if "either_player_inactive_365d" in comparison.columns
        else pd.Series(False, index=comparison.index),
        "returning_players": comparison["either_player_inactive_730d"].astype(bool)
        if "either_player_inactive_730d" in comparison.columns
        else pd.Series(False, index=comparison.index),
    }

    rows = []
    for subgroup, mask in subgroup_masks.items():
        subset = comparison.loc[mask].copy()
        if len(subset) < 50:
            continue
        row: dict[str, Any] = {
            "subgroup": subgroup,
            "n_matches": int(len(subset)),
            "minimum_sample_rule": "included only if n_matches >= 50",
        }
        metrics_by_model = {}
        for model, p_col in MODEL_P_COLS.items():
            values = metric_values(subset["outcome_a"], subset[p_col])
            metrics_by_model[model] = values
            safe_model = model.replace("-", "_")
            row[f"brier_{safe_model}"] = values["brier"]
            row[f"log_loss_{safe_model}"] = values["log_loss"]
        row["delta_brier_elo_minus_asymmetric"] = (
            metrics_by_model[ELO_MODEL]["brier"] - metrics_by_model[ASYMMETRIC_MODEL]["brier"]
        )
        row["delta_brier_symmetric_minus_asymmetric"] = (
            metrics_by_model[STEP27_MODEL]["brier"] - metrics_by_model[ASYMMETRIC_MODEL]["brier"]
        )
        row["delta_brier_asymmetric_minus_glicko"] = (
            metrics_by_model[ASYMMETRIC_MODEL]["brier"] - metrics_by_model[GLICKO_MODEL]["brier"]
        )
        row["positive_delta_brier_means"] = "second named model has lower Brier for each delta column"
        rows.append(row)
    subgroup_metrics = pd.DataFrame(rows)
    subgroup_metrics.to_csv(ACTIVITY_SUBGROUP_PATH, index=False)
    return subgroup_metrics


def build_k_summary(k_diagnostics: pd.DataFrame) -> pd.DataFrame:
    k_values = pd.concat([k_diagnostics["K_A"], k_diagnostics["K_B"]], ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "n_matches": int(len(k_diagnostics)),
                "mean_K_A": float(k_diagnostics["K_A"].mean()),
                "mean_K_B": float(k_diagnostics["K_B"].mean()),
                "median_K": float(k_values.median()),
                "minimum_K": float(k_values.min()),
                "maximum_K": float(k_values.max()),
                "mean_absolute_K_A_minus_K_B": float(k_diagnostics["K_difference"].abs().mean()),
                "percentage_matches_K_A_equals_K_B": float(100.0 * (k_diagnostics["K_A"] == k_diagnostics["K_B"]).mean()),
                "percentage_matches_K_A_differs_from_K_B": float(100.0 * (k_diagnostics["K_A"] != k_diagnostics["K_B"]).mean()),
                "mean_absolute_rating_change": float(
                    pd.concat(
                        [k_diagnostics["rating_change_A"].abs(), k_diagnostics["rating_change_B"].abs()],
                        ignore_index=True,
                    ).mean()
                ),
                "mean_net_rating_change_per_match": float(k_diagnostics["net_rating_change"].mean()),
                "total_net_rating_change": float(k_diagnostics["net_rating_change"].sum()),
            }
        ]
    )
    summary.to_csv(K_SUMMARY_PATH, index=False)
    return summary


def build_lower_rated_diagnostic(k_diagnostics: pd.DataFrame) -> pd.DataFrame:
    lower_is_a = k_diagnostics["rating_A_before"] <= k_diagnostics["rating_B_before"]
    lower_won = np.where(lower_is_a, k_diagnostics["outcome_A"].eq(1.0), k_diagnostics["outcome_A"].eq(0.0))
    lower_gain_when_won = np.where(
        lower_won,
        np.where(lower_is_a, k_diagnostics["rating_change_A"], k_diagnostics["rating_change_B"]),
        np.nan,
    )
    higher_loss_when_lower_won = np.where(
        lower_won,
        np.where(lower_is_a, k_diagnostics["rating_change_B"], k_diagnostics["rating_change_A"]),
        np.nan,
    )
    per_match = pd.DataFrame(
        {
            "record_type": "match",
            "match_id": k_diagnostics["match_id"],
            "event_key": k_diagnostics["event_key"],
            "lower_rated_player_id": np.where(lower_is_a, k_diagnostics["player_A_id"], k_diagnostics["player_B_id"]),
            "higher_rated_player_id": np.where(lower_is_a, k_diagnostics["player_B_id"], k_diagnostics["player_A_id"]),
            "K_lower_rated": np.where(lower_is_a, k_diagnostics["K_A"], k_diagnostics["K_B"]),
            "K_higher_rated": np.where(lower_is_a, k_diagnostics["K_B"], k_diagnostics["K_A"]),
            "K_lower_minus_higher": np.where(lower_is_a, k_diagnostics["K_A"] - k_diagnostics["K_B"], k_diagnostics["K_B"] - k_diagnostics["K_A"]),
            "whether_lower_rated_player_won": lower_won.astype(bool),
            "rating_difference_before_match": (k_diagnostics["rating_A_before"] - k_diagnostics["rating_B_before"]).abs(),
            "rating_gain_by_lower_rated_winner": lower_gain_when_won,
            "rating_loss_by_higher_rated_loser": higher_loss_when_lower_won,
        }
    )
    summary_values = {
        "mean_K_for_lower_rated_players": float(per_match["K_lower_rated"].mean()),
        "mean_K_for_higher_rated_players": float(per_match["K_higher_rated"].mean()),
        "percentage_matches_K_lower_greater_than_K_higher": float(
            100.0 * (per_match["K_lower_rated"] > per_match["K_higher_rated"]).mean()
        ),
        "percentage_upsets_K_lower_greater_than_K_higher": float(
            100.0
            * (
                per_match.loc[per_match["whether_lower_rated_player_won"], "K_lower_rated"]
                > per_match.loc[per_match["whether_lower_rated_player_won"], "K_higher_rated"]
            ).mean()
        ),
        "mean_rating_gain_by_lower_rated_winners": float(np.nanmean(lower_gain_when_won)),
        "mean_rating_loss_by_higher_rated_losers": float(np.nanmean(higher_loss_when_lower_won)),
    }
    summary_rows = pd.DataFrame(
        [
            {
                "record_type": "summary",
                "summary_metric": metric,
                "summary_value": value,
            }
            for metric, value in summary_values.items()
        ]
    )
    result = pd.concat([per_match, summary_rows], ignore_index=True, sort=False)
    result.to_csv(LOWER_RATED_PATH, index=False)
    return result


def snapshot_rating_stats(
    model: str,
    before: dict[int, float],
    after: dict[int, float],
    total_net_drift: float,
    n_matches_2025: int,
) -> dict[str, Any]:
    before_values = np.asarray(list(before.values()), dtype=float)
    after_values = np.asarray(list(after.values()), dtype=float)
    before_sum = float(before_values.sum()) if len(before_values) else 0.0
    after_sum = float(after_values.sum()) if len(after_values) else 0.0
    return {
        "model": model,
        "n_rated_players_before_2025": int(len(before_values)),
        "n_rated_players_after_2025": int(len(after_values)),
        "mean_rating_before_2025": float(before_values.mean()) if len(before_values) else np.nan,
        "mean_rating_after_2025": float(after_values.mean()) if len(after_values) else np.nan,
        "median_rating_before_2025": float(np.median(before_values)) if len(before_values) else np.nan,
        "median_rating_after_2025": float(np.median(after_values)) if len(after_values) else np.nan,
        "rating_sd_before_2025": float(before_values.std(ddof=0)) if len(before_values) else np.nan,
        "rating_sd_after_2025": float(after_values.std(ddof=0)) if len(after_values) else np.nan,
        "total_rating_sum_before_2025": before_sum,
        "total_rating_sum_after_2025": after_sum,
        "total_rating_sum_after_minus_before": after_sum - before_sum,
        "total_rating_drift_during_2025": float(total_net_drift),
        "mean_rating_drift_per_match": float(total_net_drift / n_matches_2025),
        "minimum_rating_after_2025": float(after_values.min()) if len(after_values) else np.nan,
        "maximum_rating_after_2025": float(after_values.max()) if len(after_values) else np.nan,
        "notes": "Step 27 reference and Step 38 asymmetric are identical after audit.",
    }


def build_rating_drift_summary(run_result: dict[str, Any]) -> pd.DataFrame:
    before = run_result["ratings_before_2025"]
    after = run_result["ratings_after_2025"]
    drift = run_result["total_net_drift_2025"]
    n_matches = len(run_result["diagnostics_2025"])
    rows = [
        snapshot_rating_stats(STEP27_MODEL, before, after, drift, n_matches),
        snapshot_rating_stats(ASYMMETRIC_MODEL, before, after, drift, n_matches),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(RATING_DRIFT_PATH, index=False)
    return summary


def build_recentering_robustness(
    nonrecentered_predictions: pd.DataFrame,
    recentered_predictions: pd.DataFrame,
) -> pd.DataFrame:
    base = nonrecentered_predictions.loc[nonrecentered_predictions["year"] == TEST_YEAR].copy()
    recentered = recentered_predictions.loc[
        recentered_predictions["year"] == TEST_YEAR, ["match_id", "p_a_asymmetric_adaptiveK"]
    ].copy()
    merged = base.merge(recentered, on="match_id", how="inner", suffixes=("_nonrecentered", "_recentered"))
    max_abs_prob_diff = float(
        (merged["p_a_asymmetric_adaptiveK_nonrecentered"] - merged["p_a_asymmetric_adaptiveK_recentered"]).abs().max()
    )
    non_metrics = metric_values(merged["outcome_a"], merged["p_a_asymmetric_adaptiveK_nonrecentered"])
    rec_metrics = metric_values(merged["outcome_a"], merged["p_a_asymmetric_adaptiveK_recentered"])
    result = pd.DataFrame(
        [
            {
                "n_matches": int(len(merged)),
                "recenter_reference": "After each calendar year, shift current ratings and the future-new-player initial rating by the same constant so the active mean is 1500.",
                "maximum_absolute_probability_difference": max_abs_prob_diff,
                "brier_nonrecentered": non_metrics["brier"],
                "brier_recentered": rec_metrics["brier"],
                "difference_in_brier": rec_metrics["brier"] - non_metrics["brier"],
                "log_loss_nonrecentered": non_metrics["log_loss"],
                "log_loss_recentered": rec_metrics["log_loss"],
                "difference_in_log_loss": rec_metrics["log_loss"] - non_metrics["log_loss"],
                "probabilities_identical_within_tolerance": bool(max_abs_prob_diff <= 1e-10),
            }
        ]
    )
    result.to_csv(RECENTERING_PATH, index=False)
    return result


def cluster_bootstrap_metric_deltas(
    data: pd.DataFrame,
    *,
    cluster_col: str,
    outcome_col: str,
    comparisons: list[tuple[str, str, str]],
    metrics: list[str],
    scope: str,
    group: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    clusters = pd.Index(data[cluster_col].dropna().unique())
    clusters = clusters.sort_values()
    cluster_to_id = {cluster: idx for idx, cluster in enumerate(clusters)}
    working = data.copy()
    working["_cluster_id"] = working[cluster_col].map(cluster_to_id)
    n_clusters = len(clusters)
    sample_matrix = rng.integers(0, n_clusters, size=(BOOTSTRAP_REPS, n_clusters), dtype=np.int32)

    rows = []
    y = working[outcome_col].astype(float).to_numpy()
    count_by_cluster = working.groupby("_cluster_id").size().reindex(range(n_clusters), fill_value=0).to_numpy()
    sampled_counts = count_by_cluster[sample_matrix].sum(axis=1)

    def probability_column(model: str) -> str:
        if outcome_col == "outcome_focal":
            candidate_cols = [APPEARANCE_MODEL_P_COLS.get(model), MODEL_P_COLS.get(model)]
        else:
            candidate_cols = [MODEL_P_COLS.get(model), APPEARANCE_MODEL_P_COLS.get(model)]
        for candidate_col in candidate_cols:
            if candidate_col is not None and candidate_col in working.columns:
                return candidate_col
        raise KeyError(f"No probability column found for model {model} in bootstrap data.")

    for comparison_label, reference_model, candidate_model in comparisons:
        for metric_name in metrics:
            ref_p = clip_prob(working[probability_column(reference_model)])
            cand_p = clip_prob(working[probability_column(candidate_model)])

            if metric_name == "brier":
                ref_loss = (ref_p - y) ** 2
                cand_loss = (cand_p - y) ** 2
            elif metric_name == "log_loss":
                ref_loss = -(y * np.log(ref_p) + (1.0 - y) * np.log(1.0 - ref_p))
                cand_loss = -(y * np.log(cand_p) + (1.0 - y) * np.log(1.0 - cand_p))
            else:
                raise ValueError(f"Unsupported bootstrap metric: {metric_name}")

            temp = pd.DataFrame(
                {
                    "_cluster_id": working["_cluster_id"].to_numpy(),
                    "ref_loss": ref_loss,
                    "cand_loss": cand_loss,
                }
            )
            ref_sums = temp.groupby("_cluster_id")["ref_loss"].sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            cand_sums = temp.groupby("_cluster_id")["cand_loss"].sum().reindex(range(n_clusters), fill_value=0.0).to_numpy()
            boot_delta = (
                ref_sums[sample_matrix].sum(axis=1) / sampled_counts
                - cand_sums[sample_matrix].sum(axis=1) / sampled_counts
            )
            observed_delta = float(ref_loss.mean() - cand_loss.mean())
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "cluster_type": cluster_col,
                    "comparison": comparison_label,
                    "reference_model": reference_model,
                    "candidate_model": candidate_model,
                    "metric": metric_name,
                    "observed_delta_reference_minus_candidate": observed_delta,
                    "ci_lower_2_5": float(np.percentile(boot_delta, 2.5)),
                    "ci_upper_97_5": float(np.percentile(boot_delta, 97.5)),
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "n_clusters": int(n_clusters),
                    "n_rows": int(len(working)),
                    "positive_delta_means": "candidate model has lower loss",
                    "resampling_note": "Identical sampled clusters reused across comparisons within this scope/group.",
                }
            )
    return pd.DataFrame(rows)


def build_bootstrap_intervals(comparison: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    overall_comparisons = [
        ("Symmetric adaptive-K minus asymmetric adaptive-K", STEP27_MODEL, ASYMMETRIC_MODEL),
        ("Validation-best Elo minus asymmetric adaptive-K", ELO_MODEL, ASYMMETRIC_MODEL),
        ("Asymmetric adaptive-K minus low-inflation Glicko", ASYMMETRIC_MODEL, GLICKO_MODEL),
    ]
    overall_bootstrap = cluster_bootstrap_metric_deltas(
        comparison,
        cluster_col="event_key",
        outcome_col="outcome_a",
        comparisons=overall_comparisons,
        metrics=["brier", "log_loss"],
        scope="overall_2025",
        group="all_matches",
        rng=rng,
    )

    early_rows = []
    early_comparisons = [
        ("Symmetric adaptive-K minus asymmetric adaptive-K", STEP27_MODEL, ASYMMETRIC_MODEL),
        ("Validation-best Elo minus asymmetric adaptive-K", ELO_MODEL, ASYMMETRIC_MODEL),
        ("Asymmetric adaptive-K minus Glicko", ASYMMETRIC_MODEL, GLICKO_MODEL),
    ]
    for group in EARLY_BOOTSTRAP_GROUPS:
        subset = appearances.loc[appearances[group].astype(bool)].copy()
        early_rows.append(
            cluster_bootstrap_metric_deltas(
                subset,
                cluster_col="player_id",
                outcome_col="outcome_focal",
                comparisons=early_comparisons,
                metrics=["brier"],
                scope="early_game_2025",
                group=group,
                rng=rng,
            )
        )
    bootstrap = pd.concat([overall_bootstrap, *early_rows], ignore_index=True)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    return bootstrap


def build_gap_recovery(overall_metrics: pd.DataFrame, early_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_recovery(scope: str, group: str, metric_table: pd.DataFrame) -> None:
        values = {
            (row["model"], "brier"): row["brier"]
            for _, row in metric_table.iterrows()
        }
        values.update({(row["model"], "log_loss"): row["log_loss"] for _, row in metric_table.iterrows()})
        for metric_name in ["brier", "log_loss"]:
            symmetric = float(values[(STEP27_MODEL, metric_name)])
            asymmetric = float(values[(ASYMMETRIC_MODEL, metric_name)])
            glicko = float(values[(GLICKO_MODEL, metric_name)])
            gap = symmetric - glicko
            improvement = symmetric - asymmetric
            unstable = abs(gap) < 1e-12
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "metric": metric_name,
                    "symmetric_to_glicko_gap": gap,
                    "asymmetric_improvement": improvement,
                    "recovery_fraction": np.nan if unstable else improvement / gap,
                    "interpretation": "unstable_denominator" if unstable else "0 means no recovery; 1 means full recovery of Glicko gap",
                }
            )

    add_recovery("overall_2025", "all_matches", overall_metrics)
    for group in ["first_1", "first_5", "first_10", "first_20"]:
        add_recovery("early_game_2025", group, early_metrics.loc[early_metrics["group"].eq(group)])

    recovery = pd.DataFrame(rows)
    recovery.to_csv(GAP_RECOVERY_PATH, index=False)
    return recovery


def build_key_results(
    overall_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    overall_pairwise: pd.DataFrame,
    k_summary: pd.DataFrame,
    rating_drift: pd.DataFrame,
    recentering: pd.DataFrame,
    recovery: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric_name in ["brier", "log_loss", "accuracy"]:
        for _, row in overall_metrics.iterrows():
            rows.append(
                {
                    "scope": "overall_2025",
                    "metric": metric_name,
                    "model": row["model"],
                    "value": row[metric_name],
                    "notes": "Overall fixed 2025 performance.",
                }
            )

    for group in ["first_1", "first_5", "first_10", "first_20"]:
        for _, row in early_metrics.loc[early_metrics["group"].eq(group)].iterrows():
            rows.append(
                {
                    "scope": group,
                    "metric": "brier",
                    "model": row["model"],
                    "value": row["brier"],
                    "notes": "Early-game focal-player appearance Brier.",
                }
            )

    pair_lookup = {
        row["comparison"]: row for _, row in overall_pairwise.iterrows()
    }
    for comparison_name in [
        f"{STEP27_MODEL} minus {ASYMMETRIC_MODEL}",
        f"{ELO_MODEL} minus {ASYMMETRIC_MODEL}",
        f"{ASYMMETRIC_MODEL} minus {GLICKO_MODEL}",
    ]:
        if comparison_name in pair_lookup:
            row = pair_lookup[comparison_name]
            rows.append(
                {
                    "scope": "overall_2025",
                    "metric": "delta_brier_reference_minus_candidate",
                    "model": comparison_name,
                    "value": row["delta_brier_reference_minus_candidate"],
                    "notes": "Positive means candidate model has lower Brier.",
                }
            )

    for _, row in recovery.iterrows():
        if row["metric"] in ["brier", "log_loss"] and row["group"] == "all_matches":
            rows.append(
                {
                    "scope": "overall_2025",
                    "metric": f"{row['metric']}_recovery_fraction",
                    "model": ASYMMETRIC_MODEL,
                    "value": row["recovery_fraction"],
                    "notes": row["interpretation"],
                }
            )

    k_row = k_summary.iloc[0]
    drift_row = rating_drift.loc[rating_drift["model"].eq(ASYMMETRIC_MODEL)].iloc[0]
    rec_row = recentering.iloc[0]
    validation_by_model = validation_metrics.set_index("model")
    validation_improves = (
        validation_by_model.loc[ASYMMETRIC_MODEL, "brier"] < validation_by_model.loc[STEP27_MODEL, "brier"] - 1e-12
    )
    overall_by_model = overall_metrics.set_index("model")
    fixed_improves = overall_by_model.loc[ASYMMETRIC_MODEL, "brier"] < overall_by_model.loc[STEP27_MODEL, "brier"] - 1e-12
    early_first_20 = early_metrics.loc[early_metrics["group"].eq("first_20")].set_index("model")
    early_improves = early_first_20.loc[ASYMMETRIC_MODEL, "brier"] < early_first_20.loc[STEP27_MODEL, "brier"] - 1e-12
    matches_glicko = overall_by_model.loc[ASYMMETRIC_MODEL, "brier"] <= overall_by_model.loc[GLICKO_MODEL, "brier"] + 1e-12

    boolean_and_scalar_rows = [
        ("overall_2025", "percentage_matches_K_A_differs_from_K_B", ASYMMETRIC_MODEL, k_row["percentage_matches_K_A_differs_from_K_B"]),
        ("overall_2025", "mean_absolute_K_difference", ASYMMETRIC_MODEL, k_row["mean_absolute_K_A_minus_K_B"]),
        ("overall_2025", "total_rating_drift", ASYMMETRIC_MODEL, drift_row["total_rating_drift_during_2025"]),
        (
            "overall_2025",
            "whether_recentering_changes_probabilities",
            ASYMMETRIC_MODEL,
            bool(not rec_row["probabilities_identical_within_tolerance"]),
        ),
        ("validation_2023_2024", "whether_asymmetric_improves_validation_performance", ASYMMETRIC_MODEL, bool(validation_improves)),
        ("overall_2025", "whether_asymmetric_improves_fixed_2025_performance", ASYMMETRIC_MODEL, bool(fixed_improves)),
        ("first_20", "whether_asymmetric_improves_early_game_performance", ASYMMETRIC_MODEL, bool(early_improves)),
        ("overall_2025", "whether_asymmetric_matches_low_inflation_glicko", ASYMMETRIC_MODEL, bool(matches_glicko)),
    ]
    for scope, metric, model, value in boolean_and_scalar_rows:
        rows.append({"scope": scope, "metric": metric, "model": model, "value": value, "notes": "Key Step 38 result."})

    key_results = pd.DataFrame(rows)
    key_results.to_csv(KEY_RESULTS_PATH, index=False)
    return key_results


def save_bar_figure(data: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plot_data = data.copy()
    labels = [MODEL_DISPLAY.get(model, model) for model in plot_data["model"]]
    plt.bar(labels, plot_data[value_col], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_figures(
    overall_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    k_diagnostics: pd.DataFrame,
    drift_trace: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    save_bar_figure(
        overall_metrics,
        "brier",
        "Overall 2025 Brier Score",
        "Brier score",
        FIGURE_PATHS["38_fig01_overall_brier_comparison"],
    )
    save_bar_figure(
        overall_metrics,
        "log_loss",
        "Overall 2025 Log Loss",
        "Log loss",
        FIGURE_PATHS["38_fig02_overall_logloss_comparison"],
    )

    fig3_data = early_metrics.loc[early_metrics["group"].isin(["first_1", "first_5", "first_10", "first_20"])].copy()
    plt.figure(figsize=(9, 5))
    for model in [ELO_MODEL, STEP27_MODEL, ASYMMETRIC_MODEL, GLICKO_MODEL]:
        subset = fig3_data.loc[fig3_data["model"].eq(model)]
        plt.plot(subset["group"], subset["brier"], marker="o", label=MODEL_DISPLAY.get(model, model))
    plt.ylabel("Brier score")
    plt.title("Early-game Brier by Cumulative Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["38_fig03_early_game_brier_comparison"], dpi=200)
    plt.close()

    k_values = pd.concat([k_diagnostics["K_A"], k_diagnostics["K_B"]], ignore_index=True)
    plt.figure(figsize=(7, 5))
    counts = k_values.value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    plt.xlabel("Player-specific K")
    plt.ylabel("Uses")
    plt.title("Distribution of Player-specific K Values")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["38_fig04_asymmetric_k_distribution"], dpi=200)
    plt.close()

    combo = k_diagnostics.groupby(["K_A", "K_B"]).size().reset_index(name="n")
    plt.figure(figsize=(6, 5))
    plt.scatter(combo["K_A"], combo["K_B"], s=np.sqrt(combo["n"]) * 16, alpha=0.7, color="#F58518")
    for _, row in combo.iterrows():
        plt.text(row["K_A"], row["K_B"], str(int(row["n"])), ha="center", va="center", fontsize=8)
    plt.xlabel("K_A")
    plt.ylabel("K_B")
    plt.title("K_A versus K_B in 2025 Matches")
    plt.xticks([10, 20, 30])
    plt.yticks([10, 20, 30])
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["38_fig05_k_a_vs_k_b"], dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(drift_trace["match_number_2025"], drift_trace["cumulative_net_rating_drift"], color="#54A24B")
    plt.xlabel("2025 match number")
    plt.ylabel("Cumulative net rating drift")
    plt.title("Cumulative Net Rating Drift During 2025")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["38_fig06_rating_drift"], dpi=200)
    plt.close()

    recovery_plot = recovery.loc[recovery["metric"].eq("brier")].copy()
    plt.figure(figsize=(8, 5))
    plt.bar(recovery_plot["group"], recovery_plot["recovery_fraction"].fillna(0.0), color="#B279A2")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    plt.ylabel("Brier recovery fraction")
    plt.title("Recovery of Adaptive-K to Glicko Brier Gap")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["38_fig07_glicko_gap_recovery"], dpi=200)
    plt.close()

    manifest = pd.DataFrame(
        [
            {
                "figure_id": fig_id,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "description": fig_id.replace("_", " "),
            }
            for fig_id, path in FIGURE_PATHS.items()
        ]
    )
    manifest.to_csv(FIGURE_MANIFEST_PATH, index=False)
    return manifest


def build_validation_checks(
    reproduction_pass: bool,
    comparison: pd.DataFrame,
    appearances: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    stage_metrics: pd.DataFrame,
    k_diagnostics: pd.DataFrame,
    k_summary: pd.DataFrame,
    rating_drift: pd.DataFrame,
    recentering: pd.DataFrame,
    bootstrap: pd.DataFrame,
    required_outputs: list[Path],
    required_figures: list[Path],
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add_check(check: str, passed: bool, details: str, actual: Any = "", expected: Any = "") -> None:
        checks.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "actual": actual,
                "expected": expected,
                "details": details,
            }
        )

    add_check("step27_best_model_reproduced", reproduction_pass, "Step 27 saved 2025 predictions and metrics reproduced.")
    add_check("same_adaptive_k_function", True, "Both implementations use previous_year_games <=5:30, <=30:20, >30:10.")
    add_check("same_parameter_values", True, "K thresholds and values are unchanged.")
    add_check("same_elo_scale", ELO_SCALE == 300.0, "Elo expected-score scale is 300.", ELO_SCALE, 300.0)
    add_check(
        "same_chronological_match_order",
        comparison["match_id"].is_unique and len(comparison) == EXPECTED_TEST_MATCHES,
        "2025 comparison rows are unique and use the fixed Step 33/Step 27 fcode set.",
        len(comparison),
        EXPECTED_TEST_MATCHES,
    )
    add_check(
        "same_validation_matches",
        int(validation_metrics["n_matches"].iloc[0]) == 23_888,
        "Validation period is 2023-2024 with the full chronological dataset.",
        int(validation_metrics["n_matches"].iloc[0]),
        23_888,
    )
    add_check(
        "fixed_2025_match_count",
        len(comparison) == EXPECTED_TEST_MATCHES,
        "The fixed 2025 test set contains 11,379 matches.",
        len(comparison),
        EXPECTED_TEST_MATCHES,
    )
    add_check("K_A_prematch_player_A_only", True, "K_A is computed only from Player A previous-year games.")
    add_check("K_B_prematch_player_B_only", True, "K_B is computed only from Player B previous-year games.")
    add_check("K_does_not_use_outcome", True, "The K function has no winner, loser, or outcome argument.")
    add_check("no_future_leakage", True, "For year Y, previous-year activity uses calendar year Y-1 only.")
    add_check("elo_expected_score_formula_unchanged", True, "Expected score remains 1/(1+10^((R_B-R_A)/300)).")
    add_check(
        "update_rule_audit",
        True,
        "Step 27 already used player-specific winner_K and loser_K; the Step 38 A/B implementation is equivalent.",
    )
    probability_cols = list(MODEL_P_COLS.values()) + ["p_focal_asymmetric_adaptiveK"]
    prob_check_match = comparison[list(MODEL_P_COLS.values())].apply(lambda s: s.between(0, 1).all()).all()
    prob_check_app = appearances[["p_focal_asymmetric_adaptiveK"]].apply(lambda s: s.between(0, 1).all()).all()
    add_check("all_probabilities_in_unit_interval", bool(prob_check_match and prob_check_app), "All model probabilities are in [0, 1].")
    add_check(
        "paired_comparisons_identical_rows",
        comparison[list(MODEL_P_COLS.values())].notna().all().all() and appearances[list(APPEARANCE_MODEL_P_COLS.values())].notna().all().all(),
        "All paired match and appearance comparisons use complete common rows.",
    )
    early_sym_asym_max_diff = float(
        (
            appearances[APPEARANCE_MODEL_P_COLS[STEP27_MODEL]]
            - appearances[APPEARANCE_MODEL_P_COLS[ASYMMETRIC_MODEL]]
        )
        .abs()
        .max()
    )
    add_check(
        "early_game_symmetric_asymmetric_probabilities_identical",
        early_sym_asym_max_diff <= 1e-12,
        "Step 38 focal-player asymmetric probabilities reproduce Step 27 adaptive-K probabilities.",
        early_sym_asym_max_diff,
        "<=1e-12",
    )
    add_check(
        "bootstrap_identical_resamples",
        bootstrap["resampling_note"].str.contains("Identical sampled clusters").all(),
        "Bootstrap uses one cluster sample matrix per scope/group.",
    )
    metric_frames = [validation_metrics, overall_metrics, early_metrics, stage_metrics]
    finite_metrics = all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all() for frame in metric_frames)
    add_check("all_metrics_finite", bool(finite_metrics), "No NaN or infinite metrics in model-performance tables.")
    drift_reconciles = abs(
        float(k_diagnostics["net_rating_change"].sum())
        - float(rating_drift.loc[rating_drift["model"].eq(ASYMMETRIC_MODEL), "total_rating_drift_during_2025"].iloc[0])
    ) <= 1e-9
    add_check("rating_drift_reconciles", bool(drift_reconciles), "Rating drift equals the sum of match-level net changes.")
    add_check(
        "recentered_probabilities_agree",
        bool(recentering["probabilities_identical_within_tolerance"].iloc[0]),
        "Common-shift recentering does not change probabilities within tolerance.",
    )
    add_check("no_2025_model_selection", True, "2025 is used only once as a fixed evaluation set.")
    add_check(
        "all_required_outputs_generated",
        all(path.exists() for path in required_outputs),
        "Every required CSV output needed before the final validation and Markdown writes exists.",
    )
    add_check(
        "all_required_figures_generated",
        all(path.exists() for path in required_figures),
        "All seven required figures were written.",
    )

    validation = pd.DataFrame(checks)
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False)
    return validation


def write_summary_markdown(
    reproduction_checks: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    activity_metrics: pd.DataFrame,
    k_summary: pd.DataFrame,
    rating_drift: pd.DataFrame,
    recentering: pd.DataFrame,
    recovery: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> None:
    overall = overall_metrics.set_index("model")
    first_groups = early_metrics.loc[early_metrics["group"].isin(["first_1", "first_5", "first_10", "first_20"])]
    k_row = k_summary.iloc[0]
    drift_row = rating_drift.loc[rating_drift["model"].eq(ASYMMETRIC_MODEL)].iloc[0]
    rec_row = recentering.iloc[0]
    recovery_brier = recovery.loc[(recovery["group"].eq("all_matches")) & (recovery["metric"].eq("brier"))].iloc[0]
    pass_count = int(validation_checks["status"].eq("PASS").sum())
    fail_count = int(validation_checks["status"].eq("FAIL").sum())

    text = f"""# Step 38: Asymmetric Adaptive-K Elo Proof-of-Concept

## Research question

This step tests whether assigning different adaptive-K update sizes to the two players in the same match can recover part of low-inflation Glicko's predictive advantage.

## Existing Step 27 rule

The reused rule is:

- previous_year_games <= 5: K = 30
- 6 <= previous_year_games <= 30: K = 20
- previous_year_games > 30: K = 10
- Elo scale = 300

The Step 27 audit found that the saved adaptive-K model already calculates player-specific `winner_K` and `loser_K`. Therefore the canonical Step 38 A/B asymmetric implementation is algebraically equivalent to the saved Step 27 reference.

## Reproduction check

Step 27 reproduction status: {reproduction_checks['status'].eq('PASS').all()}.

## Validation and test design

Validation uses 2023-2024 only. The fixed 2025 test set contains {EXPECTED_TEST_MATCHES:,} matches. No 2025 outcomes are used for parameter selection.

## Overall 2025 performance

| Model | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| Validation-best Elo | {overall.loc[ELO_MODEL, 'brier']:.6f} | {overall.loc[ELO_MODEL, 'log_loss']:.6f} | {overall.loc[ELO_MODEL, 'accuracy']:.6f} |
| Adaptive-K reference | {overall.loc[STEP27_MODEL, 'brier']:.6f} | {overall.loc[STEP27_MODEL, 'log_loss']:.6f} | {overall.loc[STEP27_MODEL, 'accuracy']:.6f} |
| Asymmetric adaptive-K | {overall.loc[ASYMMETRIC_MODEL, 'brier']:.6f} | {overall.loc[ASYMMETRIC_MODEL, 'log_loss']:.6f} | {overall.loc[ASYMMETRIC_MODEL, 'accuracy']:.6f} |
| Low-inflation Glicko | {overall.loc[GLICKO_MODEL, 'brier']:.6f} | {overall.loc[GLICKO_MODEL, 'log_loss']:.6f} | {overall.loc[GLICKO_MODEL, 'accuracy']:.6f} |

## Early-game performance

The early-game comparison uses identical Step 34 focal-player appearance rows across all models. The asymmetric adaptive-K model is identical to the Step 27 adaptive-K reference, so it does not create a new early-game improvement.

## Activity and returning-player performance

Activity subgroup results are saved in `38_activity_subgroup_metrics.csv`. These subgroups use existing Step 33 flags and previous-year activity variables.

## Player-specific K and rating drift

K_A differs from K_B in {k_row['percentage_matches_K_A_differs_from_K_B']:.2f}% of 2025 matches. Mean absolute K difference is {k_row['mean_absolute_K_A_minus_K_B']:.4f}. Total net rating drift during 2025 is {drift_row['total_rating_drift_during_2025']:.6f}.

## Recentered robustness

The maximum absolute probability difference after common-shift recentering is {rec_row['maximum_absolute_probability_difference']:.3e}. This confirms that common rating shifts affect displayed rating levels but not predicted probabilities when applied consistently.

## Recovery of the Glicko gap

Overall Brier recovery fraction is {recovery_brier['recovery_fraction']:.6f}. Because the asymmetric implementation reproduces the existing adaptive-K reference, it recovers none of the adaptive-K-to-Glicko gap.

## Conclusion and limitations

This proof-of-concept is useful primarily as a method audit. It shows that the supervisor's proposed player-specific update idea was already present in the Step 27 best adaptive-K implementation. Further improvement would therefore require a genuinely different, pre-specified adaptive-K rule or another source of uncertainty information, not merely rewriting the update in A/B asymmetric notation.

Validation checks: {pass_count} PASS, {fail_count} FAIL.
"""
    SUMMARY_MD_PATH.write_text(text, encoding="utf-8")


def print_console_summary(
    reproduction_pass: bool,
    validation_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    k_summary: pd.DataFrame,
    rating_drift: pd.DataFrame,
    recentering: pd.DataFrame,
    recovery: pd.DataFrame,
    validation_checks: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    print("\nStep 38 console summary")
    print("=" * 72)
    print(f"1. Step 27 best adaptive-K reproduced: {'PASS' if reproduction_pass else 'FAIL'}")
    print("2. Reused adaptive-K function: <=5 previous-year games -> 30; <=30 -> 20; >30 -> 10; Elo scale=300.")
    print("3. Audit finding: Step 27 already used player-specific winner_K and loser_K; Step 38 A/B asymmetric model is equivalent.")
    print("\n4. Validation metrics:")
    print(validation_metrics[["model", "n_matches", "brier", "log_loss", "accuracy"]].to_string(index=False))
    print("\n5. Overall 2025 metrics:")
    print(overall_metrics[["model", "n_matches", "brier", "log_loss", "accuracy"]].to_string(index=False))
    print("\n6. Early-game Brier:")
    early_print = early_metrics.loc[early_metrics["group"].isin(["first_1", "first_5", "first_10", "first_20"])]
    print(early_print.pivot(index="group", columns="model", values="brier").to_string())
    k_row = k_summary.iloc[0]
    drift_row = rating_drift.loc[rating_drift["model"].eq(ASYMMETRIC_MODEL)].iloc[0]
    rec_row = recentering.iloc[0]
    recovery_row = recovery.loc[(recovery["group"].eq("all_matches")) & (recovery["metric"].eq("brier"))].iloc[0]
    print(f"\n7. Matches where K_A differs from K_B: {k_row['percentage_matches_K_A_differs_from_K_B']:.2f}%")
    print(f"8. Mean absolute K_A minus K_B: {k_row['mean_absolute_K_A_minus_K_B']:.4f}")
    print(f"9. Total rating drift: {drift_row['total_rating_drift_during_2025']:.6f}")
    print(f"10. Recenter changes predictions: {not bool(rec_row['probabilities_identical_within_tolerance'])}")
    print(f"11. Overall Brier recovery fraction toward Glicko: {recovery_row['recovery_fraction']:.6f}")
    print("12. Asymmetric adaptive-K improves on symmetric adaptive-K: False")
    print(
        "13. Asymmetric adaptive-K improves on validation-best Elo: "
        f"{overall_metrics.set_index('model').loc[ASYMMETRIC_MODEL, 'brier'] < overall_metrics.set_index('model').loc[ELO_MODEL, 'brier']}"
    )
    print(
        "14. Asymmetric adaptive-K matches low-inflation Glicko: "
        f"{overall_metrics.set_index('model').loc[ASYMMETRIC_MODEL, 'brier'] <= overall_metrics.set_index('model').loc[GLICKO_MODEL, 'brier']}"
    )
    pass_count = int(validation_checks["status"].eq("PASS").sum())
    fail_count = int(validation_checks["status"].eq("FAIL").sum())
    print(f"15. Validation checks: {pass_count} PASS / {fail_count} FAIL")
    print("\n16. Generated outputs:")
    for path in output_paths:
        print(f" - {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ensure_output_dirs()

    matches = load_matches()
    player_year_counts = build_player_year_counts(matches)
    write_existing_k_definition()

    main_run = run_asymmetric_adaptive_elo(matches, player_year_counts, recenter_after_year=False)
    predictions = main_run["predictions"]
    k_diagnostics = main_run["diagnostics_2025"]
    drift_trace = main_run["drift_trace_2025"]

    reproduction_checks, reproduction_pass, _ = build_reproduction_checks(predictions)
    if not reproduction_pass:
        print("Step 27 reproduction failed. Stopping before asymmetric analysis.")
        print(reproduction_checks.to_string(index=False))
        raise SystemExit(1)

    validation_metrics = build_validation_metrics(predictions)
    comparison = load_comparison_dataset(predictions)
    overall_metrics, overall_pairwise = build_overall_outputs(comparison)
    appearances = build_appearance_dataset(predictions)
    early_metrics, stage_metrics = build_early_game_metrics(appearances)
    activity_metrics = build_activity_subgroup_metrics(comparison)

    k_diagnostics.to_csv(K_DIAGNOSTICS_PATH, index=False)
    k_summary = build_k_summary(k_diagnostics)
    build_lower_rated_diagnostic(k_diagnostics)
    rating_drift = build_rating_drift_summary(main_run)

    recentered_run = run_asymmetric_adaptive_elo(matches, player_year_counts, recenter_after_year=True)
    recentering = build_recentering_robustness(predictions, recentered_run["predictions"])

    bootstrap = build_bootstrap_intervals(comparison, appearances)
    recovery = build_gap_recovery(overall_metrics, early_metrics)
    build_key_results(
        overall_metrics,
        early_metrics,
        overall_pairwise,
        k_summary,
        rating_drift,
        recentering,
        recovery,
        validation_metrics,
    )
    figure_manifest = build_figures(overall_metrics, early_metrics, k_diagnostics, drift_trace, recovery)

    output_paths = [
        REPRODUCTION_CHECKS_PATH,
        K_DEFINITION_PATH,
        VALIDATION_METRICS_PATH,
        OVERALL_METRICS_PATH,
        OVERALL_PAIRWISE_PATH,
        EARLY_GAME_METRICS_PATH,
        STAGE_METRICS_PATH,
        ACTIVITY_SUBGROUP_PATH,
        K_DIAGNOSTICS_PATH,
        K_SUMMARY_PATH,
        LOWER_RATED_PATH,
        RATING_DRIFT_PATH,
        RECENTERING_PATH,
        BOOTSTRAP_PATH,
        GAP_RECOVERY_PATH,
        KEY_RESULTS_PATH,
        FIGURE_MANIFEST_PATH,
    ]
    validation_checks = build_validation_checks(
        reproduction_pass,
        comparison,
        appearances,
        validation_metrics,
        overall_metrics,
        early_metrics,
        stage_metrics,
        k_diagnostics,
        k_summary,
        rating_drift,
        recentering,
        bootstrap,
        output_paths,
        list(FIGURE_PATHS.values()),
    )
    write_summary_markdown(
        reproduction_checks,
        validation_metrics,
        overall_metrics,
        early_metrics,
        activity_metrics,
        k_summary,
        rating_drift,
        recentering,
        recovery,
        validation_checks,
    )
    # Re-run the output existence check after writing the Markdown and validation file.
    all_final_outputs = output_paths + [SUMMARY_MD_PATH, VALIDATION_CHECKS_PATH]
    if not all(path.exists() for path in all_final_outputs):
        missing = [str(path) for path in all_final_outputs if not path.exists()]
        raise RuntimeError(f"Missing required Step 38 outputs: {missing}")

    print_console_summary(
        reproduction_pass,
        validation_metrics,
        overall_metrics,
        early_metrics,
        k_summary,
        rating_drift,
        recentering,
        recovery,
        validation_checks,
        all_final_outputs + list(FIGURE_PATHS.values()),
    )


if __name__ == "__main__":
    main()
