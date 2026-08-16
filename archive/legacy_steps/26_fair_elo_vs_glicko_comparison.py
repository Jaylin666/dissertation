"""Meeting 5 fair Elo-vs-Glicko comparison.

This script compares fixed Elo baselines and fixed Glicko candidates under the
same full-history match data, chronological ordering, 2025 evaluation games,
and prediction metrics.

It does not tune new parameters, does not implement adaptive-K Elo, and does
not rerun or modify existing Glicko experiments. Elo baselines are regenerated
inside this script only when reusable 2025 prediction files are not available,
and all outputs are written to outputs/meeting5_fair_elo_vs_glicko/.
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

START_YEAR = 1985
END_YEAR = 2025
EXPECTED_2025_GAMES = 11_379
EPS = 1e-15

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko"

METRICS_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_metrics_2025.csv"
EVALUATION_CHECK_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_evaluation_set_check.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_predictions_2025.csv"
CALIBRATION_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_calibration.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_confidence_bins.csv"
RATING_SIMILARITY_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_rating_similarity.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_final_ratings.csv"
ISSUES_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_issues.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_summary.md"

BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_brier_bar.png"
LOGLOSS_PLOT_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_logloss_bar.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_accuracy_bar.png"
CALIBRATION_PLOT_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_calibration_plot.png"
CONFIDENCE_BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_fair_elo_vs_glicko_confidence_brier.png"
SCATTER_PLOT_PATH = OUTPUT_DIR / "meeting5_validation_elo_vs_glicko_low_scatter.png"

REQUIRED_MATCH_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_MATCH_COLUMNS = ["event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

ELO_MODELS = [
    {
        "model": "Conservative_Elo",
        "model_family": "Elo",
        "model_role": "stability reference",
        "k": 10.0,
        "scale": 500.0,
        "expected_log_loss": 0.585928,
        "expected_brier": 0.201418,
    },
    {
        "model": "Default_Elo",
        "model_family": "Elo",
        "model_role": "transparent simple baseline",
        "k": 20.0,
        "scale": 500.0,
        "expected_log_loss": 0.567633,
        "expected_brier": 0.194156,
    },
    {
        "model": "Validation_best_Elo",
        "model_family": "Elo",
        "model_role": "prediction-oriented Elo baseline",
        "k": 30.0,
        "scale": 300.0,
        "expected_log_loss": 0.556534,
        "expected_brier": 0.190073,
    },
]

GLICKO_MODELS = [
    {
        "model": "Glicko_C0_match_by_match",
        "model_family": "Glicko",
        "model_role": "basic Glicko without inactivity inflation",
        "variant": "C0_no_inflation",
        "expected_log_loss": 0.571956,
        "expected_brier": 0.195743,
        "expected_accuracy": 0.693822,
    },
    {
        "model": "Glicko_low_inflation_match_by_match",
        "model_family": "Glicko",
        "model_role": "candidate main Glicko variant",
        "variant": "low_inflation",
        "expected_log_loss": 0.552154,
        "expected_brier": 0.187724,
        "expected_accuracy": 0.711574,
    },
]

RATING_COMPARISONS = [
    ("Validation_best_Elo", "Glicko_low_inflation_match_by_match"),
    ("Default_Elo", "Glicko_low_inflation_match_by_match"),
    ("Validation_best_Elo", "Glicko_C0_match_by_match"),
]

SIMILARITY_GROUPS = [
    ("active_2025_games_ge5", 5, 0),
    ("total_games_ge100", 0, 100),
    ("active_2025_games_ge5_and_total_games_ge100", 5, 100),
]


def find_file(filename: str, preferred_relative: str | None = None) -> Path | None:
    """Find an input file with a preferred project-relative path first."""

    if preferred_relative:
        preferred = PROJECT_ROOT / preferred_relative
        if preferred.exists():
            return preferred
    matches = sorted(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def record_issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    model: str,
    detail: str,
    severity: str = "warning",
) -> None:
    """Append an issue/warning/info row."""

    issues.append(
        {
            "issue_type": issue_type,
            "model": model,
            "detail": detail,
            "severity": severity,
        }
    )


def format_code_value(value: Any) -> str:
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
    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    if code in player_names or pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date/event_date_ordering_method without modifying raw fields."""

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


def load_matches(issues: list[dict[str, Any]]) -> tuple[pd.DataFrame, Path]:
    """Load and chronologically sort the full-history checked match dataset."""

    dataset_path = find_file(
        "matches_1985_2025_checked.csv",
        "outputs/elo_optimization/matches_1985_2025_checked.csv",
    )
    if dataset_path is None:
        raise FileNotFoundError("Could not find matches_1985_2025_checked.csv")

    matches = pd.read_csv(dataset_path, low_memory=False)
    missing_required = [col for col in REQUIRED_MATCH_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{dataset_path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_MATCH_COLUMNS:
        if col not in matches.columns:
            matches[col] = pd.NA
            record_issue(issues, "missing_optional_column", "dataset", f"Filled missing {col} with NA", "info")

    for col in REQUIRED_MATCH_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    missing_ids = matches[REQUIRED_MATCH_COLUMNS].isna().sum()
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
    return matches, dataset_path


def run_elo_model(matches: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run one fixed Elo baseline over full history and return 2025 predictions."""

    start = time.perf_counter()
    model = spec["model"]
    k = float(spec["k"])
    scale = float(spec["scale"])
    ratings: dict[int, float] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}
    predictions_2025: list[dict[str, Any]] = []

    cols = ["fcode", "year", "event", "winner", "loser", "winner_name", "loser_name"]
    for row in matches[cols].itertuples(index=False):
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        update_player_name(player_names, winner, row.winner_name)
        update_player_name(player_names, loser, row.loser_name)

        winner_rating_before = ratings.get(winner, 1500.0)
        loser_rating_before = ratings.get(loser, 1500.0)
        pred_winner_win = 1.0 / (1.0 + 10.0 ** ((loser_rating_before - winner_rating_before) / scale))
        rating_change = k * (1.0 - pred_winner_win)
        ratings[winner] = winner_rating_before + rating_change
        ratings[loser] = loser_rating_before - rating_change

        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

        if int(row.year) == END_YEAR:
            predictions_2025.append(
                {
                    "model": model,
                    "model_family": "Elo",
                    "model_role": spec["model_role"],
                    "year": int(row.year),
                    "game_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "winner": winner,
                    "loser": loser,
                    "pred_winner_win": pred_winner_win,
                    "pre_rating_winner": winner_rating_before,
                    "pre_rating_loser": loser_rating_before,
                    "pre_rd_winner": np.nan,
                    "pre_rd_loser": np.nan,
                    "outcome": 1.0,
                }
            )

    final_rows = []
    for player in sorted(ratings):
        final_rows.append(
            {
                "model": model,
                "model_family": "Elo",
                "model_role": spec["model_role"],
                "player_id": player,
                "player_name": player_names.get(player, pd.NA),
                "rating": ratings[player],
                "rd": np.nan,
                "games_played": games_played[player],
                "wins": wins[player],
                "losses": losses[player],
            }
        )
    final_ratings = pd.DataFrame(final_rows)
    final_ratings["rank_by_rating"] = final_ratings["rating"].rank(method="min", ascending=False).astype(int)
    final_ratings = final_ratings.sort_values(["rank_by_rating", "player_id"]).reset_index(drop=True)
    return pd.DataFrame(predictions_2025), final_ratings, time.perf_counter() - start


def load_glicko_outputs(issues: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Load Glicko C0 and low-inflation predictions/final ratings from step 3 outputs."""

    predictions_path = find_file(
        "meeting5_glicko_rd_inflation_predictions_2025.csv",
        "outputs/meeting5_glicko_rd_inflation/meeting5_glicko_rd_inflation_predictions_2025.csv",
    )
    final_path = find_file(
        "meeting5_glicko_rd_inflation_final_ratings.csv",
        "outputs/meeting5_glicko_rd_inflation/meeting5_glicko_rd_inflation_final_ratings.csv",
    )
    metrics_path = find_file(
        "meeting5_glicko_rd_inflation_metrics_2025.csv",
        "outputs/meeting5_glicko_rd_inflation/meeting5_glicko_rd_inflation_metrics_2025.csv",
    )

    if predictions_path is None or final_path is None:
        raise FileNotFoundError("Missing required meeting5 Glicko RD inflation outputs.")

    pred_raw = pd.read_csv(predictions_path)
    final_raw = pd.read_csv(final_path)
    runtime_by_variant: dict[str, float] = {}
    if metrics_path is not None:
        metrics_raw = pd.read_csv(metrics_path)
        runtime_by_variant = dict(zip(metrics_raw["variant"], metrics_raw["runtime_seconds"]))
    else:
        record_issue(issues, "missing_input_file", "Glicko", "Glicko runtime metrics not found", "warning")

    pred_frames = []
    final_frames = []
    for spec in GLICKO_MODELS:
        variant = spec["variant"]
        model = spec["model"]
        pred = pred_raw.loc[pred_raw["variant"] == variant].copy()
        if pred.empty:
            record_issue(issues, "missing_glicko_variant", model, f"No predictions for {variant}", "warning")
            continue
        pred = pred.rename(columns={"game_id": "game_id"})
        pred["model"] = model
        pred["model_family"] = "Glicko"
        pred["model_role"] = spec["model_role"]
        pred["outcome"] = 1.0
        for col in ["pre_rd_winner", "pre_rd_loser"]:
            if col not in pred.columns:
                pred[col] = np.nan
        pred_frames.append(
            pred[
                [
                    "model",
                    "model_family",
                    "model_role",
                    "year",
                    "game_id",
                    "fcode",
                    "winner",
                    "loser",
                    "pred_winner_win",
                    "pre_rating_winner",
                    "pre_rating_loser",
                    "pre_rd_winner",
                    "pre_rd_loser",
                    "outcome",
                ]
            ]
        )

        final = final_raw.loc[final_raw["variant"] == variant].copy()
        if final.empty:
            record_issue(issues, "missing_glicko_variant", model, f"No final ratings for {variant}", "warning")
            continue
        final["model"] = model
        final["model_family"] = "Glicko"
        final["model_role"] = spec["model_role"]
        final = final.rename(columns={"final_rank_by_rating": "rank_by_rating"})
        final_frames.append(
            final[
                [
                    "model",
                    "model_family",
                    "model_role",
                    "player_id",
                    "player_name",
                    "rating",
                    "rd",
                    "games_played",
                    "wins",
                    "losses",
                    "rank_by_rating",
                ]
            ]
        )

    return pd.concat(pred_frames, ignore_index=True), pd.concat(final_frames, ignore_index=True), runtime_by_variant


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute common winner-perspective prediction metrics."""

    p = predictions["pred_winner_win"].astype(float).to_numpy()
    clipped = np.clip(p, EPS, 1.0 - EPS)
    correct = p >= 0.5
    return {
        "evaluation_games": int(len(predictions)),
        "log_loss": float(-np.mean(np.log(clipped))),
        "brier": float(np.mean((p - 1.0) ** 2)),
        "accuracy": float(np.mean(correct)),
        "mean_predicted_probability": float(np.mean(p)),
        "median_predicted_probability": float(np.median(p)),
        "std_predicted_probability": float(np.std(p)),
        "very_confident_actual_winner_ge_0_9": int(np.sum(p >= 0.9)),
        "very_unconfident_actual_winner_le_0_1": int(np.sum(p <= 0.1)),
    }


def make_calibration_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create winner-perspective calibration bins and model-level summary."""

    bins = [0.0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    labels = [
        "0.00-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]
    data = predictions.copy()
    data["prob_bin"] = pd.cut(
        data["pred_winner_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    summary_rows = []
    for model, group in data.groupby("model", sort=False):
        model_errors = []
        for prob_bin, bin_group in group.groupby("prob_bin", observed=False):
            if bin_group.empty:
                continue
            mean_pred = float(bin_group["pred_winner_win"].mean())
            actual_win_rate = 1.0
            error = actual_win_rate - mean_pred
            abs_error = abs(error)
            games = int(len(bin_group))
            model_errors.append((games, abs_error))
            rows.append(
                {
                    "model": model,
                    "prob_bin": str(prob_bin),
                    "games": games,
                    "mean_predicted_probability": mean_pred,
                    "actual_win_rate": actual_win_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs_error,
                }
            )
        total_games = sum(games for games, _ in model_errors)
        weighted = (
            sum(games * abs_error for games, abs_error in model_errors) / total_games
            if total_games
            else np.nan
        )
        max_abs = max((abs_error for _, abs_error in model_errors), default=np.nan)
        summary_rows.append(
            {
                "model": model,
                "weighted_mean_abs_calibration_error": weighted,
                "max_abs_calibration_error": max_abs,
                "bins_used": len(model_errors),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def make_confidence_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create confidence-bin diagnostics."""

    data = predictions.copy()
    data["confidence"] = np.maximum(data["pred_winner_win"], 1.0 - data["pred_winner_win"])
    data["correct"] = data["pred_winner_win"] >= 0.5
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    data["confidence_bin"] = pd.cut(data["confidence"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for (model, confidence_bin), group in data.groupby(["model", "confidence_bin"], observed=False):
        if group.empty:
            continue
        metrics = evaluate_predictions(group)
        rows.append(
            {
                "model": model,
                "confidence_bin": str(confidence_bin),
                "games": int(len(group)),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "mean_predicted_probability": metrics["mean_predicted_probability"],
                "actual_win_rate": 1.0,
            }
        )
    return pd.DataFrame(rows)


def make_metrics_table(
    predictions: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    runtime_map: dict[str, float],
) -> pd.DataFrame:
    """Build model-level metrics table."""

    rows = []
    calibration_lookup = calibration_summary.set_index("model").to_dict("index")
    for model, group in predictions.groupby("model", sort=False):
        metrics = evaluate_predictions(group)
        first = group.iloc[0]
        cal = calibration_lookup.get(model, {})
        rows.append(
            {
                "model": model,
                "model_family": first["model_family"],
                "model_role": first["model_role"],
                "evaluation_games": metrics["evaluation_games"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "mean_predicted_probability": metrics["mean_predicted_probability"],
                "median_predicted_probability": metrics["median_predicted_probability"],
                "std_predicted_probability": metrics["std_predicted_probability"],
                "very_confident_actual_winner_ge_0_9": metrics["very_confident_actual_winner_ge_0_9"],
                "very_unconfident_actual_winner_le_0_1": metrics["very_unconfident_actual_winner_le_0_1"],
                "weighted_mean_abs_calibration_error": cal.get(
                    "weighted_mean_abs_calibration_error", np.nan
                ),
                "runtime_seconds_if_available": runtime_map.get(model, np.nan),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def make_evaluation_set_check(predictions: pd.DataFrame) -> pd.DataFrame:
    """Check that all models use the same 2025 game IDs and valid probabilities."""

    game_sets = {model: set(group["game_id"].astype(int)) for model, group in predictions.groupby("model")}
    union_ids = set().union(*game_sets.values())
    reference_model = predictions["model"].drop_duplicates().iloc[0]
    reference_ids = game_sets[reference_model]

    rows = []
    for model, group in predictions.groupby("model", sort=False):
        probs = group["pred_winner_win"].astype(float)
        ids = game_sets[model]
        missing_vs_union = len(union_ids - ids)
        extra_vs_reference = len(ids - reference_ids)
        nan_count = int(probs.isna().sum())
        inf_count = int(np.isinf(probs).sum())
        out_of_range = int(((probs < 0.0) | (probs > 1.0)).sum())
        status = "ok"
        if (
            len(group) != EXPECTED_2025_GAMES
            or len(ids) != EXPECTED_2025_GAMES
            or missing_vs_union
            or extra_vs_reference
            or nan_count
            or inf_count
            or out_of_range
        ):
            status = "check_warning"
        rows.append(
            {
                "model": model,
                "evaluation_games": int(len(group)),
                "unique_game_ids": int(len(ids)),
                "missing_game_ids_vs_union": int(missing_vs_union),
                "extra_game_ids_vs_reference": int(extra_vs_reference),
                "probability_min": float(probs.min()),
                "probability_max": float(probs.max()),
                "probability_nan_count": nan_count,
                "probability_inf_count": inf_count,
                "probability_out_of_range_count": out_of_range,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def top_overlap(merged: pd.DataFrame, n: int) -> float:
    if merged.empty:
        return np.nan
    k = min(n, len(merged))
    ref_top = set(merged.sort_values("rating_ref", ascending=False).head(k)["player_id"])
    comp_top = set(merged.sort_values("rating_comp", ascending=False).head(k)["player_id"])
    return float(len(ref_top & comp_top) / k)


def make_rating_similarity(
    final_ratings: pd.DataFrame,
    active_2025_counts: dict[int, int],
    total_game_counts: dict[int, int],
) -> pd.DataFrame:
    """Compare selected Elo/Glicko final rating lists."""

    rows = []
    for left_model, right_model in RATING_COMPARISONS:
        left = final_ratings.loc[final_ratings["model"] == left_model].copy()
        right = final_ratings.loc[final_ratings["model"] == right_model].copy()
        left = left[["player_id", "rating"]].rename(columns={"rating": "rating_ref"})
        right = right[["player_id", "rating"]].rename(columns={"rating": "rating_comp"})
        merged = left.merge(right, on="player_id", how="inner")
        merged["rank_ref"] = merged["rating_ref"].rank(method="min", ascending=False)
        merged["rank_comp"] = merged["rating_comp"].rank(method="min", ascending=False)
        merged["active_2025_games"] = merged["player_id"].map(active_2025_counts).fillna(0).astype(int)
        merged["total_games"] = merged["player_id"].map(total_game_counts).fillna(0).astype(int)
        merged["abs_rank_diff"] = (merged["rank_ref"] - merged["rank_comp"]).abs()

        for group_name, min_active_2025, min_total in SIMILARITY_GROUPS:
            subset = merged.copy()
            if min_active_2025:
                subset = subset[subset["active_2025_games"] >= min_active_2025]
            if min_total:
                subset = subset[subset["total_games"] >= min_total]
            players = int(len(subset))
            if players >= 2:
                spearman = subset["rating_ref"].corr(subset["rating_comp"], method="spearman")
                pearson = subset["rating_ref"].corr(subset["rating_comp"], method="pearson")
                ref_centered = subset["rating_ref"] - subset["rating_ref"].mean()
                comp_centered = subset["rating_comp"] - subset["rating_comp"].mean()
                centered_diff = float((ref_centered - comp_centered).abs().mean())
            else:
                spearman = pearson = centered_diff = np.nan
            rows.append(
                {
                    "comparison": f"{left_model}_vs_{right_model}",
                    "group": group_name,
                    "players": players,
                    "spearman": float(spearman) if not pd.isna(spearman) else np.nan,
                    "pearson": float(pearson) if not pd.isna(pearson) else np.nan,
                    "top50_overlap": top_overlap(subset, 50),
                    "top100_overlap": top_overlap(subset, 100),
                    "mean_abs_rank_diff": float(subset["abs_rank_diff"].mean()) if players else np.nan,
                    "mean_abs_centered_rating_diff": centered_diff,
                }
            )
    return pd.DataFrame(rows)


def add_expected_metric_issues(metrics: pd.DataFrame, issues: list[dict[str, Any]]) -> None:
    """Record warnings if regenerated/read metrics differ from previous expected results."""

    expected_by_model = {
        "Conservative_Elo": (0.585928, 0.201418, None),
        "Default_Elo": (0.567633, 0.194156, None),
        "Validation_best_Elo": (0.556534, 0.190073, None),
        "Glicko_C0_match_by_match": (0.571956, 0.195743, 0.693822),
        "Glicko_low_inflation_match_by_match": (0.552154, 0.187724, 0.711574),
    }
    for model, (expected_logloss, expected_brier, expected_accuracy) in expected_by_model.items():
        row = metrics.loc[metrics["model"] == model]
        if row.empty:
            record_issue(issues, "missing_model", model, "Model missing from metrics table", "warning")
            continue
        first = row.iloc[0]
        if abs(first["log_loss"] - expected_logloss) > 5e-4:
            record_issue(
                issues,
                "metrics_mismatch_from_expected_previous_results",
                model,
                f"log_loss {first['log_loss']:.6f} vs expected approximately {expected_logloss:.6f}",
            )
        if abs(first["brier"] - expected_brier) > 5e-4:
            record_issue(
                issues,
                "metrics_mismatch_from_expected_previous_results",
                model,
                f"Brier {first['brier']:.6f} vs expected approximately {expected_brier:.6f}",
            )
        if expected_accuracy is not None and abs(first["accuracy"] - expected_accuracy) > 5e-4:
            record_issue(
                issues,
                "metrics_mismatch_from_expected_previous_results",
                model,
                f"accuracy {first['accuracy']:.6f} vs expected approximately {expected_accuracy:.6f}",
            )


def save_bar_plot(metrics: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = metrics.sort_values(metric)
    ax.bar(data["model"], data[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_calibration_plot(calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for model, group in calibration.groupby("model", sort=False):
        ax.plot(
            group["mean_predicted_probability"],
            group["actual_win_rate"],
            marker="o",
            linewidth=1.2,
            label=model,
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_title("Winner-perspective calibration")
    ax.set_xlabel("Mean predicted probability for actual winner")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_confidence_brier_plot(confidence: pd.DataFrame) -> None:
    pivot = confidence.pivot(index="confidence_bin", columns="model", values="brier")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Brier score by confidence bin")
    ax.set_ylabel("Brier score")
    ax.set_xlabel("Confidence bin")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CONFIDENCE_BRIER_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_rating_scatter(final_ratings: pd.DataFrame, active_counts: dict[int, int], total_counts: dict[int, int]) -> None:
    elo = final_ratings.loc[final_ratings["model"] == "Validation_best_Elo", ["player_id", "rating"]].rename(
        columns={"rating": "elo_rating"}
    )
    glicko = final_ratings.loc[
        final_ratings["model"] == "Glicko_low_inflation_match_by_match", ["player_id", "rating"]
    ].rename(columns={"rating": "glicko_rating"})
    merged = elo.merge(glicko, on="player_id", how="inner")
    merged["active_2025_games"] = merged["player_id"].map(active_counts).fillna(0).astype(int)
    merged["total_games"] = merged["player_id"].map(total_counts).fillna(0).astype(int)
    subset = merged[(merged["active_2025_games"] >= 5) & (merged["total_games"] >= 100)]

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(subset["elo_rating"], subset["glicko_rating"], s=14, alpha=0.65, color="#4C78A8")
    ax.set_title("Validation-best Elo vs Glicko low-inflation ratings")
    ax.set_xlabel("Validation-best Elo rating")
    ax.set_ylabel("Glicko low-inflation rating")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT_PATH, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
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


def write_summary(
    metrics: pd.DataFrame,
    evaluation_check: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    confidence: pd.DataFrame,
    rating_similarity: pd.DataFrame,
    issues: pd.DataFrame,
    dataset_path: Path,
) -> None:
    """Write meeting-ready markdown summary."""

    ordered = metrics.sort_values(["brier", "log_loss"]).copy()
    val_elo = metrics.loc[metrics["model"] == "Validation_best_Elo"].iloc[0]
    glicko_low = metrics.loc[metrics["model"] == "Glicko_low_inflation_match_by_match"].iloc[0]
    glicko_c0 = metrics.loc[metrics["model"] == "Glicko_C0_match_by_match"].iloc[0]

    if glicko_low["log_loss"] < val_elo["log_loss"] and glicko_low["brier"] < val_elo["brier"]:
        candidate_text = (
            "Glicko low_inflation gives better 2025 prediction metrics than validation-best Elo "
            "in this fixed test comparison. This supports treating it as the current candidate "
            "main predictive model, not as proof that Glicko is universally better."
        )
    else:
        candidate_text = (
            "Validation-best Elo remains very competitive in this fixed test comparison. Glicko's "
            "main value should be discussed in terms of uncertainty interpretation and inactive-player handling."
        )

    if glicko_c0["brier"] > val_elo["brier"] and glicko_low["brier"] < val_elo["brier"]:
        inflation_text = (
            "Glicko C0 is worse than validation-best Elo, while Glicko low_inflation is better. "
            "This suggests the improvement comes from Glicko's uncertainty mechanism combined with "
            "inactivity RD inflation, not simply from using Glicko without inflation."
        )
    else:
        inflation_text = (
            "The comparison does not isolate a clear C0-vs-inflation story; interpret RD inflation cautiously."
        )

    active_similarity = rating_similarity.loc[
        rating_similarity["group"] == "active_2025_games_ge5_and_total_games_ge100"
    ].copy()
    issue_lines = ["- None"] if issues.empty else [
        f"- {row.issue_type}: {row.model} - {row.detail}" for row in issues.itertuples(index=False)
    ]

    lines = [
        "# Meeting 5 Fair Elo-vs-Glicko Comparison",
        "",
        "## Purpose",
        "",
        (
            "This is a fair fixed-test comparison, not parameter tuning. The aim is to compare "
            "model families under the same data, chronological ordering, 2025 evaluation games, "
            "and metrics."
        ),
        "",
        "## Experimental Design",
        "",
        f"- Dataset: `{dataset_path.name}` covering {START_YEAR}-{END_YEAR}.",
        "- Evaluation set: 2025 games.",
        "- Prediction perspective: probability assigned to the actual winner before update.",
        "- Primary metrics: log loss and Brier score.",
        "- Secondary metrics: accuracy, calibration diagnostics, confidence bins, rating-list similarity.",
        "- No adaptive-K Elo is implemented here.",
        "- No new Glicko tuning is performed here.",
        "",
        "## Models Compared",
        "",
        "- Conservative_Elo: K=10, scale=500, stability reference.",
        "- Default_Elo: K=20, scale=500, transparent simple baseline.",
        "- Validation_best_Elo: K=30, scale=300, prediction-oriented Elo baseline.",
        "- Glicko_C0_match_by_match: basic Glicko-1 without inactivity inflation.",
        "- Glicko_low_inflation_match_by_match: candidate Glicko variant from RD inflation sensitivity.",
        "",
        "## Evaluation Set Check",
        "",
        markdown_table(
            evaluation_check,
            [
                "model",
                "evaluation_games",
                "unique_game_ids",
                "missing_game_ids_vs_union",
                "extra_game_ids_vs_reference",
                "probability_min",
                "probability_max",
                "status",
            ],
        ),
        "",
        "## Main 2025 Prediction Results",
        "",
        markdown_table(
            ordered,
            [
                "model",
                "model_family",
                "evaluation_games",
                "log_loss",
                "brier",
                "accuracy",
                "weighted_mean_abs_calibration_error",
            ],
        ),
        "",
        "## Calibration Results",
        "",
        markdown_table(
            calibration_summary.sort_values("model"),
            ["model", "weighted_mean_abs_calibration_error", "max_abs_calibration_error", "bins_used"],
        ),
        "",
        "## Confidence-Bin Results",
        "",
        "Confidence-bin diagnostics are saved in the CSV output. These show how model performance changes as prediction confidence increases.",
        "",
        "## Rating-List Similarity",
        "",
        markdown_table(
            active_similarity,
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
        candidate_text,
        "",
        inflation_text,
        "",
        (
            "If Elo and Glicko ranking lists are similar while Glicko predictions improve, the "
            "main difference may be in probabilistic confidence and uncertainty handling rather "
            "than radically different rankings."
        ),
        "",
        "## Candidate Main Comparison Result",
        "",
        (
            "Use cautious wording: this is the current fixed-test result and a candidate main "
            "comparison outcome. It is not a proof of a universally best model."
        ),
        "",
        "## Remaining Limitations",
        "",
        "- Results are based on the 2025 test set after previous model-selection decisions.",
        "- Elo baselines are transparent, while Glicko adds uncertainty and implementation complexity.",
        "- Accuracy should not be used as the only decision criterion; log loss and Brier score are primary here.",
        "- Adaptive-K Elo is not included in this step.",
        "",
        "## Issues / Warnings",
        "",
        *issue_lines,
        "",
        "## Output Files",
        "",
        f"- `{METRICS_PATH}`",
        f"- `{EVALUATION_CHECK_PATH}`",
        f"- `{PREDICTIONS_PATH}`",
        f"- `{CALIBRATION_PATH}`",
        f"- `{CONFIDENCE_PATH}`",
        f"- `{RATING_SIMILARITY_PATH}`",
        f"- `{FINAL_RATINGS_PATH}`",
        f"- `{ISSUES_PATH}`",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the fair comparison."""

    total_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []

    matches, dataset_path = load_matches(issues)
    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Loaded dataset: {dataset_path}")
    print(f"Matches: {len(matches):,}, players: {players:,}, years: {int(matches['year'].min())}-{int(matches['year'].max())}")

    all_predictions = []
    all_final_ratings = []
    runtime_map: dict[str, float] = {}

    print("\nRunning fixed Elo baselines inside this script...")
    record_issue(
        issues,
        "assumption_or_fallback_logic",
        "Elo",
        "Elo 2025 predictions were regenerated inside this script because reusable fair-comparison prediction files were not available.",
        "info",
    )
    for spec in ELO_MODELS:
        print(f"  {spec['model']} (K={spec['k']}, scale={spec['scale']})")
        pred, final, runtime = run_elo_model(matches, spec)
        all_predictions.append(pred)
        all_final_ratings.append(final)
        runtime_map[spec["model"]] = runtime
        print(f"    runtime={runtime:.1f}s, 2025 predictions={len(pred):,}")

    print("\nLoading existing Glicko outputs...")
    glicko_predictions, glicko_final_ratings, glicko_runtime_by_variant = load_glicko_outputs(issues)
    all_predictions.append(glicko_predictions)
    all_final_ratings.append(glicko_final_ratings)
    runtime_map["Glicko_C0_match_by_match"] = glicko_runtime_by_variant.get("C0_no_inflation", np.nan)
    runtime_map["Glicko_low_inflation_match_by_match"] = glicko_runtime_by_variant.get("low_inflation", np.nan)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    final_ratings_df = pd.concat(all_final_ratings, ignore_index=True)

    calibration_df, calibration_summary_df = make_calibration_table(predictions_df)
    confidence_df = make_confidence_bins(predictions_df)
    metrics_df = make_metrics_table(predictions_df, calibration_summary_df, runtime_map)
    evaluation_check_df = make_evaluation_set_check(predictions_df)

    for row in evaluation_check_df.itertuples(index=False):
        if row.status != "ok":
            record_issue(
                issues,
                "evaluation_set_or_probability_check",
                row.model,
                f"status={row.status}, games={row.evaluation_games}, missing_vs_union={row.missing_game_ids_vs_union}",
                "warning",
            )

    add_expected_metric_issues(metrics_df, issues)

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
    total_game_counts = (
        pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).value_counts().to_dict()
    )
    rating_similarity_df = make_rating_similarity(final_ratings_df, active_2025_counts, total_game_counts)

    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["issue_type", "model", "detail", "severity"])

    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    evaluation_check_df.to_csv(EVALUATION_CHECK_PATH, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    confidence_df.to_csv(CONFIDENCE_PATH, index=False, encoding="utf-8-sig")
    rating_similarity_df.to_csv(RATING_SIMILARITY_PATH, index=False, encoding="utf-8-sig")
    final_ratings_df.to_csv(FINAL_RATINGS_PATH, index=False, encoding="utf-8-sig")
    issues_df.to_csv(ISSUES_PATH, index=False, encoding="utf-8-sig")

    save_bar_plot(metrics_df, "brier", BRIER_PLOT_PATH, "2025 Brier score by model", "Brier score")
    save_bar_plot(metrics_df, "log_loss", LOGLOSS_PLOT_PATH, "2025 log loss by model", "Log loss")
    save_bar_plot(metrics_df, "accuracy", ACCURACY_PLOT_PATH, "2025 accuracy by model", "Accuracy")
    save_calibration_plot(calibration_df)
    save_confidence_brier_plot(confidence_df)
    save_rating_scatter(final_ratings_df, active_2025_counts, total_game_counts)

    write_summary(
        metrics_df,
        evaluation_check_df,
        calibration_summary_df,
        confidence_df,
        rating_similarity_df,
        issues_df,
        dataset_path,
    )

    print("\nMain metrics:")
    print(
        metrics_df[
            ["model", "evaluation_games", "log_loss", "brier", "accuracy", "weighted_mean_abs_calibration_error"]
        ].sort_values("brier").to_string(index=False)
    )

    print("\nEvaluation set check:")
    print(evaluation_check_df[["model", "evaluation_games", "unique_game_ids", "status"]].to_string(index=False))

    print("\nIssues / warnings:")
    if issues_df.empty:
        print("  None")
    else:
        print(issues_df.to_string(index=False))

    print("\nOutput files:")
    for path in [
        METRICS_PATH,
        EVALUATION_CHECK_PATH,
        PREDICTIONS_PATH,
        CALIBRATION_PATH,
        CONFIDENCE_PATH,
        RATING_SIMILARITY_PATH,
        FINAL_RATINGS_PATH,
        ISSUES_PATH,
        SUMMARY_MD_PATH,
        BRIER_PLOT_PATH,
        LOGLOSS_PLOT_PATH,
        ACCURACY_PLOT_PATH,
        CALIBRATION_PLOT_PATH,
        CONFIDENCE_BRIER_PLOT_PATH,
        SCATTER_PLOT_PATH,
    ]:
        print(f"  {path}")
    print("No adaptive-K Elo was run.")
    print("No new Glicko tuning was run.")
    print(f"Total runtime: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
