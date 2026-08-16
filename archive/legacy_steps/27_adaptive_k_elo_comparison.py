"""Compare the retained Adaptive-K Elo candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo"

METRICS_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_metrics_2025.csv"
EVALUATION_CHECK_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_evaluation_set_check.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_predictions_2025.csv"
CALIBRATION_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_corrected_calibration.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_corrected_confidence_bins.csv"
ACTIVITY_SUBGROUPS_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_activity_subgroups.csv"
RATING_SIMILARITY_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_rating_similarity.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_final_ratings.csv"
K_DIAGNOSTICS_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_k_diagnostics.csv"
ISSUES_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_issues.csv"

BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_brier_bar.png"
LOGLOSS_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_logloss_bar.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_accuracy_bar.png"
CALIBRATION_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_corrected_calibration_plot.png"
CONFIDENCE_BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_confidence_brier.png"
K_USAGE_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_elo_k_usage.png"
SCATTER_PLOT_PATH = OUTPUT_DIR / "meeting5_adaptive_k_vs_glicko_low_scatter.png"

REQUIRED_MATCH_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_MATCH_COLUMNS = ["event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

REFERENCE_MODELS = [
    "Default_Elo",
    "Validation_best_Elo",
    "Glicko_low_inflation_match_by_match",
]

REFERENCE_MODEL_ROLES = {
    "Default_Elo": "transparent simple baseline",
    "Validation_best_Elo": "prediction-oriented Elo baseline",
    "Glicko_low_inflation_match_by_match": "candidate Glicko reference from RD inflation sensitivity",
}

ADAPTIVE_MODELS = [
    {
        "model": "AdaptiveK_TotalGames_Elo",
        "model_family": "Adaptive-K Elo",
        "model_role": "simple adaptive baseline using total previous games",
        "rule": "total_previous_games",
        "scale": 500.0,
        "sensitivity": False,
    },
    {
        "model": "AdaptiveK_PreviousYearGames_Elo",
        "model_family": "Adaptive-K Elo",
        "model_role": "simple adaptive baseline using previous-year activity",
        "rule": "previous_year_games",
        "scale": 500.0,
        "sensitivity": False,
    },
    {
        "model": "AdaptiveK_TotalGames_Elo_scale300",
        "model_family": "Adaptive-K Elo",
        "model_role": "sensitivity variant: total previous games with scale=300",
        "rule": "total_previous_games",
        "scale": 300.0,
        "sensitivity": True,
    },
    {
        "model": "AdaptiveK_PreviousYearGames_Elo_scale300",
        "model_family": "Adaptive-K Elo",
        "model_role": "sensitivity variant: previous-year activity with scale=300",
        "rule": "previous_year_games",
        "scale": 300.0,
        "sensitivity": True,
    },
]

RATING_COMPARISONS = [
    ("Validation_best_Elo", "AdaptiveK_TotalGames_Elo"),
    ("Validation_best_Elo", "AdaptiveK_PreviousYearGames_Elo"),
    ("AdaptiveK_TotalGames_Elo", "Glicko_low_inflation_match_by_match"),
    ("AdaptiveK_PreviousYearGames_Elo", "Glicko_low_inflation_match_by_match"),
]

SIMILARITY_GROUPS = [
    ("active_2025_games_ge5", 5, 0),
    ("total_games_ge100", 0, 100),
    ("active_2025_games_ge5_and_total_games_ge100", 5, 100),
]

EXPECTED_METRICS = {
    "Default_Elo": (0.567633, 0.194156, 0.701204),
    "Validation_best_Elo": (0.556534, 0.190073, 0.704543),
    "Glicko_low_inflation_match_by_match": (0.552154, 0.187724, 0.711574),
}


def find_file(filename: str, preferred_relative: str | None = None) -> Path | None:
    """Find a project file using a preferred path first and Path.rglob as fallback."""

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


def player_code(value: Any) -> int:
    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    if code in player_names or pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event ordering columns without modifying raw date fields."""

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
    """Load and chronologically sort the full-history match dataset."""

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


def build_player_year_counts(matches: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Precompute player-year game counts."""

    rows = []
    rows.append(matches[["year", "winner"]].rename(columns={"winner": "player_id"}))
    rows.append(matches[["year", "loser"]].rename(columns={"loser": "player_id"}))
    long_df = pd.concat(rows, ignore_index=True)
    grouped = long_df.groupby(["player_id", "year"]).size()
    return {(int(player), int(year)): int(count) for (player, year), count in grouped.items()}


def build_2025_activity_context(
    matches: pd.DataFrame,
    player_year_counts: dict[tuple[int, int], int],
) -> pd.DataFrame:
    """Create per-2025-game activity context shared by all models."""

    total_games: defaultdict[int, int] = defaultdict(int)
    rows = []
    for row in matches[["fcode", "year", "winner", "loser"]].itertuples(index=False):
        year = int(row.year)
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        winner_prev_total = total_games[winner]
        loser_prev_total = total_games[loser]
        winner_prev_year = player_year_counts.get((winner, year - 1), 0)
        loser_prev_year = player_year_counts.get((loser, year - 1), 0)

        if year == END_YEAR:
            rows.append(
                {
                    "game_id": int(row.fcode),
                    "winner_previous_total_games": winner_prev_total,
                    "loser_previous_total_games": loser_prev_total,
                    "winner_previous_year_games": winner_prev_year,
                    "loser_previous_year_games": loser_prev_year,
                    "min_previous_total_games": min(winner_prev_total, loser_prev_total),
                    "min_previous_year_games": min(winner_prev_year, loser_prev_year),
                }
            )

        total_games[winner] += 1
        total_games[loser] += 1

    return pd.DataFrame(rows)


def adaptive_k_total(previous_total_games: int) -> float:
    if previous_total_games < 20:
        return 30.0
    if previous_total_games < 100:
        return 20.0
    return 10.0


def adaptive_k_previous_year(previous_year_games: int) -> float:
    if previous_year_games <= 5:
        return 30.0
    if previous_year_games <= 30:
        return 20.0
    return 10.0


def choose_k(spec: dict[str, Any], previous_total_games: int, previous_year_games: int) -> float:
    if spec["rule"] == "total_previous_games":
        return adaptive_k_total(previous_total_games)
    if spec["rule"] == "previous_year_games":
        return adaptive_k_previous_year(previous_year_games)
    raise ValueError(f"Unknown adaptive K rule: {spec['rule']}")


def run_adaptive_elo(
    matches: pd.DataFrame,
    spec: dict[str, Any],
    player_year_counts: dict[tuple[int, int], int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Run one adaptive-K Elo model and return 2025 predictions/final ratings/K diagnostics."""

    start = time.perf_counter()
    model = spec["model"]
    scale = float(spec["scale"])
    ratings: dict[int, float] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}
    predictions_2025: list[dict[str, Any]] = []
    k_usage: defaultdict[tuple[str, float, str], int] = defaultdict(int)

    cols = ["fcode", "year", "winner", "loser", "winner_name", "loser_name"]
    for row in matches[cols].itertuples(index=False):
        year = int(row.year)
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        update_player_name(player_names, winner, row.winner_name)
        update_player_name(player_names, loser, row.loser_name)

        winner_prev_total = games_played[winner]
        loser_prev_total = games_played[loser]
        winner_prev_year = player_year_counts.get((winner, year - 1), 0)
        loser_prev_year = player_year_counts.get((loser, year - 1), 0)
        winner_k = choose_k(spec, winner_prev_total, winner_prev_year)
        loser_k = choose_k(spec, loser_prev_total, loser_prev_year)

        winner_rating_before = ratings.get(winner, 1500.0)
        loser_rating_before = ratings.get(loser, 1500.0)
        pred_actual_winner_win = 1.0 / (
            1.0 + 10.0 ** ((loser_rating_before - winner_rating_before) / scale)
        )
        ratings[winner] = winner_rating_before + winner_k * (1.0 - pred_actual_winner_win)
        ratings[loser] = loser_rating_before + loser_k * (0.0 - (1.0 - pred_actual_winner_win))

        k_usage[("all_matches", winner_k, "winner")] += 1
        k_usage[("all_matches", loser_k, "loser")] += 1

        if year == END_YEAR:
            pred_favourite_win = max(pred_actual_winner_win, 1.0 - pred_actual_winner_win)
            favourite_actual_win = 1.0 if pred_actual_winner_win >= 0.5 else 0.0
            predictions_2025.append(
                {
                    "model": model,
                    "model_family": spec["model_family"],
                    "model_role": spec["model_role"],
                    "year": year,
                    "game_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "winner": winner,
                    "loser": loser,
                    "pred_actual_winner_win": pred_actual_winner_win,
                    "pred_favourite_win": pred_favourite_win,
                    "favourite_actual_win": favourite_actual_win,
                    "pre_rating_winner": winner_rating_before,
                    "pre_rating_loser": loser_rating_before,
                    "winner_K": winner_k,
                    "loser_K": loser_k,
                    "winner_previous_total_games": winner_prev_total,
                    "loser_previous_total_games": loser_prev_total,
                    "winner_previous_year_games": winner_prev_year,
                    "loser_previous_year_games": loser_prev_year,
                    "rd": np.nan,
                }
            )
            k_usage[("2025", winner_k, "winner")] += 1
            k_usage[("2025", loser_k, "loser")] += 1

        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

    final_rows = []
    for player in sorted(ratings):
        final_rows.append(
            {
                "model": model,
                "model_family": spec["model_family"],
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

    k_rows = []
    for scope in ["all_matches", "2025"]:
        for player_role in ["winner", "loser"]:
            total_uses = sum(
                count
                for (scope_value, _, role_value), count in k_usage.items()
                if scope_value == scope and role_value == player_role
            )
            for (scope_value, k_value, role_value), count in sorted(k_usage.items()):
                if scope_value == scope and role_value == player_role:
                    k_rows.append(
                        {
                            "model": model,
                            "scope": scope,
                            "K_value": k_value,
                            "player_role": player_role,
                            "uses": int(count),
                            "percentage": float(count / total_uses) if total_uses else np.nan,
                        }
                    )
    return pd.DataFrame(predictions_2025), final_ratings, pd.DataFrame(k_rows), time.perf_counter() - start


def load_reference_outputs(
    activity_context: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Load Default Elo, Validation-best Elo, and Glicko low from step 5 outputs."""

    pred_path = find_file(
        "meeting5_fair_elo_vs_glicko_predictions_2025.csv",
        "outputs/meeting5_fair_elo_vs_glicko/meeting5_fair_elo_vs_glicko_predictions_2025.csv",
    )
    final_path = find_file(
        "meeting5_fair_elo_vs_glicko_final_ratings.csv",
        "outputs/meeting5_fair_elo_vs_glicko/meeting5_fair_elo_vs_glicko_final_ratings.csv",
    )
    metrics_path = find_file(
        "meeting5_fair_elo_vs_glicko_metrics_2025.csv",
        "outputs/meeting5_fair_elo_vs_glicko/meeting5_fair_elo_vs_glicko_metrics_2025.csv",
    )

    if pred_path is None or final_path is None:
        raise FileNotFoundError("Missing step 5 fair comparison predictions/final ratings.")

    pred = pd.read_csv(pred_path)
    final = pd.read_csv(final_path)
    runtime_map: dict[str, float] = {}
    if metrics_path is not None:
        metrics = pd.read_csv(metrics_path)
        runtime_map = dict(zip(metrics["model"], metrics["runtime_seconds_if_available"]))
    else:
        record_issue(issues, "missing_input_file", "reference", "Step 5 metrics file missing", "warning")

    pred = pred.loc[pred["model"].isin(REFERENCE_MODELS)].copy()
    final = final.loc[final["model"].isin(REFERENCE_MODELS)].copy()
    if pred["model"].nunique() != len(REFERENCE_MODELS):
        record_issue(issues, "missing_reference_model", "reference", "Some reference model predictions are missing", "warning")
    if final["model"].nunique() != len(REFERENCE_MODELS):
        record_issue(issues, "missing_reference_model", "reference", "Some reference model final ratings are missing", "warning")

    pred = pred.rename(columns={"pred_winner_win": "pred_actual_winner_win"})
    pred["pred_favourite_win"] = np.maximum(pred["pred_actual_winner_win"], 1.0 - pred["pred_actual_winner_win"])
    pred["favourite_actual_win"] = np.where(pred["pred_actual_winner_win"] >= 0.5, 1.0, 0.0)
    pred["model_role"] = pred["model"].map(REFERENCE_MODEL_ROLES).fillna(pred.get("model_role", "reference model"))
    pred = pred.merge(activity_context, on="game_id", how="left")
    for col in ["winner_K", "loser_K"]:
        if col not in pred.columns:
            pred[col] = np.nan
    for col in ["rd"]:
        if col not in pred.columns:
            pred[col] = np.nan

    output_cols = [
        "model",
        "model_family",
        "model_role",
        "year",
        "game_id",
        "fcode",
        "winner",
        "loser",
        "pred_actual_winner_win",
        "pred_favourite_win",
        "favourite_actual_win",
        "pre_rating_winner",
        "pre_rating_loser",
        "winner_K",
        "loser_K",
        "winner_previous_total_games",
        "loser_previous_total_games",
        "winner_previous_year_games",
        "loser_previous_year_games",
        "rd",
    ]
    return pred[output_cols], final, runtime_map


def evaluate_actual_winner_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate actual-winner perspective predictions."""

    p = predictions["pred_actual_winner_win"].astype(float).to_numpy()
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return {
        "evaluation_games": int(len(predictions)),
        "log_loss": float(-np.mean(np.log(clipped))),
        "brier": float(np.mean((p - 1.0) ** 2)),
        "accuracy": float(np.mean(p >= 0.5)),
        "mean_pred_actual_winner_win": float(np.mean(p)),
        "median_pred_actual_winner_win": float(np.median(p)),
        "std_pred_actual_winner_win": float(np.std(p)),
    }


def evaluate_favourite_predictions(group: pd.DataFrame) -> dict[str, Any]:
    """Evaluate corrected favourite-perspective predictions."""

    pred = group["pred_favourite_win"].astype(float).to_numpy()
    y = group["favourite_actual_win"].astype(float).to_numpy()
    clipped = np.clip(pred, EPS, 1.0 - EPS)
    return {
        "games": int(len(group)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
        "brier": float(np.mean((pred - y) ** 2)),
        "accuracy": float(np.mean(y == 1.0)),
        "mean_predicted_favourite_probability": float(np.mean(pred)),
        "actual_favourite_win_rate": float(np.mean(y)),
    }


def make_corrected_calibration(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create corrected predicted-favourite calibration tables."""

    data = predictions.copy()
    bins = np.arange(0.50, 1.00001, 0.05)
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    data["prob_bin"] = pd.cut(
        data["pred_favourite_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    rows = []
    summary_rows = []
    for model, model_group in data.groupby("model", sort=False):
        model_errors = []
        for prob_bin, group in model_group.groupby("prob_bin", observed=False):
            if group.empty:
                continue
            mean_pred = float(group["pred_favourite_win"].mean())
            actual_rate = float(group["favourite_actual_win"].mean())
            error = actual_rate - mean_pred
            abs_error = abs(error)
            games = int(len(group))
            model_errors.append((games, abs_error))
            rows.append(
                {
                    "model": model,
                    "prob_bin": str(prob_bin),
                    "games": games,
                    "mean_predicted_favourite_probability": mean_pred,
                    "actual_favourite_win_rate": actual_rate,
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
                "weighted_mean_abs_calibration_error_corrected": weighted,
                "max_abs_calibration_error": max_abs,
                "bins_used": len(model_errors),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def make_corrected_confidence_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create corrected favourite-perspective confidence-bin diagnostics."""

    data = predictions.copy()
    bins = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    data["confidence_bin"] = pd.cut(
        data["pred_favourite_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    rows = []
    for (model, confidence_bin), group in data.groupby(["model", "confidence_bin"], observed=False):
        if group.empty:
            continue
        metrics = evaluate_favourite_predictions(group)
        rows.append(
            {
                "model": model,
                "confidence_bin": str(confidence_bin),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def make_activity_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create subgroup metrics by previous total games and previous-year activity."""

    data = predictions.copy()
    data["min_previous_total_games"] = np.minimum(
        data["winner_previous_total_games"], data["loser_previous_total_games"]
    )
    data["min_previous_year_games"] = np.minimum(
        data["winner_previous_year_games"], data["loser_previous_year_games"]
    )

    data["total_previous_games_subgroup"] = pd.cut(
        data["min_previous_total_games"],
        bins=[-1, 19, 99, np.inf],
        labels=["min_games_0_to_19", "min_games_20_to_99", "min_games_100plus"],
    )
    data["previous_year_games_subgroup"] = pd.cut(
        data["min_previous_year_games"],
        bins=[-1, 5, 30, np.inf],
        labels=[
            "min_prev_year_games_0_to_5",
            "min_prev_year_games_6_to_30",
            "min_prev_year_games_31plus",
        ],
    )

    rows = []
    for subgroup_type, subgroup_col in [
        ("total_previous_games", "total_previous_games_subgroup"),
        ("previous_year_games", "previous_year_games_subgroup"),
    ]:
        for (model, subgroup), group in data.groupby(["model", subgroup_col], observed=False):
            if group.empty:
                continue
            metrics = evaluate_actual_winner_predictions(group)
            rows.append(
                {
                    "model": model,
                    "subgroup_type": subgroup_type,
                    "subgroup": str(subgroup),
                    "games": metrics["evaluation_games"],
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "accuracy": metrics["accuracy"],
                    "mean_pred_actual_winner_win": metrics["mean_pred_actual_winner_win"],
                }
            )
    return pd.DataFrame(rows)


def make_metrics_table(
    predictions: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    runtime_map: dict[str, float],
) -> pd.DataFrame:
    """Build model-level metrics table."""

    calibration_lookup = calibration_summary.set_index("model").to_dict("index")
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        metrics = evaluate_actual_winner_predictions(group)
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
                "mean_pred_actual_winner_win": metrics["mean_pred_actual_winner_win"],
                "median_pred_actual_winner_win": metrics["median_pred_actual_winner_win"],
                "std_pred_actual_winner_win": metrics["std_pred_actual_winner_win"],
                "weighted_mean_abs_calibration_error_corrected": cal.get(
                    "weighted_mean_abs_calibration_error_corrected", np.nan
                ),
                "runtime_seconds_if_available": runtime_map.get(model, np.nan),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def make_evaluation_set_check(predictions: pd.DataFrame) -> pd.DataFrame:
    """Check all models use the same 2025 game IDs and valid probabilities."""

    game_sets = {model: set(group["game_id"].astype(int)) for model, group in predictions.groupby("model")}
    union_ids = set().union(*game_sets.values())
    reference_model = predictions["model"].drop_duplicates().iloc[0]
    reference_ids = game_sets[reference_model]
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        probs = group["pred_actual_winner_win"].astype(float)
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
    """Compare selected final rating lists."""

    rows = []
    for left_model, right_model in RATING_COMPARISONS:
        left = final_ratings.loc[final_ratings["model"] == left_model, ["player_id", "rating"]].rename(
            columns={"rating": "rating_ref"}
        )
        right = final_ratings.loc[final_ratings["model"] == right_model, ["player_id", "rating"]].rename(
            columns={"rating": "rating_comp"}
        )
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
    """Record warnings if reference model metrics differ from previous step."""

    for model, (expected_logloss, expected_brier, expected_accuracy) in EXPECTED_METRICS.items():
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
                f"log_loss {first['log_loss']:.6f} vs expected {expected_logloss:.6f}",
            )
        if abs(first["brier"] - expected_brier) > 5e-4:
            record_issue(
                issues,
                "metrics_mismatch_from_expected_previous_results",
                model,
                f"Brier {first['brier']:.6f} vs expected {expected_brier:.6f}",
            )
        if abs(first["accuracy"] - expected_accuracy) > 5e-4:
            record_issue(
                issues,
                "metrics_mismatch_from_expected_previous_results",
                model,
                f"accuracy {first['accuracy']:.6f} vs expected {expected_accuracy:.6f}",
            )


def save_bar_plot(metrics: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
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
    fig, ax = plt.subplots(figsize=(8, 5.3))
    for model, group in calibration.groupby("model", sort=False):
        ax.plot(
            group["mean_predicted_favourite_probability"],
            group["actual_favourite_win_rate"],
            marker="o",
            linewidth=1.2,
            label=model,
        )
    ax.plot([0.5, 1.0], [0.5, 1.0], color="black", linestyle="--", linewidth=1)
    ax.set_title("Corrected favourite-perspective calibration")
    ax.set_xlabel("Mean predicted favourite probability")
    ax.set_ylabel("Actual favourite win rate")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_confidence_brier_plot(confidence: pd.DataFrame) -> None:
    pivot = confidence.pivot(index="confidence_bin", columns="model", values="brier")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Corrected favourite-perspective Brier score by confidence bin")
    ax.set_ylabel("Brier score")
    ax.set_xlabel("Confidence bin")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CONFIDENCE_BRIER_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_k_usage_plot(k_diagnostics: pd.DataFrame) -> None:
    data = k_diagnostics[
        (k_diagnostics["scope"] == "2025")
        & (k_diagnostics["player_role"] == "winner")
        & (k_diagnostics["model"].isin(["AdaptiveK_TotalGames_Elo", "AdaptiveK_PreviousYearGames_Elo"]))
    ].copy()
    if data.empty:
        return
    pivot = data.pivot(index="model", columns="K_value", values="percentage").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Adaptive-K usage in 2025 winner updates")
    ax.set_ylabel("Share of winner updates")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="K")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(K_USAGE_PLOT_PATH, dpi=160)
    plt.close(fig)


def save_rating_scatter(final_ratings: pd.DataFrame, active_counts: dict[int, int], total_counts: dict[int, int], best_adaptive: str) -> None:
    adaptive = final_ratings.loc[final_ratings["model"] == best_adaptive, ["player_id", "rating"]].rename(
        columns={"rating": "adaptive_rating"}
    )
    glicko = final_ratings.loc[
        final_ratings["model"] == "Glicko_low_inflation_match_by_match", ["player_id", "rating"]
    ].rename(columns={"rating": "glicko_rating"})
    merged = adaptive.merge(glicko, on="player_id", how="inner")
    merged["active_2025_games"] = merged["player_id"].map(active_counts).fillna(0).astype(int)
    merged["total_games"] = merged["player_id"].map(total_counts).fillna(0).astype(int)
    subset = merged[(merged["active_2025_games"] >= 5) & (merged["total_games"] >= 100)]

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(subset["adaptive_rating"], subset["glicko_rating"], s=14, alpha=0.65, color="#4C78A8")
    ax.set_title(f"{best_adaptive} vs Glicko low-inflation ratings")
    ax.set_xlabel(f"{best_adaptive} rating")
    ax.set_ylabel("Glicko low-inflation rating")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT_PATH, dpi=160)
    plt.close(fig)






def main() -> None:
    """Run the adaptive-K Elo comparison."""

    total_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []

    matches, dataset_path = load_matches(issues)
    player_year_counts = build_player_year_counts(matches)
    activity_context = build_2025_activity_context(matches, player_year_counts)
    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Loaded dataset: {dataset_path}")
    print(f"Matches: {len(matches):,}, players: {players:,}, years: {int(matches['year'].min())}-{int(matches['year'].max())}")

    print("\nLoading reference models from step 5 outputs...")
    reference_predictions, reference_final_ratings, runtime_map = load_reference_outputs(activity_context, issues)

    all_predictions = [reference_predictions]
    all_final_ratings = [reference_final_ratings]
    all_k_diagnostics = []

    print("\nRunning adaptive-K Elo variants...")
    for spec in ADAPTIVE_MODELS:
        print(f"  {spec['model']} (rule={spec['rule']}, scale={spec['scale']})")
        pred, final, k_diag, runtime = run_adaptive_elo(matches, spec, player_year_counts)
        all_predictions.append(pred)
        all_final_ratings.append(final)
        all_k_diagnostics.append(k_diag)
        runtime_map[spec["model"]] = runtime
        print(f"    runtime={runtime:.1f}s, 2025 predictions={len(pred):,}")
        if spec["sensitivity"]:
            record_issue(
                issues,
                "assumption_or_fallback_logic",
                spec["model"],
                "Included as optional scale=300 sensitivity variant; not part of the main adaptive-K comparison.",
                "info",
            )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    final_ratings_df = pd.concat(all_final_ratings, ignore_index=True)
    k_diagnostics_df = pd.concat(all_k_diagnostics, ignore_index=True)

    calibration_df, calibration_summary_df = make_corrected_calibration(predictions_df)
    confidence_df = make_corrected_confidence_bins(predictions_df)
    activity_subgroups_df = make_activity_subgroups(predictions_df)
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
    activity_subgroups_df.to_csv(ACTIVITY_SUBGROUPS_PATH, index=False, encoding="utf-8-sig")
    rating_similarity_df.to_csv(RATING_SIMILARITY_PATH, index=False, encoding="utf-8-sig")
    final_ratings_df.to_csv(FINAL_RATINGS_PATH, index=False, encoding="utf-8-sig")
    k_diagnostics_df.to_csv(K_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")
    issues_df.to_csv(ISSUES_PATH, index=False, encoding="utf-8-sig")

    save_bar_plot(metrics_df, "brier", BRIER_PLOT_PATH, "2025 Brier score by model", "Brier score")
    save_bar_plot(metrics_df, "log_loss", LOGLOSS_PLOT_PATH, "2025 log loss by model", "Log loss")
    save_bar_plot(metrics_df, "accuracy", ACCURACY_PLOT_PATH, "2025 accuracy by model", "Accuracy")
    save_calibration_plot(calibration_df)
    save_confidence_brier_plot(confidence_df)
    save_k_usage_plot(k_diagnostics_df)
    best_main_adaptive = metrics_df.loc[
        metrics_df["model"].isin(["AdaptiveK_TotalGames_Elo", "AdaptiveK_PreviousYearGames_Elo"])
    ].sort_values(["brier", "log_loss"]).iloc[0]["model"]
    save_rating_scatter(final_ratings_df, active_2025_counts, total_game_counts, best_main_adaptive)


    print("\nMain metrics:")
    print(
        metrics_df[
            [
                "model",
                "evaluation_games",
                "log_loss",
                "brier",
                "accuracy",
                "weighted_mean_abs_calibration_error_corrected",
            ]
        ]
        .sort_values("brier")
        .to_string(index=False)
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
        ACTIVITY_SUBGROUPS_PATH,
        RATING_SIMILARITY_PATH,
        FINAL_RATINGS_PATH,
        K_DIAGNOSTICS_PATH,
        ISSUES_PATH,
        BRIER_PLOT_PATH,
        LOGLOSS_PLOT_PATH,
        ACCURACY_PLOT_PATH,
        CALIBRATION_PLOT_PATH,
        CONFIDENCE_BRIER_PLOT_PATH,
        K_USAGE_PLOT_PATH,
        SCATTER_PLOT_PATH,
    ]:
        print(f"  {path}")
    print("No Glicko rerun was performed.")
    print("No Glicko-2 was implemented.")
    print(f"Total runtime: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
