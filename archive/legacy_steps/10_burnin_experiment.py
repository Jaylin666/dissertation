from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import pandas as pd


AVAILABLE_START_YEAR = 2015
END_YEAR = 2025
EVALUATION_YEAR = 2025

DEFAULT_RATING = 1500.0
K = 20.0
SCALE = 500.0
EPS = 1e-15

BURNIN_CONFIGS = [
    {"run_label": "2025 only", "start_year": 2025},
    {"run_label": "2023-2025", "start_year": 2023},
    {"run_label": "2020-2025", "start_year": 2020},
    {"run_label": "2018-2025", "start_year": 2018},
    {"run_label": "2015-2025", "start_year": 2015},
]


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
DATA_PROCESSED.mkdir(exist_ok=True)

YEAR_RANGE = f"{AVAILABLE_START_YEAR}_{END_YEAR}"
MATCHES_PATH = DATA_PROCESSED / f"matches_{YEAR_RANGE}_checked.csv"
STEP09_METRICS_PATH = DATA_PROCESSED / f"elo_multiyear_metrics_{YEAR_RANGE}.csv"

RESULTS_PATH = DATA_PROCESSED / f"burnin_experiment_results_{YEAR_RANGE}.csv"
CHECKS_PATH = DATA_PROCESSED / f"burnin_experiment_evaluation_checks_{YEAR_RANGE}.csv"
FINAL_RATINGS_PATH = DATA_PROCESSED / f"burnin_experiment_final_ratings_{YEAR_RANGE}.csv"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["event_date_parsed", "winner_name", "loser_name", "eventname"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load the multi-year match table and sort it chronologically."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/07_build_multiyear_match_dataset.py first."
        )

    matches = pd.read_csv(path)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches["event_date_parsed"] = pd.to_datetime(
        matches["event_date_parsed"], errors="coerce"
    )

    missing_key_counts = matches[REQUIRED_COLUMNS].isna().sum()
    missing_key_counts = missing_key_counts[missing_key_counts > 0]
    if not missing_key_counts.empty:
        print("WARNING: missing values found in required columns:")
        print(missing_key_counts.to_string())

    sort_cols = [
        col
        for col in ["year", "event_date_parsed", "event", "code", "fcode"]
        if col in matches.columns
    ]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return matches


def expected_score(rating_a: float, rating_b: float, scale: float = SCALE) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def update_player_name(player_names: Dict[int, str], player_code: int, possible_name) -> None:
    """Store the first non-missing name seen for a player."""
    if player_code in player_names:
        return
    if pd.notna(possible_name) and str(possible_name).strip():
        player_names[player_code] = str(possible_name).strip()


def ensure_player(
    player_code: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
) -> None:
    """Initialise a player at the default rating if needed."""
    if player_code not in ratings:
        ratings[player_code] = DEFAULT_RATING
        stats[player_code] = {"n_games": 0, "n_wins": 0, "n_losses": 0}


def run_elo_for_start_year(
    matches: pd.DataFrame, start_year: int, run_label: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Run simple Elo from one start year through END_YEAR."""
    if start_year < int(matches["year"].min()):
        print(
            f"WARNING: start_year {start_year} is earlier than available data; "
            f"using {int(matches['year'].min())}."
        )
        start_year = int(matches["year"].min())

    run_matches = matches[(matches["year"] >= start_year) & (matches["year"] <= END_YEAR)].copy()
    ratings: Dict[int, float] = {}
    stats: Dict[int, Dict[str, int]] = {}
    player_names: Dict[int, str] = {}
    prediction_rows = []
    rating_change_abs_values = []
    skipped_games = 0

    for row in run_matches.itertuples(index=False):
        winner_value = getattr(row, "winner")
        loser_value = getattr(row, "loser")
        if pd.isna(winner_value) or pd.isna(loser_value):
            skipped_games += 1
            continue

        winner = int(winner_value)
        loser = int(loser_value)
        year = int(getattr(row, "year"))

        ensure_player(winner, ratings, stats)
        ensure_player(loser, ratings, stats)
        update_player_name(player_names, winner, getattr(row, "winner_name", pd.NA))
        update_player_name(player_names, loser, getattr(row, "loser_name", pd.NA))

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, SCALE)

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = pred_winner_win if player_a == winner else 1.0 - pred_winner_win

        rating_change = K * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change

        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after
        stats[winner]["n_games"] += 1
        stats[winner]["n_wins"] += 1
        stats[loser]["n_games"] += 1
        stats[loser]["n_losses"] += 1
        rating_change_abs_values.append(abs(rating_change))
        rating_change_abs_values.append(abs(rating_change))

        prediction_rows.append(
            {
                "run_label": run_label,
                "start_year": start_year,
                "fcode": int(getattr(row, "fcode")),
                "code": int(getattr(row, "code")),
                "year": year,
                "event": getattr(row, "event", pd.NA),
                "winner": winner,
                "loser": loser,
                "player_a": player_a,
                "player_b": player_b,
                "actual_a_win": actual_a_win,
                "pred_a_win": pred_a_win,
                "pred_winner_win": pred_winner_win,
                "winner_rating_before": winner_rating_before,
                "loser_rating_before": loser_rating_before,
                "winner_rating_after": winner_rating_after,
                "loser_rating_after": loser_rating_after,
                "rating_change": rating_change,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    final_ratings = make_final_ratings_df(run_label, start_year, ratings, stats, player_names)
    history_summary = {
        "number_of_games_used": len(predictions),
        "number_of_unique_players_used": len(ratings),
        "skipped_games": skipped_games,
        "rating_change_abs_values": rating_change_abs_values,
    }
    return predictions, final_ratings, history_summary


def make_final_ratings_df(
    run_label: str,
    start_year: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    player_names: Dict[int, str],
) -> pd.DataFrame:
    """Create final ratings for one burn-in configuration."""
    rows = []
    for player_code, final_rating in ratings.items():
        rows.append(
            {
                "run_label": run_label,
                "start_year": start_year,
                "player_code": player_code,
                "player_name": player_names.get(player_code, pd.NA),
                "final_rating": final_rating,
                "n_games": stats[player_code]["n_games"],
                "n_wins": stats[player_code]["n_wins"],
                "n_losses": stats[player_code]["n_losses"],
            }
        )
    final_ratings = pd.DataFrame(rows)
    if final_ratings.empty:
        return final_ratings
    return final_ratings.sort_values(
        ["final_rating", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_evaluation_metrics(predictions: pd.DataFrame, evaluation_year: int) -> Dict[str, object]:
    """Calculate predictive metrics on the fixed evaluation year."""
    eval_df = predictions[predictions["year"] == evaluation_year].copy()
    if eval_df.empty:
        raise ValueError(
            f"No evaluation games found for year {evaluation_year}. "
            "Check the burn-in start year and input matches."
        )

    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)
    clipped_pred = pred.clip(EPS, 1.0 - EPS)

    log_loss = -np.mean(y * np.log(clipped_pred) + (1.0 - y) * np.log(1.0 - clipped_pred))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))
    observed_win_rate = y.mean()

    return {
        "number_of_evaluation_games": len(eval_df),
        "log_loss": log_loss,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "baseline_accuracy": max(observed_win_rate, 1.0 - observed_win_rate),
        "evaluation_fcodes": eval_df["fcode"].astype(int).tolist(),
    }


def compute_new_players_info(predictions: pd.DataFrame, evaluation_year: int) -> Dict[str, int]:
    """Count players seen before evaluation and players new in evaluation year."""
    before_eval = predictions[predictions["year"] < evaluation_year]
    eval_games = predictions[predictions["year"] == evaluation_year]

    players_before = set(before_eval["winner"].astype(int)).union(
        set(before_eval["loser"].astype(int))
    )
    players_eval = set(eval_games["winner"].astype(int)).union(
        set(eval_games["loser"].astype(int))
    )
    new_players = players_eval - players_before

    return {
        "number_of_unique_players_seen_before_evaluation": len(players_before),
        "number_of_new_players_in_evaluation_year": len(new_players),
    }


def compute_stability_metrics(
    rating_change_abs_values: List[float], final_ratings: pd.DataFrame
) -> Dict[str, float]:
    """Calculate rating movement and final rating distribution metrics."""
    rating_changes = np.array(rating_change_abs_values, dtype=float)
    if rating_changes.size == 0:
        average_abs_rating_change = np.nan
        median_abs_rating_change = np.nan
        p95_abs_rating_change = np.nan
        max_abs_rating_change = np.nan
    else:
        average_abs_rating_change = float(np.mean(rating_changes))
        median_abs_rating_change = float(np.median(rating_changes))
        p95_abs_rating_change = float(np.percentile(rating_changes, 95))
        max_abs_rating_change = float(np.max(rating_changes))

    if final_ratings.empty:
        final_rating_mean = np.nan
        final_rating_std = np.nan
        number_of_players_final = 0
    else:
        final_rating_mean = float(final_ratings["final_rating"].mean())
        final_rating_std = float(final_ratings["final_rating"].std())
        number_of_players_final = len(final_ratings)

    return {
        "average_abs_rating_change_all_games": average_abs_rating_change,
        "median_abs_rating_change_all_games": median_abs_rating_change,
        "p95_abs_rating_change_all_games": p95_abs_rating_change,
        "max_abs_rating_change_all_games": max_abs_rating_change,
        "final_rating_mean": final_rating_mean,
        "final_rating_std": final_rating_std,
        "number_of_players_final": number_of_players_final,
    }


def hash_fcodes(fcodes: List[int]) -> str:
    """Create a stable hash for an ordered evaluation fcode list."""
    text = ",".join(str(int(fcode)) for fcode in fcodes)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def make_evaluation_check_row(
    run_label: str,
    start_year: int,
    evaluation_year: int,
    fcodes: List[int],
    reference_fcodes: List[int],
) -> Dict[str, object]:
    """Build one evaluation-set consistency check row."""
    return {
        "run_label": run_label,
        "start_year": start_year,
        "evaluation_year": evaluation_year,
        "number_of_evaluation_games": len(fcodes),
        "min_evaluation_fcode": min(fcodes) if fcodes else np.nan,
        "max_evaluation_fcode": max(fcodes) if fcodes else np.nan,
        "evaluation_fcode_hash": hash_fcodes(fcodes),
        "matches_reference_evaluation_set": fcodes == reference_fcodes,
    }


def compute_top_overlap(
    final_ratings_by_run: Dict[str, pd.DataFrame],
    reference_run_label: str = "2015-2025",
) -> pd.DataFrame:
    """Calculate top-ranking overlap against the longest burn-in run."""
    if reference_run_label not in final_ratings_by_run:
        print(f"WARNING: reference run {reference_run_label!r} not found for top overlap.")
        return pd.DataFrame()

    reference = final_ratings_by_run[reference_run_label]
    rows = []
    for run_label, ratings in final_ratings_by_run.items():
        row = {"run_label": run_label}
        for n in [20, 50, 100]:
            reference_top = set(reference.head(n)["player_code"].astype(int))
            current_top = set(ratings.head(n)["player_code"].astype(int))
            denominator = min(n, len(reference_top), len(current_top))
            if denominator == 0:
                overlap = np.nan
            else:
                overlap = len(reference_top.intersection(current_top)) / denominator
            row[f"top_{n}_overlap_with_2015_run"] = overlap
        rows.append(row)
    return pd.DataFrame(rows)


def read_step09_log_loss() -> float:
    """Read step 09 log loss for sanity checking if available."""
    if not STEP09_METRICS_PATH.exists():
        print(f"WARNING: step 09 metrics file not found: {STEP09_METRICS_PATH}")
        return np.nan
    metrics = pd.read_csv(STEP09_METRICS_PATH)
    if metrics.empty or "log_loss" not in metrics.columns:
        print("WARNING: step 09 metrics file does not contain log_loss.")
        return np.nan
    return float(metrics.iloc[0]["log_loss"])




def run_burnin_experiment(matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all burn-in configurations and collect results."""
    result_rows = []
    check_rows = []
    final_ratings_by_run: Dict[str, pd.DataFrame] = {}
    all_final_ratings = []
    reference_fcodes: List[int] = []

    for config in BURNIN_CONFIGS:
        run_label = config["run_label"]
        start_year = config["start_year"]
        print(f"\n=== Running burn-in config: {run_label} ===")
        predictions, final_ratings, history_summary = run_elo_for_start_year(
            matches, start_year, run_label
        )
        eval_metrics = compute_evaluation_metrics(predictions, EVALUATION_YEAR)
        new_player_info = compute_new_players_info(predictions, EVALUATION_YEAR)
        stability_metrics = compute_stability_metrics(
            history_summary["rating_change_abs_values"], final_ratings
        )

        eval_fcodes = eval_metrics.pop("evaluation_fcodes")
        if not reference_fcodes:
            reference_fcodes = eval_fcodes

        check_rows.append(
            make_evaluation_check_row(
                run_label=run_label,
                start_year=start_year,
                evaluation_year=EVALUATION_YEAR,
                fcodes=eval_fcodes,
                reference_fcodes=reference_fcodes,
            )
        )

        result_row = {
            "run_label": run_label,
            "start_year": start_year,
            "end_year": END_YEAR,
            "evaluation_year": EVALUATION_YEAR,
            "number_of_games_used": history_summary["number_of_games_used"],
            "number_of_unique_players_used": history_summary["number_of_unique_players_used"],
            "skipped_games": history_summary["skipped_games"],
            "default_rating": DEFAULT_RATING,
            "K": K,
            "scale": SCALE,
        }
        result_row.update(eval_metrics)
        result_row.update(new_player_info)
        result_row.update(stability_metrics)
        result_rows.append(result_row)

        final_ratings_by_run[run_label] = final_ratings
        all_final_ratings.append(final_ratings)

    results_df = pd.DataFrame(result_rows)
    checks_df = pd.DataFrame(check_rows)

    overlap_df = compute_top_overlap(final_ratings_by_run, reference_run_label="2015-2025")
    if not overlap_df.empty:
        results_df = results_df.merge(overlap_df, on="run_label", how="left")
    else:
        for n in [20, 50, 100]:
            results_df[f"top_{n}_overlap_with_2015_run"] = np.nan

    final_ratings_all = pd.concat(all_final_ratings, ignore_index=True)
    column_order = [
        "run_label",
        "start_year",
        "end_year",
        "evaluation_year",
        "number_of_games_used",
        "number_of_evaluation_games",
        "number_of_unique_players_used",
        "number_of_unique_players_seen_before_evaluation",
        "number_of_new_players_in_evaluation_year",
        "default_rating",
        "K",
        "scale",
        "log_loss",
        "brier_score",
        "accuracy",
        "baseline_accuracy",
        "average_abs_rating_change_all_games",
        "median_abs_rating_change_all_games",
        "p95_abs_rating_change_all_games",
        "max_abs_rating_change_all_games",
        "final_rating_mean",
        "final_rating_std",
        "number_of_players_final",
        "top_20_overlap_with_2015_run",
        "top_50_overlap_with_2015_run",
        "top_100_overlap_with_2015_run",
    ]
    extra_cols = [col for col in results_df.columns if col not in column_order]
    results_df = results_df[column_order + extra_cols]
    return results_df, checks_df, final_ratings_all


def save_outputs(results_df: pd.DataFrame, checks_df: pd.DataFrame, final_ratings: pd.DataFrame) -> None:
    """Save all burn-in experiment outputs."""
    results_df.to_csv(RESULTS_PATH, index=False)
    checks_df.to_csv(CHECKS_PATH, index=False)
    final_ratings.to_csv(FINAL_RATINGS_PATH, index=False)


def print_command_line_summary(results_df: pd.DataFrame, checks_df: pd.DataFrame) -> None:
    """Print key burn-in experiment results."""
    best_log_loss = results_df.sort_values("log_loss").iloc[0]
    best_brier = results_df.sort_values("brier_score").iloc[0]
    best_accuracy = results_df.sort_values("accuracy", ascending=False).iloc[0]
    identical_eval_sets = bool(checks_df["matches_reference_evaluation_set"].all())

    step09_log_loss = read_step09_log_loss()
    full_run = results_df[results_df["run_label"] == "2015-2025"]
    sanity_message = "not checked"
    if not full_run.empty and not pd.isna(step09_log_loss):
        diff = abs(float(full_run.iloc[0]["log_loss"]) - step09_log_loss)
        if diff < 1e-9:
            sanity_message = f"sanity check passed (log loss difference {diff:.12f})"
        else:
            sanity_message = f"WARNING: sanity check difference is {diff:.12f}"

    print("\n=== Burn-in Experiment Summary ===")
    print("Burn-in configurations tested:")
    for _, row in results_df.iterrows():
        print(
            f"  {row['run_label']}: start_year={int(row['start_year'])}, "
            f"log_loss={row['log_loss']:.6f}, "
            f"brier_score={row['brier_score']:.6f}, "
            f"accuracy={row['accuracy']:.6f}"
        )
    print(f"Evaluation year: {EVALUATION_YEAR}")
    print("Evaluation games per run:")
    print(checks_df[["run_label", "number_of_evaluation_games"]].to_string(index=False))
    print(
        f"Best run by log loss: {best_log_loss['run_label']} "
        f"({best_log_loss['log_loss']:.6f})"
    )
    print(
        f"Best run by Brier score: {best_brier['run_label']} "
        f"({best_brier['brier_score']:.6f})"
    )
    print(
        f"Best run by accuracy: {best_accuracy['run_label']} "
        f"({best_accuracy['accuracy']:.6f})"
    )
    print(f"Evaluation fcode sets are identical: {identical_eval_sets}")
    print(f"2015-2025 sanity check vs step 09: {sanity_message}")
    print("Output paths:")
    print(f"  results: {RESULTS_PATH}")
    print(f"  evaluation checks: {CHECKS_PATH}")
    print(f"  final ratings: {FINAL_RATINGS_PATH}")


def main() -> None:
    matches = load_matches()
    results_df, checks_df, final_ratings = run_burnin_experiment(matches)
    save_outputs(results_df, checks_df, final_ratings)
    print_command_line_summary(results_df, checks_df)


if __name__ == "__main__":
    main()
