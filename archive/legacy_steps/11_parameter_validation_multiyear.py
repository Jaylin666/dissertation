from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import pandas as pd


START_YEAR = 2015
END_YEAR = 2025
TRAIN_BURNIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

DEFAULT_RATING = 1500.0
K_VALUES = [10, 15, 20, 25, 30, 35, 40]
SCALE_VALUES = [300, 400, 500, 600]
EPS = 1e-15


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
DATA_PROCESSED.mkdir(exist_ok=True)

YEAR_RANGE = f"{START_YEAR}_{END_YEAR}"
MATCHES_PATH = DATA_PROCESSED / f"matches_{YEAR_RANGE}_checked.csv"

RESULTS_PATH = DATA_PROCESSED / f"parameter_validation_results_{YEAR_RANGE}.csv"
BEST_TEST_PATH = DATA_PROCESSED / f"best_parameter_test_result_{YEAR_RANGE}.csv"
SUMMARY_MD_PATH = DATA_PROCESSED / f"parameter_validation_summary_{YEAR_RANGE}.md"
BEST_PREDICTIONS_PATH = DATA_PROCESSED / f"parameter_validation_best_predictions_{YEAR_RANGE}.csv"
CHECK_SUMMARY_PATH = DATA_PROCESSED / f"parameter_validation_check_summary_{YEAR_RANGE}.csv"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["event_date_parsed", "eventname", "winner_name", "loser_name"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load the checked multi-year match dataset and sort it chronologically."""
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
            print(f"WARNING: optional column {col!r} is missing; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches["event_date_parsed"] = pd.to_datetime(
        matches["event_date_parsed"], errors="coerce"
    )

    missing_keys = matches[REQUIRED_COLUMNS].isna().sum()
    missing_keys = missing_keys[missing_keys > 0]
    if not missing_keys.empty:
        print("WARNING: missing values found in required columns:")
        print(missing_keys.to_string())
        print("Rows with missing winner or loser will be skipped during Elo runs.")

    sort_cols = [
        col
        for col in ["year", "event_date_parsed", "event", "code", "fcode"]
        if col in matches.columns
    ]
    return matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def run_elo_for_parameters(matches: pd.DataFrame, k: float, scale: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Run simple Elo from START_YEAR to END_YEAR for one K/scale pair."""
    run_matches = matches[
        (matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)
    ].copy()

    ratings: Dict[int, float] = {}
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

        if winner not in ratings:
            ratings[winner] = DEFAULT_RATING
        if loser not in ratings:
            ratings[loser] = DEFAULT_RATING

        if winner not in player_names:
            winner_name = getattr(row, "winner_name", pd.NA)
            if pd.notna(winner_name) and str(winner_name).strip():
                player_names[winner] = str(winner_name).strip()
        if loser not in player_names:
            loser_name = getattr(row, "loser_name", pd.NA)
            if pd.notna(loser_name) and str(loser_name).strip():
                player_names[loser] = str(loser_name).strip()

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, scale)

        # Evaluation side is fixed by player code, not by winner/loser.
        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = pred_winner_win if player_a == winner else 1.0 - pred_winner_win

        rating_change = k * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change

        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after
        rating_change_abs_values.append(abs(rating_change))
        rating_change_abs_values.append(abs(rating_change))

        prediction_rows.append(
            {
                "fcode": int(getattr(row, "fcode")),
                "code": int(getattr(row, "code")),
                "year": year,
                "event": getattr(row, "event", pd.NA),
                "eventname": getattr(row, "eventname", pd.NA),
                "winner": winner,
                "loser": loser,
                "winner_name": getattr(row, "winner_name", pd.NA),
                "loser_name": getattr(row, "loser_name", pd.NA),
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
                "K": k,
                "scale": scale,
                "is_validation_game": year in VALIDATION_YEARS,
                "is_test_game": year in TEST_YEARS,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    final_ratings = make_final_ratings_df(ratings, player_names)

    if rating_change_abs_values:
        average_abs_rating_change = float(np.mean(rating_change_abs_values))
    else:
        average_abs_rating_change = np.nan

    run_stats = {
        "average_abs_rating_change_all_years": average_abs_rating_change,
        "skipped_games": skipped_games,
        "final_rating_mean": float(final_ratings["final_rating"].mean()) if not final_ratings.empty else np.nan,
        "final_rating_std": float(final_ratings["final_rating"].std()) if not final_ratings.empty else np.nan,
    }
    return predictions, final_ratings, run_stats


def make_final_ratings_df(ratings: Dict[int, float], player_names: Dict[int, str]) -> pd.DataFrame:
    """Create final ratings for one parameter setting."""
    rows = []
    for player_code, final_rating in ratings.items():
        rows.append(
            {
                "player_code": player_code,
                "player_name": player_names.get(player_code, pd.NA),
                "final_rating": final_rating,
            }
        )
    final_ratings = pd.DataFrame(rows)
    if final_ratings.empty:
        return final_ratings
    return final_ratings.sort_values(
        ["final_rating", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_metrics(predictions: pd.DataFrame, years: List[int], label: str) -> Dict[str, object]:
    """Compute predictive metrics for validation or test years."""
    eval_df = predictions[predictions["year"].isin(years)].copy()
    if eval_df.empty:
        raise ValueError(f"No games found for {label} years: {years}")

    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)
    clipped_pred = pred.clip(EPS, 1.0 - EPS)

    log_loss = -np.mean(y * np.log(clipped_pred) + (1.0 - y) * np.log(1.0 - clipped_pred))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))
    observed_win_rate = y.mean()
    out_of_range_count = int(((pred < 0) | (pred > 1)).sum())

    return {
        f"{label}_log_loss": log_loss,
        f"{label}_brier_score": brier_score,
        f"{label}_accuracy": accuracy,
        f"{label}_baseline_accuracy": max(observed_win_rate, 1.0 - observed_win_rate),
        f"{label}_games": len(eval_df),
        f"{label}_actual_a_win_count_0": int((eval_df["actual_a_win"] == 0).sum()),
        f"{label}_actual_a_win_count_1": int((eval_df["actual_a_win"] == 1).sum()),
        f"{label}_pred_a_win_min": pred.min(),
        f"{label}_pred_a_win_max": pred.max(),
        f"{label}_pred_a_win_out_of_range_count": out_of_range_count,
        f"{label}_fcode_hash": hash_fcodes(eval_df["fcode"].astype(int).tolist()),
    }


def hash_fcodes(fcodes: List[int]) -> str:
    """Create a stable hash for an ordered list of fcodes."""
    text = ",".join(str(int(fcode)) for fcode in fcodes)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def run_parameter_grid(matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all K/scale combinations and choose best predictions by validation log loss."""
    result_rows = []
    best_predictions = None
    best_sort_key = None
    total = len(K_VALUES) * len(SCALE_VALUES)
    counter = 0

    for k in K_VALUES:
        for scale in SCALE_VALUES:
            counter += 1
            print(f"Running parameter {counter}/{total}: K={k}, scale={scale}")
            predictions, final_ratings, run_stats = run_elo_for_parameters(matches, k, scale)
            validation_metrics = compute_metrics(predictions, VALIDATION_YEARS, "validation")
            test_metrics = compute_metrics(predictions, TEST_YEARS, "test")

            row = {
                "K": k,
                "scale": scale,
                "default_rating": DEFAULT_RATING,
                "start_year": START_YEAR,
                "end_year": END_YEAR,
                "validation_years": ",".join(str(year) for year in VALIDATION_YEARS),
                "test_years": ",".join(str(year) for year in TEST_YEARS),
                "average_abs_rating_change_all_years": run_stats["average_abs_rating_change_all_years"],
                "final_rating_mean": run_stats["final_rating_mean"],
                "final_rating_std": run_stats["final_rating_std"],
                "skipped_games": run_stats["skipped_games"],
            }
            row.update(validation_metrics)
            row.update(test_metrics)
            result_rows.append(row)

            # Select only by validation metrics, never by 2025 test metrics.
            sort_key = (
                validation_metrics["validation_log_loss"],
                validation_metrics["validation_brier_score"],
            )
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_predictions = predictions.copy()

    results = pd.DataFrame(result_rows)
    results = results.sort_values(
        ["validation_log_loss", "validation_brier_score", "K", "scale"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    results["rank_by_validation_log_loss"] = np.arange(1, len(results) + 1)
    return results, best_predictions


def select_best_parameters(results: pd.DataFrame) -> pd.Series:
    """Select best K/scale by validation log loss with Brier score tie-breaker."""
    ordered = results.sort_values(
        ["validation_log_loss", "validation_brier_score", "K", "scale"],
        ascending=[True, True, True, True],
    )
    return ordered.iloc[0]


def make_check_summary(results: pd.DataFrame, best_predictions: pd.DataFrame) -> pd.DataFrame:
    """Create consistency checks for validation/test counts and best predictions."""
    validation_games_consistent = results["validation_games"].nunique() == 1
    test_games_consistent = results["test_games"].nunique() == 1
    test_fcode_sets_consistent = results["test_fcode_hash"].nunique() == 1
    validation_fcode_sets_consistent = results["validation_fcode_hash"].nunique() == 1

    validation_has_both_classes = (
        (results["validation_actual_a_win_count_0"] > 0)
        & (results["validation_actual_a_win_count_1"] > 0)
    ).all()
    test_has_both_classes = (
        (results["test_actual_a_win_count_0"] > 0)
        & (results["test_actual_a_win_count_1"] > 0)
    ).all()

    best_pred_out_of_range = int(
        ((best_predictions["pred_a_win"] < 0) | (best_predictions["pred_a_win"] > 1)).sum()
    )

    checks = {
        "number_of_parameter_combinations": len(results),
        "validation_games_per_combination": int(results["validation_games"].iloc[0]),
        "test_games_per_combination": int(results["test_games"].iloc[0]),
        "validation_games_consistent": validation_games_consistent,
        "test_games_consistent": test_games_consistent,
        "validation_fcode_sets_consistent": validation_fcode_sets_consistent,
        "test_fcode_sets_consistent": test_fcode_sets_consistent,
        "validation_actual_a_win_has_0_and_1": validation_has_both_classes,
        "test_actual_a_win_has_0_and_1": test_has_both_classes,
        "best_predictions_pred_a_win_out_of_range_count": best_pred_out_of_range,
        "best_predictions_rows": len(best_predictions),
        "best_predictions_validation_rows": int(best_predictions["is_validation_game"].sum()),
        "best_predictions_test_rows": int(best_predictions["is_test_game"].sum()),
    }
    return pd.DataFrame([checks])


def make_best_test_result(results: pd.DataFrame, best_result: pd.Series) -> pd.DataFrame:
    """Create the one-row final test result table for the chosen parameter setting."""
    default = results[(results["K"] == 20) & (results["scale"] == 500)]
    default_row = default.iloc[0] if not default.empty else None

    row = {
        "best_K": best_result["K"],
        "best_scale": best_result["scale"],
        "selection_metric": "validation_log_loss",
        "validation_log_loss": best_result["validation_log_loss"],
        "validation_brier_score": best_result["validation_brier_score"],
        "validation_accuracy": best_result["validation_accuracy"],
        "test_log_loss": best_result["test_log_loss"],
        "test_brier_score": best_result["test_brier_score"],
        "test_accuracy": best_result["test_accuracy"],
        "test_baseline_accuracy": best_result["test_baseline_accuracy"],
        "test_games": best_result["test_games"],
        "comparison_default_K20_scale500_test_log_loss": np.nan,
        "comparison_default_K20_scale500_test_brier_score": np.nan,
        "comparison_default_K20_scale500_test_accuracy": np.nan,
    }

    if default_row is not None:
        row["comparison_default_K20_scale500_test_log_loss"] = default_row["test_log_loss"]
        row["comparison_default_K20_scale500_test_brier_score"] = default_row["test_brier_score"]
        row["comparison_default_K20_scale500_test_accuracy"] = default_row["test_accuracy"]
    return pd.DataFrame([row])


def write_markdown_summary(results: pd.DataFrame, best_result: pd.Series, best_test: pd.DataFrame) -> str:
    """Write an English summary suitable for meeting notes."""
    default = results[(results["K"] == 20) & (results["scale"] == 500)]
    default_text = "The default K=20, scale=500 setting was not found in the grid."
    if not default.empty:
        default_row = default.iloc[0]
        default_text = (
            f"Default K=20, scale=500 test result: log loss = "
            f"{default_row['test_log_loss']:.6f}, Brier score = "
            f"{default_row['test_brier_score']:.6f}, accuracy = "
            f"{default_row['test_accuracy']:.6f}."
        )

    top_rows = results.head(5)
    top_lines = []
    for _, row in top_rows.iterrows():
        top_lines.append(
            f"* K={row['K']:g}, scale={row['scale']:g}: "
            f"validation log loss = {row['validation_log_loss']:.6f}, "
            f"validation Brier = {row['validation_brier_score']:.6f}, "
            f"test log loss = {row['test_log_loss']:.6f}"
        )

    summary = f"""# Multi-year Elo parameter validation summary

## Aim

The aim is to build a fairer parameter validation framework for the transparent simple Elo baseline.
Instead of choosing parameters directly on the 2025 test set, this script uses 2023-2024 as the validation period and keeps 2025 as a fixed final test period.

## Method

Each K/scale combination starts all players from the same default rating of 1500 and runs from {START_YEAR} to {END_YEAR}.
The burn-in period is 2015-2022, validation years are 2023-2024, and the test year is 2025.
The best parameters are selected using validation log loss, with validation Brier score as the tie-breaker.

## Parameter grid

* K values: {K_VALUES}
* scale values: {SCALE_VALUES}
* Number of parameter combinations: {len(results)}

## Validation results

Top validation settings:

{chr(10).join(top_lines)}

## Best parameter setting

The selected parameter setting is K = {best_result['K']:g}, scale = {best_result['scale']:g}.
It was selected by validation log loss, not by 2025 test performance.

Validation performance:

* log loss = {best_result['validation_log_loss']:.6f}
* Brier score = {best_result['validation_brier_score']:.6f}
* accuracy = {best_result['validation_accuracy']:.6f}

## Test performance on 2025

For the selected parameters, the 2025 test performance is:

* log loss = {best_result['test_log_loss']:.6f}
* Brier score = {best_result['test_brier_score']:.6f}
* accuracy = {best_result['test_accuracy']:.6f}
* baseline accuracy = {best_result['test_baseline_accuracy']:.6f}
* test games = {int(best_result['test_games'])}

## Comparison with default K=20, scale=500

{default_text}

## Interpretation

These are preliminary results for the transparent simple Elo baseline.
The best parameters are selected using the validation years 2023-2024, not the 2025 test set.
The 2025 test set is kept fixed for fair comparison.
If a more aggressive K performs better, this may reflect the need to adapt to changing player strength, but volatility should be checked separately.
The result should not be treated as a final project conclusion before comparing with Glicko and considering rating stability.

## Notes for supervisor

* Is the 2023-2024 validation and 2025 test split appropriate for this project?
* Should I repeat this validation framework with multiple test years?
* Should I combine predictive performance with rating stability before choosing an Elo setting?
* Is this a good point to compare simple Elo with Glicko or official DG-based predictions?
"""
    SUMMARY_MD_PATH.write_text(summary, encoding="utf-8")
    return summary


def save_outputs(
    results: pd.DataFrame,
    best_test: pd.DataFrame,
    best_predictions: pd.DataFrame,
    checks: pd.DataFrame,
    markdown_summary: str,
) -> None:
    """Save all parameter validation outputs."""
    results.to_csv(RESULTS_PATH, index=False)
    best_test.to_csv(BEST_TEST_PATH, index=False)
    best_predictions.to_csv(BEST_PREDICTIONS_PATH, index=False)
    checks.to_csv(CHECK_SUMMARY_PATH, index=False)
    SUMMARY_MD_PATH.write_text(markdown_summary, encoding="utf-8")


def print_summary(results: pd.DataFrame, best_result: pd.Series, best_test: pd.DataFrame, checks: pd.DataFrame) -> None:
    """Print a concise command-line summary."""
    default = results[(results["K"] == 20) & (results["scale"] == 500)]
    default_text = "not available"
    if not default.empty:
        default_row = default.iloc[0]
        default_text = (
            f"log_loss={default_row['test_log_loss']:.6f}, "
            f"brier={default_row['test_brier_score']:.6f}, "
            f"accuracy={default_row['test_accuracy']:.6f}"
        )

    check_row = checks.iloc[0]
    print("\n=== Multi-year Parameter Validation Summary ===")
    print(f"Number of parameter combinations tested: {len(results)}")
    print(
        f"Validation years: {VALIDATION_YEARS}, "
        f"validation games: {int(check_row['validation_games_per_combination'])}"
    )
    print(
        f"Test years: {TEST_YEARS}, "
        f"test games: {int(check_row['test_games_per_combination'])}"
    )
    print(
        f"Best K and scale by validation log loss: "
        f"K={best_result['K']:g}, scale={best_result['scale']:g}"
    )
    print(
        f"Best validation: log_loss={best_result['validation_log_loss']:.6f}, "
        f"brier={best_result['validation_brier_score']:.6f}, "
        f"accuracy={best_result['validation_accuracy']:.6f}"
    )
    print(
        f"Corresponding 2025 test: log_loss={best_result['test_log_loss']:.6f}, "
        f"brier={best_result['test_brier_score']:.6f}, "
        f"accuracy={best_result['test_accuracy']:.6f}"
    )
    print(f"Default K=20, scale=500 test result: {default_text}")
    print(f"Validation game counts consistent: {bool(check_row['validation_games_consistent'])}")
    print(f"Test game counts consistent: {bool(check_row['test_games_consistent'])}")
    print(f"Test fcode sets consistent: {bool(check_row['test_fcode_sets_consistent'])}")
    print("Output paths:")
    print(f"  results: {RESULTS_PATH}")
    print(f"  best test result: {BEST_TEST_PATH}")
    print(f"  best predictions: {BEST_PREDICTIONS_PATH}")
    print(f"  check summary: {CHECK_SUMMARY_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")


def main() -> None:
    matches = load_matches()
    results, best_predictions = run_parameter_grid(matches)
    best_result = select_best_parameters(results)
    checks = make_check_summary(results, best_predictions)
    best_test = make_best_test_result(results, best_result)
    markdown_summary = write_markdown_summary(results, best_result, best_test)
    save_outputs(results, best_test, best_predictions, checks, markdown_summary)
    print_summary(results, best_result, best_test, checks)


if __name__ == "__main__":
    main()
