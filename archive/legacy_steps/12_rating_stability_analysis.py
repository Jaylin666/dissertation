from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


START_YEAR = 2015
END_YEAR = 2025
TEST_YEAR = 2025
DEFAULT_RATING = 1500.0

PARAMETER_SETTINGS = [
    {
        "setting_name": "default_elo",
        "K": 20.0,
        "scale": 500.0,
        "reason": "Original default baseline from previous steps",
    },
    {
        "setting_name": "selected_validation_best",
        "K": 30.0,
        "scale": 300.0,
        "reason": "Selected by 2023-2024 validation log loss",
    },
    {
        "setting_name": "nearby_test_strong",
        "K": 35.0,
        "scale": 300.0,
        "reason": "Had slightly lower 2025 test log loss but was not selected by validation",
    },
    {
        "setting_name": "validation_tie_candidate",
        "K": 40.0,
        "scale": 400.0,
        "reason": "Tied or nearly tied validation performance; useful to compare stability",
    },
    {
        "setting_name": "conservative_update",
        "K": 10.0,
        "scale": 500.0,
        "reason": "Lower K to represent smoother / less volatile ratings",
    },
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

YEAR_RANGE = f"{START_YEAR}_{END_YEAR}"
MATCHES_PATH = DATA_PROCESSED / f"matches_{YEAR_RANGE}_checked.csv"
PARAMETER_RESULTS_PATH = DATA_PROCESSED / f"parameter_validation_results_{YEAR_RANGE}.csv"
BEST_PARAMETER_PATH = DATA_PROCESSED / f"best_parameter_test_result_{YEAR_RANGE}.csv"

STABILITY_RESULTS_PATH = DATA_PROCESSED / f"rating_stability_results_{YEAR_RANGE}.csv"
YEARLY_SUMMARY_PATH = DATA_PROCESSED / f"rating_stability_yearly_summary_{YEAR_RANGE}.csv"
TOP_OVERLAP_PATH = DATA_PROCESSED / f"rating_stability_top_rank_overlap_{YEAR_RANGE}.csv"
PLAYER_LEVEL_PATH = DATA_PROCESSED / f"rating_stability_player_level_{YEAR_RANGE}.csv"
SUMMARY_MD_PATH = DATA_PROCESSED / f"rating_stability_summary_{YEAR_RANGE}.md"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["event_date_parsed", "winner_name", "loser_name", "eventname"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load checked multi-year matches and sort them in chronological order."""
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

    sort_cols = [
        col
        for col in ["year", "event_date_parsed", "event", "code", "fcode"]
        if col in matches.columns
    ]
    return matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def load_parameter_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictive parameter validation results if available."""
    if PARAMETER_RESULTS_PATH.exists():
        parameter_results = pd.read_csv(PARAMETER_RESULTS_PATH)
        print(f"Loaded parameter validation results: {PARAMETER_RESULTS_PATH}")
    else:
        print(f"WARNING: parameter validation results not found: {PARAMETER_RESULTS_PATH}")
        parameter_results = pd.DataFrame()

    if BEST_PARAMETER_PATH.exists():
        best_result = pd.read_csv(BEST_PARAMETER_PATH)
        print(f"Loaded best parameter result: {BEST_PARAMETER_PATH}")
    else:
        print(f"WARNING: best parameter result not found: {BEST_PARAMETER_PATH}")
        best_result = pd.DataFrame()

    return parameter_results, best_result


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def initialise_player(
    player_code: int,
    year: int,
    ratings: Dict[int, float],
    player_stats: Dict[int, Dict[str, object]],
) -> None:
    """Add a new player to rating and player-level tracking dictionaries."""
    if player_code not in ratings:
        ratings[player_code] = DEFAULT_RATING
        player_stats[player_code] = {
            "n_games": 0,
            "first_year_seen": year,
            "last_year_seen": year,
            "rating_min": DEFAULT_RATING,
            "rating_max": DEFAULT_RATING,
            "total_abs_update": 0.0,
            "n_updates": 0,
            "player_name": pd.NA,
        }


def update_player_name(player_stats: Dict[int, Dict[str, object]], player_code: int, possible_name) -> None:
    """Store the first non-missing player name seen for this player."""
    if pd.isna(player_stats[player_code].get("player_name")):
        if pd.notna(possible_name) and str(possible_name).strip():
            player_stats[player_code]["player_name"] = str(possible_name).strip()


def record_player_update(
    player_stats: Dict[int, Dict[str, object]],
    player_code: int,
    year: int,
    rating_after: float,
    abs_update: float,
) -> None:
    """Update player-level stability information after one match."""
    stats = player_stats[player_code]
    stats["n_games"] += 1
    stats["last_year_seen"] = year
    stats["rating_min"] = min(float(stats["rating_min"]), rating_after)
    stats["rating_max"] = max(float(stats["rating_max"]), rating_after)
    stats["total_abs_update"] = float(stats["total_abs_update"]) + abs_update
    stats["n_updates"] = int(stats["n_updates"]) + 1


def snapshot_year_end_ratings(
    setting_name: str,
    k: float,
    scale: float,
    year: int,
    ratings: Dict[int, float],
    player_stats: Dict[int, Dict[str, object]],
    active_players: set,
) -> List[Dict[str, object]]:
    """Record one year-end rating snapshot for all players seen so far."""
    rows = []
    for player_code, rating in ratings.items():
        rows.append(
            {
                "setting_name": setting_name,
                "K": k,
                "scale": scale,
                "year": year,
                "player_code": player_code,
                "player_name": player_stats[player_code].get("player_name", pd.NA),
                "rating": rating,
                "active_in_year": player_code in active_players,
            }
        )
    return rows


def run_elo_with_history(
    matches: pd.DataFrame, k: float, scale: float, setting_name: str
) -> Dict[str, pd.DataFrame]:
    """Run Elo and record predictions, rating history, year-end ratings and player summaries."""
    run_matches = matches[(matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)].copy()

    ratings: Dict[int, float] = {}
    player_stats: Dict[int, Dict[str, object]] = {}
    predictions = []
    rating_history = []
    year_end_rows = []
    active_players_by_year: Dict[int, set] = {}
    skipped_games = 0
    current_year = None

    for row in run_matches.itertuples(index=False):
        winner_value = getattr(row, "winner")
        loser_value = getattr(row, "loser")
        if pd.isna(winner_value) or pd.isna(loser_value):
            skipped_games += 1
            continue

        year = int(getattr(row, "year"))
        if current_year is None:
            current_year = year
        elif year != current_year:
            year_end_rows.extend(
                snapshot_year_end_ratings(
                    setting_name,
                    k,
                    scale,
                    current_year,
                    ratings,
                    player_stats,
                    active_players_by_year.get(current_year, set()),
                )
            )
            current_year = year

        winner = int(winner_value)
        loser = int(loser_value)
        initialise_player(winner, year, ratings, player_stats)
        initialise_player(loser, year, ratings, player_stats)
        update_player_name(player_stats, winner, getattr(row, "winner_name", pd.NA))
        update_player_name(player_stats, loser, getattr(row, "loser_name", pd.NA))
        active_players_by_year.setdefault(year, set()).update([winner, loser])

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, scale)

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = pred_winner_win if player_a == winner else 1.0 - pred_winner_win

        rating_change = k * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change
        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after

        record_player_update(player_stats, winner, year, winner_rating_after, abs(rating_change))
        record_player_update(player_stats, loser, year, loser_rating_after, abs(rating_change))

        predictions.append(
            {
                "setting_name": setting_name,
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
                "rating_change": rating_change,
                "K": k,
                "scale": scale,
            }
        )

        rating_history.extend(
            [
                {
                    "setting_name": setting_name,
                    "fcode": int(getattr(row, "fcode")),
                    "year": year,
                    "player_code": winner,
                    "opponent_code": loser,
                    "result": 1,
                    "rating_before": winner_rating_before,
                    "rating_after": winner_rating_after,
                    "rating_change_abs": abs(rating_change),
                },
                {
                    "setting_name": setting_name,
                    "fcode": int(getattr(row, "fcode")),
                    "year": year,
                    "player_code": loser,
                    "opponent_code": winner,
                    "result": 0,
                    "rating_before": loser_rating_before,
                    "rating_after": loser_rating_after,
                    "rating_change_abs": abs(rating_change),
                },
            ]
        )

    if current_year is not None:
        year_end_rows.extend(
            snapshot_year_end_ratings(
                setting_name,
                k,
                scale,
                current_year,
                ratings,
                player_stats,
                active_players_by_year.get(current_year, set()),
            )
        )

    final_ratings = make_final_ratings_df(setting_name, k, scale, ratings, player_stats)

    return {
        "predictions": pd.DataFrame(predictions),
        "rating_history": pd.DataFrame(rating_history),
        "year_end_ratings": pd.DataFrame(year_end_rows),
        "player_level": final_ratings,
        "skipped_games": skipped_games,
    }


def make_final_ratings_df(
    setting_name: str,
    k: float,
    scale: float,
    ratings: Dict[int, float],
    player_stats: Dict[int, Dict[str, object]],
) -> pd.DataFrame:
    """Create player-level stability output for one setting."""
    rows = []
    for player_code, rating in ratings.items():
        stats = player_stats[player_code]
        n_updates = int(stats["n_updates"])
        rows.append(
            {
                "setting_name": setting_name,
                "K": k,
                "scale": scale,
                "player_code": player_code,
                "player_name": stats.get("player_name", pd.NA),
                "n_games": int(stats["n_games"]),
                "first_year_seen": int(stats["first_year_seen"]),
                "last_year_seen": int(stats["last_year_seen"]),
                "final_rating": rating,
                "rating_min": float(stats["rating_min"]),
                "rating_max": float(stats["rating_max"]),
                "rating_range": float(stats["rating_max"]) - float(stats["rating_min"]),
                "average_abs_update": float(stats["total_abs_update"]) / n_updates
                if n_updates > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_update_volatility(rating_history: pd.DataFrame) -> Dict[str, float]:
    """Compute match-level absolute update volatility metrics."""
    updates = rating_history["rating_change_abs"].astype(float)
    return {
        "average_abs_rating_update": updates.mean(),
        "median_abs_rating_update": updates.median(),
        "p90_abs_rating_update": updates.quantile(0.90),
        "p95_abs_rating_update": updates.quantile(0.95),
        "max_abs_rating_update": updates.max(),
    }


def compute_yearly_rating_summary(year_end_ratings: pd.DataFrame) -> pd.DataFrame:
    """Summarise year-end rating distribution by setting and year."""
    rows = []
    grouped = year_end_ratings.groupby(["setting_name", "K", "scale", "year"], sort=True)
    for (setting_name, k, scale, year), group in grouped:
        rows.append(
            {
                "setting_name": setting_name,
                "K": k,
                "scale": scale,
                "year": year,
                "active_players_in_year": int(group["active_in_year"].sum()),
                "players_seen_so_far": int(group["player_code"].nunique()),
                "year_end_rating_mean": group["rating"].mean(),
                "year_end_rating_std": group["rating"].std(),
                "top_rating": group["rating"].max(),
                "bottom_rating": group["rating"].min(),
            }
        )
    return pd.DataFrame(rows)


def compute_year_to_year_changes(year_end_ratings: pd.DataFrame) -> pd.DataFrame:
    """Calculate absolute rating changes between consecutive year-end snapshots."""
    rows = []
    sort_cols = ["setting_name", "player_code", "year"]
    ordered = year_end_ratings.sort_values(sort_cols)
    for (setting_name, player_code), group in ordered.groupby(["setting_name", "player_code"]):
        group = group.sort_values("year")
        previous_year = None
        previous_rating = None
        for row in group.itertuples(index=False):
            if previous_year is not None and int(row.year) == previous_year + 1:
                rows.append(
                    {
                        "setting_name": setting_name,
                        "player_code": player_code,
                        "year_previous": previous_year,
                        "year_current": int(row.year),
                        "annual_abs_rating_change": abs(float(row.rating) - previous_rating),
                    }
                )
            previous_year = int(row.year)
            previous_rating = float(row.rating)
    return pd.DataFrame(rows)


def compute_top_rank_overlap(year_end_ratings: pd.DataFrame) -> pd.DataFrame:
    """Compute adjacent-year top ranking overlap and 2025 overlap vs selected setting."""
    rows = []
    for setting_name, setting_df in year_end_ratings.groupby("setting_name"):
        k = setting_df["K"].iloc[0]
        scale = setting_df["scale"].iloc[0]
        years = sorted(setting_df["year"].unique())
        for previous_year, current_year in zip(years[:-1], years[1:]):
            prev = setting_df[setting_df["year"] == previous_year].sort_values("rating", ascending=False)
            curr = setting_df[setting_df["year"] == current_year].sort_values("rating", ascending=False)
            for top_n in [50, 100]:
                prev_top = set(prev.head(top_n)["player_code"])
                curr_top = set(curr.head(top_n)["player_code"])
                denominator = min(top_n, len(prev_top), len(curr_top))
                overlap_count = len(prev_top.intersection(curr_top))
                rows.append(
                    {
                        "setting_name": setting_name,
                        "K": k,
                        "scale": scale,
                        "year_previous": previous_year,
                        "year_current": current_year,
                        "top_n": top_n,
                        "overlap_count": overlap_count,
                        "overlap_rate": overlap_count / denominator if denominator else np.nan,
                        "comparison_type": "adjacent_year",
                    }
                )

    reference = year_end_ratings[
        (year_end_ratings["setting_name"] == "selected_validation_best")
        & (year_end_ratings["year"] == TEST_YEAR)
    ].sort_values("rating", ascending=False)
    if not reference.empty:
        for setting_name, setting_df in year_end_ratings[year_end_ratings["year"] == TEST_YEAR].groupby("setting_name"):
            k = setting_df["K"].iloc[0]
            scale = setting_df["scale"].iloc[0]
            setting_sorted = setting_df.sort_values("rating", ascending=False)
            for top_n in [50, 100]:
                reference_top = set(reference.head(top_n)["player_code"])
                setting_top = set(setting_sorted.head(top_n)["player_code"])
                denominator = min(top_n, len(reference_top), len(setting_top))
                overlap_count = len(reference_top.intersection(setting_top))
                rows.append(
                    {
                        "setting_name": setting_name,
                        "K": k,
                        "scale": scale,
                        "year_previous": TEST_YEAR,
                        "year_current": TEST_YEAR,
                        "top_n": top_n,
                        "overlap_count": overlap_count,
                        "overlap_rate": overlap_count / denominator if denominator else np.nan,
                        "comparison_type": "vs_selected_validation_best_2025",
                    }
                )
    return pd.DataFrame(rows)


def summarise_annual_changes(annual_changes: pd.DataFrame, setting_name: str) -> Dict[str, float]:
    """Summarise year-to-year absolute rating changes for one setting."""
    subset = annual_changes[annual_changes["setting_name"] == setting_name]
    if subset.empty:
        return {
            "average_annual_abs_rating_change": np.nan,
            "median_annual_abs_rating_change": np.nan,
            "p95_annual_abs_rating_change": np.nan,
        }
    values = subset["annual_abs_rating_change"].astype(float)
    return {
        "average_annual_abs_rating_change": values.mean(),
        "median_annual_abs_rating_change": values.median(),
        "p95_annual_abs_rating_change": values.quantile(0.95),
    }


def combine_with_predictive_results(stability_results: pd.DataFrame, parameter_results: pd.DataFrame) -> pd.DataFrame:
    """Merge stability metrics with validation/test metrics from step 11 if available."""
    if parameter_results.empty:
        for col in [
            "validation_log_loss",
            "validation_brier_score",
            "validation_accuracy",
            "test_log_loss",
            "test_brier_score",
            "test_accuracy",
        ]:
            stability_results[col] = np.nan
        return stability_results

    metric_cols = [
        "K",
        "scale",
        "validation_log_loss",
        "validation_brier_score",
        "validation_accuracy",
        "test_log_loss",
        "test_brier_score",
        "test_accuracy",
    ]
    available_cols = [col for col in metric_cols if col in parameter_results.columns]
    merged = stability_results.merge(
        parameter_results[available_cols],
        on=["K", "scale"],
        how="left",
    )
    return merged


def build_stability_results(
    setting_outputs: Dict[str, Dict[str, pd.DataFrame]],
    yearly_summary: pd.DataFrame,
    top_overlap: pd.DataFrame,
    annual_changes: pd.DataFrame,
    parameter_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create the one-row-per-setting stability results table."""
    rows = []
    for setting in PARAMETER_SETTINGS:
        setting_name = setting["setting_name"]
        output = setting_outputs[setting_name]
        predictions = output["predictions"]
        rating_history = output["rating_history"]
        player_level = output["player_level"]
        volatility = compute_update_volatility(rating_history)
        annual = summarise_annual_changes(annual_changes, setting_name)

        final_mean = player_level["final_rating"].mean()
        final_std = player_level["final_rating"].std()
        final_min = player_level["final_rating"].min()
        final_max = player_level["final_rating"].max()

        overlap_2024_2025 = top_overlap[
            (top_overlap["setting_name"] == setting_name)
            & (top_overlap["year_previous"] == 2024)
            & (top_overlap["year_current"] == 2025)
            & (top_overlap["comparison_type"] == "adjacent_year")
        ]
        top50 = overlap_2024_2025[overlap_2024_2025["top_n"] == 50]["overlap_rate"]
        top100 = overlap_2024_2025[overlap_2024_2025["top_n"] == 100]["overlap_rate"]

        row = {
            "setting_name": setting_name,
            "K": setting["K"],
            "scale": setting["scale"],
            "reason": setting["reason"],
            "number_of_games": len(predictions),
            "number_of_players": player_level["player_code"].nunique(),
            "final_rating_mean": final_mean,
            "final_rating_std": final_std,
            "final_rating_min": final_min,
            "final_rating_max": final_max,
            "top50_overlap_2024_2025": float(top50.iloc[0]) if not top50.empty else np.nan,
            "top100_overlap_2024_2025": float(top100.iloc[0]) if not top100.empty else np.nan,
        }
        row.update(volatility)
        row.update(annual)
        rows.append(row)

    stability_results = pd.DataFrame(rows)
    stability_results = combine_with_predictive_results(stability_results, parameter_results)

    column_order = [
        "setting_name",
        "K",
        "scale",
        "reason",
        "number_of_games",
        "number_of_players",
        "average_abs_rating_update",
        "median_abs_rating_update",
        "p90_abs_rating_update",
        "p95_abs_rating_update",
        "max_abs_rating_update",
        "final_rating_mean",
        "final_rating_std",
        "final_rating_min",
        "final_rating_max",
        "average_annual_abs_rating_change",
        "median_annual_abs_rating_change",
        "p95_annual_abs_rating_change",
        "top50_overlap_2024_2025",
        "top100_overlap_2024_2025",
        "validation_log_loss",
        "validation_brier_score",
        "validation_accuracy",
        "test_log_loss",
        "test_brier_score",
        "test_accuracy",
    ]
    remaining = [col for col in stability_results.columns if col not in column_order]
    return stability_results[column_order + remaining]


def make_sanity_checks(setting_outputs: Dict[str, Dict[str, pd.DataFrame]], stability_results: pd.DataFrame) -> Dict[str, object]:
    """Run basic validation checks and print warnings if needed."""
    games_counts = {}
    fcode_sets = {}
    all_pred_in_range = True
    all_actual_has_both = True
    all_history_lengths_ok = True

    for setting_name, output in setting_outputs.items():
        predictions = output["predictions"]
        history = output["rating_history"]
        games_counts[setting_name] = len(predictions)
        test_fcodes = predictions[predictions["year"] == TEST_YEAR]["fcode"].astype(int).tolist()
        fcode_sets[setting_name] = tuple(test_fcodes)

        out_of_range = ((predictions["pred_a_win"] < 0) | (predictions["pred_a_win"] > 1)).sum()
        if out_of_range > 0:
            all_pred_in_range = False
            print(f"WARNING: {setting_name} has {out_of_range} pred_a_win values outside [0, 1].")

        actual_values = set(predictions["actual_a_win"].dropna().astype(int).unique().tolist())
        if actual_values != {0, 1}:
            all_actual_has_both = False
            print(f"WARNING: {setting_name} actual_a_win values are {sorted(actual_values)}.")

        if len(history) != len(predictions) * 2:
            all_history_lengths_ok = False
            print(
                f"WARNING: {setting_name} history rows {len(history)} != "
                f"2 * predictions rows {len(predictions)}."
            )

    processed_games_consistent = len(set(games_counts.values())) == 1
    test_fcode_sets_consistent = len(set(fcode_sets.values())) == 1
    final_rating_mean_close = bool(
        (stability_results["final_rating_mean"] - DEFAULT_RATING).abs().lt(1e-6).all()
    )
    if not final_rating_mean_close:
        print(
            "WARNING: at least one final_rating_mean is not close to 1500. "
            "This may indicate skipped games or non-zero-sum updates."
        )

    return {
        "processed_games_consistent": processed_games_consistent,
        "test_fcode_sets_consistent": test_fcode_sets_consistent,
        "all_pred_a_win_in_range": all_pred_in_range,
        "all_actual_a_win_has_0_and_1": all_actual_has_both,
        "all_history_rows_equal_predictions_times_two": all_history_lengths_ok,
        "final_rating_mean_close_to_1500": final_rating_mean_close,
    }


def write_markdown_summary(
    stability_results: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    top_overlap: pd.DataFrame,
    checks: Dict[str, object],
) -> str:
    """Write an English markdown summary for meeting notes."""
    selected = stability_results[
        stability_results["setting_name"] == "selected_validation_best"
    ].iloc[0]
    default = stability_results[stability_results["setting_name"] == "default_elo"].iloc[0]

    rows = []
    for _, row in stability_results.iterrows():
        test_text = "not available"
        if pd.notna(row.get("test_log_loss", np.nan)):
            test_text = (
                f"test log loss = {row['test_log_loss']:.6f}, "
                f"test accuracy = {row['test_accuracy']:.6f}"
            )
        rows.append(
            f"* {row['setting_name']} (K={row['K']:g}, scale={row['scale']:g}): "
            f"average abs update = {row['average_abs_rating_update']:.3f}, "
            f"final rating std = {row['final_rating_std']:.3f}, "
            f"top50 2024-2025 overlap = {row['top50_overlap_2024_2025']:.3f}, "
            f"{test_text}"
        )

    markdown = f"""# Rating stability / volatility analysis ({START_YEAR}-{END_YEAR})

## Aim

The aim is to compare rating stability and volatility across several transparent simple Elo parameter settings.
This step is not trying to find a lower log loss. Instead, it checks the trade-off between predictive performance and rating smoothness.

## Method

Each setting starts all players from 1500 and runs the same simple Elo update rule over the 2015-2025 match dataset.
For every setting, the script records match-level rating updates, year-end rating distributions, top-ranking overlap, and player-level rating ranges.

## Parameter settings compared

{chr(10).join(rows)}

## Main stability results

The selected validation-best setting is K={selected['K']:g}, scale={selected['scale']:g}.
Compared with the default setting, its average absolute rating update is {selected['average_abs_rating_update']:.3f} versus {default['average_abs_rating_update']:.3f}, and its final rating standard deviation is {selected['final_rating_std']:.3f} versus {default['final_rating_std']:.3f}.

## Prediction vs stability comparison

The selected validation-best setting improves the 2025 test predictive metrics relative to the original default setting, but it also uses a larger K and smaller scale. This can make ratings more responsive, and potentially more volatile.
The stability table should therefore be read together with the validation/test log loss, Brier score and accuracy from the parameter validation step.

## Interpretation

These are preliminary stability checks for the transparent simple Elo baseline. The selected validation-best setting improves predictive performance on the 2025 test set, but a larger K may also make ratings more responsive or more volatile. Therefore, predictive performance should be considered together with stability before treating any Elo parameter setting as final.

The result should not be treated as a final project conclusion before comparing with Glicko and considering rating stability in a more formal way.

## Sanity checks

* Processed games consistent across settings: {checks['processed_games_consistent']}
* 2025 test fcode sets consistent across settings: {checks['test_fcode_sets_consistent']}
* pred_a_win always in [0, 1]: {checks['all_pred_a_win_in_range']}
* actual_a_win contains both 0 and 1: {checks['all_actual_a_win_has_0_and_1']}
* rating history rows equal two times prediction rows: {checks['all_history_rows_equal_predictions_times_two']}
* final rating mean close to 1500: {checks['final_rating_mean_close_to_1500']}

## Notes for supervisor

* Is this level of rating volatility acceptable for a sports rating system?
* Should predictive performance and stability be combined into one model selection criterion?
* Should I compare these Elo stability results with Glicko rating deviation later?
* Should the final project report include both predictive scoring rules and rating stability diagnostics?
"""
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def save_outputs(
    stability_results: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    top_overlap: pd.DataFrame,
    player_level: pd.DataFrame,
    markdown_summary: str,
) -> None:
    """Save all stability analysis outputs."""
    stability_results.to_csv(STABILITY_RESULTS_PATH, index=False)
    yearly_summary.to_csv(YEARLY_SUMMARY_PATH, index=False)
    top_overlap.to_csv(TOP_OVERLAP_PATH, index=False)
    player_level.to_csv(PLAYER_LEVEL_PATH, index=False)
    SUMMARY_MD_PATH.write_text(markdown_summary, encoding="utf-8")


def print_command_line_summary(stability_results: pd.DataFrame, checks: Dict[str, object]) -> None:
    """Print key results after the script finishes."""
    selected = stability_results[stability_results["setting_name"] == "selected_validation_best"]
    best_test = stability_results.sort_values("test_log_loss").head(1)

    print("\n=== Rating Stability Analysis Summary ===")
    print(f"Number of parameter settings compared: {len(stability_results)}")
    print("Number of games processed for each setting:")
    print(stability_results[["setting_name", "number_of_games"]].to_string(index=False))
    print("Number of players:")
    print(stability_results[["setting_name", "number_of_players"]].to_string(index=False))
    if not selected.empty:
        row = selected.iloc[0]
        print(
            f"Selected validation-best setting: {row['setting_name']} "
            f"(K={row['K']:g}, scale={row['scale']:g})"
        )
    if not best_test.empty and pd.notna(best_test.iloc[0].get("test_log_loss", np.nan)):
        row = best_test.iloc[0]
        print(
            f"Best test log loss setting if available, not selected by test: "
            f"{row['setting_name']} (test_log_loss={row['test_log_loss']:.6f})"
        )
    print("average_abs_rating_update for each setting:")
    print(stability_results[["setting_name", "average_abs_rating_update"]].to_string(index=False))
    print("final_rating_std for each setting:")
    print(stability_results[["setting_name", "final_rating_std"]].to_string(index=False))
    print("Sanity checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print("Output paths:")
    print(f"  stability results: {STABILITY_RESULTS_PATH}")
    print(f"  yearly summary: {YEARLY_SUMMARY_PATH}")
    print(f"  top rank overlap: {TOP_OVERLAP_PATH}")
    print(f"  player level: {PLAYER_LEVEL_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")


def main() -> None:
    matches = load_matches()
    parameter_results, _ = load_parameter_results()
    setting_outputs = {}
    all_year_end = []
    all_player_level = []

    for setting in PARAMETER_SETTINGS:
        print(
            f"\n=== Running stability setting: {setting['setting_name']} "
            f"(K={setting['K']:g}, scale={setting['scale']:g}) ==="
        )
        output = run_elo_with_history(
            matches=matches,
            k=setting["K"],
            scale=setting["scale"],
            setting_name=setting["setting_name"],
        )
        setting_outputs[setting["setting_name"]] = output
        all_year_end.append(output["year_end_ratings"])
        all_player_level.append(output["player_level"])

    year_end_ratings = pd.concat(all_year_end, ignore_index=True)
    player_level = pd.concat(all_player_level, ignore_index=True)
    yearly_summary = compute_yearly_rating_summary(year_end_ratings)
    annual_changes = compute_year_to_year_changes(year_end_ratings)
    top_overlap = compute_top_rank_overlap(year_end_ratings)
    stability_results = build_stability_results(
        setting_outputs=setting_outputs,
        yearly_summary=yearly_summary,
        top_overlap=top_overlap,
        annual_changes=annual_changes,
        parameter_results=parameter_results,
    )
    checks = make_sanity_checks(setting_outputs, stability_results)
    markdown_summary = write_markdown_summary(
        stability_results=stability_results,
        yearly_summary=yearly_summary,
        top_overlap=top_overlap,
        checks=checks,
    )
    save_outputs(stability_results, yearly_summary, top_overlap, player_level, markdown_summary)
    print_command_line_summary(stability_results, checks)


if __name__ == "__main__":
    main()
