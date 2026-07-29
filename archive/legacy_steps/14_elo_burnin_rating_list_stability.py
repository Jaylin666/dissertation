"""
This script checks how sensitive the final Elo rating list is to the length
of the historical burn-in period.

The aim is not to keep chasing the lowest log loss. Instead, this script
checks whether different burn-in start years lead to materially different
2025 final rating lists.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import time

import numpy as np
import pandas as pd


START_YEARS = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2018, 2020, 2023, 2025]
RUN_ALL_ANNUAL_START_YEARS = False
FULL_HISTORY_START_YEAR = 1985
END_YEAR = 2025
INITIAL_RATING = 1500.0
EPS = 1e-15

ELO_SETTINGS = [
    {
        "setting_name": "conservative_k10_scale500",
        "k": 10.0,
        "scale": 500.0,
        "label": "Conservative Elo",
    },
    {
        "setting_name": "default_k20_scale500",
        "k": 20.0,
        "scale": 500.0,
        "label": "Default Elo",
    },
    {
        "setting_name": "validation_best_k30_scale300",
        "k": 30.0,
        "scale": 300.0,
        "label": "Validation-best Elo",
    },
]

ACTIVE_PLAYER_SUBSETS = [
    ("all_common_players", 0),
    ("active_2025_games_ge1", 1),
    ("active_2025_games_ge5", 5),
    ("active_2025_games_ge10", 10),
]


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"
MATCHES_PATH = OUTPUT_DIR / "matches_1985_2025_checked.csv"

PREDICTION_METRICS_PATH = OUTPUT_DIR / "elo_burnin_prediction_metrics.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "elo_burnin_final_ratings_all_runs.csv"
UPDATE_HISTORY_PATH = OUTPUT_DIR / "elo_burnin_update_history_all_runs.csv"
VS_REFERENCE_PATH = OUTPUT_DIR / "elo_burnin_vs_1985_reference.csv"
ADJACENT_COMPARISONS_PATH = OUTPUT_DIR / "elo_burnin_adjacent_start_year_comparisons.csv"
ACTIVE_PLAYER_COUNTS_PATH = OUTPUT_DIR / "elo_burnin_active_player_counts.csv"
DATE_ORDERING_SUMMARY_PATH = OUTPUT_DIR / "elo_burnin_date_ordering_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "elo_burnin_rating_list_stability_summary.md"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]


def get_start_years() -> List[int]:
    """Return selected or annual start years."""
    if RUN_ALL_ANNUAL_START_YEARS:
        return list(range(FULL_HISTORY_START_YEAR, END_YEAR + 1))
    return START_YEARS


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load the full-history checked match-level dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/13_build_full_history_match_dataset.py first."
        )

    matches = pd.read_csv(path, low_memory=False)
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

    matches["event_date_parsed"] = pd.to_datetime(matches["event_date_parsed"], errors="coerce")
    matches = add_event_ordering_columns(matches)

    sort_cols = ["year", "event_order_date", "event", "code", "fcode"]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    print(f"Loaded full-history dataset: {path}")
    print(f"Matches: {len(matches)}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    print(f"Players: {players}")
    return matches


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date and event_date_ordering_method without modifying raw dates."""
    matches = matches.copy()
    matches["event_order_date"] = matches["event_date_parsed"]
    matches["event_date_ordering_method"] = np.where(
        matches["event_date_parsed"].notna(),
        "parsed_full_date",
        "fallback_no_date",
    )

    missing_parsed = matches["event_date_parsed"].isna()
    raw = matches.loc[missing_parsed, "event_date_raw"].astype("string").str.strip()
    extracted = raw.str.extract(r"^(?P<month>\d{1,2})\.(?P<year>\d{2}|\d{4})$")

    valid_month_year = extracted["month"].notna()
    if valid_month_year.any():
        months = pd.to_numeric(extracted.loc[valid_month_year, "month"], errors="coerce")
        raw_years = extracted.loc[valid_month_year, "year"].astype(str)
        years = raw_years.astype(int)
        years = np.where(raw_years.str.len().eq(2), np.where(years >= 85, 1900 + years, 2000 + years), years)

        valid_month = months.between(1, 12).fillna(False)
        valid_month_mask = valid_month.to_numpy(dtype=bool)
        valid_index = extracted.loc[valid_month_year].index[valid_month_mask]
        imputed_dates = pd.to_datetime(
            {
                "year": np.asarray(years)[valid_month_mask],
                "month": months.loc[valid_index].astype(int).to_numpy(),
                "day": np.repeat(15, len(valid_index)),
            },
            errors="coerce",
        )

        matches.loc[valid_index, "event_order_date"] = imputed_dates.to_numpy()
        matches.loc[valid_index, "event_date_ordering_method"] = "month_year_imputed"

    return matches


def make_date_ordering_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise how event_order_date was obtained."""
    summary = (
        matches.groupby("event_date_ordering_method", dropna=False)
        .size()
        .reset_index(name="match_count")
        .sort_values("event_date_ordering_method")
    )
    summary["share_of_matches"] = summary["match_count"] / len(matches)
    return summary


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def update_player_name(player_names: Dict[int, str], player_code: int, possible_name: object) -> None:
    """Store the first non-missing name seen for a player."""
    if player_code in player_names:
        return
    if pd.notna(possible_name) and str(possible_name).strip():
        player_names[player_code] = str(possible_name).strip()


def initialise_player(
    player_code: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    initial_rating: float,
) -> None:
    """Initialise player rating and run statistics."""
    if player_code not in ratings:
        ratings[player_code] = initial_rating
        stats[player_code] = {"games": 0, "wins": 0, "losses": 0}


def run_elo_for_period(
    matches: pd.DataFrame,
    start_year: int,
    end_year: int,
    k: float,
    scale: float,
    setting_name: str,
    games_played_2025: Dict[int, int],
    initial_rating: float = INITIAL_RATING,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Run Elo over one burn-in period and return predictions, ratings and update history."""
    run_matches = matches[(matches["year"] >= start_year) & (matches["year"] <= end_year)].copy()

    ratings: Dict[int, float] = {}
    stats: Dict[int, Dict[str, int]] = {}
    player_names: Dict[int, str] = {}
    prediction_rows = []
    history_rows = []
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

        initialise_player(winner, ratings, stats, initial_rating)
        initialise_player(loser, ratings, stats, initial_rating)
        update_player_name(player_names, winner, getattr(row, "winner_name", pd.NA))
        update_player_name(player_names, loser, getattr(row, "loser_name", pd.NA))

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, scale)

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = pred_winner_win if player_a == winner else 1.0 - pred_winner_win
        player_a_rating_before = ratings[player_a]
        player_b_rating_before = ratings[player_b]

        rating_change = k * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change
        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after

        stats[winner]["games"] += 1
        stats[winner]["wins"] += 1
        stats[loser]["games"] += 1
        stats[loser]["losses"] += 1

        player_a_rating_after = ratings[player_a]
        player_b_rating_after = ratings[player_b]
        event_order_date = getattr(row, "event_order_date", pd.NaT)
        event_date_parsed = getattr(row, "event_date_parsed", pd.NaT)

        prediction_rows.append(
            {
                "setting_name": setting_name,
                "k": k,
                "scale": scale,
                "start_year": start_year,
                "end_year": end_year,
                "fcode": int(getattr(row, "fcode")),
                "code": int(getattr(row, "code")),
                "year": year,
                "event": getattr(row, "event", pd.NA),
                "eventname": getattr(row, "eventname", pd.NA),
                "event_date_raw": getattr(row, "event_date_raw", pd.NA),
                "event_date_parsed": event_date_parsed,
                "event_order_date": event_order_date,
                "event_date_ordering_method": getattr(row, "event_date_ordering_method", pd.NA),
                "winner": winner,
                "loser": loser,
                "player_a": player_a,
                "player_b": player_b,
                "actual_a_win": actual_a_win,
                "pred_a_win": pred_a_win,
                "pred_winner_win": pred_winner_win,
                "winner_rating_before": winner_rating_before,
                "loser_rating_before": loser_rating_before,
                "player_a_rating_before": player_a_rating_before,
                "player_b_rating_before": player_b_rating_before,
                "winner_rating_after": winner_rating_after,
                "loser_rating_after": loser_rating_after,
                "player_a_rating_after": player_a_rating_after,
                "player_b_rating_after": player_b_rating_after,
                "rating_change_abs": abs(rating_change),
            }
        )

        history_rows.extend(
            [
                {
                    "setting_name": setting_name,
                    "k": k,
                    "scale": scale,
                    "start_year": start_year,
                    "end_year": end_year,
                    "fcode": int(getattr(row, "fcode")),
                    "year": year,
                    "event": getattr(row, "event", pd.NA),
                    "player_code": winner,
                    "opponent_code": loser,
                    "result": 1,
                    "rating_before": winner_rating_before,
                    "rating_after": winner_rating_after,
                    "rating_change": rating_change,
                    "rating_change_abs": abs(rating_change),
                },
                {
                    "setting_name": setting_name,
                    "k": k,
                    "scale": scale,
                    "start_year": start_year,
                    "end_year": end_year,
                    "fcode": int(getattr(row, "fcode")),
                    "year": year,
                    "event": getattr(row, "event", pd.NA),
                    "player_code": loser,
                    "opponent_code": winner,
                    "result": 0,
                    "rating_before": loser_rating_before,
                    "rating_after": loser_rating_after,
                    "rating_change": -rating_change,
                    "rating_change_abs": abs(rating_change),
                },
            ]
        )

    predictions = pd.DataFrame(prediction_rows)
    update_history = pd.DataFrame(history_rows)
    final_ratings = make_final_ratings_df(
        ratings=ratings,
        stats=stats,
        player_names=player_names,
        setting_name=setting_name,
        k=k,
        scale=scale,
        start_year=start_year,
        end_year=end_year,
        games_played_2025=games_played_2025,
    )
    run_info = {
        "number_of_matches_used": len(predictions),
        "number_of_players": len(final_ratings),
        "skipped_games": skipped_games,
    }
    return predictions, final_ratings, update_history, run_info


def make_final_ratings_df(
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    player_names: Dict[int, str],
    setting_name: str,
    k: float,
    scale: float,
    start_year: int,
    end_year: int,
    games_played_2025: Dict[int, int],
) -> pd.DataFrame:
    """Build final ratings table for one run."""
    rows = []
    for player_code, final_rating in ratings.items():
        player_stats = stats[player_code]
        rows.append(
            {
                "setting_name": setting_name,
                "k": k,
                "scale": scale,
                "start_year": start_year,
                "end_year": end_year,
                "player_code": player_code,
                "player_name": player_names.get(player_code, pd.NA),
                "final_rating": final_rating,
                "games_played_in_run": player_stats["games"],
                "wins_in_run": player_stats["wins"],
                "losses_in_run": player_stats["losses"],
                "games_played_2025": games_played_2025.get(player_code, 0),
            }
        )

    final_ratings = pd.DataFrame(rows)
    if final_ratings.empty:
        return final_ratings

    final_ratings = final_ratings.sort_values(
        ["final_rating", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)
    final_ratings["final_rank"] = np.arange(1, len(final_ratings) + 1)
    return final_ratings


def hash_fcodes(fcodes: Iterable[int]) -> str:
    """Create a stable hash for an ordered fcode list."""
    text = ",".join(str(int(fcode)) for fcode in fcodes)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_prediction_metrics(
    predictions: pd.DataFrame,
    setting_name: str,
    k: float,
    scale: float,
    start_year: int,
    end_year: int,
    number_of_matches_used: int,
    number_of_players: int,
    reference_fcodes: Optional[List[int]],
) -> Tuple[Dict[str, object], List[int]]:
    """Compute fixed 2025 prediction metrics for one run."""
    eval_df = predictions[predictions["year"] == END_YEAR].copy()
    if eval_df.empty:
        raise ValueError(f"No {END_YEAR} evaluation games for {setting_name}, start_year={start_year}")

    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)
    clipped_pred = pred.clip(EPS, 1.0 - EPS)
    log_loss = -np.mean(y * np.log(clipped_pred) + (1.0 - y) * np.log(1.0 - clipped_pred))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))
    observed_win_rate = y.mean()
    fcodes = eval_df["fcode"].astype(int).tolist()
    fcode_hash = hash_fcodes(fcodes)

    row = {
        "setting_name": setting_name,
        "k": k,
        "scale": scale,
        "start_year": start_year,
        "end_year": end_year,
        "number_of_matches_used": number_of_matches_used,
        "number_of_players": number_of_players,
        "evaluation_year": END_YEAR,
        "evaluation_games": len(eval_df),
        "log_loss": log_loss,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "baseline_accuracy": max(observed_win_rate, 1.0 - observed_win_rate),
        "mean_predicted_probability": pred.mean(),
        "observed_win_rate": observed_win_rate,
        "pred_a_win_min": pred.min(),
        "pred_a_win_max": pred.max(),
        "pred_a_win_out_of_range_count": int(((pred < 0) | (pred > 1)).sum()),
        "actual_a_win_count_0": int((eval_df["actual_a_win"] == 0).sum()),
        "actual_a_win_count_1": int((eval_df["actual_a_win"] == 1).sum()),
        "evaluation_fcode_hash": fcode_hash,
        "evaluation_fcode_set_matches_reference": True
        if reference_fcodes is None
        else fcodes == reference_fcodes,
    }
    return row, fcodes


def compute_games_played_2025(matches: pd.DataFrame) -> Dict[int, int]:
    """Count 2025 appearances per player."""
    matches_2025 = matches[matches["year"] == END_YEAR]
    players = pd.concat([matches_2025["winner"], matches_2025["loser"]]).dropna().astype(int)
    return players.value_counts().to_dict()


def make_active_player_counts(games_played_2025: Dict[int, int], total_players: int) -> pd.DataFrame:
    """Summarise active player subset sizes."""
    rows = []
    for subset_name, min_games in ACTIVE_PLAYER_SUBSETS:
        if min_games == 0:
            count = total_players
        else:
            count = sum(1 for games in games_played_2025.values() if games >= min_games)
        rows.append(
            {
                "player_subset": subset_name,
                "min_2025_games": min_games,
                "number_of_players": count,
            }
        )
    return pd.DataFrame(rows)


def get_subset_players(games_played_2025: Dict[int, int], min_games: int) -> Optional[set]:
    """Return active subset players, or None for all common players."""
    if min_games == 0:
        return None
    return {player for player, games in games_played_2025.items() if games >= min_games}


def compare_rating_lists(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    setting_name: str,
    k: float,
    scale: float,
    reference_start_year: int,
    comparison_start_year: int,
    end_year: int,
    player_subset: str,
    min_2025_games: int,
    subset_players: Optional[set],
) -> Dict[str, object]:
    """Compare two final rating lists on common players and optional active subset."""
    ref_cols = ["player_code", "final_rating", "final_rank"]
    cmp_cols = ["player_code", "final_rating", "final_rank"]
    merged = reference[ref_cols].merge(
        comparison[cmp_cols],
        on="player_code",
        how="inner",
        suffixes=("_reference", "_comparison"),
    )

    if subset_players is not None:
        merged = merged[merged["player_code"].isin(subset_players)].copy()

    n_common = len(merged)
    if n_common > 0:
        rating_diff = (merged["final_rating_reference"] - merged["final_rating_comparison"]).abs()
        rank_diff = (merged["final_rank_reference"] - merged["final_rank_comparison"]).abs()
        pearson = merged["final_rating_reference"].corr(merged["final_rating_comparison"]) if n_common >= 2 else np.nan
        spearman = merged["final_rank_reference"].corr(merged["final_rank_comparison"], method="spearman") if n_common >= 2 else np.nan
    else:
        rating_diff = pd.Series(dtype=float)
        rank_diff = pd.Series(dtype=float)
        pearson = np.nan
        spearman = np.nan

    row = {
        "setting_name": setting_name,
        "k": k,
        "scale": scale,
        "reference_start_year": reference_start_year,
        "comparison_start_year": comparison_start_year,
        "end_year": end_year,
        "player_subset": player_subset,
        "min_2025_games": min_2025_games,
        "number_of_common_players": n_common,
        "pearson_rating_correlation": pearson,
        "spearman_rank_correlation": spearman,
        "mean_abs_rating_difference": rating_diff.mean() if n_common else np.nan,
        "median_abs_rating_difference": rating_diff.median() if n_common else np.nan,
        "p90_abs_rating_difference": rating_diff.quantile(0.90) if n_common else np.nan,
        "max_abs_rating_difference": rating_diff.max() if n_common else np.nan,
        "mean_abs_rank_difference": rank_diff.mean() if n_common else np.nan,
        "median_abs_rank_difference": rank_diff.median() if n_common else np.nan,
    }

    common_players = set(merged["player_code"].astype(int))
    subset_reference = reference[reference["player_code"].isin(common_players)]
    subset_comparison = comparison[comparison["player_code"].isin(common_players)]

    for top_n in [10, 25, 50, 100]:
        row[f"top{top_n}_overlap"] = compute_top_overlap(subset_reference, subset_comparison, top_n)

    return row


def compute_top_overlap(reference: pd.DataFrame, comparison: pd.DataFrame, top_n: int) -> float:
    """Compute top-N set overlap between two rating lists."""
    ref_top = set(reference.sort_values(["final_rating", "player_code"], ascending=[False, True]).head(top_n)["player_code"].astype(int))
    cmp_top = set(comparison.sort_values(["final_rating", "player_code"], ascending=[False, True]).head(top_n)["player_code"].astype(int))
    denominator = min(top_n, len(ref_top), len(cmp_top))
    if denominator == 0:
        return np.nan
    return len(ref_top.intersection(cmp_top)) / denominator


def build_stability_comparisons(
    final_ratings_by_run: Dict[Tuple[str, int], pd.DataFrame],
    games_played_2025: Dict[int, int],
    start_years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build vs-1985 and adjacent start-year comparison tables."""
    vs_reference_rows = []
    adjacent_rows = []

    for setting in ELO_SETTINGS:
        setting_name = setting["setting_name"]
        k = setting["k"]
        scale = setting["scale"]
        reference = final_ratings_by_run[(setting_name, FULL_HISTORY_START_YEAR)]

        for comparison_start_year in start_years:
            comparison = final_ratings_by_run[(setting_name, comparison_start_year)]
            for subset_name, min_games in ACTIVE_PLAYER_SUBSETS:
                subset_players = get_subset_players(games_played_2025, min_games)
                vs_reference_rows.append(
                    compare_rating_lists(
                        reference=reference,
                        comparison=comparison,
                        setting_name=setting_name,
                        k=k,
                        scale=scale,
                        reference_start_year=FULL_HISTORY_START_YEAR,
                        comparison_start_year=comparison_start_year,
                        end_year=END_YEAR,
                        player_subset=subset_name,
                        min_2025_games=min_games,
                        subset_players=subset_players,
                    )
                )

        for earlier, later in zip(start_years[:-1], start_years[1:]):
            earlier_ratings = final_ratings_by_run[(setting_name, earlier)]
            later_ratings = final_ratings_by_run[(setting_name, later)]
            adjacent_rows.append(
                compare_rating_lists(
                    reference=earlier_ratings,
                    comparison=later_ratings,
                    setting_name=setting_name,
                    k=k,
                    scale=scale,
                    reference_start_year=earlier,
                    comparison_start_year=later,
                    end_year=END_YEAR,
                    player_subset="all_common_players",
                    min_2025_games=0,
                    subset_players=None,
                )
            )

    return pd.DataFrame(vs_reference_rows), pd.DataFrame(adjacent_rows)


def append_update_history(update_history: pd.DataFrame, first_write: bool) -> None:
    """Append one run's update history to the combined CSV."""
    update_history.to_csv(
        UPDATE_HISTORY_PATH,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )


def write_markdown_summary(
    matches: pd.DataFrame,
    date_summary: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    vs_reference: pd.DataFrame,
    adjacent: pd.DataFrame,
    active_counts: pd.DataFrame,
    all_fcodes_consistent: bool,
    output_path: Path,
) -> str:
    """Write a meeting-ready markdown summary."""
    total_matches = len(matches)
    players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    unique_events = matches[["year", "event"]].drop_duplicates().shape[0]

    date_lines = []
    for _, row in date_summary.iterrows():
        date_lines.append(
            f"* {row['event_date_ordering_method']}: {int(row['match_count'])} matches "
            f"({row['share_of_matches']:.3%})"
        )

    setting_lines = [
        f"* {setting['label']}: `{setting['setting_name']}`, K={setting['k']:g}, scale={setting['scale']:g}"
        for setting in ELO_SETTINGS
    ]

    prediction_lines = []
    for setting_name, group in prediction_metrics.groupby("setting_name", sort=False):
        ordered = group.sort_values("start_year")
        longest = ordered[ordered["start_year"] == FULL_HISTORY_START_YEAR].iloc[0]
        shortest = ordered[ordered["start_year"] == END_YEAR].iloc[0]
        best = ordered.sort_values(["log_loss", "brier_score"]).iloc[0]
        prediction_lines.append(
            f"* {setting_name}: 1985-start log loss {longest['log_loss']:.6f}, "
            f"2025-only log loss {shortest['log_loss']:.6f}; best selected-start log loss "
            f"{best['log_loss']:.6f} at start year {int(best['start_year'])}."
        )

    stability_lines = []
    focus = vs_reference[
        (vs_reference["player_subset"] == "all_common_players")
        & (vs_reference["comparison_start_year"].isin([1995, 2005, 2015, 2020, 2025]))
    ]
    for setting_name, group in focus.groupby("setting_name", sort=False):
        parts = []
        for _, row in group.sort_values("comparison_start_year").iterrows():
            parts.append(
                f"{int(row['comparison_start_year'])}: mean abs rating diff "
                f"{row['mean_abs_rating_difference']:.2f}, top50 {row['top50_overlap']:.3f}"
            )
        stability_lines.append(f"* {setting_name}: " + "; ".join(parts))

    active_lines = []
    active_focus = vs_reference[
        (vs_reference["player_subset"].isin(["active_2025_games_ge1", "active_2025_games_ge5", "active_2025_games_ge10"]))
        & (vs_reference["comparison_start_year"].isin([2015, 2020, 2025]))
    ]
    for setting_name, group in active_focus.groupby("setting_name", sort=False):
        best_parts = []
        for subset_name, subset_group in group.groupby("player_subset", sort=False):
            row = subset_group.sort_values("comparison_start_year").tail(1).iloc[0]
            best_parts.append(
                f"{subset_name} at start {int(row['comparison_start_year'])}: "
                f"mean abs rating diff {row['mean_abs_rating_difference']:.2f}, "
                f"top50 {row['top50_overlap']:.3f}"
            )
        active_lines.append(f"* {setting_name}: " + "; ".join(best_parts))

    k_scale_lines = []
    k_scale_focus = vs_reference[
        (vs_reference["player_subset"] == "active_2025_games_ge5")
        & (vs_reference["comparison_start_year"].isin([2015, 2020, 2025]))
    ]
    for setting_name, group in k_scale_focus.groupby("setting_name", sort=False):
        row = group.sort_values("comparison_start_year").tail(1).iloc[0]
        k_scale_lines.append(
            f"* {setting_name}: 2025-only vs 1985 reference on active_2025_games_ge5 has "
            f"mean abs rating diff {row['mean_abs_rating_difference']:.2f}, "
            f"Spearman {row['spearman_rank_correlation']:.4f}, top50 {row['top50_overlap']:.3f}."
        )

    active_count_lines = [
        f"* {row['player_subset']} (min_2025_games={int(row['min_2025_games'])}): "
        f"{int(row['number_of_players'])} players"
        for _, row in active_counts.iterrows()
    ]

    candidate_text = choose_provisional_burnin_text(vs_reference)

    markdown = f"""# Elo burn-in rating list stability experiment

## 1. Aim of this experiment

This experiment checks how sensitive the final Elo rating list is to the historical burn-in period.
It responds to the supervisor's suggestion to run a long historical period, remove early years, and compare the final rating list.
The purpose is not to optimise log loss alone, but to diagnose whether different start years materially change the 2025 final ratings.

## 2. Data used

Input file: `outputs/elo_optimization/matches_1985_2025_checked.csv`.

* Year range: 1985-2025
* Total matches: {total_matches}
* Unique players: {players}
* Unique events: {unique_events}
* Core merge checks from the full-history dataset passed: duplicated fcode, missing event rows, missing hidx rows, missing winner names and missing loser names were all zero.

## 3. Event date ordering limitation

The full-history dataset contains some early events where `event_date_raw` has only month-year information and no specific day.
This script does not delete those matches. It adds `event_order_date` and `event_date_ordering_method`.
Full dates use the parsed event date. Month-year dates are imputed to the 15th of that month only for ordering.
The original `event_date_raw` and `event_date_parsed` columns are preserved.

{chr(10).join(date_lines)}

## 4. Elo settings tested

{chr(10).join(setting_lines)}

## 5. Start years tested

Selected start years: {', '.join(str(year) for year in get_start_years())}.
All runs end in {END_YEAR}. All players start from rating {INITIAL_RATING:g}.

## 6. Prediction metrics on 2025

All 2025 fcode sets consistent across runs: {all_fcodes_consistent}.

{chr(10).join(prediction_lines)}

## 7. Final rating list stability vs 1985 reference

For each setting, the start-year 1985 run is the full-history reference.
The table `elo_burnin_vs_1985_reference.csv` reports rating correlations, rank correlations, rating differences and top-N overlap.

{chr(10).join(stability_lines)}

## 8. Active player stability

Active player subsets are based on 2025 match appearances:

{chr(10).join(active_count_lines)}

{chr(10).join(active_lines)}

## 9. Does the answer depend on K and scale?

{chr(10).join(k_scale_lines)}

More aggressive settings can produce larger match-by-match updates and may also make final rating lists more sensitive to burn-in length.
The comparison tables should therefore be read by setting, not only in aggregate.

## 10. Provisional conclusion for Elo baseline

This does not prove a theoretically correct burn-in period.
It provides an empirical diagnostic of how much historical data is needed before the 2025 final rating list becomes stable relative to the full-history reference.
{candidate_text}

## 11. Next step

The next Elo optimisation step is the single-year rerun convergence experiment, which is the supervisor's second suggested burn-in diagnostic.
"""
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def choose_provisional_burnin_text(vs_reference: pd.DataFrame) -> str:
    """Create cautious automatic wording for the provisional conclusion."""
    focus = vs_reference[
        (vs_reference["setting_name"] == "validation_best_k30_scale300")
        & (vs_reference["player_subset"] == "active_2025_games_ge5")
        & (vs_reference["comparison_start_year"] != FULL_HISTORY_START_YEAR)
    ].copy()
    if focus.empty:
        return "A defensible burn-in choice should be made after inspecting the stability tables."

    stable = focus[
        (focus["top50_overlap"] >= 0.90)
        & (focus["spearman_rank_correlation"] >= 0.95)
        & (focus["mean_abs_rating_difference"] <= 25.0)
    ]
    if stable.empty:
        return (
            "A defensible burn-in choice appears to favour using the longest available history "
            "as the reference, because the selected stability thresholds were not met by shorter burn-ins."
        )

    candidate = int(stable["comparison_start_year"].max())
    return (
        f"Using the active_2025_games_ge5 subset and conservative empirical thresholds, "
        f"a defensible shorter burn-in candidate appears to be start year {candidate}; "
        "this should still be treated as provisional and checked against Glicko later."
    )


def remove_existing_outputs() -> None:
    """Remove this script's own output files before a fresh run."""
    for path in [
        PREDICTION_METRICS_PATH,
        FINAL_RATINGS_PATH,
        UPDATE_HISTORY_PATH,
        VS_REFERENCE_PATH,
        ADJACENT_COMPARISONS_PATH,
        ACTIVE_PLAYER_COUNTS_PATH,
        DATE_ORDERING_SUMMARY_PATH,
        SUMMARY_MD_PATH,
    ]:
        if path.exists():
            path.unlink()


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs()

    start_years = get_start_years()
    print("=== Elo burn-in rating list stability experiment ===")
    print(f"RUN_ALL_ANNUAL_START_YEARS: {RUN_ALL_ANNUAL_START_YEARS}")
    print(f"Start years: {start_years}")

    matches = load_matches()
    date_summary = make_date_ordering_summary(matches)
    date_summary.to_csv(DATE_ORDERING_SUMMARY_PATH, index=False)
    print("\nEvent date ordering method counts:")
    print(date_summary.to_string(index=False))

    games_played_2025 = compute_games_played_2025(matches)
    total_players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int).nunique()
    active_counts = make_active_player_counts(games_played_2025, total_players)
    active_counts.to_csv(ACTIVE_PLAYER_COUNTS_PATH, index=False)

    all_prediction_metrics = []
    all_final_ratings = []
    final_ratings_by_run: Dict[Tuple[str, int], pd.DataFrame] = {}
    reference_fcodes: Optional[List[int]] = None
    history_first_write = True

    for setting in ELO_SETTINGS:
        setting_name = setting["setting_name"]
        k = setting["k"]
        scale = setting["scale"]
        print(f"\n=== Running setting: {setting_name} (K={k:g}, scale={scale:g}) ===")

        for start_year in start_years:
            run_start = time.time()
            print(f"Running start_year={start_year}, end_year={END_YEAR}")
            predictions, final_ratings, update_history, run_info = run_elo_for_period(
                matches=matches,
                start_year=start_year,
                end_year=END_YEAR,
                k=k,
                scale=scale,
                setting_name=setting_name,
                games_played_2025=games_played_2025,
                initial_rating=INITIAL_RATING,
            )

            metrics_row, fcodes = compute_prediction_metrics(
                predictions=predictions,
                setting_name=setting_name,
                k=k,
                scale=scale,
                start_year=start_year,
                end_year=END_YEAR,
                number_of_matches_used=run_info["number_of_matches_used"],
                number_of_players=run_info["number_of_players"],
                reference_fcodes=reference_fcodes,
            )
            if reference_fcodes is None:
                reference_fcodes = fcodes

            all_prediction_metrics.append(metrics_row)
            all_final_ratings.append(final_ratings)
            final_ratings_by_run[(setting_name, start_year)] = final_ratings
            append_update_history(update_history, first_write=history_first_write)
            history_first_write = False

            print(
                f"  matches used: {run_info['number_of_matches_used']}, "
                f"players: {run_info['number_of_players']}, "
                f"2025 evaluation games: {metrics_row['evaluation_games']}, "
                f"fcode set consistent: {metrics_row['evaluation_fcode_set_matches_reference']}, "
                f"elapsed: {time.time() - run_start:.1f}s"
            )

            del predictions
            del update_history

    prediction_metrics = pd.DataFrame(all_prediction_metrics)
    final_ratings_all = pd.concat(all_final_ratings, ignore_index=True)

    all_fcodes_consistent = bool(prediction_metrics["evaluation_fcode_set_matches_reference"].all())
    if not all_fcodes_consistent:
        print("WARNING: at least one 2025 fcode set differs from the reference run.")
    else:
        print("\nAll 2025 fcode sets are consistent across runs.")

    print("\nBuilding final rating list stability comparisons...")
    vs_reference, adjacent = build_stability_comparisons(
        final_ratings_by_run=final_ratings_by_run,
        games_played_2025=games_played_2025,
        start_years=start_years,
    )

    prediction_metrics.to_csv(PREDICTION_METRICS_PATH, index=False)
    final_ratings_all.to_csv(FINAL_RATINGS_PATH, index=False)
    vs_reference.to_csv(VS_REFERENCE_PATH, index=False)
    adjacent.to_csv(ADJACENT_COMPARISONS_PATH, index=False)

    write_markdown_summary(
        matches=matches,
        date_summary=date_summary,
        prediction_metrics=prediction_metrics,
        vs_reference=vs_reference,
        adjacent=adjacent,
        active_counts=active_counts,
        all_fcodes_consistent=all_fcodes_consistent,
        output_path=SUMMARY_MD_PATH,
    )

    print("\nOutput paths:")
    print(f"  prediction metrics: {PREDICTION_METRICS_PATH}")
    print(f"  final ratings: {FINAL_RATINGS_PATH}")
    print(f"  update history: {UPDATE_HISTORY_PATH}")
    print(f"  vs 1985 reference: {VS_REFERENCE_PATH}")
    print(f"  adjacent comparisons: {ADJACENT_COMPARISONS_PATH}")
    print(f"  active player counts: {ACTIVE_PLAYER_COUNTS_PATH}")
    print(f"  date ordering summary: {DATE_ORDERING_SUMMARY_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
