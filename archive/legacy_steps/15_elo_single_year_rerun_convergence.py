"""
This script implements a single-year repeated rerun diagnostic: run one year
of matches, use the final ratings as the initial ratings for the next
iteration, and repeat until the rating list stabilises.

Important limitation: this is not a fair predictive evaluation. The same
year's results are reused across iterations, so log loss and Brier score
from these repeated runs would not represent out-of-sample performance.
This script is only a rating convergence / stability diagnostic.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
import pandas as pd


YEARS_TO_TEST = [2025, 2024, 2023]
MAX_ITERATIONS = 50
MEAN_ABS_CHANGE_THRESHOLD = 0.1
MAX_ABS_CHANGE_THRESHOLD = 1.0
SPEARMAN_THRESHOLD = 0.999
DEFAULT_RATING = 1500.0

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


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"
MATCHES_PATH = OUTPUT_DIR / "matches_1985_2025_checked.csv"

ITERATION_SUMMARY_PATH = OUTPUT_DIR / "elo_single_year_rerun_iteration_summary.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "elo_single_year_rerun_final_ratings.csv"
UPDATE_SUMMARY_PATH = OUTPUT_DIR / "elo_single_year_rerun_update_summary.csv"
DECISIONS_PATH = OUTPUT_DIR / "elo_single_year_rerun_convergence_decisions.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "elo_single_year_rerun_convergence_summary.md"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load full-history matches and keep only the requested single-year tests."""
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
    matches = matches[matches["year"].isin(YEARS_TO_TEST)].copy()

    sort_cols = ["year", "event_order_date", "event", "code", "fcode"]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    print(f"Loaded dataset: {path}")
    print(f"Testing years: {YEARS_TO_TEST}")
    print("Matches by year:")
    print(matches.groupby("year").size().sort_index().to_string())
    return matches


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date and event_date_ordering_method without changing raw date columns."""
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


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def initialise_player(
    player_code: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    player_names: Dict[int, str],
    initial_ratings: Optional[Dict[int, float]],
    default_rating: float,
    possible_name: object,
) -> None:
    """Initialise one player if needed."""
    if player_code not in ratings:
        if initial_ratings is not None and player_code in initial_ratings:
            ratings[player_code] = float(initial_ratings[player_code])
        else:
            ratings[player_code] = default_rating
        stats[player_code] = {"games": 0, "wins": 0, "losses": 0}

    if player_code not in player_names and pd.notna(possible_name) and str(possible_name).strip():
        player_names[player_code] = str(possible_name).strip()


def run_elo_one_year(
    year_matches: pd.DataFrame,
    k: float,
    scale: float,
    initial_ratings: Optional[Dict[int, float]] = None,
    default_rating: float = DEFAULT_RATING,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run one year of Elo and return final ratings plus compact update summary."""
    ratings: Dict[int, float] = {}
    stats: Dict[int, Dict[str, int]] = {}
    player_names: Dict[int, str] = {}
    abs_updates: List[float] = []
    skipped_games = 0

    for row in year_matches.itertuples(index=False):
        winner_value = getattr(row, "winner")
        loser_value = getattr(row, "loser")
        if pd.isna(winner_value) or pd.isna(loser_value):
            skipped_games += 1
            continue

        winner = int(winner_value)
        loser = int(loser_value)
        initialise_player(
            winner,
            ratings,
            stats,
            player_names,
            initial_ratings,
            default_rating,
            getattr(row, "winner_name", pd.NA),
        )
        initialise_player(
            loser,
            ratings,
            stats,
            player_names,
            initial_ratings,
            default_rating,
            getattr(row, "loser_name", pd.NA),
        )

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, scale)

        rating_change = k * (1.0 - pred_winner_win)
        ratings[winner] = winner_rating_before + rating_change
        ratings[loser] = loser_rating_before - rating_change
        abs_updates.append(abs(rating_change))

        stats[winner]["games"] += 1
        stats[winner]["wins"] += 1
        stats[loser]["games"] += 1
        stats[loser]["losses"] += 1

    final_ratings = build_final_ratings(ratings, stats, player_names)
    updates = pd.Series(abs_updates, dtype=float)
    update_summary = {
        "number_of_matches": len(abs_updates),
        "skipped_games": skipped_games,
        "mean_abs_match_update": updates.mean() if not updates.empty else np.nan,
        "median_abs_match_update": updates.median() if not updates.empty else np.nan,
        "p90_abs_match_update": updates.quantile(0.90) if not updates.empty else np.nan,
        "max_abs_match_update": updates.max() if not updates.empty else np.nan,
    }
    return final_ratings, update_summary


def build_final_ratings(
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    player_names: Dict[int, str],
) -> pd.DataFrame:
    """Create final ratings table after one yearly rerun iteration."""
    rows = []
    for player_code, final_rating in ratings.items():
        player_stats = stats[player_code]
        rows.append(
            {
                "player_code": player_code,
                "player_name": player_names.get(player_code, pd.NA),
                "final_rating": final_rating,
                "games_played_in_year": player_stats["games"],
                "wins_in_year": player_stats["wins"],
                "losses_in_year": player_stats["losses"],
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


def ratings_to_initial_dict(final_ratings: pd.DataFrame) -> Dict[int, float]:
    """Convert final ratings dataframe to initial rating dictionary."""
    return dict(zip(final_ratings["player_code"].astype(int), final_ratings["final_rating"].astype(float)))


def compute_top_overlap(previous: pd.DataFrame, current: pd.DataFrame, top_n: int) -> float:
    """Compute top-N overlap between consecutive iterations."""
    previous_top = set(previous.sort_values(["final_rating", "player_code"], ascending=[False, True]).head(top_n)["player_code"].astype(int))
    current_top = set(current.sort_values(["final_rating", "player_code"], ascending=[False, True]).head(top_n)["player_code"].astype(int))
    denominator = min(top_n, len(previous_top), len(current_top))
    if denominator == 0:
        return np.nan
    return len(previous_top.intersection(current_top)) / denominator


def compare_with_previous(previous: Optional[pd.DataFrame], current: pd.DataFrame) -> Dict[str, object]:
    """Compare current final ratings against previous iteration final ratings."""
    if previous is None:
        return {
            "mean_abs_rating_change_from_previous_iteration": np.nan,
            "median_abs_rating_change_from_previous_iteration": np.nan,
            "p90_abs_rating_change_from_previous_iteration": np.nan,
            "max_abs_rating_change_from_previous_iteration": np.nan,
            "pearson_rating_correlation_with_previous_iteration": np.nan,
            "spearman_rank_correlation_with_previous_iteration": np.nan,
            "mean_abs_rank_change_from_previous_iteration": np.nan,
            "median_abs_rank_change_from_previous_iteration": np.nan,
            "top10_overlap_with_previous_iteration": np.nan,
            "top25_overlap_with_previous_iteration": np.nan,
            "top50_overlap_with_previous_iteration": np.nan,
            "top100_overlap_with_previous_iteration": np.nan,
            "converged_flag": False,
            "convergence_reason": "first_iteration_no_previous_comparison",
        }

    merged = previous[["player_code", "final_rating", "final_rank"]].merge(
        current[["player_code", "final_rating", "final_rank"]],
        on="player_code",
        how="inner",
        suffixes=("_previous", "_current"),
    )

    rating_change = (merged["final_rating_current"] - merged["final_rating_previous"]).abs()
    rank_change = (merged["final_rank_current"] - merged["final_rank_previous"]).abs()
    mean_abs_change = rating_change.mean()
    max_abs_change = rating_change.max()
    pearson = merged["final_rating_previous"].corr(merged["final_rating_current"]) if len(merged) >= 2 else np.nan
    spearman = merged["final_rank_previous"].corr(merged["final_rank_current"], method="spearman") if len(merged) >= 2 else np.nan

    converged = False
    reason = "not_converged"
    if pd.notna(mean_abs_change) and mean_abs_change < MEAN_ABS_CHANGE_THRESHOLD:
        converged = True
        reason = f"mean_abs_rating_change_below_{MEAN_ABS_CHANGE_THRESHOLD:g}"
    elif pd.notna(max_abs_change) and max_abs_change < MAX_ABS_CHANGE_THRESHOLD:
        converged = True
        reason = f"max_abs_rating_change_below_{MAX_ABS_CHANGE_THRESHOLD:g}"

    return {
        "mean_abs_rating_change_from_previous_iteration": mean_abs_change,
        "median_abs_rating_change_from_previous_iteration": rating_change.median(),
        "p90_abs_rating_change_from_previous_iteration": rating_change.quantile(0.90),
        "max_abs_rating_change_from_previous_iteration": max_abs_change,
        "pearson_rating_correlation_with_previous_iteration": pearson,
        "spearman_rank_correlation_with_previous_iteration": spearman,
        "mean_abs_rank_change_from_previous_iteration": rank_change.mean(),
        "median_abs_rank_change_from_previous_iteration": rank_change.median(),
        "top10_overlap_with_previous_iteration": compute_top_overlap(previous, current, 10),
        "top25_overlap_with_previous_iteration": compute_top_overlap(previous, current, 25),
        "top50_overlap_with_previous_iteration": compute_top_overlap(previous, current, 50),
        "top100_overlap_with_previous_iteration": compute_top_overlap(previous, current, 100),
        "converged_flag": converged,
        "convergence_reason": reason,
    }


def make_iteration_summary_row(
    year: int,
    setting_name: str,
    k: float,
    scale: float,
    iteration: int,
    final_ratings: pd.DataFrame,
    number_of_matches: int,
    comparison_metrics: Dict[str, object],
) -> Dict[str, object]:
    """Build one row for iteration-level convergence summary."""
    return {
        "year": year,
        "setting_name": setting_name,
        "k": k,
        "scale": scale,
        "iteration": iteration,
        "number_of_matches": number_of_matches,
        "number_of_players": len(final_ratings),
        "mean_final_rating": final_ratings["final_rating"].mean(),
        "std_final_rating": final_ratings["final_rating"].std(),
        "min_final_rating": final_ratings["final_rating"].min(),
        "max_final_rating": final_ratings["final_rating"].max(),
        **comparison_metrics,
    }


def add_run_columns(
    final_ratings: pd.DataFrame,
    year: int,
    setting_name: str,
    k: float,
    scale: float,
    iteration: int,
) -> pd.DataFrame:
    """Add year, setting and iteration metadata to final ratings."""
    final_ratings = final_ratings.copy()
    final_ratings.insert(0, "iteration", iteration)
    final_ratings.insert(0, "scale", scale)
    final_ratings.insert(0, "k", k)
    final_ratings.insert(0, "setting_name", setting_name)
    final_ratings.insert(0, "year", year)
    return final_ratings


def run_single_year_setting(
    year: int,
    year_matches: pd.DataFrame,
    setting: Dict[str, object],
) -> Tuple[List[Dict[str, object]], List[pd.DataFrame], List[Dict[str, object]], Dict[str, object]]:
    """Run repeated single-year reruns for one year and one Elo setting."""
    setting_name = str(setting["setting_name"])
    k = float(setting["k"])
    scale = float(setting["scale"])

    print(f"\n=== Running year={year}, setting={setting_name} (K={k:g}, scale={scale:g}) ===")

    iteration_rows: List[Dict[str, object]] = []
    final_rating_tables: List[pd.DataFrame] = []
    update_rows: List[Dict[str, object]] = []
    previous_final: Optional[pd.DataFrame] = None
    initial_ratings: Optional[Dict[int, float]] = None
    decision: Optional[Dict[str, object]] = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        final_ratings, update_summary = run_elo_one_year(
            year_matches=year_matches,
            k=k,
            scale=scale,
            initial_ratings=initial_ratings,
            default_rating=DEFAULT_RATING,
        )
        comparison_metrics = compare_with_previous(previous_final, final_ratings)
        iteration_rows.append(
            make_iteration_summary_row(
                year=year,
                setting_name=setting_name,
                k=k,
                scale=scale,
                iteration=iteration,
                final_ratings=final_ratings,
                number_of_matches=int(update_summary["number_of_matches"]),
                comparison_metrics=comparison_metrics,
            )
        )

        update_rows.append(
            {
                "year": year,
                "setting_name": setting_name,
                "k": k,
                "scale": scale,
                "iteration": iteration,
                **update_summary,
            }
        )
        final_rating_tables.append(add_run_columns(final_ratings, year, setting_name, k, scale, iteration))

        mean_change = comparison_metrics["mean_abs_rating_change_from_previous_iteration"]
        max_change = comparison_metrics["max_abs_rating_change_from_previous_iteration"]
        print(
            f"  iteration {iteration}: "
            f"mean_abs_change={format_optional_float(mean_change)}, "
            f"max_abs_change={format_optional_float(max_change)}, "
            f"converged={comparison_metrics['converged_flag']}, "
            f"reason={comparison_metrics['convergence_reason']}"
        )

        if comparison_metrics["converged_flag"]:
            decision = make_decision_row(year, setting_name, k, scale, iteration_rows[-1], True)
            print(f"  reached convergence threshold at iteration {iteration}")
            break

        previous_final = final_ratings
        initial_ratings = ratings_to_initial_dict(final_ratings)

    if decision is None:
        decision = make_decision_row(year, setting_name, k, scale, iteration_rows[-1], False)

    return iteration_rows, final_rating_tables, update_rows, decision


def format_optional_float(value: object) -> str:
    """Format floats for terminal output."""
    if pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def make_decision_row(
    year: int,
    setting_name: str,
    k: float,
    scale: float,
    final_iteration_row: Dict[str, object],
    converged: bool,
) -> Dict[str, object]:
    """Create one convergence decision row for a year-setting combination."""
    return {
        "year": year,
        "setting_name": setting_name,
        "k": k,
        "scale": scale,
        "converged": converged,
        "convergence_iteration": final_iteration_row["iteration"] if converged else np.nan,
        "convergence_reason": final_iteration_row["convergence_reason"],
        "final_mean_abs_change": final_iteration_row["mean_abs_rating_change_from_previous_iteration"],
        "final_median_abs_change": final_iteration_row["median_abs_rating_change_from_previous_iteration"],
        "final_p90_abs_change": final_iteration_row["p90_abs_rating_change_from_previous_iteration"],
        "final_max_abs_change": final_iteration_row["max_abs_rating_change_from_previous_iteration"],
        "final_spearman_rank_correlation": final_iteration_row["spearman_rank_correlation_with_previous_iteration"],
        "total_iterations_run": final_iteration_row["iteration"],
    }


def write_markdown_summary(
    matches: pd.DataFrame,
    iteration_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    update_summary: pd.DataFrame,
    output_path: Path,
) -> str:
    """Write a meeting-ready markdown summary."""
    year_lines = []
    for year in YEARS_TO_TEST:
        year_matches = matches[matches["year"] == year]
        players = pd.concat([year_matches["winner"], year_matches["loser"]]).dropna().astype(int).nunique()
        year_lines.append(f"* {year}: {len(year_matches)} matches, {players} players")

    setting_lines = [
        f"* {setting['label']}: `{setting['setting_name']}`, K={setting['k']:g}, scale={setting['scale']:g}"
        for setting in ELO_SETTINGS
    ]

    convergence_lines = []
    for _, row in decisions.sort_values(["year", "setting_name"], ascending=[False, True]).iterrows():
        if bool(row["converged"]):
            status = f"converged at iteration {int(row['convergence_iteration'])}"
        else:
            status = f"not converged after {int(row['total_iterations_run'])} iterations"
        convergence_lines.append(
            f"* {int(row['year'])}, {row['setting_name']}: {status}; "
            f"final mean abs change = {format_optional_float(row['final_mean_abs_change'])}, "
            f"final max abs change = {format_optional_float(row['final_max_abs_change'])}, "
            f"final Spearman = {format_optional_float(row['final_spearman_rank_correlation'])}."
        )

    effect_lines = []
    grouped = decisions.groupby("setting_name", sort=False)
    for setting_name, group in grouped:
        converged_count = int(group["converged"].sum())
        median_iterations = group["total_iterations_run"].median()
        median_final_change = group["final_mean_abs_change"].median()
        effect_lines.append(
            f"* {setting_name}: {converged_count}/{len(group)} year tests converged; "
            f"median iterations run = {median_iterations:.1f}; "
            f"median final mean abs change = {median_final_change:.4f}."
        )

    markdown = f"""# Single-year repeated rerun convergence diagnostic

## 1. Aim of this experiment

This is the supervisor-suggested single-year repeated rerun diagnostic.
For a chosen year, the script runs that year's matches once, uses the final ratings as the initial ratings for the next iteration, and repeats until the rating list stabilises or the maximum iteration count is reached.

## 2. Important limitation

This is not a fair prediction test.
The same year's match results are reused across iterations, so log loss, Brier score or accuracy from these reruns would not be valid out-of-sample performance measures.
This experiment is only a rating convergence / stability diagnostic.
It complements the full-history burn-in stability experiment by asking whether a single year of repeated information can stabilise an Elo rating list.

## 3. Data and years used

Input file: `outputs/elo_optimization/matches_1985_2025_checked.csv`.
The default years are:

{chr(10).join(year_lines)}

## 4. Elo settings tested

{chr(10).join(setting_lines)}

Convergence thresholds:

* mean absolute rating change from previous iteration < {MEAN_ABS_CHANGE_THRESHOLD}
* or max absolute rating change from previous iteration < {MAX_ABS_CHANGE_THRESHOLD}
* Spearman rank correlation is recorded, with reference threshold {SPEARMAN_THRESHOLD}, but is not used alone to decide convergence.

## 5. Convergence results

{chr(10).join(convergence_lines)}

## 6. Effect of K and scale

{chr(10).join(effect_lines)}

The setting-level comparison should be read cautiously because this diagnostic reuses the same matches repeatedly.
Different K/scale values can change both how quickly rating values move and how quickly the final ranking stabilises.

## 7. Interpretation

The full-history burn-in experiment checks sensitivity to historical data length.
This single-year rerun diagnostic checks whether one year's results, repeatedly reused, can produce a stable rating list.
Together, they help explain why historical burn-in matters and why short-window ratings can be sensitive to parameter choices.

## 8. Implication for Elo baseline

This experiment does not directly choose the final Elo baseline.
It helps interpret the need for historical burn-in and the role of K/scale in rating stability.
If a setting does not stabilise under repeated single-year reruns, that is evidence that one-year data alone is not enough to define a robust rating list.

## 9. Next step

The next Elo optimisation step is event-level volatility analysis, which addresses the difference between match-by-match volatility and tournament/event-level resolution.
"""
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def remove_existing_outputs() -> None:
    """Remove this script's own outputs before a fresh run."""
    for path in [
        ITERATION_SUMMARY_PATH,
        FINAL_RATINGS_PATH,
        UPDATE_SUMMARY_PATH,
        DECISIONS_PATH,
        SUMMARY_MD_PATH,
    ]:
        if path.exists():
            path.unlink()


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs()

    print("=== Single-year repeated rerun convergence diagnostic ===")
    print(f"Years to test: {YEARS_TO_TEST}")
    print("Elo settings:")
    for setting in ELO_SETTINGS:
        print(
            f"  {setting['setting_name']}: K={setting['k']:g}, scale={setting['scale']:g}"
        )

    matches = load_matches()
    all_iteration_rows: List[Dict[str, object]] = []
    all_final_rating_tables: List[pd.DataFrame] = []
    all_update_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []

    for year in YEARS_TO_TEST:
        year_matches = matches[matches["year"] == year].copy()
        sort_cols = ["year", "event_order_date", "event", "code", "fcode"]
        year_matches = year_matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        for setting in ELO_SETTINGS:
            iteration_rows, final_rating_tables, update_rows, decision = run_single_year_setting(
                year=year,
                year_matches=year_matches,
                setting=setting,
            )
            all_iteration_rows.extend(iteration_rows)
            all_final_rating_tables.extend(final_rating_tables)
            all_update_rows.extend(update_rows)
            decision_rows.append(decision)

    iteration_summary = pd.DataFrame(all_iteration_rows)
    final_ratings = pd.concat(all_final_rating_tables, ignore_index=True)
    update_summary = pd.DataFrame(all_update_rows)
    decisions = pd.DataFrame(decision_rows)

    iteration_summary.to_csv(ITERATION_SUMMARY_PATH, index=False)
    final_ratings.to_csv(FINAL_RATINGS_PATH, index=False)
    update_summary.to_csv(UPDATE_SUMMARY_PATH, index=False)
    decisions.to_csv(DECISIONS_PATH, index=False)
    write_markdown_summary(matches, iteration_summary, decisions, update_summary, SUMMARY_MD_PATH)

    print("\nOutput paths:")
    print(f"  iteration summary: {ITERATION_SUMMARY_PATH}")
    print(f"  final ratings: {FINAL_RATINGS_PATH}")
    print(f"  update summary: {UPDATE_SUMMARY_PATH}")
    print(f"  convergence decisions: {DECISIONS_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
