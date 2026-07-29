"""Meeting 8 technical diagnostic: burn-in entry status and rating-scale drift.

This script answers the two targeted technical questions raised after Meeting 7:

1. Distinguish players who are first observed at the start of the historical
   model run from players whose first recorded appearance occurs after a
   burn-in period.
2. Diagnose whether the low-inflation Glicko rating scale moves over calendar
   time and where the fixed new-player anchor of 1500 sits relative to
   established players and debut opponents.

The script does not tune a new model. It imports and reuses the validated
Meeting 5 Step 24 low-inflation Glicko configuration and checks its 2025 debut
rows against the frozen Meeting 7 Step 34 appearance dataset.
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting8_technical"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP34_APPEARANCE_PATH = (
    PROJECT_ROOT / "outputs" / "meeting7" / "34_early_game_appearance_dataset.csv"
)

MODEL_START_YEAR = 1985
MODEL_END_YEAR = 2025
TEST_YEAR = 2025
PRIMARY_BURN_IN_YEARS = 5
POST_BURN_IN_START_YEAR = MODEL_START_YEAR + PRIMARY_BURN_IN_YEARS
INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
EPS = 1e-15

PLAYER_ENTRY_PATH = OUTPUT_DIR / "41_player_entry_classification.csv"
DEBUT_DETAIL_PATH = OUTPUT_DIR / "41_debut_appearance_diagnostics.csv"
DEBUT_COHORT_PATH = OUTPUT_DIR / "41_debut_cohort_summary.csv"
YEARLY_SCALE_PATH = OUTPUT_DIR / "41_yearly_rating_scale_drift.csv"
YEARLY_DEBUT_PATH = OUTPUT_DIR / "41_yearly_debut_anchor_diagnostics.csv"
BURN_IN_SENSITIVITY_PATH = OUTPUT_DIR / "41_burnin_definition_sensitivity.csv"
TEST_YEAR_VALIDATION_PATH = OUTPUT_DIR / "41_2025_first_appearance_validation.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "41_validation_checks.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "41_figure_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "41_meeting8_technical_summary.md"

RATING_SCALE_FIGURE_PATH = FIGURE_DIR / "41_fig01_glicko_rating_scale_by_year.png"
DEBUT_ANCHOR_FIGURE_PATH = FIGURE_DIR / "41_fig02_debut_anchor_by_year.png"
COHORT_BIAS_FIGURE_PATH = FIGURE_DIR / "41_fig03_debut_prediction_by_cohort.png"

EXPECTED_2025_FIRST_APPEARANCES = 76
EXPECTED_2025_FIRST_MATCHES = 74
EXPECTED_2025_MEAN_P = 0.743448
EXPECTED_2025_WIN_RATE = 0.407895
EXPECTED_2025_BRIER = 0.322316
EXPECTED_2025_MEAN_OPPONENT_RATING = 1180.755


def configure_output_root(output_root: str | Path) -> Path:
    """Redirect Step 41 outputs while retaining frozen default inputs."""

    global OUTPUT_DIR, FIGURE_DIR
    global PLAYER_ENTRY_PATH, DEBUT_DETAIL_PATH, DEBUT_COHORT_PATH
    global YEARLY_SCALE_PATH, YEARLY_DEBUT_PATH, BURN_IN_SENSITIVITY_PATH
    global TEST_YEAR_VALIDATION_PATH, VALIDATION_CHECKS_PATH
    global FIGURE_MANIFEST_PATH, SUMMARY_PATH
    global RATING_SCALE_FIGURE_PATH, DEBUT_ANCHOR_FIGURE_PATH
    global COHORT_BIAS_FIGURE_PATH

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    OUTPUT_DIR = root.resolve() / "meeting8_technical"
    FIGURE_DIR = OUTPUT_DIR / "figures"
    PLAYER_ENTRY_PATH = OUTPUT_DIR / "41_player_entry_classification.csv"
    DEBUT_DETAIL_PATH = OUTPUT_DIR / "41_debut_appearance_diagnostics.csv"
    DEBUT_COHORT_PATH = OUTPUT_DIR / "41_debut_cohort_summary.csv"
    YEARLY_SCALE_PATH = OUTPUT_DIR / "41_yearly_rating_scale_drift.csv"
    YEARLY_DEBUT_PATH = OUTPUT_DIR / "41_yearly_debut_anchor_diagnostics.csv"
    BURN_IN_SENSITIVITY_PATH = OUTPUT_DIR / "41_burnin_definition_sensitivity.csv"
    TEST_YEAR_VALIDATION_PATH = OUTPUT_DIR / "41_2025_first_appearance_validation.csv"
    VALIDATION_CHECKS_PATH = OUTPUT_DIR / "41_validation_checks.csv"
    FIGURE_MANIFEST_PATH = OUTPUT_DIR / "41_figure_manifest.csv"
    SUMMARY_PATH = OUTPUT_DIR / "41_meeting8_technical_summary.md"
    RATING_SCALE_FIGURE_PATH = FIGURE_DIR / "41_fig01_glicko_rating_scale_by_year.png"
    DEBUT_ANCHOR_FIGURE_PATH = FIGURE_DIR / "41_fig02_debut_anchor_by_year.png"
    COHORT_BIAS_FIGURE_PATH = FIGURE_DIR / "41_fig03_debut_prediction_by_cohort.png"
    return OUTPUT_DIR


def load_step24_module():
    """Return the canonical validated Step 24 implementation."""

    from code.pipelines import glicko_pipeline

    return glicko_pipeline


def format_player_name(value: Any) -> str | pd._libs.missing.NAType:
    """Return a clean player name or pandas NA."""

    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def cohort_for_first_year(first_year: int) -> str:
    """Assign the primary five-year burn-in cohort."""

    if first_year == MODEL_START_YEAR:
        return "system_start_left_censored"
    if first_year < POST_BURN_IN_START_YEAR:
        return "within_5y_burn_in_recorded_entry"
    if first_year < TEST_YEAR:
        return "post_burn_in_recorded_entry"
    if first_year == TEST_YEAR:
        return "test_year_recorded_entry"
    raise ValueError(f"Unexpected first recorded year: {first_year}")


def build_player_entry_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one row per player with first-recorded timing and cohort."""

    winner_rows = matches[
        [
            "fcode",
            "year",
            "event",
            "event_order_date",
            "winner",
            "winner_name",
        ]
    ].copy()
    winner_rows = winner_rows.rename(
        columns={"winner": "player_id", "winner_name": "player_name"}
    )
    winner_rows["result_in_first_recorded_match"] = 1

    loser_rows = matches[
        [
            "fcode",
            "year",
            "event",
            "event_order_date",
            "loser",
            "loser_name",
        ]
    ].copy()
    loser_rows = loser_rows.rename(
        columns={"loser": "player_id", "loser_name": "player_name"}
    )
    loser_rows["result_in_first_recorded_match"] = 0

    long = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long["player_id"] = pd.to_numeric(long["player_id"], errors="raise").astype(int)
    long["year"] = pd.to_numeric(long["year"], errors="raise").astype(int)
    long["player_name"] = long["player_name"].apply(format_player_name)
    long["event_order_date"] = pd.to_datetime(long["event_order_date"], errors="coerce")
    long["date_missing"] = long["event_order_date"].isna()
    long = long.sort_values(
        ["year", "date_missing", "event_order_date", "event", "fcode", "player_id"],
        na_position="last",
    )

    first = long.drop_duplicates("player_id", keep="first").copy()
    first = first.rename(
        columns={
            "year": "first_recorded_year",
            "event_order_date": "first_recorded_date",
            "fcode": "first_recorded_match_id",
            "event": "first_recorded_event",
        }
    )
    match_counts = long.groupby("player_id").size().rename("recorded_matches_1985_2025")
    first = first.merge(match_counts, on="player_id", how="left", validate="one_to_one")
    first["entry_cohort"] = first["first_recorded_year"].map(cohort_for_first_year)
    first["is_model_start_player"] = first["first_recorded_year"].eq(MODEL_START_YEAR)
    first["is_within_primary_burn_in"] = first["first_recorded_year"].lt(
        POST_BURN_IN_START_YEAR
    )
    first["is_post_burn_in_recorded_entry"] = first["first_recorded_year"].ge(
        POST_BURN_IN_START_YEAR
    )
    first["is_test_year_recorded_entry"] = first["first_recorded_year"].eq(TEST_YEAR)
    first["burn_in_definition"] = (
        f"{MODEL_START_YEAR}-{POST_BURN_IN_START_YEAR - 1} excluded; "
        f"recorded entry from {POST_BURN_IN_START_YEAR}"
    )

    columns = [
        "player_id",
        "player_name",
        "first_recorded_year",
        "first_recorded_date",
        "first_recorded_match_id",
        "first_recorded_event",
        "result_in_first_recorded_match",
        "recorded_matches_1985_2025",
        "entry_cohort",
        "is_model_start_player",
        "is_within_primary_burn_in",
        "is_post_burn_in_recorded_entry",
        "is_test_year_recorded_entry",
        "burn_in_definition",
    ]
    return first[columns].sort_values(
        ["first_recorded_year", "first_recorded_match_id", "player_id"]
    ).reset_index(drop=True)


def safe_log_loss(probability: pd.Series, outcome: pd.Series) -> float:
    """Return mean binary log loss."""

    p = probability.astype(float).clip(EPS, 1.0 - EPS)
    y = outcome.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def summarise_debut_group(group: pd.DataFrame, label: str) -> dict[str, Any]:
    """Summarise first-recorded appearances for one cohort or year."""

    p = group["p_focal_glicko_low_current"].astype(float)
    y = group["outcome_focal"].astype(float)
    opponent_rating = group["opponent_rating_before"].astype(float)
    rating_gap = group["focal_minus_opponent_rating"].astype(float)
    both_debut_matches = group.loc[group["both_players_first_recorded"], "match_id"].nunique()
    return {
        "group": label,
        "n_first_appearances": int(len(group)),
        "n_unique_players": int(group["player_id"].nunique()),
        "n_unique_matches": int(group["match_id"].nunique()),
        "n_both_debut_matches": int(both_debut_matches),
        "mean_predicted_win_probability": float(p.mean()),
        "empirical_win_rate": float(y.mean()),
        "prediction_bias_mean_p_minus_win_rate": float(p.mean() - y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": safe_log_loss(p, y),
        "mean_focal_rating": float(group["focal_rating_before"].mean()),
        "mean_opponent_rating": float(opponent_rating.mean()),
        "median_opponent_rating": float(opponent_rating.median()),
        "mean_focal_minus_opponent_rating": float(rating_gap.mean()),
        "median_focal_minus_opponent_rating": float(rating_gap.median()),
        "mean_focal_rd": float(group["focal_rd_before"].mean()),
        "mean_opponent_rd": float(group["opponent_rd_before"].mean()),
        "mean_opponent_prior_games": float(group["opponent_games_before"].mean()),
    }


def run_low_inflation_diagnostic(
    matches: pd.DataFrame,
    player_entries: pd.DataFrame,
    step24: Any,
    inactivity_unit: str,
    low_variant: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen low-inflation Glicko setting and retain diagnostic state."""

    c_value = float(low_variant["c_value"])
    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_period_index: dict[int, int] = {}
    games_before: defaultdict[int, int] = defaultdict(int)

    first_year = (
        player_entries.set_index("player_id")["first_recorded_year"].astype(int).to_dict()
    )

    debut_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    current_year: int | None = None
    active_players: set[int] = set()
    year_net_update_change = 0.0
    cumulative_net_update_change = 0.0
    year_start_rating_sum = 0.0
    year_match_count = 0

    def finish_year(year: int) -> None:
        nonlocal cumulative_net_update_change
        active = sorted(active_players)
        established = [player for player in active if first_year[player] < year]
        all_known = sorted(ratings)
        new_players = [player for player in all_known if first_year[player] == year]

        active_ratings = np.asarray([ratings[player] for player in active], dtype=float)
        established_ratings = np.asarray(
            [ratings[player] for player in established], dtype=float
        )
        all_known_ratings = np.asarray(
            [ratings[player] for player in all_known], dtype=float
        )
        cumulative_net_update_change += year_net_update_change

        expected_end_sum = (
            year_start_rating_sum
            + len(new_players) * INITIAL_RATING
            + year_net_update_change
        )
        observed_end_sum = float(all_known_ratings.sum())

        def stat(values: np.ndarray, fn: str) -> float:
            if len(values) == 0:
                return float("nan")
            if fn == "mean":
                return float(np.mean(values))
            if fn == "median":
                return float(np.median(values))
            if fn == "p10":
                return float(np.quantile(values, 0.10))
            if fn == "p90":
                return float(np.quantile(values, 0.90))
            raise ValueError(fn)

        established_median = stat(established_ratings, "median")
        yearly_rows.append(
            {
                "year": year,
                "matches": int(year_match_count),
                "n_active_players": int(len(active)),
                "n_established_active_players": int(len(established)),
                "n_known_players_end_year": int(len(all_known)),
                "n_new_recorded_players": int(len(new_players)),
                "mean_rating_active": stat(active_ratings, "mean"),
                "median_rating_active": stat(active_ratings, "median"),
                "p10_rating_active": stat(active_ratings, "p10"),
                "p90_rating_active": stat(active_ratings, "p90"),
                "mean_rating_established_active": stat(established_ratings, "mean"),
                "median_rating_established_active": established_median,
                "p10_rating_established_active": stat(established_ratings, "p10"),
                "p90_rating_established_active": stat(established_ratings, "p90"),
                "mean_rating_all_known": stat(all_known_ratings, "mean"),
                "median_rating_all_known": stat(all_known_ratings, "median"),
                "initial_rating": INITIAL_RATING,
                "initial_minus_established_active_median": (
                    INITIAL_RATING - established_median
                    if np.isfinite(established_median)
                    else float("nan")
                ),
                "net_glicko_rating_update_change": float(year_net_update_change),
                "cumulative_net_glicko_rating_update_change": float(
                    cumulative_net_update_change
                ),
                "rating_sum_end_year": observed_end_sum,
                "rating_sum_reconciliation_error": float(
                    observed_end_sum - expected_end_sum
                ),
            }
        )

    sim_columns = [
        "fcode",
        "year",
        "event",
        "event_order_date",
        "inactivity_period_index",
        "winner",
        "loser",
    ]

    for row in matches[sim_columns].itertuples(index=False):
        year = int(row.year)
        if current_year is None:
            current_year = year
            year_start_rating_sum = float(sum(ratings.values()))
        elif year != current_year:
            finish_year(current_year)
            current_year = year
            active_players = set()
            year_net_update_change = 0.0
            year_start_rating_sum = float(sum(ratings.values()))
            year_match_count = 0

        winner = int(row.winner)
        loser = int(row.loser)
        period_index = int(row.inactivity_period_index)
        winner_is_first = games_before[winner] == 0
        loser_is_first = games_before[loser] == 0

        if winner not in ratings:
            ratings[winner] = INITIAL_RATING
            rds[winner] = INITIAL_RD
        else:
            elapsed = period_index - last_period_index[winner]
            rds[winner] = step24.inflate_rd_for_inactivity(
                rds[winner], elapsed, c_value
            )

        if loser not in ratings:
            ratings[loser] = INITIAL_RATING
            rds[loser] = INITIAL_RD
        else:
            elapsed = period_index - last_period_index[loser]
            rds[loser] = step24.inflate_rd_for_inactivity(
                rds[loser], elapsed, c_value
            )

        winner_rating_before = float(ratings[winner])
        winner_rd_before = float(rds[winner])
        loser_rating_before = float(ratings[loser])
        loser_rd_before = float(rds[loser])

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        if player_a == winner:
            rating_a, rd_a = winner_rating_before, winner_rd_before
            rating_b, rd_b = loser_rating_before, loser_rd_before
        else:
            rating_a, rd_a = loser_rating_before, loser_rd_before
            rating_b, rd_b = winner_rating_before, winner_rd_before

        p_a_current = step24.expected_score(rating_a, rating_b, rd_b)
        match_id = int(row.fcode)
        event_date = pd.to_datetime(row.event_order_date, errors="coerce")

        participant_state = {
            winner: {
                "opponent": loser,
                "outcome": 1,
                "rating": winner_rating_before,
                "rd": winner_rd_before,
                "opponent_rating": loser_rating_before,
                "opponent_rd": loser_rd_before,
                "opponent_games_before": games_before[loser],
                "is_first": winner_is_first,
            },
            loser: {
                "opponent": winner,
                "outcome": 0,
                "rating": loser_rating_before,
                "rd": loser_rd_before,
                "opponent_rating": winner_rating_before,
                "opponent_rd": winner_rd_before,
                "opponent_games_before": games_before[winner],
                "is_first": loser_is_first,
            },
        }

        for focal, state in participant_state.items():
            if not state["is_first"]:
                continue
            p_focal = p_a_current if focal == player_a else 1.0 - p_a_current
            focal_first_year = first_year[focal]
            debut_rows.append(
                {
                    "match_id": match_id,
                    "year": year,
                    "event": row.event,
                    "event_order_date": event_date,
                    "player_id": focal,
                    "opponent_id": state["opponent"],
                    "focal_side": "A" if focal == player_a else "B",
                    "outcome_focal": int(state["outcome"]),
                    "p_focal_glicko_low_current": float(p_focal),
                    "focal_rating_before": float(state["rating"]),
                    "opponent_rating_before": float(state["opponent_rating"]),
                    "focal_minus_opponent_rating": float(
                        state["rating"] - state["opponent_rating"]
                    ),
                    "focal_rd_before": float(state["rd"]),
                    "opponent_rd_before": float(state["opponent_rd"]),
                    "opponent_games_before": int(state["opponent_games_before"]),
                    "both_players_first_recorded": bool(
                        winner_is_first and loser_is_first
                    ),
                    "first_recorded_year": int(focal_first_year),
                    "entry_cohort": cohort_for_first_year(focal_first_year),
                    "post_primary_burn_in": bool(
                        focal_first_year >= POST_BURN_IN_START_YEAR
                    ),
                    "probability_convention": (
                        "canonical player-A probability; complement for player B"
                    ),
                    "initial_rating": INITIAL_RATING,
                    "initial_rd": INITIAL_RD,
                    "low_inflation_c": c_value,
                }
            )

        update = step24.update_two_players_single_game(
            winner_rating_before,
            winner_rd_before,
            loser_rating_before,
            loser_rd_before,
            1.0,
        )
        ratings[winner] = float(update.player1_rating_after)
        rds[winner] = float(update.player1_rd_after)
        ratings[loser] = float(update.player2_rating_after)
        rds[loser] = float(update.player2_rd_after)

        winner_change = ratings[winner] - winner_rating_before
        loser_change = ratings[loser] - loser_rating_before
        year_net_update_change += winner_change + loser_change

        last_period_index[winner] = max(
            last_period_index.get(winner, period_index), period_index
        )
        last_period_index[loser] = max(
            last_period_index.get(loser, period_index), period_index
        )
        games_before[winner] += 1
        games_before[loser] += 1
        active_players.update([winner, loser])
        year_match_count += 1

    if current_year is not None:
        finish_year(current_year)

    debut = pd.DataFrame(debut_rows)
    yearly = pd.DataFrame(yearly_rows)
    return debut, yearly


def build_debut_cohort_summary(debut: pd.DataFrame) -> pd.DataFrame:
    """Build primary cohort summaries plus useful combined rows."""

    order = [
        "system_start_left_censored",
        "within_5y_burn_in_recorded_entry",
        "post_burn_in_recorded_entry",
        "test_year_recorded_entry",
    ]
    rows = []
    for cohort in order:
        rows.append(summarise_debut_group(debut.loc[debut["entry_cohort"].eq(cohort)], cohort))

    rows.append(
        summarise_debut_group(
            debut.loc[debut["post_primary_burn_in"]],
            "all_post_burn_in_recorded_entries_1990_2025",
        )
    )
    rows.append(summarise_debut_group(debut, "all_first_recorded_appearances_1985_2025"))
    return pd.DataFrame(rows)


def build_yearly_debut_summary(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise the relative new-player anchor for each calendar year."""

    rows = []
    for year, group in debut.groupby("year", sort=True):
        row = summarise_debut_group(group, str(int(year)))
        row["year"] = int(year)
        row["entry_cohort"] = cohort_for_first_year(int(year))
        rows.append(row)
    result = pd.DataFrame(rows)
    preferred = ["year", "entry_cohort"] + [
        col for col in result.columns if col not in {"year", "entry_cohort", "group"}
    ]
    return result[preferred]


def build_burn_in_sensitivity(debut: pd.DataFrame) -> pd.DataFrame:
    """Show how reasonable burn-in cutoffs change the historical diagnostic."""

    rows = []
    for burn_in_years in [1, 3, 5, 10]:
        post_start = MODEL_START_YEAR + burn_in_years
        subset = debut.loc[debut["year"].ge(post_start)].copy()
        summary = summarise_debut_group(
            subset, f"recorded_entry_from_{post_start}"
        )
        rows.append(
            {
                "burn_in_years": burn_in_years,
                "burn_in_calendar_years": f"{MODEL_START_YEAR}-{post_start - 1}",
                "post_burn_in_start_year": post_start,
                "n_post_burn_in_first_appearances": summary["n_first_appearances"],
                "mean_predicted_win_probability": summary[
                    "mean_predicted_win_probability"
                ],
                "empirical_win_rate": summary["empirical_win_rate"],
                "prediction_bias": summary[
                    "prediction_bias_mean_p_minus_win_rate"
                ],
                "brier": summary["brier"],
                "mean_opponent_rating": summary["mean_opponent_rating"],
                "mean_focal_minus_opponent_rating": summary[
                    "mean_focal_minus_opponent_rating"
                ],
                "n_2025_first_appearances_retained": int(
                    subset["year"].eq(TEST_YEAR).sum()
                ),
                "interpretation": (
                    "Sensitivity only. The primary definition uses five calendar years."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_2025_validation(
    debut: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare rerun 2025 debut rows with the frozen Step 34 appearance data."""

    reference = pd.read_csv(STEP34_APPEARANCE_PATH, low_memory=False)
    reference["first_1"] = reference["first_1"].astype(str).str.lower().eq("true")
    reference = reference.loc[reference["first_1"]].copy()
    reference = reference[
        [
            "match_id",
            "player_id",
            "opponent_id",
            "p_focal_Glicko_low_fixed",
            "rating_focal_Glicko_low",
            "rating_opponent_Glicko_low",
            "rd_focal_Glicko_low",
            "rd_opponent_Glicko_low",
        ]
    ]
    for col in ["match_id", "player_id", "opponent_id"]:
        reference[col] = pd.to_numeric(reference[col], errors="raise").astype(int)

    current = debut.loc[debut["year"].eq(TEST_YEAR)].copy()
    current = current.rename(
        columns={
            "p_focal_glicko_low_current": "rerun_probability",
            "focal_rating_before": "rerun_focal_rating",
            "opponent_rating_before": "rerun_opponent_rating",
            "focal_rd_before": "rerun_focal_rd",
            "opponent_rd_before": "rerun_opponent_rd",
        }
    )
    current = current[
        [
            "match_id",
            "player_id",
            "opponent_id",
            "outcome_focal",
            "rerun_probability",
            "rerun_focal_rating",
            "rerun_opponent_rating",
            "rerun_focal_rd",
            "rerun_opponent_rd",
            "entry_cohort",
        ]
    ]

    merged = current.merge(
        reference,
        on=["match_id", "player_id", "opponent_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged["probability_difference_rerun_minus_step34"] = (
        merged["rerun_probability"] - merged["p_focal_Glicko_low_fixed"]
    )
    merged["opponent_rating_difference_rerun_minus_step34"] = (
        merged["rerun_opponent_rating"] - merged["rating_opponent_Glicko_low"]
    )
    merged["opponent_rd_difference_rerun_minus_step34"] = (
        merged["rerun_opponent_rd"] - merged["rd_opponent_Glicko_low"]
    )

    matched = merged.loc[merged["_merge"].eq("both")].copy()
    aggregates = {
        "rerun_rows": float(len(current)),
        "reference_rows": float(len(reference)),
        "matched_rows": float(len(matched)),
        "rerun_unique_matches": float(current["match_id"].nunique()),
        "rerun_mean_p": float(current["rerun_probability"].mean()),
        "rerun_win_rate": float(current["outcome_focal"].mean()),
        "rerun_brier": float(
            np.mean(
                (
                    current["rerun_probability"].astype(float)
                    - current["outcome_focal"].astype(float)
                )
                ** 2
            )
        ),
        "rerun_mean_opponent_rating": float(current["rerun_opponent_rating"].mean()),
        "max_abs_probability_difference": float(
            matched["probability_difference_rerun_minus_step34"].abs().max()
        ),
        "max_abs_opponent_rating_difference": float(
            matched["opponent_rating_difference_rerun_minus_step34"].abs().max()
        ),
        "max_abs_opponent_rd_difference": float(
            matched["opponent_rd_difference_rerun_minus_step34"].abs().max()
        ),
    }
    return merged, aggregates


def add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    actual: Any,
    expected: Any,
    details: str,
) -> None:
    """Append a machine-readable validation check."""

    rows.append(
        {
            "check": check,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "details": details,
        }
    )


def build_validation_checks(
    matches: pd.DataFrame,
    player_entries: pd.DataFrame,
    debut: pd.DataFrame,
    yearly: pd.DataFrame,
    validation: pd.DataFrame,
    aggregates: dict[str, float],
    low_variant: dict[str, Any],
) -> pd.DataFrame:
    """Build input, model-reuse, reconciliation, and regression checks."""

    rows: list[dict[str, Any]] = []
    all_players = pd.concat([matches["winner"], matches["loser"]]).astype(int).nunique()
    add_check(
        rows,
        "one_entry_row_per_player",
        len(player_entries) == all_players,
        len(player_entries),
        all_players,
        "Every observed player has exactly one first-recorded classification row.",
    )
    add_check(
        rows,
        "history_year_range",
        int(matches["year"].min()) == MODEL_START_YEAR
        and int(matches["year"].max()) == MODEL_END_YEAR,
        f"{int(matches['year'].min())}-{int(matches['year'].max())}",
        f"{MODEL_START_YEAR}-{MODEL_END_YEAR}",
        "The diagnostic uses the full frozen historical range.",
    )
    add_check(
        rows,
        "one_debut_appearance_per_player",
        len(debut) == all_players and debut["player_id"].nunique() == all_players,
        f"rows={len(debut)}, unique={debut['player_id'].nunique()}",
        all_players,
        "Each player contributes one first-recorded focal appearance.",
    )
    add_check(
        rows,
        "all_debut_focal_ratings_equal_1500",
        bool(np.isclose(debut["focal_rating_before"], INITIAL_RATING).all()),
        float(debut["focal_rating_before"].mean()),
        INITIAL_RATING,
        "The fixed new-player rating anchor is unchanged.",
    )
    add_check(
        rows,
        "all_debut_focal_rds_equal_350",
        bool(np.isclose(debut["focal_rd_before"], INITIAL_RD).all()),
        float(debut["focal_rd_before"].mean()),
        INITIAL_RD,
        "The fixed new-player RD anchor is unchanged.",
    )
    add_check(
        rows,
        "probabilities_in_unit_interval",
        bool(debut["p_focal_glicko_low_current"].between(0.0, 1.0).all()),
        f"min={debut['p_focal_glicko_low_current'].min():.12f}, "
        f"max={debut['p_focal_glicko_low_current'].max():.12f}",
        "[0, 1]",
        "All saved probabilities are valid.",
    )
    add_check(
        rows,
        "primary_low_inflation_variant_reused",
        str(low_variant["variant"]) == "low_inflation",
        low_variant["variant"],
        "low_inflation",
        "The diagnostic imports the existing Step 24 configuration.",
    )
    add_check(
        rows,
        "yearly_rating_sum_reconciles",
        bool(yearly["rating_sum_reconciliation_error"].abs().max() < 1e-6),
        float(yearly["rating_sum_reconciliation_error"].abs().max()),
        "< 1e-6",
        "End-year rating sums equal prior sums plus new anchors and net updates.",
    )
    add_check(
        rows,
        "2025_first_appearance_rows",
        int(aggregates["rerun_rows"]) == EXPECTED_2025_FIRST_APPEARANCES,
        int(aggregates["rerun_rows"]),
        EXPECTED_2025_FIRST_APPEARANCES,
        "Regression against the Meeting 7 first_1 sample.",
    )
    add_check(
        rows,
        "2025_first_appearance_matches",
        int(aggregates["rerun_unique_matches"]) == EXPECTED_2025_FIRST_MATCHES,
        int(aggregates["rerun_unique_matches"]),
        EXPECTED_2025_FIRST_MATCHES,
        "Two matches contain two first-recorded players.",
    )
    add_check(
        rows,
        "2025_all_first_appearances_are_post_burn_in",
        bool(
            debut.loc[debut["year"].eq(TEST_YEAR), "post_primary_burn_in"].all()
        ),
        int(
            debut.loc[
                debut["year"].eq(TEST_YEAR), "post_primary_burn_in"
            ].sum()
        ),
        EXPECTED_2025_FIRST_APPEARANCES,
        "The test-year first_1 group is not the left-censored model-start cohort.",
    )
    add_check(
        rows,
        "2025_rows_match_step34_keys",
        int(aggregates["matched_rows"]) == EXPECTED_2025_FIRST_APPEARANCES
        and bool(validation["_merge"].eq("both").all()),
        int(aggregates["matched_rows"]),
        EXPECTED_2025_FIRST_APPEARANCES,
        "Match/player/opponent keys agree with the frozen Step 34 dataset.",
    )
    add_check(
        rows,
        "2025_probabilities_match_step34",
        aggregates["max_abs_probability_difference"] < 1e-10,
        aggregates["max_abs_probability_difference"],
        "< 1e-10",
        "Probability convention and chronological model state are unchanged.",
    )
    add_check(
        rows,
        "2025_opponent_ratings_match_step34",
        aggregates["max_abs_opponent_rating_difference"] < 1e-8,
        aggregates["max_abs_opponent_rating_difference"],
        "< 1e-8",
        "Prematch rating state agrees with Meeting 7 within saved-CSV precision.",
    )
    add_check(
        rows,
        "2025_opponent_rds_match_step34",
        aggregates["max_abs_opponent_rd_difference"] < 1e-9,
        aggregates["max_abs_opponent_rd_difference"],
        "< 1e-9",
        "Prematch RD state agrees with Meeting 7.",
    )
    add_check(
        rows,
        "2025_mean_probability_regression",
        abs(aggregates["rerun_mean_p"] - EXPECTED_2025_MEAN_P) < 1e-6,
        aggregates["rerun_mean_p"],
        EXPECTED_2025_MEAN_P,
        "Matches the rounded Meeting 7 headline.",
    )
    add_check(
        rows,
        "2025_win_rate_regression",
        abs(aggregates["rerun_win_rate"] - EXPECTED_2025_WIN_RATE) < 1e-6,
        aggregates["rerun_win_rate"],
        EXPECTED_2025_WIN_RATE,
        "Matches the rounded Meeting 7 headline.",
    )
    add_check(
        rows,
        "2025_brier_regression",
        abs(aggregates["rerun_brier"] - EXPECTED_2025_BRIER) < 1e-6,
        aggregates["rerun_brier"],
        EXPECTED_2025_BRIER,
        "Matches the rounded Meeting 7 headline.",
    )
    add_check(
        rows,
        "2025_mean_opponent_rating_regression",
        abs(
            aggregates["rerun_mean_opponent_rating"]
            - EXPECTED_2025_MEAN_OPPONENT_RATING
        )
        < 1e-3,
        aggregates["rerun_mean_opponent_rating"],
        EXPECTED_2025_MEAN_OPPONENT_RATING,
        "Matches the rounded Meeting 7 initialisation diagnostic.",
    )
    return pd.DataFrame(rows)


def create_figures(
    yearly: pd.DataFrame,
    yearly_debut: pd.DataFrame,
    cohort_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create three meeting-ready diagnostic figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    plt.figure(figsize=(9.2, 5.2))
    plt.plot(
        yearly["year"],
        yearly["median_rating_established_active"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Established active median",
    )
    plt.plot(
        yearly["year"],
        yearly["mean_rating_established_active"],
        linewidth=1.4,
        label="Established active mean",
    )
    plt.axhline(INITIAL_RATING, color="#C44E52", linestyle="--", label="New-player anchor (1500)")
    plt.xlabel("Calendar year")
    plt.ylabel("Glicko rating")
    plt.title("Low-inflation Glicko rating scale among established active players")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(RATING_SCALE_FIGURE_PATH, dpi=200)
    plt.close()
    manifest.append(
        {
            "figure": "41_fig01",
            "path": str(RATING_SCALE_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "description": (
                "Annual mean and median ratings for active players first observed before "
                "that year, compared with the fixed 1500 new-player anchor."
            ),
        }
    )

    post = yearly_debut.loc[yearly_debut["year"].ge(POST_BURN_IN_START_YEAR)].copy()
    plt.figure(figsize=(9.2, 5.2))
    plt.plot(
        post["year"],
        post["mean_opponent_rating"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Mean debut-opponent rating",
    )
    plt.plot(
        post["year"],
        post["median_opponent_rating"],
        linewidth=1.3,
        label="Median debut-opponent rating",
    )
    plt.axhline(INITIAL_RATING, color="#C44E52", linestyle="--", label="New-player anchor (1500)")
    plt.xlabel("First recorded year")
    plt.ylabel("Prematch Glicko rating")
    plt.title("Fixed new-player anchor relative to debut opponents after burn-in")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(DEBUT_ANCHOR_FIGURE_PATH, dpi=200)
    plt.close()
    manifest.append(
        {
            "figure": "41_fig02",
            "path": str(DEBUT_ANCHOR_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "description": (
                "Mean and median opponent rating faced on first recorded appearance "
                "for post-burn-in entry years."
            ),
        }
    )

    plot_groups = [
        "system_start_left_censored",
        "within_5y_burn_in_recorded_entry",
        "post_burn_in_recorded_entry",
        "test_year_recorded_entry",
    ]
    plot = cohort_summary.set_index("group").loc[plot_groups].reset_index()
    labels = ["1985 start", "1986-89 burn-in", "1990-2024 entry", "2025 test entry"]
    x = np.arange(len(plot))
    width = 0.36
    plt.figure(figsize=(9.2, 5.2))
    plt.bar(
        x - width / 2,
        plot["mean_predicted_win_probability"],
        width,
        label="Mean predicted probability",
    )
    plt.bar(
        x + width / 2,
        plot["empirical_win_rate"],
        width,
        label="Empirical win rate",
    )
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("Probability / win rate")
    plt.ylim(0.0, 1.0)
    plt.title("First-recorded prediction bias by entry cohort")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(COHORT_BIAS_FIGURE_PATH, dpi=200)
    plt.close()
    manifest.append(
        {
            "figure": "41_fig03",
            "path": str(COHORT_BIAS_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "description": (
                "Mean current-convention Glicko probability and empirical win rate "
                "for the model-start, burn-in, post-burn-in, and 2025 entry cohorts."
            ),
        }
    )
    return pd.DataFrame(manifest)


def write_summary(
    dataset_path: Path,
    inactivity_unit: str,
    low_variant: dict[str, Any],
    player_entries: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    yearly_debut: pd.DataFrame,
    aggregates: dict[str, float],
    checks: pd.DataFrame,
) -> None:
    """Write a concise dissertation- and meeting-ready interpretation."""

    cohort = cohort_summary.set_index("group")
    start = cohort.loc["system_start_left_censored"]
    within = cohort.loc["within_5y_burn_in_recorded_entry"]
    post = cohort.loc["post_burn_in_recorded_entry"]
    test = cohort.loc["test_year_recorded_entry"]
    first_defined_established_row = yearly.loc[
        yearly["n_established_active_players"].gt(0)
    ].iloc[0]
    final_year_row = yearly.iloc[-1]
    post_yearly = yearly_debut.loc[
        yearly_debut["year"].ge(POST_BURN_IN_START_YEAR)
    ]
    max_anchor_gap = post_yearly.loc[
        post_yearly["mean_focal_minus_opponent_rating"].idxmax()
    ]
    failed = checks.loc[~checks["passed"].astype(bool)]

    lines = [
        "# Meeting 8 Technical Diagnostic: Burn-in, Recorded Entry, and Rating-Scale Drift",
        "",
        "## Scope",
        "",
        "This is a targeted diagnostic, not a new model search. It reuses the frozen low-inflation Glicko configuration and adds the two checks requested after Meeting 7: a defensible recorded-entry definition and an audit of the evolving rating scale.",
        "",
        "## Reused model and data",
        "",
        f"- Full-history data: `{dataset_path.relative_to(PROJECT_ROOT)}`",
        f"- Years: {MODEL_START_YEAR}-{MODEL_END_YEAR}",
        f"- Low-inflation variant: `{low_variant['variant']}`",
        f"- Inactivity unit: {inactivity_unit}",
        f"- C value: {float(low_variant['c_value']):.12f}",
        f"- Fixed new-player state: rating {INITIAL_RATING:.0f}, RD {INITIAL_RD:.0f}",
        "- Probability convention: canonical Player A probability, complemented for Player B, matching the frozen Meeting 7 reporting convention.",
        "",
        "## Operational definitions",
        "",
        f"- **System-start / left-censored player:** first observed in {MODEL_START_YEAR}. The data cannot establish that this was the player's true career debut.",
        f"- **Within-burn-in recorded entry:** first observed during {MODEL_START_YEAR + 1}-{POST_BURN_IN_START_YEAR - 1}. These rows are excluded from the primary post-burn-in definition.",
        f"- **Post-burn-in recorded entry:** first observed from {POST_BURN_IN_START_YEAR} onward after a five-calendar-year burn-in.",
        f"- **2025 test-year recorded entry:** first observed in {TEST_YEAR}. This is a subset of post-burn-in recorded entries and is the source of the frozen first_1 test result.",
        "- The phrase *recorded entry* is deliberate: the available data do not prove true career debut status.",
        "",
        "## Cohort sizes",
        "",
        f"- System-start players: {int(start['n_unique_players']):,}",
        f"- Within five-year burn-in entries: {int(within['n_unique_players']):,}",
        f"- Post-burn-in entries before the test year: {int(post['n_unique_players']):,}",
        f"- 2025 test-year entries: {int(test['n_unique_players']):,}",
        f"- Total classified players: {len(player_entries):,}",
        "",
        "## 2025 first-appearance regression",
        "",
        f"- First appearances: {int(aggregates['rerun_rows'])}",
        f"- Unique matches: {int(aggregates['rerun_unique_matches'])}",
        f"- Mean predicted win probability: {aggregates['rerun_mean_p']:.6f}",
        f"- Empirical win rate: {aggregates['rerun_win_rate']:.6f}",
        f"- Brier score: {aggregates['rerun_brier']:.6f}",
        f"- Mean opponent rating: {aggregates['rerun_mean_opponent_rating']:.3f}",
        f"- Maximum absolute probability difference from Step 34: {aggregates['max_abs_probability_difference']:.3e}",
        "",
        "All 2025 first appearances occur after the five-year burn-in. Therefore, the 2025 first_1 weakness is not an artefact of treating the 1985 model-start population as genuinely new. It remains a limitation for players newly entering the recorded system in the held-out test year.",
        "",
        "## Rating-scale audit",
        "",
        f"- Established-player scale is undefined in {MODEL_START_YEAR}, because every observed player is in the system-start cohort.",
        f"- First defined established-active median ({int(first_defined_established_row['year'])}): {first_defined_established_row['median_rating_established_active']:.3f}",
        f"- Established active median rating in {MODEL_END_YEAR}: {final_year_row['median_rating_established_active']:.3f}",
        f"- Fixed anchor minus established-active median in {MODEL_END_YEAR}: {final_year_row['initial_minus_established_active_median']:.3f}",
        f"- Mean rating across all {int(final_year_row['n_known_players_end_year']):,} known players in {MODEL_END_YEAR}: {final_year_row['mean_rating_all_known']:.3f}",
        f"- Largest annual mean anchor-minus-debut-opponent gap after burn-in: {max_anchor_gap['mean_focal_minus_opponent_rating']:.3f} in {int(max_anchor_gap['year'])}",
        f"- Cumulative net two-player Glicko update change by {MODEL_END_YEAR}: {final_year_row['cumulative_net_glicko_rating_update_change']:.3f}",
        "",
        "The absolute rating level has no standalone substantive meaning because predictions depend on relative ratings. These scale summaries are therefore diagnostic: they show whether the fixed 1500 entry anchor stays aligned with the contemporaneous established-player scale. The direct quantity for the new-player mechanism is the focal-minus-opponent prematch rating gap.",
        "",
        "## What should go into the dissertation",
        "",
        "1. State the four operational definitions above in the methodology or limitations section.",
        "2. Report that the held-out 2025 first_1 sample consists entirely of post-burn-in recorded entrants.",
        "3. Use the annual scale figure only to explain the mechanism; do not interpret 1500 as an absolute skill level.",
        "4. Present the adaptive-K work separately as a short negative result. No additional adaptive-K experiment is required here.",
        "",
        "## Validation",
        "",
        f"- Checks passed: {int(checks['passed'].sum())}/{len(checks)}",
        f"- Failed checks: {len(failed)}",
    ]
    if len(failed):
        lines.extend(
            [
                "",
                "### Failed checks",
                "",
                *[
                    f"- {row.check}: actual={row.actual}, expected={row.expected}"
                    for row in failed.itertuples(index=False)
                ],
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the complete Meeting 8 technical diagnostic."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    step24 = load_step24_module()
    matches, inactivity_unit, dataset_path = step24.load_matches()
    variants = step24.build_variants(inactivity_unit)
    low_variant = next(
        variant for variant in variants if variant["variant"] == "low_inflation"
    )

    player_entries = build_player_entry_table(matches)
    debut, yearly = run_low_inflation_diagnostic(
        matches,
        player_entries,
        step24,
        inactivity_unit,
        low_variant,
    )
    cohort_summary = build_debut_cohort_summary(debut)
    yearly_debut = build_yearly_debut_summary(debut)
    burn_in_sensitivity = build_burn_in_sensitivity(debut)
    test_validation, aggregates = build_2025_validation(debut)
    checks = build_validation_checks(
        matches,
        player_entries,
        debut,
        yearly,
        test_validation,
        aggregates,
        low_variant,
    )
    figure_manifest = create_figures(yearly, yearly_debut, cohort_summary)
    write_summary(
        dataset_path,
        inactivity_unit,
        low_variant,
        player_entries,
        cohort_summary,
        yearly,
        yearly_debut,
        aggregates,
        checks,
    )

    player_entries.to_csv(PLAYER_ENTRY_PATH, index=False, encoding="utf-8-sig")
    debut.to_csv(DEBUT_DETAIL_PATH, index=False, encoding="utf-8-sig")
    cohort_summary.to_csv(DEBUT_COHORT_PATH, index=False, encoding="utf-8-sig")
    yearly.to_csv(YEARLY_SCALE_PATH, index=False, encoding="utf-8-sig")
    yearly_debut.to_csv(YEARLY_DEBUT_PATH, index=False, encoding="utf-8-sig")
    burn_in_sensitivity.to_csv(
        BURN_IN_SENSITIVITY_PATH, index=False, encoding="utf-8-sig"
    )
    test_validation.to_csv(
        TEST_YEAR_VALIDATION_PATH, index=False, encoding="utf-8-sig"
    )
    checks.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")
    figure_manifest.to_csv(
        FIGURE_MANIFEST_PATH, index=False, encoding="utf-8-sig"
    )

    print("\nMeeting 8 Step 41 validation")
    print(checks[["check", "passed", "actual", "expected"]].to_string(index=False))
    print(f"\nChecks passed: {int(checks['passed'].sum())}/{len(checks)}")
    print(f"Summary: {SUMMARY_PATH}")

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise AssertionError(f"Step 41 validation failed: {failed}")


if __name__ == "__main__":
    main()
