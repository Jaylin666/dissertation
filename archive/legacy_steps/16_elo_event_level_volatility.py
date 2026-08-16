"""
This script compares Elo rating volatility at match resolution and event resolution.

The aim is not to re-select the best K or evaluate prediction accuracy. Instead,
it quantifies how large match-by-match updates are, how large net event-level
rating changes are, and whether an aggressive K looks different at event
resolution than at match resolution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd


BURNIN_START_YEAR = 1985
OPTIONAL_CANDIDATE_START_YEAR = 2005
RUN_OPTIONAL_CANDIDATE_START = False
END_YEAR = 2025
INITIAL_RATING = 1500.0

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
    {
        "setting_name": "aggressive_k35_scale300",
        "k": 35.0,
        "scale": 300.0,
        "label": "Nearby aggressive Elo",
    },
    {
        "setting_name": "aggressive_k40_scale400",
        "k": 40.0,
        "scale": 400.0,
        "label": "Very aggressive Elo",
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

PLAYER_EVENT_PATH = OUTPUT_DIR / "elo_event_level_volatility_player_event.csv"
MATCH_SUMMARY_PATH = OUTPUT_DIR / "elo_event_level_volatility_match_summary.csv"
EVENT_SUMMARY_PATH = OUTPUT_DIR / "elo_event_level_volatility_event_summary.csv"
EVENT_SIZE_SUMMARY_PATH = OUTPUT_DIR / "elo_event_level_volatility_by_event_size.csv"
EXAMPLES_PATH = OUTPUT_DIR / "elo_event_level_volatility_examples.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "elo_event_level_volatility_summary.md"


REQUIRED_COLUMNS = ["fcode", "code", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = [
    "event_fcode",
    "eventname",
    "event_date_raw",
    "event_date_parsed",
    "winner_name",
    "loser_name",
]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "event_fcode", "winner", "loser"]


def get_start_years() -> List[int]:
    """Return burn-in start years for this volatility analysis."""
    start_years = [BURNIN_START_YEAR]
    if RUN_OPTIONAL_CANDIDATE_START and OPTIONAL_CANDIDATE_START_YEAR not in start_years:
        start_years.append(OPTIONAL_CANDIDATE_START_YEAR)
    return start_years


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load full-history matches and prepare event ordering columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/13_build_full_history_match_dataset.py first."
        )

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if col in header]
    missing_required = [col for col in REQUIRED_COLUMNS if col not in usecols]
    if missing_required:
        raise ValueError(f"{path.name} is missing required columns: {missing_required}")

    matches = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in OPTIONAL_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches["event_date_parsed"] = pd.to_datetime(matches["event_date_parsed"], errors="coerce")
    matches = add_event_ordering_columns(matches)
    matches = add_event_key(matches)

    sort_cols = ["year", "event_order_date", "event", "code", "fcode"]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    print(f"Loaded dataset: {path}")
    print(f"Rows: {len(matches)}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    return matches


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add event_order_date and event_date_ordering_method without changing raw dates."""
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


def format_id(value: object) -> str:
    """Format numeric IDs without a trailing .0 where possible."""
    if pd.isna(value):
        return "missing"
    return str(int(value))


def add_event_key(matches: pd.DataFrame) -> pd.DataFrame:
    """Create a year_event key because event codes are only year-local."""
    matches = matches.copy()
    matches["event_key"] = matches.apply(
        lambda row: f"{format_id(row['year'])}_{format_id(row['event'])}", axis=1
    )
    return matches


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def ensure_player(
    player_code: int,
    ratings: Dict[int, float],
    player_names: Dict[int, str],
    possible_name: object,
    initial_rating: float,
) -> None:
    """Initialise a player rating and store first available name."""
    if player_code not in ratings:
        ratings[player_code] = initial_rating
    if player_code not in player_names and pd.notna(possible_name) and str(possible_name).strip():
        player_names[player_code] = str(possible_name).strip()


def update_player_event_record(
    player_events: Dict[Tuple[int, str, int], Dict[str, object]],
    row: object,
    setting_name: str,
    k: float,
    scale: float,
    start_year: int,
    end_year: int,
    player_code: int,
    player_name: object,
    rating_before: float,
    rating_after: float,
    result: int,
    abs_update: float,
) -> None:
    """Accumulate one player appearance into a player-event record."""
    year = int(getattr(row, "year"))
    event_key = getattr(row, "event_key")
    key = (year, event_key, player_code)

    if key not in player_events:
        player_events[key] = {
            "setting_name": setting_name,
            "k": k,
            "scale": scale,
            "start_year": start_year,
            "end_year": end_year,
            "year": year,
            "event": getattr(row, "event", pd.NA),
            "event_key": event_key,
            "event_fcode": getattr(row, "event_fcode", pd.NA),
            "eventname": getattr(row, "eventname", pd.NA),
            "event_order_date": getattr(row, "event_order_date", pd.NaT),
            "event_date_ordering_method": getattr(row, "event_date_ordering_method", pd.NA),
            "player_code": player_code,
            "player_name": player_name,
            "games_in_event": 0,
            "wins_in_event": 0,
            "losses_in_event": 0,
            "rating_before_first_game_in_event": rating_before,
            "rating_after_last_game_in_event": rating_after,
            "cumulative_abs_match_updates_in_event": 0.0,
            "max_abs_match_update_in_event": 0.0,
        }

    record = player_events[key]
    if pd.isna(record.get("player_name")) and pd.notna(player_name):
        record["player_name"] = player_name
    record["games_in_event"] = int(record["games_in_event"]) + 1
    record["wins_in_event"] = int(record["wins_in_event"]) + int(result == 1)
    record["losses_in_event"] = int(record["losses_in_event"]) + int(result == 0)
    record["rating_after_last_game_in_event"] = rating_after
    record["cumulative_abs_match_updates_in_event"] = (
        float(record["cumulative_abs_match_updates_in_event"]) + abs_update
    )
    record["max_abs_match_update_in_event"] = max(
        float(record["max_abs_match_update_in_event"]), abs_update
    )


def run_elo_collect_event_volatility(
    matches: pd.DataFrame,
    start_year: int,
    end_year: int,
    k: float,
    scale: float,
    setting_name: str,
    initial_rating: float = INITIAL_RATING,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run Elo and collect compact match-level and player-event volatility."""
    run_matches = matches[(matches["year"] >= start_year) & (matches["year"] <= end_year)].copy()
    ratings: Dict[int, float] = {}
    player_names: Dict[int, str] = {}
    player_events: Dict[Tuple[int, str, int], Dict[str, object]] = {}
    abs_match_updates: List[float] = []
    skipped_games = 0

    for row in run_matches.itertuples(index=False):
        winner_value = getattr(row, "winner")
        loser_value = getattr(row, "loser")
        if pd.isna(winner_value) or pd.isna(loser_value):
            skipped_games += 1
            continue

        winner = int(winner_value)
        loser = int(loser_value)
        ensure_player(winner, ratings, player_names, getattr(row, "winner_name", pd.NA), initial_rating)
        ensure_player(loser, ratings, player_names, getattr(row, "loser_name", pd.NA), initial_rating)

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, scale)
        rating_change = k * (1.0 - pred_winner_win)
        abs_update = abs(rating_change)

        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change
        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after
        abs_match_updates.append(abs_update)

        update_player_event_record(
            player_events=player_events,
            row=row,
            setting_name=setting_name,
            k=k,
            scale=scale,
            start_year=start_year,
            end_year=end_year,
            player_code=winner,
            player_name=player_names.get(winner, pd.NA),
            rating_before=winner_rating_before,
            rating_after=winner_rating_after,
            result=1,
            abs_update=abs_update,
        )
        update_player_event_record(
            player_events=player_events,
            row=row,
            setting_name=setting_name,
            k=k,
            scale=scale,
            start_year=start_year,
            end_year=end_year,
            player_code=loser,
            player_name=player_names.get(loser, pd.NA),
            rating_before=loser_rating_before,
            rating_after=loser_rating_after,
            result=0,
            abs_update=abs_update,
        )

    player_event = pd.DataFrame(player_events.values())
    if not player_event.empty:
        player_event["net_rating_change_in_event"] = (
            player_event["rating_after_last_game_in_event"]
            - player_event["rating_before_first_game_in_event"]
        )
        player_event["abs_net_rating_change_in_event"] = player_event[
            "net_rating_change_in_event"
        ].abs()
        player_event["mean_abs_match_update_in_event"] = (
            player_event["cumulative_abs_match_updates_in_event"]
            / player_event["games_in_event"]
        )
        player_event["event_cancellation_ratio"] = np.where(
            player_event["cumulative_abs_match_updates_in_event"] > 0,
            player_event["abs_net_rating_change_in_event"]
            / player_event["cumulative_abs_match_updates_in_event"],
            np.nan,
        )
        player_event["games_in_event_bucket"] = player_event["games_in_event"].apply(
            make_games_bucket
        )

    match_summary = make_match_summary(
        abs_match_updates=abs_match_updates,
        setting_name=setting_name,
        k=k,
        scale=scale,
        start_year=start_year,
        end_year=end_year,
        skipped_games=skipped_games,
    )
    return player_event, match_summary


def make_match_summary(
    abs_match_updates: List[float],
    setting_name: str,
    k: float,
    scale: float,
    start_year: int,
    end_year: int,
    skipped_games: int,
) -> Dict[str, object]:
    """Create match-level absolute update summary for one setting."""
    updates = pd.Series(abs_match_updates, dtype=float)
    return {
        "setting_name": setting_name,
        "k": k,
        "scale": scale,
        "start_year": start_year,
        "end_year": end_year,
        "number_of_matches": len(updates),
        "skipped_games": skipped_games,
        "mean_abs_match_update": updates.mean(),
        "median_abs_match_update": updates.median(),
        "p75_abs_match_update": updates.quantile(0.75),
        "p90_abs_match_update": updates.quantile(0.90),
        "p95_abs_match_update": updates.quantile(0.95),
        "p99_abs_match_update": updates.quantile(0.99),
        "max_abs_match_update": updates.max(),
    }


def make_games_bucket(games_in_event: int) -> str:
    """Bucket games per player-event."""
    if games_in_event == 1:
        return "1 game"
    if games_in_event == 2:
        return "2 games"
    if 3 <= games_in_event <= 4:
        return "3-4 games"
    if 5 <= games_in_event <= 9:
        return "5-9 games"
    return "10+ games"


def q(series: pd.Series, quantile: float) -> float:
    """Small quantile helper."""
    return series.quantile(quantile)


def summarise_player_events(player_event: pd.DataFrame) -> Dict[str, object]:
    """Summarise event-level net volatility for one setting."""
    return {
        "setting_name": player_event["setting_name"].iloc[0],
        "k": player_event["k"].iloc[0],
        "scale": player_event["scale"].iloc[0],
        "start_year": player_event["start_year"].iloc[0],
        "end_year": player_event["end_year"].iloc[0],
        "number_of_player_event_records": len(player_event),
        "mean_games_per_player_event": player_event["games_in_event"].mean(),
        "median_games_per_player_event": player_event["games_in_event"].median(),
        "mean_abs_event_net_change": player_event["abs_net_rating_change_in_event"].mean(),
        "median_abs_event_net_change": player_event["abs_net_rating_change_in_event"].median(),
        "p75_abs_event_net_change": q(player_event["abs_net_rating_change_in_event"], 0.75),
        "p90_abs_event_net_change": q(player_event["abs_net_rating_change_in_event"], 0.90),
        "p95_abs_event_net_change": q(player_event["abs_net_rating_change_in_event"], 0.95),
        "p99_abs_event_net_change": q(player_event["abs_net_rating_change_in_event"], 0.99),
        "max_abs_event_net_change": player_event["abs_net_rating_change_in_event"].max(),
        "mean_cumulative_abs_match_updates_in_event": player_event[
            "cumulative_abs_match_updates_in_event"
        ].mean(),
        "median_cumulative_abs_match_updates_in_event": player_event[
            "cumulative_abs_match_updates_in_event"
        ].median(),
        "p90_cumulative_abs_match_updates_in_event": q(
            player_event["cumulative_abs_match_updates_in_event"], 0.90
        ),
        "mean_event_cancellation_ratio": player_event["event_cancellation_ratio"].mean(),
        "median_event_cancellation_ratio": player_event["event_cancellation_ratio"].median(),
    }


def summarise_by_event_size(player_event: pd.DataFrame) -> pd.DataFrame:
    """Summarise event-level volatility by games-in-event bucket."""
    bucket_order = ["1 game", "2 games", "3-4 games", "5-9 games", "10+ games"]
    rows = []
    for bucket in bucket_order:
        subset = player_event[player_event["games_in_event_bucket"] == bucket]
        if subset.empty:
            continue
        rows.append(
            {
                "setting_name": subset["setting_name"].iloc[0],
                "k": subset["k"].iloc[0],
                "scale": subset["scale"].iloc[0],
                "start_year": subset["start_year"].iloc[0],
                "end_year": subset["end_year"].iloc[0],
                "games_in_event_bucket": bucket,
                "player_event_records": len(subset),
                "mean_abs_event_net_change": subset["abs_net_rating_change_in_event"].mean(),
                "median_abs_event_net_change": subset["abs_net_rating_change_in_event"].median(),
                "p90_abs_event_net_change": q(subset["abs_net_rating_change_in_event"], 0.90),
                "mean_cumulative_abs_match_updates": subset[
                    "cumulative_abs_match_updates_in_event"
                ].mean(),
                "mean_event_cancellation_ratio": subset["event_cancellation_ratio"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_examples(player_event: pd.DataFrame) -> pd.DataFrame:
    """Create small examples table for meeting inspection."""
    example_frames = []
    cols = [
        "setting_name",
        "k",
        "scale",
        "start_year",
        "end_year",
        "year",
        "event",
        "event_key",
        "eventname",
        "event_order_date",
        "player_code",
        "player_name",
        "games_in_event",
        "wins_in_event",
        "losses_in_event",
        "rating_before_first_game_in_event",
        "rating_after_last_game_in_event",
        "net_rating_change_in_event",
        "abs_net_rating_change_in_event",
        "cumulative_abs_match_updates_in_event",
        "event_cancellation_ratio",
    ]

    top_net = player_event.nlargest(20, "abs_net_rating_change_in_event")[cols].copy()
    top_net.insert(0, "example_type", "largest_abs_net_rating_change")
    example_frames.append(top_net)

    top_cumulative = player_event.nlargest(20, "cumulative_abs_match_updates_in_event")[cols].copy()
    top_cumulative.insert(0, "example_type", "largest_cumulative_abs_match_updates")
    example_frames.append(top_cumulative)

    cancellation_candidates = player_event[
        (player_event["games_in_event"] >= 5)
        & (player_event["event_cancellation_ratio"].notna())
    ].copy()
    low_cancel = cancellation_candidates.nsmallest(20, "event_cancellation_ratio")[cols].copy()
    low_cancel.insert(0, "example_type", "lowest_cancellation_ratio_games_ge5")
    example_frames.append(low_cancel)

    return pd.concat(example_frames, ignore_index=True)


def append_player_event(player_event: pd.DataFrame, first_write: bool) -> None:
    """Append one setting's player-event table."""
    player_event.to_csv(
        PLAYER_EVENT_PATH,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )


def remove_existing_outputs() -> None:
    """Remove this script's own outputs before a fresh run."""
    for path in [
        PLAYER_EVENT_PATH,
        MATCH_SUMMARY_PATH,
        EVENT_SUMMARY_PATH,
        EVENT_SIZE_SUMMARY_PATH,
        EXAMPLES_PATH,
        SUMMARY_MD_PATH,
    ]:
        if path.exists():
            path.unlink()


def write_markdown_summary(
    matches: pd.DataFrame,
    match_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_size_summary: pd.DataFrame,
    output_path: Path,
) -> str:
    """Write a meeting-ready markdown summary."""
    total_matches = len(matches[(matches["year"] >= BURNIN_START_YEAR) & (matches["year"] <= END_YEAR)])
    unique_events = matches[
        (matches["year"] >= BURNIN_START_YEAR) & (matches["year"] <= END_YEAR)
    ][["year", "event"]].drop_duplicates().shape[0]

    setting_lines = [
        f"* {setting['label']}: `{setting['setting_name']}`, K={setting['k']:g}, scale={setting['scale']:g}"
        for setting in ELO_SETTINGS
    ]

    match_lines = []
    for _, row in match_summary.sort_values(["k", "scale"]).iterrows():
        match_lines.append(
            f"* {row['setting_name']}: mean abs match update {row['mean_abs_match_update']:.3f}, "
            f"p95 {row['p95_abs_match_update']:.3f}, p99 {row['p99_abs_match_update']:.3f}, "
            f"max {row['max_abs_match_update']:.3f}."
        )

    event_lines = []
    for _, row in event_summary.sort_values(["k", "scale"]).iterrows():
        event_lines.append(
            f"* {row['setting_name']}: mean abs event net change {row['mean_abs_event_net_change']:.3f}, "
            f"p95 {row['p95_abs_event_net_change']:.3f}, "
            f"mean cumulative match movement {row['mean_cumulative_abs_match_updates_in_event']:.3f}, "
            f"mean cancellation ratio {row['mean_event_cancellation_ratio']:.3f}."
        )

    interpretation_lines = []
    merged = event_summary.merge(
        match_summary[["setting_name", "mean_abs_match_update"]],
        on="setting_name",
        how="left",
    )
    for _, row in merged.sort_values(["k", "scale"]).iterrows():
        interpretation_lines.append(
            f"* {row['setting_name']}: mean event net change is "
            f"{row['mean_abs_event_net_change']:.3f}, while the mean cumulative absolute "
            f"within-event movement is {row['mean_cumulative_abs_match_updates_in_event']:.3f}. "
            f"The mean cancellation ratio is {row['mean_event_cancellation_ratio']:.3f}."
        )

    size_lines = []
    focus_setting = "validation_best_k30_scale300"
    focus = event_size_summary[event_size_summary["setting_name"] == focus_setting]
    if focus.empty:
        focus = event_size_summary
    for _, row in focus.iterrows():
        size_lines.append(
            f"* {row['setting_name']}, {row['games_in_event_bucket']}: "
            f"records {int(row['player_event_records'])}, "
            f"mean abs net {row['mean_abs_event_net_change']:.3f}, "
            f"mean cumulative movement {row['mean_cumulative_abs_match_updates']:.3f}, "
            f"mean cancellation ratio {row['mean_event_cancellation_ratio']:.3f}."
        )

    markdown = f"""# Elo event-level volatility analysis

## 1. Aim of this experiment

This experiment responds to the supervisor's point about match-by-match versus tournament/event resolution.
Croquet data is naturally match-level, but rating systems such as Elo are often interpreted over a broader event or tournament context.
The aim is to compare Elo volatility at match resolution and event resolution.

## 2. Data and period used

Input file: `outputs/elo_optimization/matches_1985_2025_checked.csv`.

* Burn-in period: {BURNIN_START_YEAR}-{END_YEAR}
* Total matches used per setting: {total_matches}
* Unique year-event records: {unique_events}

The script reruns Elo directly from the checked full-history dataset and does not read the large update-history file from the burn-in stability experiment.

## 3. Elo settings tested

{chr(10).join(setting_lines)}

## 4. Match-level volatility

The match-level summary records each match once because the winner and loser have the same absolute update size.

{chr(10).join(match_lines)}

## 5. Event-level volatility

The event-level table groups by player within year-event.
For each player-event, net event change is the rating after the player's last game in the event minus the rating before the player's first game in the event.

{chr(10).join(event_lines)}

## 6. Match-level versus event-level interpretation

`cumulative_abs_match_updates_in_event` adds all within-event absolute updates.
`abs_net_rating_change_in_event` measures the event-level net movement.
If a player wins and loses within the same event, these match-level movements can cancel.
Therefore, event resolution can look smoother than match-by-match resolution.

{chr(10).join(interpretation_lines)}

## 7. Effect of K and scale

Larger or more aggressive settings generally increase both match-level updates and event-level net changes.
However, the cancellation ratio shows that event-level interpretation is not identical to summing match-level movement.
The aggressive settings should therefore be assessed alongside stability diagnostics, not only prediction metrics.

## 8. Event size effect

The table below focuses on the validation-best setting when available.
It shows how games per player-event changes the relationship between cumulative movement and net event change.

{chr(10).join(size_lines)}

## 9. Implication for Elo baseline

This experiment does not directly choose the final Elo parameters.
It helps explain why K=30 can look volatile when inspected after every match, while event-level net changes may provide a smoother and more interpretable view.
For the final Elo baseline, prediction performance should be reported together with rating stability and volatility diagnostics.

## 10. Next step

The next step is the Elo baseline decision summary, combining burn-in stability, single-year rerun convergence, event-level volatility and previous validation results before moving to Glicko comparison.
"""
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs()

    print("=== Elo event-level volatility analysis ===")
    print(f"Burn-in start year: {BURNIN_START_YEAR}")
    print(f"End year: {END_YEAR}")
    print(f"RUN_OPTIONAL_CANDIDATE_START: {RUN_OPTIONAL_CANDIDATE_START}")
    print("Elo settings:")
    for setting in ELO_SETTINGS:
        print(f"  {setting['setting_name']}: K={setting['k']:g}, scale={setting['scale']:g}")

    matches = load_matches()
    start_years = get_start_years()

    match_summary_rows = []
    event_summary_rows = []
    size_summary_frames = []
    example_frames = []
    first_player_event_write = True

    for start_year in start_years:
        for setting in ELO_SETTINGS:
            setting_name = setting["setting_name"]
            k = float(setting["k"])
            scale = float(setting["scale"])
            print(f"\nRunning setting={setting_name}, start_year={start_year}, end_year={END_YEAR}")
            run_start = time.time()
            player_event, match_summary = run_elo_collect_event_volatility(
                matches=matches,
                start_year=start_year,
                end_year=END_YEAR,
                k=k,
                scale=scale,
                setting_name=setting_name,
                initial_rating=INITIAL_RATING,
            )
            append_player_event(player_event, first_write=first_player_event_write)
            first_player_event_write = False

            match_summary_rows.append(match_summary)
            event_summary_rows.append(summarise_player_events(player_event))
            size_summary_frames.append(summarise_by_event_size(player_event))
            example_frames.append(make_examples(player_event))

            print(f"  matches: {match_summary['number_of_matches']}")
            print(f"  player-event records: {len(player_event)}")
            print(f"  elapsed: {time.time() - run_start:.1f}s")

    match_summary_df = pd.DataFrame(match_summary_rows)
    event_summary_df = pd.DataFrame(event_summary_rows)
    size_summary_df = pd.concat(size_summary_frames, ignore_index=True)
    examples_df = pd.concat(example_frames, ignore_index=True)

    match_summary_df.to_csv(MATCH_SUMMARY_PATH, index=False)
    event_summary_df.to_csv(EVENT_SUMMARY_PATH, index=False)
    size_summary_df.to_csv(EVENT_SIZE_SUMMARY_PATH, index=False)
    examples_df.to_csv(EXAMPLES_PATH, index=False)
    write_markdown_summary(matches, match_summary_df, event_summary_df, size_summary_df, SUMMARY_MD_PATH)

    print("\nOutput paths:")
    print(f"  player-event table: {PLAYER_EVENT_PATH}")
    print(f"  match-level summary: {MATCH_SUMMARY_PATH}")
    print(f"  event-level summary: {EVENT_SUMMARY_PATH}")
    print(f"  event-size summary: {EVENT_SIZE_SUMMARY_PATH}")
    print(f"  examples: {EXAMPLES_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
