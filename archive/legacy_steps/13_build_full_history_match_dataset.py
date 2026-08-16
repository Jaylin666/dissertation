from pathlib import Path
import importlib.util
from typing import Dict

import numpy as np
import pandas as pd


START_YEAR = 1985
END_YEAR = 2025


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"

YEAR_RANGE = f"{START_YEAR}_{END_YEAR}"
MATCHES_OUTPUT_PATH = OUTPUT_DIR / f"matches_{YEAR_RANGE}_checked.csv"
YEARLY_SUMMARY_PATH = OUTPUT_DIR / f"full_history_yearly_check_summary_{YEAR_RANGE}.csv"
OVERALL_SUMMARY_PATH = OUTPUT_DIR / f"full_history_data_check_summary_{YEAR_RANGE}.csv"


def load_step07_module():
    """Load the existing step 07 dataset builder despite its numeric filename."""
    module_path = CODE_DIR / "07_build_multiyear_match_dataset.py"
    spec = importlib.util.spec_from_file_location("step07_build_multiyear", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import step 07 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_required_summary_aliases(yearly_summary: pd.DataFrame) -> pd.DataFrame:
    """Add requested concise column names while preserving step 07 check columns."""
    yearly_summary = yearly_summary.copy()
    if "number_of_unique_players" in yearly_summary.columns:
        yearly_summary["unique_players"] = yearly_summary["number_of_unique_players"]
    elif "unique_players" not in yearly_summary.columns:
        yearly_summary["unique_players"] = np.nan

    if "number_of_unique_events" in yearly_summary.columns:
        yearly_summary["unique_events"] = yearly_summary["number_of_unique_events"]
    elif "unique_events" not in yearly_summary.columns:
        yearly_summary["unique_events"] = np.nan

    preferred = [
        "year",
        "games_rows",
        "events_rows",
        "hidx_rows",
        "matches_rows",
        "duplicated_game_fcode",
        "duplicated_hidx_fcode",
        "missing_event_rows",
        "missing_hidx_rows",
        "missing_winner_names",
        "missing_loser_names",
        "missing_event_date_parsed",
        "unique_players",
        "unique_events",
        "number_of_unique_players",
        "number_of_unique_events",
        "status",
    ]
    existing = [col for col in preferred if col in yearly_summary.columns]
    remaining = [col for col in yearly_summary.columns if col not in existing]
    return yearly_summary[existing + remaining]


def create_full_history_summary(matches: pd.DataFrame, step07_module) -> pd.DataFrame:
    """Create overall checks, extending step 07 with missing event-date counts."""
    overall = step07_module.create_data_check_summary(matches).copy()
    if matches.empty or "event_date_parsed" not in matches.columns:
        missing_event_date_parsed = np.nan if matches.empty else len(matches)
    else:
        missing_event_date_parsed = int(matches["event_date_parsed"].isna().sum())

    if "missing_event_date_parsed" not in overall.columns:
        insert_at = overall.columns.get_loc("missing_hidx_rows") + 1
        overall.insert(insert_at, "missing_event_date_parsed", missing_event_date_parsed)
    else:
        overall["missing_event_date_parsed"] = missing_event_date_parsed
    return overall


def validate_year_coverage(matches: pd.DataFrame, yearly_summary: pd.DataFrame) -> None:
    """Warn if the full-history dataset does not contain every requested year."""
    expected_years = set(range(START_YEAR, END_YEAR + 1))
    actual_years = set(matches["year"].dropna().astype(int).unique()) if not matches.empty else set()
    missing_years = sorted(expected_years - actual_years)
    extra_years = sorted(actual_years - expected_years)

    if missing_years:
        print(f"WARNING: missing years in final matches: {missing_years}")
    if extra_years:
        print(f"WARNING: unexpected years in final matches: {extra_years}")

    if "status" in yearly_summary.columns:
        failed = yearly_summary[yearly_summary["status"].astype(str).ne("ok")]
        if not failed.empty:
            print("WARNING: at least one year did not load with status ok:")
            print(failed[["year", "status"]].to_string(index=False))


def collect_check_flags(overall_summary: pd.DataFrame) -> Dict[str, bool]:
    """Return pass/fail flags for the key duplicate and missing-value checks."""
    row = overall_summary.iloc[0]
    checks = {
        "duplicated_fcode_count_is_zero": int(row["duplicated_fcode_count"]) == 0,
        "missing_event_rows_is_zero": int(row["missing_event_rows"]) == 0,
        "missing_hidx_rows_is_zero": int(row["missing_hidx_rows"]) == 0,
        "missing_winner_names_is_zero": int(row["missing_winner_names"]) == 0,
        "missing_loser_names_is_zero": int(row["missing_loser_names"]) == 0,
        "missing_event_date_parsed_is_zero": int(row["missing_event_date_parsed"]) == 0,
    }
    return checks




def save_outputs(
    matches: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
) -> None:
    """Save all full-history dataset outputs without touching data_processed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_OUTPUT_PATH, index=False)
    yearly_summary.to_csv(YEARLY_SUMMARY_PATH, index=False)
    overall_summary.to_csv(OVERALL_SUMMARY_PATH, index=False)


def print_yearly_counts(yearly_summary: pd.DataFrame) -> None:
    """Print one concise line per year after the reused builder finishes."""
    print("\n=== Yearly Match Counts ===")
    for row in yearly_summary.sort_values("year").itertuples(index=False):
        year = getattr(row, "year")
        matches_rows = getattr(row, "matches_rows", np.nan)
        status = getattr(row, "status", "unknown")
        print(f"{int(year)}: matches={matches_rows}, status={status}")


def print_command_line_summary(overall_summary: pd.DataFrame) -> None:
    """Print final checks and output paths."""
    row = overall_summary.iloc[0]
    print("\n=== Full-history Match Dataset Summary ===")
    print(f"Year range: {int(row['min_year'])}-{int(row['max_year'])}")
    print(f"Total matches: {int(row['total_number_of_matches'])}")
    print(f"Unique players: {int(row['number_of_unique_players'])}")
    print(f"Unique events: {int(row['number_of_unique_events'])}")
    print(f"Duplicated fcode count: {int(row['duplicated_fcode_count'])}")
    print(f"Missing event rows: {int(row['missing_event_rows'])}")
    print(f"Missing hidx rows: {int(row['missing_hidx_rows'])}")
    print(f"Missing winner names: {int(row['missing_winner_names'])}")
    print(f"Missing loser names: {int(row['missing_loser_names'])}")
    print(f"Missing parsed event dates: {int(row['missing_event_date_parsed'])}")
    print("Output paths:")
    print(f"  matches: {MATCHES_OUTPUT_PATH}")
    print(f"  yearly summary: {YEARLY_SUMMARY_PATH}")
    print(f"  overall summary: {OVERALL_SUMMARY_PATH}")


def main() -> None:
    print("=== Building full-history checked match-level dataset ===")
    print(f"Requested years: {START_YEAR}-{END_YEAR}")
    print("Reusing code/07_build_multiyear_match_dataset.py")

    step07_module = load_step07_module()
    matches, yearly_summary = step07_module.build_multiyear_dataset(START_YEAR, END_YEAR)
    yearly_summary = add_required_summary_aliases(yearly_summary)
    overall_summary = create_full_history_summary(matches, step07_module)

    sort_cols = [
        col
        for col in ["year", "event_date_parsed", "event", "code", "fcode"]
        if col in matches.columns
    ]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    validate_year_coverage(matches, yearly_summary)
    save_outputs(matches, yearly_summary, overall_summary)
    print_yearly_counts(yearly_summary)
    print_command_line_summary(overall_summary)


if __name__ == "__main__":
    main()
