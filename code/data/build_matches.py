"""Build the checked historical game dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from code.config import (
    EXPECTED_FULL_HISTORY_MATCHES,
    EXPECTED_TEST_MATCHES,
    EXPECTED_UNIQUE_PLAYERS,
    FULL_HISTORY_END_YEAR,
    FULL_HISTORY_START_YEAR,
    TEST_YEAR,
)


START_YEAR = FULL_HISTORY_START_YEAR
END_YEAR = FULL_HISTORY_END_YEAR


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / "data_raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"

MATCHES_OUTPUT_PATH = OUTPUT_DIR / f"matches_{START_YEAR}_{END_YEAR}_checked.csv"
YEARLY_SUMMARY_PATH = (
    OUTPUT_DIR / f"full_history_yearly_check_summary_{START_YEAR}_{END_YEAR}.csv"
)
OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR / f"full_history_data_check_summary_{START_YEAR}_{END_YEAR}.csv"
)


GAME_COLS = ["fcode", "code", "year", "event", "winner", "loser", "country", "tp"]

EVENT_COLS = [
    "fcode",
    "code",
    "year",
    "eventname",
    "date",
    "e_class",
    "e_games",
    "game_start",
    "game_end",
    "country",
    "players",
    "avdg",
    "winner",
    "super",
]

HIDX_COLS = [
    "fcode",
    "code",
    "year",
    "idx_win",
    "idx_lose",
    "grd_win",
    "grd_lose",
    "dgrd_win",
    "dgrd_lose",
    "pt_win",
    "pt_lose",
    "idx20_win",
    "idx20_lose",
    "cinc",
    "rdgrd_win",
    "rdgrd_lose",
    "rpt_win",
    "rpt_lose",
    "idgw",
    "idgl",
]

NAME_COLS = [
    "code",
    "surname",
    "initials",
    "mf",
    "country",
    "grade",
    "firstname",
    "dead",
    "oldname",
    "state",
    "country2",
    "grip",
    "birthy",
    "birthm",
    "birthd",
    "deathy",
    "deathm",
    "deathd",
    "namechange",
    "rebased",
    "newgrade",
    "altfirstname",
    "oldfirstname",
    "firstnamechange",
]

COUNTRY_COLS = ["country_code", "country_name", "country_symbol", "nationality"]

HIDX_FIELDS = [col for col in HIDX_COLS if col not in {"fcode", "code", "year"}]

IMPORTANT_OUTPUT_COLUMNS = [
    "fcode",
    "code",
    "year",
    "event",
    "event_fcode",
    "eventname",
    "event_date_raw",
    "event_date_parsed",
    "country",
    "country_name",
    "winner",
    "loser",
    "winner_name",
    "loser_name",
    "idx_win",
    "idx_lose",
    "grd_win",
    "grd_lose",
    "dgrd_win",
    "dgrd_lose",
    "pt_win",
    "pt_lose",
    "cinc",
    "rdgrd_win",
    "rdgrd_lose",
    "rpt_win",
    "rpt_lose",
]


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].mask(df[col].eq(""), pd.NA)
    return df


def assign_columns_safely(
    df: pd.DataFrame, columns: List[str], file_label: str
) -> pd.DataFrame:
    df = df.copy()
    actual_count = df.shape[1]
    expected_count = len(columns)

    if actual_count == expected_count:
        df.columns = columns
        return df

    print(
        f"WARNING: {file_label} has {actual_count} columns; "
        f"expected {expected_count}."
    )

    if actual_count > expected_count:
        extra_cols = [f"extra_{i}" for i in range(1, actual_count - expected_count + 1)]
        df.columns = columns + extra_cols
    else:
        df.columns = columns[:actual_count]
        for missing_col in columns[actual_count:]:
            print(f"WARNING: {file_label} missing column {missing_col}; filling with NA.")
            df[missing_col] = pd.NA

    return df


def load_csv_no_header(path: Path, columns: List[str], file_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{file_label}: file not found: {path}")

    errors = []
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(
                path,
                header=None,
                dtype="string",
                keep_default_na=False,
                skipinitialspace=True,
                encoding=encoding,
            )
            df = clean_strings(df)
            df = assign_columns_safely(df, columns, file_label)
            print(f"{file_label}: loaded {path.name}, shape={df.shape}")
            return df
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"{file_label}: could not read {path}\n{joined_errors}")


def to_numeric_id(df: pd.DataFrame, columns: List[str], file_label: str = "table") -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            print(f"WARNING: {file_label} has no column {col}; cannot convert to numeric.")
            continue

        missing_before = df[col].isna().sum()
        converted = pd.to_numeric(df[col], errors="coerce")
        missing_after = converted.isna().sum()
        newly_missing = int(missing_after - missing_before)
        if newly_missing > 0:
            print(
                f"WARNING: {file_label}.{col} created {newly_missing} missing values "
                "during numeric conversion."
            )
        df[col] = converted.astype("Int64")
    return df


def to_numeric_fields(df: pd.DataFrame, columns: List[str], file_label: str) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"WARNING: optional field {file_label}.{col} is missing; filling with NaN.")
            df[col] = np.nan
    return df


def make_player_name(row: pd.Series) -> object:
    surname = "" if pd.isna(row.get("surname")) else str(row.get("surname")).strip()
    firstname = "" if pd.isna(row.get("firstname")) else str(row.get("firstname")).strip()
    initials = "" if pd.isna(row.get("initials")) else str(row.get("initials")).strip()

    if firstname and surname:
        return f"{firstname} {surname}"
    if initials and surname:
        return f"{initials} {surname}"
    if surname:
        return surname
    return pd.NA


def load_names() -> pd.DataFrame:
    names = load_csv_no_header(DATA_RAW / "names.dat", NAME_COLS, "names")
    names = to_numeric_id(names, ["code"], "names")

    for col in ["surname", "firstname", "initials", "country"]:
        if col not in names.columns:
            print(f"WARNING: names missing {col}; filling with NA.")
            names[col] = pd.NA

    names["player_name"] = names.apply(make_player_name, axis=1)
    lookup = names[["code", "surname", "firstname", "initials", "country", "player_name"]].copy()
    duplicate_codes = lookup["code"].duplicated().sum()
    if duplicate_codes > 0:
        print(f"WARNING: names has {duplicate_codes} duplicate player codes; keeping first.")
        lookup = lookup.drop_duplicates("code", keep="first")
    return lookup


def load_country() -> pd.DataFrame:
    country = load_csv_no_header(DATA_RAW / "country.csv", COUNTRY_COLS, "country")
    country = to_numeric_id(country, ["country_code"], "country")

    for col in COUNTRY_COLS:
        if col not in country.columns:
            print(f"WARNING: country missing {col}; filling with NA.")
            country[col] = pd.NA

    lookup = country[["country_code", "country_name", "country_symbol", "nationality"]].copy()
    duplicate_codes = lookup["country_code"].duplicated().sum()
    if duplicate_codes > 0:
        print(f"WARNING: country has {duplicate_codes} duplicate country codes; keeping first.")
        lookup = lookup.drop_duplicates("country_code", keep="first")
    return lookup


def parse_event_dates(event_date_raw: pd.Series) -> pd.Series:
    """Parse a sortable date from each event date field when possible."""
    raw = event_date_raw.astype("string")

    # Date ranges retain one complete date for ordering; unmatched rows remain.
    extracted = raw.str.extract(r"(\d{1,2}\.\d{1,2}\.\d{2,4})", expand=False)
    parsed_from_extracted = parse_dot_dates(extracted)
    parsed_direct = parse_dot_dates(raw)
    return parsed_direct.fillna(parsed_from_extracted)


def parse_dot_dates(date_text: pd.Series) -> pd.Series:
    """Parse dates written with dot separators."""
    parsed = pd.Series(pd.NaT, index=date_text.index, dtype="datetime64[ns]")
    for date_format in ("%d.%m.%y", "%d.%m.%Y"):
        parsed = parsed.fillna(pd.to_datetime(date_text, format=date_format, errors="coerce"))
    return parsed


def prepare_events_for_merge(events: pd.DataFrame) -> pd.DataFrame:
    events = events.rename(
        columns={
            "fcode": "event_fcode",
            "code": "event_code",
            "date": "event_date_raw",
            "country": "event_country",
            "winner": "event_winner",
        }
    )
    events["event_date_parsed"] = parse_event_dates(events["event_date_raw"])
    return events


def prepare_player_lookup(names: pd.DataFrame, prefix: str) -> pd.DataFrame:
    lookup = names[["code", "player_name", "country"]].copy()
    lookup = lookup.rename(
        columns={
            "code": prefix,
            "player_name": f"{prefix}_name",
            "country": f"{prefix}_country",
        }
    )
    return lookup


def count_missing(series: pd.Series) -> int:
    cleaned = series.fillna("").astype(str).str.strip()
    return int(cleaned.eq("").sum())


def calculate_year_summary(
    year: int,
    games: pd.DataFrame,
    events: pd.DataFrame,
    hidx: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    missing_hidx_rows = int(matches[HIDX_FIELDS].isna().all(axis=1).sum())
    unique_players = pd.concat([matches["winner"], matches["loser"]]).dropna().nunique()

    summary = {
        "year": year,
        "games_rows": len(games),
        "events_rows": len(events),
        "hidx_rows": len(hidx),
        "matches_rows": len(matches),
        "duplicated_game_fcode": int(games["fcode"].duplicated().sum()),
        "duplicated_hidx_fcode": int(hidx["fcode"].duplicated().sum()),
        "missing_event_rows": count_missing(matches["eventname"]),
        "missing_hidx_rows": missing_hidx_rows,
        "missing_winner_names": count_missing(matches["winner_name"]),
        "missing_loser_names": count_missing(matches["loser_name"]),
        "missing_event_date_parsed": int(matches["event_date_parsed"].isna().sum()),
        "number_of_unique_players": int(unique_players),
        "number_of_unique_events": int(matches[["year", "event"]].drop_duplicates().shape[0]),
        "status": "ok",
    }
    return pd.DataFrame([summary])


def ensure_important_columns(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    for col in IMPORTANT_OUTPUT_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: final matches missing optional column {col}; filling with NaN.")
            matches[col] = np.nan
    return matches


def order_output_columns(matches: pd.DataFrame) -> pd.DataFrame:
    matches = ensure_important_columns(matches)
    first_cols = [col for col in IMPORTANT_OUTPUT_COLUMNS if col in matches.columns]
    remaining_cols = [col for col in matches.columns if col not in first_cols]
    return matches[first_cols + remaining_cols]


def load_one_year(
    year: int,
    names: Optional[pd.DataFrame] = None,
    country: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine one year of source data."""
    if names is None:
        names = load_names()
    if country is None:
        country = load_country()

    games = load_csv_no_header(DATA_RAW / f"game{year}.csv", GAME_COLS, f"game{year}")
    events = load_csv_no_header(DATA_RAW / f"evnt{year}.csv", EVENT_COLS, f"evnt{year}")
    hidx = load_csv_no_header(DATA_RAW / f"hidx{year}.csv", HIDX_COLS, f"hidx{year}")

    games = to_numeric_id(
        games,
        ["fcode", "code", "year", "event", "winner", "loser", "country"],
        f"game{year}",
    )
    events = to_numeric_id(
        events,
        ["fcode", "code", "year", "country", "winner"],
        f"evnt{year}",
    )
    hidx = to_numeric_id(hidx, ["fcode", "code", "year"], f"hidx{year}")
    hidx = to_numeric_fields(hidx, HIDX_FIELDS, f"hidx{year}")

    for df, label in [(games, "games"), (events, "events"), (hidx, "hidx")]:
        if "year" in df.columns and df["year"].isna().any():
            print(f"WARNING: {label} for {year} has missing year values.")
        df["year"] = df["year"].fillna(year).astype("Int64")

    events_for_merge = prepare_events_for_merge(events)
    event_cols = [
        "event_fcode",
        "event_code",
        "year",
        "eventname",
        "event_date_raw",
        "event_date_parsed",
        "e_class",
        "e_games",
        "game_start",
        "game_end",
        "event_country",
        "players",
        "avdg",
        "event_winner",
        "super",
    ]
    event_cols = [col for col in event_cols if col in events_for_merge.columns]

    matches = games.merge(
        events_for_merge[event_cols],
        left_on=["year", "event"],
        right_on=["year", "event_code"],
        how="left",
    )

    hidx_cols = ["fcode", "code", "year"] + HIDX_FIELDS
    hidx_cols = [col for col in hidx_cols if col in hidx.columns]
    matches = matches.merge(hidx[hidx_cols], on=["fcode", "code", "year"], how="left")

    winner_lookup = prepare_player_lookup(names, "winner")
    loser_lookup = prepare_player_lookup(names, "loser")
    matches = matches.merge(winner_lookup, on="winner", how="left")
    matches = matches.merge(loser_lookup, on="loser", how="left")

    matches = matches.merge(
        country,
        left_on="country",
        right_on="country_code",
        how="left",
    )

    matches = order_output_columns(matches)
    summary = calculate_year_summary(year, games, events, hidx, matches)
    return matches, summary


def build_multiyear_dataset(start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the chronologically ordered game dataset."""
    names = load_names()
    country = load_country()
    all_matches = []
    yearly_summaries = []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Loading {year} ===")
        try:
            year_matches, year_summary = load_one_year(year, names=names, country=country)
            all_matches.append(year_matches)
            yearly_summaries.append(year_summary)
        except Exception as exc:
            print(f"WARNING: failed to load {year}: {exc}")
            yearly_summaries.append(
                pd.DataFrame(
                    [
                        {
                            "year": year,
                            "status": f"failed: {exc}",
                        }
                    ]
                )
            )

    if all_matches:
        matches = pd.concat(all_matches, ignore_index=True)
        sort_cols = [
            col
            for col in ["year", "event_date_parsed", "event", "code", "fcode"]
            if col in matches.columns
        ]
        matches = matches.sort_values(
            sort_cols,
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    else:
        matches = pd.DataFrame()

    yearly_summary = pd.concat(yearly_summaries, ignore_index=True)
    return matches, yearly_summary


def create_data_check_summary(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(
            [
                {
                    "total_number_of_matches": 0,
                    "number_of_years": 0,
                    "min_year": np.nan,
                    "max_year": np.nan,
                    "duplicated_fcode_count": 0,
                    "missing_event_rows": np.nan,
                    "missing_hidx_rows": np.nan,
                    "missing_winner_names": np.nan,
                    "missing_loser_names": np.nan,
                    "number_of_unique_players": 0,
                    "number_of_unique_events": 0,
                }
            ]
        )

    players = pd.concat([matches["winner"], matches["loser"]]).dropna()
    missing_hidx_rows = int(matches[HIDX_FIELDS].isna().all(axis=1).sum())

    summary = {
        "total_number_of_matches": len(matches),
        "number_of_years": int(matches["year"].nunique(dropna=True)),
        "min_year": int(matches["year"].min()),
        "max_year": int(matches["year"].max()),
        "duplicated_fcode_count": int(matches["fcode"].duplicated().sum()),
        "missing_event_rows": count_missing(matches["eventname"]),
        "missing_hidx_rows": missing_hidx_rows,
        "missing_winner_names": count_missing(matches["winner_name"]),
        "missing_loser_names": count_missing(matches["loser_name"]),
        "missing_event_date_parsed": (
            int(matches["event_date_parsed"].isna().sum())
            if "event_date_parsed" in matches.columns
            else len(matches)
        ),
        "number_of_unique_players": int(players.nunique()),
        "number_of_unique_events": int(matches[["year", "event"]].drop_duplicates().shape[0]),
    }
    return pd.DataFrame([summary])


def save_outputs(
    matches: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_OUTPUT_PATH, index=False)
    yearly_summary.to_csv(YEARLY_SUMMARY_PATH, index=False)
    overall_summary.to_csv(OVERALL_SUMMARY_PATH, index=False)
    return {
        "matches": MATCHES_OUTPUT_PATH,
        "yearly_summary": YEARLY_SUMMARY_PATH,
        "overall_summary": OVERALL_SUMMARY_PATH,
    }


def configure_output_root(output_root: str | Path) -> Path:
    global OUTPUT_DIR, MATCHES_OUTPUT_PATH, YEARLY_SUMMARY_PATH, OVERALL_SUMMARY_PATH

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    OUTPUT_DIR = root.resolve() / "elo_optimization"
    MATCHES_OUTPUT_PATH = OUTPUT_DIR / f"matches_{START_YEAR}_{END_YEAR}_checked.csv"
    YEARLY_SUMMARY_PATH = (
        OUTPUT_DIR / f"full_history_yearly_check_summary_{START_YEAR}_{END_YEAR}.csv"
    )
    OVERALL_SUMMARY_PATH = (
        OUTPUT_DIR / f"full_history_data_check_summary_{START_YEAR}_{END_YEAR}.csv"
    )
    return OUTPUT_DIR


def validate_canonical_counts(matches: pd.DataFrame) -> Dict[str, bool]:
    """Validate the frozen dataset counts."""

    players = pd.concat([matches["winner"], matches["loser"]]).dropna().nunique()
    test_matches = int((pd.to_numeric(matches["year"], errors="coerce") == TEST_YEAR).sum())
    checks = {
        "full_history_match_count": len(matches) == EXPECTED_FULL_HISTORY_MATCHES,
        "unique_player_count": int(players) == EXPECTED_UNIQUE_PLAYERS,
        "test_year_match_count": test_matches == EXPECTED_TEST_MATCHES,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "Canonical checked-match counts failed: "
            + ", ".join(failures)
            + f"; observed matches={len(matches)}, players={int(players)}, "
            + f"{TEST_YEAR} matches={test_matches}"
        )
    return checks


def print_command_line_summary(
    start_year: int,
    end_year: int,
    overall_summary: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> None:
    row = overall_summary.iloc[0]

    print("\n=== Multi-year Match Dataset Summary ===")
    print(f"Start year: {start_year}")
    print(f"End year: {end_year}")
    print(f"Total matches: {row['total_number_of_matches']}")
    print(f"Total unique players: {row['number_of_unique_players']}")
    print(f"Duplicated fcode count: {row['duplicated_fcode_count']}")
    print(f"Missing event rows: {row['missing_event_rows']}")
    print(f"Missing hidx rows: {row['missing_hidx_rows']}")
    print(f"Missing winner names: {row['missing_winner_names']}")
    print(f"Missing loser names: {row['missing_loser_names']}")
    print("Output paths:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    matches, yearly_summary = build_multiyear_dataset(START_YEAR, END_YEAR)
    validate_canonical_counts(matches)
    overall_summary = create_data_check_summary(matches)
    output_paths = save_outputs(matches, yearly_summary, overall_summary)
    print_command_line_summary(START_YEAR, END_YEAR, overall_summary, output_paths)


if __name__ == "__main__":
    main()
