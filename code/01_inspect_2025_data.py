from pathlib import Path
from typing import Dict, List

import pandas as pd


YEAR = 2025
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"


GAME_COLS = ["fcode", "code", "year", "event", "winner", "loser", "country", "tp"]

EVENT_COLS = [
    "event_fcode",
    "event_code",
    "year",
    "eventname",
    "date",
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

COUNTRY_COLS = ["country_code", "country_name", "governing_body", "nationality"]
TEAM_COLS = ["team_code", "team_name"]


def read_csv_flexible(path: Path, table_name: str, sep: str = ",") -> pd.DataFrame:
    """Read a raw croquet file while trying common encodings."""
    if not path.exists():
        raise FileNotFoundError(f"{table_name}: file not found: {path}")

    errors = []
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(
                path,
                header=None,
                sep=sep,
                dtype="string",
                keep_default_na=False,
                skipinitialspace=True,
                encoding=encoding,
                engine="python",
            )
            print(f"{table_name}: read {path.name} with encoding={encoding}")
            return clean_strings(df)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"{table_name}: could not read {path}\n{joined_errors}")


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and turn blank cells into proper missing values."""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].mask(df[col].eq(""), pd.NA)
    return df


def assign_columns_safely(
    df: pd.DataFrame, expected_cols: List[str], table_name: str
) -> pd.DataFrame:
    """Assign expected names, preserving unexpected columns as extra_1, extra_2, ..."""
    df = df.copy()
    actual_count = df.shape[1]
    expected_count = len(expected_cols)

    if actual_count == expected_count:
        df.columns = expected_cols
        return df

    print(
        f"\nWARNING: {table_name} has {actual_count} columns; "
        f"expected {expected_count}."
    )
    print(f"{table_name}: first rows before safe column assignment:")
    print(df.head().to_string(index=False))

    if actual_count > expected_count:
        extra_cols = [f"extra_{i}" for i in range(1, actual_count - expected_count + 1)]
        df.columns = expected_cols + extra_cols
    else:
        df.columns = expected_cols[:actual_count]
        for missing_col in expected_cols[actual_count:]:
            df[missing_col] = pd.NA
            print(f"{table_name}: added missing column {missing_col!r} as NA")

    return df


def assign_generic_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{i:02d}" for i in range(1, df.shape[1] + 1)]
    return df


def convert_key_columns(df: pd.DataFrame, cols: List[str], table_name: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            print(f"WARNING: {table_name} is missing key column {col!r}")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def print_table_overview(table_name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {table_name} ---")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("head:")
    print(df.head().to_string(index=False))


def read_names_table() -> pd.DataFrame:
    """Read names.dat, falling back to whitespace parsing if comma parsing looks wrong."""
    path = DATA_RAW / "names.dat"
    names = read_csv_flexible(path, "names")
    if names.shape[1] < 7:
        print("WARNING: names.dat comma parsing produced too few columns.")
        print("Trying whitespace parsing for inspection.")
        names = read_csv_flexible(path, "names", sep=r"\s+")
    names = assign_columns_safely(names, NAME_COLS, "names")
    return convert_key_columns(names, ["code"], "names")


def make_player_lookup(names: pd.DataFrame) -> pd.DataFrame:
    required = ["code", "firstname", "surname", "country"]
    missing = [col for col in required if col not in names.columns]
    if missing:
        raise ValueError(f"names: missing required columns for player lookup: {missing}")

    players = names[required].copy()
    duplicate_codes = players["code"].duplicated().sum()
    if duplicate_codes:
        print(f"WARNING: names has {duplicate_codes} duplicate player codes; keeping first.")
        players = players.drop_duplicates("code", keep="first")

    first = players["firstname"].fillna("")
    surname = players["surname"].fillna("")
    players["player_name"] = (first + " " + surname).str.strip()
    players["player_name"] = players["player_name"].mask(players["player_name"].eq(""), pd.NA)
    players = players.rename(columns={"code": "player_code", "country": "player_country"})
    return players[["player_code", "player_name", "player_country"]]


def merge_matches(games: pd.DataFrame, events: pd.DataFrame, hidx: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    event_cols = [
        "event_code",
        "year",
        "eventname",
        "date",
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
    event_cols = [col for col in event_cols if col in events.columns]

    matches = games.merge(
        events[event_cols],
        left_on=["year", "event"],
        right_on=["year", "event_code"],
        how="left",
    )

    hidx_cols = [col for col in hidx.columns if col not in {"extra_1", "extra_2"}]
    matches = matches.merge(hidx[hidx_cols], on=["fcode", "code", "year"], how="left")

    players = make_player_lookup(names)
    winner_lookup = players.rename(
        columns={
            "player_code": "winner",
            "player_name": "winner_name",
            "player_country": "winner_country",
        }
    )
    loser_lookup = players.rename(
        columns={
            "player_code": "loser",
            "player_name": "loser_name",
            "player_country": "loser_country",
        }
    )

    matches = matches.merge(winner_lookup, on="winner", how="left")
    matches = matches.merge(loser_lookup, on="loser", how="left")
    return matches.sort_values(["year", "code"]).reset_index(drop=True)


def print_merge_checks(games: pd.DataFrame, events: pd.DataFrame, hidx: pd.DataFrame, matches: pd.DataFrame) -> None:
    print("\n--- Merge checks ---")
    print(f"games.fcode duplicate count: {games['fcode'].duplicated().sum()}")
    print(f"hidx.fcode duplicate count: {hidx['fcode'].duplicated().sum()}")
    print(f"games rows before merge: {len(games)}")
    print(f"matches rows after merge: {len(matches)}")

    event_keys = events[["year", "event_code"]].drop_duplicates()
    event_key_matches = games.merge(
        event_keys,
        left_on=["year", "event"],
        right_on=["year", "event_code"],
        how="left",
        indicator=True,
    )
    hidx_keys = hidx[["fcode", "code", "year"]].drop_duplicates()
    hidx_key_matches = games.merge(
        hidx_keys,
        on=["fcode", "code", "year"],
        how="left",
        indicator=True,
    )
    print(f"games without matching event row: {(event_key_matches['_merge'] == 'left_only').sum()}")
    print(f"games without matching hidx row: {(hidx_key_matches['_merge'] == 'left_only').sum()}")


def print_missing_statistics(matches: pd.DataFrame) -> Dict[str, int]:
    hidx_metric_cols = [col for col in HIDX_COLS if col not in {"fcode", "code", "year"}]
    missing_hidx_rows = matches[hidx_metric_cols].isna().all(axis=1).sum()
    missing_hidx_fields = matches[hidx_metric_cols].isna().sum()

    print("\n--- Missing statistics ---")
    print(f"missing eventname: {matches['eventname'].isna().sum()}")
    print(f"missing hidx rows: {missing_hidx_rows}")
    print("missing hidx fields:")
    print(missing_hidx_fields.to_string())
    print(f"missing winner name: {matches['winner_name'].isna().sum()}")
    print(f"missing loser name: {matches['loser_name'].isna().sum()}")

    return {
        "missing_event_rows": int(matches["eventname"].isna().sum()),
        "missing_hidx_rows": int(missing_hidx_rows),
        "missing_winner_names": int(matches["winner_name"].isna().sum()),
        "missing_loser_names": int(matches["loser_name"].isna().sum()),
    }


def print_summary(
    games: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    missing: Dict[str, int],
    output_path: Path,
) -> None:
    players_in_games = pd.concat([matches["winner"], matches["loser"]]).dropna()

    print("\n=== Summary ===")
    print(f"Number of games: {len(games)}")
    print(f"Number of events: {len(events)}")
    print(f"Number of unique winners: {matches['winner'].nunique(dropna=True)}")
    print(f"Number of unique losers: {matches['loser'].nunique(dropna=True)}")
    print(f"Number of unique players appearing in games: {players_in_games.nunique()}")
    print(f"Missing event rows: {missing['missing_event_rows']}")
    print(f"Missing hidx rows: {missing['missing_hidx_rows']}")
    print(f"Missing winner names: {missing['missing_winner_names']}")
    print(f"Missing loser names: {missing['missing_loser_names']}")
    print(f"Output path: {output_path}")


def main() -> None:
    DATA_PROCESSED.mkdir(exist_ok=True)

    games = read_csv_flexible(DATA_RAW / f"game{YEAR}.csv", "games")
    events = read_csv_flexible(DATA_RAW / f"evnt{YEAR}.csv", "events")
    hidx = read_csv_flexible(DATA_RAW / f"hidx{YEAR}.csv", "hidx")
    out = read_csv_flexible(DATA_RAW / f"out{YEAR}.csv", "out")
    names = read_names_table()
    country = read_csv_flexible(DATA_RAW / "country.csv", "country")
    teams = read_csv_flexible(DATA_RAW / "teams.csv", "teams")

    games = assign_columns_safely(games, GAME_COLS, "games")
    events = assign_columns_safely(events, EVENT_COLS, "events")
    hidx = assign_columns_safely(hidx, HIDX_COLS, "hidx")
    out = assign_generic_columns(out, "out_col")
    country = assign_columns_safely(country, COUNTRY_COLS, "country")
    teams = assign_columns_safely(teams, TEAM_COLS, "teams")

    games = convert_key_columns(games, ["fcode", "code", "year", "event", "winner", "loser"], "games")
    events = convert_key_columns(events, ["event_fcode", "event_code", "year"], "events")
    hidx = convert_key_columns(hidx, ["fcode", "code", "year"], "hidx")

    print_table_overview("games", games)
    print_table_overview("events", events)
    print_table_overview("hidx", hidx)
    print_table_overview("out", out)
    print_table_overview("names", names)
    print_table_overview("country", country)
    print_table_overview("teams", teams)

    print(
        "\nout2025.csv: column meanings are not confirmed yet, "
        "so player_start_ratings_2025.csv is not created in this stage."
    )

    matches = merge_matches(games, events, hidx, names)
    print_table_overview("matches", matches)
    print_merge_checks(games, events, hidx, matches)
    missing = print_missing_statistics(matches)

    output_path = DATA_PROCESSED / "matches_2025_checked.csv"
    matches.to_csv(output_path, index=False)
    print_summary(games, events, matches, missing, output_path)


if __name__ == "__main__":
    main()
