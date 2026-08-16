"""Export the name-free processed game dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from code.config import (
    EXPECTED_FULL_HISTORY_MATCHES,
    EXPECTED_TEST_MATCHES,
    EXPECTED_UNIQUE_PLAYERS,
    FULL_HISTORY_END_YEAR,
    FULL_HISTORY_START_YEAR,
    TEST_YEAR,
)
from code.io_utils import PROJECT_ROOT, PUBLIC_MATCHES_PATH


DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "elo_optimization"
    / "matches_1985_2025_checked.csv"
)
PUBLIC_MANIFEST_PATH = PUBLIC_MATCHES_PATH.with_name("manifest.json")
REQUIRED_COLUMNS = ("fcode", "year", "event", "winner", "loser")
DIRECT_PERSONAL_COLUMNS = (
    "winner_name",
    "loser_name",
    "player_name",
    "surname",
    "firstname",
    "initials",
    "oldname",
    "altfirstname",
    "oldfirstname",
    "birthy",
    "birthm",
    "birthd",
    "deathy",
    "deathm",
    "deathd",
    "namechange",
    "firstnamechange",
)
PUBLIC_CHECK_NAMES = (
    "file_exists",
    "gzip_csv_readable",
    "required_columns_present",
    "full_history_match_count",
    "unique_coded_player_count",
    "test_year_match_count",
    "year_range",
    "fcode_unique",
    "fcode_non_missing",
    "winner_non_missing",
    "loser_non_missing",
    "winner_differs_from_loser",
    "direct_personal_columns_absent",
    "numeric_player_ids_retained",
    "manifest_sha256_matches",
)
MAX_PUBLIC_FILE_BYTES = 95 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_table(table: pd.DataFrame) -> dict[str, int]:
    """Validate the frozen game-table invariants."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in table]
    if missing_columns:
        raise ValueError(f"Processed dataset is missing columns: {missing_columns}")

    core = table.loc[:, REQUIRED_COLUMNS]
    empty = core.astype("string").apply(lambda column: column.str.strip().eq(""))
    numeric = core.apply(pd.to_numeric, errors="coerce")
    players = pd.concat([numeric["winner"], numeric["loser"]], ignore_index=True)
    failures: list[str] = []

    if len(table) != EXPECTED_FULL_HISTORY_MATCHES:
        failures.append(f"rows={len(table)}")
    if numeric.isna().any().any() or empty.any().any():
        failures.append("required fields contain missing values")
    if not core["fcode"].is_unique:
        failures.append("fcode is not unique")
    if numeric["winner"].eq(numeric["loser"]).any():
        failures.append("winner equals loser")
    if int(players.nunique()) != EXPECTED_UNIQUE_PLAYERS:
        failures.append(f"coded players={int(players.nunique())}")

    years = numeric["year"]
    first_year = int(years.min()) if years.notna().any() else -1
    last_year = int(years.max()) if years.notna().any() else -1
    test_rows = int(years.eq(TEST_YEAR).sum())
    if (first_year, last_year) != (FULL_HISTORY_START_YEAR, FULL_HISTORY_END_YEAR):
        failures.append(f"year range={first_year}-{last_year}")
    if test_rows != EXPECTED_TEST_MATCHES:
        failures.append(f"{TEST_YEAR} rows={test_rows}")

    if failures:
        raise ValueError("Frozen processed-dataset checks failed: " + "; ".join(failures))

    return {
        "rows": len(table),
        "unique_coded_players": int(players.nunique()),
        "first_year": first_year,
        "last_year": last_year,
        "test_year_rows": test_rows,
    }


def public_dataset_check_results(
    path: str | Path = PUBLIC_MATCHES_PATH,
    manifest_path: str | Path = PUBLIC_MANIFEST_PATH,
) -> dict[str, bool]:
    """Return the fixed public-dataset validation checks."""

    source = Path(path)
    manifest_source = Path(manifest_path)
    checks = dict.fromkeys(PUBLIC_CHECK_NAMES, False)
    checks["file_exists"] = source.is_file() and source.stat().st_size > 0
    if not checks["file_exists"]:
        return checks

    try:
        columns = pd.read_csv(source, nrows=0).columns.tolist()
        checks["gzip_csv_readable"] = True
    except Exception:
        return checks

    checks["required_columns_present"] = all(
        column in columns for column in REQUIRED_COLUMNS
    )
    checks["direct_personal_columns_absent"] = not any(
        column in columns for column in DIRECT_PERSONAL_COLUMNS
    )
    if not checks["required_columns_present"]:
        return checks

    try:
        core = pd.read_csv(source, usecols=list(REQUIRED_COLUMNS), low_memory=False)
    except Exception:
        checks["gzip_csv_readable"] = False
        return checks

    numeric = core.apply(pd.to_numeric, errors="coerce")
    players = pd.concat([numeric["winner"], numeric["loser"]], ignore_index=True)
    checks["full_history_match_count"] = len(core) == EXPECTED_FULL_HISTORY_MATCHES
    checks["unique_coded_player_count"] = (
        int(players.nunique()) == EXPECTED_UNIQUE_PLAYERS
    )
    checks["test_year_match_count"] = int(numeric["year"].eq(TEST_YEAR).sum()) == (
        EXPECTED_TEST_MATCHES
    )
    checks["year_range"] = (
        int(numeric["year"].min()),
        int(numeric["year"].max()),
    ) == (FULL_HISTORY_START_YEAR, FULL_HISTORY_END_YEAR)
    checks["fcode_unique"] = core["fcode"].is_unique
    checks["fcode_non_missing"] = core["fcode"].notna().all()
    checks["winner_non_missing"] = core["winner"].notna().all()
    checks["loser_non_missing"] = core["loser"].notna().all()
    checks["winner_differs_from_loser"] = numeric["winner"].ne(
        numeric["loser"]
    ).all()
    checks["numeric_player_ids_retained"] = (
        numeric[["winner", "loser"]].notna().all().all()
        and numeric[["winner", "loser"]].mod(1).eq(0).all().all()
    )

    try:
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
        checks["manifest_sha256_matches"] = manifest.get("sha256") == sha256_file(
            source
        )
    except (OSError, json.JSONDecodeError):
        pass
    return checks


def export_public_dataset(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export and verify the name-free gzip dataset."""

    source_path = Path(input_path)
    destination = Path(output_path)
    manifest_destination = (
        Path(manifest_path) if manifest_path is not None else destination.with_name("manifest.json")
    )
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise FileNotFoundError(f"Checked source dataset not found: {source_path}")

    source = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    counts = validate_frozen_table(source)
    removed_columns = [
        column for column in source.columns if column in DIRECT_PERSONAL_COLUMNS
    ]
    retained_columns = [
        column for column in source.columns if column not in DIRECT_PERSONAL_COLUMNS
    ]
    public = source.loc[:, retained_columns]

    destination.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(
        destination,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    if destination.stat().st_size >= MAX_PUBLIC_FILE_BYTES:
        raise ValueError(
            f"Public dataset is {destination.stat().st_size} bytes; limit is below "
            f"{MAX_PUBLIC_FILE_BYTES} bytes"
        )

    round_trip = pd.read_csv(
        destination,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    pd.testing.assert_frame_equal(public, round_trip, check_dtype=False, check_exact=True)
    round_trip_counts = validate_frozen_table(round_trip)
    if counts != round_trip_counts:
        raise ValueError("Frozen counts changed during the gzip round trip")
    if any(column in round_trip.columns for column in DIRECT_PERSONAL_COLUMNS):
        raise ValueError("Direct personal columns remain in the public dataset")

    manifest: dict[str, Any] = {
        "file": destination.name,
        "sha256": sha256_file(destination),
        "compression": "gzip",
        **counts,
        "test_year": TEST_YEAR,
        "player_names_removed": True,
        "stable_player_ids_retained": True,
        "source_processed_file": source_path.name,
        "export_script": "code/data/export_public_dataset.py",
    }
    manifest_destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported: {destination}")
    print(f"Rows: {counts['rows']:,}")
    print(f"Retained columns: {len(retained_columns)}")
    print(f"Removed columns: {removed_columns}")
    print(f"SHA-256: {manifest['sha256']}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the export command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=PUBLIC_MATCHES_PATH)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public-data export."""

    args = build_parser().parse_args(argv)
    export_public_dataset(args.input, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
