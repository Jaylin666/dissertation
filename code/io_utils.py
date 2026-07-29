"""Shared project-relative file and table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def project_path(*parts: str) -> Path:
    """Return a path below the repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def resolve_output_root(output_root: str | Path | None = None) -> Path:
    """Resolve an optional output root without requiring it to exist."""

    if output_root is None:
        return DEFAULT_OUTPUT_ROOT
    candidate = Path(output_root)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create and return one directory."""

    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def require_columns(
    table: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    """Raise a clear error if a table lacks required columns."""

    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_csv_checked(
    path: str | Path,
    required_columns: Sequence[str] = (),
    *,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a CSV and apply an optional required-column check."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required input not found: {source}")
    table = pd.read_csv(source, low_memory=low_memory)
    require_columns(table, required_columns, source.name)
    return table


def write_csv(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV after creating its parent directory."""

    destination = Path(path)
    ensure_directory(destination.parent)
    table.to_csv(destination, index=False)
    return destination


def stable_match_sort(
    matches: pd.DataFrame,
    *,
    year_column: str = "year",
    date_column: str = "event_order_date",
    event_column: str = "event",
    match_column: str = "fcode",
) -> pd.DataFrame:
    """Apply the frozen stable chronological order without dropping missing dates."""

    require_columns(matches, [year_column, event_column, match_column], "matches")
    ordered = matches.copy()
    sort_columns = [year_column]
    temporary_missing = "__event_order_date_missing"
    if date_column in ordered.columns:
        ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
        ordered[temporary_missing] = ordered[date_column].isna()
        sort_columns.extend([temporary_missing, date_column])
    sort_columns.extend([event_column, match_column])
    ordered = ordered.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    )
    if temporary_missing in ordered.columns:
        ordered = ordered.drop(columns=[temporary_missing])
    return ordered.reset_index(drop=True)
