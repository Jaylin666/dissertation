"""Project file and table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PUBLIC_MATCHES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "association_croquet_games_1985_2025_no_names.csv.gz"
)


def project_path(*parts: str) -> Path:
    """Return a path below the repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def resolve_output_root(output_root: str | Path | None = None) -> Path:
    """Resolve an optional output root."""

    if output_root is None:
        return DEFAULT_OUTPUT_ROOT
    candidate = Path(output_root)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory."""

    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def require_columns(
    table: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    """Raise an error when required columns are missing."""

    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_csv_checked(
    path: str | Path,
    required_columns: Sequence[str] = (),
    *,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a CSV and check its required columns."""

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
    """Sort games chronologically while retaining missing dates."""

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


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ordering dates without changing the raw date fields."""

    ordered = matches.copy()
    if "event_date_raw" not in ordered.columns:
        ordered["event_date_raw"] = pd.NA
    if "event_date_parsed" not in ordered.columns:
        ordered["event_date_parsed"] = pd.NA

    if (
        "event_order_date" in ordered.columns
        and "event_date_ordering_method" in ordered.columns
    ):
        ordered["event_order_date"] = pd.to_datetime(
            ordered["event_order_date"],
            errors="coerce",
        )
        return ordered

    parsed = pd.to_datetime(ordered["event_date_parsed"], errors="coerce")
    ordered["event_order_date"] = parsed
    ordered["event_date_ordering_method"] = np.where(
        parsed.notna(),
        "parsed_full_date",
        "fallback_no_date",
    )

    missing_parsed = parsed.isna()
    raw = ordered.loc[missing_parsed, "event_date_raw"].astype("string").str.strip()
    extracted = raw.str.extract(r"^(?P<month>\d{1,2})\.(?P<year>\d{2}|\d{4})$")
    valid_month_year = extracted["month"].notna()

    if valid_month_year.any():
        months = pd.to_numeric(
            extracted.loc[valid_month_year, "month"],
            errors="coerce",
        )
        raw_years = extracted.loc[valid_month_year, "year"].astype(str)
        years_numeric = raw_years.astype(int)
        years = np.where(
            raw_years.str.len().eq(2),
            np.where(
                years_numeric >= 85,
                1900 + years_numeric,
                2000 + years_numeric,
            ),
            years_numeric,
        )
        valid_month = months.between(1, 12).fillna(False)
        valid_mask = valid_month.to_numpy(dtype=bool)
        valid_index = extracted.loc[valid_month_year].index[valid_mask]
        # The 15th orders month-year records only; it is not a recorded game date.
        imputed_dates = pd.to_datetime(
            {
                "year": np.asarray(years)[valid_mask],
                "month": months.loc[valid_index].astype(int).to_numpy(),
                "day": np.repeat(15, len(valid_index)),
            },
            errors="coerce",
        )
        ordered.loc[valid_index, "event_order_date"] = imputed_dates.to_numpy()
        ordered.loc[
            valid_index,
            "event_date_ordering_method",
        ] = "month_year_imputed"

    return ordered
