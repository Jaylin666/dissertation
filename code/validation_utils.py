"""Machine-readable validation checks with preserved severity semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from code.io_utils import write_csv


VALID_SEVERITIES = {"error", "warning"}


@dataclass(frozen=True)
class CheckRecord:
    """One validation observation."""

    check_name: str
    passed: bool
    observed: Any
    expected: Any
    severity: str = "error"
    detail: str = ""

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"Unsupported severity: {self.severity}")


class ValidationCollector:
    """Collect validation checks without weakening error-level failures."""

    def __init__(self) -> None:
        self._records: list[CheckRecord] = []

    @property
    def records(self) -> tuple[CheckRecord, ...]:
        return tuple(self._records)

    def add(
        self,
        check_name: str,
        passed: bool,
        observed: Any,
        expected: Any,
        severity: str = "error",
        detail: str = "",
    ) -> CheckRecord:
        record = CheckRecord(
            check_name=check_name,
            passed=bool(passed),
            observed=observed,
            expected=expected,
            severity=severity,
            detail=detail,
        )
        self._records.append(record)
        return record

    def extend(self, records: Iterable[CheckRecord]) -> None:
        for record in records:
            if not isinstance(record, CheckRecord):
                raise TypeError("ValidationCollector accepts CheckRecord instances")
            self._records.append(record)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self._records])

    def write(self, path: str | Path) -> Path:
        return write_csv(self.to_frame(), path)

    def failed_errors(self) -> list[CheckRecord]:
        return [
            record
            for record in self._records
            if record.severity == "error" and not record.passed
        ]

    def active_warnings(self) -> list[CheckRecord]:
        return [
            record
            for record in self._records
            if record.severity == "warning" and not record.passed
        ]

    def raise_for_errors(self) -> None:
        failures = self.failed_errors()
        if failures:
            names = ", ".join(record.check_name for record in failures)
            raise RuntimeError(f"Error-level validation checks failed: {names}")


def robust_bool(value: Any) -> bool:
    """Convert supported boolean representations without truthiness shortcuts."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return bool(int(value))
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1"}:
            return True
        if normalised in {"false", "0"}:
            return False
    raise ValueError(f"Unsupported boolean value: {value!r}")
