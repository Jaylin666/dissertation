"""Validation records and collectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np

from code.io_utils import PROJECT_ROOT, write_csv


VALID_SEVERITIES = {"error", "warning"}


@dataclass(frozen=True)
class CheckRecord:
    """One validation result."""

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
    """Collect validation results by severity."""

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
    """Convert a supported value to bool."""

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


REPRODUCTION_EVIDENCE_TABLES = (
    ("elo_validation/elo_validation_grid.csv", "chapter4/elo_validation_grid.csv", ()),
    ("elo_validation/elo_validation_selected_model.csv", "chapter4/elo_validation_selected_model.csv", ()),
    ("glicko_inflation/glicko_inflation_metrics_2025.csv", "chapter4/glicko_inflation_sensitivity.csv", ("runtime_seconds",)),
    ("glicko_rating_period/glicko_rating_period_metrics.csv", "chapter4/glicko_rating_period_metrics.csv", ()),
    ("comparison/overall_model_metrics.csv", "chapter4/overall_model_metrics.csv", ()),
    ("comparison/overall_pairwise_comparisons.csv", "chapter4/overall_pairwise_comparisons.csv", ()),
    ("comparison/overall_bootstrap_confidence_intervals.csv", "chapter4/overall_bootstrap_confidence_intervals.csv", ()),
    ("comparison/calibration_bins.csv", "chapter4/calibration_bins.csv", ()),
    ("comparison/calibration_summary.csv", "chapter4/calibration_summary.csv", ()),
    ("comparison/adaptive_k_recovery.csv", "chapter4/adaptive_k_recovery.csv", ()),
    ("early_game/early_game_cumulative_core.csv", "chapter5/early_game_cumulative_core.csv", ()),
    ("early_game/early_game_stage_core.csv", "chapter5/early_game_stage_core.csv", ()),
    ("early_game/early_game_event_cluster_ci_core.csv", "chapter5/early_game_event_cluster_ci_core.csv", ()),
    ("early_game/first_appearance_mechanism_core.csv", "chapter5/first_appearance_mechanism_core.csv", ()),
    ("initial_rating_sensitivity/initial_rating_invariance_core.csv", "chapter5/initial_rating_invariance_core.csv", ()),
    ("entry_diagnostics/entry_cohort_definitions_core.csv", "chapter5/entry_cohort_definitions_core.csv", ()),
    ("entry_diagnostics/burnin_sensitivity_core.csv", "chapter5/burnin_sensitivity_core.csv", ()),
    ("entry_diagnostics/prematch_scale_alignment_2025_core.csv", "chapter5/prematch_scale_alignment_2025_core.csv", ()),
    ("entry_diagnostics/orientation_sensitivity_2025_core.csv", "chapter5/orientation_sensitivity_2025_core.csv", ()),
)


def compare_csv_tables(
    reproduced: pd.DataFrame,
    reference: pd.DataFrame,
    excluded_columns: Iterable[str] = (),
    tolerance: float = 1e-9,
) -> list[str]:
    """Return scientific differences without changing row order."""

    differences: list[str] = []
    excluded = set(excluded_columns)
    if len(reproduced) != len(reference):
        return [f"row count {len(reproduced)} != {len(reference)}"]
    columns = [column for column in reference.columns if column not in excluded]
    missing = [column for column in columns if column not in reproduced.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for column in columns:
        left = reproduced[column]
        right = reference[column]
        numeric = (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
            and not pd.api.types.is_bool_dtype(left)
            and not pd.api.types.is_bool_dtype(right)
        )
        if numeric:
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            if not np.allclose(
                left_values,
                right_values,
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            ):
                maximum = float(np.nanmax(np.abs(left_values - right_values)))
                differences.append(f"{column}: max absolute difference {maximum}")
        else:
            left_text = left.fillna("<NA>").astype(str).reset_index(drop=True)
            right_text = right.fillna("<NA>").astype(str).reset_index(drop=True)
            if not left_text.equals(right_text):
                differences.append(f"{column}: row values or order differ")
    return differences


def compare_reproduction_to_evidence(
    output_root: str | Path,
    evidence_root: str | Path | None = None,
) -> pd.DataFrame:
    """Compare generated compact tables with tracked direct evidence."""

    generated_root = Path(output_root)
    if not generated_root.is_absolute():
        generated_root = PROJECT_ROOT / generated_root
    tracked_root = (
        Path(evidence_root)
        if evidence_root is not None
        else PROJECT_ROOT / "outputs" / "dissertation_evidence"
    )
    rows: list[dict[str, Any]] = []
    for generated_name, tracked_name, exclusions in REPRODUCTION_EVIDENCE_TABLES:
        generated_path = generated_root / generated_name
        tracked_path = tracked_root / tracked_name
        if not generated_path.exists() or not tracked_path.exists():
            differences = [
                "missing "
                + ", ".join(
                    str(path)
                    for path in (generated_path, tracked_path)
                    if not path.exists()
                )
            ]
        else:
            differences = compare_csv_tables(
                pd.read_csv(generated_path, low_memory=False),
                pd.read_csv(tracked_path, low_memory=False),
                exclusions,
            )
        rows.append(
            {
                "generated_table": generated_name,
                "tracked_table": tracked_name,
                "passed": not differences,
                "detail": "; ".join(differences),
            }
        )
    return pd.DataFrame(rows)
