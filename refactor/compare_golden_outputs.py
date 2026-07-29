"""Compare protected compact outputs with refactor validation reruns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "outputs"
GENERATED_ROOT = REFERENCE_ROOT / "refactor_validation"
REPORT_PATH = PROJECT_ROOT / "refactor" / "golden_output_comparison.csv"
TOLERANCE = 1e-10


@dataclass(frozen=True)
class CsvPair:
    name: str
    keys: tuple[str, ...]
    directory: str = "meeting8_technical"


MEETING7_PAIRS: tuple[CsvPair, ...] = (
    CsvPair(
        "34_bootstrap_confidence_intervals.csv",
        ("group_type", "group", "comparison", "metric"),
        "meeting7",
    ),
    CsvPair(
        "34_bootstrap_metadata.csv",
        ("threshold", "comparison", "metric"),
        "meeting7",
    ),
    CsvPair(
        "34_bootstrap_method_audit_checks.csv",
        ("check_name",),
        "meeting7",
    ),
    CsvPair(
        "34_bootstrap_robustness_comparison.csv",
        ("threshold", "metric"),
        "meeting7",
    ),
    CsvPair(
        "34_bootstrap_robustness_validation_checks.csv",
        ("check_name",),
        "meeting7",
    ),
    CsvPair(
        "34_cumulative_threshold_model_performance.csv",
        ("group_type", "group", "model"),
        "meeting7",
    ),
    CsvPair(
        "34_early_game_appearance_dataset.csv",
        ("match_id", "player_id"),
        "meeting7",
    ),
    CsvPair(
        "34_exact_appearance_counts.csv",
        ("appearance_number",),
        "meeting7",
    ),
    CsvPair(
        "34_exact_appearance_model_performance.csv",
        ("appearance_number", "model"),
        "meeting7",
    ),
    CsvPair("34_input_validation_checks.csv", ("check_name",), "meeting7"),
    CsvPair(
        "34_match_level_bootstrap_confidence_intervals.csv",
        ("group", "metric"),
        "meeting7",
    ),
    CsvPair(
        "34_match_level_early_game_robustness.csv",
        ("group",),
        "meeting7",
    ),
    CsvPair("34_metric_validation_checks.csv", ("check_name",), "meeting7"),
    CsvPair(
        "34_pairwise_model_differences.csv",
        ("group_type", "group", "comparison"),
        "meeting7",
    ),
    CsvPair(
        "34_stage_bin_model_performance.csv",
        ("group_type", "group", "model"),
        "meeting7",
    ),
    CsvPair(
        "34_bootstrap_figure_validation_checks.csv",
        ("check_name",),
        "meeting7",
    ),
    CsvPair("34_figure_manifest.csv", ("figure_id",), "meeting7"),
)


MEETING8_PAIRS: tuple[CsvPair, ...] = (
    CsvPair("41_burnin_definition_sensitivity.csv", ("burn_in_years",)),
    CsvPair("41_debut_cohort_summary.csv", ("group",)),
    CsvPair(
        "41_yearly_debut_anchor_diagnostics.csv",
        ("year", "entry_cohort"),
    ),
    CsvPair("41_yearly_rating_scale_drift.csv", ("year",)),
    CsvPair(
        "41_2025_first_appearance_validation.csv",
        ("match_id", "player_id"),
    ),
    CsvPair("41_validation_checks.csv", ("check",)),
    CsvPair("42_entry_year_scale_summary.csv", ("year", "entry_cohort")),
    CsvPair("42_entry_cohort_scale_summary.csv", ("group",)),
    CsvPair(
        "42_2025_crossfile_entry_audit.csv",
        ("match_id", "player_id"),
    ),
    CsvPair("42_probability_orientation_sensitivity.csv", ("group",)),
    CsvPair("42_burnin_sensitivity_audit.csv", ("burn_in_years",)),
    CsvPair("42_validation_checks.csv", ("check_name",)),
)


def _sorted(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def _numeric_columns(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in reference.columns
        if pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(generated[column])
    ]


def _max_numeric_difference(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[float, bool]:
    maximum = 0.0
    nan_patterns_match = True
    for column in columns:
        left = reference[column].to_numpy(dtype=float)
        right = generated[column].to_numpy(dtype=float)
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        nan_patterns_match &= bool(np.array_equal(left_nan, right_nan))
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(left[finite] - right[finite]))),
            )
    return maximum, nan_patterns_match


def _text_matches(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> bool:
    text_columns = [
        column
        for column in reference.columns
        if column not in numeric_columns
    ]
    for column in text_columns:
        left = reference[column].fillna("<NA>").astype(str)
        right = generated[column].fillna("<NA>").astype(str)
        if column == "path":
            left = left.str.replace("\\", "/", regex=False).str.rsplit(
                "/",
                n=1,
            ).str[-1]
            right = right.str.replace("\\", "/", regex=False).str.rsplit(
                "/",
                n=1,
            ).str[-1]
        if not left.equals(right):
            return False
    return True


def compare_same_schema(pair: CsvPair) -> dict[str, object]:
    reference_path = REFERENCE_ROOT / pair.directory / pair.name
    generated_path = GENERATED_ROOT / pair.directory / pair.name
    reference = _sorted(pd.read_csv(reference_path), pair.keys)
    generated = _sorted(pd.read_csv(generated_path), pair.keys)
    columns_match = reference.columns.equals(generated.columns)
    row_count_match = len(reference) == len(generated)
    key_match = (
        row_count_match
        and reference[list(pair.keys)]
        .fillna("<NA>")
        .astype(str)
        .equals(
            generated[list(pair.keys)].fillna("<NA>").astype(str)
        )
    )
    if not columns_match or not row_count_match:
        max_difference = float("inf")
        text_match = False
        nan_patterns_match = False
    else:
        numeric_columns = _numeric_columns(reference, generated)
        max_difference, nan_patterns_match = _max_numeric_difference(
            reference,
            generated,
            numeric_columns,
        )
        text_match = _text_matches(reference, generated, numeric_columns)
    passed = (
        columns_match
        and row_count_match
        and key_match
        and nan_patterns_match
        and text_match
        and max_difference <= TOLERANCE
    )
    return {
        "comparison_id": pair.name[:-4] if pair.name.endswith(".csv") else pair.name,
        "reference_file": reference_path.relative_to(PROJECT_ROOT).as_posix(),
        "new_file": generated_path.relative_to(PROJECT_ROOT).as_posix(),
        "key_columns": "|".join(pair.keys),
        "row_count_reference": len(reference),
        "row_count_new": len(generated),
        "column_match": columns_match,
        "key_match": key_match,
        "max_absolute_numeric_difference": max_difference,
        "text_columns_match": text_match and nan_patterns_match,
        "tolerance": TOLERANCE,
        "status": "PASS" if passed else "FAIL",
        "notes": (
            "Exact schema comparison after stable key sort; path compared by filename."
            if "path" in reference.columns
            else "Exact schema comparison after stable key sort."
        ),
    }


def compare_elo_metrics() -> dict[str, object]:
    reference_path = (
        REFERENCE_ROOT / "meeting6" / "33_overall_model_metrics.csv"
    )
    generated_path = (
        GENERATED_ROOT / "elo_pipeline" / "elo_metrics_2025.csv"
    )
    generated = pd.read_csv(generated_path).sort_values("model").reset_index(
        drop=True
    )
    reference = pd.read_csv(reference_path)
    reference = (
        reference.loc[reference["model"].isin(generated["model"])]
        .sort_values("model")
        .reset_index(drop=True)
    )
    metrics = ["brier", "log_loss", "accuracy"]
    key_match = reference["model"].equals(generated["model"])
    differences = [
        float(
            np.max(
                np.abs(
                    reference[metric].to_numpy(dtype=float)
                    - generated[metric].to_numpy(dtype=float)
                )
            )
        )
        for metric in metrics
    ]
    maximum = max(differences)
    passed = len(reference) == len(generated) == 3 and key_match and maximum <= TOLERANCE
    return {
        "comparison_id": "elo_2025_core_metrics",
        "reference_file": reference_path.relative_to(PROJECT_ROOT).as_posix(),
        "new_file": generated_path.relative_to(PROJECT_ROOT).as_posix(),
        "key_columns": "model",
        "row_count_reference": len(reference),
        "row_count_new": len(generated),
        "column_match": "mapped",
        "key_match": key_match,
        "max_absolute_numeric_difference": maximum,
        "text_columns_match": True,
        "tolerance": TOLERANCE,
        "status": "PASS" if passed else "FAIL",
        "notes": "Mapped comparison of brier, log_loss, and accuracy.",
    }


def main() -> None:
    rows = [compare_elo_metrics()]
    rows.extend(compare_same_schema(pair) for pair in MEETING7_PAIRS)
    rows.extend(compare_same_schema(pair) for pair in MEETING8_PAIRS)
    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    failed = report.loc[report["status"].ne("PASS")]
    print(report.to_string(index=False))
    print(f"\nReport: {REPORT_PATH}")
    if len(failed):
        raise RuntimeError(
            "Golden-output comparison failures: "
            + ", ".join(failed["comparison_id"].tolist())
        )


if __name__ == "__main__":
    main()
