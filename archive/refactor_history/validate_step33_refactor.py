"""Regenerate and compare the complete Step 33 refactor output set."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "outputs" / "meeting6"
DEFAULT_NEW_ROOT = (
    PROJECT_ROOT / "outputs" / "refactor_validation" / "meeting6"
)
DEFAULT_RUN_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "refactor_validation"
REFACTOR_DIR = PROJECT_ROOT / "refactor"

INVENTORY_PATH = REFACTOR_DIR / "step33_output_inventory.csv"
COMMAND_RECORD_PATH = REFACTOR_DIR / "step33_full_regression_command.txt"
COMPARISON_PATH = REFACTOR_DIR / "step33_full_golden_comparison.csv"
SUMMARY_PATH = REFACTOR_DIR / "step33_full_regression_summary.md"
FIGURE_VALIDATION_PATH = REFACTOR_DIR / "step33_figure_validation.csv"
MARKDOWN_HEADLINE_PATH = (
    REFACTOR_DIR / "step33_markdown_headline_comparison.csv"
)
HEADLINE_CHECK_PATH = REFACTOR_DIR / "step33_headline_regression_checks.csv"

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-10
BOOTSTRAP_REPETITIONS = 2_000
RANDOM_SEED = 20260715
EXPECTED_GAMES = 11_379


@dataclass(frozen=True)
class OutputSpec:
    output_name: str
    output_category: str
    comparison_method: str
    key_columns: tuple[str, ...] = ()
    large_row_level_file: bool = False
    required_for_regression: bool = True
    notes: str = ""


CSV_SPECS: tuple[OutputSpec, ...] = (
    OutputSpec(
        "33_adaptive_k_improvement_recovered.csv",
        "adaptive-k comparison",
        "full keyed CSV comparison",
        ("range",),
    ),
    OutputSpec(
        "33_brier_decomposition_bootstrap.csv",
        "bootstrap",
        "full keyed bootstrap comparison",
        ("sample",),
    ),
    OutputSpec(
        "33_canonical_player_orientation_checks.csv",
        "validation",
        "full keyed validation comparison",
        ("check_name",),
    ),
    OutputSpec(
        "33_debut_corrected_model_summary.csv",
        "debut analysis",
        "full keyed CSV comparison",
        ("sample", "model"),
    ),
    OutputSpec(
        "33_debut_corrected_player_perspective.csv",
        "large row-level debut analysis",
        "full keyed row-level comparison",
        ("match_id", "player_side", "model"),
        True,
    ),
    OutputSpec(
        "33_final_validation_checks.csv",
        "validation",
        "full keyed validation comparison",
        ("check_name",),
    ),
    OutputSpec(
        "33_glicko_probability_reconstruction_checks.csv",
        "large row-level orientation audit",
        "full keyed row-level comparison",
        ("match_id",),
        True,
    ),
    OutputSpec(
        "33_meeting6_final_results.csv",
        "final result summary",
        "full keyed CSV comparison",
        ("subgroup",),
    ),
    OutputSpec(
        "33_orientation_corrected_per_match_scores_2025.csv",
        "large row-level model scores",
        "full keyed row-level comparison",
        ("match_id",),
        True,
    ),
    OutputSpec(
        "33_orientation_sensitivity_bootstrap.csv",
        "bootstrap",
        "full keyed bootstrap comparison",
        ("model", "orientation", "sample"),
    ),
    OutputSpec(
        "33_orientation_sensitivity_metrics.csv",
        "orientation sensitivity",
        "full keyed CSV comparison",
        ("model", "orientation", "sample"),
    ),
    OutputSpec(
        "33_overall_bootstrap_confidence_intervals.csv",
        "bootstrap",
        "full keyed bootstrap comparison",
        ("diff_name",),
    ),
    OutputSpec(
        "33_overall_exclusion_robustness.csv",
        "robustness",
        "full keyed CSV comparison",
        ("sample",),
    ),
    OutputSpec(
        "33_overall_model_metrics.csv",
        "overall model comparison",
        "full keyed CSV comparison",
        ("model",),
    ),
    OutputSpec(
        "33_overall_pairwise_comparisons.csv",
        "overall paired comparison",
        "full keyed CSV comparison",
        ("comparison",),
    ),
    OutputSpec(
        "33_returning_exclusive_bins.csv",
        "returning-player analysis",
        "full keyed CSV comparison",
        ("subgroup",),
    ),
    OutputSpec(
        "33_returning_player_corrected_results.csv",
        "returning-player analysis",
        "full keyed CSV comparison",
        ("subgroup",),
    ),
    OutputSpec(
        "33_standard_brier_decomposition_bins.csv",
        "Brier decomposition",
        "full keyed CSV comparison",
        ("sample", "model", "bin_index"),
    ),
    OutputSpec(
        "33_standard_brier_decomposition_summary.csv",
        "Brier decomposition",
        "full keyed CSV comparison",
        ("sample", "model"),
    ),
    OutputSpec(
        "33_standard_calibration_bins.csv",
        "calibration",
        "full keyed CSV comparison",
        ("sample", "model", "bin_label"),
    ),
    OutputSpec(
        "33_standard_calibration_summary.csv",
        "calibration",
        "full keyed CSV comparison",
        ("sample", "model"),
    ),
    OutputSpec(
        "33_subgroup_bootstrap_confidence_intervals.csv",
        "bootstrap",
        "full keyed bootstrap comparison",
        ("subgroup_variable", "subgroup", "diff_name"),
    ),
    OutputSpec(
        "33_subgroup_model_performance_long.csv",
        "subgroup analysis",
        "full keyed CSV comparison",
        ("subgroup_variable", "subgroup", "model"),
    ),
    OutputSpec(
        "33_subgroup_pairwise_comparisons.csv",
        "subgroup paired comparison",
        "full keyed CSV comparison",
        ("subgroup_variable", "subgroup"),
    ),
    OutputSpec(
        "33_supersession_map.csv",
        "provenance",
        "full keyed text comparison",
        ("old_file",),
    ),
)

MARKDOWN_SPEC = OutputSpec(
    "33_meeting6_orientation_corrected_final_summary.md",
    "Markdown scientific summary",
    "normalised full-text and structured headline comparison",
    notes="Validation-root path prefixes are normalised before comparison.",
)

FIGURE_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "33_fig01_overall_brier_zoomed.png": (
        "33_overall_model_metrics.csv",
    ),
    "33_fig02_exclusion_robustness_delta_brier.png": (
        "33_overall_exclusion_robustness.csv",
    ),
    "33_fig03_debut_probability_vs_actual.png": (
        "33_debut_corrected_model_summary.csv",
        "33_debut_corrected_player_perspective.csv",
    ),
    "33_fig04_zero_activity_debut_decomposition.png": (
        "33_subgroup_pairwise_comparisons.csv",
        "33_subgroup_bootstrap_confidence_intervals.csv",
    ),
    "33_fig05_returner_inflation_gain.png": (
        "33_returning_exclusive_bins.csv",
    ),
    "33_fig06_no_debut_rd_quartiles.png": (
        "33_subgroup_pairwise_comparisons.csv",
    ),
    "33_fig07_standard_player_a_calibration.png": (
        "33_standard_calibration_bins.csv",
    ),
    "33_fig08_orientation_sensitivity.png": (
        "33_orientation_sensitivity_metrics.csv",
    ),
    "33_fig09_prediction_confidence_mechanism.png": (
        "33_subgroup_pairwise_comparisons.csv",
    ),
    "33_fig10_debut_opponent_rating_distribution.png": (
        "33_debut_corrected_player_perspective.csv",
    ),
}

FIGURE_SPECS: tuple[OutputSpec, ...] = tuple(
    OutputSpec(
        figure_name,
        "figure",
        "existence, image-open, dimension, content, and visual QA",
        notes="Pixel-identical rendering is not required.",
    )
    for figure_name in FIGURE_SOURCE_MAP
)

ALL_SPECS: tuple[OutputSpec, ...] = (
    *CSV_SPECS,
    MARKDOWN_SPEC,
    *FIGURE_SPECS,
)


def relative(path: Path) -> str:
    """Return a project-relative POSIX path for reports."""

    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def ensure_safe_roots(
    reference_root: Path,
    new_root: Path,
    run_output_root: Path,
    run_requested: bool,
) -> None:
    """Protect golden outputs and constrain destructive cleanup to temp data."""

    reference = reference_root.resolve()
    new = new_root.resolve()
    run_root = run_output_root.resolve()
    if reference == new:
        raise ValueError("Reference and regenerated output roots must differ.")
    if run_requested:
        allowed_root = (
            PROJECT_ROOT / "outputs" / "refactor_validation"
        ).resolve()
        try:
            new.relative_to(allowed_root)
            run_root.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                "Full regeneration is restricted to outputs/refactor_validation."
            ) from exc
        expected_new = run_root / "meeting6"
        if new != expected_new:
            raise ValueError(
                f"New root {new} does not match CLI output {expected_new}."
            )


def remove_previous_temp_step33_outputs(new_root: Path) -> None:
    """Remove only prior Step 33 files from the validated temporary root."""

    for path in new_root.glob("33_*"):
        if path.is_file():
            path.unlink()
    figure_root = new_root / "figures"
    for path in figure_root.glob("33_*"):
        if path.is_file():
            path.unlink()


def run_step33_pipeline(
    reference_root: Path,
    new_root: Path,
    run_output_root: Path,
) -> tuple[str, float]:
    """Run the real active CLI pipeline into the protected validation root."""

    remove_previous_temp_step33_outputs(new_root)
    command_display = (
        "python -m code.cli compare-models --full-run "
        f"--output-root {relative(run_output_root)}"
    )
    command = [
        sys.executable,
        "-m",
        "code.cli",
        "compare-models",
        "--full-run",
        "--output-root",
        str(run_output_root),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    runtime = time.perf_counter() - started
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Step 33 CLI failed with exit code {result.returncode}."
        )

    input_files = (
        reference_root / "29_per_match_model_scores_2025.csv",
        reference_root / "31_debut_probability_mechanism.csv",
        reference_root / "31_debut_opponent_rating_summary.csv",
        reference_root / "31_unique_player_rating_snapshot.csv",
        reference_root / "32_glicko_direct_probability_comparison.csv",
        reference_root / "32_orientation_impact_on_metrics.csv",
    )
    figure_count = len(list((new_root / "figures").glob("33_*.png")))
    record_lines = [
        f"command={command_display}",
        "working_directory=repository root",
        "inputs=" + "|".join(relative(path) for path in input_files),
        (
            "selected_models=Glicko_low_fixed|Validation_best_Elo|"
            "best_AdaptiveK|Default_Elo|Glicko_C0_fixed|Conservative_Elo"
        ),
        f"bootstrap_repetitions={BOOTSTRAP_REPETITIONS}",
        f"random_seed={RANDOM_SEED}",
        (
            "probability_convention=direct player-A probability: "
            "expected_score(rating_A, rating_B, RD_B)"
        ),
        f"output_root={relative(run_output_root)}",
        f"resolved_step33_output={relative(new_root)}",
        f"runtime_seconds={runtime:.6f}",
        f"figures_generated={figure_count}",
    ]
    COMMAND_RECORD_PATH.write_text(
        "\n".join(record_lines) + "\n",
        encoding="utf-8",
    )
    return command_display, runtime


def read_recorded_run_metadata(
    fallback_command: str,
) -> tuple[str, float | None]:
    """Read the most recent full-run command and runtime when available."""

    if not COMMAND_RECORD_PATH.exists():
        return fallback_command, None
    values: dict[str, str] = {}
    for line in COMMAND_RECORD_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    command = values.get("command", fallback_command)
    runtime_text = values.get("runtime_seconds")
    runtime = float(runtime_text) if runtime_text is not None else None
    return command, runtime


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving case-sensitive scientific column names."""

    return pd.read_csv(path, low_memory=False)


def boolean_columns(frame: pd.DataFrame) -> list[str]:
    """Identify native and text-encoded boolean columns."""

    columns: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = {
            str(value).strip().lower()
            for value in series.dropna().unique().tolist()
        }
        if values and values.issubset({"true", "false"}):
            columns.append(column)
    return columns


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns parsed as numeric by pandas."""

    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def normalise_text_series(series: pd.Series, column: str) -> pd.Series:
    """Normalise only explicitly harmless text-format differences."""

    result = series.fillna("<NA>").astype(str)
    if "path" in column.lower() or "file" in column.lower():
        result = result.str.replace("\\", "/", regex=False)
        result = result.str.replace(
            "outputs/refactor_validation/meeting6",
            "outputs/meeting6",
            regex=False,
        )
    return result


def key_tuples(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[str, ...]]:
    """Return stable string key tuples without relying on row order."""

    normalised = pd.DataFrame(
        {
            key: normalise_text_series(frame[key], key)
            for key in keys
        }
    )
    return list(normalised.itertuples(index=False, name=None))


def empty_comparison_row(
    spec: OutputSpec,
    reference_path: Path,
    new_path: Path,
) -> dict[str, Any]:
    """Create the required comparison-report schema."""

    return {
        "reference_file": relative(reference_path),
        "new_file": relative(new_path),
        "output_category": spec.output_category,
        "required": spec.required_for_regression,
        "reference_exists": reference_path.exists(),
        "new_exists": new_path.exists(),
        "row_count_reference": "",
        "row_count_new": "",
        "column_sets_match": "",
        "key_columns": "|".join(spec.key_columns),
        "key_sets_match": "",
        "duplicate_keys_reference": "",
        "duplicate_keys_new": "",
        "text_columns_match": "",
        "boolean_columns_match": "",
        "max_absolute_numeric_difference": "",
        "max_relative_numeric_difference": "",
        "columns_with_differences": "",
        "rows_exceeding_tolerance": "",
        "comparison_status": "",
        "substantive_difference": "",
        "notes": spec.notes,
    }


def compare_csv_output(
    spec: OutputSpec,
    reference_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Perform a full keyed comparison of one CSV output."""

    reference_path = reference_root / spec.output_name
    new_path = new_root / spec.output_name
    row = empty_comparison_row(spec, reference_path, new_path)
    if not reference_path.exists() or not new_path.exists():
        row["comparison_status"] = "FAIL_MISSING_OUTPUT"
        row["substantive_difference"] = True
        row["notes"] = "A required Step 33 CSV is missing."
        return row

    reference = read_csv(reference_path)
    generated = read_csv(new_path)
    row["row_count_reference"] = len(reference)
    row["row_count_new"] = len(generated)
    column_sets_match = set(reference.columns) == set(generated.columns)
    row["column_sets_match"] = column_sets_match
    if not column_sets_match:
        row["comparison_status"] = "FAIL_VALUE_MISMATCH"
        row["substantive_difference"] = True
        row["columns_with_differences"] = "|".join(
            sorted(set(reference.columns).symmetric_difference(generated.columns))
        )
        row["notes"] = "Column sets differ."
        return row

    missing_keys = [
        key for key in spec.key_columns if key not in reference.columns
    ]
    if missing_keys:
        row["comparison_status"] = "FAIL_KEY_MISMATCH"
        row["substantive_difference"] = True
        row["notes"] = "Missing key columns: " + "|".join(missing_keys)
        return row

    reference_duplicates = int(
        reference.duplicated(list(spec.key_columns)).sum()
    )
    generated_duplicates = int(
        generated.duplicated(list(spec.key_columns)).sum()
    )
    row["duplicate_keys_reference"] = reference_duplicates
    row["duplicate_keys_new"] = generated_duplicates
    reference_keys = key_tuples(reference, spec.key_columns)
    generated_keys = key_tuples(generated, spec.key_columns)
    key_sets_match = set(reference_keys) == set(generated_keys)
    row["key_sets_match"] = key_sets_match
    if (
        not key_sets_match
        or reference_duplicates
        or generated_duplicates
        or len(reference) != len(generated)
    ):
        row["comparison_status"] = "FAIL_KEY_MISMATCH"
        row["substantive_difference"] = True
        unmatched = len(set(reference_keys).symmetric_difference(generated_keys))
        row["notes"] = f"Unmatched key tuples: {unmatched}."
        return row

    reference = reference.assign(
        __comparison_key__=reference_keys
    ).sort_values("__comparison_key__", kind="mergesort")
    generated = generated.assign(
        __comparison_key__=generated_keys
    ).sort_values("__comparison_key__", kind="mergesort")
    reference = reference.reset_index(drop=True)
    generated = generated.reset_index(drop=True)
    generated = generated[reference.columns]

    bool_columns = sorted(
        set(boolean_columns(reference)).union(boolean_columns(generated))
    )
    number_columns = [
        column
        for column in numeric_columns(reference)
        if column in numeric_columns(generated)
        and column not in bool_columns
        and column != "__comparison_key__"
    ]
    text_columns = [
        column
        for column in reference.columns
        if column not in number_columns
        and column not in bool_columns
        and column != "__comparison_key__"
    ]

    differing_columns: set[str] = set()
    differing_rows = np.zeros(len(reference), dtype=bool)

    text_match = True
    for column in text_columns:
        left = normalise_text_series(reference[column], column)
        right = normalise_text_series(generated[column], column)
        different = left.to_numpy() != right.to_numpy()
        if different.any():
            text_match = False
            differing_columns.add(column)
            differing_rows |= different
    row["text_columns_match"] = text_match

    bool_match = True
    for column in bool_columns:
        left = normalise_text_series(reference[column], column).str.lower()
        right = normalise_text_series(generated[column], column).str.lower()
        different = left.to_numpy() != right.to_numpy()
        if different.any():
            bool_match = False
            differing_columns.add(column)
            differing_rows |= different
    row["boolean_columns_match"] = bool_match

    max_absolute = 0.0
    max_relative = 0.0
    numeric_within_tolerance = True
    any_numeric_representation_difference = False
    for column in number_columns:
        left = reference[column].to_numpy(dtype=float)
        right = generated[column].to_numpy(dtype=float)
        close = np.isclose(
            left,
            right,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
            equal_nan=True,
        )
        exact = (left == right) | (np.isnan(left) & np.isnan(right))
        if not exact.all():
            any_numeric_representation_difference = True
            differing_columns.add(column)
        if not close.all():
            numeric_within_tolerance = False
            differing_rows |= ~close
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            absolute = np.abs(left[finite] - right[finite])
            denominator = np.maximum(
                np.maximum(np.abs(left[finite]), np.abs(right[finite])),
                1e-15,
            )
            relative_difference = absolute / denominator
            max_absolute = max(max_absolute, float(absolute.max()))
            max_relative = max(
                max_relative,
                float(relative_difference.max()),
            )

    row["max_absolute_numeric_difference"] = max_absolute
    row["max_relative_numeric_difference"] = max_relative
    row["columns_with_differences"] = "|".join(sorted(differing_columns))
    row["rows_exceeding_tolerance"] = int(differing_rows.sum())

    if not text_match or not bool_match or not numeric_within_tolerance:
        row["comparison_status"] = "FAIL_VALUE_MISMATCH"
        row["substantive_difference"] = True
        row["notes"] = "One or more values exceed strict comparison rules."
    elif any_numeric_representation_difference:
        row["comparison_status"] = "PASS_NUMERIC_TOLERANCE"
        row["substantive_difference"] = False
        row["notes"] = (
            "All numeric differences are within absolute and relative "
            "tolerances."
        )
    elif list(reference.columns[:-1]) != list(generated.columns[:-1]):
        row["comparison_status"] = "PASS_NORMALISED_FORMATTING"
        row["substantive_difference"] = False
        row["notes"] = "Column order was normalised."
    else:
        row["comparison_status"] = "PASS_EXACT"
        row["substantive_difference"] = False
        row["notes"] = "Full keyed comparison is exact."
    return row


def normalise_markdown(
    text: str,
    reference_root: Path,
    new_root: Path,
) -> str:
    """Normalise line endings and validation-root path prefixes only."""

    normalised = text.replace("\r\n", "\n").replace("\\", "/")
    new_relative = relative(new_root)
    reference_relative = relative(reference_root)
    return normalised.replace(new_relative, reference_relative).strip()


def extract_markdown_headlines(text: str) -> dict[str, float]:
    """Extract structured scientific values from the Step 33 summary."""

    patterns = {
        "glicko_low_brier": r"Glicko low inflation: Brier=([-+0-9.eE]+)",
        "glicko_low_log_loss": (
            r"Glicko low inflation: Brier=[-+0-9.eE]+, "
            r"log loss=([-+0-9.eE]+)"
        ),
        "glicko_low_accuracy": (
            r"Glicko low inflation: Brier=[-+0-9.eE]+, "
            r"log loss=[-+0-9.eE]+, accuracy=([-+0-9.eE]+)"
        ),
        "validation_best_elo_brier": (
            r"Validation-best Elo: Brier=([-+0-9.eE]+)"
        ),
        "validation_best_elo_log_loss": (
            r"Validation-best Elo: Brier=[-+0-9.eE]+, "
            r"log loss=([-+0-9.eE]+)"
        ),
        "validation_best_elo_accuracy": (
            r"Validation-best Elo: Brier=[-+0-9.eE]+, "
            r"log loss=[-+0-9.eE]+, accuracy=([-+0-9.eE]+)"
        ),
        "adaptive_k_brier": (
            r"Best adaptive-K Elo: Brier=([-+0-9.eE]+)"
        ),
        "adaptive_k_log_loss": (
            r"Best adaptive-K Elo: Brier=[-+0-9.eE]+, "
            r"log loss=([-+0-9.eE]+)"
        ),
        "adaptive_k_accuracy": (
            r"Best adaptive-K Elo: Brier=[-+0-9.eE]+, "
            r"log loss=[-+0-9.eE]+, accuracy=([-+0-9.eE]+)"
        ),
        "glicko_c0_brier": r"Glicko C0: Brier=([-+0-9.eE]+)",
        "glicko_c0_log_loss": (
            r"Glicko C0: Brier=[-+0-9.eE]+, "
            r"log loss=([-+0-9.eE]+)"
        ),
        "glicko_c0_accuracy": (
            r"Glicko C0: Brier=[-+0-9.eE]+, "
            r"log loss=[-+0-9.eE]+, accuracy=([-+0-9.eE]+)"
        ),
        "main_brier_improvement": (
            r"Main paired Brier improvement, Elo minus Glicko low: "
            r"([-+0-9.eE]+)"
        ),
        "main_log_loss_improvement": (
            r"Main paired log-loss improvement, Elo minus Glicko low: "
            r"([-+0-9.eE]+)"
        ),
        "inflation_brier_improvement": (
            r"Overall inflation contribution, C0 Brier minus "
            r"low-inflation Brier: ([-+0-9.eE]+)"
        ),
        "validation_checks_passed": (
            r"Validation checks passed: ([0-9]+) / [0-9]+"
        ),
        "validation_checks_total": (
            r"Validation checks passed: [0-9]+ / ([0-9]+)"
        ),
    }
    values: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Markdown headline not found: {name}")
        values[name] = float(match.group(1).rstrip("."))
    return values


def compare_markdown_output(
    reference_root: Path,
    new_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, bool]:
    """Compare the Markdown summary and its scientific headline values."""

    spec = MARKDOWN_SPEC
    reference_path = reference_root / spec.output_name
    new_path = new_root / spec.output_name
    row = empty_comparison_row(spec, reference_path, new_path)
    if not reference_path.exists() or not new_path.exists():
        row["comparison_status"] = "FAIL_MISSING_OUTPUT"
        row["substantive_difference"] = True
        return row, pd.DataFrame(), False

    reference_text = reference_path.read_text(encoding="utf-8-sig")
    generated_text = new_path.read_text(encoding="utf-8-sig")
    reference_values = extract_markdown_headlines(reference_text)
    generated_values = extract_markdown_headlines(generated_text)
    headline_rows: list[dict[str, Any]] = []
    headlines_pass = True
    for name in reference_values:
        reference_value = reference_values[name]
        generated_value = generated_values[name]
        absolute = abs(reference_value - generated_value)
        denominator = max(
            abs(reference_value),
            abs(generated_value),
            1e-15,
        )
        relative_difference = absolute / denominator
        passed = math.isclose(
            reference_value,
            generated_value,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
        headlines_pass &= passed
        headline_rows.append(
            {
                "headline": name,
                "reference_value": reference_value,
                "new_value": generated_value,
                "absolute_difference": absolute,
                "relative_difference": relative_difference,
                "passed": passed,
            }
        )

    required_statements = (
        "The debut anomaly remains after correction",
        "low inactivity RD inflation still improves Glicko relative to C0",
        "The correction is only in evaluation orientation",
        "Glicko low remains better overall than validation-best Elo",
    )
    statement_match = all(
        statement in reference_text and statement in generated_text
        for statement in required_statements
    )
    raw_equal = reference_text == generated_text
    normalised_equal = normalise_markdown(
        reference_text,
        reference_root,
        new_root,
    ) == normalise_markdown(
        generated_text,
        reference_root,
        new_root,
    )
    row["text_columns_match"] = normalised_equal
    row["boolean_columns_match"] = True
    row["max_absolute_numeric_difference"] = max(
        item["absolute_difference"] for item in headline_rows
    )
    row["max_relative_numeric_difference"] = max(
        item["relative_difference"] for item in headline_rows
    )
    row["rows_exceeding_tolerance"] = sum(
        not item["passed"] for item in headline_rows
    )
    row["columns_with_differences"] = (
        "" if headlines_pass else "structured_headline_values"
    )
    row["key_sets_match"] = True
    row["duplicate_keys_reference"] = 0
    row["duplicate_keys_new"] = 0
    passed = normalised_equal and headlines_pass and statement_match
    if not passed:
        row["comparison_status"] = "FAIL_VALUE_MISMATCH"
        row["substantive_difference"] = True
        row["notes"] = (
            "Markdown scientific text or structured headline values differ."
        )
    elif raw_equal:
        row["comparison_status"] = "PASS_EXACT"
        row["substantive_difference"] = False
        row["notes"] = "Markdown files are byte-equivalent after decoding."
    else:
        row["comparison_status"] = "PASS_NORMALISED_FORMATTING"
        row["substantive_difference"] = False
        row["notes"] = (
            "Only the protected validation-root path prefix differs."
        )
    return row, pd.DataFrame(headline_rows), passed


def inspect_image(path: Path) -> tuple[bool, int, int, bool, str]:
    """Open an image and apply conservative non-blank content checks."""

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            grayscale = image.convert("L")
            statistics = ImageStat.Stat(grayscale)
            standard_deviation = float(statistics.stddev[0])
            extrema = grayscale.getextrema()
            content_ok = (
                width > 100
                and height > 100
                and standard_deviation > 2.0
                and extrema[1] - extrema[0] > 10
            )
        return True, width, height, content_ok, (
            f"grayscale_std={standard_deviation:.3f}; extrema={extrema}"
        )
    except Exception as exc:
        return False, 0, 0, False, f"{type(exc).__name__}: {exc}"


def validate_figures(
    reference_root: Path,
    new_root: Path,
    source_status: dict[str, str],
    manual_visual_check: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate all Step 33 figures without requiring identical pixels."""

    figure_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for spec in FIGURE_SPECS:
        reference_path = reference_root / "figures" / spec.output_name
        new_path = new_root / "figures" / spec.output_name
        original_exists = reference_path.exists()
        new_exists = new_path.exists()
        nonempty = new_exists and new_path.stat().st_size > 0
        opened, width, height, content_ok, image_notes = (
            inspect_image(new_path)
            if new_exists
            else (False, 0, 0, False, "New figure is missing.")
        )
        sources = FIGURE_SOURCE_MAP[spec.output_name]
        source_validated = all(
            source_status.get(source, "").startswith("PASS_")
            for source in sources
        )
        automated_pass = (
            original_exists
            and new_exists
            and nonempty
            and opened
            and content_ok
            and source_validated
        )
        if not automated_pass or manual_visual_check == "fail":
            status = "FAIL"
        elif manual_visual_check == "pass":
            status = "PASS"
        else:
            status = "PENDING_MANUAL"
        figure_rows.append(
            {
                "figure_name": spec.output_name,
                "original_exists": original_exists,
                "new_exists": new_exists,
                "new_file_nonempty": nonempty,
                "width": width,
                "height": height,
                "source_data_validated": source_validated,
                "visual_check": manual_visual_check,
                "status": status,
                "notes": image_notes,
            }
        )
        comparison = empty_comparison_row(
            spec,
            reference_path,
            new_path,
        )
        comparison["key_sets_match"] = True
        comparison["duplicate_keys_reference"] = 0
        comparison["duplicate_keys_new"] = 0
        comparison["text_columns_match"] = True
        comparison["boolean_columns_match"] = True
        comparison["rows_exceeding_tolerance"] = 0
        if status == "PASS":
            comparison["comparison_status"] = "PASS_NORMALISED_FORMATTING"
            comparison["substantive_difference"] = False
            comparison["notes"] = (
                "Image opens, is non-blank, has valid dimensions, uses "
                "validated source data, and passed manual visual review."
            )
        elif status == "PENDING_MANUAL":
            comparison["comparison_status"] = "FAIL_VALUE_MISMATCH"
            comparison["substantive_difference"] = True
            comparison["notes"] = "Manual visual review has not been recorded."
        else:
            comparison["comparison_status"] = (
                "FAIL_MISSING_OUTPUT"
                if not new_exists
                else "FAIL_VALUE_MISMATCH"
            )
            comparison["substantive_difference"] = True
            comparison["notes"] = "Figure validation failed. " + image_notes
        comparison_rows.append(comparison)
    return pd.DataFrame(figure_rows), comparison_rows


def create_figure_contact_sheet(new_root: Path) -> Path:
    """Create an ignored contact sheet to support manual figure review."""

    figure_paths = [
        new_root / "figures" / spec.output_name
        for spec in FIGURE_SPECS
    ]
    cell_width = 900
    cell_height = 590
    label_height = 34
    columns = 2
    rows = math.ceil(len(figure_paths) / columns)
    sheet = Image.new(
        "RGB",
        (cell_width * columns, cell_height * rows),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(figure_paths):
        column = index % columns
        row = index // columns
        x = column * cell_width
        y = row * cell_height
        draw.text((x + 12, y + 10), path.name, fill="black")
        with Image.open(path) as image:
            preview = ImageOps.contain(
                image.convert("RGB"),
                (cell_width - 24, cell_height - label_height - 18),
            )
        preview_x = x + (cell_width - preview.width) // 2
        preview_y = y + label_height
        sheet.paste(preview, (preview_x, preview_y))
    output_path = (
        new_root / "figures" / "step33_validation_contact_sheet.png"
    )
    sheet.save(output_path)
    return output_path


def build_inventory(
    reference_root: Path,
    new_root: Path,
) -> pd.DataFrame:
    """Build the machine-readable inventory requested for Step 33."""

    rows: list[dict[str, Any]] = []
    for spec in ALL_SPECS:
        reference_path = (
            reference_root / "figures" / spec.output_name
            if spec.output_category == "figure"
            else reference_root / spec.output_name
        )
        new_path = (
            new_root / "figures" / spec.output_name
            if spec.output_category == "figure"
            else new_root / spec.output_name
        )
        numeric = ""
        if reference_path.suffix.lower() == ".csv" and reference_path.exists():
            numeric = "|".join(numeric_columns(read_csv(reference_path)))
        rows.append(
            {
                "output_name": spec.output_name,
                "original_path": relative(reference_path),
                "refactored_expected_path": relative(new_path),
                "output_category": spec.output_category,
                "comparison_method": spec.comparison_method,
                "key_columns": "|".join(spec.key_columns),
                "numeric_columns": numeric,
                "large_row_level_file": spec.large_row_level_file,
                "required_for_regression": spec.required_for_regression,
                "notes": spec.notes,
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(INVENTORY_PATH, index=False)
    return inventory


def headline_check_row(
    check_name: str,
    reference_value: Any,
    generated_value: Any,
    expected: Any,
    source_file: str,
    exact: bool = False,
) -> dict[str, Any]:
    """Create one explicit headline regression record."""

    if isinstance(reference_value, (bool, np.bool_)):
        passed = bool(reference_value) == bool(generated_value)
        absolute = 0.0 if passed else 1.0
    elif isinstance(reference_value, str):
        passed = reference_value == str(generated_value)
        absolute = 0.0 if passed else float("inf")
    else:
        left = float(reference_value)
        right = float(generated_value)
        passed = (
            left == right
            if exact
            else math.isclose(
                left,
                right,
                rel_tol=RELATIVE_TOLERANCE,
                abs_tol=ABSOLUTE_TOLERANCE,
            )
        )
        absolute = abs(left - right)
    return {
        "check_name": check_name,
        "reference_value": reference_value,
        "new_value": generated_value,
        "expected": expected,
        "absolute_difference": absolute,
        "passed": passed,
        "source_file": source_file,
    }


def build_headline_checks(
    reference_root: Path,
    new_root: Path,
) -> pd.DataFrame:
    """Verify every explicitly required Step 33 headline result."""

    reference_metrics = read_csv(
        reference_root / "33_overall_model_metrics.csv"
    ).set_index("model")
    generated_metrics = read_csv(
        new_root / "33_overall_model_metrics.csv"
    ).set_index("model")
    reference_pairwise = read_csv(
        reference_root / "33_overall_pairwise_comparisons.csv"
    ).set_index("comparison")
    generated_pairwise = read_csv(
        new_root / "33_overall_pairwise_comparisons.csv"
    ).set_index("comparison")
    reference_results = read_csv(
        reference_root / "33_meeting6_final_results.csv"
    ).set_index("subgroup")
    generated_results = read_csv(
        new_root / "33_meeting6_final_results.csv"
    ).set_index("subgroup")
    reference_validation = read_csv(
        reference_root / "33_final_validation_checks.csv"
    )
    generated_validation = read_csv(
        new_root / "33_final_validation_checks.csv"
    )
    reference_scores = read_csv(
        reference_root / "33_orientation_corrected_per_match_scores_2025.csv"
    )
    generated_scores = read_csv(
        new_root / "33_orientation_corrected_per_match_scores_2025.csv"
    )

    rows: list[dict[str, Any]] = []
    metric_specs = (
        (
            "evaluation_games",
            "Glicko_low_fixed",
            "evaluation_games",
            EXPECTED_GAMES,
            True,
        ),
        (
            "glicko_low_brier",
            "Glicko_low_fixed",
            "brier",
            "original golden; approximately 0.187604",
            False,
        ),
        (
            "glicko_low_log_loss",
            "Glicko_low_fixed",
            "log_loss",
            "original golden; approximately 0.551779",
            False,
        ),
        (
            "glicko_low_accuracy",
            "Glicko_low_fixed",
            "accuracy",
            "original golden; approximately 0.711486",
            False,
        ),
        (
            "validation_best_elo_brier",
            "Validation_best_Elo",
            "brier",
            "original golden; approximately 0.190073",
            False,
        ),
        (
            "validation_best_elo_log_loss",
            "Validation_best_Elo",
            "log_loss",
            "original golden; approximately 0.556534",
            False,
        ),
        (
            "validation_best_elo_accuracy",
            "Validation_best_Elo",
            "accuracy",
            "original golden; approximately 0.704456",
            False,
        ),
        (
            "glicko_c0_brier",
            "Glicko_C0_fixed",
            "brier",
            "original golden; approximately 0.195708",
            False,
        ),
    )
    for check_name, model, column, expected, exact in metric_specs:
        rows.append(
            headline_check_row(
                check_name,
                reference_metrics.loc[model, column],
                generated_metrics.loc[model, column],
                expected,
                "33_overall_model_metrics.csv",
                exact,
            )
        )

    comparison_name = "Glicko low fixed vs validation-best Elo"
    rows.append(
        headline_check_row(
            "main_elo_minus_glicko_brier_improvement",
            reference_pairwise.loc[comparison_name, "delta_brier"],
            generated_pairwise.loc[comparison_name, "delta_brier"],
            "original golden; approximately 0.002469",
            "33_overall_pairwise_comparisons.csv",
        )
    )
    for check_name, subgroup, expected in (
        ("exactly_one_debut_matches", "Exactly one debut", 72),
        ("no_debut_matches", "Overall excluding debut", 11_305),
    ):
        rows.append(
            headline_check_row(
                check_name,
                reference_results.loc[subgroup, "games"],
                generated_results.loc[subgroup, "games"],
                expected,
                "33_meeting6_final_results.csv",
                True,
            )
        )

    probability_columns = [
        column
        for column in generated_scores.columns
        if column.startswith("p_a_")
    ]
    probabilities_in_range_reference = bool(
        (
            reference_scores[probability_columns].ge(0.0)
            & reference_scores[probability_columns].le(1.0)
        ).all().all()
    )
    probabilities_in_range_generated = bool(
        (
            generated_scores[probability_columns].ge(0.0)
            & generated_scores[probability_columns].le(1.0)
        ).all().all()
    )
    rows.append(
        headline_check_row(
            "all_core_probabilities_in_unit_interval",
            probabilities_in_range_reference,
            probabilities_in_range_generated,
            True,
            "33_orientation_corrected_per_match_scores_2025.csv",
            True,
        )
    )

    orientation_reference = reference_validation.loc[
        reference_validation["check_name"].eq(
            "fixed_probability_definition_does_not_use_outcome"
        ),
        "passed",
    ].iloc[0]
    orientation_generated = generated_validation.loc[
        generated_validation["check_name"].eq(
            "fixed_probability_definition_does_not_use_outcome"
        ),
        "passed",
    ].iloc[0]
    rows.append(
        headline_check_row(
            "fixed_direct_player_a_orientation",
            bool(orientation_reference),
            bool(orientation_generated),
            True,
            "33_final_validation_checks.csv",
            True,
        )
    )
    rows.append(
        headline_check_row(
            "all_step33_final_validation_checks_pass",
            bool(reference_validation["passed"].astype(bool).all()),
            bool(generated_validation["passed"].astype(bool).all()),
            True,
            "33_final_validation_checks.csv",
            True,
        )
    )
    checks = pd.DataFrame(rows)
    checks.to_csv(HEADLINE_CHECK_PATH, index=False)
    return checks


def report_numeric_maximum(
    comparison: pd.DataFrame,
    column: str,
) -> float:
    """Return the maximum finite numeric comparison value."""

    values = pd.to_numeric(comparison[column], errors="coerce")
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else 0.0


def write_summary(
    command_display: str,
    runtime: float | None,
    reference_root: Path,
    new_root: Path,
    inventory: pd.DataFrame,
    comparison: pd.DataFrame,
    headline_checks: pd.DataFrame,
    markdown_headlines: pd.DataFrame,
    figure_validation: pd.DataFrame,
) -> bool:
    """Write the complete regression summary and return equivalence status."""

    statuses = comparison["comparison_status"].value_counts().to_dict()
    required = comparison.loc[comparison["required"].astype(bool)]
    substantive = required["substantive_difference"].astype(bool)
    failed = required.loc[substantive]
    exact_passes = int(statuses.get("PASS_EXACT", 0))
    tolerance_passes = int(statuses.get("PASS_NUMERIC_TOLERANCE", 0))
    formatting_passes = int(statuses.get("PASS_NORMALISED_FORMATTING", 0))
    historical_only = int(
        statuses.get("NOT_REGENERATED_HISTORICAL_ONLY", 0)
    )
    max_absolute = report_numeric_maximum(
        comparison,
        "max_absolute_numeric_difference",
    )
    max_relative = report_numeric_maximum(
        comparison,
        "max_relative_numeric_difference",
    )
    all_headlines_pass = bool(headline_checks["passed"].astype(bool).all())
    markdown_headlines_pass = bool(
        markdown_headlines["passed"].astype(bool).all()
    )
    figures_pass = bool(figure_validation["status"].eq("PASS").all())
    equivalent = (
        failed.empty
        and all_headlines_pass
        and markdown_headlines_pass
        and figures_pass
    )
    runtime_text = "not rerun" if runtime is None else f"{runtime:.6f} seconds"
    lines = [
        "# Complete Step 33 Refactor Regression",
        "",
        "## Execution",
        "",
        f"- Command: `{command_display}`",
        f"- Runtime: {runtime_text}",
        f"- Reference root: `{relative(reference_root)}`",
        f"- Regenerated root: `{relative(new_root)}`",
        f"- Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}",
        f"- Random seed: {RANDOM_SEED}",
        (
            "- Probability convention: direct player-A probability, "
            "`expected_score(rating_A, rating_B, RD_B)`."
        ),
        "",
        "## Coverage",
        "",
        f"- Outputs expected: {len(inventory)}",
        (
            "- Outputs regenerated: "
            f"{int(comparison['new_exists'].astype(bool).sum())}"
        ),
        f"- Outputs compared: {len(comparison)}",
        f"- Exact passes: {exact_passes}",
        f"- Numeric-tolerance passes: {tolerance_passes}",
        f"- Normalised-formatting or figure passes: {formatting_passes}",
        f"- Historical-only outputs: {historical_only}",
        f"- Failed or missing outputs: {len(failed)}",
        "",
        "## Numerical Equivalence",
        "",
        f"- Maximum absolute numerical difference: {max_absolute:.17g}",
        f"- Maximum relative numerical difference: {max_relative:.17g}",
        (
            "- All required key sets match: "
            f"{bool(required['key_sets_match'].replace('', True).astype(bool).all())}"
        ),
        (
            "- All bootstrap outputs match: "
            f"{not bool(required.loc[required['output_category'].eq('bootstrap'), 'substantive_difference'].astype(bool).any())}"
        ),
        f"- All explicit headline checks pass: {all_headlines_pass}",
        f"- All Markdown headline checks pass: {markdown_headlines_pass}",
        f"- All figures pass automated and manual checks: {figures_pass}",
        "",
        "## Conclusion",
        "",
        (
            "The refactored comparison pipeline is scientifically equivalent "
            "to the original Step 33 workflow."
            if equivalent
            else (
                "Scientific equivalence is not established. Review failed "
                "rows before merge."
            )
        ),
        "",
    ]
    if len(failed):
        lines.extend(
            [
                "## Failed Outputs",
                "",
                *[
                    (
                        f"- `{Path(row.new_file).name}`: "
                        f"{row.comparison_status}; {row.notes}"
                    )
                    for row in failed.itertuples(index=False)
                ],
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return equivalent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse validation command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate and compare every active Step 33 output without "
            "overwriting golden files."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Directory containing protected original Step 33 outputs.",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=DEFAULT_NEW_ROOT,
        help="Directory containing regenerated Step 33 outputs.",
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=DEFAULT_RUN_OUTPUT_ROOT,
        help="Parent output root passed to code.cli compare-models.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full active Step 33 pipeline before comparing.",
    )
    parser.add_argument(
        "--manual-visual-check",
        choices=("pending", "pass", "fail"),
        default="pending",
        help="Record the result of human visual inspection of all figures.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete Step 33 regression workflow."""

    args = parse_args(argv)
    reference_root = args.reference_root.resolve()
    new_root = args.new_root.resolve()
    run_output_root = args.run_output_root.resolve()
    ensure_safe_roots(
        reference_root,
        new_root,
        run_output_root,
        args.run,
    )

    command_display = (
        "python -m code.cli compare-models --full-run "
        f"--output-root {relative(run_output_root)}"
    )
    runtime: float | None = None
    if args.run:
        command_display, runtime = run_step33_pipeline(
            reference_root,
            new_root,
            run_output_root,
        )
    elif not new_root.exists():
        raise FileNotFoundError(
            f"Regenerated output root does not exist: {new_root}"
        )
    else:
        command_display, runtime = read_recorded_run_metadata(
            command_display
        )

    inventory = build_inventory(reference_root, new_root)
    comparison_rows = [
        compare_csv_output(spec, reference_root, new_root)
        for spec in CSV_SPECS
    ]
    markdown_row, markdown_headlines, _ = compare_markdown_output(
        reference_root,
        new_root,
    )
    comparison_rows.append(markdown_row)

    source_status = {
        Path(row["new_file"]).name: str(row["comparison_status"])
        for row in comparison_rows
    }
    figure_validation, figure_comparisons = validate_figures(
        reference_root,
        new_root,
        source_status,
        args.manual_visual_check,
    )
    contact_sheet_path = create_figure_contact_sheet(new_root)
    comparison_rows.extend(figure_comparisons)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)
    figure_validation.to_csv(FIGURE_VALIDATION_PATH, index=False)
    markdown_headlines.to_csv(MARKDOWN_HEADLINE_PATH, index=False)
    headline_checks = build_headline_checks(reference_root, new_root)

    equivalent = write_summary(
        command_display,
        runtime,
        reference_root,
        new_root,
        inventory,
        comparison,
        headline_checks,
        markdown_headlines,
        figure_validation,
    )
    failed_outputs = int(
        comparison["substantive_difference"].astype(bool).sum()
    )
    print(f"Step 33 outputs inventoried: {len(inventory)}")
    print(f"Step 33 outputs compared: {len(comparison)}")
    print(f"Substantive output failures: {failed_outputs}")
    print(
        "Headline checks passed: "
        f"{int(headline_checks['passed'].astype(bool).sum())}/"
        f"{len(headline_checks)}"
    )
    print(
        "Figure checks passed: "
        f"{int(figure_validation['status'].eq('PASS').sum())}/"
        f"{len(figure_validation)}"
    )
    print(f"Scientific equivalence: {equivalent}")
    print(f"Figure contact sheet: {contact_sheet_path}")
    print(f"Comparison report: {COMPARISON_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0 if equivalent else 1


if __name__ == "__main__":
    sys.exit(main())
