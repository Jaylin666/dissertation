"""Command-line entry points for the canonical dissertation code."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Callable, Sequence

import pandas as pd

from code.config import (
    BOOTSTRAP_REPETITIONS,
    ELO_CONFIGURATIONS,
    EXPECTED_FULL_HISTORY_MATCHES,
    EXPECTED_TEST_MATCHES,
    EXPECTED_UNIQUE_PLAYERS,
    FIRST_APPEARANCE_GOLDEN,
    FULL_HISTORY_END_YEAR,
    FULL_HISTORY_START_YEAR,
    GLICKO_LOW_INFLATION,
    TEST_YEAR,
    VALIDATION_YEARS,
)
from code.io_utils import PROJECT_ROOT
from code.validation_utils import robust_bool


DEFAULT_VALIDATION_ROOT = "outputs/refactor_validation"
EVIDENCE_ROOT = PROJECT_ROOT / "outputs" / "dissertation_evidence"
ARCHIVED_OUTPUT_ROOT = PROJECT_ROOT / "archive" / "research_outputs"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run the full command instead of validation-only checks.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_VALIDATION_ROOT,
        help="Root for generated validation artifacts.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m code.cli",
        description=(
            "Canonical entry points for the frozen Elo-Glicko dissertation "
            "analysis. Commands default to validation-only mode."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("build-data", "Build or validate the checked 1985-2025 game dataset."),
        ("run-elo", "Run or validate the frozen Elo configurations."),
        ("run-glicko", "Run or validate the frozen Glicko configurations."),
        ("compare-models", "Recompute or validate orientation-corrected comparisons."),
        ("early-game", "Run or validate the focal-player early-game workflow."),
        ("entry-diagnostics", "Run or validate burn-in and entry-scale diagnostics."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_run_options(command_parser)

    subparsers.add_parser(
        "validate",
        help="Run lightweight checks against tracked compact golden outputs.",
    )
    return parser


def _print_run_header(
    command: str,
    mode: str,
    inputs: Sequence[Path | str],
    outputs: Sequence[Path | str],
    parameters: Sequence[str],
) -> None:
    print(f"Command: {command}")
    print(f"Mode: {mode}")
    print("Inputs:")
    for item in inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in outputs:
        print(f"  - {item}")
    print("Parameter configuration:")
    for item in parameters:
        print(f"  - {item}")


def _validation_files() -> list[Path]:
    return [
        ARCHIVED_OUTPUT_ROOT / "meeting6" / "33_canonical_player_orientation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting6" / "33_final_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting7" / "34_input_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting7" / "34_metric_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting7" / "34_bootstrap_figure_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting7" / "34_bootstrap_method_audit_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting7" / "34_bootstrap_robustness_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting8_technical" / "41_validation_checks.csv",
        ARCHIVED_OUTPUT_ROOT / "meeting8_technical" / "42_validation_checks.csv",
    ]


def validate_compact_outputs() -> bool:
    """Validate tracked error-level checks without rerunning expensive models."""

    failures: list[str] = []
    total_error_checks = 0
    for path in _validation_files():
        if not path.exists():
            failures.append(f"missing validation file: {path}")
            continue
        checks = pd.read_csv(path)
        if "passed" not in checks.columns and "status" not in checks.columns:
            failures.append(f"missing passed/status column: {path}")
            continue
        if "severity" in checks.columns:
            error_mask = checks["severity"].fillna("error").astype(str).str.lower().eq("error")
        else:
            error_mask = pd.Series(True, index=checks.index)
        error_checks = checks.loc[error_mask].copy()
        total_error_checks += len(error_checks)
        for row in error_checks.itertuples(index=False):
            if hasattr(row, "passed"):
                try:
                    passed = robust_bool(getattr(row, "passed"))
                except ValueError:
                    passed = False
            else:
                passed = str(getattr(row, "status")).strip().upper() == "PASS"
            if not passed:
                name = getattr(row, "check_name", "unnamed_check")
                failures.append(f"{path.name}: {name}")

    overall_path = EVIDENCE_ROOT / "chapter4" / "overall_model_metrics.csv"
    overall = pd.read_csv(overall_path)
    if not (overall["evaluation_games"].astype(int) == EXPECTED_TEST_MATCHES).all():
        failures.append("Step 33 evaluation row counts differ from 11,379")

    entry_path = EVIDENCE_ROOT / "chapter5" / "entry_cohort_definitions_core.csv"
    entry = pd.read_csv(entry_path)
    test_rows = entry[entry["cohort"] == "test_year_recorded_entry"]
    if len(test_rows) != 1 or int(test_rows.iloc[0]["players"]) != 76:
        failures.append("Step 42 test-year entrant count differs from 76")

    first_path = EVIDENCE_ROOT / "chapter5" / "first_appearance_mechanism_core.csv"
    first = pd.read_csv(first_path)
    first_rows = first[first["model"] == "Glicko low inflation"]
    if len(first_rows) != 1:
        failures.append("Chapter 5 Glicko first-appearance row is missing or duplicated")
    else:
        first_row = first_rows.iloc[0]
        if int(first_row["appearances"]) != 76:
            failures.append("Chapter 5 first-appearance count differs from 76")
        if not math.isclose(
            float(first_row["brier"]),
            FIRST_APPEARANCE_GOLDEN.brier_score,
            abs_tol=1e-6,
        ):
            failures.append("Chapter 5 first-appearance Brier score changed")

    print(f"Tracked error-level checks inspected: {total_error_checks}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"  - {failure}")
    if not failures:
        print("Validation result: PASS")
    else:
        print("Validation result: FAIL")
    return not failures


def command_build_data(args: argparse.Namespace) -> int:
    from code.data import build_matches

    output = Path(args.output_root) / "elo_optimization"
    _print_run_header(
        "build-data",
        "full run" if args.full_run else "validation-only",
        [PROJECT_ROOT / "data_raw"],
        [output],
        [
            f"years={FULL_HISTORY_START_YEAR}-{FULL_HISTORY_END_YEAR}",
            f"expected_games={EXPECTED_FULL_HISTORY_MATCHES}",
            f"expected_players={EXPECTED_UNIQUE_PLAYERS}",
            "missing parsed dates are retained",
            "stable chronological ordering",
        ],
    )
    if args.full_run:
        build_matches.configure_output_root(args.output_root)
        build_matches.main()
    else:
        matches = pd.read_csv(build_matches.MATCHES_OUTPUT_PATH, low_memory=False)
        build_matches.validate_canonical_counts(matches)
        print("Existing checked-game dataset counts: PASS")
    return 0


def command_run_elo(args: argparse.Namespace) -> int:
    from code.pipelines import elo_pipeline

    _print_run_header(
        "run-elo",
        "full run" if args.full_run else "validation-only",
        [elo_pipeline.FULL_HISTORY_PATH],
        [Path(args.output_root) / "elo_pipeline"],
        [
            f"years={FULL_HISTORY_START_YEAR}-{FULL_HISTORY_END_YEAR}",
            f"test_year={TEST_YEAR}",
            "configs="
            + ", ".join(
                f"{item.name}(K={item.k_factor},scale={item.scale})"
                for item in ELO_CONFIGURATIONS
            ),
        ],
    )
    if args.full_run:
        paths = elo_pipeline.run_pipeline(args.output_root)
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0
    return 0 if validate_compact_outputs() else 1


def command_run_glicko(args: argparse.Namespace) -> int:
    from code.pipelines import glicko_pipeline

    _print_run_header(
        "run-glicko",
        "full run" if args.full_run else "validation-only",
        [
            PROJECT_ROOT
            / "outputs"
            / "elo_optimization"
            / "matches_1985_2025_checked.csv"
        ],
        [Path(args.output_root) / "meeting5_glicko_rd_inflation"],
        [
            f"initial_rating={GLICKO_LOW_INFLATION.initial_rating}",
            f"initial_rd={GLICKO_LOW_INFLATION.initial_rd}",
            f"rd_bounds={GLICKO_LOW_INFLATION.minimum_rd}-{GLICKO_LOW_INFLATION.maximum_rd}",
            f"low_inflation_C={GLICKO_LOW_INFLATION.inactivity_c:.12f}",
            "rating_period=one_game_per_period",
        ],
    )
    if args.full_run:
        glicko_pipeline.configure_output_root(args.output_root)
        glicko_pipeline.main()
        return 0
    return 0 if validate_compact_outputs() else 1


def command_compare_models(args: argparse.Namespace) -> int:
    from code.pipelines import comparison_pipeline

    _print_run_header(
        "compare-models",
        "full run" if args.full_run else "validation-only",
        [
            PROJECT_ROOT / "outputs" / "meeting6" / "29_per_match_model_scores_2025.csv",
            PROJECT_ROOT / "outputs" / "meeting6" / "32_glicko_direct_probability_comparison.csv",
        ],
        [Path(args.output_root) / "meeting6"],
        [
            "formal Glicko convention=direct player-A probability",
            f"bootstrap_repetitions={BOOTSTRAP_REPETITIONS}",
            f"test_games={EXPECTED_TEST_MATCHES}",
        ],
    )
    if args.full_run:
        comparison_pipeline.configure_output_root(args.output_root)
        comparison_pipeline.main()
        return 0
    return 0 if validate_compact_outputs() else 1


def command_early_game(args: argparse.Namespace) -> int:
    from code.analysis import early_game

    _print_run_header(
        "early-game",
        "full run" if args.full_run else "validation-only",
        [
            PROJECT_ROOT
            / "outputs"
            / "meeting6"
            / "33_orientation_corrected_per_match_scores_2025.csv"
        ],
        [Path(args.output_root) / "meeting7"],
        [
            "probability_orientation=focal player",
            "cumulative_groups=first 1, 5, 10, 20, 30, 50",
            "stages=1; 2-5; 6-10; 11-20; 21-50; 51+",
            f"bootstrap_repetitions={BOOTSTRAP_REPETITIONS}",
        ],
    )
    if args.full_run:
        early_game.configure_output_root(args.output_root)
        early_game.main()
        return 0
    return 0 if validate_compact_outputs() else 1


def command_entry_diagnostics(args: argparse.Namespace) -> int:
    from code.analysis import entry_diagnostics

    _print_run_header(
        "entry-diagnostics",
        "full run" if args.full_run else "validation-only",
        [
            PROJECT_ROOT
            / "outputs"
            / "elo_optimization"
            / "matches_1985_2025_checked.csv",
            PROJECT_ROOT
            / "outputs"
            / "meeting6"
            / "33_orientation_corrected_per_match_scores_2025.csv",
            PROJECT_ROOT
            / "outputs"
            / "meeting7"
            / "34_early_game_appearance_dataset.csv",
        ],
        [Path(args.output_root) / "meeting8_technical"],
        [
            "model_start_year=1985",
            "post_burn_in_start_year=1990",
            f"test_year={TEST_YEAR}",
            f"low_inflation_C={GLICKO_LOW_INFLATION.inactivity_c:.12f}",
            "Step 42 strict classification is primary",
            "Step 41 end-of-year scale summaries are retained",
        ],
    )
    if args.full_run:
        entry_diagnostics.run_all_entry_diagnostics(args.output_root)
        return 0
    return 0 if validate_compact_outputs() else 1


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build-data": command_build_data,
    "run-elo": command_run_elo,
    "run-glicko": command_run_glicko,
    "compare-models": command_compare_models,
    "early-game": command_early_game,
    "entry-diagnostics": command_entry_diagnostics,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        _print_run_header(
            "validate",
            "validation-only",
            _validation_files(),
            ["console validation report"],
            [
                f"years={FULL_HISTORY_START_YEAR}-{FULL_HISTORY_END_YEAR}",
                f"validation_years={list(VALIDATION_YEARS)}",
                f"test_year={TEST_YEAR}",
            ],
        )
        return 0 if validate_compact_outputs() else 1
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
