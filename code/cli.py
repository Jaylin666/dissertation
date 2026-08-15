"""Command-line interface for supported dissertation workflows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
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
    TEST_YEAR,
    VALIDATION_YEARS,
)
from code.io_utils import PROJECT_ROOT
from code.validation_utils import compare_reproduction_to_evidence, robust_bool


DEFAULT_OUTPUT_ROOT = "outputs/reproduction"
DEFAULT_MATCHES_PATH = "outputs/elo_optimization/matches_1985_2025_checked.csv"
EVIDENCE_ROOT = PROJECT_ROOT / "outputs" / "dissertation_evidence"
VALIDATION_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "validation"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Execute the workflow. Without this flag, only paths and parameters are shown.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Ignored root for generated reproduction artifacts.",
    )
    parser.add_argument(
        "--matches-path",
        default=DEFAULT_MATCHES_PATH,
        help="Checked historical game table.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Explicit unified comparison input when required.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m code.cli",
        description="Supported reproduction commands for the frozen dissertation analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = (
        ("build-data", "Build the checked 1985-2025 game dataset."),
        ("select-elo", "Run the frozen 2015-2024 Elo validation grid."),
        ("run-elo", "Run the three fixed Elo histories."),
        ("run-glicko", "Run the four Glicko inactivity variants."),
        ("glicko-periods", "Run the C=0 rating-period sensitivity."),
        ("run-adaptive-elo", "Run the four frozen adaptive-K candidates."),
        ("build-comparison-inputs", "Build leakage-free 2025 comparison inputs."),
        ("compare-models", "Run comparison, calibration, and event bootstrap."),
        ("early-game", "Run appearance and unique-game early-game analyses."),
        ("initial-rating-sensitivity", "Run the common-origin Glicko sensitivity."),
        ("entry-diagnostics", "Run recorded-entry and prematch-scale diagnostics."),
        ("reproduce-dissertation", "Run all supported workflows in order."),
    )
    for command, help_text in commands:
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_run_options(command_parser)
    subparsers.add_parser("validate", help="Validate tracked compact evidence.")
    return parser


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _comparison_input(args: argparse.Namespace) -> Path:
    return _path(
        args.input_path
        or Path(args.output_root) / "comparison_inputs" / "comparison_input_2025.csv"
    )


def _print_run_header(
    command: str,
    full_run: bool,
    inputs: Sequence[Path | str],
    outputs: Sequence[Path | str],
    parameters: Sequence[str],
) -> None:
    print(f"Command: {command}")
    print(f"Mode: {'full run' if full_run else 'preview'}")
    print("Inputs:")
    for item in inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in outputs:
        print(f"  - {item}")
    print("Parameters:")
    for item in parameters:
        print(f"  - {item}")


def _validation_files() -> list[Path]:
    return [
        VALIDATION_FIXTURE_ROOT / "player_orientation_checks.csv",
        VALIDATION_FIXTURE_ROOT / "model_comparison_checks.csv",
        VALIDATION_FIXTURE_ROOT / "early_game_input_checks.csv",
        VALIDATION_FIXTURE_ROOT / "early_game_metric_checks.csv",
        VALIDATION_FIXTURE_ROOT / "early_game_bootstrap_figure_checks.csv",
        VALIDATION_FIXTURE_ROOT / "early_game_bootstrap_method_checks.csv",
        VALIDATION_FIXTURE_ROOT / "early_game_bootstrap_robustness_checks.csv",
        VALIDATION_FIXTURE_ROOT / "entry_classification_checks.csv",
        VALIDATION_FIXTURE_ROOT / "entry_evidence_checks.csv",
    ]


def validate_compact_outputs() -> bool:
    """Validate tracked compact outputs without rerunning models."""

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
        error_mask = (
            checks["severity"].fillna("error").astype(str).str.lower().eq("error")
            if "severity" in checks.columns
            else pd.Series(True, index=checks.index)
        )
        error_checks = checks.loc[error_mask]
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
                failures.append(f"{path.name}: {getattr(row, 'check_name', 'unnamed_check')}")

    overall = pd.read_csv(EVIDENCE_ROOT / "chapter4" / "overall_model_metrics.csv")
    if not (overall["evaluation_games"].astype(int) == EXPECTED_TEST_MATCHES).all():
        failures.append("Chapter 4 evaluation row counts differ from 11,379")
    entry = pd.read_csv(EVIDENCE_ROOT / "chapter5" / "entry_cohort_definitions_core.csv")
    test_rows = entry.loc[entry["cohort"].eq("test_year_recorded_entry")]
    if len(test_rows) != 1 or int(test_rows.iloc[0]["players"]) != 76:
        failures.append("Chapter 5 test-year entrant count differs from 76")
    first = pd.read_csv(EVIDENCE_ROOT / "chapter5" / "first_appearance_mechanism_core.csv")
    first_rows = first.loc[first["model"].eq("Glicko low inflation")]
    if len(first_rows) != 1:
        failures.append("Chapter 5 Glicko first-appearance row is missing or duplicated")
    else:
        row = first_rows.iloc[0]
        if int(row["appearances"]) != 76:
            failures.append("Chapter 5 first-appearance count differs from 76")
        if not math.isclose(float(row["brier"]), FIRST_APPEARANCE_GOLDEN.brier_score, abs_tol=1e-6):
            failures.append("Chapter 5 first-appearance Brier score changed")

    print(f"Tracked error-level checks inspected: {total_error_checks}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"Validation result: {'PASS' if not failures else 'FAIL'}")
    return not failures


def _show_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        print(f"{label}: {path}")


def command_build_data(args: argparse.Namespace) -> int:
    from code.data import build_matches

    destination = _path(args.output_root) / "elo_optimization"
    _print_run_header(
        "build-data",
        args.full_run,
        [PROJECT_ROOT / "data_raw"],
        [destination],
        [
            f"years={FULL_HISTORY_START_YEAR}-{FULL_HISTORY_END_YEAR}",
            f"expected_games={EXPECTED_FULL_HISTORY_MATCHES}",
            f"expected_players={EXPECTED_UNIQUE_PLAYERS}",
            "stable chronological ordering with undated records retained",
        ],
    )
    if args.full_run:
        build_matches.configure_output_root(args.output_root)
        build_matches.main()
    return 0


def command_select_elo(args: argparse.Namespace) -> int:
    from code.pipelines import elo_validation

    _print_run_header(
        "select-elo",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "elo_validation"],
        ["history=2015-2022", "validation=2023-2024", "grid=7 K values x 4 scales", "2025 excluded from selection"],
    )
    if args.full_run:
        _show_paths(elo_validation.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_run_elo(args: argparse.Namespace) -> int:
    from code.pipelines import elo_pipeline

    _print_run_header(
        "run-elo",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "elo_pipeline"],
        [", ".join(f"{item.name}(K={item.k_factor}, S={item.scale})" for item in ELO_CONFIGURATIONS)],
    )
    if args.full_run:
        _show_paths(elo_pipeline.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_run_glicko(args: argparse.Namespace) -> int:
    from code.pipelines import glicko_pipeline

    _print_run_header(
        "run-glicko",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "glicko_inflation"],
        [
            "initial_rating=1500",
            "initial_rd=350",
            "variants=C0, low, medium, high inflation",
            "rating_period=one game",
        ],
    )
    if args.full_run:
        _show_paths(glicko_pipeline.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_glicko_periods(args: argparse.Namespace) -> int:
    from code.pipelines import glicko_rating_period

    _print_run_header(
        "glicko-periods",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "glicko_rating_period"],
        ["C=0", "periods=one game, event, month, year", "grouped predictions use period-opening state"],
    )
    if args.full_run:
        _show_paths(glicko_rating_period.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_run_adaptive(args: argparse.Namespace) -> int:
    from code.pipelines import adaptive_k_pipeline

    _print_run_header(
        "run-adaptive-elo",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "adaptive_k"],
        ["candidates=total-games and previous-year rules at S=500 and S=300", "retained comparator=previous-year rule, S=300"],
    )
    if args.full_run:
        _show_paths(adaptive_k_pipeline.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_build_comparison_inputs(args: argparse.Namespace) -> int:
    from code.pipelines import comparison_inputs

    root = _path(args.output_root)
    _print_run_header(
        "build-comparison-inputs",
        args.full_run,
        [
            _path(args.matches_path),
            root / "elo_pipeline" / "elo_predictions_2025.csv",
            root / "glicko_inflation" / "glicko_inflation_predictions_2025.csv",
            root / "adaptive_k" / "adaptive_k_predictions_2025.csv",
        ],
        [root / "comparison_inputs"],
        ["Player A=smaller database ID", "all activity features are pre-game", "formal Glicko probability uses opponent RD"],
    )
    if args.full_run:
        _show_paths(comparison_inputs.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_compare_models(args: argparse.Namespace) -> int:
    from code.pipelines import comparison_pipeline

    source = _comparison_input(args)
    _print_run_header(
        "compare-models",
        args.full_run,
        [source],
        [_path(args.output_root) / "comparison"],
        ["Brier and log loss primary", f"event-cluster bootstrap={BOOTSTRAP_REPETITIONS}", "fixed Player-A probabilities"],
    )
    if args.full_run:
        _show_paths(comparison_pipeline.run_pipeline(source, args.output_root))
    return 0


def command_early_game(args: argparse.Namespace) -> int:
    from code.analysis import early_game

    source = _comparison_input(args)
    _print_run_header(
        "early-game",
        args.full_run,
        [source],
        [_path(args.output_root) / "early_game"],
        ["ranges=1,5,10,20,30,50", "stages=1;2-5;6-10;11-20;21-50;51+", "formal intervals use unique games clustered by event"],
    )
    if args.full_run:
        _show_paths(early_game.run_pipeline(source, args.output_root))
    return 0


def command_initial_rating(args: argparse.Namespace) -> int:
    from code.analysis import initial_rating_sensitivity

    _print_run_header(
        "initial-rating-sensitivity",
        args.full_run,
        [_path(args.matches_path)],
        [_path(args.output_root) / "initial_rating_sensitivity"],
        ["common origins=1000,1100,1200,1300,1400,1500", "full low-inflation history rebuilt for each origin"],
    )
    if args.full_run:
        _show_paths(initial_rating_sensitivity.run_pipeline(args.matches_path, args.output_root))
    return 0


def command_entry_diagnostics(args: argparse.Namespace) -> int:
    from code.analysis import entry_diagnostics

    source = _comparison_input(args)
    _print_run_header(
        "entry-diagnostics",
        args.full_run,
        [_path(args.matches_path), source],
        [_path(args.output_root) / "entry_diagnostics"],
        ["burn-in sensitivity=1,3,5,10 years", "direct-focal orientation is sensitivity only", "fixed entry anchor=1500"],
    )
    if args.full_run:
        _show_paths(entry_diagnostics.run_pipeline(source, args.matches_path, args.output_root))
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    root = _path(args.output_root)
    generated_matches = root / "elo_optimization" / "matches_1985_2025_checked.csv"
    _print_run_header(
        "reproduce-dissertation",
        args.full_run,
        [PROJECT_ROOT / "data_raw"],
        [root],
        ["eleven supported workflows in frozen order", "no tracked evidence is overwritten", "no parameter search beyond the frozen grid"],
    )
    if not args.full_run:
        return 0

    from code.data import build_matches
    from code.pipelines import (
        adaptive_k_pipeline,
        comparison_inputs,
        comparison_pipeline,
        elo_pipeline,
        elo_validation,
        glicko_pipeline,
        glicko_rating_period,
    )
    from code.analysis import early_game, entry_diagnostics, initial_rating_sensitivity

    started = time.perf_counter()
    build_matches.configure_output_root(root)
    build_matches.main()
    elo_validation.run_pipeline(generated_matches, root)
    elo_pipeline.run_pipeline(generated_matches, root)
    glicko_pipeline.run_pipeline(generated_matches, root)
    glicko_rating_period.run_pipeline(generated_matches, root)
    adaptive_k_pipeline.run_pipeline(generated_matches, root)
    input_paths = comparison_inputs.run_pipeline(generated_matches, root)
    comparison_pipeline.run_pipeline(input_paths["comparison"], root)
    early_game.run_pipeline(input_paths["comparison"], root)
    initial_rating_sensitivity.run_pipeline(generated_matches, root)
    entry_diagnostics.run_pipeline(input_paths["comparison"], generated_matches, root)
    regression = compare_reproduction_to_evidence(root)
    regression_path = root / "compact_output_regression.csv"
    regression.to_csv(regression_path, index=False)
    failures = regression.loc[~regression["passed"]]
    print(f"Compact evidence tables matched: {int(regression['passed'].sum())}/{len(regression)}")
    if len(failures):
        print(failures.to_string(index=False))
        raise RuntimeError("Reproduced compact results differ from tracked evidence")
    print(f"Full reproduction runtime: {time.perf_counter() - started:.1f} seconds")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build-data": command_build_data,
    "select-elo": command_select_elo,
    "run-elo": command_run_elo,
    "run-glicko": command_run_glicko,
    "glicko-periods": command_glicko_periods,
    "run-adaptive-elo": command_run_adaptive,
    "build-comparison-inputs": command_build_comparison_inputs,
    "compare-models": command_compare_models,
    "early-game": command_early_game,
    "initial-rating-sensitivity": command_initial_rating,
    "entry-diagnostics": command_entry_diagnostics,
    "reproduce-dissertation": command_reproduce,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        _print_run_header(
            "validate",
            False,
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
