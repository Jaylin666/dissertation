"""This script implements and tests the core Glicko-1 update formula before applying it to the full croquet dataset.

The purpose of this first Glicko step is only to verify the core formula and a
small reusable engine. It does not read the full croquet dataset, does not
compare against Elo, does not tune parameters, and does not implement Glicko-2.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import pandas as pd

from glicko_core import (
    C,
    DEFAULT_RATING,
    DEFAULT_RD,
    MAX_RD,
    MIN_RD,
    Q,
    expected_score,
    g_function,
    update_player_glicko,
    update_two_players_single_game,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "glicko_implementation"

SANITY_RESULTS_FILE = OUTPUT_DIR / "glicko_core_sanity_results.csv"
EXAMPLE_UPDATES_FILE = OUTPUT_DIR / "glicko_core_example_updates.csv"
SUMMARY_FILE = OUTPUT_DIR / "glicko_core_sanity_summary.md"


def format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def add_result(
    rows: list[dict[str, Any]],
    check_name: str,
    input_description: str,
    result_value: Any,
    expected_condition: str,
    passed: bool,
    notes: str = "",
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "input_description": input_description,
            "result_value": result_value,
            "expected_condition": expected_condition,
            "passed": bool(passed),
            "notes": notes,
        }
    )


def add_single_game_examples(
    rows: list[dict[str, Any]],
    scenario: str,
    player1_label: str,
    player2_label: str,
    update,
) -> None:
    rows.append(
        {
            "scenario": scenario,
            "player": player1_label,
            "rating_before": update.player1_rating_before,
            "rd_before": update.player1_rd_before,
            "opponent_rating": update.player2_rating_before,
            "opponent_rd": update.player2_rd_before,
            "score": update.score1,
            "predicted_probability": update.predicted_player1_win,
            "rating_after": update.player1_rating_after,
            "rd_after": update.player1_rd_after,
            "rating_change": update.player1_rating_after - update.player1_rating_before,
            "rd_change": update.player1_rd_after - update.player1_rd_before,
        }
    )
    rows.append(
        {
            "scenario": scenario,
            "player": player2_label,
            "rating_before": update.player2_rating_before,
            "rd_before": update.player2_rd_before,
            "opponent_rating": update.player1_rating_before,
            "opponent_rd": update.player1_rd_before,
            "score": 1.0 - update.score1,
            "predicted_probability": 1.0 - update.predicted_player1_win,
            "rating_after": update.player2_rating_after,
            "rd_after": update.player2_rd_after,
            "rating_change": update.player2_rating_after - update.player2_rating_before,
            "rd_change": update.player2_rd_after - update.player2_rd_before,
        }
    )


def run_sanity_checks() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sanity_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    equal_expected = expected_score(1500, 1500, 350)
    higher_expected = expected_score(1600, 1500, 350)
    lower_expected = expected_score(1400, 1500, 350)

    add_result(
        sanity_rows,
        "expected_probability_equal_ratings",
        "rating=1500, opponent_rating=1500, opponent_rd=350",
        equal_expected,
        "Expected score should be close to 0.5",
        abs(equal_expected - 0.5) < 1e-12,
    )
    add_result(
        sanity_rows,
        "expected_probability_higher_rating",
        "rating=1600, opponent_rating=1500, opponent_rd=350",
        higher_expected,
        "Expected score should be greater than 0.5",
        higher_expected > 0.5,
    )
    add_result(
        sanity_rows,
        "expected_probability_lower_rating",
        "rating=1400, opponent_rating=1500, opponent_rd=350",
        lower_expected,
        "Expected score should be less than 0.5",
        lower_expected < 0.5,
    )

    equal_players_update = update_two_players_single_game(1500, 350, 1500, 350, 1)
    add_single_game_examples(
        example_rows,
        "equal_rating_single_game_a_wins",
        "A",
        "B",
        equal_players_update,
    )
    equal_players_passed = (
        equal_players_update.player1_rating_after > 1500
        and equal_players_update.player2_rating_after < 1500
        and equal_players_update.player1_rd_after <= 350
        and equal_players_update.player2_rd_after <= 350
        and abs(equal_players_update.predicted_player1_win - 0.5) < 1e-12
    )
    add_result(
        sanity_rows,
        "single_game_direction_equal_players",
        "A rating=1500 RD=350, B rating=1500 RD=350, A wins",
        (
            f"A_after={equal_players_update.player1_rating_after:.3f}; "
            f"B_after={equal_players_update.player2_rating_after:.3f}; "
            f"pred_A={equal_players_update.predicted_player1_win:.6f}"
        ),
        "Winner rating increases, loser rating decreases, RDs do not increase, prediction near 0.5",
        equal_players_passed,
    )

    upset_update = update_two_players_single_game(1400, 100, 1700, 100, 1)
    add_single_game_examples(
        example_rows,
        "weaker_player_beats_stronger_player",
        "weaker_player",
        "stronger_player",
        upset_update,
    )
    weaker_change = upset_update.player1_rating_after - upset_update.player1_rating_before
    stronger_change = upset_update.player2_rating_after - upset_update.player2_rating_before
    upset_passed = (
        upset_update.player1_rating_after > 1400
        and upset_update.player2_rating_after < 1700
        and weaker_change > 10
        and stronger_change < -10
    )
    add_result(
        sanity_rows,
        "upset_direction_and_size",
        "weaker rating=1400 RD=100 beats stronger rating=1700 RD=100",
        f"weaker_change={weaker_change:.3f}; stronger_change={stronger_change:.3f}",
        "Weaker rating increases and stronger rating decreases by a noticeable amount",
        upset_passed,
    )

    batch_new_rating, batch_new_rd = update_player_glicko(
        rating=1500,
        rd=200,
        opponent_ratings=[1400, 1550, 1700],
        opponent_rds=[30, 100, 300],
        scores=[1, 0, 0],
    )
    batch_rating_close = abs(batch_new_rating - 1464) < 2
    batch_rd_close = abs(batch_new_rd - 152) < 2
    batch_passed = batch_rating_close and batch_rd_close
    batch_notes = ""
    if not batch_passed:
        batch_notes = (
            "Warning: result is not close to the common Glicko example. "
            "Check formula constants and RD handling before using on full data."
        )
    add_result(
        sanity_rows,
        "official_style_batch_example",
        "player rating=1500 RD=200; opponents=(1400,30,1),(1550,100,0),(1700,300,0)",
        f"new_rating={batch_new_rating:.3f}; new_rd={batch_new_rd:.3f}",
        "Expected approximately new rating 1464 and new RD 152, tolerance +/-2",
        batch_passed,
        batch_notes,
    )
    example_rows.append(
        {
            "scenario": "official_style_batch_example",
            "player": "target_player",
            "rating_before": 1500.0,
            "rd_before": 200.0,
            "opponent_rating": "1400;1550;1700",
            "opponent_rd": "30;100;300",
            "score": "1;0;0",
            "predicted_probability": (
                f"{expected_score(1500, 1400, 30):.6f};"
                f"{expected_score(1500, 1550, 100):.6f};"
                f"{expected_score(1500, 1700, 300):.6f}"
            ),
            "rating_after": batch_new_rating,
            "rd_after": batch_new_rd,
            "rating_change": batch_new_rating - 1500.0,
            "rd_change": batch_new_rd - 200.0,
        }
    )

    metadata = {
        "equal_expected": equal_expected,
        "higher_expected": higher_expected,
        "lower_expected": lower_expected,
        "equal_players_update": equal_players_update,
        "upset_update": upset_update,
        "batch_new_rating": batch_new_rating,
        "batch_new_rd": batch_new_rd,
        "batch_passed": batch_passed,
        "all_passed": all(row["passed"] for row in sanity_rows),
        "warning_count": sum(1 for row in sanity_rows if row["notes"]),
    }

    return pd.DataFrame(sanity_rows), pd.DataFrame(example_rows), metadata


def write_summary(
    sanity_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    passed_count = int(sanity_df["passed"].sum())
    total_count = len(sanity_df)
    failed = sanity_df.loc[~sanity_df["passed"], "check_name"].tolist()
    warnings = sanity_df.loc[sanity_df["notes"].astype(str).str.len() > 0, ["check_name", "notes"]]

    lines = [
        "# Glicko-1 Core Sanity Check Summary",
        "",
        "## Aim",
        "",
        "This script implements and tests the core Glicko-1 update formula before applying it to the full croquet dataset.",
        "",
        "The goal is to verify the reusable Glicko-1 engine only. This step does not read the full croquet match dataset, does not compare against Elo, does not tune parameters, and does not implement Glicko-2.",
        "",
        "## Implemented Formulas",
        "",
        f"- Default rating: {DEFAULT_RATING:.1f}",
        f"- Default RD: {DEFAULT_RD:.1f}",
        f"- RD bounds: [{MIN_RD:.1f}, {MAX_RD:.1f}]",
        f"- q = ln(10) / 400 = {Q:.10f}",
        f"- Inactivity RD inflation C = {C:.1f}",
        "- g(RD) = 1 / sqrt(1 + 3 * q^2 * RD^2 / pi^2)",
        "- Expected score = 1 / (1 + 10^(-g(RD_j) * (r - r_j) / 400))",
        "- Rating-period update uses all games in the period before computing the new rating and RD.",
        "",
        "## Sanity Check Results",
        "",
        f"- Passed checks: {passed_count} / {total_count}",
        f"- Overall status: {'PASS' if metadata['all_passed'] else 'CHECK WARNINGS/FAILURES'}",
        "",
    ]

    for _, row in sanity_df.iterrows():
        lines.append(
            f"- {row['check_name']}: {format_bool(bool(row['passed']))}; result = {row['result_value']}"
        )

    lines.extend(
        [
            "",
            "## Official-Style Batch Example",
            "",
            "The batch example uses a player rated 1500 with RD 200 against three opponents: 1400/RD30 win, 1550/RD100 loss, and 1700/RD300 loss.",
            "",
            f"- New rating: {metadata['batch_new_rating']:.3f}",
            f"- New RD: {metadata['batch_new_rd']:.3f}",
            "- Common reference expectation: approximately rating 1464 and RD 152.",
            f"- Tolerance check: {'PASS' if metadata['batch_passed'] else 'WARNING'}",
            "",
            "## Readiness for Next Step",
            "",
            "The core engine is ready for a small full-data match-by-match Glicko baseline script, provided the next step keeps the same match ordering and prediction column conventions as the Elo pipeline.",
            "",
            "## Remaining Limitations",
            "",
            "- Inactivity RD inflation is not implemented yet because rating-period design for croquet still needs to be decided.",
            "- Monthly, event-level, and yearly rating-period sensitivity are not implemented in this first core sanity step.",
            "- Glicko-2 volatility is intentionally out of scope for this baseline implementation.",
        ]
    )

    if failed:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- {name}" for name in failed)

    if not warnings.empty:
        lines.extend(["", "## Warnings", ""])
        for _, row in warnings.iterrows():
            lines.append(f"- {row['check_name']}: {row['notes']}")

    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Glicko-1 core sanity check")
    print("Default parameters:")
    print(f"  DEFAULT_RATING = {DEFAULT_RATING}")
    print(f"  DEFAULT_RD = {DEFAULT_RD}")
    print(f"  MIN_RD = {MIN_RD}")
    print(f"  MAX_RD = {MAX_RD}")
    print(f"  Q = {Q}")
    print(f"  C = {C}")
    print(f"  g(DEFAULT_RD) = {g_function(DEFAULT_RD):.6f}")
    print("This script does not read the full croquet dataset.")
    print()

    sanity_df, examples_df, metadata = run_sanity_checks()

    for _, row in sanity_df.iterrows():
        print(f"{format_bool(bool(row['passed']))}: {row['check_name']} -> {row['result_value']}")
        if row["notes"]:
            print(f"  WARNING: {row['notes']}")

    print()
    print(
        "Official-style batch example: "
        f"new_rating={metadata['batch_new_rating']:.3f}, "
        f"new_rd={metadata['batch_new_rd']:.3f}"
    )

    sanity_df.to_csv(SANITY_RESULTS_FILE, index=False, encoding="utf-8-sig")
    examples_df.to_csv(EXAMPLE_UPDATES_FILE, index=False, encoding="utf-8-sig")
    write_summary(sanity_df, examples_df, metadata)

    print()
    print("Output files:")
    print(f"  {SANITY_RESULTS_FILE}")
    print(f"  {EXAMPLE_UPDATES_FILE}")
    print(f"  {SUMMARY_FILE}")
    print()
    if metadata["all_passed"]:
        print("Core Glicko-1 engine is ready for the next full-data match-by-match baseline step.")
    else:
        print("Core Glicko-1 engine produced warnings/failures; review outputs before the full-data step.")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
