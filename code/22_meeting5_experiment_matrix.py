"""Create the Meeting 5 experiment matrix planning files.

This script does not run any new rating model. It only checks whether existing
key scripts/outputs are present and writes a structured experiment plan for the
next stage of Elo-vs-Glicko work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPERIMENT_MATRIX_PATH = OUTPUT_DIR / "meeting5_experiment_matrix.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "meeting5_experiment_matrix_summary.md"
FILE_CHECK_PATH = OUTPUT_DIR / "meeting5_existing_file_check.csv"


MATRIX_COLUMNS = [
    "experiment_id",
    "experiment_group",
    "research_question",
    "model_or_variant",
    "comparison_baseline",
    "single_difference_tested",
    "dataset",
    "burn_in_or_history",
    "evaluation_set",
    "rating_period",
    "rd_inflation_setting",
    "min_rd_setting",
    "max_rd_setting",
    "elo_k_or_scale_setting",
    "main_metrics",
    "secondary_metrics",
    "expected_output_files",
    "expected_plots",
    "current_status",
    "priority_for_meeting5",
    "notes_for_dissertation",
]


REQUIRED_EXISTING_FILES = [
    {
        "filename": "13_build_full_history_match_dataset.py",
        "role": "Builds the 1985-2025 full-history checked match-level dataset.",
    },
    {
        "filename": "14_elo_burnin_rating_list_stability.py",
        "role": "Runs Elo burn-in start-year and final rating-list stability diagnostics.",
    },
    {
        "filename": "15_elo_single_year_rerun_convergence.py",
        "role": "Runs single-year repeated rerun convergence diagnostic for Elo.",
    },
    {
        "filename": "16_elo_event_level_volatility.py",
        "role": "Compares Elo match-level volatility with event-level net changes.",
    },
    {
        "filename": "17_elo_baseline_decision_summary.py",
        "role": "Combines Elo diagnostics into defensible baseline recommendations.",
    },
    {
        "filename": "glicko_core.py",
        "role": "Reusable Glicko-1 expected-score and rating/RD update functions.",
    },
    {
        "filename": "18_glicko_core_sanity_check.py",
        "role": "Checks Glicko-1 formula behaviour on small sanity examples.",
    },
    {
        "filename": "19_glicko_match_by_match_baseline.py",
        "role": "Runs first full-history Glicko-1 match-by-match baseline.",
    },
    {
        "filename": "20_glicko_rating_period_sensitivity.py",
        "role": "Compares match-by-match, event-level, monthly, and yearly Glicko periods.",
    },
]


def relative_or_missing(path: Path | None) -> str:
    """Return a project-relative path if possible, otherwise an empty string."""

    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def find_file_by_name(filename: str) -> Path | None:
    """Find a file anywhere under the project root by filename."""

    matches = sorted(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def build_existing_file_check() -> pd.DataFrame:
    """Check whether required previous scripts exist."""

    rows: list[dict[str, Any]] = []
    for item in REQUIRED_EXISTING_FILES:
        found_path = find_file_by_name(item["filename"])
        rows.append(
            {
                "filename": item["filename"],
                "found": found_path is not None,
                "path": relative_or_missing(found_path) if found_path else "missing",
                "role": item["role"],
            }
        )
    return pd.DataFrame(rows, columns=["filename", "found", "path", "role"])


def row(
    experiment_id: str,
    experiment_group: str,
    research_question: str,
    model_or_variant: str,
    comparison_baseline: str,
    single_difference_tested: str,
    rating_period: str,
    rd_inflation_setting: str,
    min_rd_setting: str,
    max_rd_setting: str,
    elo_k_or_scale_setting: str,
    main_metrics: str,
    secondary_metrics: str,
    expected_output_files: str,
    expected_plots: str,
    current_status: str,
    priority_for_meeting5: int,
    notes_for_dissertation: str,
    dataset: str = "1985-2025 full-history checked match-level dataset",
    burn_in_or_history: str = "1985-2025 full history unless otherwise stated",
    evaluation_set: str = "2025 test games; expected n=11,379",
) -> dict[str, Any]:
    """Create one experiment matrix row."""

    return {
        "experiment_id": experiment_id,
        "experiment_group": experiment_group,
        "research_question": research_question,
        "model_or_variant": model_or_variant,
        "comparison_baseline": comparison_baseline,
        "single_difference_tested": single_difference_tested,
        "dataset": dataset,
        "burn_in_or_history": burn_in_or_history,
        "evaluation_set": evaluation_set,
        "rating_period": rating_period,
        "rd_inflation_setting": rd_inflation_setting,
        "min_rd_setting": min_rd_setting,
        "max_rd_setting": max_rd_setting,
        "elo_k_or_scale_setting": elo_k_or_scale_setting,
        "main_metrics": main_metrics,
        "secondary_metrics": secondary_metrics,
        "expected_output_files": expected_output_files,
        "expected_plots": expected_plots,
        "current_status": current_status,
        "priority_for_meeting5": priority_for_meeting5,
        "notes_for_dissertation": notes_for_dissertation,
    }


def build_experiment_matrix() -> pd.DataFrame:
    """Build the Meeting 5 experiment matrix."""

    rows = [
        row(
            "A1",
            "Glicko implementation validation",
            "Does the Glicko-1 implementation behave correctly before model comparison?",
            "Glicko-1 probability sanity checks",
            "No performance baseline; formula-level validation only",
            "Not comparing model performance; verifies expected probabilities follow Glicko behaviour.",
            "Not applicable",
            "C=0; not part of this validation",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "Expected score for equal/higher/lower ratings",
            "Pass/fail notes; probability range checks",
            "outputs/glicko_implementation/glicko_core_sanity_results.csv",
            "None required; simple validation table is sufficient",
            "Completed in code/18_glicko_core_sanity_check.py",
            1,
            "Use this to show confidence in the implementation before discussing prediction results.",
        ),
        row(
            "A2",
            "Glicko implementation validation",
            "Does the Glicko update match a known official-style batch example?",
            "Glicko-1 official-style batch update",
            "Expected example values around rating 1464 and RD 152",
            "Not comparing model performance; verifies update formula and simultaneous batch update.",
            "One synthetic rating period",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "New rating and new RD vs expected values",
            "Pass/fail tolerance check",
            "outputs/glicko_implementation/glicko_core_example_updates.csv",
            "None required",
            "Completed; example gave rating 1464.106 and RD 151.399",
            1,
            "This is useful evidence that the Glicko formula itself is implemented correctly.",
            dataset="Synthetic Glicko example",
            burn_in_or_history="Not applicable",
            evaluation_set="Not applicable",
        ),
        row(
            "A3",
            "Glicko implementation validation",
            "Do RD bounds behave sensibly at MIN_RD and MAX_RD?",
            "Glicko-1 RD boundary checks",
            "Glicko core constants",
            "Not comparing model performance; checks that RD is bounded and not allowed to exceed new-player RD.",
            "Synthetic edge cases",
            "C=0 initially; future inflation must still respect MAX_RD",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "RD after update; number below MIN_RD; number above MAX_RD",
            "Warning table for boundary behaviour",
            "Future: outputs/glicko_implementation/glicko_rd_boundary_check.csv",
            "None required",
            "Planned; core clamp exists in glicko_core.py but explicit boundary table not yet generated",
            1,
            "Boundary checks will help justify later inactivity RD inflation settings.",
            dataset="Synthetic edge cases",
            burn_in_or_history="Not applicable",
            evaluation_set="Not applicable",
        ),
        row(
            "A4",
            "Glicko implementation validation",
            "Are predictions recorded before rating/RD updates?",
            "Prediction-before-update audit",
            "Existing Elo/Glicko pipeline convention",
            "Not comparing model performance; verifies no result leakage in prediction rows.",
            "Match-by-match initially",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "Audit pass/fail; pre-match state equals state before update",
            "Selected example rows; prediction probability range",
            "Future: outputs/glicko_implementation/glicko_prediction_before_update_audit.csv",
            "None required",
            "Partly completed by design in code/19 and code/20; explicit audit table still useful",
            1,
            "This addresses fair predictive evaluation: each match must be predicted before its result is used.",
            dataset="1985-2025 full-history checked match-level dataset",
            burn_in_or_history="1985-2025 full history",
            evaluation_set="2025 test games plus selected earlier audit rows",
        ),
        row(
            "A5",
            "Glicko implementation validation",
            "Do active high-volume players have broadly comparable Elo and Glicko rating lists?",
            "Active-player Elo-vs-Glicko rating-list similarity",
            "Validation-best Elo and default Elo final ratings",
            "Implementation behaviour check, especially for players with many games; not the final performance comparison.",
            "Use selected Glicko period assumption",
            "C=0 first, then selected RD inflation variant",
            "MIN_RD=30",
            "MAX_RD=350",
            "Elo K=20 scale=500 and K=30 scale=300",
            "Spearman rank correlation; Top50/Top100 overlap",
            "Mean abs rating/rank difference; active player counts",
            "Future: outputs/model_comparison/active_player_rating_list_similarity.csv",
            "Rank scatter plot; Top100 overlap bar chart",
            "Planned",
            1,
            "This helps identify whether Glicko is producing plausible rankings for well-observed players.",
            dataset="1985-2025 full-history checked match-level dataset",
            burn_in_or_history="1985-2025 full history",
            evaluation_set="Active 2025 players; include games >=100 lifetime or in-run threshold",
        ),
        row(
            "B1",
            "Glicko inactivity RD inflation sensitivity",
            "What happens when inactive players do not regain uncertainty?",
            "Glicko C=0 current baseline",
            "Current match-by-match Glicko baseline",
            "Reference variant: no inactivity RD inflation.",
            "Selected after rating-period diagnostic; match-by-match currently strongest",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "2025 log loss; Brier score; final RD distribution",
            "Players at MIN_RD; inactive returning-player examples; calibration",
            "outputs/glicko_implementation/glicko_mbm_metrics_2025.csv; glicko_mbm_rd_summary.csv",
            "RD distribution histogram; RD vs games played",
            "Completed as current baseline",
            2,
            "This baseline shows the limitation that inactive players do not become more uncertain over time.",
        ),
        row(
            "B2",
            "Glicko inactivity RD inflation sensitivity",
            "Does low inactivity RD inflation improve uncertainty behaviour without hurting predictions?",
            "Glicko low RD inflation",
            "Glicko C=0",
            "Only C changes; dataset, rating period, initial rating/RD, MIN_RD, MAX_RD stay fixed.",
            "Selected Glicko period assumption",
            "Low C; exact value to be selected before coding",
            "MIN_RD=30, unless sensitivity requires minor adjustment",
            "MAX_RD=350; must not exceed new-player RD",
            "Not applicable",
            "2025 log loss; Brier score; final RD distribution",
            "Inactive-player RD increase; calibration; active rating-list stability",
            "Future: outputs/glicko_implementation/glicko_rd_inflation_low_*.csv",
            "RD distribution; inactive returner case studies",
            "Planned",
            2,
            "Tests whether Glicko's uncertainty advantage appears once inactivity is represented.",
        ),
        row(
            "B3",
            "Glicko inactivity RD inflation sensitivity",
            "Does medium inactivity RD inflation provide a better trade-off?",
            "Glicko medium RD inflation",
            "Glicko C=0 and low C",
            "Only C changes; all other design choices fixed.",
            "Selected Glicko period assumption",
            "Medium C; exact value to be selected before coding",
            "MIN_RD=30, unless sensitivity requires minor adjustment",
            "MAX_RD=350; must not exceed new-player RD",
            "Not applicable",
            "2025 log loss; Brier score; final RD distribution",
            "Returning-player prediction metrics; calibration; rating-list stability",
            "Future: outputs/glicko_implementation/glicko_rd_inflation_medium_*.csv",
            "RD distribution; metric comparison bar chart",
            "Planned",
            2,
            "Medium C may be a plausible final candidate if it improves uncertainty without large prediction cost.",
        ),
        row(
            "B4",
            "Glicko inactivity RD inflation sensitivity",
            "Does high inactivity RD inflation make ratings too unstable?",
            "Glicko high RD inflation",
            "Glicko C=0, low C, medium C",
            "Only C changes; all other design choices fixed.",
            "Selected Glicko period assumption",
            "High C; exact value to be selected before coding",
            "MIN_RD=30, unless sensitivity requires minor adjustment",
            "MAX_RD=350; must not exceed new-player RD",
            "Not applicable",
            "2025 log loss; Brier score; RD distribution",
            "Rating volatility; returning-player update sizes; calibration",
            "Future: outputs/glicko_implementation/glicko_rd_inflation_high_*.csv",
            "RD distribution; update-size comparison",
            "Planned",
            2,
            "High C is mainly a stress test to understand when uncertainty inflation becomes too aggressive.",
        ),
        row(
            "C1",
            "Glicko rating-period runtime comparison",
            "How expensive is match-by-match compared with coarser rating periods?",
            "Glicko match-by-match period",
            "Event-level, monthly, yearly Glicko periods",
            "Only rating-period grouping changes; C remains 0.",
            "Match-by-match",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "Runtime seconds; number of rating periods; 2025 log loss; Brier score",
            "Number of update operations; active-player rating-list similarity",
            "Future: outputs/glicko_implementation/glicko_rating_period_runtime_summary.csv",
            "Runtime vs rating-period count; metrics vs period type",
            "Partly completed: metrics and period counts exist; per-setting runtime not recorded",
            3,
            "Shorter rating periods normally take longer because ratings/RDs are updated more frequently.",
        ),
        row(
            "C2",
            "Glicko rating-period runtime comparison",
            "Is event-level a reasonable alternative to match-by-match?",
            "Glicko event-level period",
            "Glicko match-by-match period",
            "Only rating-period grouping changes; C remains 0.",
            "Year + event",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "2025 log loss; Brier score; runtime seconds",
            "Top50/Top100 active-player overlap; number of periods",
            "outputs/glicko_implementation/glicko_rating_period_metrics_2025.csv; future runtime summary",
            "Active Top100 overlap; runtime bar chart",
            "Metrics completed; runtime missing",
            3,
            "Event-level is defensible but did not improve prediction under C=0 in the current results.",
        ),
        row(
            "C3",
            "Glicko rating-period runtime comparison",
            "Does monthly grouping lose useful chronological information?",
            "Glicko monthly period",
            "Glicko match-by-match period",
            "Only rating-period grouping changes; C remains 0.",
            "Calendar month using event_order_date",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "2025 log loss; Brier score; runtime seconds",
            "Active-player rating-list similarity; period sizes",
            "outputs/glicko_implementation/glicko_rating_period_metrics_2025.csv; future runtime summary",
            "Metrics vs runtime trade-off plot",
            "Metrics completed; runtime missing",
            3,
            "Monthly has far fewer periods than match-by-match but should be judged against prediction and ranking stability.",
        ),
        row(
            "C4",
            "Glicko rating-period runtime comparison",
            "Is yearly grouping too coarse for croquet?",
            "Glicko yearly period",
            "Glicko match-by-match period",
            "Only rating-period grouping changes; C remains 0.",
            "Calendar year",
            "C=0",
            "MIN_RD=30",
            "MAX_RD=350",
            "Not applicable",
            "2025 log loss; Brier score; runtime seconds",
            "Active-player rating-list similarity; calibration",
            "outputs/glicko_implementation/glicko_rating_period_metrics_2025.csv; future runtime summary",
            "Metrics vs period coarseness plot",
            "Metrics completed; runtime missing",
            3,
            "Yearly is mainly a diagnostic setting and currently appears too coarse.",
        ),
        row(
            "D1",
            "Fair Elo-vs-Glicko comparison",
            "How does transparent Default Elo compare with Glicko under the same evaluation design?",
            "Default Elo K=20 scale=500",
            "Selected Glicko C=0 and selected RD inflation variant",
            "Model family differs; dataset, ordering, evaluation games, and metrics fixed.",
            "Elo: match-by-match; Glicko: selected period assumption",
            "Glicko C=0 first, then selected C",
            "Glicko MIN_RD=30",
            "Glicko MAX_RD=350",
            "Elo K=20, scale=500",
            "Log loss; Brier score",
            "Accuracy; calibration; confidence table; rating-list stability",
            "Future: outputs/model_comparison/fair_elo_glicko_metrics_2025.csv",
            "Metric comparison bar chart; calibration curves",
            "Planned",
            4,
            "Accuracy should be secondary; log loss and Brier are the main prediction metrics.",
        ),
        row(
            "D2",
            "Fair Elo-vs-Glicko comparison",
            "Does Glicko outperform the prediction-oriented Elo baseline?",
            "Validation-best Elo K=30 scale=300",
            "Selected Glicko C=0 and selected RD inflation variant",
            "Model family differs; dataset, ordering, evaluation games, and metrics fixed.",
            "Elo: match-by-match; Glicko: selected period assumption",
            "Glicko C=0 first, then selected C",
            "Glicko MIN_RD=30",
            "Glicko MAX_RD=350",
            "Elo K=30, scale=300",
            "Log loss; Brier score",
            "Accuracy; calibration; volatility/stability diagnostics",
            "Future: outputs/model_comparison/fair_elo_glicko_metrics_2025.csv",
            "Metric comparison bar chart; calibration curves",
            "Planned",
            4,
            "This is the strictest Elo comparator because it was selected by validation.",
        ),
        row(
            "D3",
            "Fair Elo-vs-Glicko comparison",
            "How does low-volatility Elo behave as a stability reference?",
            "Conservative Elo K=10 scale=500",
            "Selected Glicko variant",
            "Model family differs; conservative Elo is not expected to be the best predictor.",
            "Elo: match-by-match; Glicko: selected period assumption",
            "Glicko selected C",
            "Glicko MIN_RD=30",
            "Glicko MAX_RD=350",
            "Elo K=10, scale=500",
            "Rating stability; volatility; log loss; Brier score",
            "Accuracy; calibration; active-player ranking overlap",
            "Future: outputs/model_comparison/fair_elo_glicko_stability_reference.csv",
            "Volatility comparison; rating movement distribution",
            "Planned",
            4,
            "Use conservative Elo mainly to contextualise stability and volatility, not as the main predictive baseline.",
        ),
        row(
            "D4",
            "Fair Elo-vs-Glicko comparison",
            "Does the best selected Glicko RD inflation variant improve on C=0?",
            "Glicko with best selected RD inflation",
            "Glicko C=0 and Elo baselines",
            "Only selected Glicko uncertainty handling changes relative to C=0.",
            "Selected Glicko period assumption",
            "Selected after RD inflation sensitivity",
            "Glicko MIN_RD selected/fixed",
            "Glicko MAX_RD=350",
            "Not applicable",
            "Log loss; Brier score; RD behaviour",
            "Calibration; returning-player examples; active ranking stability",
            "Future: outputs/model_comparison/fair_elo_glicko_metrics_2025.csv",
            "RD behaviour plot; metric comparison",
            "Blocked until RD inflation sensitivity is completed",
            4,
            "This should be the Glicko candidate carried into the final comparison if RD inflation is helpful.",
        ),
        row(
            "E1",
            "Adaptive-K Elo comparison",
            "Can a simple adaptive-K Elo capture part of Glicko's uncertainty advantage?",
            "Adaptive-K Elo based on total previous games",
            "Default Elo and selected Glicko variant",
            "Only Elo K changes as a function of total previous games.",
            "Match-by-match Elo updates",
            "Not applicable",
            "Not applicable",
            "Not applicable",
            "Higher K for <20 previous games; medium K for 20-99; lower K for 100+",
            "Log loss; Brier score",
            "Calibration; new-player/low-experience subgroup performance; rating volatility",
            "Future: outputs/adaptive_elo/adaptive_k_total_games_metrics_2025.csv",
            "Subgroup metric plot; K distribution plot",
            "Planned",
            5,
            "This is a transparent bridge model between fixed-K Elo and Glicko uncertainty.",
            burn_in_or_history="1985-2025 full history",
        ),
        row(
            "E2",
            "Adaptive-K Elo comparison",
            "Can recent activity adjust Elo updates in a way similar to Glicko RD?",
            "Adaptive-K Elo based on previous-year games",
            "Default Elo and selected Glicko variant",
            "Only Elo K changes as a function of previous-year activity.",
            "Match-by-match Elo updates",
            "Not applicable",
            "Not applicable",
            "Not applicable",
            "Higher K for few games last year; medium K for medium activity; lower K for many games",
            "Log loss; Brier score",
            "Calibration; inactive/returning-player subgroup performance; rating volatility",
            "Future: outputs/adaptive_elo/adaptive_k_previous_year_metrics_2025.csv",
            "Subgroup metric plot; K distribution plot",
            "Planned",
            5,
            "This tests whether simple activity-aware Elo can capture some of Glicko's practical benefit.",
            burn_in_or_history="1985-2025 full history",
        ),
        row(
            "F1",
            "Final plots and summary tables",
            "What evidence should be prepared for Meeting 5 and dissertation writing?",
            "Summary tables and plots",
            "All completed experiments",
            "Not a model experiment; presentation layer only.",
            "All selected period assumptions",
            "All selected RD inflation variants",
            "As selected",
            "As selected",
            "As selected",
            "Summary of log loss, Brier, calibration, RD behaviour, runtime, stability",
            "Meeting-ready tables; dissertation-ready interpretation notes",
            "Future: outputs/meeting5_summary_tables/*.csv",
            "Metric comparison; calibration; RD distribution; runtime; rating-list overlap",
            "Planned after experiments B-D",
            6,
            "This converts experiments into evidence the supervisor can evaluate quickly.",
            dataset="All selected experiment outputs",
            burn_in_or_history="All selected history settings",
            evaluation_set="2025 evaluation plus stability/volatility subsets",
        ),
    ]

    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


def build_markdown_summary(file_check: pd.DataFrame, matrix: pd.DataFrame) -> str:
    """Create a meeting-ready Markdown summary."""

    found_count = int(file_check["found"].sum())
    total_count = len(file_check)
    missing = file_check.loc[~file_check["found"], "filename"].tolist()
    group_counts = matrix.groupby("experiment_group").size().reset_index(name="n_experiments")

    missing_text = "None" if not missing else ", ".join(missing)
    group_lines = [
        f"- {row.experiment_group}: {int(row.n_experiments)} planned rows"
        for row in group_counts.itertuples(index=False)
    ]

    priority_rows = matrix.sort_values(["priority_for_meeting5", "experiment_id"])[
        ["priority_for_meeting5", "experiment_group"]
    ].drop_duplicates()
    priority_lines = [
        f"{int(row.priority_for_meeting5)}. {row.experiment_group}"
        for row in priority_rows.itertuples(index=False)
    ]

    completed_work = [
        "Elo baseline built from 2025 prototype to full-history 1985-2025 framework.",
        "Elo burn-in, rating-list stability, single-year rerun, and event-level volatility diagnostics completed.",
        "Elo baseline decision summary completed with conservative, default, and validation-best Elo roles.",
        "Glicko-1 core implemented and sanity checked.",
        "Full-history match-by-match Glicko baseline completed.",
        "Glicko rating-period sensitivity completed for match-by-match, event-level, monthly, and yearly periods under C=0.",
    ]

    outputs_for_supervisor = [
        "A concise implementation validation table for Glicko-1.",
        "RD inflation sensitivity table showing C=0, low C, medium C, and high C.",
        "Rating-period runtime table including runtime seconds, number of periods, update operations, and 2025 metrics.",
        "Fair Elo-vs-Glicko metric table using identical 2025 evaluation games.",
        "Calibration and confidence diagnostics for selected Elo and Glicko variants.",
        "Active-player rating-list similarity table, especially for high-volume players.",
    ]

    short_english_summary = (
        "For Meeting 5, the next stage is to turn the completed Elo and initial Glicko work into a controlled "
        "comparison framework. The priority is to validate the Glicko implementation, isolate the effect of "
        "inactivity RD inflation, add runtime evidence for rating-period choices, and then compare Elo and Glicko "
        "under the same dataset, chronological ordering, prediction-before-update rule, and 2025 evaluation metrics."
    )

    lines = [
        "# Meeting 5 Experiment Matrix",
        "",
        "## Current Completed Work",
        "",
        *[f"- {item}" for item in completed_work],
        "",
        "## Existing File Check",
        "",
        f"- Required files found: {found_count} / {total_count}",
        f"- Missing files: {missing_text}",
        "",
        "## Why This Experiment Matrix Is Needed",
        "",
        "The previous work produced defensible Elo baselines and an initial Glicko-1 implementation. The next stage should avoid changing multiple things at once. This matrix separates implementation validation, Glicko inactivity RD inflation, rating-period runtime, fair Elo-vs-Glicko comparison, and adaptive-K Elo into distinct experiment groups.",
        "",
        "## Fixed Evaluation Design",
        "",
        "- Dataset: 1985-2025 full-history checked match-level dataset.",
        "- Evaluation set: fixed 2025 games, expected n=11,379.",
        "- Prediction rule: record pre-match prediction before updating ratings.",
        "- Main prediction metrics: log loss and Brier score.",
        "- Secondary metrics: accuracy, calibration, confidence bins, rating-list stability, RD behaviour, runtime, and active-player ranking overlap.",
        "- Fair comparison principle: isolate one single difference whenever possible.",
        "",
        "## Experiment Groups",
        "",
        *group_lines,
        "",
        "## Priority Order Before Meeting 5",
        "",
        *priority_lines,
        "",
        "## Outputs To Prepare For Supervisor",
        "",
        *[f"- {item}" for item in outputs_for_supervisor],
        "",
        "## Short English Summary For Meeting Notes",
        "",
        short_english_summary,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_check = build_existing_file_check()
    matrix = build_experiment_matrix()
    summary = build_markdown_summary(file_check, matrix)

    file_check.to_csv(FILE_CHECK_PATH, index=False, encoding="utf-8-sig")
    matrix.to_csv(EXPERIMENT_MATRIX_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_MD_PATH.write_text(summary, encoding="utf-8")

    print("Meeting 5 experiment planning files generated:")
    print(f"  {EXPERIMENT_MATRIX_PATH}")
    print(f"  {SUMMARY_MD_PATH}")
    print(f"  {FILE_CHECK_PATH}")
    print()
    print("Existing file check:")
    print(file_check[["filename", "found", "path"]].to_string(index=False))
    print()
    print(f"Experiment matrix rows: {len(matrix)}")
    print("This script did not run new models or modify existing model outputs.")


if __name__ == "__main__":
    main()
