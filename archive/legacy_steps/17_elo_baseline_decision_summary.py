"""
This script combines the Elo burn-in, convergence, parameter validation,
and volatility results into a meeting-ready baseline decision summary.

The purpose is not to find a single universally optimal Elo model, but to
identify transparent and defensible Elo baselines for later comparison with
Glicko.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd


FULL_HISTORY_START_YEAR = 1985
END_YEAR = 2025

MAIN_CANDIDATES = [
    {
        "candidate_name": "Conservative Elo",
        "setting_name": "conservative_k10_scale500",
        "k": 10.0,
        "scale": 500.0,
        "role": "stability-oriented reference",
        "recommended_use": "Use as a low-volatility sensitivity/reference baseline.",
    },
    {
        "candidate_name": "Default Elo",
        "setting_name": "default_k20_scale500",
        "k": 20.0,
        "scale": 500.0,
        "role": "transparent simple baseline",
        "recommended_use": "Carry forward as the transparent and easy-to-explain simple Elo baseline.",
    },
    {
        "candidate_name": "Validation-best Elo",
        "setting_name": "validation_best_k30_scale300",
        "k": 30.0,
        "scale": 300.0,
        "role": "prediction-oriented baseline selected by validation",
        "recommended_use": "Carry forward as the prediction-oriented Elo baseline, with stability/volatility reported alongside prediction metrics.",
    },
]

SENSITIVITY_SETTINGS = ["aggressive_k35_scale300", "aggressive_k40_scale400"]
KEY_BURNIN_START_YEARS = [2005, 2010, 2015, 2020, 2025]
SHORTER_BURNIN_CANDIDATE_YEARS = [2005, 2010, 2015, 2020]

BURNIN_THRESHOLDS = {
    "top50_overlap": 0.90,
    "top100_overlap": 0.90,
    "spearman_rank_correlation": 0.98,
    "mean_abs_rating_difference": 25.0,
}


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

INPUT_PATHS = {
    "full_history_md": OUTPUT_DIR / "full_history_dataset_summary.md",
    "full_history_checks": OUTPUT_DIR / "full_history_data_check_summary_1985_2025.csv",
    "burnin_prediction_metrics": OUTPUT_DIR / "elo_burnin_prediction_metrics.csv",
    "burnin_vs_reference": OUTPUT_DIR / "elo_burnin_vs_1985_reference.csv",
    "burnin_adjacent": OUTPUT_DIR / "elo_burnin_adjacent_start_year_comparisons.csv",
    "burnin_active_counts": OUTPUT_DIR / "elo_burnin_active_player_counts.csv",
    "burnin_date_ordering": OUTPUT_DIR / "elo_burnin_date_ordering_summary.csv",
    "rerun_decisions": OUTPUT_DIR / "elo_single_year_rerun_convergence_decisions.csv",
    "rerun_iteration_summary": OUTPUT_DIR / "elo_single_year_rerun_iteration_summary.csv",
    "event_match_summary": OUTPUT_DIR / "elo_event_level_volatility_match_summary.csv",
    "event_summary": OUTPUT_DIR / "elo_event_level_volatility_event_summary.csv",
    "event_size_summary": OUTPUT_DIR / "elo_event_level_volatility_by_event_size.csv",
}

OPTIONAL_INPUT_PATHS = {
    "best_parameter_test": DATA_PROCESSED / "best_parameter_test_result_2015_2025.csv",
    "parameter_validation": DATA_PROCESSED / "parameter_validation_results_2015_2025.csv",
    "rating_stability": DATA_PROCESSED / "rating_stability_results_2015_2025.csv",
}

SKIPPED_LARGE_FILE = OUTPUT_DIR / "elo_burnin_update_history_all_runs.csv"

CANDIDATES_PATH = OUTPUT_DIR / "elo_candidate_baselines.csv"
EVIDENCE_TABLE_PATH = OUTPUT_DIR / "elo_baseline_evidence_table.csv"
BURNIN_KEY_FINDINGS_PATH = OUTPUT_DIR / "elo_burnin_key_findings_for_meeting.csv"
VOLATILITY_KEY_FINDINGS_PATH = OUTPUT_DIR / "elo_volatility_key_findings_for_meeting.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "elo_baseline_decision_summary.md"


def read_required_inputs() -> Dict[str, pd.DataFrame]:
    """Read all required compact CSV inputs."""
    inputs = {}
    missing = []
    for name, path in INPUT_PATHS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        if path.suffix.lower() == ".csv":
            inputs[name] = pd.read_csv(path)
        else:
            inputs[name] = path.read_text(encoding="utf-8")

    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return inputs


def read_optional_inputs() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Read optional previous validation/stability files if available."""
    optional = {}
    missing = []
    for name, path in OPTIONAL_INPUT_PATHS.items():
        if path.exists():
            optional[name] = pd.read_csv(path)
        else:
            missing.append(str(path))
    return optional, missing


def print_input_status(missing_optional: List[str]) -> None:
    """Print input file status for terminal log."""
    print("Required input files:")
    for path in INPUT_PATHS.values():
        print(f"  found: {path}")
    print("Optional input files:")
    for name, path in OPTIONAL_INPUT_PATHS.items():
        status = "found" if path.exists() else "missing"
        print(f"  {status}: {path}")
    if missing_optional:
        print("Missing optional inputs:")
        for path in missing_optional:
            print(f"  {path}")
    else:
        print("Missing optional inputs: none")
    print(f"Skipped large file: {SKIPPED_LARGE_FILE}")


def get_full_history_dataset_info(full_history_checks: pd.DataFrame) -> Dict[str, int]:
    """Extract full-history dataset size and checks."""
    row = full_history_checks.iloc[0]
    return {
        "total_matches": int(row["total_number_of_matches"]),
        "unique_players": int(row["number_of_unique_players"]),
        "unique_events": int(row["number_of_unique_events"]),
        "missing_event_date_parsed": int(row.get("missing_event_date_parsed", 0)),
        "duplicated_fcode_count": int(row["duplicated_fcode_count"]),
        "missing_event_rows": int(row["missing_event_rows"]),
        "missing_hidx_rows": int(row["missing_hidx_rows"]),
        "missing_winner_names": int(row["missing_winner_names"]),
        "missing_loser_names": int(row["missing_loser_names"]),
    }


def setting_row(df: pd.DataFrame, setting_name: str) -> pd.Series:
    """Return one row for a setting name."""
    subset = df[df["setting_name"] == setting_name]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.iloc[0]


def validation_row(optional_inputs: Dict[str, pd.DataFrame], k: float, scale: float) -> pd.Series:
    """Return validation result for a K/scale setting if available."""
    validation = optional_inputs.get("parameter_validation")
    if validation is None or validation.empty:
        return pd.Series(dtype=object)
    subset = validation[(validation["K"].astype(float) == float(k)) & (validation["scale"].astype(float) == float(scale))]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.iloc[0]


def compute_burnin_candidate(
    burnin_vs_reference: pd.DataFrame,
    setting_name: str,
) -> Tuple[object, str]:
    """Find the shortest selected start year meeting diagnostic stability thresholds."""
    subset = burnin_vs_reference[
        (burnin_vs_reference["setting_name"] == setting_name)
        & (burnin_vs_reference["player_subset"] == "active_2025_games_ge5")
        & (burnin_vs_reference["comparison_start_year"].isin(SHORTER_BURNIN_CANDIDATE_YEARS))
    ].copy()
    if subset.empty:
        return np.nan, "No active_2025_games_ge5 burn-in rows were available."

    subset["meets_thresholds"] = (
        (subset["top50_overlap"] >= BURNIN_THRESHOLDS["top50_overlap"])
        & (subset["top100_overlap"] >= BURNIN_THRESHOLDS["top100_overlap"])
        & (subset["spearman_rank_correlation"] >= BURNIN_THRESHOLDS["spearman_rank_correlation"])
        & (subset["mean_abs_rating_difference"] <= BURNIN_THRESHOLDS["mean_abs_rating_difference"])
    )
    candidates = subset[subset["meets_thresholds"]].sort_values("comparison_start_year", ascending=False)
    if candidates.empty:
        best = subset.sort_values(
            ["top50_overlap", "top100_overlap", "spearman_rank_correlation"],
            ascending=[False, False, False],
        ).iloc[0]
        return (
            np.nan,
            "No shorter selected start year met all diagnostic thresholds; "
            f"best inspected row was {int(best['comparison_start_year'])} with "
            f"top50={best['top50_overlap']:.3f}, top100={best['top100_overlap']:.3f}, "
            f"Spearman={best['spearman_rank_correlation']:.4f}, "
            f"mean abs diff={best['mean_abs_rating_difference']:.2f}.",
        )

    chosen = candidates.iloc[0]
    return (
        int(chosen["comparison_start_year"]),
        "Under the provisional diagnostic thresholds, "
        f"{int(chosen['comparison_start_year'])} is the shortest selected start year meeting "
        f"top50/top100 overlap, Spearman and mean absolute rating-difference checks.",
    )


def make_rerun_comment(rerun_decisions: pd.DataFrame, setting_name: str) -> str:
    """Summarise single-year rerun convergence for one setting."""
    subset = rerun_decisions[rerun_decisions["setting_name"] == setting_name]
    if subset.empty:
        return "Single-year rerun result not available."
    converged_count = int(subset["converged"].sum())
    total = len(subset)
    median_mean_change = subset["final_mean_abs_change"].median()
    median_spearman = subset["final_spearman_rank_correlation"].median()
    return (
        f"{converged_count}/{total} tested years converged within 50 iterations; "
        f"median final mean abs change={median_mean_change:.3f}, "
        f"median final Spearman={median_spearman:.6f}. Rank ordering was nearly stable, "
        "but rating values did not meet strict numerical convergence thresholds."
    )


def make_final_recommendation(setting_name: str) -> str:
    """Return recommendation wording by candidate setting."""
    if setting_name == "conservative_k10_scale500":
        return "Keep as a stability-oriented reference, not necessarily as the main predictive baseline."
    if setting_name == "default_k20_scale500":
        return "Carry forward as the transparent simple Elo baseline for Glicko comparison."
    if setting_name == "validation_best_k30_scale300":
        return "Carry forward as the prediction-oriented Elo baseline, with volatility/stability diagnostics reported."
    return "Use only as sensitivity evidence."


def build_candidate_baselines(
    inputs: Dict[str, pd.DataFrame],
    optional_inputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one-row-per-candidate baseline table."""
    prediction_metrics = inputs["burnin_prediction_metrics"]
    burnin_vs_reference = inputs["burnin_vs_reference"]
    rerun_decisions = inputs["rerun_decisions"]
    match_summary = inputs["event_match_summary"]
    event_summary = inputs["event_summary"]

    rows = []
    for candidate in MAIN_CANDIDATES:
        setting_name = candidate["setting_name"]
        k = candidate["k"]
        scale = candidate["scale"]

        pred = prediction_metrics[
            (prediction_metrics["setting_name"] == setting_name)
            & (prediction_metrics["start_year"] == FULL_HISTORY_START_YEAR)
        ].iloc[0]
        val = validation_row(optional_inputs, k, scale)
        match = setting_row(match_summary, setting_name)
        event = setting_row(event_summary, setting_name)
        burnin_start, burnin_comment = compute_burnin_candidate(burnin_vs_reference, setting_name)
        rerun_comment = make_rerun_comment(rerun_decisions, setting_name)

        row = {
            **candidate,
            "2025_full_history_evaluation_games": pred["evaluation_games"],
            "2025_full_history_log_loss": pred["log_loss"],
            "2025_full_history_brier": pred["brier_score"],
            "2025_full_history_accuracy": pred["accuracy"],
            "2025_full_history_baseline_accuracy": pred["baseline_accuracy"],
            "validation_log_loss": val.get("validation_log_loss", np.nan),
            "validation_brier": val.get("validation_brier_score", np.nan),
            "validation_accuracy": val.get("validation_accuracy", np.nan),
            "previous_2025_test_log_loss": val.get("test_log_loss", np.nan),
            "previous_2025_test_brier": val.get("test_brier_score", np.nan),
            "previous_2025_test_accuracy": val.get("test_accuracy", np.nan),
            "mean_abs_match_update": match.get("mean_abs_match_update", np.nan),
            "p95_abs_match_update": match.get("p95_abs_match_update", np.nan),
            "mean_abs_event_net_change": event.get("mean_abs_event_net_change", np.nan),
            "mean_cumulative_abs_match_updates_in_event": event.get(
                "mean_cumulative_abs_match_updates_in_event", np.nan
            ),
            "mean_event_cancellation_ratio": event.get("mean_event_cancellation_ratio", np.nan),
            "burnin_candidate_start_year": burnin_start,
            "burnin_stability_comment": burnin_comment,
            "rerun_convergence_comment": rerun_comment,
            "final_recommendation": make_final_recommendation(setting_name),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_evidence(
    rows: List[Dict[str, object]],
    setting_name: str,
    k: float,
    scale: float,
    evidence_type: str,
    metric_name: str,
    metric_value: object,
    interpretation: str,
) -> None:
    """Append a row to the evidence table."""
    rows.append(
        {
            "setting_name": setting_name,
            "k": k,
            "scale": scale,
            "evidence_type": evidence_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "interpretation": interpretation,
        }
    )


def build_evidence_table(
    candidate_baselines: pd.DataFrame,
    inputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build long-form evidence table."""
    rows: List[Dict[str, object]] = []
    burnin_vs = inputs["burnin_vs_reference"]
    rerun = inputs["rerun_decisions"]
    event_size = inputs["event_size_summary"]

    for _, candidate in candidate_baselines.iterrows():
        setting_name = candidate["setting_name"]
        k = candidate["k"]
        scale = candidate["scale"]
        add_evidence(rows, setting_name, k, scale, "prediction", "2025_full_history_log_loss", candidate["2025_full_history_log_loss"], "Lower is better; computed on fixed 2025 games after 1985-2025 run.")
        add_evidence(rows, setting_name, k, scale, "prediction", "2025_full_history_accuracy", candidate["2025_full_history_accuracy"], "Auxiliary accuracy on the fixed 2025 games.")
        add_evidence(rows, setting_name, k, scale, "burnin", "burnin_candidate_start_year", candidate["burnin_candidate_start_year"], candidate["burnin_stability_comment"])
        add_evidence(rows, setting_name, k, scale, "rerun_convergence", "converged_year_tests", int(rerun[rerun["setting_name"] == setting_name]["converged"].sum()), "Number of 2023-2025 single-year repeated rerun tests converged within 50 iterations.")
        add_evidence(rows, setting_name, k, scale, "volatility", "mean_abs_match_update", candidate["mean_abs_match_update"], "Mean absolute per-match Elo update.")
        add_evidence(rows, setting_name, k, scale, "volatility", "mean_abs_event_net_change", candidate["mean_abs_event_net_change"], "Mean absolute player-event net rating change.")
        add_evidence(rows, setting_name, k, scale, "volatility", "mean_event_cancellation_ratio", candidate["mean_event_cancellation_ratio"], "Event net movement divided by cumulative within-event movement; lower means more within-event cancellation.")

    vb_size = event_size[event_size["setting_name"] == "validation_best_k30_scale300"]
    for bucket in ["1 game", "10+ games"]:
        subset = vb_size[vb_size["games_in_event_bucket"] == bucket]
        if not subset.empty:
            row = subset.iloc[0]
            add_evidence(
                rows,
                "validation_best_k30_scale300",
                float(row["k"]),
                float(row["scale"]),
                "event_size_effect",
                f"{bucket}_mean_event_cancellation_ratio",
                row["mean_event_cancellation_ratio"],
                "Shows how event size changes the relationship between cumulative match movement and net event change.",
            )

    return pd.DataFrame(rows)


def build_burnin_key_findings(burnin_vs_reference: pd.DataFrame) -> pd.DataFrame:
    """Extract meeting-focused burn-in rows."""
    key = burnin_vs_reference[
        (burnin_vs_reference["setting_name"].isin([c["setting_name"] for c in MAIN_CANDIDATES]))
        & (burnin_vs_reference["player_subset"] == "active_2025_games_ge5")
        & (burnin_vs_reference["comparison_start_year"].isin(KEY_BURNIN_START_YEARS))
    ].copy()

    cols = [
        "setting_name",
        "k",
        "scale",
        "reference_start_year",
        "comparison_start_year",
        "player_subset",
        "min_2025_games",
        "number_of_common_players",
        "mean_abs_rating_difference",
        "median_abs_rating_difference",
        "p90_abs_rating_difference",
        "spearman_rank_correlation",
        "top50_overlap",
        "top100_overlap",
    ]
    return key[cols].sort_values(["setting_name", "comparison_start_year"])


def build_volatility_key_findings(
    match_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build meeting-focused volatility key rows."""
    merged = match_summary.merge(
        event_summary,
        on=["setting_name", "k", "scale", "start_year", "end_year"],
        how="inner",
        suffixes=("_match", "_event"),
    )
    cols = [
        "setting_name",
        "k",
        "scale",
        "start_year",
        "end_year",
        "number_of_matches",
        "mean_abs_match_update",
        "p95_abs_match_update",
        "p99_abs_match_update",
        "mean_abs_event_net_change",
        "p95_abs_event_net_change",
        "mean_cumulative_abs_match_updates_in_event",
        "mean_event_cancellation_ratio",
        "median_event_cancellation_ratio",
    ]
    return merged[cols].sort_values(["k", "scale"])


def format_metric(value: object, digits: int = 3) -> str:
    """Format optional numeric metric for markdown."""
    if pd.isna(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def make_candidate_markdown_lines(candidate_baselines: pd.DataFrame) -> List[str]:
    """Create markdown bullets for candidate baselines."""
    lines = []
    for _, row in candidate_baselines.iterrows():
        lines.append(
            f"* {row['candidate_name']} (`{row['setting_name']}`, K={row['k']:g}, scale={row['scale']:g}): "
            f"{row['role']}. 2025 full-history log loss {row['2025_full_history_log_loss']:.6f}, "
            f"accuracy {row['2025_full_history_accuracy']:.3f}; "
            f"mean abs match update {row['mean_abs_match_update']:.3f}; "
            f"mean event cancellation ratio {row['mean_event_cancellation_ratio']:.3f}. "
            f"Recommendation: {row['final_recommendation']}"
        )
    return lines


def write_markdown_summary(
    inputs: Dict[str, pd.DataFrame],
    optional_inputs: Dict[str, pd.DataFrame],
    missing_optional: List[str],
    candidate_baselines: pd.DataFrame,
    burnin_key: pd.DataFrame,
    volatility_key: pd.DataFrame,
    output_path: Path,
) -> str:
    """Write the main meeting-ready Elo baseline decision summary."""
    info = get_full_history_dataset_info(inputs["full_history_checks"])
    date_ordering = inputs["burnin_date_ordering"]
    rerun = inputs["rerun_decisions"]
    event_size = inputs["event_size_summary"]
    validation_best = candidate_baselines[candidate_baselines["setting_name"] == "validation_best_k30_scale300"].iloc[0]
    default = candidate_baselines[candidate_baselines["setting_name"] == "default_k20_scale500"].iloc[0]

    date_lines = [
        f"* {row['event_date_ordering_method']}: {int(row['match_count'])} matches"
        for _, row in date_ordering.iterrows()
    ]
    candidate_lines = make_candidate_markdown_lines(candidate_baselines)

    burnin_focus = burnin_key[
        (burnin_key["setting_name"] == "validation_best_k30_scale300")
        & (burnin_key["comparison_start_year"].isin([2005, 2010, 2015, 2020]))
    ]
    burnin_lines = []
    for _, row in burnin_focus.iterrows():
        burnin_lines.append(
            f"* start {int(row['comparison_start_year'])}: mean abs rating diff "
            f"{row['mean_abs_rating_difference']:.2f}, Spearman {row['spearman_rank_correlation']:.4f}, "
            f"top50 {row['top50_overlap']:.3f}, top100 {row['top100_overlap']:.3f}."
        )

    rerun_lines = []
    for setting_name, group in rerun.groupby("setting_name", sort=False):
        converged_count = int(group["converged"].sum())
        median_mean = group["final_mean_abs_change"].median()
        median_spearman = group["final_spearman_rank_correlation"].median()
        rerun_lines.append(
            f"* {setting_name}: {converged_count}/{len(group)} year-setting tests converged within 50 iterations; "
            f"median final mean abs change {median_mean:.3f}, median Spearman {median_spearman:.6f}."
        )

    volatility_lines = []
    for _, row in volatility_key.iterrows():
        if row["setting_name"] in [c["setting_name"] for c in MAIN_CANDIDATES] + SENSITIVITY_SETTINGS:
            volatility_lines.append(
                f"* {row['setting_name']}: mean abs match update {row['mean_abs_match_update']:.3f}, "
                f"p95 match update {row['p95_abs_match_update']:.3f}, "
                f"mean event net change {row['mean_abs_event_net_change']:.3f}, "
                f"mean cumulative event movement {row['mean_cumulative_abs_match_updates_in_event']:.3f}, "
                f"cancellation ratio {row['mean_event_cancellation_ratio']:.3f}."
            )

    vb_size = event_size[event_size["setting_name"] == "validation_best_k30_scale300"]
    one_game_ratio = vb_size[vb_size["games_in_event_bucket"] == "1 game"]["mean_event_cancellation_ratio"]
    ten_plus_ratio = vb_size[vb_size["games_in_event_bucket"] == "10+ games"]["mean_event_cancellation_ratio"]
    one_game_text = format_metric(one_game_ratio.iloc[0]) if not one_game_ratio.empty else "not available"
    ten_plus_text = format_metric(ten_plus_ratio.iloc[0]) if not ten_plus_ratio.empty else "not available"

    optional_text = "None" if not missing_optional else "\n".join(f"* {path}" for path in missing_optional)
    vb_burnin = validation_best["burnin_candidate_start_year"]
    if pd.isna(vb_burnin):
        vb_burnin_text = "No shorter selected start year met all diagnostic thresholds."
    else:
        vb_burnin_text = (
            f"For the validation-best setting and active 2025 players with at least 5 games, "
            f"{int(vb_burnin)} appears to be a defensible shorter burn-in candidate under the chosen diagnostic thresholds. "
            "This is an empirical diagnostic, not a theoretical rule."
        )

    markdown = f"""# Elo baseline decision summary

## 1. Aim of this Elo baseline decision stage

The aim is to stop expanding Elo indefinitely and instead consolidate a defensible set of Elo baselines before implementing Glicko.
The purpose is not to find a single universally optimal Elo model, but to identify transparent and defensible Elo baselines for later comparison with Glicko.

## 2. Data used

The full-history checked match-level dataset covers 1985-2025.

* Matches: {info['total_matches']}
* Players: {info['unique_players']}
* Events: {info['unique_events']}
* duplicated fcode: {info['duplicated_fcode_count']}
* missing event rows: {info['missing_event_rows']}
* missing hidx rows: {info['missing_hidx_rows']}
* missing winner names: {info['missing_winner_names']}
* missing loser names: {info['missing_loser_names']}

Small numbers of early event dates had incomplete day-level information. These matches were not deleted. The burn-in scripts used `event_order_date` and `event_date_ordering_method`:

{chr(10).join(date_lines)}

## 3. What has been completed

* Full-history burn-in stability: compares different start years against the 1985-2025 reference final rating list.
* Single-year repeated rerun convergence: repeats a single year of matches to check whether rating values stabilise.
* Event-level volatility: compares match-level updates with player-event net rating changes.
* Previous parameter validation and stability analysis: uses 2015-2022 burn-in, 2023-2024 validation and 2025 test, where available.

Missing optional inputs:

{optional_text}

## 4. Burn-in stability findings

The full-history start year 1985 has stronger 2025 prediction metrics than 2025-only runs across the tested Elo settings.
The final rating list diverges more from the 1985 reference as the start year moves later.
For current rankings, the active 2025 subsets are more interpretable than all historical players because many older players are inactive.

For the validation-best setting on `active_2025_games_ge5`:

{chr(10).join(burnin_lines)}

{vb_burnin_text}

The diagnostic thresholds used here are provisional: top50 overlap >= 0.90, top100 overlap >= 0.90, Spearman >= 0.98 and mean absolute rating difference <= 25.

## 5. Single-year rerun convergence findings

The single-year repeated rerun experiment is not a prediction evaluation because it reuses the same year's matches repeatedly.
Across the 9 year-setting combinations, none reached the strict numerical convergence thresholds within 50 iterations.
Rank ordering became almost stable, but rating values continued to move by more than the strict thresholds.

{chr(10).join(rerun_lines)}

This supports the idea that a single season of repeated information is not a substitute for historical burn-in.

## 6. Event-level volatility findings

Larger K values increase both match-level updates and event-level net changes.
The validation-best K=30, scale=300 setting has stronger prediction results than the default, but it also has larger match-level updates.
However, event-level net change is smaller than cumulative match movement because wins and losses within an event can cancel.

{chr(10).join(volatility_lines)}

For validation-best Elo, mean cumulative event movement is approximately {validation_best['mean_cumulative_abs_match_updates_in_event']:.3f}, while mean event net change is approximately {validation_best['mean_abs_event_net_change']:.3f}, with cancellation ratio {validation_best['mean_event_cancellation_ratio']:.3f}.
The validation-best cancellation ratio is about {one_game_text} for 1-game player-events and {ten_plus_text} for 10+ game player-events, showing stronger cancellation in longer events.

## 7. Candidate Elo baselines

{chr(10).join(candidate_lines)}

Aggressive settings such as `aggressive_k35_scale300` and `aggressive_k40_scale400` are useful as sensitivity checks, but they should not be treated as main baselines because they are more volatile.

## 8. Provisional recommendation

* Carry forward default Elo as the transparent baseline.
* Carry forward validation-best Elo as the prediction-oriented baseline.
* Keep conservative Elo as a stability reference.
* Do not use K=35 or K=40 aggressive variants as main baselines; keep them only as sensitivity/warning examples.

This keeps the Elo side defensible without pretending that one parameter set is universally optimal.

## 9. Implication for Glicko comparison

Glicko should be compared with Elo on the same match list, the same chronological ordering and the same evaluation metrics.
If Glicko improves performance, the interpretation should not be based only on log loss.
The comparison should also consider rating uncertainty, new or inactive players, rating stability and interpretability.

## 10. Remaining questions for supervisor

* Is it reasonable to carry forward both default Elo and validation-best Elo for comparison with Glicko?
* Should the shorter burn-in candidate, such as 2005, be used for computational convenience, or should I use the full 1985 history as the main reference?
* For the dissertation, should I present conservative Elo only as a sensitivity check rather than a main baseline?
* Does the event-level volatility analysis address the match-by-match versus tournament-resolution concern adequately?
* Before implementing Glicko, are there any additional Elo checks you would expect?
"""
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def remove_existing_outputs() -> None:
    """Remove this script's own outputs before a fresh summary run."""
    for path in [
        CANDIDATES_PATH,
        EVIDENCE_TABLE_PATH,
        BURNIN_KEY_FINDINGS_PATH,
        VOLATILITY_KEY_FINDINGS_PATH,
        SUMMARY_MD_PATH,
    ]:
        if path.exists():
            path.unlink()


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs()

    print("=== Elo baseline decision summary ===")
    print("This script reads compact outputs only and does not rerun Elo.")
    print(f"Skipping large file: {SKIPPED_LARGE_FILE}")

    inputs = read_required_inputs()
    optional_inputs, missing_optional = read_optional_inputs()
    print_input_status(missing_optional)

    candidate_baselines = build_candidate_baselines(inputs, optional_inputs)
    evidence_table = build_evidence_table(candidate_baselines, inputs)
    burnin_key = build_burnin_key_findings(inputs["burnin_vs_reference"])
    volatility_key = build_volatility_key_findings(inputs["event_match_summary"], inputs["event_summary"])

    candidate_baselines.to_csv(CANDIDATES_PATH, index=False)
    evidence_table.to_csv(EVIDENCE_TABLE_PATH, index=False)
    burnin_key.to_csv(BURNIN_KEY_FINDINGS_PATH, index=False)
    volatility_key.to_csv(VOLATILITY_KEY_FINDINGS_PATH, index=False)
    write_markdown_summary(
        inputs=inputs,
        optional_inputs=optional_inputs,
        missing_optional=missing_optional,
        candidate_baselines=candidate_baselines,
        burnin_key=burnin_key,
        volatility_key=volatility_key,
        output_path=SUMMARY_MD_PATH,
    )

    print(f"Candidate baselines: {len(candidate_baselines)}")
    print("Output paths:")
    print(f"  candidate baselines: {CANDIDATES_PATH}")
    print(f"  evidence table: {EVIDENCE_TABLE_PATH}")
    print(f"  burn-in key findings: {BURNIN_KEY_FINDINGS_PATH}")
    print(f"  volatility key findings: {VOLATILITY_KEY_FINDINGS_PATH}")
    print(f"  markdown summary: {SUMMARY_MD_PATH}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
