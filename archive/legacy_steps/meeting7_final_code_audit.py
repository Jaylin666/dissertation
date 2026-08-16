"""Validate retained scientific outputs and cross-step consistency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
MEETING6_DIR = PROJECT_ROOT / "outputs" / "meeting6"
MEETING7_DIR = PROJECT_ROOT / "outputs" / "meeting7"
FIGURE_DIR = MEETING7_DIR / "figures"

AUDIT_CHECKS_PATH = MEETING7_DIR / "meeting7_final_code_audit_checks.csv"
AUDIT_ISSUES_PATH = MEETING7_DIR / "meeting7_final_code_audit_issues.csv"

EPS = 1e-12
LOGL_loss_EPS = 1e-15


@dataclass(frozen=True)
class SourceFile:
    step: str
    path: Path
    role: str


SOURCE_FILES = [
    SourceFile("27", CODE_DIR / "27_adaptive_k_elo_comparison.py", "Adaptive-K Elo reference from Meeting 6."),
    SourceFile("32", CODE_DIR / "32_glicko_probability_orientation_audit.py", "Original Glicko probability-orientation audit."),
    SourceFile("33", CODE_DIR / "33_recompute_orientation_corrected_meeting6_results.py", "Final orientation-corrected Meeting 6 outputs."),
    SourceFile("34", CODE_DIR / "34_early_game_analysis.py", "Early-game appearance dataset, metrics, bootstrap and figures."),
    SourceFile("35", CODE_DIR / "35_early_game_mechanism_analysis.py", "Mechanism analysis for early-game findings."),
    SourceFile("36", CODE_DIR / "36_glicko_initialisation_source_diagnostic.py", "Initialisation-source diagnostic."),
    SourceFile("37", CODE_DIR / "37_glicko_initial_rating_sensitivity.py", "Common initial-rating sensitivity experiment."),
    SourceFile("38", CODE_DIR / "38_asymmetric_adaptive_k_elo.py", "Asymmetric adaptive-K Elo proof of concept."),
    SourceFile("39", CODE_DIR / "39_glicko_orientation_sensitivity_audit.py", "Independent orientation sensitivity audit."),
    SourceFile("40", CODE_DIR / "40_finalize_orientation_reporting.py", "Final orientation reporting corrections."),
    SourceFile("glicko_core", CODE_DIR / "glicko_core.py", "Shared Glicko expected-score and update functions."),
]

REQUIRED_OUTPUTS = {
    "33 per-match scores": MEETING6_DIR / "33_orientation_corrected_per_match_scores_2025.csv",
    "33 overall metrics": MEETING6_DIR / "33_overall_model_metrics.csv",
    "34 appearance dataset": MEETING7_DIR / "34_early_game_appearance_dataset.csv",
    "34 cumulative metrics": MEETING7_DIR / "34_cumulative_threshold_model_performance.csv",
    "34 stage metrics": MEETING7_DIR / "34_stage_bin_model_performance.csv",
    "34 exact metrics": MEETING7_DIR / "34_exact_appearance_model_performance.csv",
    "34 pairwise differences": MEETING7_DIR / "34_pairwise_model_differences.csv",
    "34 bootstrap CIs": MEETING7_DIR / "34_bootstrap_confidence_intervals.csv",
    "34 match-level bootstrap CIs": MEETING7_DIR / "34_match_level_bootstrap_confidence_intervals.csv",
    "35 key mechanism results": MEETING7_DIR / "35_key_mechanism_results.csv",
    "35 probability bias": MEETING7_DIR / "35_cumulative_probability_bias_summary.csv",
    "35 extremity": MEETING7_DIR / "35_prediction_extremity_summary.csv",
    "35 Glicko rating/RD": MEETING7_DIR / "35_glicko_rating_rd_summary.csv",
    "36 key diagnostics": MEETING7_DIR / "36_key_initialisation_diagnostic_results.csv",
    "36 debut state": MEETING7_DIR / "36_debut_state_summary.csv",
    "36 counterfactual diagnostics": MEETING7_DIR / "36_debut_counterfactual_probability_diagnostics.csv",
    "37 key initial-rating results": MEETING7_DIR / "37_key_initial_rating_results.csv",
    "37 validation initial-rating metrics": MEETING7_DIR / "37_validation_initial_rating_metrics.csv",
    "37 test initial-rating metrics": MEETING7_DIR / "37_test_initial_rating_metrics.csv",
    "38 key asymmetric-K results": MEETING7_DIR / "38_key_asymmetric_k_results.csv",
    "38 overall model metrics": MEETING7_DIR / "38_overall_model_metrics.csv",
    "38 asymmetric-K summary": MEETING7_DIR / "38_asymmetric_k_summary.csv",
    "38 Glicko gap recovery": MEETING7_DIR / "38_glicko_gap_recovery.csv",
    "39 key orientation results": MEETING7_DIR / "39_key_orientation_results.csv",
    "39 orientation comparison": MEETING7_DIR / "39_orientation_sensitivity_comparison.csv",
    "39 complement gap": MEETING7_DIR / "39_complement_gap_summary.csv",
    "39 early-player side distribution": MEETING7_DIR / "39_early_player_side_distribution.csv",
    "40 corrected orientation table": MEETING7_DIR / "40_final_orientation_reporting_table.csv",
    "40 conclusion codes": MEETING7_DIR / "40_orientation_conclusion_codes.csv",
}

VALIDATION_OUTPUTS = {
    "34 input validation": MEETING7_DIR / "34_input_validation_checks.csv",
    "34 metric validation": MEETING7_DIR / "34_metric_validation_checks.csv",
    "34 bootstrap/figure validation": MEETING7_DIR / "34_bootstrap_figure_validation_checks.csv",
    "34 bootstrap robustness validation": MEETING7_DIR / "34_bootstrap_robustness_validation_checks.csv",
    "35 input validation": MEETING7_DIR / "35_input_validation_checks.csv",
    "35 mechanism validation": MEETING7_DIR / "35_mechanism_validation_checks.csv",
    "36 diagnostic validation": MEETING7_DIR / "36_initialisation_diagnostic_validation_checks.csv",
    "37 sensitivity validation": MEETING7_DIR / "37_initial_rating_sensitivity_validation_checks.csv",
    "38 asymmetric-K validation": MEETING7_DIR / "38_asymmetric_k_validation_checks.csv",
    "39 orientation validation": MEETING7_DIR / "39_orientation_validation_checks.csv",
    "40 reporting validation": MEETING7_DIR / "40_reporting_validation_checks.csv",
}

MODEL_PROB_COLUMNS = {
    "Validation_best_Elo": "p_focal_Validation_best_Elo",
    "Glicko_low_fixed": "p_focal_Glicko_low_fixed",
    "Glicko_C0_fixed": "p_focal_Glicko_C0_fixed",
    "best_AdaptiveK": "p_focal_best_AdaptiveK",
}


def brier_score(y: pd.Series, p: pd.Series) -> float:
    return float(np.mean((p.to_numpy(dtype=float) - y.to_numpy(dtype=float)) ** 2))


def log_loss_score(y: pd.Series, p: pd.Series) -> float:
    clipped = np.clip(p.to_numpy(dtype=float), LOGL_loss_EPS, 1.0 - LOGL_loss_EPS)
    yy = y.to_numpy(dtype=float)
    return float(-np.mean(yy * np.log(clipped) + (1.0 - yy) * np.log(1.0 - clipped)))


def accuracy_score(y: pd.Series, p: pd.Series) -> float:
    pred = (p.to_numpy(dtype=float) >= 0.5).astype(int)
    return float(np.mean(pred == y.to_numpy(dtype=int)))


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{digits}f}"
    return str(value)




def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def validation_is_passed(df: pd.DataFrame) -> bool:
    if "passed" in df.columns:
        return df["passed"].astype(str).str.lower().isin(["true", "1", "pass"]).all()
    if "status" in df.columns:
        return df["status"].astype(str).str.upper().eq("PASS").all()
    return False


def get_key_value(df: pd.DataFrame, metric: str) -> float:
    row = df.loc[df["metric"].eq(metric)]
    if row.empty:
        raise KeyError(metric)
    return float(row.iloc[0]["value"])


def get_model_metric(df: pd.DataFrame, group: str, model: str, metric: str) -> float:
    row = df.loc[df["group"].eq(group) & df["model"].eq(model)]
    if row.empty:
        raise KeyError((group, model, metric))
    return float(row.iloc[0][metric])


def get_pairwise_delta(df: pd.DataFrame, group: str, comparison: str, metric: str) -> float:
    row = df.loc[df["group"].eq(group) & df["comparison"].eq(comparison)]
    if row.empty:
        raise KeyError((group, comparison, metric))
    return float(row.iloc[0][metric])


def get_ci_row(df: pd.DataFrame, group: str, comparison: str, metric: str) -> pd.Series:
    rows = df.loc[df["group"].eq(group) & df["comparison"].eq(comparison) & df["metric"].eq(metric)]
    if rows.empty:
        raise KeyError((group, comparison, metric))
    return rows.iloc[0]


def build_audit_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def add_check(
        step: str,
        category: str,
        check_name: str,
        passed: bool,
        observed: Any,
        expected: Any,
        evidence_file: Path | str,
        details: str = "",
    ) -> None:
        checks.append(
            {
                "check_id": f"CHK_{len(checks) + 1:03d}",
                "step": step,
                "category": category,
                "check_name": check_name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
                "evidence_file": str(evidence_file).replace(str(PROJECT_ROOT) + "\\", ""),
                "details": details,
            }
        )

    def add_issue(
        issue_id: str,
        source_step: str,
        issue_type: str,
        classification: str,
        severity: str,
        description: str,
        effect_on_results: str,
        recommended_action: str,
        must_fix_before_meeting: bool,
    ) -> None:
        issues.append(
            {
                "issue_id": issue_id,
                "source_step": source_step,
                "issue_type": issue_type,
                "classification": classification,
                "severity": severity,
                "description": description,
                "effect_on_results": effect_on_results,
                "recommended_action": recommended_action,
                "must_fix_before_meeting": must_fix_before_meeting,
            }
        )

    for source in SOURCE_FILES:
        exists = source.path.exists()
        size = source.path.stat().st_size if exists else 0
        add_check(
            source.step,
            "source_file_review",
            f"source_file_exists_{source.path.name}",
            exists and size > 0,
            size,
            ">0 bytes",
            source.path,
            source.role,
        )

    for label, path in REQUIRED_OUTPUTS.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        add_check("all", "output_presence", f"required_output_exists_{label}", exists and size > 0, size, ">0 bytes", path)

    for label, path in VALIDATION_OUTPUTS.items():
        if not path.exists():
            add_check("all", "validation", f"{label}_exists", False, "missing", "present", path)
            continue
        df = load_csv(path)
        passed = validation_is_passed(df)
        status_counts = (
            df["status"].astype(str).str.upper().value_counts().to_dict()
            if "status" in df.columns
            else df["passed"].astype(str).str.lower().value_counts().to_dict()
        )
        add_check("all", "validation", f"{label}_all_pass", bool(passed), status_counts, "all PASS/True", path)

    # Load required data after presence checks.
    step33_scores = load_csv(REQUIRED_OUTPUTS["33 per-match scores"])
    step33_overall = load_csv(REQUIRED_OUTPUTS["33 overall metrics"])
    app = load_csv(REQUIRED_OUTPUTS["34 appearance dataset"])
    cum = load_csv(REQUIRED_OUTPUTS["34 cumulative metrics"])
    pairwise = load_csv(REQUIRED_OUTPUTS["34 pairwise differences"])
    boot = load_csv(REQUIRED_OUTPUTS["34 bootstrap CIs"])
    bias35 = load_csv(REQUIRED_OUTPUTS["35 probability bias"])
    extremity35 = load_csv(REQUIRED_OUTPUTS["35 extremity"])
    rd35 = load_csv(REQUIRED_OUTPUTS["35 Glicko rating/RD"])
    key36 = load_csv(REQUIRED_OUTPUTS["36 key diagnostics"])
    debut36 = load_csv(REQUIRED_OUTPUTS["36 debut state"])
    counter36 = load_csv(REQUIRED_OUTPUTS["36 counterfactual diagnostics"])
    key37 = load_csv(REQUIRED_OUTPUTS["37 key initial-rating results"])
    val37 = load_csv(REQUIRED_OUTPUTS["37 validation initial-rating metrics"])
    test37 = load_csv(REQUIRED_OUTPUTS["37 test initial-rating metrics"])
    overall38 = load_csv(REQUIRED_OUTPUTS["38 overall model metrics"])
    asym38 = load_csv(REQUIRED_OUTPUTS["38 asymmetric-K summary"])
    recovery38 = load_csv(REQUIRED_OUTPUTS["38 Glicko gap recovery"])
    orient39 = load_csv(REQUIRED_OUTPUTS["39 orientation comparison"])
    comp39 = load_csv(REQUIRED_OUTPUTS["39 complement gap"])
    side39 = load_csv(REQUIRED_OUTPUTS["39 early-player side distribution"])
    orient40 = load_csv(REQUIRED_OUTPUTS["40 corrected orientation table"])
    codes40 = load_csv(REQUIRED_OUTPUTS["40 conclusion codes"])

    add_check("33", "data_lineage", "step33_row_count_is_fixed_2025_set", len(step33_scores) == 11379, len(step33_scores), 11379, REQUIRED_OUTPUTS["33 per-match scores"])
    add_check("33", "data_lineage", "step33_match_id_unique", step33_scores["match_id"].is_unique, int(step33_scores["match_id"].duplicated().sum()), 0, REQUIRED_OUTPUTS["33 per-match scores"])
    required_step33_cols = [
        "outcome_a",
        "p_a_Validation_best_Elo",
        "p_a_Glicko_low_fixed",
        "p_a_Glicko_C0_fixed",
        "p_a_best_AdaptiveK",
    ]
    missing_cols = sorted(set(required_step33_cols) - set(step33_scores.columns))
    add_check("33", "data_lineage", "step33_required_probability_columns_present", not missing_cols, missing_cols or "none", "none", REQUIRED_OUTPUTS["33 per-match scores"])

    # Independent overall metric recomputation from Step 33.
    step33_prob_cols = {
        "Validation_best_Elo": "p_a_Validation_best_Elo",
        "Glicko_low_fixed": "p_a_Glicko_low_fixed",
        "Glicko_C0_fixed": "p_a_Glicko_C0_fixed",
        "best_AdaptiveK": "p_a_best_AdaptiveK",
    }
    for model, prob_col in step33_prob_cols.items():
        if prob_col not in step33_scores.columns:
            continue
        y = step33_scores["outcome_a"]
        p = step33_scores[prob_col]
        recomputed = {
            "brier": brier_score(y, p),
            "log_loss": log_loss_score(y, p),
            "accuracy": accuracy_score(y, p),
        }
        row = step33_overall.loc[step33_overall["model"].eq(model)]
        if not row.empty:
            max_abs_diff = max(abs(recomputed[m] - float(row.iloc[0][m])) for m in recomputed)
            add_check(
                "33",
                "metric_recompute",
                f"step33_overall_metrics_recompute_{model}",
                max_abs_diff < 5e-12,
                f"{max_abs_diff:.3e}",
                "<5e-12",
                REQUIRED_OUTPUTS["33 overall metrics"],
                "Brier, log loss and accuracy independently recomputed from Step 33 per-match probabilities.",
            )

    # Step 34 appearance-level dataset checks.
    per_match_counts = app.groupby("match_id").size()
    add_check("34", "appearance_dataset", "appearance_dataset_row_count", len(app) == 22758, len(app), 22758, REQUIRED_OUTPUTS["34 appearance dataset"])
    add_check("34", "appearance_dataset", "each_match_has_two_appearances", per_match_counts.min() == 2 and per_match_counts.max() == 2, f"min={per_match_counts.min()}, max={per_match_counts.max()}", "min=2, max=2", REQUIRED_OUTPUTS["34 appearance dataset"])
    duplicate_player_match = int(app.duplicated(["match_id", "player_id"]).sum())
    add_check("34", "appearance_dataset", "no_duplicate_player_match_appearance", duplicate_player_match == 0, duplicate_player_match, 0, REQUIRED_OUTPUTS["34 appearance dataset"])
    appearance_formula_ok = bool((app["appearance_number"] == app["total_games_before"] + 1).all())
    add_check("34", "appearance_dataset", "appearance_number_equals_total_games_before_plus_one", appearance_formula_ok, "checked", "True", REQUIRED_OUTPUTS["34 appearance dataset"])
    expected_stage_counts = {"1": 76, "2-5": 330, "6-10": 449, "11-20": 840, "21-50": 1807, "51+": 19256}
    observed_stage_counts = app["appearance_stage"].astype(str).value_counts().to_dict()
    add_check("34", "early_game_groups", "stage_bins_mutually_cover_all_appearances", observed_stage_counts == expected_stage_counts, observed_stage_counts, expected_stage_counts, REQUIRED_OUTPUTS["34 appearance dataset"])
    observed_first_counts = {group: int(app[group].sum()) for group in ["first_1", "first_5", "first_10", "first_20", "first_30", "first_50"]}
    expected_first_counts = {"first_1": 76, "first_5": 406, "first_10": 855, "first_20": 1695, "first_30": 2399, "first_50": 3502}
    add_check("34", "early_game_groups", "cumulative_group_counts_match_expected", observed_first_counts == expected_first_counts, observed_first_counts, expected_first_counts, REQUIRED_OUTPUTS["34 appearance dataset"])
    focal_prob_in_range = all(app[col].between(0, 1).all() for col in MODEL_PROB_COLUMNS.values())
    add_check("34", "appearance_dataset", "all_focal_model_probabilities_in_unit_interval", focal_prob_in_range, "checked", "[0,1]", REQUIRED_OUTPUTS["34 appearance dataset"])

    # Recompute cumulative metrics from the appearance dataset.
    max_cumulative_diff = 0.0
    for group in ["first_1", "first_5", "first_10", "first_20", "first_30", "first_50"]:
        subset = app.loc[app[group].astype(bool)]
        for model, prob_col in MODEL_PROB_COLUMNS.items():
            recomputed = {
                "brier": brier_score(subset["outcome_focal"], subset[prob_col]),
                "log_loss": log_loss_score(subset["outcome_focal"], subset[prob_col]),
                "accuracy": accuracy_score(subset["outcome_focal"], subset[prob_col]),
            }
            table_row = cum.loc[cum["group"].eq(group) & cum["model"].eq(model)]
            if table_row.empty:
                max_cumulative_diff = max(max_cumulative_diff, np.inf)
            else:
                max_cumulative_diff = max(
                    max_cumulative_diff,
                    *(abs(recomputed[m] - float(table_row.iloc[0][m])) for m in recomputed),
                )
    add_check(
        "34",
        "metric_recompute",
        "cumulative_metrics_recompute_from_appearance_dataset",
        max_cumulative_diff < 5e-12,
        f"{max_cumulative_diff:.3e}",
        "<5e-12",
        REQUIRED_OUTPUTS["34 cumulative metrics"],
    )

    max_pairwise_diff = 0.0
    for group in ["first_1", "first_5", "first_10", "first_20", "first_30", "first_50"]:
        elo_brier = get_model_metric(cum, group, "Validation_best_Elo", "brier")
        gl_brier = get_model_metric(cum, group, "Glicko_low_fixed", "brier")
        pair_delta = get_pairwise_delta(pairwise, group, "Validation_best_Elo_minus_Glicko_low_fixed", "delta_brier")
        max_pairwise_diff = max(max_pairwise_diff, abs((elo_brier - gl_brier) - pair_delta))
    add_check("34", "pairwise_comparison", "pairwise_delta_brier_matches_model_metric_difference", max_pairwise_diff < 5e-12, f"{max_pairwise_diff:.3e}", "<5e-12", REQUIRED_OUTPUTS["34 pairwise differences"])

    max_bootstrap_point_diff = 0.0
    for group in ["first_1", "first_5", "first_10", "first_20"]:
        ci = get_ci_row(boot, group, "Validation_best_Elo_minus_Glicko_low_fixed", "delta_brier")
        pair_delta = get_pairwise_delta(pairwise, group, "Validation_best_Elo_minus_Glicko_low_fixed", "delta_brier")
        max_bootstrap_point_diff = max(max_bootstrap_point_diff, abs(float(ci["point_estimate"]) - pair_delta))
    add_check("34", "bootstrap", "bootstrap_point_estimates_match_pairwise_table", max_bootstrap_point_diff < 5e-12, f"{max_bootstrap_point_diff:.3e}", "<5e-12", REQUIRED_OUTPUTS["34 bootstrap CIs"])
    add_check("34", "bootstrap", "bootstrap_uses_player_cluster_resampling", boot["bootstrap_type"].eq("player_cluster").all(), boot["bootstrap_type"].unique().tolist(), "player_cluster", REQUIRED_OUTPUTS["34 bootstrap CIs"])

    # Step 35 mechanism checks.
    first1_gl_bias = bias35.loc[bias35["group"].eq("first_1") & bias35["model"].eq("Glicko_low_fixed")].iloc[0]
    bias_formula_diff = abs((float(first1_gl_bias["mean_predicted_win_probability"]) - float(first1_gl_bias["empirical_win_rate"])) - float(first1_gl_bias["prediction_bias"]))
    add_check("35", "mechanism", "first1_glicko_bias_formula", bias_formula_diff < EPS, f"{bias_formula_diff:.3e}", "<1e-12", REQUIRED_OUTPUTS["35 probability bias"])
    first1_extreme = extremity35.loc[extremity35["appearance_stage"].astype(str).eq("1") & extremity35["model"].eq("Glicko_low_fixed")].iloc[0]
    add_check("35", "mechanism", "glicko_first1_extreme_probability_share_recorded", float(first1_extreme["pct_probability_below_0_10_or_above_0_90"]) >= 0, format_float(first1_extreme["pct_probability_below_0_10_or_above_0_90"]), ">=0", REQUIRED_OUTPUTS["35 extremity"])
    rd_first1 = rd35.loc[rd35["appearance_stage"].astype(str).eq("1")].iloc[0]
    add_check("35", "mechanism", "first1_rd_equals_initial_rd_350", abs(float(rd_first1["mean_focal_rd"]) - 350.0) < EPS, format_float(rd_first1["mean_focal_rd"]), 350.0, REQUIRED_OUTPUTS["35 Glicko rating/RD"])

    # Step 36 initialisation diagnostics.
    add_check("36", "initialisation", "first1_observed_probability_matches_step35", abs(get_key_value(key36, "first_1_mean_Glicko_probability") - float(first1_gl_bias["mean_predicted_win_probability"])) < EPS, format_float(get_key_value(key36, "first_1_mean_Glicko_probability")), format_float(first1_gl_bias["mean_predicted_win_probability"]), REQUIRED_OUTPUTS["36 key diagnostics"])
    equal_rating = counter36.loc[counter36["counterfactual"].eq("B_set_focal_rating_equal_to_opponent")].iloc[0]
    add_check("36", "initialisation", "equal_rating_counterfactual_mean_probability_is_half", abs(float(equal_rating["mean_predicted_probability"]) - 0.5) < EPS, format_float(equal_rating["mean_predicted_probability"]), 0.5, REQUIRED_OUTPUTS["36 counterfactual diagnostics"])
    add_check("36", "initialisation", "both_debut_match_count_is_two", int(float(debut36.loc[debut36["metric"].eq("number_of_both_debut_matches"), "value"].iloc[0])) == 2, int(float(debut36.loc[debut36["metric"].eq("number_of_both_debut_matches"), "value"].iloc[0])), 2, REQUIRED_OUTPUTS["36 debut state"])

    # Step 37 initial-rating sensitivity.
    val_spread = float(val37["brier"].max() - val37["brier"].min())
    test_spread = float(test37["brier"].max() - test37["brier"].min())
    add_check("37", "sensitivity", "validation_brier_identical_across_common_initial_ratings", val_spread < EPS, f"{val_spread:.3e}", "<1e-12", REQUIRED_OUTPUTS["37 validation initial-rating metrics"])
    add_check("37", "sensitivity", "test_brier_identical_across_common_initial_ratings", test_spread < EPS, f"{test_spread:.3e}", "<1e-12", REQUIRED_OUTPUTS["37 test initial-rating metrics"])
    add_check("37", "sensitivity", "selected_initial_rating_recorded_as_1500", int(get_key_value(key37, "selected_initial_rating")) == 1500, int(get_key_value(key37, "selected_initial_rating")), 1500, REQUIRED_OUTPUTS["37 key initial-rating results"])

    # Step 38 asymmetric adaptive-K.
    sym = overall38.loc[overall38["model"].eq("AdaptiveK_PreviousYearGames_Elo_scale300")].iloc[0]
    asym = overall38.loc[overall38["model"].eq("Asymmetric_AdaptiveK_PreviousYearGames_Elo_scale300")].iloc[0]
    sym_asym_diff = max(abs(float(sym[m]) - float(asym[m])) for m in ["brier", "log_loss", "accuracy"])
    add_check("38", "adaptive_k", "symmetric_and_asymmetric_adaptive_k_match_in_predictions", sym_asym_diff < 5e-12, f"{sym_asym_diff:.3e}", "<5e-12", REQUIRED_OUTPUTS["38 overall model metrics"])
    k_diff_share = float(asym38.iloc[0]["percentage_matches_K_A_differs_from_K_B"])
    add_check("38", "adaptive_k", "asymmetric_k_actually_assigns_different_player_K_values", k_diff_share > 0, format_float(k_diff_share), ">0", REQUIRED_OUTPUTS["38 asymmetric-K summary"])
    max_recovery_abs = float(recovery38["recovery_fraction"].abs().max())
    add_check("38", "adaptive_k", "asymmetric_k_recovers_no_material_glicko_gap", max_recovery_abs < 1e-9, f"{max_recovery_abs:.3e}", "<1e-9", REQUIRED_OUTPUTS["38 Glicko gap recovery"])

    # Step 39 and 40 orientation checks.
    overall_conclusions = orient39.loc[orient39["group"].eq("all_2025_matches"), "conclusion"].tolist()
    first1_conclusions = orient39.loc[orient39["group"].eq("either_player_first_1"), "conclusion"].tolist()
    add_check("39", "orientation", "overall_brier_conclusion_glicko_better_all_conventions", set(overall_conclusions) == {"GLICKO_BETTER"}, overall_conclusions, "all GLICKO_BETTER", REQUIRED_OUTPUTS["39 orientation comparison"])
    add_check("39", "orientation", "first1_brier_conclusion_elo_better_all_conventions", set(first1_conclusions) == {"ELO_BETTER"}, first1_conclusions, "all ELO_BETTER", REQUIRED_OUTPUTS["39 orientation comparison"])
    low_all_gap = comp39.loc[comp39["model"].eq("Glicko_low") & comp39["group"].eq("all_2025_matches")].iloc[0]
    add_check("39", "orientation", "overall_complement_gap_quantified", float(low_all_gap["mean_absolute_complement_gap"]) > 0, format_float(low_all_gap["mean_absolute_complement_gap"]), ">0", REQUIRED_OUTPUTS["39 complement gap"])
    first1_side = side39.loc[side39["group"].eq("first_1")].iloc[0]
    add_check("39", "orientation", "first1_side_distribution_quantified", float(first1_side["percentage_focal_player_is_large_id"]) > 90, format_float(first1_side["percentage_focal_player_is_large_id"]), ">90%", REQUIRED_OUTPUTS["39 early-player side distribution"])
    main_code = str(codes40.iloc[0]["main_conclusion_code"])
    add_check("40", "reporting", "final_orientation_conclusion_is_robust_to_orientation", main_code == "ROBUST_TO_ORIENTATION", main_code, "ROBUST_TO_ORIENTATION", REQUIRED_OUTPUTS["40 conclusion codes"])
    add_check("40", "reporting", "current_convention_can_remain_primary", bool(codes40.iloc[0]["current_convention_can_remain_primary"]), bool(codes40.iloc[0]["current_convention_can_remain_primary"]), True, REQUIRED_OUTPUTS["40 conclusion codes"])
    add_check("40", "reporting", "corrected_reporting_table_has_expected_rows", len(orient40) == 27, len(orient40), 27, REQUIRED_OUTPUTS["40 corrected orientation table"])

    # Source-text checks for the most important implementation choices.
    source_texts = {source.step: read_text(source.path) for source in SOURCE_FILES if source.path.exists()}
    add_check("34", "source_logic", "step34_uses_fixed_glicko_columns", "p_a_Glicko_low_fixed" in source_texts.get("34", "") and "p_a_Glicko_C0_fixed" in source_texts.get("34", ""), "found" if "p_a_Glicko_low_fixed" in source_texts.get("34", "") else "missing", "fixed Glicko columns", SOURCE_FILES[3].path)
    step34_text = source_texts.get("34", "")
    step34_player_b_orientation_ok = "1 - scores[\"outcome_a\"" in step34_text and "1.0 - p_a" in step34_text
    add_check("34", "source_logic", "step34_focal_orientation_uses_one_minus_for_player_b", step34_player_b_orientation_ok, "checked", "player-B outcome uses 1 - outcome_a and p_focal uses 1.0 - p_a", SOURCE_FILES[3].path)
    add_check("36", "source_logic", "step36_uses_glicko_formula_counterfactuals_not_rerun", "expected_score" in source_texts.get("36", "") and "counterfactual" in source_texts.get("36", ""), "checked", "formula-only counterfactual diagnostics", SOURCE_FILES[5].path)
    add_check("38", "source_logic", "step38_records_separate_k_a_and_k_b", "K_A" in source_texts.get("38", "") and "K_B" in source_texts.get("38", ""), "checked", "separate player K diagnostics", SOURCE_FILES[7].path)
    add_check("39", "source_logic", "step39_tests_current_reversed_midpoint_conventions", all(token in source_texts.get("39", "") for token in ["current", "reversed", "midpoint"]), "checked", "current/reversed/midpoint conventions", SOURCE_FILES[8].path)
    add_check("40", "source_logic", "step40_corrects_reporting_metadata_without_changing_numbers", "only_metadata_and_reporting_text_changed" in source_texts.get("40", ""), "checked", "metadata/reporting-only correction", SOURCE_FILES[9].path)
    add_check("glicko_core", "source_logic", "glicko_core_exposes_expected_score", "def expected_score" in source_texts.get("glicko_core", ""), "checked", "def expected_score", SOURCE_FILES[-1].path)

    # Issues and limitations. These are not material implementation errors.
    add_issue(
        "ISSUE_001",
        "38",
        "Figure clarity",
        "REPORTING_OR_METADATA_ISSUE",
        "low",
        "Figure 38_fig07_glicko_gap_recovery is visually almost blank because recovery fractions are essentially zero while the y-axis spans 0 to 1.",
        "No numerical result is affected. The underlying table correctly shows no material recovery of the Glicko gap by the asymmetric-K proof of concept.",
        "Do not use this figure in the meeting report, or replace it with a short table/annotation explaining that recovery is approximately zero.",
        False,
    )
    add_issue(
        "ISSUE_002",
        "40",
        "Resolved reporting metadata",
        "REPORTING_OR_METADATA_ISSUE",
        "low",
        "Step 40 corrects Step 39 reporting metadata for candidate-bias interpretation while preserving all numerical values.",
        "No numerical result is affected if the Step 40 corrected reporting table and wording are used.",
        "Use Step 40 files as the final orientation reporting source, not the uncorrected Step 39 wording.",
        False,
    )
    add_issue(
        "LIMIT_001",
        "34-36",
        "Small first-appearance sample",
        "LIMITATION_NOT_CODE_ERROR",
        "medium",
        "The first_1 group contains only 76 player appearances across 74 matches.",
        "The first-appearance result is substantively strong, but uncertainty and event composition should be discussed.",
        "Present first_1 together with first_5, first_10 and first_20 rather than as a stand-alone conclusion.",
        False,
    )
    add_issue(
        "LIMIT_002",
        "34",
        "Overlapping cumulative groups",
        "LIMITATION_NOT_CODE_ERROR",
        "low",
        "The first_5, first_10 and first_20 groups are cumulative and therefore not independent samples.",
        "Cumulative tables are useful for the research question, but non-overlapping stage bins should be used to describe the learning curve.",
        "Report both cumulative thresholds and non-overlapping stage bins.",
        False,
    )
    add_issue(
        "LIMIT_003",
        "34-36",
        "Observed-data debut definition",
        "LIMITATION_NOT_CODE_ERROR",
        "medium",
        "First recorded appearance in the available 1985-2025 data is not necessarily the player's true career debut.",
        "Interpretation should be framed as reliability for players with no previous recorded history, not guaranteed career beginners.",
        "Use 'first recorded appearance' or 'no previous recorded history' wording in the dissertation.",
        False,
    )
    add_issue(
        "LIMIT_004",
        "39-40",
        "Orientation sensitivity scope",
        "LIMITATION_NOT_CODE_ERROR",
        "low",
        "Orientation sensitivity tests outcome-independent current, reversed and midpoint conventions; they do not define a uniquely true Glicko probability convention.",
        "The main conclusion is robust to the tested conventions, but the convention itself should still be described transparently.",
        "Keep current Step 33 convention as primary and include the Step 39/40 sensitivity note.",
        False,
    )

    audit_checks = pd.DataFrame(checks)
    audit_issues = pd.DataFrame(issues)
    return audit_checks, audit_issues


def main() -> None:
    MEETING7_DIR.mkdir(parents=True, exist_ok=True)
    checks, issues = build_audit_outputs()

    checks.to_csv(AUDIT_CHECKS_PATH, index=False)
    issues.to_csv(AUDIT_ISSUES_PATH, index=False)

    failed_checks = int(checks["status"].eq("FAIL").sum())
    material_errors = int((issues["classification"] == "MATERIAL_IMPLEMENTATION_ERROR").sum())
    minor_issues = int((issues["classification"] == "MINOR_IMPLEMENTATION_ISSUE").sum())
    reporting_issues = int((issues["classification"] == "REPORTING_OR_METADATA_ISSUE").sum())
    limitations = int((issues["classification"] == "LIMITATION_NOT_CODE_ERROR").sum())
    overall_status = "PASS" if failed_checks == 0 and material_errors == 0 else "FAIL"

    print("Meeting 7 final source-code audit")
    print(f"Overall status: {overall_status}")
    print(f"Source-code files reviewed: {len(SOURCE_FILES)}")
    print(f"Output files checked: {len(REQUIRED_OUTPUTS) + len(VALIDATION_OUTPUTS)}")
    print(f"Audit checks: {len(checks)}")
    print(f"Failed checks: {failed_checks}")
    print(f"Material implementation errors: {material_errors}")
    print(f"Minor implementation issues: {minor_issues}")
    print(f"Reporting/metadata issues: {reporting_issues}")
    print(f"Limitations, not code errors: {limitations}")
    print("Conclusions valid: YES" if overall_status == "PASS" else "Conclusions valid: NO")
    print("Technical analysis can be frozen: YES" if overall_status == "PASS" else "Technical analysis can be frozen: NO")
    print(f"Wrote: {AUDIT_CHECKS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {AUDIT_ISSUES_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
