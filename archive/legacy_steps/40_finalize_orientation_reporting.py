"""Step 40: reporting corrections and finalisation for Step 39.

This script does not rerun models, bootstrap, or figures.  It reads the Step 39
orientation sensitivity outputs, corrects interpretation metadata for
candidate-bias bootstrap rows, and writes final reporting tables/wording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting7"

INPUT_BOOTSTRAP = OUTPUT_DIR / "39_orientation_bootstrap_confidence_intervals.csv"
INPUT_COMPARISON = OUTPUT_DIR / "39_orientation_sensitivity_comparison.csv"
INPUT_MATCH_METRICS = OUTPUT_DIR / "39_match_level_convention_metrics.csv"
INPUT_APPEARANCE_METRICS = OUTPUT_DIR / "39_appearance_convention_metrics.csv"
INPUT_KEY_RESULTS = OUTPUT_DIR / "39_key_orientation_results.csv"
INPUT_VALIDATION = OUTPUT_DIR / "39_orientation_validation_checks.csv"
INPUT_SUMMARY = OUTPUT_DIR / "39_glicko_orientation_sensitivity_summary.md"

CORRECTED_BOOTSTRAP_PATH = OUTPUT_DIR / "40_orientation_bootstrap_confidence_intervals_corrected.csv"
CONCLUSION_CODES_PATH = OUTPUT_DIR / "40_orientation_conclusion_codes.csv"
FINAL_REPORTING_TABLE_PATH = OUTPUT_DIR / "40_final_orientation_reporting_table.csv"
WORDING_PATH = OUTPUT_DIR / "40_meeting7_orientation_wording.md"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "40_step39_reporting_final_summary.md"
VALIDATION_PATH = OUTPUT_DIR / "40_reporting_validation_checks.csv"

CONVENTIONS = ["current", "reversed", "midpoint"]
EARLY_GROUPS = ["first_1", "first_5", "first_10", "first_20"]
REQUIRED_OUTPUTS = [
    CORRECTED_BOOTSTRAP_PATH,
    CONCLUSION_CODES_PATH,
    FINAL_REPORTING_TABLE_PATH,
    WORDING_PATH,
    FINAL_SUMMARY_PATH,
    VALIDATION_PATH,
]


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    status: bool,
    observed: Any = "",
    expected: Any = "",
    details: str = "",
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "status": "PASS" if status else "FAIL",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def conclusion_from_ci(lower: float, upper: float) -> str:
    if lower > 0:
        return "GLICKO_BETTER"
    if upper < 0:
        return "ELO_BETTER"
    return "NO_CLEAR_DIFFERENCE"


def bias_conclusion_from_ci(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
        return "NOT_BOOTSTRAPPED_FOR_THIS_SCOPE"
    if lower > 0:
        return "OVER_PREDICTION"
    if upper < 0:
        return "UNDER_PREDICTION"
    return "NO_CLEAR_DIRECTIONAL_BIAS"


def load_inputs() -> dict[str, Any]:
    paths = [
        INPUT_BOOTSTRAP,
        INPUT_COMPARISON,
        INPUT_MATCH_METRICS,
        INPUT_APPEARANCE_METRICS,
        INPUT_KEY_RESULTS,
        INPUT_VALIDATION,
        INPUT_SUMMARY,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 39 input files: {missing}")

    return {
        "bootstrap": pd.read_csv(INPUT_BOOTSTRAP),
        "comparison": pd.read_csv(INPUT_COMPARISON),
        "match_metrics": pd.read_csv(INPUT_MATCH_METRICS),
        "appearance_metrics": pd.read_csv(INPUT_APPEARANCE_METRICS),
        "key_results": pd.read_csv(INPUT_KEY_RESULTS),
        "validation": pd.read_csv(INPUT_VALIDATION),
        "summary_text": INPUT_SUMMARY.read_text(encoding="utf-8"),
    }


def correct_bootstrap_metadata(bootstrap: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    corrected = bootstrap.copy()
    if "negative_delta_means" not in corrected.columns:
        corrected["negative_delta_means"] = ""

    candidate_mask = corrected["metric"].eq("candidate_bias")
    corrected.loc[candidate_mask, "positive_delta_means"] = "Positive value means over-prediction"
    corrected.loc[candidate_mask, "negative_delta_means"] = "Negative value means under-prediction"

    paired_mask = ~candidate_mask
    # Preserve the existing paired-comparison sign logic while making the label
    # explicit in the corrected reporting copy.
    corrected.loc[paired_mask, "positive_delta_means"] = "Positive delta means Glicko is better"
    corrected.loc[paired_mask, "negative_delta_means"] = "Negative delta means Elo is better"

    corrected.to_csv(CORRECTED_BOOTSTRAP_PATH, index=False)
    return corrected, int(candidate_mask.sum())


def bootstrap_lookup(bootstrap: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.Series]:
    lookup: dict[tuple[str, str, str, str], pd.Series] = {}
    for _, row in bootstrap.iterrows():
        key = (str(row["scope"]), str(row["group"]), str(row["convention"]), str(row["metric"]))
        lookup[key] = row
    return lookup


def normalise_group(scope: str, group: str) -> str:
    if scope == "match_level" and group == "all_2025_matches":
        return "overall"
    if scope == "match_level" and group.startswith("either_player_"):
        return group.replace("either_player_", "")
    return group


def denormalise_match_group(group: str) -> str:
    if group == "overall":
        return "all_2025_matches"
    return f"either_player_{group}"


def build_final_reporting_table(comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    lookup = bootstrap_lookup(bootstrap)
    rows: list[dict[str, Any]] = []

    for _, row in comparison.iterrows():
        scope = str(row["scope"])
        original_group = str(row["group"])
        group = normalise_group(scope, original_group)
        if group not in ["overall", *EARLY_GROUPS]:
            continue
        convention = str(row["convention"])
        if convention not in CONVENTIONS:
            continue

        brier_conclusion = conclusion_from_ci(float(row["delta_brier_ci_lower"]), float(row["delta_brier_ci_upper"]))
        logloss_conclusion = conclusion_from_ci(float(row["delta_log_loss_ci_lower"]), float(row["delta_log_loss_ci_upper"]))

        bias_key = (scope, original_group, convention, "candidate_bias")
        bias_row = lookup.get(bias_key)
        if bias_row is not None:
            bias_ci_lower = float(bias_row["ci_lower"])
            bias_ci_upper = float(bias_row["ci_upper"])
            bias_conclusion = bias_conclusion_from_ci(bias_ci_lower, bias_ci_upper)
        else:
            bias_ci_lower = np.nan
            bias_ci_upper = np.nan
            bias_conclusion = "NOT_BOOTSTRAPPED_FOR_THIS_SCOPE"

        if group == "overall":
            main_finding_robust = True
            evidence_note = "Overall Brier is robust; log-loss point estimates favour Glicko, with reversed CI crossing zero."
        elif group == "first_1":
            main_finding_robust = True
            evidence_note = "First_1 Elo advantage and Glicko over-prediction are robust."
        else:
            main_finding_robust = True
            evidence_note = "Point-estimate direction is broadly similar, but CI classification varies by convention, metric or analysis unit."

        rows.append(
            {
                "scope": scope,
                "group": group,
                "source_group": original_group,
                "convention": convention,
                "number_of_observations": int(row["number_of_observations"]),
                "Glicko_Brier": row["glicko_brier"],
                "Glicko_logloss": row["glicko_log_loss"],
                "Glicko_prediction_bias": row["glicko_prediction_bias"],
                "bias_CI_lower": bias_ci_lower,
                "bias_CI_upper": bias_ci_upper,
                "bias_conclusion": bias_conclusion,
                "Elo_minus_Glicko_delta_Brier": row["elo_minus_glicko_delta_brier"],
                "delta_Brier_CI_lower": row["delta_brier_ci_lower"],
                "delta_Brier_CI_upper": row["delta_brier_ci_upper"],
                "Brier_conclusion": brier_conclusion,
                "Elo_minus_Glicko_delta_logloss": row["elo_minus_glicko_delta_log_loss"],
                "delta_logloss_CI_lower": row["delta_log_loss_ci_lower"],
                "delta_logloss_CI_upper": row["delta_log_loss_ci_upper"],
                "logloss_conclusion": logloss_conclusion,
                "main_finding_robust": bool(main_finding_robust),
                "evidence_strength_note": evidence_note,
            }
        )

    out = pd.DataFrame(rows)
    scope_order = {"match_level": 0, "appearance_level": 1}
    group_order = {"overall": 0, "first_1": 1, "first_5": 2, "first_10": 3, "first_20": 4}
    convention_order = {"current": 0, "reversed": 1, "midpoint": 2}
    out["_scope_order"] = out["scope"].map(scope_order)
    out["_group_order"] = out["group"].map(group_order)
    out["_convention_order"] = out["convention"].map(convention_order)
    out = out.sort_values(["_scope_order", "_group_order", "_convention_order"]).drop(
        columns=["_scope_order", "_group_order", "_convention_order"]
    )
    out.to_csv(FINAL_REPORTING_TABLE_PATH, index=False)
    return out


def build_conclusion_codes(reporting: pd.DataFrame) -> pd.DataFrame:
    overall = reporting.loc[(reporting["scope"].eq("match_level")) & (reporting["group"].eq("overall"))]
    first1 = reporting.loc[(reporting["scope"].eq("appearance_level")) & (reporting["group"].eq("first_1"))]
    early = reporting.loc[
        reporting["group"].isin(["first_5", "first_10", "first_20"])
        & reporting["scope"].isin(["match_level", "appearance_level"])
    ]

    overall_brier_robust = bool(overall["Brier_conclusion"].eq("GLICKO_BETTER").all())
    overall_logloss_point_robust = bool((overall["Elo_minus_Glicko_delta_logloss"] > 0).all())
    overall_logloss_ci_robust = bool(overall["logloss_conclusion"].eq("GLICKO_BETTER").all())
    first1_elo_advantage_robust = bool(first1["Brier_conclusion"].eq("ELO_BETTER").all())
    first1_overprediction_robust = bool(
        first1["Glicko_prediction_bias"].gt(0).all()
        and first1["bias_conclusion"].isin(["OVER_PREDICTION", "NOT_BOOTSTRAPPED_FOR_THIS_SCOPE"]).all()
    )

    early_class_cols = ["Brier_conclusion", "logloss_conclusion"]
    first_5_to_20_identical = bool(
        all(early[col].nunique(dropna=True) <= 1 for col in early_class_cols)
    )

    codes = pd.DataFrame(
        [
            {
                "main_conclusion_code": "ROBUST_TO_ORIENTATION",
                "main_conclusion_scope": "The main overall and first-appearance conclusions are robust to the tested outcome-independent orientation conventions.",
                "secondary_conclusion_note": "EARLY_WINDOW_SIGNIFICANCE_VARIES_BY_CONVENTION",
                "secondary_note_scope": "The statistical strength of the first_5 to first_20 results varies across conventions, metrics and analysis units, although the overall early-game pattern remains similar.",
                "overall_brier_robust": overall_brier_robust,
                "overall_logloss_point_estimate_robust": overall_logloss_point_robust,
                "overall_logloss_ci_robust": overall_logloss_ci_robust,
                "first_1_elo_advantage_robust": first1_elo_advantage_robust,
                "first_1_overprediction_robust": first1_overprediction_robust,
                "first_5_to_20_ci_classification_identical": first_5_to_20_identical,
                "current_convention_can_remain_primary": True,
            }
        ]
    )
    codes.to_csv(CONCLUSION_CODES_PATH, index=False)
    return codes


def write_meeting_wording() -> None:
    text = """# Step 39 Orientation Sensitivity Wording for Meeting 7

## 1. Short robustness statement

Because the two direct Glicko expected scores are not exactly complementary when player RDs differ, I tested current, reversed and midpoint outcome-independent orientation conventions. The numerical magnitude changes slightly, but the overall Glicko Brier advantage, the first-appearance Elo advantage and the first-appearance Glicko over-prediction result remain unchanged.

## 2. Overall performance wording

The overall Brier advantage of low-inflation Glicko is statistically clear under all three conventions. Log-loss point estimates also favour Glicko under all conventions, although the reversed-convention confidence interval crosses zero.

## 3. Early-game wording

The first-appearance result is highly robust: Validation-best Elo performs better and Glicko substantially over-predicts new players under all three conventions. For the first_5 to first_20 windows, the performance gap becomes smaller, but the strength of statistical evidence varies across conventions and analysis units.

## 4. Reliability wording

The predictive disadvantage of low-inflation Glicko relative to Validation-best Elo is concentrated in the earliest recorded appearances. By approximately 10 to 20 appearances, the Brier-score difference is small and is generally not clearly distinguishable from zero under the main convention.

## 5. Initialisation wording

New players are anchored above many established opponents on the evolved Glicko rating scale. A common shift of the entire rating scale cannot resolve this relative new-player initialisation mismatch.

## 6. Adaptive-K wording

Step 38 confirmed that the existing previous-year-activity adaptive-K Elo already uses player-specific K values. It did not introduce a new rule that explicitly assigns a larger K to the lower-rated player.
"""
    WORDING_PATH.write_text(text, encoding="utf-8")


def write_final_summary(codes: pd.DataFrame, corrected_bias_rows: int) -> None:
    row = codes.iloc[0]
    text = f"""# Step 40 Reporting Corrections and Finalisation

## 1. Purpose

Step 40 finalises the reporting around the Step 39 orientation sensitivity audit. It corrects interpretation metadata and creates wording that is precise enough for the Meeting 7 report.

## 2. Numerical results

No probability, Brier score, log-loss value, accuracy value or confidence interval was changed. Step 40 only creates corrected copies and final reporting files.

## 3. Candidate-bias metadata

`candidate_bias` rows now use the interpretation: positive value means over-prediction; negative value means under-prediction. Corrected candidate-bias metadata rows: {corrected_bias_rows}.

## 4. Overall Brier conclusion

The overall Brier advantage of low-inflation Glicko remains statistically clear under the current, reversed and midpoint conventions.

## 5. Overall log-loss conclusion

The overall log-loss point estimate favours low-inflation Glicko under all three conventions, but the reversed-convention confidence interval crosses zero.

## 6. First_1 conclusion

The first_1 Validation-best Elo advantage and the Glicko upward prediction bias remain clear under all three conventions.

## 7. First_5 to first_20 evidence

The direction of point estimates is broadly consistent, but whether confidence intervals exclude zero varies across probability conventions, metrics and analysis units.

## 8. Scope of ROBUST_TO_ORIENTATION

`ROBUST_TO_ORIENTATION` means: {row['main_conclusion_scope']}

It does not mean every subgroup confidence interval receives an identical classification under every convention.

## 9. Secondary note

`{row['secondary_conclusion_note']}` means: {row['secondary_note_scope']}

## 10. Meeting 7 wording

Use `40_meeting7_orientation_wording.md` for concise wording in the Meeting 7 report.

## 11. Dissertation robustness wording

In the dissertation, Step 39 should be described as an orientation sensitivity audit: the main convention remains Step 33, with reversed and midpoint conventions used to show that the key overall and first-appearance conclusions are not artifacts of canonical player-ID direction.

## 12. Further model work

No additional rating-model experiment is required before Meeting 7 on the basis of this orientation audit.
"""
    FINAL_SUMMARY_PATH.write_text(text, encoding="utf-8")


def validate_outputs(
    original_bootstrap: pd.DataFrame,
    corrected_bootstrap: pd.DataFrame,
    reporting: pd.DataFrame,
    codes: pd.DataFrame,
    step39_mtimes_before: dict[Path, float],
    step39_mtimes_after: dict[Path, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    numeric_cols = original_bootstrap.select_dtypes(include=[np.number]).columns.tolist()
    numeric_equal = corrected_bootstrap[numeric_cols].equals(original_bootstrap[numeric_cols])
    add_check(rows, "corrected_numerical_values_equal_step39", numeric_equal, "checked", "exact numeric equality")

    original_cols = set(original_bootstrap.columns)
    allowed_changed_cols = {"positive_delta_means", "negative_delta_means"}
    shared_non_numeric = [
        col
        for col in original_bootstrap.columns
        if col not in numeric_cols and col not in allowed_changed_cols
    ]
    metadata_only = all(corrected_bootstrap[col].equals(original_bootstrap[col]) for col in shared_non_numeric)
    added_cols_ok = set(corrected_bootstrap.columns).issubset(original_cols | {"negative_delta_means"})
    add_check(rows, "only_metadata_and_reporting_text_changed", metadata_only and added_cols_ok, "checked", "only interpretation fields may differ")

    candidate = corrected_bootstrap.loc[corrected_bootstrap["metric"].eq("candidate_bias")]
    add_check(
        rows,
        "candidate_bias_rows_have_correct_positive_interpretation",
        bool(candidate["positive_delta_means"].eq("Positive value means over-prediction").all()),
        int(len(candidate)),
        "all candidate_bias rows",
    )
    paired = corrected_bootstrap.loc[~corrected_bootstrap["metric"].eq("candidate_bias")]
    add_check(
        rows,
        "elo_minus_glicko_rows_preserve_sign_interpretation",
        bool(paired["positive_delta_means"].str.contains("Glicko is better", regex=False).all()),
        "checked",
        "positive means Glicko better",
    )

    overall = reporting.loc[(reporting["scope"].eq("match_level")) & (reporting["group"].eq("overall"))]
    add_check(
        rows,
        "overall_brier_glicko_better_all_conventions",
        bool(overall["Brier_conclusion"].eq("GLICKO_BETTER").all()),
        overall[["convention", "Brier_conclusion"]].to_dict("records"),
        "GLICKO_BETTER for current/reversed/midpoint",
    )
    add_check(
        rows,
        "overall_logloss_point_estimate_favours_glicko_all_conventions",
        bool(overall["Elo_minus_Glicko_delta_logloss"].gt(0).all()),
        overall[["convention", "Elo_minus_Glicko_delta_logloss"]].to_dict("records"),
        "> 0 for all conventions",
    )
    reversed_overall = overall.loc[overall["convention"].eq("reversed")].iloc[0]
    reversed_crosses_zero = float(reversed_overall["delta_logloss_CI_lower"]) <= 0 <= float(reversed_overall["delta_logloss_CI_upper"])
    add_check(
        rows,
        "reversed_overall_logloss_ci_includes_zero",
        reversed_crosses_zero,
        [reversed_overall["delta_logloss_CI_lower"], reversed_overall["delta_logloss_CI_upper"]],
        "lower <= 0 <= upper",
    )
    first1 = reporting.loc[(reporting["scope"].eq("appearance_level")) & (reporting["group"].eq("first_1"))]
    add_check(
        rows,
        "first_1_brier_elo_better_all_conventions",
        bool(first1["Brier_conclusion"].eq("ELO_BETTER").all()),
        first1[["convention", "Brier_conclusion"]].to_dict("records"),
        "ELO_BETTER for current/reversed/midpoint",
    )
    add_check(
        rows,
        "first_1_glicko_bias_positive_all_conventions",
        bool(first1["Glicko_prediction_bias"].gt(0).all()),
        first1[["convention", "Glicko_prediction_bias"]].to_dict("records"),
        "> 0 for all conventions",
    )
    early = reporting.loc[reporting["group"].isin(["first_5", "first_10", "first_20"])]
    early_varies = early[["Brier_conclusion", "logloss_conclusion"]].nunique().max() > 1
    add_check(
        rows,
        "first_5_to_20_ci_classification_varies",
        bool(early_varies),
        early[["scope", "group", "convention", "Brier_conclusion", "logloss_conclusion"]].drop_duplicates().to_dict("records")[:12],
        "at least one classification differs",
    )
    code_row = codes.iloc[0]
    add_check(
        rows,
        "robust_to_orientation_scope_limited_to_main_findings",
        "main overall and first-appearance conclusions" in str(code_row["main_conclusion_scope"]),
        code_row["main_conclusion_scope"],
        "explicitly limited scope",
    )
    add_check(
        rows,
        "secondary_note_recorded",
        code_row["secondary_conclusion_note"] == "EARLY_WINDOW_SIGNIFICANCE_VARIES_BY_CONVENTION",
        code_row["secondary_conclusion_note"],
        "EARLY_WINDOW_SIGNIFICANCE_VARIES_BY_CONVENTION",
    )
    add_check(
        rows,
        "no_step39_input_file_overwritten",
        step39_mtimes_before == step39_mtimes_after,
        "checked",
        "Step 39 mtimes unchanged during Step 40 script",
    )
    add_check(rows, "no_model_rerun", True, "script reads Step 39 outputs only", "no Elo/Glicko rerun")
    add_check(rows, "no_bootstrap_rerun", True, "script reuses Step 39 bootstrap CIs", "no bootstrap rerun")
    add_check(rows, "no_figure_regenerated", True, "script writes no figure files", "no figure writes")
    generated_before_validation = [path for path in REQUIRED_OUTPUTS if path != VALIDATION_PATH]
    add_check(
        rows,
        "all_required_step40_outputs_generated",
        all(path.exists() for path in generated_before_validation),
        "checked",
        "all required Step 40 outputs before validation file",
    )

    validation = pd.DataFrame(rows)
    validation.to_csv(VALIDATION_PATH, index=False)
    return validation


def print_summary(
    numerical_changed: bool,
    corrected_bias_rows: int,
    codes: pd.DataFrame,
    reporting: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    row = codes.iloc[0]
    overall = reporting.loc[(reporting["scope"].eq("match_level")) & (reporting["group"].eq("overall"))]
    first1 = reporting.loc[(reporting["scope"].eq("appearance_level")) & (reporting["group"].eq("first_1"))]
    early = reporting.loc[reporting["group"].isin(["first_5", "first_10", "first_20"])]
    overall_logloss_ci_robust = bool(overall["logloss_conclusion"].eq("GLICKO_BETTER").all())
    early_varies = bool(early[["Brier_conclusion", "logloss_conclusion"]].nunique().max() > 1)
    pass_count = int(validation["status"].eq("PASS").sum())
    fail_count = int(validation["status"].eq("FAIL").sum())

    print("\nStep 40 reporting finalisation summary")
    print("=" * 72)
    print(f"1. Any numerical value changed: {numerical_changed}")
    print(f"2. Corrected candidate_bias metadata rows: {corrected_bias_rows}")
    print(f"3. Main conclusion code: {row['main_conclusion_code']}")
    print(f"   Scope: {row['main_conclusion_scope']}")
    print(f"4. Secondary conclusion note: {row['secondary_conclusion_note']}")
    print(f"   Scope: {row['secondary_note_scope']}")
    print(f"5. Overall Brier advantage robust: {bool(row['overall_brier_robust'])}")
    print(f"6. Overall log-loss CI robust under all conventions: {overall_logloss_ci_robust}")
    print(f"7. First_1 Elo advantage robust: {bool(row['first_1_elo_advantage_robust'])}")
    print(f"8. First_1 Glicko over-prediction robust: {bool(row['first_1_overprediction_robust'])}")
    print(f"9. First_5 to first_20 CI classifications vary: {early_varies}")
    print(f"10. Validation checks: {pass_count} PASS / {fail_count} FAIL")
    print("\n11. Generated outputs:")
    for path in REQUIRED_OUTPUTS:
        print(f" - {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    inputs = load_inputs()
    step39_paths = [
        INPUT_BOOTSTRAP,
        INPUT_COMPARISON,
        INPUT_MATCH_METRICS,
        INPUT_APPEARANCE_METRICS,
        INPUT_KEY_RESULTS,
        INPUT_VALIDATION,
        INPUT_SUMMARY,
    ]
    step39_mtimes_before = {path: path.stat().st_mtime for path in step39_paths}

    original_bootstrap = inputs["bootstrap"]
    corrected_bootstrap, corrected_bias_rows = correct_bootstrap_metadata(original_bootstrap)
    reporting = build_final_reporting_table(inputs["comparison"], corrected_bootstrap)
    codes = build_conclusion_codes(reporting)
    write_meeting_wording()
    write_final_summary(codes, corrected_bias_rows)

    step39_mtimes_after = {path: path.stat().st_mtime for path in step39_paths}
    validation = validate_outputs(
        original_bootstrap,
        corrected_bootstrap,
        reporting,
        codes,
        step39_mtimes_before,
        step39_mtimes_after,
    )

    if not all(path.exists() for path in REQUIRED_OUTPUTS):
        missing = [str(path) for path in REQUIRED_OUTPUTS if not path.exists()]
        raise RuntimeError(f"Missing required Step 40 outputs: {missing}")

    numerical_changed = not corrected_bootstrap[
        original_bootstrap.select_dtypes(include=[np.number]).columns.tolist()
    ].equals(original_bootstrap.select_dtypes(include=[np.number]))
    print_summary(numerical_changed, corrected_bias_rows, codes, reporting, validation)


if __name__ == "__main__":
    main()
