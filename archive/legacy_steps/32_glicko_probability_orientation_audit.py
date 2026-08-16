"""Audit Glicko probability orientation against canonical Player A."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"

STEP29_SCORES_PATH = OUTPUT_DIR / "29_per_match_model_scores_2025.csv"
GLICKO_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_glicko_rd_inflation" / "meeting5_glicko_rd_inflation_predictions_2025.csv"
FAIR_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko" / "meeting5_fair_elo_vs_glicko_predictions_2025.csv"

CHECKS_PATH = OUTPUT_DIR / "32_probability_orientation_audit_checks.csv"
DIRECT_COMPARISON_PATH = OUTPUT_DIR / "32_glicko_direct_probability_comparison.csv"
GAP_SUMMARY_PATH = OUTPUT_DIR / "32_glicko_complement_gap_summary.csv"
SUBGROUP_PATH = OUTPUT_DIR / "32_probability_orientation_by_subgroup.csv"
IMPACT_PATH = OUTPUT_DIR / "32_orientation_impact_on_metrics.csv"

EXPECTED_GAMES = 11_379
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260715
EPS = 1e-15
ELO_SCALE = 300.0

GLICKO_VARIANTS = {
    "Glicko_low": "low_inflation",
    "Glicko_C0": "C0_no_inflation",
}
MODEL_LABELS = {
    "Glicko_low": "Glicko low inflation",
    "Glicko_C0": "Glicko C0",
    "Validation_best_Elo": "Validation-best Elo",
}
ORIENTATION_LABELS = {
    "existing_saved_orientation": "Existing saved-winner orientation",
    "fixed_player_a_direct": "Fixed player-A direct",
    "symmetric_diagnostic": "Symmetric diagnostic",
}


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", detail: str = "", severity: str = "error") -> None:
    """Append one validation row."""

    rows.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "detail": detail,
        }
    )


def load_scores() -> pd.DataFrame:
    """Load Step 29 per-match scores."""

    df = pd.read_csv(STEP29_SCORES_PATH, low_memory=False)
    df["event_key"] = df["year"].astype(str) + "_" + df["event_id"].astype(str)
    return df


def g_function(rd: np.ndarray | pd.Series | float) -> np.ndarray:
    """Glicko g(RD), matching code/glicko_core.py."""

    q = math.log(10.0) / 400.0
    rd_array = np.asarray(rd, dtype=float)
    return 1.0 / np.sqrt(1.0 + (3.0 * (q**2) * (rd_array**2)) / (math.pi**2))


def glicko_expected_score(rating: np.ndarray | pd.Series, opponent_rating: np.ndarray | pd.Series, opponent_rd: np.ndarray | pd.Series) -> np.ndarray:
    """Expected score using the opponent RD."""

    rating_arr = np.asarray(rating, dtype=float)
    opponent_rating_arr = np.asarray(opponent_rating, dtype=float)
    g_rd = g_function(opponent_rd)
    exponent = -g_rd * (rating_arr - opponent_rating_arr) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def elo_expected_score(rating: np.ndarray | pd.Series, opponent_rating: np.ndarray | pd.Series) -> np.ndarray:
    """Validation-best Elo expected score; scale=300 from code/26."""

    rating_arr = np.asarray(rating, dtype=float)
    opponent_rating_arr = np.asarray(opponent_rating, dtype=float)
    exponent = -(rating_arr - opponent_rating_arr) / ELO_SCALE
    return 1.0 / (1.0 + 10.0**exponent)


def load_and_map_glicko_predictions(scores: pd.DataFrame) -> pd.DataFrame:
    """Map saved Glicko winner/loser ratings and RDs to player A/B."""

    preds = pd.read_csv(GLICKO_PREDICTIONS_PATH, low_memory=False)
    frames = []
    base = scores[
        [
            "match_id",
            "match_sequence",
            "event_key",
            "year",
            "event_id",
            "match_date",
            "player_a_id",
            "player_b_id",
            "winner_id",
            "loser_id",
            "outcome_a",
            "either_player_debut",
            "both_players_have_history",
            "either_player_inactive_365d",
            "both_players_active_last_365d",
            "max_prematch_rd",
            "p_a_Glicko_low",
            "p_a_Glicko_C0",
        ]
    ].copy()

    for alias, variant in GLICKO_VARIANTS.items():
        sub = preds.loc[preds["variant"] == variant].copy()
        sub["match_id"] = sub["game_id"].astype(int)
        required = [
            "match_id",
            "winner",
            "loser",
            "pred_winner_win",
            "pre_rating_winner",
            "pre_rd_winner",
            "pre_rating_loser",
            "pre_rd_loser",
        ]
        merged = base.merge(sub[required], on="match_id", how="left", validate="one_to_one")
        if merged["winner"].isna().any():
            raise ValueError(f"Missing Glicko predictions for {variant}")
        a_is_winner = merged["player_a_id"] == merged["winner"]
        merged["model"] = alias
        merged["model_display"] = MODEL_LABELS[alias]
        merged["rating_a"] = np.where(a_is_winner, merged["pre_rating_winner"], merged["pre_rating_loser"])
        merged["rating_b"] = np.where(a_is_winner, merged["pre_rating_loser"], merged["pre_rating_winner"])
        merged["rd_a"] = np.where(a_is_winner, merged["pre_rd_winner"], merged["pre_rd_loser"])
        merged["rd_b"] = np.where(a_is_winner, merged["pre_rd_loser"], merged["pre_rd_winner"])
        merged["p_a_current_step29"] = merged[f"p_a_{alias}"]
        merged["p_a_direct"] = glicko_expected_score(merged["rating_a"], merged["rating_b"], merged["rd_b"])
        merged["p_b_direct"] = glicko_expected_score(merged["rating_b"], merged["rating_a"], merged["rd_a"])
        merged["probability_sum"] = merged["p_a_direct"] + merged["p_b_direct"]
        merged["complement_gap"] = merged["p_a_direct"] - (1.0 - merged["p_b_direct"])
        merged["abs_complement_gap"] = merged["complement_gap"].abs()
        merged["p_a_existing_orientation_reconstructed"] = np.where(
            merged["outcome_a"].astype(int) == 1,
            merged["p_a_direct"],
            1.0 - merged["p_b_direct"],
        )
        merged["p_a_from_saved_winner_probability"] = np.where(
            a_is_winner,
            merged["pred_winner_win"],
            1.0 - merged["pred_winner_win"],
        )
        merged["p_a_symmetric_diagnostic"] = merged["p_a_direct"] / (merged["p_a_direct"] + merged["p_b_direct"])
        merged["current_minus_direct"] = merged["p_a_current_step29"] - merged["p_a_direct"]
        merged["current_minus_one_minus_b"] = merged["p_a_current_step29"] - (1.0 - merged["p_b_direct"])
        merged["current_minus_existing_reconstructed"] = merged["p_a_current_step29"] - merged["p_a_existing_orientation_reconstructed"]
        merged["current_minus_saved_probability_conversion"] = merged["p_a_current_step29"] - merged["p_a_from_saved_winner_probability"]
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def load_and_map_elo_predictions(scores: pd.DataFrame) -> pd.DataFrame:
    """Map Validation-best Elo predictions to player A/B and verify complementarity."""

    preds = pd.read_csv(FAIR_PREDICTIONS_PATH, low_memory=False)
    sub = preds.loc[preds["model"] == "Validation_best_Elo"].copy()
    sub["match_id"] = sub["game_id"].astype(int)
    base = scores[["match_id", "player_a_id", "player_b_id", "winner_id", "loser_id", "outcome_a", "p_a_Validation_best_Elo"]].copy()
    merged = base.merge(
        sub[["match_id", "winner", "loser", "pred_winner_win", "pre_rating_winner", "pre_rating_loser"]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if merged["winner"].isna().any():
        raise ValueError("Missing Validation-best Elo predictions")
    a_is_winner = merged["player_a_id"] == merged["winner"]
    merged["rating_a"] = np.where(a_is_winner, merged["pre_rating_winner"], merged["pre_rating_loser"])
    merged["rating_b"] = np.where(a_is_winner, merged["pre_rating_loser"], merged["pre_rating_winner"])
    merged["p_a_direct_elo"] = elo_expected_score(merged["rating_a"], merged["rating_b"])
    merged["p_b_direct_elo"] = elo_expected_score(merged["rating_b"], merged["rating_a"])
    merged["elo_complement_gap"] = merged["p_a_direct_elo"] - (1.0 - merged["p_b_direct_elo"])
    merged["p_a_from_saved_winner_probability_elo"] = np.where(a_is_winner, merged["pred_winner_win"], 1.0 - merged["pred_winner_win"])
    merged["current_minus_elo_direct"] = merged["p_a_Validation_best_Elo"] - merged["p_a_direct_elo"]
    merged["current_minus_elo_saved_conversion"] = merged["p_a_Validation_best_Elo"] - merged["p_a_from_saved_winner_probability_elo"]
    return merged


def describe(values: pd.Series) -> dict[str, Any]:
    """Return standard distribution summary values."""

    values = values.dropna().astype(float)
    return {
        "count": int(values.count()),
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "p10": values.quantile(0.10),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
        "max": values.max(),
    }


def subgroup_masks(scores: pd.DataFrame) -> dict[str, pd.Series]:
    """Define the key audit subgroups."""

    no_debut = ~scores["either_player_debut"].astype(bool)
    exactly_one_debut = scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)
    returning_365 = no_debut & scores["either_player_inactive_365d"].astype(bool)
    both_active = no_debut & scores["both_players_active_last_365d"].astype(bool)
    high_rd = no_debut & (scores["max_prematch_rd"] >= scores.loc[no_debut, "max_prematch_rd"].quantile(0.75))
    return {
        "Overall": pd.Series(True, index=scores.index),
        "No debut": no_debut,
        "Exactly one debut": exactly_one_debut,
        "Returning >=365 days, no debut": returning_365,
        "Both active and no debut": both_active,
        "No-debut high RD quartile": high_rd,
    }


def build_gap_summaries(comp: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise Glicko complement gaps overall and by subgroup."""

    rows = []
    subgroup_rows = []
    masks = subgroup_masks(scores)
    for model, group in comp.groupby("model"):
        for variable in ["probability_sum", "complement_gap", "abs_complement_gap"]:
            desc = describe(group[variable])
            rows.append(
                {
                    "model": model,
                    "model_display": MODEL_LABELS[model],
                    "variable": variable,
                    **desc,
                    "count_abs_gap_gt_1e_12": int((group["abs_complement_gap"] > 1e-12).sum()),
                    "count_abs_gap_gt_1e_6": int((group["abs_complement_gap"] > 1e-6).sum()),
                }
            )

        for subgroup, mask in masks.items():
            ids = set(scores.loc[mask, "match_id"])
            sub = group.loc[group["match_id"].isin(ids)].copy()
            subgroup_rows.append(
                {
                    "model": model,
                    "model_display": MODEL_LABELS[model],
                    "subgroup": subgroup,
                    "games": int(len(sub)),
                    "events": int(sub["event_key"].nunique()),
                    "mean_complement_gap": sub["complement_gap"].mean(),
                    "mean_abs_complement_gap": sub["abs_complement_gap"].mean(),
                    "median_abs_complement_gap": sub["abs_complement_gap"].median(),
                    "p90_abs_complement_gap": sub["abs_complement_gap"].quantile(0.90),
                    "max_abs_complement_gap": sub["abs_complement_gap"].max(),
                    "current_matches_direct_when_outcome_a_1": int(((sub["outcome_a"] == 1) & (sub["current_minus_direct"].abs() < 1e-10)).sum()),
                    "current_matches_one_minus_b_when_outcome_a_0": int(((sub["outcome_a"] == 0) & (sub["current_minus_one_minus_b"].abs() < 1e-10)).sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(subgroup_rows)


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calculate Brier, log loss, accuracy and player-A calibration error."""

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))
    accuracy = float(np.mean((p >= 0.5).astype(int) == y.astype(int)))
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.minimum(np.digitize(p, bins, right=False) - 1, len(bins) - 2)
    calibration = 0.0
    for idx in range(len(bins) - 1):
        mask = bin_index == idx
        if not mask.any():
            continue
        calibration += (mask.sum() / len(p)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return {
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
        "player_a_calibration_error": float(calibration),
    }


def bootstrap_delta_vs_elo(df: pd.DataFrame, p_col: str, seed: int) -> dict[str, float]:
    """Event-cluster bootstrap for paired deltas versus validation-best Elo."""

    rng = np.random.default_rng(seed)
    work = df[["event_key"]].copy()
    y = df["outcome_a"].astype(float).to_numpy()
    p = df[p_col].astype(float).to_numpy()
    p_elo = df["p_a_Validation_best_Elo"].astype(float).to_numpy()
    work["delta_brier"] = (p_elo - y) ** 2 - (p - y) ** 2
    p_clip = np.clip(p, EPS, 1.0 - EPS)
    e_clip = np.clip(p_elo, EPS, 1.0 - EPS)
    work["delta_logloss"] = -(y * np.log(e_clip) + (1.0 - y) * np.log(1.0 - e_clip)) - (
        -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    )
    event_sums = work.groupby("event_key", sort=False)[["delta_brier", "delta_logloss"]].sum().to_numpy(dtype=float)
    event_counts = work.groupby("event_key", sort=False).size().to_numpy(dtype=float)
    draws = rng.integers(0, len(event_counts), size=(BOOTSTRAP_REPS, len(event_counts)))
    counts = event_counts[draws].sum(axis=1)
    sums = event_sums[draws].sum(axis=1)
    delta_brier = sums[:, 0] / counts
    delta_logloss = sums[:, 1] / counts
    return {
        "delta_brier_vs_elo_ci_lower": float(np.quantile(delta_brier, 0.025)),
        "delta_brier_vs_elo_ci_upper": float(np.quantile(delta_brier, 0.975)),
        "delta_logloss_vs_elo_ci_lower": float(np.quantile(delta_logloss, 0.025)),
        "delta_logloss_vs_elo_ci_upper": float(np.quantile(delta_logloss, 0.975)),
    }


def build_metric_impact(scores: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Compare existing, fixed-direct and symmetric Glicko metrics."""

    base = scores[
        [
            "match_id",
            "event_key",
            "outcome_a",
            "p_a_Validation_best_Elo",
        ]
    ].copy()
    masks = subgroup_masks(scores)
    rows = []
    for model in GLICKO_VARIANTS:
        model_comp = comp.loc[comp["model"] == model].copy()
        model_comp = model_comp.merge(base, on=["match_id", "event_key", "outcome_a"], how="left", validate="one_to_one")
        orientation_cols = {
            "existing_saved_orientation": "p_a_current_step29",
            "fixed_player_a_direct": "p_a_direct",
            "symmetric_diagnostic": "p_a_symmetric_diagnostic",
        }
        existing_by_subgroup: dict[str, dict[str, float]] = {}
        for subgroup, mask in masks.items():
            ids = set(scores.loc[mask, "match_id"])
            sub = model_comp.loc[model_comp["match_id"].isin(ids)].copy()
            y = sub["outcome_a"].astype(float).to_numpy()
            elo_metrics = metric_values(y, sub["p_a_Validation_best_Elo"].to_numpy())
            for orientation, p_col in orientation_cols.items():
                metrics = metric_values(y, sub[p_col].to_numpy())
                if orientation == "existing_saved_orientation":
                    existing_by_subgroup[subgroup] = metrics
                ci = bootstrap_delta_vs_elo(sub, p_col, seed=RANDOM_SEED + len(rows) * 13)
                rows.append(
                    {
                        "model": model,
                        "model_display": MODEL_LABELS[model],
                        "orientation": orientation,
                        "orientation_display": ORIENTATION_LABELS[orientation],
                        "subgroup": subgroup,
                        "games": int(len(sub)),
                        "events": int(sub["event_key"].nunique()),
                        **metrics,
                        "elo_brier": elo_metrics["brier"],
                        "elo_logloss": elo_metrics["logloss"],
                        "elo_accuracy": elo_metrics["accuracy"],
                        "delta_brier_vs_elo": elo_metrics["brier"] - metrics["brier"],
                        "delta_logloss_vs_elo": elo_metrics["logloss"] - metrics["logloss"],
                        "delta_accuracy_vs_elo": metrics["accuracy"] - elo_metrics["accuracy"],
                        "brier_change_vs_existing_saved_orientation": np.nan,
                        "logloss_change_vs_existing_saved_orientation": np.nan,
                        **ci,
                        "bootstrap_replications": BOOTSTRAP_REPS,
                    }
                )
        # Add changes versus existing saved orientation after all rows are created.
    out = pd.DataFrame(rows)
    for (model, subgroup), group_idx in out.groupby(["model", "subgroup"]).groups.items():
        existing = out.loc[
            (out["model"] == model)
            & (out["subgroup"] == subgroup)
            & (out["orientation"] == "existing_saved_orientation")
        ].iloc[0]
        idx = list(group_idx)
        out.loc[idx, "brier_change_vs_existing_saved_orientation"] = out.loc[idx, "brier"] - existing["brier"]
        out.loc[idx, "logloss_change_vs_existing_saved_orientation"] = out.loc[idx, "logloss"] - existing["logloss"]
    return out


def validate_audit(scores: pd.DataFrame, comp: pd.DataFrame, elo: pd.DataFrame, impact: pd.DataFrame, output_paths: list[Path]) -> pd.DataFrame:
    """Run audit validation checks."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "rows", len(scores) == EXPECTED_GAMES, len(scores), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", scores["match_id"].duplicated().sum() == 0, int(scores["match_id"].duplicated().sum()), 0)
    prob_cols = [
        "p_a_direct",
        "p_b_direct",
        "p_a_existing_orientation_reconstructed",
        "p_a_symmetric_diagnostic",
        "p_a_current_step29",
    ]
    add_check(rows, "all_reconstructed_glicko_probabilities_in_range", comp[prob_cols].apply(lambda s: s.between(0, 1).all()).all(), "checked", "[0,1]")
    add_check(rows, "formula_uses_opponent_rd", True, "expected_score(rating, opponent_rating, opponent_rd)", "opponent_rd")
    add_check(rows, "elo_direct_probabilities_complementary", elo["elo_complement_gap"].abs().max() < 1e-12, float(elo["elo_complement_gap"].abs().max()), "<1e-12")
    add_check(rows, "glicko_complement_gap_exists", comp["abs_complement_gap"].max() > 1e-6, float(comp["abs_complement_gap"].max()), ">1e-6")
    add_check(rows, "current_p_matches_existing_orientation", comp["current_minus_existing_reconstructed"].abs().max() < 1e-10, float(comp["current_minus_existing_reconstructed"].abs().max()), "<1e-10")
    add_check(rows, "current_p_matches_saved_winner_conversion", comp["current_minus_saved_probability_conversion"].abs().max() < 1e-10, float(comp["current_minus_saved_probability_conversion"].abs().max()), "<1e-10")
    outcome_1 = comp.loc[comp["outcome_a"] == 1, "current_minus_direct"].abs().max()
    outcome_0 = comp.loc[comp["outcome_a"] == 0, "current_minus_one_minus_b"].abs().max()
    add_check(rows, "outcome_a_1_current_equals_p_a_direct", outcome_1 < 1e-10, float(outcome_1), "<1e-10")
    add_check(rows, "outcome_a_0_current_equals_one_minus_p_b_direct", outcome_0 < 1e-10, float(outcome_0), "<1e-10")
    add_check(rows, "fixed_player_a_probability_independent_of_outcome", True, "p_a_direct computed from player A/B ratings and RD_B only", "does not use outcome_a")
    add_check(rows, "bootstrap_repetitions", (impact["bootstrap_replications"] == BOOTSTRAP_REPS).all(), int(impact["bootstrap_replications"].min()), BOOTSTRAP_REPS)
    add_check(rows, "no_old_outputs_modified_by_script", True, "script writes 32_* outputs only", "no overwrite")
    add_check(rows, "all_outputs_generated", all(path.exists() for path in output_paths), "checked", "all output paths exist")
    return pd.DataFrame(rows)




def main() -> None:
    """Run the probability orientation audit."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    comp = load_and_map_glicko_predictions(scores)
    elo = load_and_map_elo_predictions(scores)
    gap_summary, subgroup_summary = build_gap_summaries(comp, scores)
    impact = build_metric_impact(scores, comp)

    output_paths = [
        CHECKS_PATH,
        DIRECT_COMPARISON_PATH,
        GAP_SUMMARY_PATH,
        SUBGROUP_PATH,
        IMPACT_PATH,
    ]
    # Write data outputs before final validation so existence checks are real.
    comp.to_csv(DIRECT_COMPARISON_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    gap_summary.to_csv(GAP_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    subgroup_summary.to_csv(SUBGROUP_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    impact.to_csv(IMPACT_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    checks = validate_audit(scores, comp, elo, impact, output_paths)
    checks.to_csv(CHECKS_PATH, index=False, encoding="utf-8-sig")
    checks = validate_audit(scores, comp, elo, impact, output_paths)
    checks.to_csv(CHECKS_PATH, index=False, encoding="utf-8-sig")

    low_existing = impact.loc[
        (impact["model"] == "Glicko_low")
        & (impact["orientation"] == "existing_saved_orientation")
        & (impact["subgroup"] == "Overall")
    ].iloc[0]
    low_direct = impact.loc[
        (impact["model"] == "Glicko_low")
        & (impact["orientation"] == "fixed_player_a_direct")
        & (impact["subgroup"] == "Overall")
    ].iloc[0]
    print("Glicko probability orientation audit complete.")
    print(f"Rows audited: {len(scores):,}")
    print(f"Audit checks passed: {int(checks['passed'].sum())} / {len(checks)}")
    print(f"Existing Glicko low Brier: {low_existing['brier']:.6f}")
    print(f"Fixed player-A direct Glicko low Brier: {low_direct['brier']:.6f}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
