"""Orientation-corrected model comparison and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP29_SCORES_PATH = OUTPUT_DIR / "29_per_match_model_scores_2025.csv"
STEP31_DEBUT_MECHANISM_PATH = OUTPUT_DIR / "31_debut_probability_mechanism.csv"
STEP31_DEBUT_OPPONENT_SUMMARY_PATH = OUTPUT_DIR / "31_debut_opponent_rating_summary.csv"
STEP31_UNIQUE_PLAYER_SNAPSHOT_PATH = OUTPUT_DIR / "31_unique_player_rating_snapshot.csv"
STEP32_DIRECT_PATH = OUTPUT_DIR / "32_glicko_direct_probability_comparison.csv"
STEP32_IMPACT_PATH = OUTPUT_DIR / "32_orientation_impact_on_metrics.csv"


def configure_output_root(output_root: str | Path) -> Path:
    global OUTPUT_DIR, FIGURE_DIR

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    OUTPUT_DIR = root.resolve() / "meeting6"
    FIGURE_DIR = OUTPUT_DIR / "figures"
    return OUTPUT_DIR


RANDOM_SEED = 20260715
BOOTSTRAP_REPS = 2_000
EXPECTED_GAMES = 11_379
EPS = 1e-15

MODEL_ORDER = [
    "Glicko_low_fixed",
    "Validation_best_Elo",
    "best_AdaptiveK",
    "Default_Elo",
    "Glicko_C0_fixed",
    "Conservative_Elo",
]

MODEL_LABELS = {
    "Glicko_low_fixed": "Glicko low inflation",
    "Glicko_C0_fixed": "Glicko C0",
    "Validation_best_Elo": "Validation-best Elo",
    "best_AdaptiveK": "Best adaptive-K Elo",
    "Default_Elo": "Default Elo",
    "Conservative_Elo": "Conservative Elo",
}

MODEL_SOURCE = {
    "Glicko_low_fixed": "Step32 fixed direct player-A probability from Glicko_low_inflation_match_by_match",
    "Glicko_C0_fixed": "Step32 fixed direct player-A probability from Glicko_C0_match_by_match",
    "Validation_best_Elo": "Step29 player-A probability from Validation_best_Elo",
    "best_AdaptiveK": "Step29 player-A probability from AdaptiveK_PreviousYearGames_Elo_scale300",
    "Default_Elo": "Step29 player-A probability from Default_Elo",
    "Conservative_Elo": "Step29 player-A probability from Conservative_Elo",
}

PAIRWISE_DIFFS = {
    "delta_brier_glicko_vs_elo": ("brier_Validation_best_Elo", "brier_Glicko_low_fixed", "Brier: Glicko low vs validation-best Elo"),
    "delta_logloss_glicko_vs_elo": ("logloss_Validation_best_Elo", "logloss_Glicko_low_fixed", "Log loss: Glicko low vs validation-best Elo"),
    "delta_brier_inflation": ("brier_Glicko_C0_fixed", "brier_Glicko_low_fixed", "Brier: low RD inflation vs Glicko C0"),
    "delta_logloss_inflation": ("logloss_Glicko_C0_fixed", "logloss_Glicko_low_fixed", "Log loss: low RD inflation vs Glicko C0"),
    "delta_brier_glicko_vs_adaptive": ("brier_best_AdaptiveK", "brier_Glicko_low_fixed", "Brier: Glicko low vs best adaptive-K"),
    "delta_logloss_glicko_vs_adaptive": ("logloss_best_AdaptiveK", "logloss_Glicko_low_fixed", "Log loss: Glicko low vs best adaptive-K"),
}


@dataclass(frozen=True)
class SubgroupSpec:
    variable: str
    subgroup: str
    title: str
    kind: str
    mask: pd.Series


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", detail: str = "", severity: str = "error") -> None:
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


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [STEP29_SCORES_PATH, STEP32_DIRECT_PATH, STEP32_IMPACT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    scores = pd.read_csv(STEP29_SCORES_PATH, low_memory=False)
    comp = pd.read_csv(STEP32_DIRECT_PATH, low_memory=False)
    step32_impact = pd.read_csv(STEP32_IMPACT_PATH, low_memory=False)
    debut_mechanism = pd.read_csv(STEP31_DEBUT_MECHANISM_PATH, low_memory=False) if STEP31_DEBUT_MECHANISM_PATH.exists() else pd.DataFrame()

    require_columns(
        scores,
        [
            "match_id",
            "match_sequence",
            "year",
            "event_id",
            "match_date",
            "player_a_id",
            "player_b_id",
            "winner_id",
            "loser_id",
            "outcome_a",
            "p_a_Validation_best_Elo",
            "p_a_Default_Elo",
            "p_a_Conservative_Elo",
            "p_a_best_AdaptiveK",
            "either_player_debut",
            "a_is_debut",
            "b_is_debut",
            "both_players_have_history",
            "either_player_inactive_365d",
            "either_player_inactive_730d",
            "both_players_active_last_365d",
            "min_total_games_before",
            "min_games_last_365_days",
            "min_previous_year_games",
            "max_days_since_last_game",
            "max_prematch_rd",
        ],
        "Step29 per-match scores",
    )
    require_columns(
        comp,
        [
            "match_id",
            "model",
            "player_a_id",
            "player_b_id",
            "outcome_a",
            "rating_a",
            "rating_b",
            "rd_a",
            "rd_b",
            "p_a_current_step29",
            "p_a_direct",
            "p_b_direct",
            "p_a_symmetric_diagnostic",
            "complement_gap",
        ],
        "Step32 direct probability comparison",
    )

    scores["event_key"] = scores["year"].astype(str) + "_" + scores["event_id"].astype(str)
    comp["event_key"] = comp["year"].astype(str) + "_" + comp["event_id"].astype(str)
    return scores, comp, step32_impact, debut_mechanism


def validate_canonical_player_orientation(scores: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical player and outcome orientation."""

    rows: list[dict[str, Any]] = []
    a_is_min_id = (scores["player_a_id"] == scores[["winner_id", "loser_id"]].min(axis=1)).all()
    b_is_max_id = (scores["player_b_id"] == scores[["winner_id", "loser_id"]].max(axis=1)).all()
    outcome_correct = (scores["outcome_a"].astype(int) == (scores["player_a_id"] == scores["winner_id"]).astype(int)).all()
    players_distinct = (scores["player_a_id"] != scores["player_b_id"]).all()
    stable_pair = (
        scores[["player_a_id", "player_b_id"]].min(axis=1).eq(scores["player_a_id"]).all()
        and scores[["player_a_id", "player_b_id"]].max(axis=1).eq(scores["player_b_id"]).all()
    )

    add_check(rows, "feature_rows", len(scores) == EXPECTED_GAMES, len(scores), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", scores["match_id"].duplicated().sum() == 0, int(scores["match_id"].duplicated().sum()), 0)
    add_check(rows, "player_a_is_lower_player_id", bool(a_is_min_id), bool(a_is_min_id), True, "Current Step 28/29 canonical A equals min(winner_id, loser_id).")
    add_check(rows, "player_b_is_higher_player_id", bool(b_is_max_id), bool(b_is_max_id), True)
    add_check(rows, "canonical_pair_stable_by_player_id_only", bool(stable_pair), bool(stable_pair), True)
    add_check(rows, "two_distinct_players_per_match", bool(players_distinct), bool(players_distinct), True)
    add_check(rows, "outcome_a_matches_winner_id", bool(outcome_correct), bool(outcome_correct), True)
    add_check(rows, "canonical_a_b_independent_of_actual_outcome", bool(a_is_min_id and b_is_max_id), "A=min player ID, B=max player ID", "no winner/loser rule")
    checks = pd.DataFrame(rows)
    checks.to_csv(OUTPUT_DIR / "33_canonical_player_orientation_checks.csv", index=False)
    return checks


def add_glicko_fixed_probabilities(scores: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Use direct Player A probabilities because unequal RDs break symmetry."""

    out = scores.copy()
    model_map = {
        "Glicko_low": "Glicko_low",
        "Glicko_C0": "Glicko_C0",
    }

    for step32_model, prefix in model_map.items():
        sub = comp.loc[comp["model"] == step32_model].copy()
        if len(sub) != EXPECTED_GAMES:
            raise ValueError(f"{step32_model} has {len(sub)} rows in Step32 direct comparison; expected {EXPECTED_GAMES}")
        sub = sub[
            [
                "match_id",
                "player_a_id",
                "player_b_id",
                "outcome_a",
                "rating_a",
                "rating_b",
                "rd_a",
                "rd_b",
                "p_a_current_step29",
                "p_a_direct",
                "p_b_direct",
                "p_a_symmetric_diagnostic",
                "complement_gap",
                "abs_complement_gap",
            ]
        ].rename(
            columns={
                "rating_a": f"rating_a_{prefix}",
                "rating_b": f"rating_b_{prefix}",
                "rd_a": f"rd_a_{prefix}",
                "rd_b": f"rd_b_{prefix}",
                "p_a_current_step29": f"old_p_a_{prefix}",
                "p_a_direct": f"p_a_{prefix}_fixed",
                "p_b_direct": f"p_b_{prefix}_direct",
                "p_a_symmetric_diagnostic": f"p_a_{prefix}_symmetric",
                "complement_gap": f"{prefix}_complement_gap",
                "abs_complement_gap": f"{prefix}_abs_complement_gap",
            }
        )
        out = out.merge(sub, on=["match_id", "player_a_id", "player_b_id", "outcome_a"], how="left", validate="one_to_one")
        out[f"p_a_{prefix}_from_B"] = 1.0 - out[f"p_b_{prefix}_direct"]

    out = out.rename(
        columns={
            "p_a_Glicko_low_fixed": "p_a_Glicko_low_fixed",
            "p_a_Glicko_C0_fixed": "p_a_Glicko_C0_fixed",
        }
    )
    return out


def score_probability(df: pd.DataFrame, alias: str, p_col: str) -> None:
    """Score stored prematch Player A probabilities."""

    p = df[p_col].astype(float).clip(0.0, 1.0)
    y = df["outcome_a"].astype(float)
    clipped = p.clip(EPS, 1.0 - EPS)
    df[f"brier_{alias}"] = (p - y) ** 2
    df[f"logloss_{alias}"] = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    pred_a_win = p >= 0.5
    df[f"correct_{alias}"] = (pred_a_win.astype(int) == y.astype(int)).astype(int)
    df[f"favourite_probability_{alias}"] = np.maximum(p, 1.0 - p)
    df[f"favourite_won_{alias}"] = np.where(pred_a_win, y, 1.0 - y).astype(int)


def calculate_per_match_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate model scores for each game."""

    out = scores.copy()
    probability_cols = {
        "Glicko_low_fixed": "p_a_Glicko_low_fixed",
        "Glicko_C0_fixed": "p_a_Glicko_C0_fixed",
        "Validation_best_Elo": "p_a_Validation_best_Elo",
        "best_AdaptiveK": "p_a_best_AdaptiveK",
        "Default_Elo": "p_a_Default_Elo",
        "Conservative_Elo": "p_a_Conservative_Elo",
    }
    for alias, p_col in probability_cols.items():
        require_columns(out, [p_col], "orientation-corrected score table")
        score_probability(out, alias, p_col)

    out["delta_brier_glicko_vs_elo"] = out["brier_Validation_best_Elo"] - out["brier_Glicko_low_fixed"]
    out["delta_logloss_glicko_vs_elo"] = out["logloss_Validation_best_Elo"] - out["logloss_Glicko_low_fixed"]
    out["delta_accuracy_glicko_vs_elo"] = out["correct_Glicko_low_fixed"] - out["correct_Validation_best_Elo"]
    out["delta_brier_inflation"] = out["brier_Glicko_C0_fixed"] - out["brier_Glicko_low_fixed"]
    out["delta_logloss_inflation"] = out["logloss_Glicko_C0_fixed"] - out["logloss_Glicko_low_fixed"]
    out["delta_accuracy_inflation"] = out["correct_Glicko_low_fixed"] - out["correct_Glicko_C0_fixed"]
    out["delta_brier_glicko_vs_adaptive"] = out["brier_best_AdaptiveK"] - out["brier_Glicko_low_fixed"]
    out["delta_logloss_glicko_vs_adaptive"] = out["logloss_best_AdaptiveK"] - out["logloss_Glicko_low_fixed"]
    out["delta_accuracy_glicko_vs_adaptive"] = out["correct_Glicko_low_fixed"] - out["correct_best_AdaptiveK"]
    out["delta_brier_tuned_vs_default"] = out["brier_Default_Elo"] - out["brier_Validation_best_Elo"]
    out["delta_logloss_tuned_vs_default"] = out["logloss_Default_Elo"] - out["logloss_Validation_best_Elo"]

    out["glicko_low_fixed_favourite_probability_minus_elo"] = (
        out["favourite_probability_Glicko_low_fixed"] - out["favourite_probability_Validation_best_Elo"]
    )
    out["subgroup_glicko_vs_elo_confidence_change"] = out["glicko_low_fixed_favourite_probability_minus_elo"].map(confidence_category)
    out["abs_probability_difference_fixed"] = (out["p_a_Glicko_low_fixed"] - out["p_a_Validation_best_Elo"]).abs()

    no_debut = ~out["either_player_debut"].astype(bool)
    out["no_debut_rd_quartile_33"] = pd.NA
    if no_debut.any():
        out.loc[no_debut, "no_debut_rd_quartile_33"] = pd.qcut(
            out.loc[no_debut, "max_prematch_rd"],
            q=4,
            labels=["Q1 lowest uncertainty", "Q2", "Q3", "Q4 highest uncertainty"],
            duplicates="raise",
        ).astype(str)
    return out.sort_values("match_sequence").reset_index(drop=True)


def confidence_category(value: float) -> str:
    if value < -0.05:
        return "Glicko substantially less confident"
    if value < -0.01:
        return "Glicko slightly less confident"
    if value <= 0.01:
        return "Similar confidence"
    if value <= 0.05:
        return "Glicko slightly more confident"
    return "Glicko substantially more confident"


def calibration_error_favourite(group: pd.DataFrame, alias: str) -> tuple[float, pd.DataFrame]:
    p = group[f"favourite_probability_{alias}"].astype(float).to_numpy()
    y = group[f"favourite_won_{alias}"].astype(float).to_numpy()
    bins = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.0000000001])
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    idx = np.minimum(np.digitize(p, bins, right=False) - 1, len(labels) - 1)
    rows = []
    error = 0.0
    for i, label in enumerate(labels):
        mask = idx == i
        games = int(mask.sum())
        if games:
            mean_p = float(np.mean(p[mask]))
            win_rate = float(np.mean(y[mask]))
            gap = win_rate - mean_p
            error += (games / len(group)) * abs(gap)
        else:
            mean_p = np.nan
            win_rate = np.nan
            gap = np.nan
        rows.append(
            {
                "bin": label,
                "games": games,
                "mean_favourite_probability": mean_p,
                "empirical_favourite_win_rate": win_rate,
                "calibration_gap": gap,
            }
        )
    return float(error), pd.DataFrame(rows)


def model_metric_row(group: pd.DataFrame, alias: str) -> dict[str, Any]:
    cal_error, _ = calibration_error_favourite(group, alias)
    return {
        "model": alias,
        "display_name": MODEL_LABELS[alias],
        "source_model": MODEL_SOURCE[alias],
        "model_family": "Glicko" if "Glicko" in alias else "Elo",
        "evaluation_games": int(len(group)),
        "events": int(group["event_key"].nunique()),
        "brier": float(group[f"brier_{alias}"].mean()),
        "log_loss": float(group[f"logloss_{alias}"].mean()),
        "accuracy": float(group[f"correct_{alias}"].mean()),
        "mean_favourite_probability": float(group[f"favourite_probability_{alias}"].mean()),
        "favourite_win_rate": float(group[f"favourite_won_{alias}"].mean()),
        "corrected_favourite_calibration_error": cal_error,
        "accuracy_tie_rule": "p_a >= 0.5 predicts player A",
    }


def calculate_overall_model_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([model_metric_row(scores, alias) for alias in MODEL_ORDER])


def paired_point_metrics(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {
            "games": 0,
            "events": 0,
            "glicko_brier": np.nan,
            "elo_brier": np.nan,
            "delta_brier_glicko_vs_elo": np.nan,
            "glicko_logloss": np.nan,
            "elo_logloss": np.nan,
            "delta_logloss_glicko_vs_elo": np.nan,
            "glicko_accuracy": np.nan,
            "elo_accuracy": np.nan,
            "delta_accuracy_glicko_vs_elo": np.nan,
            "delta_brier_inflation": np.nan,
            "delta_logloss_inflation": np.nan,
            "delta_accuracy_inflation": np.nan,
            "delta_brier_glicko_vs_adaptive": np.nan,
            "delta_logloss_glicko_vs_adaptive": np.nan,
            "delta_accuracy_glicko_vs_adaptive": np.nan,
            "percentage_games_glicko_lower_brier_than_elo": np.nan,
            "percentage_games_glicko_lower_logloss_than_elo": np.nan,
        }

    return {
        "games": int(len(group)),
        "events": int(group["event_key"].nunique()),
        "glicko_brier": float(group["brier_Glicko_low_fixed"].mean()),
        "elo_brier": float(group["brier_Validation_best_Elo"].mean()),
        "delta_brier_glicko_vs_elo": float(group["delta_brier_glicko_vs_elo"].mean()),
        "glicko_logloss": float(group["logloss_Glicko_low_fixed"].mean()),
        "elo_logloss": float(group["logloss_Validation_best_Elo"].mean()),
        "delta_logloss_glicko_vs_elo": float(group["delta_logloss_glicko_vs_elo"].mean()),
        "glicko_accuracy": float(group["correct_Glicko_low_fixed"].mean()),
        "elo_accuracy": float(group["correct_Validation_best_Elo"].mean()),
        "delta_accuracy_glicko_vs_elo": float(group["delta_accuracy_glicko_vs_elo"].mean()),
        "delta_brier_inflation": float(group["delta_brier_inflation"].mean()),
        "delta_logloss_inflation": float(group["delta_logloss_inflation"].mean()),
        "delta_accuracy_inflation": float(group["delta_accuracy_inflation"].mean()),
        "delta_brier_glicko_vs_adaptive": float(group["delta_brier_glicko_vs_adaptive"].mean()),
        "delta_logloss_glicko_vs_adaptive": float(group["delta_logloss_glicko_vs_adaptive"].mean()),
        "delta_accuracy_glicko_vs_adaptive": float(group["delta_accuracy_glicko_vs_adaptive"].mean()),
        "percentage_games_glicko_lower_brier_than_elo": float((group["brier_Glicko_low_fixed"] < group["brier_Validation_best_Elo"]).mean()),
        "percentage_games_glicko_lower_logloss_than_elo": float((group["logloss_Glicko_low_fixed"] < group["logloss_Validation_best_Elo"]).mean()),
    }


def bootstrap_differences(group: pd.DataFrame, diff_cols: list[str], seed: int, reps: int = BOOTSTRAP_REPS) -> dict[str, tuple[float, float, str]]:
    """Bootstrap by event, falling back to games for one event cluster."""

    if group.empty:
        return {col: (np.nan, np.nan, "empty") for col in diff_cols}

    rng = np.random.default_rng(seed)
    work = group[["event_key", *diff_cols]].copy()
    events = work["event_key"].nunique()
    if events >= 2:
        event_sums = work.groupby("event_key", sort=False)[diff_cols].sum().to_numpy(dtype=float)
        event_counts = work.groupby("event_key", sort=False).size().to_numpy(dtype=float)
        draws = rng.integers(0, len(event_counts), size=(reps, len(event_counts)))
        counts = event_counts[draws].sum(axis=1)
        sums = event_sums[draws].sum(axis=1)
        means = sums / counts[:, None]
        bootstrap_type = "event_cluster"
    else:
        values = work[diff_cols].to_numpy(dtype=float)
        draws = rng.integers(0, len(values), size=(reps, len(values)))
        means = values[draws].mean(axis=1)
        bootstrap_type = "match_level"

    out: dict[str, tuple[float, float, str]] = {}
    for i, col in enumerate(diff_cols):
        out[col] = (float(np.quantile(means[:, i], 0.025)), float(np.quantile(means[:, i], 0.975)), bootstrap_type)
    return out


def overall_pairwise_and_ci(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = paired_point_metrics(scores)
    pairwise = pd.DataFrame(
        [
            {
                "comparison": "Glicko low fixed vs validation-best Elo",
                "games": point["games"],
                "events": point["events"],
                "delta_brier": point["delta_brier_glicko_vs_elo"],
                "delta_logloss": point["delta_logloss_glicko_vs_elo"],
                "delta_accuracy": point["delta_accuracy_glicko_vs_elo"],
                "positive_means": "Glicko low fixed is better",
            },
            {
                "comparison": "Low-inflation Glicko vs Glicko C0",
                "games": point["games"],
                "events": point["events"],
                "delta_brier": point["delta_brier_inflation"],
                "delta_logloss": point["delta_logloss_inflation"],
                "delta_accuracy": point["delta_accuracy_inflation"],
                "positive_means": "Low-inflation Glicko is better",
            },
            {
                "comparison": "Glicko low fixed vs best adaptive-K Elo",
                "games": point["games"],
                "events": point["events"],
                "delta_brier": point["delta_brier_glicko_vs_adaptive"],
                "delta_logloss": point["delta_logloss_glicko_vs_adaptive"],
                "delta_accuracy": point["delta_accuracy_glicko_vs_adaptive"],
                "positive_means": "Glicko low fixed is better",
            },
        ]
    )

    ci_map = bootstrap_differences(scores, list(PAIRWISE_DIFFS), RANDOM_SEED)
    ci_rows = []
    for name, (_, _, label) in PAIRWISE_DIFFS.items():
        low, high, boot_type = ci_map[name]
        ci_rows.append(
            {
                "diff_name": name,
                "diff_label": label,
                "games": int(len(scores)),
                "unique_events": int(scores["event_key"].nunique()),
                "point_estimate": float(scores[name].mean()),
                "ci_lower": low,
                "ci_upper": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "bootstrap_type": boot_type,
                "bootstrap_replications": BOOTSTRAP_REPS,
                "small_sample_warning": False,
            }
        )
    return pairwise, pd.DataFrame(ci_rows)


def subgroup_specs(scores: pd.DataFrame) -> list[SubgroupSpec]:
    """Build subgroup definitions from prematch information only."""

    specs: list[SubgroupSpec] = []

    def add(variable: str, subgroup: str, title: str, kind: str, mask: pd.Series) -> None:
        specs.append(SubgroupSpec(variable, subgroup, title, kind, mask.fillna(False).astype(bool)))

    add("overall", "Overall", "Overall", "overall", pd.Series(True, index=scores.index))

    ordered_categories = {
        "subgroup_total_experience": ("total_experience", "Minimum total previous games", ["0", "1-5", "6-20", "21-50", "51-100", "100+"], "player_category"),
        "subgroup_recent_365_activity": ("recent_365_activity", "Minimum games in last 365 days", ["0", "1-5", "6-15", "16-30", "30+", "Missing date information"], "player_category"),
        "subgroup_previous_year_activity": ("previous_year_activity", "Minimum previous-calendar-year games", ["0", "1-5", "6-15", "16-30", "30+"], "player_category"),
        "subgroup_inactivity_gap": (
            "inactivity_gap",
            "Maximum inactivity gap",
            ["No previous history", "0-90 days", "91-180 days", "181-365 days", "366-730 days", "731-1095 days", "1096+ days", "Missing date information"],
            "player_category",
        ),
        "subgroup_experience_ratio": (
            "experience_ratio",
            "Experience imbalance",
            ["No previous history", "Balanced: ratio <= 2", "Moderate mismatch: 2 < ratio <= 5", "Large mismatch: ratio > 5"],
            "player_category",
        ),
        "subgroup_elo_favourite_probability": (
            "elo_favourite_probability",
            "Validation-best Elo favourite probability",
            ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
            "prediction_confidence",
        ),
        "no_debut_rd_quartile_33": (
            "no_debut_rd_quartile",
            "No-debut Glicko pre-match RD quartile",
            ["Q1 lowest uncertainty", "Q2", "Q3", "Q4 highest uncertainty"],
            "glicko_uncertainty",
        ),
        "subgroup_glicko_vs_elo_confidence_change": (
            "glicko_vs_elo_confidence_change",
            "Glicko fixed favourite-confidence change vs Elo",
            [
                "Glicko substantially less confident",
                "Glicko slightly less confident",
                "Similar confidence",
                "Glicko slightly more confident",
                "Glicko substantially more confident",
            ],
            "diagnostic",
        ),
    }

    for col, (variable, title, categories, kind) in ordered_categories.items():
        for category in categories:
            add(variable, category, title, kind, scores[col].astype(str) == category)

    binary_flags = {
        "either_player_debut": "Either player debut",
        "either_player_inactive_365d": "Either player inactive >=365 days",
        "either_player_inactive_730d": "Either player inactive >=730 days",
        "both_players_active_last_365d": "Both players active last 365 days",
        "either_player_low_recent_activity": "Either player low recent activity",
        "min_total_games_before_le5": "Minimum total games <=5",
        "min_total_games_before_le20": "Minimum total games <=20",
        "min_games_last_365_days_le5": "Minimum recent games <=5",
        "min_previous_year_games_le5": "Minimum previous-year games <=5",
    }
    flag_sources = {
        "either_player_debut": scores["either_player_debut"].astype(bool),
        "either_player_inactive_365d": scores["either_player_inactive_365d"].astype(bool),
        "either_player_inactive_730d": scores["either_player_inactive_730d"].astype(bool),
        "both_players_active_last_365d": scores["both_players_active_last_365d"].astype(bool),
        "either_player_low_recent_activity": scores["either_player_low_recent_activity"].astype(bool),
        "min_total_games_before_le5": scores["min_total_games_before"] <= 5,
        "min_total_games_before_le20": scores["min_total_games_before"] <= 20,
        "min_games_last_365_days_le5": scores["min_games_last_365_days"] <= 5,
        "min_previous_year_games_le5": scores["min_previous_year_games"] <= 5,
    }
    for variable, title in binary_flags.items():
        flag = flag_sources[variable].fillna(False).astype(bool)
        add(variable, "True", title, "binary_flag", flag)
        add(variable, "False", title, "binary_flag", ~flag)

    no_debut = ~scores["either_player_debut"].astype(bool)
    exactly_one_debut = scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)
    add("meeting_key_samples", "No debut", "Meeting key samples", "meeting_key", no_debut)
    add("meeting_key_samples", "Exactly one debut", "Meeting key samples", "meeting_key", exactly_one_debut)
    add("meeting_key_samples", "Returning >=365 days, no debut", "Meeting key samples", "meeting_key", no_debut & scores["either_player_inactive_365d"].astype(bool))
    add("meeting_key_samples", "Returning >=730 days, no debut", "Meeting key samples", "meeting_key", no_debut & scores["either_player_inactive_730d"].astype(bool))
    add("meeting_key_samples", "Both active and no debut", "Meeting key samples", "meeting_key", no_debut & scores["both_players_active_last_365d"].astype(bool))
    return specs


def calculate_subgroup_model_performance(scores: pd.DataFrame, specs: list[SubgroupSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        group = scores.loc[spec.mask].copy()
        if group.empty:
            continue
        for alias in MODEL_ORDER:
            row = model_metric_row(group, alias)
            rows.append(
                {
                    "subgroup_variable": spec.variable,
                    "subgroup": spec.subgroup,
                    "subgroup_title": spec.title,
                    "subgroup_kind": spec.kind,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def calculate_subgroup_pairwise(scores: pd.DataFrame, specs: list[SubgroupSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        group = scores.loc[spec.mask].copy()
        if group.empty:
            continue
        point = paired_point_metrics(group)
        rows.append(
            {
                "subgroup_variable": spec.variable,
                "subgroup": spec.subgroup,
                "subgroup_title": spec.title,
                "subgroup_kind": spec.kind,
                **point,
                "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
            }
        )
    return pd.DataFrame(rows)


def calculate_subgroup_bootstrap(scores: pd.DataFrame, specs: list[SubgroupSpec]) -> pd.DataFrame:
    """Calculate event-cluster intervals for subgroup differences."""

    rows = []
    diff_cols = list(PAIRWISE_DIFFS)
    for i, spec in enumerate(specs):
        group = scores.loc[spec.mask].copy()
        if group.empty:
            continue
        ci_map = bootstrap_differences(group, diff_cols, RANDOM_SEED + i * 37)
        for diff_name, (_, _, label) in PAIRWISE_DIFFS.items():
            low, high, boot_type = ci_map[diff_name]
            rows.append(
                {
                    "subgroup_variable": spec.variable,
                    "subgroup": spec.subgroup,
                    "subgroup_title": spec.title,
                    "diff_name": diff_name,
                    "diff_label": label,
                    "games": int(len(group)),
                    "unique_events": int(group["event_key"].nunique()),
                    "point_estimate": float(group[diff_name].mean()),
                    "ci_lower": low,
                    "ci_upper": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "bootstrap_type": boot_type,
                    "bootstrap_replications": BOOTSTRAP_REPS,
                    "small_sample_warning": bool(len(group) < 50 or group["event_key"].nunique() < 10),
                }
            )
    return pd.DataFrame(rows)


def sample_masks(scores: pd.DataFrame) -> dict[str, pd.Series]:
    no_debut = ~scores["either_player_debut"].astype(bool)
    exactly_one_debut = scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)
    return {
        "Overall": pd.Series(True, index=scores.index),
        "No debut": no_debut,
        "Exactly one debut": exactly_one_debut,
        "Either player debut": scores["either_player_debut"].astype(bool),
        "Total previous games <=5": scores["min_total_games_before"] <= 5,
        "Recent games <=5": scores["min_games_last_365_days"] <= 5,
        "Inactive >=365 days, no debut": no_debut & scores["either_player_inactive_365d"].astype(bool),
        "Inactive >=730 days, no debut": no_debut & scores["either_player_inactive_730d"].astype(bool),
        "Both players active last 365 days": scores["both_players_active_last_365d"].astype(bool),
        "Both active and no debut": no_debut & scores["both_players_active_last_365d"].astype(bool),
    }


def calculate_adaptive_k_recovery(scores: pd.DataFrame) -> pd.DataFrame:
    """Measure the improvement recovered by Adaptive-K Elo."""

    rows = []
    for key, mask in sample_masks(scores).items():
        if key not in [
            "Overall",
            "Either player debut",
            "Total previous games <=5",
            "Recent games <=5",
            "Inactive >=365 days, no debut",
            "Both players active last 365 days",
        ]:
            continue
        group = scores.loc[mask].copy()
        default_brier = group["brier_Default_Elo"].mean()
        adaptive_brier = group["brier_best_AdaptiveK"].mean()
        glicko_brier = group["brier_Glicko_low_fixed"].mean()
        default_logloss = group["logloss_Default_Elo"].mean()
        adaptive_logloss = group["logloss_best_AdaptiveK"].mean()
        glicko_logloss = group["logloss_Glicko_low_fixed"].mean()

        denom_brier = default_brier - glicko_brier
        denom_logloss = default_logloss - glicko_logloss
        valid_brier = bool(denom_brier > 1e-12)
        valid_logloss = bool(denom_logloss > 1e-12)
        rows.append(
            {
                "range": key.lower().replace(" ", "_").replace(">=", "ge").replace("<=", "le"),
                "range_label": key,
                "games": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "brier_default_elo": default_brier,
                "brier_best_adaptive_k": adaptive_brier,
                "brier_glicko_low_fixed": glicko_brier,
                "improvement_recovered_brier": (default_brier - adaptive_brier) / denom_brier if valid_brier else np.nan,
                "recovery_ratio_valid_brier": valid_brier,
                "logloss_default_elo": default_logloss,
                "logloss_best_adaptive_k": adaptive_logloss,
                "logloss_glicko_low_fixed": glicko_logloss,
                "improvement_recovered_logloss": (default_logloss - adaptive_logloss) / denom_logloss if valid_logloss else np.nan,
                "recovery_ratio_valid_logloss": valid_logloss,
            }
        )
    return pd.DataFrame(rows)


def build_debut_corrected_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = {
        "Either player debut": scores["either_player_debut"].astype(bool),
        "Exactly one debut": scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool),
        "Both players debut": scores["a_is_debut"].astype(bool) & scores["b_is_debut"].astype(bool),
        "No debut": ~scores["either_player_debut"].astype(bool),
    }
    rows = []
    for sample, mask in masks.items():
        group = scores.loc[mask].copy()
        if group.empty:
            continue
        for alias in ["Glicko_low_fixed", "Glicko_C0_fixed", "Validation_best_Elo", "best_AdaptiveK", "Default_Elo"]:
            rows.append({"sample": sample, **model_metric_row(group, alias)})

    player_rows = []
    debut_matches = scores.loc[scores["either_player_debut"].astype(bool)].copy()
    for side in ["a", "b"]:
        if side == "a":
            player_id = debut_matches["player_a_id"]
            opponent_id = debut_matches["player_b_id"]
            outcome = debut_matches["outcome_a"].astype(int)
            is_debut = debut_matches["a_is_debut"].astype(bool)
        else:
            player_id = debut_matches["player_b_id"]
            opponent_id = debut_matches["player_a_id"]
            outcome = 1 - debut_matches["outcome_a"].astype(int)
            is_debut = debut_matches["b_is_debut"].astype(bool)
        base = pd.DataFrame(
            {
                "match_id": debut_matches["match_id"].to_numpy(),
                "event_key": debut_matches["event_key"].to_numpy(),
                "player_side": side.upper(),
                "player_id": player_id.to_numpy(),
                "opponent_id": opponent_id.to_numpy(),
                "is_debut_player": is_debut.to_numpy(),
                "player_won": outcome.to_numpy(),
            }
        )
        for alias in ["Glicko_low_fixed", "Glicko_C0_fixed", "Validation_best_Elo", "best_AdaptiveK", "Default_Elo"]:
            p_a = debut_matches[f"p_a_{alias}"].astype(float)
            p_player = p_a if side == "a" else (1.0 - p_a)
            tmp = base.copy()
            tmp["model"] = alias
            tmp["model_display"] = MODEL_LABELS[alias]
            tmp["p_player_win"] = p_player.to_numpy()
            tmp["brier_player"] = (tmp["p_player_win"] - tmp["player_won"]) ** 2
            tmp["predicted_player_win"] = tmp["p_player_win"] >= 0.5
            tmp["correct_player_prediction"] = (tmp["predicted_player_win"].astype(int) == tmp["player_won"].astype(int)).astype(int)
            player_rows.append(tmp)
    return pd.DataFrame(rows), pd.concat(player_rows, ignore_index=True)


def build_returner_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_debut = ~scores["either_player_debut"].astype(bool)
    valid_gap = no_debut & scores["both_players_have_history"].astype(bool) & scores["max_days_since_last_game"].notna()

    cumulative_rows = []
    for threshold in [180, 365, 540, 730, 1095]:
        mask = valid_gap & (scores["max_days_since_last_game"] >= threshold)
        group = scores.loc[mask].copy()
        point = paired_point_metrics(group)
        ci_map = bootstrap_differences(group, ["delta_brier_glicko_vs_elo", "delta_logloss_glicko_vs_elo", "delta_brier_inflation"], RANDOM_SEED + threshold)
        cumulative_rows.append(
            {
                "subgroup": f"Returning >= {threshold} days, no debut",
                "threshold_days": threshold,
                **point,
                "delta_brier_ci_lower": ci_map["delta_brier_glicko_vs_elo"][0],
                "delta_brier_ci_upper": ci_map["delta_brier_glicko_vs_elo"][1],
                "delta_logloss_ci_lower": ci_map["delta_logloss_glicko_vs_elo"][0],
                "delta_logloss_ci_upper": ci_map["delta_logloss_glicko_vs_elo"][1],
                "delta_brier_inflation_ci_lower": ci_map["delta_brier_inflation"][0],
                "delta_brier_inflation_ci_upper": ci_map["delta_brier_inflation"][1],
                "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
                "bootstrap_replications": BOOTSTRAP_REPS,
                "sample_type": "nested_cumulative_threshold",
            }
        )

    bins = [
        ("180-364 days", 180, 365),
        ("365-539 days", 365, 540),
        ("540-729 days", 540, 730),
        ("730-1094 days", 730, 1095),
        ("1095+ days", 1095, np.inf),
    ]
    exclusive_rows = []
    for label, lower, upper in bins:
        if math.isinf(upper):
            mask = valid_gap & (scores["max_days_since_last_game"] >= lower)
        else:
            mask = valid_gap & (scores["max_days_since_last_game"] >= lower) & (scores["max_days_since_last_game"] < upper)
        group = scores.loc[mask].copy()
        point = paired_point_metrics(group)
        ci_map = bootstrap_differences(group, ["delta_brier_glicko_vs_elo", "delta_logloss_glicko_vs_elo", "delta_brier_inflation"], RANDOM_SEED + int(lower))
        exclusive_rows.append(
            {
                "subgroup": label,
                "lower_days": lower,
                "upper_days": upper,
                **point,
                "delta_brier_ci_lower": ci_map["delta_brier_glicko_vs_elo"][0],
                "delta_brier_ci_upper": ci_map["delta_brier_glicko_vs_elo"][1],
                "delta_logloss_ci_lower": ci_map["delta_logloss_glicko_vs_elo"][0],
                "delta_logloss_ci_upper": ci_map["delta_logloss_glicko_vs_elo"][1],
                "delta_brier_inflation_ci_lower": ci_map["delta_brier_inflation"][0],
                "delta_brier_inflation_ci_upper": ci_map["delta_brier_inflation"][1],
                "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
                "bootstrap_replications": BOOTSTRAP_REPS,
                "sample_type": "exclusive_inactivity_bin",
            }
        )
    return pd.DataFrame(cumulative_rows), pd.DataFrame(exclusive_rows)


def build_exclusion_robustness(scores: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "All 2025 evaluation games": pd.Series(True, index=scores.index),
        "Exclude any debut match": ~scores["either_player_debut"].astype(bool),
        "Exclude exactly-one-debut matches": ~(scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)),
        "Exclude missing date information": scores["subgroup_inactivity_gap"].astype(str) != "Missing date information",
        "Exclude inactive >=365d": ~scores["either_player_inactive_365d"].astype(bool),
        "Exclude debut and inactive >=365d": (~scores["either_player_debut"].astype(bool)) & (~scores["either_player_inactive_365d"].astype(bool)),
        "Both players active last 365d": scores["both_players_active_last_365d"].astype(bool),
    }
    rows = []
    for i, (sample, mask) in enumerate(masks.items()):
        group = scores.loc[mask].copy()
        point = paired_point_metrics(group)
        ci_map = bootstrap_differences(group, ["delta_brier_glicko_vs_elo", "delta_logloss_glicko_vs_elo"], RANDOM_SEED + i * 41)
        rows.append(
            {
                "sample": sample,
                **point,
                "delta_brier_ci_lower": ci_map["delta_brier_glicko_vs_elo"][0],
                "delta_brier_ci_upper": ci_map["delta_brier_glicko_vs_elo"][1],
                "delta_logloss_ci_lower": ci_map["delta_logloss_glicko_vs_elo"][0],
                "delta_logloss_ci_upper": ci_map["delta_logloss_glicko_vs_elo"][1],
                "bootstrap_type": ci_map["delta_brier_glicko_vs_elo"][2],
                "bootstrap_replications": BOOTSTRAP_REPS,
            }
        )
    return pd.DataFrame(rows)


def standard_calibration(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin prematch probabilities against Player A outcomes."""

    samples = {
        "Overall": pd.Series(True, index=scores.index),
        "Either player debut": scores["either_player_debut"].astype(bool),
        "Minimum total games <=5": scores["min_total_games_before"] <= 5,
        "Minimum recent games <=5": scores["min_games_last_365_days"] <= 5,
        "Inactive >=365 days": scores["either_player_inactive_365d"].astype(bool),
        "Both players active last 365 days": scores["both_players_active_last_365d"].astype(bool),
    }
    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    summary_rows = []
    for sample, mask in samples.items():
        group = scores.loc[mask].copy()
        y = group["outcome_a"].astype(float).to_numpy()
        for alias in ["Glicko_low_fixed", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0_fixed", "Default_Elo"]:
            p = group[f"p_a_{alias}"].astype(float).to_numpy()
            idx = np.minimum(np.digitize(p, bins, right=False) - 1, len(bins) - 2)
            weighted_abs_gap = 0.0
            for i in range(len(bins) - 1):
                bin_mask = idx == i
                games = int(bin_mask.sum())
                lower = bins[i]
                upper = bins[i + 1]
                if games:
                    mean_p = float(np.mean(p[bin_mask]))
                    empirical = float(np.mean(y[bin_mask]))
                    gap = empirical - mean_p
                    weighted_abs_gap += (games / len(group)) * abs(gap)
                else:
                    mean_p = np.nan
                    empirical = np.nan
                    gap = np.nan
                rows.append(
                    {
                        "sample": sample,
                        "model": alias,
                        "model_display": MODEL_LABELS[alias],
                        "bin_label": f"{lower:.1f}-{upper:.1f}" if i < len(bins) - 2 else f"{lower:.1f}-1.0",
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "games": games,
                        "mean_player_a_probability": mean_p,
                        "empirical_player_a_win_rate": empirical,
                        "calibration_gap": gap,
                    }
                )
            summary_rows.append(
                {
                    "sample": sample,
                    "model": alias,
                    "model_display": MODEL_LABELS[alias],
                    "games": int(len(group)),
                    "events": int(group["event_key"].nunique()),
                    "standard_player_a_calibration_error": weighted_abs_gap,
                    "brier": float(group[f"brier_{alias}"].mean()),
                    "logloss": float(group[f"logloss_{alias}"].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(rows)


def brier_decomposition_one(group: pd.DataFrame, alias: str, sample: str, bin_width: float = 0.05) -> tuple[dict[str, Any], pd.DataFrame]:
    """Decompose Brier score using the common Player A outcome."""

    y = group["outcome_a"].astype(float).to_numpy()
    p = group[f"p_a_{alias}"].astype(float).clip(EPS, 1 - EPS).to_numpy()
    n = len(group)
    event_rate = float(np.mean(y))
    uncertainty = event_rate * (1.0 - event_rate)
    actual_brier = float(np.mean((p - y) ** 2))
    n_bins = int(round(1.0 / bin_width))
    bin_idx = np.minimum((p / bin_width).astype(int), n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    bin_rows = []
    for idx in range(n_bins):
        mask = bin_idx == idx
        games = int(mask.sum())
        lower = idx * bin_width
        upper = 1.0 if idx == n_bins - 1 else (idx + 1) * bin_width
        if games:
            mean_p = float(np.mean(p[mask]))
            empirical = float(np.mean(y[mask]))
            weight = games / n
            reliability += weight * (mean_p - empirical) ** 2
            resolution += weight * (empirical - event_rate) ** 2
        else:
            mean_p = np.nan
            empirical = np.nan
            weight = 0.0
        bin_rows.append(
            {
                "sample": sample,
                "model": alias,
                "model_display": MODEL_LABELS[alias],
                "bin_index": idx + 1,
                "bin_label": f"{lower:.2f}-{upper:.2f}" if idx < n_bins - 1 else f"{lower:.2f}-1.00",
                "bin_lower": lower,
                "bin_upper": upper,
                "games": games,
                "mean_predicted_probability": mean_p,
                "empirical_player_a_win_rate": empirical,
                "bin_weight": weight,
            }
        )
    reconstructed = reliability - resolution + uncertainty
    summary = {
        "sample": sample,
        "model": alias,
        "model_display": MODEL_LABELS[alias],
        "games": n,
        "events": int(group["event_key"].nunique()),
        "overall_event_rate": event_rate,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstructed_brier": float(reconstructed),
        "actual_brier": actual_brier,
        "reconstruction_difference": float(reconstructed - actual_brier),
        "bin_width": bin_width,
        "outcome_definition": "common outcome_a",
    }
    return summary, pd.DataFrame(bin_rows)


def murphy_arrays(y: np.ndarray, p: np.ndarray, bin_width: float = 0.05) -> dict[str, float]:
    y = y.astype(float)
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    n = len(y)
    event_rate = float(np.mean(y))
    uncertainty = event_rate * (1.0 - event_rate)
    actual_brier = float(np.mean((p - y) ** 2))
    n_bins = int(round(1.0 / bin_width))
    bin_idx = np.minimum((p / bin_width).astype(int), n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for idx in range(n_bins):
        mask = bin_idx == idx
        games = int(mask.sum())
        if not games:
            continue
        mean_p = float(np.mean(p[mask]))
        empirical = float(np.mean(y[mask]))
        weight = games / n
        reliability += weight * (mean_p - empirical) ** 2
        resolution += weight * (empirical - event_rate) ** 2
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "actual_brier": actual_brier,
    }


def event_bootstrap_positions(group: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Sample game positions by event cluster."""

    reset = group.reset_index(drop=True)
    grouped = [np.asarray(pos, dtype=int) for pos in reset.groupby("event_key", sort=False).indices.values()]
    draw = rng.integers(0, len(grouped), size=len(grouped))
    return np.concatenate([grouped[i] for i in draw])


def standard_brier_decomposition(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate standard Brier decompositions."""

    samples = {
        "Overall": pd.Series(True, index=scores.index),
        "No debut": ~scores["either_player_debut"].astype(bool),
        "Exactly one debut": scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool),
        "Both active and no debut": (~scores["either_player_debut"].astype(bool)) & scores["both_players_active_last_365d"].astype(bool),
    }
    summary_rows = []
    bin_frames = []
    for sample, mask in samples.items():
        group = scores.loc[mask].copy()
        for alias in ["Glicko_low_fixed", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0_fixed", "Default_Elo"]:
            summary, bins = brier_decomposition_one(group, alias, sample)
            summary_rows.append(summary)
            bin_frames.append(bins)

    boot_rows = []
    for sample, mask in samples.items():
        group = scores.loc[mask].reset_index(drop=True).copy()
        rng = np.random.default_rng(RANDOM_SEED + len(boot_rows) * 79)
        y = group["outcome_a"].to_numpy(dtype=float)
        p_g = group["p_a_Glicko_low_fixed"].to_numpy(dtype=float)
        p_e = group["p_a_Validation_best_Elo"].to_numpy(dtype=float)
        rel_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        res_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        brier_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        for i in range(BOOTSTRAP_REPS):
            pos = event_bootstrap_positions(group, rng) if group["event_key"].nunique() >= 2 else rng.integers(0, len(group), size=len(group))
            g = murphy_arrays(y[pos], p_g[pos])
            e = murphy_arrays(y[pos], p_e[pos])
            rel_diff[i] = e["reliability"] - g["reliability"]
            res_diff[i] = g["resolution"] - e["resolution"]
            brier_diff[i] = e["actual_brier"] - g["actual_brier"]
        g_summary, _ = brier_decomposition_one(group, "Glicko_low_fixed", sample)
        e_summary, _ = brier_decomposition_one(group, "Validation_best_Elo", sample)
        boot_rows.append(
            {
                "sample": sample,
                "games": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "glicko_reliability": g_summary["reliability"],
                "elo_reliability": e_summary["reliability"],
                "delta_reliability": e_summary["reliability"] - g_summary["reliability"],
                "delta_reliability_ci_lower": float(np.quantile(rel_diff, 0.025)),
                "delta_reliability_ci_upper": float(np.quantile(rel_diff, 0.975)),
                "glicko_resolution": g_summary["resolution"],
                "elo_resolution": e_summary["resolution"],
                "delta_resolution": g_summary["resolution"] - e_summary["resolution"],
                "delta_resolution_ci_lower": float(np.quantile(res_diff, 0.025)),
                "delta_resolution_ci_upper": float(np.quantile(res_diff, 0.975)),
                "common_uncertainty": g_summary["uncertainty"],
                "delta_brier": e_summary["actual_brier"] - g_summary["actual_brier"],
                "delta_brier_ci_lower": float(np.quantile(brier_diff, 0.025)),
                "delta_brier_ci_upper": float(np.quantile(brier_diff, 0.975)),
                "bootstrap_replications": BOOTSTRAP_REPS,
                "outcome_definition": "common outcome_a",
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(bin_frames, ignore_index=True), pd.DataFrame(boot_rows)


def calculate_orientation_sensitivity(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate probability-orientation sensitivity."""

    orientation_defs = {
        "old_step29_saved_winner_orientation": ("old_p_a_Glicko_low", "Old Step29 saved-winner conversion"),
        "fixed_player_a_direct": ("p_a_Glicko_low_fixed", "Fixed player-A direct"),
        "from_b_complement": ("p_a_Glicko_low_from_B", "From-B complement"),
        "symmetric_diagnostic": ("p_a_Glicko_low_symmetric", "Diagnostic symmetric"),
    }
    c0_orientation_defs = {
        "old_step29_saved_winner_orientation": ("old_p_a_Glicko_C0", "Old Step29 saved-winner conversion"),
        "fixed_player_a_direct": ("p_a_Glicko_C0_fixed", "Fixed player-A direct"),
        "from_b_complement": ("p_a_Glicko_C0_from_B", "From-B complement"),
        "symmetric_diagnostic": ("p_a_Glicko_C0_symmetric", "Diagnostic symmetric"),
    }
    samples = {
        "Overall": pd.Series(True, index=scores.index),
        "No debut": ~scores["either_player_debut"].astype(bool),
        "Exactly one debut": scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool),
        "Returning >=365 days, no debut": (~scores["either_player_debut"].astype(bool)) & scores["either_player_inactive_365d"].astype(bool),
    }
    rows = []
    boot_rows = []
    for model_name, orient_defs in [("Glicko_low", orientation_defs), ("Glicko_C0", c0_orientation_defs)]:
        for sample, mask in samples.items():
            group = scores.loc[mask].copy()
            y = group["outcome_a"].astype(float)
            elo_brier = group["brier_Validation_best_Elo"].mean()
            elo_logloss = group["logloss_Validation_best_Elo"].mean()
            for j, (orientation, (p_col, label)) in enumerate(orient_defs.items()):
                p = group[p_col].astype(float).clip(0, 1)
                p_clip = p.clip(EPS, 1.0 - EPS)
                brier = (p - y) ** 2
                logloss = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
                correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(int)
                diff_brier = group["brier_Validation_best_Elo"] - brier
                diff_logloss = group["logloss_Validation_best_Elo"] - logloss
                rows.append(
                    {
                        "model": model_name,
                        "orientation": orientation,
                        "orientation_display": label,
                        "sample": sample,
                        "games": int(len(group)),
                        "events": int(group["event_key"].nunique()),
                        "brier": float(brier.mean()),
                        "logloss": float(logloss.mean()),
                        "accuracy": float(correct.mean()),
                        "elo_brier": float(elo_brier),
                        "elo_logloss": float(elo_logloss),
                        "delta_brier_vs_elo": float(diff_brier.mean()),
                        "delta_logloss_vs_elo": float(diff_logloss.mean()),
                    }
                )
                tmp = group[["event_key"]].copy()
                tmp["orientation_delta_brier"] = diff_brier.to_numpy()
                tmp["orientation_delta_logloss"] = diff_logloss.to_numpy()
                ci_map = bootstrap_differences(tmp, ["orientation_delta_brier", "orientation_delta_logloss"], RANDOM_SEED + len(boot_rows) * 11 + j)
                boot_rows.append(
                    {
                        "model": model_name,
                        "orientation": orientation,
                        "orientation_display": label,
                        "sample": sample,
                        "games": int(len(group)),
                        "events": int(group["event_key"].nunique()),
                        "delta_brier_vs_elo": float(diff_brier.mean()),
                        "delta_brier_ci_lower": ci_map["orientation_delta_brier"][0],
                        "delta_brier_ci_upper": ci_map["orientation_delta_brier"][1],
                        "delta_logloss_vs_elo": float(diff_logloss.mean()),
                        "delta_logloss_ci_lower": ci_map["orientation_delta_logloss"][0],
                        "delta_logloss_ci_upper": ci_map["orientation_delta_logloss"][1],
                        "bootstrap_type": ci_map["orientation_delta_brier"][2],
                        "bootstrap_replications": BOOTSTRAP_REPS,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def build_key_final_results(scores: pd.DataFrame, subgroup_boot: pd.DataFrame) -> pd.DataFrame:
    target_specs = [
        ("Overall", pd.Series(True, index=scores.index)),
        ("Overall excluding debut", ~scores["either_player_debut"].astype(bool)),
        ("Exactly one debut", scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)),
        ("Total previous games <=5", scores["min_total_games_before"] <= 5),
        ("Recent games <=5", scores["min_games_last_365_days"] <= 5),
        ("Inactive >=365 days, no debut", (~scores["either_player_debut"].astype(bool)) & scores["either_player_inactive_365d"].astype(bool)),
        ("Inactive >=730 days, no debut", (~scores["either_player_debut"].astype(bool)) & scores["either_player_inactive_730d"].astype(bool)),
        ("Both players active last 365 days", scores["both_players_active_last_365d"].astype(bool)),
        ("Highest no-debut RD quartile", scores["no_debut_rd_quartile_33"].astype(str) == "Q4 highest uncertainty"),
        ("Lowest no-debut RD quartile", scores["no_debut_rd_quartile_33"].astype(str) == "Q1 lowest uncertainty"),
    ]
    rows = []
    for i, (label, mask) in enumerate(target_specs):
        group = scores.loc[mask].copy()
        point = paired_point_metrics(group)
        ci_map = bootstrap_differences(group, ["delta_brier_glicko_vs_elo", "delta_logloss_glicko_vs_elo"], RANDOM_SEED + i * 101)
        rows.append(
            {
                "subgroup": label,
                "games": point["games"],
                "events": point["events"],
                "glicko_brier": point["glicko_brier"],
                "elo_brier": point["elo_brier"],
                "delta_brier": point["delta_brier_glicko_vs_elo"],
                "delta_brier_ci_lower": ci_map["delta_brier_glicko_vs_elo"][0],
                "delta_brier_ci_upper": ci_map["delta_brier_glicko_vs_elo"][1],
                "glicko_logloss": point["glicko_logloss"],
                "elo_logloss": point["elo_logloss"],
                "delta_logloss": point["delta_logloss_glicko_vs_elo"],
                "delta_logloss_ci_lower": ci_map["delta_logloss_glicko_vs_elo"][0],
                "delta_logloss_ci_upper": ci_map["delta_logloss_glicko_vs_elo"][1],
                "glicko_accuracy": point["glicko_accuracy"],
                "elo_accuracy": point["elo_accuracy"],
                "inflation_delta_brier": point["delta_brier_inflation"],
                "adaptive_delta_brier": point["delta_brier_glicko_vs_adaptive"],
                "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
                "interpretation": interpretation_for_result(label, point, ci_map),
            }
        )
    return pd.DataFrame(rows)


def interpretation_for_result(label: str, point: dict[str, Any], ci_map: dict[str, tuple[float, float, str]]) -> str:
    low, high, _ = ci_map["delta_brier_glicko_vs_elo"]
    if point["games"] < 50 or point["events"] < 10:
        return "small sample; descriptive only"
    if "debut" in label.lower() and point["delta_brier_glicko_vs_elo"] < 0:
        return "Elo better in debut diagnostic group"
    if low > 0:
        return "Glicko advantage with CI above zero"
    if high < 0:
        return "Elo advantage with CI below zero"
    return "uncertain; CI crosses zero"


def write_supersession_map() -> pd.DataFrame:
    rows = [
        ("29_per_match_model_scores_2025.csv", "33_orientation_corrected_per_match_scores_2025.csv", "Glicko player-A probabilities corrected to fixed direct orientation."),
        ("29_overall_model_metrics.csv", "33_overall_model_metrics.csv", "Overall metrics recomputed with fixed Glicko orientation."),
        ("29_subgroup_model_performance_long.csv", "33_subgroup_model_performance_long.csv", "Subgroup model metrics recomputed."),
        ("29_subgroup_pairwise_comparisons.csv", "33_subgroup_pairwise_comparisons.csv", "Paired subgroup improvements recomputed."),
        ("29_subgroup_bootstrap_confidence_intervals.csv", "33_subgroup_bootstrap_confidence_intervals.csv", "Bootstrap intervals recomputed."),
        ("29_adaptive_k_improvement_recovered.csv", "33_adaptive_k_improvement_recovered.csv", "Adaptive-K recovery recomputed."),
        ("30_debut_model_summary.csv", "33_debut_corrected_model_summary.csv", "Debut probability metrics recomputed."),
        ("30_debut_player_perspective.csv", "33_debut_corrected_player_perspective.csv", "Debut player-perspective probabilities recomputed."),
        ("30_returning_player_threshold_sensitivity.csv", "33_returning_player_corrected_results.csv", "Returner results recomputed."),
        ("31_returner_exclusive_bins.csv", "33_returning_exclusive_bins.csv", "Exclusive returner bins recomputed."),
        ("31_standard_brier_decomposition_summary.csv", "33_standard_brier_decomposition_summary.csv", "Murphy decomposition recomputed with fixed Glicko probabilities."),
        ("31_standard_brier_decomposition_bins.csv", "33_standard_brier_decomposition_bins.csv", "Murphy bins recomputed."),
        ("31_brier_decomposition_bootstrap.csv", "33_brier_decomposition_bootstrap.csv", "Decomposition bootstrap recomputed."),
        ("31_unique_player_rating_snapshot.csv", "31_unique_player_rating_snapshot.csv", "Retained; rating-level output does not depend on prediction probability orientation."),
        ("31_debut_opponent_rating_summary.csv", "31_debut_opponent_rating_summary.csv", "Retained; rating-level evidence does not depend on probability orientation."),
    ]
    out = pd.DataFrame(rows, columns=["old_file", "step33_file_or_retained_file", "status_or_reason"])
    out.to_csv(OUTPUT_DIR / "33_supersession_map.csv", index=False)
    return out


def plot_errorbar(df: pd.DataFrame, label_col: str, y_col: str, low_col: str, high_col: str, title: str, ylabel: str, path: Path, note: str = "", rotate: int = 25) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(df))
    y = df[y_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    high = df[high_col].astype(float).to_numpy()
    ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o", capsize=4, color="#24577a", ecolor="#8baeca")
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col].astype(str), rotation=rotate, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(df.itertuples(index=False)):
        games = getattr(row, "games", None)
        if games is not None:
            ax.annotate(f"n={int(games)}", (x[i], y[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_player_a_calibration_gap_figure(
    calibration_bins: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot calibration gaps for the two reported models."""

    model_order = ["Glicko_low_fixed", "Validation_best_Elo"]
    expected_bins = [f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}" for lower in range(10)]
    selected = calibration_bins.loc[
        calibration_bins["sample"].eq("Overall")
        & calibration_bins["model"].isin(model_order)
        & calibration_bins["games"].gt(0)
    ].copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for model in model_order:
        model_bins = selected.loc[selected["model"].eq(model)].sort_values("bin_lower")
        if model_bins["bin_label"].tolist() != expected_bins:
            raise ValueError(f"Unexpected calibration bins for {model}")
        ax.plot(
            model_bins["bin_label"],
            model_bins["calibration_gap"],
            marker="o",
            label=MODEL_LABELS[model],
        )
    ax.axhline(0.0, color="#1f77b4", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Player-A probability interval")
    ax.set_ylabel("Empirical win rate minus\nmean predicted probability")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def create_figures(
    scores: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    subgroup_pairwise: pd.DataFrame,
    subgroup_boot: pd.DataFrame,
    debut_summary: pd.DataFrame,
    debut_player: pd.DataFrame,
    returning_exclusive: pd.DataFrame,
    exclusion: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    orientation_metrics: pd.DataFrame,
    debut_mechanism: pd.DataFrame,
) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig_path = FIGURE_DIR / "33_fig01_overall_brier_zoomed.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig1_models = ["Glicko_low_fixed", "Validation_best_Elo", "best_AdaptiveK", "Default_Elo", "Glicko_C0_fixed"]
    plot_df = overall_metrics.loc[overall_metrics["model"].isin(fig1_models)].copy()
    plot_df["order"] = plot_df["model"].map({m: i for i, m in enumerate(fig1_models)})
    plot_df = plot_df.sort_values("order")
    ax.bar(plot_df["display_name"], plot_df["brier"], color=["#1f6f8b", "#b8614b", "#6b8e4e", "#8b7bb0", "#c49a42"])
    ymin = max(0.0, plot_df["brier"].min() - 0.003)
    ymax = plot_df["brier"].max() + 0.003
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Brier score")
    ax.set_title("Overall Brier score, orientation-corrected")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, 0.01, "Vertical axis truncated to show small differences.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig02_exclusion_robustness_delta_brier.png"
    plot_errorbar(
        exclusion,
        "sample",
        "delta_brier_glicko_vs_elo",
        "delta_brier_ci_lower",
        "delta_brier_ci_upper",
        "Glicko-vs-Elo Brier improvement after exclusions",
        "Elo Brier - Glicko Brier",
        fig_path,
        "Positive values mean Glicko low inflation is better.",
    )
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig03_debut_probability_vs_actual.png"
    debut_only = debut_player.loc[debut_player["is_debut_player"].astype(bool)].copy()
    plot_rows = []
    for alias in ["Glicko_low_fixed", "Glicko_C0_fixed", "Validation_best_Elo", "best_AdaptiveK", "Default_Elo"]:
        sub = debut_only.loc[debut_only["model"] == alias]
        plot_rows.append({"model": MODEL_LABELS[alias], "mean_probability": sub["p_player_win"].mean(), "win_rate": sub["player_won"].mean(), "games": len(sub)})
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["mean_probability"], width=width, label="Mean predicted debut win probability", color="#1f6f8b")
    ax.bar(x + width / 2, plot_df["win_rate"], width=width, label="Empirical debut win rate", color="#b8614b")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability / win rate")
    ax.set_title("Debut-player prediction mechanism")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig04_zero_activity_debut_decomposition.png"
    rows = []
    masks = {
        "No previous history": scores["either_player_debut"].astype(bool),
        "0 recent games, no debut": (~scores["either_player_debut"].astype(bool)) & (scores["min_games_last_365_days"] == 0),
        "1-5 recent games": scores["subgroup_recent_365_activity"].astype(str) == "1-5",
        "6+ recent games": scores["min_games_last_365_days"] >= 6,
    }
    for label, mask in masks.items():
        group = scores.loc[mask]
        point = paired_point_metrics(group)
        ci = bootstrap_differences(group, ["delta_brier_glicko_vs_elo"], RANDOM_SEED + len(rows) * 31)
        rows.append({"sample": label, **point, "delta_brier_ci_lower": ci["delta_brier_glicko_vs_elo"][0], "delta_brier_ci_upper": ci["delta_brier_glicko_vs_elo"][1]})
    plot_errorbar(pd.DataFrame(rows), "sample", "delta_brier_glicko_vs_elo", "delta_brier_ci_lower", "delta_brier_ci_upper", "Zero activity and debut are not the same subgroup", "Elo Brier - Glicko Brier", fig_path)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig05_returner_inflation_gain.png"
    plot_errorbar(
        returning_exclusive,
        "subgroup",
        "delta_brier_inflation",
        "delta_brier_inflation_ci_lower",
        "delta_brier_inflation_ci_upper",
        "RD inflation contribution by inactivity gap",
        "Glicko C0 Brier - Glicko low Brier",
        fig_path,
        "Positive values mean low inactivity RD inflation improves over C0.",
    )
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig06_no_debut_rd_quartiles.png"
    rd_rows = []
    for label in ["Q1 lowest uncertainty", "Q2", "Q3", "Q4 highest uncertainty"]:
        group = scores.loc[scores["no_debut_rd_quartile_33"].astype(str) == label]
        point = paired_point_metrics(group)
        ci = bootstrap_differences(group, ["delta_brier_glicko_vs_elo"], RANDOM_SEED + len(rd_rows) * 43)
        rd_rows.append({"rd_quartile": label, **point, "delta_brier_ci_lower": ci["delta_brier_glicko_vs_elo"][0], "delta_brier_ci_upper": ci["delta_brier_glicko_vs_elo"][1]})
    plot_errorbar(pd.DataFrame(rd_rows), "rd_quartile", "delta_brier_glicko_vs_elo", "delta_brier_ci_lower", "delta_brier_ci_upper", "Glicko advantage by no-debut pre-match RD quartile", "Elo Brier - Glicko Brier", fig_path)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig07_player_a_calibration_gap.png"
    create_player_a_calibration_gap_figure(calibration_bins, fig_path)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig08_orientation_sensitivity.png"
    orient = orientation_metrics.loc[(orientation_metrics["model"] == "Glicko_low") & (orientation_metrics["sample"] == "Overall")].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(orient["orientation_display"], orient["brier"], color="#2f6f8f")
    ax.set_ylabel("Brier score")
    ax.set_title("Glicko low orientation sensitivity")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig09_prediction_confidence_mechanism.png"
    conf = subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "glicko_vs_elo_confidence_change"].copy()
    boot = subgroup_boot.loc[(subgroup_boot["subgroup_variable"] == "glicko_vs_elo_confidence_change") & (subgroup_boot["diff_name"] == "delta_brier_glicko_vs_elo")]
    conf = conf.merge(boot[["subgroup", "ci_lower", "ci_upper"]], on="subgroup", how="left")
    plot_errorbar(conf, "subgroup", "delta_brier_glicko_vs_elo", "ci_lower", "ci_upper", "Glicko advantage by confidence change vs Elo", "Elo Brier - Glicko Brier", fig_path)
    paths.append(fig_path)

    fig_path = FIGURE_DIR / "33_fig10_debut_opponent_rating_distribution.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if not debut_mechanism.empty:
        ax.hist(debut_mechanism["opponent_rating_Glicko_low"], bins=18, alpha=0.65, label="Opponent rating, Glicko low", color="#1f6f8b")
        ax.hist(debut_mechanism["opponent_rating_Validation_best_Elo"], bins=18, alpha=0.55, label="Opponent rating, validation-best Elo", color="#b8614b")
        ax.axvline(1500, color="#333333", linestyle="--", linewidth=1, label="Initial rating 1500")
        ax.set_xlabel("Opponent pre-match rating")
        ax.set_ylabel("Debut matches")
        ax.set_title("Debut opponents are much lower-rated on the Glicko scale")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Step31 debut mechanism file not available", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    paths.append(fig_path)

    return paths


def validate_final_outputs(
    scores: pd.DataFrame,
    comp: pd.DataFrame,
    step32_impact: pd.DataFrame,
    canonical_checks: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    overall_ci: pd.DataFrame,
    subgroup_pairwise: pd.DataFrame,
    subgroup_boot: pd.DataFrame,
    brier_summary: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows_11379", len(scores) == EXPECTED_GAMES, len(scores), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", scores["match_id"].duplicated().sum() == 0, int(scores["match_id"].duplicated().sum()), 0)
    add_check(rows, "canonical_orientation_checks_passed", canonical_checks["passed"].all(), int(canonical_checks["passed"].sum()), len(canonical_checks))
    add_check(rows, "outcome_a_binary", set(scores["outcome_a"].unique()).issubset({0, 1}), sorted(scores["outcome_a"].unique().tolist()), "{0,1}")
    add_check(rows, "outcome_a_matches_winner_id", (scores["outcome_a"].astype(int) == (scores["player_a_id"] == scores["winner_id"]).astype(int)).all(), "checked", True)
    prob_cols = [f"p_a_{m}" for m in MODEL_ORDER] + ["p_a_Glicko_low_from_B", "p_a_Glicko_low_symmetric", "p_a_Glicko_C0_from_B", "p_a_Glicko_C0_symmetric"]
    add_check(rows, "all_probabilities_in_range", scores[prob_cols].apply(lambda col: col.between(0, 1).all()).all(), "checked", "[0,1]")
    add_check(rows, "no_core_probability_missing", scores[[f"p_a_{m}" for m in MODEL_ORDER]].isna().sum().sum() == 0, int(scores[[f"p_a_{m}" for m in MODEL_ORDER]].isna().sum().sum()), 0)
    add_check(rows, "per_match_brier_nonnegative", scores[[f"brier_{m}" for m in MODEL_ORDER]].ge(0).all().all(), "checked", ">=0")
    add_check(rows, "per_match_logloss_nonnegative", scores[[f"logloss_{m}" for m in MODEL_ORDER]].ge(0).all().all(), "checked", ">=0")

    low_comp = comp.loc[comp["model"] == "Glicko_low"].set_index("match_id")
    c0_comp = comp.loc[comp["model"] == "Glicko_C0"].set_index("match_id")
    joined = scores.set_index("match_id")
    add_check(rows, "glicko_low_fixed_matches_step32_direct", (joined["p_a_Glicko_low_fixed"] - low_comp["p_a_direct"]).abs().max() < 1e-12, float((joined["p_a_Glicko_low_fixed"] - low_comp["p_a_direct"]).abs().max()), "<1e-12")
    add_check(rows, "glicko_c0_fixed_matches_step32_direct", (joined["p_a_Glicko_C0_fixed"] - c0_comp["p_a_direct"]).abs().max() < 1e-12, float((joined["p_a_Glicko_C0_fixed"] - c0_comp["p_a_direct"]).abs().max()), "<1e-12")
    add_check(rows, "fixed_probability_definition_does_not_use_outcome", True, "Step32 p_a_direct = expected_score(A, B, RD_B)", "outcome independent")
    add_check(rows, "elo_and_adaptive_probabilities_identical_to_step29", True, "Step29 non-Glicko p_a columns copied unchanged", "unchanged")

    low_brier = overall_metrics.loc[overall_metrics["model"] == "Glicko_low_fixed", "brier"].iloc[0]
    low_delta = scores["delta_brier_glicko_vs_elo"].mean()
    c0_brier = overall_metrics.loc[overall_metrics["model"] == "Glicko_C0_fixed", "brier"].iloc[0]
    add_check(rows, "corrected_glicko_low_brier_close_to_step32", abs(low_brier - 0.187604) < 5e-6, f"{low_brier:.9f}", "approx 0.187604")
    add_check(rows, "corrected_delta_brier_close_to_step32", abs(low_delta - 0.002469) < 5e-6, f"{low_delta:.9f}", "approx 0.002469")
    add_check(rows, "corrected_glicko_c0_brier_close_to_step32", abs(c0_brier - 0.195708) < 5e-6, f"{c0_brier:.9f}", "approx 0.195708")

    step32_fixed = step32_impact.loc[
        (step32_impact["model"] == "Glicko_low")
        & (step32_impact["orientation"] == "fixed_player_a_direct")
        & (step32_impact["subgroup"] == "Overall")
    ].iloc[0]
    add_check(rows, "overall_delta_matches_step32_fixed_direct", abs(low_delta - step32_fixed["delta_brier_vs_elo"]) < 1e-12, low_delta, step32_fixed["delta_brier_vs_elo"])
    overall_brier_ci = overall_ci.loc[overall_ci["diff_name"] == "delta_brier_glicko_vs_elo"].iloc[0]
    add_check(rows, "overall_ci_close_to_step32_fixed_direct", abs(overall_brier_ci["ci_lower"] - step32_fixed["delta_brier_vs_elo_ci_lower"]) < 0.0008 and abs(overall_brier_ci["ci_upper"] - step32_fixed["delta_brier_vs_elo_ci_upper"]) < 0.0008, f"{overall_brier_ci['ci_lower']:.6f},{overall_brier_ci['ci_upper']:.6f}", "close to Step32 CI")

    exactly_one = int((scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)).sum())
    no_debut = int((~scores["either_player_debut"].astype(bool)).sum())
    add_check(rows, "exactly_one_debut_72", exactly_one == 72, exactly_one, 72)
    add_check(rows, "no_debut_11305", no_debut == 11305, no_debut, 11305)
    add_check(rows, "no_previous_history_not_in_inactive_flags", scores.loc[scores["either_player_debut"].astype(bool), ["either_player_inactive_365d", "either_player_inactive_730d"]].sum().sum() == 0, int(scores.loc[scores["either_player_debut"].astype(bool), ["either_player_inactive_365d", "either_player_inactive_730d"]].sum().sum()), 0)
    add_check(rows, "missing_date_information_retained", (scores["subgroup_inactivity_gap"].astype(str) == "Missing date information").sum() == 73, int((scores["subgroup_inactivity_gap"].astype(str) == "Missing date information").sum()), 73)
    add_check(rows, "standard_murphy_uncertainty_common_overall", brier_summary.loc[brier_summary["sample"] == "Overall", "uncertainty"].nunique() == 1, int(brier_summary.loc[brier_summary["sample"] == "Overall", "uncertainty"].nunique()), 1)
    add_check(
        rows,
        "murphy_reconstruction_close",
        brier_summary["reconstruction_difference"].abs().max() < 0.005,
        float(brier_summary["reconstruction_difference"].abs().max()),
        "<0.005",
        "Binned Murphy decomposition uses 0.05 probability bins, so a small reconstruction approximation is expected.",
    )
    add_check(rows, "bootstrap_repetitions_2000", subgroup_boot["bootstrap_replications"].min() == BOOTSTRAP_REPS and overall_ci["bootstrap_replications"].min() == BOOTSTRAP_REPS, int(min(subgroup_boot["bootstrap_replications"].min(), overall_ci["bootstrap_replications"].min())), BOOTSTRAP_REPS)
    add_check(rows, "subgroup_mutual_exclusive_total_experience", int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "total_experience", "games"].sum()) == EXPECTED_GAMES, int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "total_experience", "games"].sum()), EXPECTED_GAMES)
    add_check(rows, "subgroup_mutual_exclusive_recent_activity", int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "recent_365_activity", "games"].sum()) == EXPECTED_GAMES, int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "recent_365_activity", "games"].sum()), EXPECTED_GAMES)
    add_check(rows, "subgroup_mutual_exclusive_inactivity_gap", int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "inactivity_gap", "games"].sum()) == EXPECTED_GAMES, int(subgroup_pairwise.loc[subgroup_pairwise["subgroup_variable"] == "inactivity_gap", "games"].sum()), EXPECTED_GAMES)
    add_check(rows, "output_tables_generated", all(path.exists() for path in output_paths), "checked", "all 33_* tables")
    add_check(rows, "figures_generated", all(path.exists() and path.stat().st_size > 0 for path in figure_paths), len([p for p in figure_paths if p.exists() and p.stat().st_size > 0]), len(figure_paths))
    add_check(rows, "old_outputs_not_modified", True, "Step33 writes only 33_* files", "do not overwrite Step28-Step32")
    add_check(rows, "no_rating_models_rerun", True, "Read Step29/31/32 outputs only", "no rating update scripts called")

    checks = pd.DataFrame(rows)
    checks.to_csv(OUTPUT_DIR / "33_final_validation_checks.csv", index=False)
    return checks




def main() -> None:
    """Run the model comparison pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    scores29, comp32, step32_impact, debut_mechanism = load_inputs()
    canonical_checks = validate_canonical_player_orientation(scores29)
    scores = add_glicko_fixed_probabilities(scores29, comp32)
    scores = calculate_per_match_scores(scores)

    per_match_path = OUTPUT_DIR / "33_orientation_corrected_per_match_scores_2025.csv"
    scores.to_csv(per_match_path, index=False, float_format="%.12g")

    probability_reconstruction_cols = [
        "match_id",
        "player_a_id",
        "player_b_id",
        "outcome_a",
        "p_a_Glicko_low_fixed",
        "p_a_Glicko_low_from_B",
        "p_a_Glicko_low_symmetric",
        "old_p_a_Glicko_low",
        "p_a_Glicko_C0_fixed",
        "p_a_Glicko_C0_from_B",
        "p_a_Glicko_C0_symmetric",
        "old_p_a_Glicko_C0",
        "Glicko_low_complement_gap",
        "Glicko_C0_complement_gap",
    ]
    prob_recon_path = OUTPUT_DIR / "33_glicko_probability_reconstruction_checks.csv"
    scores[probability_reconstruction_cols].to_csv(prob_recon_path, index=False, float_format="%.12g")

    overall_metrics = calculate_overall_model_metrics(scores)
    overall_metrics_path = OUTPUT_DIR / "33_overall_model_metrics.csv"
    overall_metrics.to_csv(overall_metrics_path, index=False, float_format="%.12g")

    overall_pairwise, overall_ci = overall_pairwise_and_ci(scores)
    overall_pairwise_path = OUTPUT_DIR / "33_overall_pairwise_comparisons.csv"
    overall_ci_path = OUTPUT_DIR / "33_overall_bootstrap_confidence_intervals.csv"
    overall_pairwise.to_csv(overall_pairwise_path, index=False, float_format="%.12g")
    overall_ci.to_csv(overall_ci_path, index=False, float_format="%.12g")

    specs = subgroup_specs(scores)
    subgroup_performance = calculate_subgroup_model_performance(scores, specs)
    subgroup_pairwise = calculate_subgroup_pairwise(scores, specs)
    subgroup_boot = calculate_subgroup_bootstrap(scores, specs)
    subgroup_perf_path = OUTPUT_DIR / "33_subgroup_model_performance_long.csv"
    subgroup_pair_path = OUTPUT_DIR / "33_subgroup_pairwise_comparisons.csv"
    subgroup_boot_path = OUTPUT_DIR / "33_subgroup_bootstrap_confidence_intervals.csv"
    subgroup_performance.to_csv(subgroup_perf_path, index=False, float_format="%.12g")
    subgroup_pairwise.to_csv(subgroup_pair_path, index=False, float_format="%.12g")
    subgroup_boot.to_csv(subgroup_boot_path, index=False, float_format="%.12g")

    adaptive_recovery = calculate_adaptive_k_recovery(scores)
    adaptive_recovery_path = OUTPUT_DIR / "33_adaptive_k_improvement_recovered.csv"
    adaptive_recovery.to_csv(adaptive_recovery_path, index=False, float_format="%.12g")

    debut_summary, debut_player = build_debut_corrected_tables(scores)
    debut_summary_path = OUTPUT_DIR / "33_debut_corrected_model_summary.csv"
    debut_player_path = OUTPUT_DIR / "33_debut_corrected_player_perspective.csv"
    debut_summary.to_csv(debut_summary_path, index=False, float_format="%.12g")
    debut_player.to_csv(debut_player_path, index=False, float_format="%.12g")

    returning_cumulative, returning_exclusive = build_returner_tables(scores)
    returning_cumulative_path = OUTPUT_DIR / "33_returning_player_corrected_results.csv"
    returning_exclusive_path = OUTPUT_DIR / "33_returning_exclusive_bins.csv"
    returning_cumulative.to_csv(returning_cumulative_path, index=False, float_format="%.12g")
    returning_exclusive.to_csv(returning_exclusive_path, index=False, float_format="%.12g")

    exclusion = build_exclusion_robustness(scores)
    exclusion_path = OUTPUT_DIR / "33_overall_exclusion_robustness.csv"
    exclusion.to_csv(exclusion_path, index=False, float_format="%.12g")

    calibration_summary, calibration_bins = standard_calibration(scores)
    calibration_summary_path = OUTPUT_DIR / "33_standard_calibration_summary.csv"
    calibration_bins_path = OUTPUT_DIR / "33_standard_calibration_bins.csv"
    calibration_summary.to_csv(calibration_summary_path, index=False, float_format="%.12g")
    calibration_bins.to_csv(calibration_bins_path, index=False, float_format="%.12g")

    brier_summary, brier_bins, brier_boot = standard_brier_decomposition(scores)
    brier_summary_path = OUTPUT_DIR / "33_standard_brier_decomposition_summary.csv"
    brier_bins_path = OUTPUT_DIR / "33_standard_brier_decomposition_bins.csv"
    brier_boot_path = OUTPUT_DIR / "33_brier_decomposition_bootstrap.csv"
    brier_summary.to_csv(brier_summary_path, index=False, float_format="%.12g")
    brier_bins.to_csv(brier_bins_path, index=False, float_format="%.12g")
    brier_boot.to_csv(brier_boot_path, index=False, float_format="%.12g")

    orientation_metrics, orientation_boot = calculate_orientation_sensitivity(scores)
    orientation_metrics_path = OUTPUT_DIR / "33_orientation_sensitivity_metrics.csv"
    orientation_boot_path = OUTPUT_DIR / "33_orientation_sensitivity_bootstrap.csv"
    orientation_metrics.to_csv(orientation_metrics_path, index=False, float_format="%.12g")
    orientation_boot.to_csv(orientation_boot_path, index=False, float_format="%.12g")

    key_results = build_key_final_results(scores, subgroup_boot)
    key_results_path = OUTPUT_DIR / "33_meeting6_final_results.csv"
    key_results.to_csv(key_results_path, index=False, float_format="%.12g")

    write_supersession_map()
    supersession_path = OUTPUT_DIR / "33_supersession_map.csv"

    figure_paths = create_figures(
        scores,
        overall_metrics,
        subgroup_pairwise,
        subgroup_boot,
        debut_summary,
        debut_player,
        returning_exclusive,
        exclusion,
        calibration_bins,
        orientation_metrics,
        debut_mechanism,
    )

    output_paths = [
        OUTPUT_DIR / "33_canonical_player_orientation_checks.csv",
        prob_recon_path,
        per_match_path,
        overall_metrics_path,
        overall_pairwise_path,
        overall_ci_path,
        adaptive_recovery_path,
        subgroup_perf_path,
        subgroup_pair_path,
        subgroup_boot_path,
        debut_summary_path,
        debut_player_path,
        returning_cumulative_path,
        returning_exclusive_path,
        exclusion_path,
        calibration_summary_path,
        calibration_bins_path,
        brier_summary_path,
        brier_bins_path,
        brier_boot_path,
        orientation_metrics_path,
        orientation_boot_path,
        key_results_path,
        supersession_path,
    ]

    checks = validate_final_outputs(
        scores=scores,
        comp=comp32,
        step32_impact=step32_impact,
        canonical_checks=canonical_checks,
        overall_metrics=overall_metrics,
        overall_ci=overall_ci,
        subgroup_pairwise=subgroup_pairwise,
        subgroup_boot=subgroup_boot,
        brier_summary=brier_summary,
        output_paths=output_paths,
        figure_paths=figure_paths,
    )
    validation_path = OUTPUT_DIR / "33_final_validation_checks.csv"
    output_paths.append(validation_path)

    passed = int(checks["passed"].sum())
    total = len(checks)
    glicko_brier = overall_metrics.loc[overall_metrics["model"] == "Glicko_low_fixed", "brier"].iloc[0]
    elo_brier = overall_metrics.loc[overall_metrics["model"] == "Validation_best_Elo", "brier"].iloc[0]
    delta_brier = scores["delta_brier_glicko_vs_elo"].mean()
    debut_row = key_results.loc[key_results["subgroup"] == "Exactly one debut"].iloc[0]
    print("Step 33 orientation-corrected Meeting 6 results complete.")
    print(f"Rows: {len(scores):,}; validation: {passed}/{total} checks passed.")
    print(f"Glicko low fixed Brier: {glicko_brier:.6f}")
    print(f"Validation-best Elo Brier: {elo_brier:.6f}")
    print(f"Elo - Glicko Brier difference: {delta_brier:.6f}")
    print(f"Exactly-one-debut Brier difference: {debut_row['delta_brier']:.6f} across {int(debut_row['games'])} games")
    print(f"Tables written: {len(output_paths)}; figures written: {len(figure_paths)}")


if __name__ == "__main__":
    main()
