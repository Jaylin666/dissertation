"""Meeting 6 step 2: analyse where Glicko helps.

This script merges leakage-safe pre-match features from step 28 with existing
meeting 5 prediction files. It does not rerun Elo, Glicko, or adaptive-K Elo.
All outputs are written to outputs/meeting6/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"
FIGURE_DIR = OUTPUT_DIR / "figures"

FEATURE_PATH = OUTPUT_DIR / "28_prematch_match_features_2025.csv"
FAIR_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko" / "meeting5_fair_elo_vs_glicko_predictions_2025.csv"
FAIR_METRICS_PATH = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko" / "meeting5_fair_elo_vs_glicko_metrics_2025.csv"
ADAPTIVE_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_predictions_2025.csv"
ADAPTIVE_METRICS_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_metrics_2025.csv"
ADAPTIVE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_summary.md"

ALIGNMENT_CHECKS_PATH = OUTPUT_DIR / "29_model_alignment_checks.csv"
PER_MATCH_SCORES_PATH = OUTPUT_DIR / "29_per_match_model_scores_2025.csv"
OVERALL_METRICS_PATH = OUTPUT_DIR / "29_overall_model_metrics.csv"
SUBGROUP_MODEL_PERFORMANCE_PATH = OUTPUT_DIR / "29_subgroup_model_performance_long.csv"
SUBGROUP_PAIRWISE_PATH = OUTPUT_DIR / "29_subgroup_pairwise_comparisons.csv"
BOOTSTRAP_CI_PATH = OUTPUT_DIR / "29_subgroup_bootstrap_confidence_intervals.csv"
CALIBRATION_SUMMARY_PATH = OUTPUT_DIR / "29_corrected_calibration_summary.csv"
CALIBRATION_BINS_PATH = OUTPUT_DIR / "29_corrected_calibration_bins.csv"
ADAPTIVE_RECOVERY_PATH = OUTPUT_DIR / "29_adaptive_k_improvement_recovered.csv"
KEY_RESULTS_PATH = OUTPUT_DIR / "29_key_meeting6_results.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "29_where_glicko_helps_validation_checks.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "29_where_glicko_helps_summary.md"
RD_CUTPOINTS_PATH = OUTPUT_DIR / "29_glicko_rd_quartile_cutpoints.csv"

EPS = 1e-15
EXPECTED_GAMES = 11_379
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20260713
METRIC_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    source_model: str
    display_name: str
    source: str
    probability_field: str
    family: str


MODEL_SPECS = [
    ModelSpec(
        alias="Glicko_low",
        source_model="Glicko_low_inflation_match_by_match",
        display_name="Glicko low inflation",
        source="fair",
        probability_field="pred_winner_win",
        family="Glicko",
    ),
    ModelSpec(
        alias="Validation_best_Elo",
        source_model="Validation_best_Elo",
        display_name="Validation-best Elo",
        source="fair",
        probability_field="pred_winner_win",
        family="Elo",
    ),
    ModelSpec(
        alias="Glicko_C0",
        source_model="Glicko_C0_match_by_match",
        display_name="Glicko C0",
        source="fair",
        probability_field="pred_winner_win",
        family="Glicko",
    ),
    ModelSpec(
        alias="Default_Elo",
        source_model="Default_Elo",
        display_name="Default Elo",
        source="fair",
        probability_field="pred_winner_win",
        family="Elo",
    ),
    ModelSpec(
        alias="Conservative_Elo",
        source_model="Conservative_Elo",
        display_name="Conservative Elo",
        source="fair",
        probability_field="pred_winner_win",
        family="Elo",
    ),
    ModelSpec(
        alias="best_AdaptiveK",
        source_model="AdaptiveK_PreviousYearGames_Elo_scale300",
        display_name="Best adaptive-K Elo",
        source="adaptive",
        probability_field="pred_actual_winner_win",
        family="Adaptive-K Elo",
    ),
]

MODEL_BY_ALIAS = {spec.alias: spec for spec in MODEL_SPECS}
DISPLAY_BY_ALIAS = {spec.alias: spec.display_name for spec in MODEL_SPECS}

CORE_ALIASES = [
    "Glicko_low",
    "Validation_best_Elo",
    "Glicko_C0",
    "Default_Elo",
    "best_AdaptiveK",
]

OVERALL_MODEL_ORDER = [
    "Glicko_low",
    "Validation_best_Elo",
    "best_AdaptiveK",
    "Default_Elo",
    "Glicko_C0",
    "Conservative_Elo",
]

PAIRWISE_DIFFS = [
    ("delta_brier_glicko_vs_elo", "Brier: Glicko low vs validation-best Elo"),
    ("delta_logloss_glicko_vs_elo", "Log loss: Glicko low vs validation-best Elo"),
    ("delta_brier_inflation", "Brier: low RD inflation vs Glicko C0"),
    ("delta_logloss_inflation", "Log loss: low RD inflation vs Glicko C0"),
    ("delta_brier_glicko_vs_adaptive", "Brier: Glicko low vs best adaptive-K"),
    ("delta_logloss_glicko_vs_adaptive", "Log loss: Glicko low vs best adaptive-K"),
]

FEATURE_COLUMNS_TO_KEEP = [
    "match_id",
    "match_sequence",
    "year",
    "event_id",
    "match_date",
    "player_a_id",
    "player_b_id",
    "winner_id",
    "loser_id",
    "player_a_is_winner",
    "outcome_a",
    "a_total_games_before",
    "b_total_games_before",
    "a_games_last_90_days",
    "b_games_last_90_days",
    "a_games_last_365_days",
    "b_games_last_365_days",
    "a_games_previous_calendar_year",
    "b_games_previous_calendar_year",
    "a_days_since_last_game",
    "b_days_since_last_game",
    "a_career_days_before",
    "b_career_days_before",
    "a_is_debut",
    "b_is_debut",
    "a_has_previous_history",
    "b_has_previous_history",
    "a_date_features_available",
    "b_date_features_available",
    "a_date_quality",
    "b_date_quality",
    "min_total_games_before",
    "max_total_games_before",
    "abs_diff_total_games_before",
    "min_games_last_90_days",
    "max_games_last_90_days",
    "abs_diff_games_last_90_days",
    "min_games_last_365_days",
    "max_games_last_365_days",
    "abs_diff_games_last_365_days",
    "min_previous_year_games",
    "max_previous_year_games",
    "abs_diff_previous_year_games",
    "min_days_since_last_game",
    "max_days_since_last_game",
    "either_player_debut",
    "both_players_have_history",
    "either_player_inactive_365d",
    "either_player_inactive_730d",
    "both_players_active_last_365d",
    "either_player_low_recent_activity",
]


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any = "",
    severity: str = "error",
    detail: str = "",
) -> None:
    """Append one validation or alignment check."""

    rows.append(
        {
            "check_name": check_name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "detail": detail,
        }
    )


def discover_prediction_files() -> dict[str, Path]:
    """Return the prediction and metric files used by this script."""

    required = {
        "features": FEATURE_PATH,
        "fair_predictions": FAIR_PREDICTIONS_PATH,
        "fair_metrics": FAIR_METRICS_PATH,
        "adaptive_predictions": ADAPTIVE_PREDICTIONS_PATH,
        "adaptive_metrics": ADAPTIVE_METRICS_PATH,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    return required


def load_prematch_features() -> pd.DataFrame:
    """Load step 28 pre-match match-level features."""

    features = pd.read_csv(FEATURE_PATH, low_memory=False)
    missing = [col for col in FEATURE_COLUMNS_TO_KEEP if col not in features.columns]
    if missing:
        raise ValueError(f"Prematch feature file is missing required columns: {missing}")
    features = features[FEATURE_COLUMNS_TO_KEEP].copy()
    for col in ["match_id", "match_sequence", "event_id", "player_a_id", "player_b_id", "winner_id", "loser_id", "outcome_a"]:
        features[col] = pd.to_numeric(features[col], errors="raise").astype(int)
    features = features.sort_values("match_sequence").reset_index(drop=True)
    return features


def load_and_normalise_fair_predictions() -> pd.DataFrame:
    """Load fair Elo-vs-Glicko predictions in winner-probability orientation."""

    pred = pd.read_csv(FAIR_PREDICTIONS_PATH, low_memory=False)
    required = {"model", "game_id", "fcode", "winner", "loser", "pred_winner_win"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Fair prediction file is missing required columns: {sorted(missing)}")

    rows = []
    fair_specs = [spec for spec in MODEL_SPECS if spec.source == "fair"]
    for spec in fair_specs:
        sub = pred.loc[pred["model"] == spec.source_model].copy()
        if sub.empty:
            raise ValueError(f"Fair prediction file has no rows for model {spec.source_model}")
        sub["alias"] = spec.alias
        sub["source_model"] = spec.source_model
        sub["display_name"] = spec.display_name
        sub["match_id"] = sub["game_id"].astype(int)
        sub["winner"] = sub["winner"].astype(int)
        sub["loser"] = sub["loser"].astype(int)
        sub["p_winner"] = sub[spec.probability_field].astype(float)
        for col in ["pre_rd_winner", "pre_rd_loser"]:
            if col not in sub.columns:
                sub[col] = np.nan
        rows.append(
            sub[
                [
                    "alias",
                    "source_model",
                    "display_name",
                    "match_id",
                    "game_id",
                    "fcode",
                    "winner",
                    "loser",
                    "p_winner",
                    "pre_rd_winner",
                    "pre_rd_loser",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_and_normalise_adaptive_k_predictions() -> pd.DataFrame:
    """Load the selected adaptive-K predictions in winner-probability orientation."""

    pred = pd.read_csv(ADAPTIVE_PREDICTIONS_PATH, low_memory=False)
    required = {"model", "game_id", "fcode", "winner", "loser", "pred_actual_winner_win"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Adaptive-K prediction file is missing required columns: {sorted(missing)}")

    rows = []
    adaptive_specs = [spec for spec in MODEL_SPECS if spec.source == "adaptive"]
    for spec in adaptive_specs:
        sub = pred.loc[pred["model"] == spec.source_model].copy()
        if sub.empty:
            raise ValueError(f"Adaptive-K prediction file has no rows for model {spec.source_model}")
        sub["alias"] = spec.alias
        sub["source_model"] = spec.source_model
        sub["display_name"] = spec.display_name
        sub["match_id"] = sub["game_id"].astype(int)
        sub["winner"] = sub["winner"].astype(int)
        sub["loser"] = sub["loser"].astype(int)
        sub["p_winner"] = sub[spec.probability_field].astype(float)
        sub["pre_rd_winner"] = np.nan
        sub["pre_rd_loser"] = np.nan
        rows.append(
            sub[
                [
                    "alias",
                    "source_model",
                    "display_name",
                    "match_id",
                    "game_id",
                    "fcode",
                    "winner",
                    "loser",
                    "p_winner",
                    "pre_rd_winner",
                    "pre_rd_loser",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def convert_probabilities_to_player_a(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    alignment_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert model predictions from actual-winner probability to player-A probability."""

    base_ids = features[["match_id", "match_sequence", "player_a_id", "player_b_id", "winner_id", "loser_id"]]
    feature_set = set(features["match_id"])
    model_frames = []
    rd_frame = features[["match_id"]].copy()

    for spec in MODEL_SPECS:
        sub = predictions.loc[predictions["alias"] == spec.alias].copy()
        sub["match_id"] = sub["match_id"].astype(int)
        sub["game_id"] = sub["game_id"].astype(int)
        sub["fcode"] = sub["fcode"].astype(int)
        pred_set = set(sub["match_id"])
        missing_ids = sorted(feature_set - pred_set)
        extra_ids = sorted(pred_set - feature_set)

        add_check(
            alignment_rows,
            "prediction_rows",
            len(sub) == EXPECTED_GAMES,
            len(sub),
            EXPECTED_GAMES,
            detail=spec.alias,
        )
        add_check(
            alignment_rows,
            "prediction_match_id_unique",
            not sub["match_id"].duplicated().any(),
            int(sub["match_id"].duplicated().sum()),
            0,
            detail=spec.alias,
        )
        add_check(
            alignment_rows,
            "game_id_equals_fcode",
            bool((sub["game_id"] == sub["fcode"]).all()),
            bool((sub["game_id"] == sub["fcode"]).all()),
            True,
            detail=spec.alias,
        )
        add_check(
            alignment_rows,
            "feature_prediction_match_id_set",
            not missing_ids and not extra_ids,
            f"missing={len(missing_ids)}; extra={len(extra_ids)}",
            "missing=0; extra=0",
            detail=spec.alias,
        )

        merged = base_ids.merge(
            sub[
                [
                    "match_id",
                    "winner",
                    "loser",
                    "p_winner",
                    "pre_rd_winner",
                    "pre_rd_loser",
                    "source_model",
                ]
            ],
            on="match_id",
            how="left",
            validate="one_to_one",
        )
        add_check(
            alignment_rows,
            "merged_rows",
            len(merged) == EXPECTED_GAMES,
            len(merged),
            EXPECTED_GAMES,
            detail=spec.alias,
        )
        add_check(
            alignment_rows,
            "winner_loser_alignment",
            bool(((merged["winner"] == merged["winner_id"]) & (merged["loser"] == merged["loser_id"])).all()),
            "checked",
            "prediction winner/loser equal feature winner_id/loser_id",
            detail=spec.alias,
        )
        player_a_is_winner = merged["player_a_id"] == merged["winner"]
        player_a_is_loser = merged["player_a_id"] == merged["loser"]
        add_check(
            alignment_rows,
            "player_a_maps_to_prediction_players",
            bool((player_a_is_winner | player_a_is_loser).all()),
            "checked",
            "player_a_id is either prediction winner or loser",
            detail=spec.alias,
        )
        p_a = np.where(player_a_is_winner, merged["p_winner"], 1.0 - merged["p_winner"])
        p_df = pd.DataFrame(
            {
                "match_id": merged["match_id"],
                f"p_a_{spec.alias}": p_a.astype(float),
            }
        )
        model_frames.append(p_df)

        if spec.alias == "Glicko_low":
            rd_a = np.where(player_a_is_winner, merged["pre_rd_winner"], merged["pre_rd_loser"])
            rd_b = np.where(player_a_is_winner, merged["pre_rd_loser"], merged["pre_rd_winner"])
            rd_frame["rd_a_glicko_low"] = rd_a.astype(float)
            rd_frame["rd_b_glicko_low"] = rd_b.astype(float)

    out = features.copy()
    for frame in model_frames:
        out = out.merge(frame, on="match_id", how="left", validate="one_to_one")
    out = out.merge(rd_frame, on="match_id", how="left", validate="one_to_one")
    return out.sort_values("match_sequence").reset_index(drop=True), pd.DataFrame(alignment_rows)


def score_one_model(df: pd.DataFrame, alias: str) -> None:
    """Add per-match scores for one model."""

    p = df[f"p_a_{alias}"].astype(float)
    y = df["outcome_a"].astype(float)
    clipped = p.clip(EPS, 1.0 - EPS)
    tie = p == 0.5
    predicted_a_win = p >= 0.5
    # Meeting 5 evaluated actual-winner probability with p >= 0.5 counted as
    # correct. After converting to player-A orientation, an exact 0.5 tie has
    # no side direction, so count it as a correct favourite/tie prediction to
    # preserve the existing fixed-test accuracy definition.
    favourite_won = np.where(tie, 1.0, np.where(predicted_a_win, y, 1.0 - y))
    df[f"brier_{alias}"] = (p - y) ** 2
    df[f"logloss_{alias}"] = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    df[f"correct_{alias}"] = np.where(tie, 1, (predicted_a_win.astype(int) == y.astype(int)).astype(int))
    df[f"favourite_probability_{alias}"] = np.maximum(p, 1.0 - p)
    df[f"favourite_won_{alias}"] = favourite_won.astype(int)


def calculate_per_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-match Brier, log loss, accuracy, and paired differences."""

    out = df.copy()
    for spec in MODEL_SPECS:
        score_one_model(out, spec.alias)

    out["delta_brier_glicko_vs_elo"] = out["brier_Validation_best_Elo"] - out["brier_Glicko_low"]
    out["delta_logloss_glicko_vs_elo"] = out["logloss_Validation_best_Elo"] - out["logloss_Glicko_low"]
    out["delta_accuracy_glicko_vs_elo"] = out["correct_Glicko_low"] - out["correct_Validation_best_Elo"]
    out["delta_brier_inflation"] = out["brier_Glicko_C0"] - out["brier_Glicko_low"]
    out["delta_logloss_inflation"] = out["logloss_Glicko_C0"] - out["logloss_Glicko_low"]
    out["delta_accuracy_inflation"] = out["correct_Glicko_low"] - out["correct_Glicko_C0"]
    out["delta_brier_glicko_vs_adaptive"] = out["brier_best_AdaptiveK"] - out["brier_Glicko_low"]
    out["delta_logloss_glicko_vs_adaptive"] = out["logloss_best_AdaptiveK"] - out["logloss_Glicko_low"]
    out["delta_accuracy_glicko_vs_adaptive"] = out["correct_Glicko_low"] - out["correct_best_AdaptiveK"]
    out["delta_brier_tuned_vs_default"] = out["brier_Default_Elo"] - out["brier_Validation_best_Elo"]
    out["delta_logloss_tuned_vs_default"] = out["logloss_Default_Elo"] - out["logloss_Validation_best_Elo"]
    return out


def corrected_calibration_error(group: pd.DataFrame, alias: str) -> float:
    """Return sample-weighted absolute favourite-perspective calibration error."""

    bins = calibration_bins_for_model(group, alias)
    if bins.empty or bins["games"].sum() == 0:
        return np.nan
    return float(np.average(bins["abs_calibration_gap"], weights=bins["games"]))


def calibration_bins_for_model(group: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Build fixed favourite-probability calibration bins for one model."""

    if group.empty:
        return pd.DataFrame()
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    fav = group[f"favourite_probability_{alias}"].astype(float)
    won = group[f"favourite_won_{alias}"].astype(float)
    bin_index = np.floor((fav.clip(0.5, 1.0) - 0.5) / 0.1).astype(int)
    bin_index = np.clip(bin_index, 0, len(labels) - 1)
    temp = pd.DataFrame({"bin": [labels[i] for i in bin_index], "fav": fav, "won": won})
    rows = []
    for label in labels:
        sub = temp.loc[temp["bin"] == label]
        if sub.empty:
            rows.append(
                {
                    "probability_bin": label,
                    "games": 0,
                    "mean_favourite_probability": np.nan,
                    "empirical_favourite_win_rate": np.nan,
                    "calibration_gap": np.nan,
                    "abs_calibration_gap": np.nan,
                }
            )
        else:
            mean_prob = float(sub["fav"].mean())
            win_rate = float(sub["won"].mean())
            rows.append(
                {
                    "probability_bin": label,
                    "games": int(len(sub)),
                    "mean_favourite_probability": mean_prob,
                    "empirical_favourite_win_rate": win_rate,
                    "calibration_gap": win_rate - mean_prob,
                    "abs_calibration_gap": abs(win_rate - mean_prob),
                }
            )
    return pd.DataFrame(rows)


def load_reference_metrics() -> dict[str, dict[str, float]]:
    """Load meeting 5 reference metrics for exact consistency checks."""

    references: dict[str, dict[str, float]] = {}
    fair = pd.read_csv(FAIR_METRICS_PATH)
    adaptive = pd.read_csv(ADAPTIVE_METRICS_PATH)
    for spec in MODEL_SPECS:
        source = fair if spec.source == "fair" else adaptive
        row = source.loc[source["model"] == spec.source_model]
        if row.empty:
            continue
        r = row.iloc[0]
        references[spec.alias] = {
            "reference_logloss": float(r["log_loss"]),
            "reference_brier": float(r["brier"]),
            "reference_accuracy": float(r["accuracy"]),
        }
    return references


def validate_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute overall metrics and compare them against meeting 5 outputs."""

    references = load_reference_metrics()
    rows = []
    for spec in MODEL_SPECS:
        alias = spec.alias
        brier = float(df[f"brier_{alias}"].mean())
        logloss = float(df[f"logloss_{alias}"].mean())
        accuracy = float(df[f"correct_{alias}"].mean())
        fav_prob = float(df[f"favourite_probability_{alias}"].mean())
        fav_win = float(df[f"favourite_won_{alias}"].mean())
        calibration_error = corrected_calibration_error(df, alias)
        ref = references.get(alias, {})
        rows.append(
            {
                "model": alias,
                "source_model": spec.source_model,
                "display_name": spec.display_name,
                "model_family": spec.family,
                "evaluation_games": int(len(df)),
                "brier": brier,
                "log_loss": logloss,
                "accuracy": accuracy,
                "mean_favourite_probability": fav_prob,
                "favourite_win_rate": fav_win,
                "corrected_calibration_error": calibration_error,
                "reference_brier": ref.get("reference_brier", np.nan),
                "reference_log_loss": ref.get("reference_logloss", np.nan),
                "reference_accuracy": ref.get("reference_accuracy", np.nan),
                "brier_abs_diff_vs_reference": abs(brier - ref.get("reference_brier", np.nan)),
                "logloss_abs_diff_vs_reference": abs(logloss - ref.get("reference_logloss", np.nan)),
                "accuracy_abs_diff_vs_reference": abs(accuracy - ref.get("reference_accuracy", np.nan)),
                "matches_reference_within_tolerance": bool(
                    abs(brier - ref.get("reference_brier", np.nan)) <= METRIC_TOLERANCE
                    and abs(logloss - ref.get("reference_logloss", np.nan)) <= METRIC_TOLERANCE
                    and abs(accuracy - ref.get("reference_accuracy", np.nan)) <= METRIC_TOLERANCE
                ),
            }
        )
    return pd.DataFrame(rows)


def bin_min_total_games(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    value = float(value)
    if value == 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 20:
        return "6-20"
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return "100+"


def bin_min_games_last_365(value: Any) -> str:
    if pd.isna(value):
        return "Missing date information"
    value = float(value)
    if value == 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 15:
        return "6-15"
    if value <= 30:
        return "16-30"
    return "30+"


def bin_min_previous_year_games(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    value = float(value)
    if value == 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 15:
        return "6-15"
    if value <= 30:
        return "16-30"
    return "30+"


def bin_inactivity(row: pd.Series) -> str:
    if bool(row["either_player_debut"]):
        return "No previous history"
    if not bool(row["both_players_have_history"]) or pd.isna(row["max_days_since_last_game"]):
        return "Missing date information"
    value = float(row["max_days_since_last_game"])
    if value <= 90:
        return "0-90 days"
    if value <= 180:
        return "91-180 days"
    if value <= 365:
        return "181-365 days"
    if value <= 730:
        return "366-730 days"
    if value <= 1095:
        return "731-1095 days"
    return "1096+ days"


def bin_experience_ratio(row: pd.Series) -> str:
    if bool(row["either_player_debut"]):
        return "No previous history"
    ratio = float(row["experience_ratio"])
    if ratio <= 2:
        return "Balanced: ratio <= 2"
    if ratio <= 5:
        return "Moderate mismatch: 2 < ratio <= 5"
    return "Large mismatch: ratio > 5"


def bin_elo_confidence(value: Any) -> str:
    value = float(value)
    if value < 0.60:
        return "0.50-0.60"
    if value < 0.70:
        return "0.60-0.70"
    if value < 0.80:
        return "0.70-0.80"
    if value < 0.90:
        return "0.80-0.90"
    return "0.90-1.00"


def qcut_labels(series: pd.Series, prefix: str, q: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Create stable quantile labels and return cutpoint metadata."""

    labels = [f"Q{i + 1}" for i in range(q)]
    cats, bins = pd.qcut(series, q=q, labels=labels, retbins=True, duplicates="drop")
    actual_labels = list(cats.cat.categories)
    rows = []
    for idx, label in enumerate(actual_labels):
        rows.append(
            {
                "variable": prefix,
                "group": label,
                "lower": float(bins[idx]),
                "upper": float(bins[idx + 1]),
            }
        )
    return cats.astype(str), pd.DataFrame(rows)


def build_subgroup_definitions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, bool]:
    """Add all pre-specified subgroup labels."""

    out = df.copy()
    out["subgroup_total_experience"] = out["min_total_games_before"].map(bin_min_total_games)
    out["subgroup_recent_365_activity"] = out["min_games_last_365_days"].map(bin_min_games_last_365)
    out["subgroup_previous_year_activity"] = out["min_previous_year_games"].map(bin_min_previous_year_games)
    out["subgroup_inactivity_gap"] = out.apply(bin_inactivity, axis=1)
    out["min_total_games_before_le5"] = out["min_total_games_before"] <= 5
    out["min_total_games_before_le20"] = out["min_total_games_before"] <= 20
    out["min_games_last_365_days_le5"] = out["min_games_last_365_days"] <= 5
    out["min_previous_year_games_le5"] = out["min_previous_year_games"] <= 5
    out["experience_ratio"] = (out["max_total_games_before"] + 1.0) / (out["min_total_games_before"] + 1.0)
    out["log_experience_ratio"] = np.log1p(out["max_total_games_before"]) - np.log1p(out["min_total_games_before"])
    out["subgroup_experience_ratio"] = out.apply(bin_experience_ratio, axis=1)
    out["elo_favourite_probability"] = out["favourite_probability_Validation_best_Elo"]
    out["subgroup_elo_favourite_probability"] = out["elo_favourite_probability"].map(bin_elo_confidence)
    out["abs_probability_difference"] = (out["p_a_Glicko_low"] - out["p_a_Validation_best_Elo"]).abs()
    out["subgroup_probability_difference_quintile"], prob_qcuts = qcut_labels(
        out["abs_probability_difference"],
        "abs_probability_difference",
        q=5,
    )

    rd_available = (
        "rd_a_glicko_low" in out.columns
        and "rd_b_glicko_low" in out.columns
        and out[["rd_a_glicko_low", "rd_b_glicko_low"]].notna().all(axis=None)
    )
    rd_qcuts = pd.DataFrame()
    if rd_available:
        out["max_prematch_rd"] = out[["rd_a_glicko_low", "rd_b_glicko_low"]].max(axis=1)
        out["min_prematch_rd"] = out[["rd_a_glicko_low", "rd_b_glicko_low"]].min(axis=1)
        out["mean_prematch_rd"] = out[["rd_a_glicko_low", "rd_b_glicko_low"]].mean(axis=1)
        rd_labels, rd_qcuts = qcut_labels(out["max_prematch_rd"], "max_prematch_rd", q=4)
        mapping = {
            "Q1": "Q1 lowest uncertainty",
            "Q2": "Q2",
            "Q3": "Q3",
            "Q4": "Q4 highest uncertainty",
        }
        out["subgroup_glicko_rd_quartile"] = rd_labels.map(mapping).fillna(rd_labels)
        rd_qcuts["group"] = rd_qcuts["group"].map(mapping).fillna(rd_qcuts["group"])
    else:
        out["max_prematch_rd"] = np.nan
        out["min_prematch_rd"] = np.nan
        out["mean_prematch_rd"] = np.nan
        out["subgroup_glicko_rd_quartile"] = "RD unavailable"

    prob_qcuts.to_csv(OUTPUT_DIR / "29_probability_difference_quintile_cutpoints.csv", index=False, encoding="utf-8-sig")
    if rd_available:
        rd_qcuts.to_csv(RD_CUTPOINTS_PATH, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            [
                {
                    "variable": "max_prematch_rd",
                    "group": "RD unavailable",
                    "lower": np.nan,
                    "upper": np.nan,
                    "note": "Reliable pre-match RD fields were not available.",
                }
            ]
        ).to_csv(RD_CUTPOINTS_PATH, index=False, encoding="utf-8-sig")

    subgroup_specs = [
        {
            "variable": "overall",
            "column": "subgroup_overall",
            "title": "Overall",
            "order": ["Overall"],
            "mutually_exclusive": True,
            "kind": "overall",
        },
        {
            "variable": "total_experience",
            "column": "subgroup_total_experience",
            "title": "Minimum total previous games",
            "order": ["0", "1-5", "6-20", "21-50", "51-100", "100+"],
            "mutually_exclusive": True,
            "kind": "player_category",
        },
        {
            "variable": "recent_365_activity",
            "column": "subgroup_recent_365_activity",
            "title": "Minimum games in last 365 days",
            "order": ["0", "1-5", "6-15", "16-30", "30+", "Missing date information"],
            "mutually_exclusive": True,
            "kind": "player_category",
        },
        {
            "variable": "previous_year_activity",
            "column": "subgroup_previous_year_activity",
            "title": "Minimum previous-year games",
            "order": ["0", "1-5", "6-15", "16-30", "30+"],
            "mutually_exclusive": True,
            "kind": "player_category",
        },
        {
            "variable": "inactivity_gap",
            "column": "subgroup_inactivity_gap",
            "title": "Maximum inactivity gap",
            "order": [
                "No previous history",
                "0-90 days",
                "91-180 days",
                "181-365 days",
                "366-730 days",
                "731-1095 days",
                "1096+ days",
                "Missing date information",
            ],
            "mutually_exclusive": True,
            "kind": "player_category",
        },
        {
            "variable": "experience_ratio",
            "column": "subgroup_experience_ratio",
            "title": "Experience imbalance",
            "order": [
                "No previous history",
                "Balanced: ratio <= 2",
                "Moderate mismatch: 2 < ratio <= 5",
                "Large mismatch: ratio > 5",
            ],
            "mutually_exclusive": True,
            "kind": "diagnostic",
        },
        {
            "variable": "elo_favourite_probability",
            "column": "subgroup_elo_favourite_probability",
            "title": "Validation-best Elo favourite probability",
            "order": ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
            "mutually_exclusive": True,
            "kind": "prediction_confidence",
        },
        {
            "variable": "probability_difference_quintile",
            "column": "subgroup_probability_difference_quintile",
            "title": "Absolute Glicko-Elo probability difference",
            "order": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "mutually_exclusive": True,
            "kind": "diagnostic",
        },
    ]
    if rd_available:
        subgroup_specs.append(
            {
                "variable": "glicko_rd_quartile",
                "column": "subgroup_glicko_rd_quartile",
                "title": "Glicko pre-match RD quartile",
                "order": ["Q1 lowest uncertainty", "Q2", "Q3", "Q4 highest uncertainty"],
                "mutually_exclusive": True,
                "kind": "rd_uncertainty",
            }
        )

    flag_columns = [
        ("either_player_debut", "Either player debut"),
        ("either_player_inactive_365d", "Either player inactive >= 365 days"),
        ("either_player_inactive_730d", "Either player inactive >= 730 days"),
        ("both_players_active_last_365d", "Both players active in last 365 days"),
        ("either_player_low_recent_activity", "Either player low recent activity"),
        ("min_total_games_before_le5", "Minimum total games <= 5"),
        ("min_total_games_before_le20", "Minimum total games <= 20"),
        ("min_games_last_365_days_le5", "Minimum games last 365 days <= 5"),
        ("min_previous_year_games_le5", "Minimum previous-year games <= 5"),
    ]
    for col, title in flag_columns:
        label_col = f"subgroup_flag_{col}"
        out[label_col] = np.where(out[col].astype(bool), "True", "False")
        subgroup_specs.append(
            {
                "variable": f"flag_{col}",
                "column": label_col,
                "title": title,
                "order": ["True", "False"],
                "mutually_exclusive": False,
                "kind": "binary_flag",
            }
        )

    out["subgroup_overall"] = "Overall"
    return out, subgroup_specs, rd_qcuts, rd_available


def calculate_subgroup_model_metrics(df: pd.DataFrame, subgroup_specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Calculate model metrics in long format for every subgroup."""

    rows = []
    for spec_group in subgroup_specs:
        col = spec_group["column"]
        order = spec_group["order"]
        for subgroup in order:
            sub = df.loc[df[col] == subgroup]
            if sub.empty:
                continue
            for model in MODEL_SPECS:
                alias = model.alias
                rows.append(
                    {
                        "subgroup_variable": spec_group["variable"],
                        "subgroup": subgroup,
                        "subgroup_title": spec_group["title"],
                        "subgroup_kind": spec_group["kind"],
                        "model": alias,
                        "model_display": model.display_name,
                        "games": int(len(sub)),
                        "events": int(sub["event_id"].nunique()),
                        "brier": float(sub[f"brier_{alias}"].mean()),
                        "logloss": float(sub[f"logloss_{alias}"].mean()),
                        "accuracy": float(sub[f"correct_{alias}"].mean()),
                        "mean_favourite_probability": float(sub[f"favourite_probability_{alias}"].mean()),
                        "favourite_win_rate": float(sub[f"favourite_won_{alias}"].mean()),
                        "corrected_calibration_error": corrected_calibration_error(sub, alias),
                    }
                )
    return pd.DataFrame(rows)


def calculate_pairwise_differences(df: pd.DataFrame, subgroup_specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Calculate paired Glicko/Elo/adaptive comparisons by subgroup."""

    rows = []
    for spec_group in subgroup_specs:
        col = spec_group["column"]
        for subgroup in spec_group["order"]:
            sub = df.loc[df[col] == subgroup]
            if sub.empty:
                continue
            rows.append(
                {
                    "subgroup_variable": spec_group["variable"],
                    "subgroup": subgroup,
                    "subgroup_title": spec_group["title"],
                    "subgroup_kind": spec_group["kind"],
                    "games": int(len(sub)),
                    "events": int(sub["event_id"].nunique()),
                    "glicko_brier": float(sub["brier_Glicko_low"].mean()),
                    "elo_brier": float(sub["brier_Validation_best_Elo"].mean()),
                    "delta_brier_glicko_vs_elo": float(sub["delta_brier_glicko_vs_elo"].mean()),
                    "glicko_logloss": float(sub["logloss_Glicko_low"].mean()),
                    "elo_logloss": float(sub["logloss_Validation_best_Elo"].mean()),
                    "delta_logloss_glicko_vs_elo": float(sub["delta_logloss_glicko_vs_elo"].mean()),
                    "glicko_accuracy": float(sub["correct_Glicko_low"].mean()),
                    "elo_accuracy": float(sub["correct_Validation_best_Elo"].mean()),
                    "delta_accuracy_glicko_vs_elo": float(sub["delta_accuracy_glicko_vs_elo"].mean()),
                    "delta_brier_inflation": float(sub["delta_brier_inflation"].mean()),
                    "delta_logloss_inflation": float(sub["delta_logloss_inflation"].mean()),
                    "delta_accuracy_inflation": float(sub["delta_accuracy_inflation"].mean()),
                    "delta_brier_glicko_vs_adaptive": float(sub["delta_brier_glicko_vs_adaptive"].mean()),
                    "delta_logloss_glicko_vs_adaptive": float(sub["delta_logloss_glicko_vs_adaptive"].mean()),
                    "delta_accuracy_glicko_vs_adaptive": float(sub["delta_accuracy_glicko_vs_adaptive"].mean()),
                    "percentage_games_glicko_lower_brier_than_elo": float((sub["delta_brier_glicko_vs_elo"] > 0).mean()),
                    "percentage_games_glicko_lower_logloss_than_elo": float((sub["delta_logloss_glicko_vs_elo"] > 0).mean()),
                    "small_sample_warning": bool(len(sub) < 50 or sub["event_id"].nunique() < 10),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_difference(
    sub: pd.DataFrame,
    diff_col: str,
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Run paired event-cluster bootstrap, with match-level fallback for tiny event counts."""

    games = int(len(sub))
    events = int(sub["event_id"].nunique())
    point = float(sub[diff_col].mean()) if games else np.nan
    small_warning = bool(games < 50 or events < 10)
    if games == 0:
        return {
            "point_estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_excludes_zero": False,
            "bootstrap_type": "not_run",
            "bootstrap_replications": 0,
            "small_sample_warning": small_warning,
        }

    if events >= 10:
        grouped = sub.groupby("event_id", sort=False)[diff_col].agg(["sum", "count"]).reset_index(drop=True)
        sums = grouped["sum"].to_numpy(dtype=float)
        counts = grouped["count"].to_numpy(dtype=float)
        n_units = len(grouped)
        bootstrap_type = "event_cluster"
        estimates = np.empty(reps, dtype=float)
        for i in range(reps):
            idx = rng.integers(0, n_units, size=n_units)
            estimates[i] = sums[idx].sum() / counts[idx].sum()
    else:
        values = sub[diff_col].to_numpy(dtype=float)
        n_units = len(values)
        bootstrap_type = "match_level"
        estimates = np.empty(reps, dtype=float)
        for i in range(reps):
            idx = rng.integers(0, n_units, size=n_units)
            estimates[i] = values[idx].mean()

    ci_lower, ci_upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
        "bootstrap_type": bootstrap_type,
        "bootstrap_replications": reps,
        "small_sample_warning": small_warning,
    }


def run_event_cluster_bootstrap(df: pd.DataFrame, subgroup_specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Bootstrap paired score differences for each subgroup."""

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for spec_group in subgroup_specs:
        col = spec_group["column"]
        for subgroup in spec_group["order"]:
            sub = df.loc[df[col] == subgroup]
            if sub.empty:
                continue
            for diff_col, diff_label in PAIRWISE_DIFFS:
                result = bootstrap_mean_difference(sub, diff_col, rng, BOOTSTRAP_REPS)
                rows.append(
                    {
                        "subgroup_variable": spec_group["variable"],
                        "subgroup": subgroup,
                        "subgroup_title": spec_group["title"],
                        "diff_name": diff_col,
                        "diff_label": diff_label,
                        "games": int(len(sub)),
                        "unique_events": int(sub["event_id"].nunique()),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def calculate_corrected_calibration(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate favourite-perspective calibration for overall and selected ranges."""

    ranges = [
        ("overall", "Overall", pd.Series(True, index=df.index)),
        ("either_player_debut", "Either player debut", df["either_player_debut"].astype(bool)),
        ("min_total_games_le5", "Minimum total games <= 5", df["min_total_games_before"] <= 5),
        ("min_recent_games_le5", "Minimum games last 365 days <= 5", df["min_games_last_365_days"] <= 5),
        ("inactive_365", "Inactive >= 365 days", df["either_player_inactive_365d"].astype(bool)),
        ("both_active_365", "Both players active last 365 days", df["both_players_active_last_365d"].astype(bool)),
    ]
    summary_rows = []
    bin_rows = []
    for range_key, range_label, mask in ranges:
        sub = df.loc[mask.fillna(False)]
        for spec in MODEL_SPECS:
            bins = calibration_bins_for_model(sub, spec.alias)
            if bins.empty:
                calibration_error = np.nan
            else:
                nonempty = bins.loc[bins["games"] > 0]
                calibration_error = (
                    float(np.average(nonempty["abs_calibration_gap"], weights=nonempty["games"]))
                    if not nonempty.empty
                    else np.nan
                )
            summary_rows.append(
                {
                    "range": range_key,
                    "range_label": range_label,
                    "model": spec.alias,
                    "model_display": spec.display_name,
                    "games": int(len(sub)),
                    "corrected_calibration_error": calibration_error,
                }
            )
            for row in bins.itertuples(index=False):
                bin_rows.append(
                    {
                        "range": range_key,
                        "range_label": range_label,
                        "model": spec.alias,
                        "model_display": spec.display_name,
                        **row._asdict(),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def calculate_adaptive_k_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much of Glicko's improvement over default Elo adaptive-K recovers."""

    ranges = [
        ("overall", "Overall", pd.Series(True, index=df.index)),
        ("debut", "Either player debut", df["either_player_debut"].astype(bool)),
        ("total_previous_games_le5", "Minimum total games <= 5", df["min_total_games_before"] <= 5),
        ("recent_games_le5", "Minimum games last 365 days <= 5", df["min_games_last_365_days"] <= 5),
        ("inactive_365", "Inactive >= 365 days", df["either_player_inactive_365d"].astype(bool)),
        ("both_active_last_365", "Both players active last 365 days", df["both_players_active_last_365d"].astype(bool)),
    ]
    rows = []
    for key, label, mask in ranges:
        sub = df.loc[mask.fillna(False)]
        if sub.empty:
            continue
        default_brier = float(sub["brier_Default_Elo"].mean())
        adaptive_brier = float(sub["brier_best_AdaptiveK"].mean())
        glicko_brier = float(sub["brier_Glicko_low"].mean())
        default_logloss = float(sub["logloss_Default_Elo"].mean())
        adaptive_logloss = float(sub["logloss_best_AdaptiveK"].mean())
        glicko_logloss = float(sub["logloss_Glicko_low"].mean())
        denom_brier = default_brier - glicko_brier
        denom_logloss = default_logloss - glicko_logloss
        valid_brier = bool(denom_brier > 1e-12)
        valid_logloss = bool(denom_logloss > 1e-12)
        rows.append(
            {
                "range": key,
                "range_label": label,
                "games": int(len(sub)),
                "events": int(sub["event_id"].nunique()),
                "brier_default_elo": default_brier,
                "brier_best_adaptive_k": adaptive_brier,
                "brier_glicko_low": glicko_brier,
                "improvement_recovered_brier": (
                    (default_brier - adaptive_brier) / denom_brier if valid_brier else np.nan
                ),
                "recovery_ratio_valid_brier": valid_brier,
                "logloss_default_elo": default_logloss,
                "logloss_best_adaptive_k": adaptive_logloss,
                "logloss_glicko_low": glicko_logloss,
                "improvement_recovered_logloss": (
                    (default_logloss - adaptive_logloss) / denom_logloss if valid_logloss else np.nan
                ),
                "recovery_ratio_valid_logloss": valid_logloss,
            }
        )
    return pd.DataFrame(rows)


def find_bootstrap_row(
    bootstrap_df: pd.DataFrame,
    subgroup_variable: str,
    subgroup: str,
    diff_name: str,
) -> pd.Series | None:
    mask = (
        (bootstrap_df["subgroup_variable"] == subgroup_variable)
        & (bootstrap_df["subgroup"] == subgroup)
        & (bootstrap_df["diff_name"] == diff_name)
    )
    rows = bootstrap_df.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_key_results_table(pairwise: pd.DataFrame, bootstrap: pd.DataFrame, rd_available: bool) -> pd.DataFrame:
    """Build a concise meeting table for the main result rows."""

    requested = [
        ("Overall", "overall", "Overall"),
        ("Either player debut", "flag_either_player_debut", "True"),
        ("Total previous games <= 5", "flag_min_total_games_before_le5", "True"),
        ("Recent games <= 5", "flag_min_games_last_365_days_le5", "True"),
        ("Inactive >= 365 days", "flag_either_player_inactive_365d", "True"),
        ("Inactive >= 730 days", "flag_either_player_inactive_730d", "True"),
        ("Both players active last 365 days", "flag_both_players_active_last_365d", "True"),
    ]
    if rd_available:
        requested.append(("Highest RD quartile", "glicko_rd_quartile", "Q4 highest uncertainty"))
        requested.append(("Lowest RD quartile", "glicko_rd_quartile", "Q1 lowest uncertainty"))

    rows = []
    for label, subgroup_variable, subgroup in requested:
        row = pairwise.loc[
            (pairwise["subgroup_variable"] == subgroup_variable) & (pairwise["subgroup"] == subgroup)
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        brier_ci = find_bootstrap_row(bootstrap, subgroup_variable, subgroup, "delta_brier_glicko_vs_elo")
        logloss_ci = find_bootstrap_row(bootstrap, subgroup_variable, subgroup, "delta_logloss_glicko_vs_elo")
        rows.append(
            {
                "subgroup": label,
                "games": int(r["games"]),
                "events": int(r["events"]),
                "glicko_brier": r["glicko_brier"],
                "elo_brier": r["elo_brier"],
                "delta_brier": r["delta_brier_glicko_vs_elo"],
                "delta_brier_ci_lower": brier_ci["ci_lower"] if brier_ci is not None else np.nan,
                "delta_brier_ci_upper": brier_ci["ci_upper"] if brier_ci is not None else np.nan,
                "glicko_logloss": r["glicko_logloss"],
                "elo_logloss": r["elo_logloss"],
                "delta_logloss": r["delta_logloss_glicko_vs_elo"],
                "delta_logloss_ci_lower": logloss_ci["ci_lower"] if logloss_ci is not None else np.nan,
                "delta_logloss_ci_upper": logloss_ci["ci_upper"] if logloss_ci is not None else np.nan,
                "glicko_accuracy": r["glicko_accuracy"],
                "elo_accuracy": r["elo_accuracy"],
                "inflation_delta_brier": r["delta_brier_inflation"],
                "adaptive_delta_brier": r["delta_brier_glicko_vs_adaptive"],
                "small_sample_warning": bool(r["small_sample_warning"]),
            }
        )
    return pd.DataFrame(rows)


def plot_overall_brier(overall: pd.DataFrame) -> Path:
    """Create Figure 1: zoomed overall Brier score."""

    path = FIGURE_DIR / "29_fig01_overall_brier_zoomed.png"
    data = overall.set_index("model").loc[[m for m in OVERALL_MODEL_ORDER if m in set(overall["model"])]].reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2C7FB8", "#7A7A7A", "#66A61E", "#B8B8B8", "#D95F02", "#AAAAAA"][: len(data)]
    ax.bar(data["display_name"], data["brier"], color=colors)
    low = max(0, data["brier"].min() - 0.002)
    high = data["brier"].max() + 0.002
    ax.set_ylim(low, high)
    ax.set_ylabel("Brier score")
    ax.set_title("Overall 2025 Brier score\nVertical axis truncated to show small differences.")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delta_with_ci(
    bootstrap: pd.DataFrame,
    subgroup_variable: str,
    diff_name: str,
    order: list[str],
    title: str,
    ylabel: str,
    output_name: str,
    short_labels: dict[str, str] | None = None,
) -> Path:
    """Create a delta Brier figure with bootstrap confidence intervals."""

    path = FIGURE_DIR / output_name
    data = bootstrap.loc[
        (bootstrap["subgroup_variable"] == subgroup_variable) & (bootstrap["diff_name"] == diff_name)
    ].copy()
    data["order"] = data["subgroup"].map({label: idx for idx, label in enumerate(order)})
    data = data.dropna(subset=["order"]).sort_values("order")
    if data.empty:
        return path
    labels = [short_labels.get(x, x) if short_labels else x for x in data["subgroup"]]
    y = data["point_estimate"].to_numpy(float)
    yerr = np.vstack([y - data["ci_lower"].to_numpy(float), data["ci_upper"].to_numpy(float) - y])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(np.arange(len(data)), y, yerr=yerr, fmt="o", color="#2C7FB8", ecolor="#7A7A7A", capsize=4)
    ax.axhline(0, color="#333333", linewidth=1)
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.text(idx, row.point_estimate, f"n={int(row.games)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.01, 0.02, "Positive values mean Glicko low inflation is better.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def create_figures(
    overall: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rd_available: bool,
) -> list[Path]:
    """Create the requested meeting figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [plot_overall_brier(overall)]
    paths.append(
        plot_delta_with_ci(
            bootstrap,
            "total_experience",
            "delta_brier_glicko_vs_elo",
            ["0", "1-5", "6-20", "21-50", "51-100", "100+"],
            "Glicko vs Elo Brier improvement by total experience",
            "Elo Brier - Glicko Brier",
            "29_fig02_delta_brier_by_total_games.png",
        )
    )
    paths.append(
        plot_delta_with_ci(
            bootstrap,
            "recent_365_activity",
            "delta_brier_glicko_vs_elo",
            ["0", "1-5", "6-15", "16-30", "30+", "Missing date information"],
            "Glicko vs Elo Brier improvement by recent activity",
            "Elo Brier - Glicko Brier",
            "29_fig03_delta_brier_by_recent_activity.png",
            {"Missing date information": "Missing"},
        )
    )
    paths.append(
        plot_delta_with_ci(
            bootstrap,
            "inactivity_gap",
            "delta_brier_glicko_vs_elo",
            [
                "No previous history",
                "0-90 days",
                "91-180 days",
                "181-365 days",
                "366-730 days",
                "731-1095 days",
                "1096+ days",
                "Missing date information",
            ],
            "Glicko vs Elo Brier improvement by inactivity gap",
            "Elo Brier - Glicko Brier",
            "29_fig04_delta_brier_by_inactivity.png",
            {
                "No previous history": "No history",
                "Missing date information": "Missing",
                "366-730 days": "366-730d",
                "731-1095 days": "731-1095d",
                "1096+ days": "1096+d",
            },
        )
    )
    paths.append(
        plot_delta_with_ci(
            bootstrap,
            "inactivity_gap",
            "delta_brier_inflation",
            [
                "No previous history",
                "0-90 days",
                "91-180 days",
                "181-365 days",
                "366-730 days",
                "731-1095 days",
                "1096+ days",
                "Missing date information",
            ],
            "RD inflation contribution by inactivity gap",
            "Glicko C0 Brier - Glicko low Brier",
            "29_fig05_inflation_gain_by_inactivity.png",
            {
                "No previous history": "No history",
                "Missing date information": "Missing",
                "366-730 days": "366-730d",
                "731-1095 days": "731-1095d",
                "1096+ days": "1096+d",
            },
        )
    )
    paths.append(
        plot_delta_with_ci(
            bootstrap,
            "elo_favourite_probability",
            "delta_brier_glicko_vs_elo",
            ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
            "Glicko vs Elo Brier improvement by Elo prediction confidence",
            "Elo Brier - Glicko Brier",
            "29_fig06_delta_brier_by_prediction_confidence.png",
        )
    )
    if rd_available:
        paths.append(
            plot_delta_with_ci(
                bootstrap,
                "glicko_rd_quartile",
                "delta_brier_glicko_vs_elo",
                ["Q1 lowest uncertainty", "Q2", "Q3", "Q4 highest uncertainty"],
                "Glicko advantage by pre-match RD",
                "Elo Brier - Glicko Brier",
                "29_fig07_delta_brier_by_glicko_rd.png",
                {
                    "Q1 lowest uncertainty": "Q1 low RD",
                    "Q4 highest uncertainty": "Q4 high RD",
                },
            )
        )
    return paths


def run_validation_checks(
    df: pd.DataFrame,
    alignment: pd.DataFrame,
    overall: pd.DataFrame,
    subgroup_specs: list[dict[str, Any]],
    bootstrap: pd.DataFrame,
    output_paths: list[Path],
    rd_available: bool,
) -> pd.DataFrame:
    """Run final validation checks for the where-Glicko-helps script."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "feature_rows", len(df) == EXPECTED_GAMES, len(df), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", not df["match_id"].duplicated().any(), int(df["match_id"].duplicated().sum()), 0)
    add_check(rows, "outcome_a_binary", set(df["outcome_a"].unique()).issubset({0, 1}), sorted(df["outcome_a"].unique()), "0/1")
    add_check(rows, "alignment_checks_passed", bool(alignment["passed"].all()), f"{int(alignment['passed'].sum())}/{len(alignment)}", f"{len(alignment)}/{len(alignment)}")
    for alias in CORE_ALIASES:
        add_check(
            rows,
            f"probability_present_{alias}",
            df[f"p_a_{alias}"].notna().all(),
            int(df[f"p_a_{alias}"].isna().sum()),
            0,
        )
    for spec in MODEL_SPECS:
        alias = spec.alias
        prob = df[f"p_a_{alias}"]
        add_check(rows, f"probability_range_{alias}", bool(prob.between(0, 1).all()), f"{prob.min()} to {prob.max()}", "[0,1]")
        add_check(rows, f"brier_nonnegative_{alias}", bool((df[f"brier_{alias}"] >= 0).all()), "checked", ">=0")
        add_check(rows, f"logloss_nonnegative_{alias}", bool((df[f"logloss_{alias}"] >= 0).all()), "checked", ">=0")
    add_check(
        rows,
        "overall_metrics_match_meeting5",
        bool(overall["matches_reference_within_tolerance"].all()),
        f"{int(overall['matches_reference_within_tolerance'].sum())}/{len(overall)}",
        f"{len(overall)}/{len(overall)}",
    )
    for spec_group in subgroup_specs:
        counts = df[spec_group["column"]].value_counts(dropna=False)
        if spec_group["mutually_exclusive"]:
            add_check(
                rows,
                f"mutually_exclusive_sum_{spec_group['variable']}",
                int(counts.sum()) == EXPECTED_GAMES,
                int(counts.sum()),
                EXPECTED_GAMES,
            )
        if spec_group["kind"] == "binary_flag":
            add_check(
                rows,
                f"binary_flag_sum_{spec_group['variable']}",
                int(counts.sum()) == EXPECTED_GAMES and set(counts.index.astype(str)).issubset({"True", "False"}),
                int(counts.sum()),
                EXPECTED_GAMES,
            )
    add_check(
        rows,
        "no_previous_history_not_in_inactive_flags",
        bool(
            (
                ~df["either_player_debut"].astype(bool)
                | (~df["either_player_inactive_365d"].astype(bool) & ~df["either_player_inactive_730d"].astype(bool))
            ).all()
        ),
        "checked",
        "debut matches do not enter inactive flags",
    )
    add_check(
        rows,
        "missing_date_information_not_deleted",
        int((df["subgroup_recent_365_activity"] == "Missing date information").sum()) > 0
        and int((df["subgroup_inactivity_gap"] == "Missing date information").sum()) > 0,
        f"recent_missing={(df['subgroup_recent_365_activity'] == 'Missing date information').sum()}; inactivity_missing={(df['subgroup_inactivity_gap'] == 'Missing date information').sum()}",
        "missing groups retained",
    )
    add_check(
        rows,
        "bootstrap_repetitions",
        bool((bootstrap["bootstrap_replications"] == BOOTSTRAP_REPS).all()),
        sorted(bootstrap["bootstrap_replications"].unique()),
        BOOTSTRAP_REPS,
    )
    add_check(rows, "prematch_rd_available", rd_available, rd_available, True, severity="info")
    missing_outputs = [str(path.relative_to(PROJECT_ROOT)) for path in output_paths if not path.exists()]
    add_check(rows, "all_requested_outputs_generated", not missing_outputs, "; ".join(missing_outputs), "none")
    return pd.DataFrame(rows)


def format_float(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_main_conclusions(key_results: pd.DataFrame) -> list[str]:
    """Build concise cautious meeting conclusions from the key results table."""

    conclusions = []
    for row in key_results.itertuples(index=False):
        evidence = "CI excludes 0" if row.delta_brier_ci_lower > 0 or row.delta_brier_ci_upper < 0 else "CI includes 0"
        direction = "favours Glicko" if row.delta_brier > 0 else "favours Elo"
        conclusions.append(
            f"- {row.subgroup}: games={int(row.games)}, Brier diff={format_float(row.delta_brier)}, "
            f"log-loss diff={format_float(row.delta_logloss)}, 95% Brier CI "
            f"[{format_float(row.delta_brier_ci_lower)}, {format_float(row.delta_brier_ci_upper)}]; "
            f"{direction}, {evidence}."
        )
    return conclusions[:8]


def write_markdown_summary(
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    bootstrap: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    adaptive_recovery: pd.DataFrame,
    key_results: pd.DataFrame,
    rd_available: bool,
    output_paths: list[Path],
) -> str:
    """Write the meeting-ready markdown summary."""

    overall_display = overall.set_index("model")
    glicko = overall_display.loc["Glicko_low"]
    elo = overall_display.loc["Validation_best_Elo"]
    adaptive = overall_display.loc["best_AdaptiveK"]
    main_row = pairwise.loc[(pairwise["subgroup_variable"] == "overall") & (pairwise["subgroup"] == "Overall")].iloc[0]
    main_brier_ci = find_bootstrap_row(bootstrap, "overall", "Overall", "delta_brier_glicko_vs_elo")
    main_logloss_ci = find_bootstrap_row(bootstrap, "overall", "Overall", "delta_logloss_glicko_vs_elo")

    inflation = pairwise.loc[(pairwise["subgroup_variable"] == "inactivity_gap")].copy()
    inflation_focus = inflation.sort_values("delta_brier_inflation", ascending=False).head(3)

    recovery_overall = adaptive_recovery.loc[adaptive_recovery["range"] == "overall"].iloc[0]
    conclusions = build_main_conclusions(key_results)

    lines = [
        "# Meeting 6 Step 2: Where Glicko Helps",
        "",
        "## Purpose",
        "",
        "This script merges validated pre-match features with existing meeting 5 prediction files to analyse where Glicko low-inflation improves over validation-best Elo.",
        "",
        "## Inputs and model alignment",
        "",
        f"- Prematch features: `{FEATURE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Fair Elo-vs-Glicko predictions: `{FAIR_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Adaptive-K predictions: `{ADAPTIVE_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}`",
        "- Probability orientation: fair and adaptive files store actual-winner probability; this script converts to neutral player-A probability using player IDs.",
        "- All model predictions are merged by `match_id == game_id == fcode`, not by row number.",
        "",
        "## Overall model comparison",
        "",
        f"- Glicko low inflation: Brier={format_float(glicko['brier'])}, log loss={format_float(glicko['log_loss'])}, accuracy={format_float(glicko['accuracy'])}.",
        f"- Validation-best Elo: Brier={format_float(elo['brier'])}, log loss={format_float(elo['log_loss'])}, accuracy={format_float(elo['accuracy'])}.",
        f"- Best adaptive-K Elo: Brier={format_float(adaptive['brier'])}, log loss={format_float(adaptive['log_loss'])}, accuracy={format_float(adaptive['accuracy'])}.",
        f"- Main paired Brier difference, Elo minus Glicko: {format_float(main_row['delta_brier_glicko_vs_elo'])}, 95% CI [{format_float(main_brier_ci['ci_lower'])}, {format_float(main_brier_ci['ci_upper'])}].",
        f"- Main paired log-loss difference, Elo minus Glicko: {format_float(main_row['delta_logloss_glicko_vs_elo'])}, 95% CI [{format_float(main_logloss_ci['ci_lower'])}, {format_float(main_logloss_ci['ci_upper'])}].",
        "",
        "## Where Glicko improves over validation-best Elo",
        "",
    ]
    top_groups = pairwise.loc[
        (pairwise["subgroup_kind"].isin(["player_category", "prediction_confidence", "rd_uncertainty"]))
        & (pairwise["games"] >= 50)
    ].sort_values("delta_brier_glicko_vs_elo", ascending=False).head(6)
    for row in top_groups.itertuples(index=False):
        ci = find_bootstrap_row(bootstrap, row.subgroup_variable, row.subgroup, "delta_brier_glicko_vs_elo")
        ci_text = f"[{format_float(ci['ci_lower'])}, {format_float(ci['ci_upper'])}]" if ci is not None else "NA"
        lines.append(
            f"- {row.subgroup_title} = {row.subgroup}: games={int(row.games)}, "
            f"Brier diff={format_float(row.delta_brier_glicko_vs_elo)}, 95% CI {ci_text}."
        )

    lines.extend(
        [
            "",
            "## Contribution of inactivity RD inflation",
            "",
        ]
    )
    for row in inflation_focus.itertuples(index=False):
        lines.append(
            f"- {row.subgroup}: games={int(row.games)}, Glicko C0 Brier - Glicko low Brier = {format_float(row.delta_brier_inflation)}."
        )

    lines.extend(
        [
            "",
            "## Comparison with adaptive-K Elo",
            "",
            f"- Overall adaptive-K Brier recovery ratio: {format_float(recovery_overall['improvement_recovered_brier'])} "
            f"(valid={bool(recovery_overall['recovery_ratio_valid_brier'])}).",
            f"- Overall adaptive-K log-loss recovery ratio: {format_float(recovery_overall['improvement_recovered_logloss'])} "
            f"(valid={bool(recovery_overall['recovery_ratio_valid_logloss'])}).",
            "",
            "## New and low-experience players",
            "",
        ]
    )
    for variable, subgroup in [
        ("flag_either_player_debut", "True"),
        ("flag_min_total_games_before_le5", "True"),
        ("flag_min_total_games_before_le20", "True"),
    ]:
        row = pairwise.loc[(pairwise["subgroup_variable"] == variable) & (pairwise["subgroup"] == subgroup)]
        if not row.empty:
            r = row.iloc[0]
            lines.append(
                f"- {r['subgroup_title']}: games={int(r['games'])}, Brier diff={format_float(r['delta_brier_glicko_vs_elo'])}, log-loss diff={format_float(r['delta_logloss_glicko_vs_elo'])}."
            )

    lines.extend(
        [
            "",
            "## Low recent activity and returning players",
            "",
        ]
    )
    for variable, subgroup in [
        ("flag_min_games_last_365_days_le5", "True"),
        ("flag_either_player_inactive_365d", "True"),
        ("flag_either_player_inactive_730d", "True"),
    ]:
        row = pairwise.loc[(pairwise["subgroup_variable"] == variable) & (pairwise["subgroup"] == subgroup)]
        if not row.empty:
            r = row.iloc[0]
            lines.append(
                f"- {r['subgroup_title']}: games={int(r['games'])}, Brier diff={format_float(r['delta_brier_glicko_vs_elo'])}, log-loss diff={format_float(r['delta_logloss_glicko_vs_elo'])}."
            )

    lines.extend(
        [
            "",
            "## Prediction confidence",
            "",
        ]
    )
    conf = pairwise.loc[pairwise["subgroup_variable"] == "elo_favourite_probability"]
    for row in conf.itertuples(index=False):
        lines.append(
            f"- Elo favourite probability {row.subgroup}: games={int(row.games)}, Brier diff={format_float(row.delta_brier_glicko_vs_elo)}."
        )

    lines.extend(
        [
            "",
            "## Pre-match RD analysis",
            "",
        ]
    )
    if rd_available:
        lines.append("- Reliable pre-match Glicko RD was available and RD quartile analysis was generated.")
    else:
        lines.append("- RD subgroup analysis was skipped because reliable pre-match RD was not available.")

    lines.extend(
        [
            "",
            "## Bootstrap uncertainty",
            "",
            f"- Bootstrap type is event-cluster when a subgroup has at least 10 events, with {BOOTSTRAP_REPS:,} replications.",
            "- Small groups are retained and marked with `small_sample_warning`.",
            "",
            "## Corrected calibration",
            "",
        ]
    )
    cal_overall = calibration_summary.loc[calibration_summary["range"] == "overall"].set_index("model")
    for alias in ["Glicko_low", "Validation_best_Elo", "best_AdaptiveK"]:
        if alias in cal_overall.index:
            lines.append(
                f"- {DISPLAY_BY_ALIAS[alias]} corrected calibration error: {format_float(cal_overall.loc[alias, 'corrected_calibration_error'])}."
            )

    lines.extend(
        [
            "",
            "## Main conclusions for Meeting 6",
            "",
            *conclusions,
            "",
            "## Limitations",
            "",
            "- Subgroup results are exploratory and should not be interpreted as causal proof.",
            "- Some debut and inactive groups are small, so confidence intervals can be wide.",
            "- The script uses fixed meeting5 model outputs and does not retune models on 2025 results.",
            "",
            "## Files written",
            "",
        ]
    )
    for path in output_paths:
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")

    markdown = "\n".join(lines)
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    discover_prediction_files()

    features = load_prematch_features()
    fair_predictions = load_and_normalise_fair_predictions()
    adaptive_predictions = load_and_normalise_adaptive_k_predictions()
    all_predictions = pd.concat([fair_predictions, adaptive_predictions], ignore_index=True)

    alignment_rows: list[dict[str, Any]] = []
    per_match, alignment = convert_probabilities_to_player_a(features, all_predictions, alignment_rows)
    per_match = calculate_per_match_scores(per_match)
    per_match, subgroup_specs, rd_qcuts, rd_available = build_subgroup_definitions(per_match)

    overall = validate_overall_metrics(per_match)
    subgroup_performance = calculate_subgroup_model_metrics(per_match, subgroup_specs)
    pairwise = calculate_pairwise_differences(per_match, subgroup_specs)
    bootstrap = run_event_cluster_bootstrap(per_match, subgroup_specs)
    calibration_summary, calibration_bins = calculate_corrected_calibration(per_match)
    adaptive_recovery = calculate_adaptive_k_recovery(per_match)
    key_results = build_key_results_table(pairwise, bootstrap, rd_available)
    figure_paths = create_figures(overall, bootstrap, rd_available)

    output_paths = [
        ALIGNMENT_CHECKS_PATH,
        PER_MATCH_SCORES_PATH,
        OVERALL_METRICS_PATH,
        SUBGROUP_MODEL_PERFORMANCE_PATH,
        SUBGROUP_PAIRWISE_PATH,
        BOOTSTRAP_CI_PATH,
        CALIBRATION_SUMMARY_PATH,
        CALIBRATION_BINS_PATH,
        ADAPTIVE_RECOVERY_PATH,
        KEY_RESULTS_PATH,
        VALIDATION_CHECKS_PATH,
        SUMMARY_MD_PATH,
        RD_CUTPOINTS_PATH,
        *figure_paths,
    ]

    validation = run_validation_checks(
        per_match,
        alignment,
        overall,
        subgroup_specs,
        bootstrap,
        output_paths,
        rd_available,
    )

    alignment.to_csv(ALIGNMENT_CHECKS_PATH, index=False, encoding="utf-8-sig")
    per_match.to_csv(PER_MATCH_SCORES_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    overall.to_csv(OVERALL_METRICS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    subgroup_performance.to_csv(SUBGROUP_MODEL_PERFORMANCE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    pairwise.to_csv(SUBGROUP_PAIRWISE_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    bootstrap.to_csv(BOOTSTRAP_CI_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    calibration_summary.to_csv(CALIBRATION_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    calibration_bins.to_csv(CALIBRATION_BINS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    adaptive_recovery.to_csv(ADAPTIVE_RECOVERY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    key_results.to_csv(KEY_RESULTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")

    output_paths = [
        ALIGNMENT_CHECKS_PATH,
        PER_MATCH_SCORES_PATH,
        OVERALL_METRICS_PATH,
        SUBGROUP_MODEL_PERFORMANCE_PATH,
        SUBGROUP_PAIRWISE_PATH,
        BOOTSTRAP_CI_PATH,
        CALIBRATION_SUMMARY_PATH,
        CALIBRATION_BINS_PATH,
        ADAPTIVE_RECOVERY_PATH,
        KEY_RESULTS_PATH,
        VALIDATION_CHECKS_PATH,
        SUMMARY_MD_PATH,
        RD_CUTPOINTS_PATH,
        *figure_paths,
    ]
    write_markdown_summary(
        overall,
        pairwise,
        bootstrap,
        calibration_summary,
        adaptive_recovery,
        key_results,
        rd_available,
        output_paths,
    )
    validation = run_validation_checks(
        per_match,
        alignment,
        overall,
        subgroup_specs,
        bootstrap,
        output_paths,
        rd_available,
    )
    validation.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")

    main_overall = pairwise.loc[(pairwise["subgroup_variable"] == "overall") & (pairwise["subgroup"] == "Overall")].iloc[0]
    print("Meeting 6 step 2 where-Glicko-helps analysis complete.")
    print(f"Rows analysed: {len(per_match):,}")
    print(f"Alignment checks passed: {int(alignment['passed'].sum())} / {len(alignment)}")
    print(f"Validation checks passed: {int(validation['passed'].sum())} / {len(validation)}")
    print(f"Overall delta Brier, Elo - Glicko: {main_overall['delta_brier_glicko_vs_elo']:.6f}")
    print(f"Overall delta log loss, Elo - Glicko: {main_overall['delta_logloss_glicko_vs_elo']:.6f}")
    print(f"RD subgroup available: {rd_available}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
