"""Apply methodological corrections and produce scientific robustness outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP29_SCORES_PATH = OUTPUT_DIR / "29_per_match_model_scores_2025.csv"
STEP29_OVERALL_METRICS_PATH = OUTPUT_DIR / "29_overall_model_metrics.csv"
STEP30_DEBUT_PATH = OUTPUT_DIR / "30_debut_player_perspective.csv"
STEP30_DEBUT_SUMMARY_PATH = OUTPUT_DIR / "30_debut_model_summary.csv"
STEP30_INIT_DIAG_PATH = OUTPUT_DIR / "30_initialisation_rating_scale_diagnostics.csv"
STEP30_EXCLUSION_PATH = OUTPUT_DIR / "30_overall_exclusion_robustness.csv"
STEP30_RETURNERS_PATH = OUTPUT_DIR / "30_returning_player_threshold_sensitivity.csv"
STEP30_ZERO_RECENT_PATH = OUTPUT_DIR / "30_zero_recent_activity_decomposition.csv"
STEP30_NO_DEBUT_SUBGROUP_PATH = OUTPUT_DIR / "30_no_debut_subgroup_results.csv"
STEP30_RD_QUARTILE_PATH = OUTPUT_DIR / "30_no_debut_rd_quartile_results.csv"
STEP30_CONFIDENCE_PATH = OUTPUT_DIR / "30_prediction_confidence_diagnostics.csv"
STEP30_KEY_RESULTS_PATH = OUTPUT_DIR / "30_key_diagnostic_results.csv"

FAIR_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_fair_elo_vs_glicko" / "meeting5_fair_elo_vs_glicko_predictions_2025.csv"
ADAPTIVE_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "meeting5_adaptive_k_elo" / "meeting5_adaptive_k_elo_predictions_2025.csv"

INPUT_VALIDATION_PATH = OUTPUT_DIR / "31_input_validation_checks.csv"
RATING_DISTRIBUTION_SUMMARY_PATH = OUTPUT_DIR / "31_rating_distribution_summary.csv"
UNIQUE_PLAYER_SNAPSHOT_PATH = OUTPUT_DIR / "31_unique_player_rating_snapshot.csv"
DEBUT_OPPONENT_RATING_SUMMARY_PATH = OUTPUT_DIR / "31_debut_opponent_rating_summary.csv"
STANDARD_BRIER_SUMMARY_PATH = OUTPUT_DIR / "31_standard_brier_decomposition_summary.csv"
STANDARD_BRIER_BINS_PATH = OUTPUT_DIR / "31_standard_brier_decomposition_bins.csv"
BRIER_BOOTSTRAP_PATH = OUTPUT_DIR / "31_brier_decomposition_bootstrap.csv"
DEBUT_MECHANISM_PATH = OUTPUT_DIR / "31_debut_probability_mechanism.csv"
DEBUT_MECHANISM_SUMMARY_PATH = OUTPUT_DIR / "31_debut_probability_mechanism_summary.csv"
RETURNER_CUMULATIVE_INFLATION_PATH = OUTPUT_DIR / "31_returner_cumulative_inflation_bootstrap.csv"
RETURNER_EXCLUSIVE_BINS_PATH = OUTPUT_DIR / "31_returner_exclusive_bins.csv"
MEETING6_FINAL_RESULTS_PATH = OUTPUT_DIR / "31_meeting6_final_results.csv"
FIGURE_MANIFEST_PATH = OUTPUT_DIR / "31_meeting6_figure_manifest.csv"
FINAL_VALIDATION_PATH = OUTPUT_DIR / "31_final_validation_checks.csv"

RANDOM_SEED = 20260714
BOOTSTRAP_REPS = 2_000
EXPECTED_GAMES = 11_379
EPS = 1e-15
INITIAL_RATING = 1500.0
VALIDATION_ELO_SCALE = 300.0

MODEL_ALIASES = ["Glicko_low", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0", "Default_Elo"]
DECOMPOSITION_MODELS = ["Glicko_low", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0", "Default_Elo"]
CALIBRATION_MODELS = ["Glicko_low", "Validation_best_Elo", "best_AdaptiveK", "Glicko_C0"]
MODEL_LABELS = {
    "Glicko_low": "Glicko low inflation",
    "Validation_best_Elo": "Validation-best Elo",
    "best_AdaptiveK": "Best adaptive-K",
    "Glicko_C0": "Glicko C0",
    "Default_Elo": "Default Elo",
}
FAIR_MODEL_MAP = {
    "Glicko_low": "Glicko_low_inflation_match_by_match",
    "Glicko_C0": "Glicko_C0_match_by_match",
    "Validation_best_Elo": "Validation_best_Elo",
    "Default_Elo": "Default_Elo",
}
ADAPTIVE_MODEL_MAP = {"best_AdaptiveK": "AdaptiveK_PreviousYearGames_Elo_scale300"}


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", detail: str = "", severity: str = "error") -> None:
    """Append a validation check row."""

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


def read_float_constant(name: str, text: str) -> float:
    """Read a simple float constant from Python source without importing it."""

    match = re.search(rf"^{name}\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {name} in glicko_core.py")
    return float(match.group(1))


def load_step29_scores() -> pd.DataFrame:
    """Load step 29 per-match scores and add an event cluster key."""

    df = pd.read_csv(STEP29_SCORES_PATH, low_memory=False)
    df["event_key"] = df["year"].astype(str) + "_" + df["event_id"].astype(str)
    return df


def add_no_debut_rd_quartiles(scores: pd.DataFrame) -> pd.DataFrame:
    """Rebuild RD quartiles within the no-debut sample for final diagnostics."""

    out = scores.copy()
    out["no_debut_rd_quartile_31"] = pd.NA
    mask = (~out["either_player_debut"].astype(bool)) & out["max_prematch_rd"].notna()
    labels = ["quartile_1", "quartile_2", "quartile_3", "quartile_4"]
    out.loc[mask, "no_debut_rd_quartile_31"] = pd.qcut(
        out.loc[mask, "max_prematch_rd"],
        q=4,
        labels=labels,
        duplicates="raise",
    ).astype(str)
    return out


def load_step30_outputs() -> dict[str, pd.DataFrame]:
    """Load step 30 diagnostic outputs used by this finalisation step."""

    return {
        "debut": pd.read_csv(STEP30_DEBUT_PATH, low_memory=False),
        "debut_summary": pd.read_csv(STEP30_DEBUT_SUMMARY_PATH),
        "initialisation": pd.read_csv(STEP30_INIT_DIAG_PATH),
        "exclusion": pd.read_csv(STEP30_EXCLUSION_PATH),
        "returners": pd.read_csv(STEP30_RETURNERS_PATH),
        "zero_recent": pd.read_csv(STEP30_ZERO_RECENT_PATH),
        "no_debut_subgroups": pd.read_csv(STEP30_NO_DEBUT_SUBGROUP_PATH),
        "rd_quartiles": pd.read_csv(STEP30_RD_QUARTILE_PATH),
        "confidence": pd.read_csv(STEP30_CONFIDENCE_PATH),
        "key_results": pd.read_csv(STEP30_KEY_RESULTS_PATH),
        "overall_metrics": pd.read_csv(STEP29_OVERALL_METRICS_PATH),
    }


def load_model_constants() -> dict[str, float | str]:
    """Read model constants and formula text from existing code."""

    text = (SCRIPT_DIR / "glicko_core.py").read_text(encoding="utf-8")
    return {
        "initial_rating": INITIAL_RATING,
        "glicko_default_rating": read_float_constant("DEFAULT_RATING", text),
        "glicko_default_rd": read_float_constant("DEFAULT_RD", text),
        "glicko_min_rd": read_float_constant("MIN_RD", text),
        "glicko_max_rd": read_float_constant("MAX_RD", text),
        "q": math.log(10.0) / 400.0,
        "expected_score_uses_opponent_rd": "def expected_score(rating: float, opponent_rating: float, opponent_rd: float)" in text,
        "formula": "1 / (1 + 10 ** (-g(opponent_rd) * (rating - opponent_rating) / 400))",
    }


def merge_pre_match_ratings(scores: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-match ratings/RDs from meeting 5 prediction files."""

    out = scores.copy()
    fair = pd.read_csv(FAIR_PREDICTIONS_PATH, low_memory=False)

    for alias, source_model in FAIR_MODEL_MAP.items():
        sub = fair.loc[fair["model"] == source_model].copy()
        if sub.empty:
            raise ValueError(f"Missing fair prediction rows for {source_model}")
        sub["match_id"] = sub["game_id"].astype(int)
        sub = sub[
            [
                "match_id",
                "winner",
                "loser",
                "pre_rating_winner",
                "pre_rating_loser",
                "pre_rd_winner",
                "pre_rd_loser",
            ]
        ]
        tmp = out[["match_id", "player_a_id", "player_b_id"]].merge(sub, on="match_id", how="left", validate="one_to_one")
        if tmp["winner"].isna().any():
            raise ValueError(f"Missing prediction alignment rows for {source_model}")
        a_is_winner = tmp["player_a_id"] == tmp["winner"]
        b_is_winner = tmp["player_b_id"] == tmp["winner"]
        if not (a_is_winner | b_is_winner).all():
            raise ValueError(f"Could not map winner to player A/B for {source_model}")
        out[f"rating_a_{alias}"] = np.where(a_is_winner, tmp["pre_rating_winner"], tmp["pre_rating_loser"])
        out[f"rating_b_{alias}"] = np.where(a_is_winner, tmp["pre_rating_loser"], tmp["pre_rating_winner"])
        out[f"rd_a_{alias}"] = np.where(a_is_winner, tmp["pre_rd_winner"], tmp["pre_rd_loser"])
        out[f"rd_b_{alias}"] = np.where(a_is_winner, tmp["pre_rd_loser"], tmp["pre_rd_winner"])

    adaptive = pd.read_csv(ADAPTIVE_PREDICTIONS_PATH, low_memory=False)
    for alias, source_model in ADAPTIVE_MODEL_MAP.items():
        sub = adaptive.loc[adaptive["model"] == source_model].copy()
        if sub.empty:
            raise ValueError(f"Missing adaptive prediction rows for {source_model}")
        sub["match_id"] = sub["game_id"].astype(int)
        sub = sub[["match_id", "winner", "loser", "pre_rating_winner", "pre_rating_loser"]]
        tmp = out[["match_id", "player_a_id", "player_b_id"]].merge(sub, on="match_id", how="left", validate="one_to_one")
        if tmp["winner"].isna().any():
            raise ValueError(f"Missing adaptive alignment rows for {source_model}")
        a_is_winner = tmp["player_a_id"] == tmp["winner"]
        out[f"rating_a_{alias}"] = np.where(a_is_winner, tmp["pre_rating_winner"], tmp["pre_rating_loser"])
        out[f"rating_b_{alias}"] = np.where(a_is_winner, tmp["pre_rating_loser"], tmp["pre_rating_winner"])

    return out


def validate_inputs(scores: pd.DataFrame, step30: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validate that inputs reproduce the fixed evaluation set and key results."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows", len(scores) == EXPECTED_GAMES, len(scores), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", scores["match_id"].duplicated().sum() == 0, int(scores["match_id"].duplicated().sum()), 0)
    add_check(rows, "outcome_a_binary", set(scores["outcome_a"].dropna().unique()).issubset({0, 1}), sorted(scores["outcome_a"].dropna().unique()), "0/1")

    for model in MODEL_ALIASES:
        p_col = f"p_a_{model}"
        add_check(rows, f"probability_present_{model}", scores[p_col].isna().sum() == 0, int(scores[p_col].isna().sum()), 0)
        add_check(
            rows,
            f"probability_range_{model}",
            scores[p_col].between(0.0, 1.0).all(),
            f"{scores[p_col].min():.12g} to {scores[p_col].max():.12g}",
            "[0,1]",
        )

    metrics_ref = step30["overall_metrics"].set_index("model")
    reproduced = 0
    for model in MODEL_ALIASES:
        brier = float(scores[f"brier_{model}"].mean())
        logloss = float(scores[f"logloss_{model}"].mean())
        accuracy = float(scores[f"correct_{model}"].mean())
        ref = metrics_ref.loc[model]
        ok = (
            abs(brier - float(ref["brier"])) <= 1e-12
            and abs(logloss - float(ref["log_loss"])) <= 1e-12
            and abs(accuracy - float(ref["accuracy"])) <= 1e-12
        )
        reproduced += int(ok)
    add_check(rows, "step29_overall_metrics_reproduced", reproduced == len(MODEL_ALIASES), f"{reproduced}/{len(MODEL_ALIASES)}", f"{len(MODEL_ALIASES)}/{len(MODEL_ALIASES)}")

    no_debut = int((~scores["either_player_debut"]).sum())
    exactly_one = int((scores["a_is_debut"] ^ scores["b_is_debut"]).sum())
    both_debut = int((scores["a_is_debut"] & scores["b_is_debut"]).sum())
    add_check(rows, "no_debut_games", no_debut == 11_305, no_debut, 11_305)
    add_check(rows, "exactly_one_debut_games", exactly_one == 72, exactly_one, 72)
    add_check(rows, "both_players_debut_games", both_debut == 2, both_debut, 2)
    return pd.DataFrame(rows)


def build_player_match_rating_observations(scores: pd.DataFrame) -> pd.DataFrame:
    """Build match-weighted established-player pre-match rating observations."""

    side_rows = []
    for side in ["a", "b"]:
        tmp = pd.DataFrame(
            {
                "match_id": scores["match_id"],
                "match_sequence": scores["match_sequence"],
                "match_date": scores["match_date"],
                "player_id": scores[f"player_{side}_id"],
                "player_side": side.upper(),
                "total_games_before": scores[f"{side}_total_games_before"],
                "rating_Glicko_low": scores[f"rating_{side}_Glicko_low"],
                "rating_Validation_best_Elo": scores[f"rating_{side}_Validation_best_Elo"],
            }
        )
        side_rows.append(tmp)

    wide = pd.concat(side_rows, ignore_index=True)
    established = wide.loc[wide["total_games_before"] > 0].copy()

    long_rows = []
    for model, col in [("Glicko_low", "rating_Glicko_low"), ("Validation_best_Elo", "rating_Validation_best_Elo")]:
        tmp = established[
            [
                "match_id",
                "match_sequence",
                "match_date",
                "player_id",
                "player_side",
                "total_games_before",
                col,
            ]
        ].rename(columns={col: "rating"})
        tmp["model"] = model
        tmp["observation_type"] = "match_weighted"
        long_rows.append(tmp)
    return pd.concat(long_rows, ignore_index=True)


def build_unique_player_first_appearance_snapshot(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep each player's first 2025 pre-match rating snapshot if they are established."""

    rows = []
    for side in ["a", "b"]:
        rows.append(
            pd.DataFrame(
                {
                    "player_id": scores[f"player_{side}_id"],
                    "first_match_id": scores["match_id"],
                    "first_match_sequence": scores["match_sequence"],
                    "first_match_date": scores["match_date"],
                    "player_side": side.upper(),
                    "total_games_before": scores[f"{side}_total_games_before"],
                    "rating_Glicko_low": scores[f"rating_{side}_Glicko_low"],
                    "rating_Validation_best_Elo": scores[f"rating_{side}_Validation_best_Elo"],
                    "rd_Glicko_low": scores[f"rd_{side}_Glicko_low"],
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    long = long.sort_values(["player_id", "first_match_sequence", "first_match_id"])
    first = long.groupby("player_id", as_index=False).first()
    first = first.loc[first["total_games_before"] > 0].copy()
    first["observation_type"] = "unique_player_first_2025_appearance"
    return first


def percentile_of_initial(values: pd.Series) -> float:
    """Return percentile rank of the initial rating within a distribution."""

    values = values.dropna()
    if values.empty:
        return np.nan
    return float((values <= INITIAL_RATING).mean() * 100.0)


def summarise_rating_values(values: pd.Series, model: str, observation_type: str, observations: int, unique_players: int) -> dict[str, Any]:
    """Summarise one rating distribution."""

    values = values.dropna().astype(float)
    return {
        "model": model,
        "model_display": MODEL_LABELS.get(model, model),
        "observation_type": observation_type,
        "observations": int(observations),
        "unique_players": int(unique_players),
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "p10": values.quantile(0.10),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
        "max": values.max(),
        "initial_rating": INITIAL_RATING,
        "initial_percentile": percentile_of_initial(values),
        "initial_minus_median": INITIAL_RATING - values.median(),
    }


def build_rating_distribution_summary(match_obs: pd.DataFrame, unique_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Summarise match-weighted and unique-player rating distributions."""

    rows = []
    for model in ["Glicko_low", "Validation_best_Elo"]:
        obs = match_obs.loc[match_obs["model"] == model]
        rows.append(
            summarise_rating_values(
                obs["rating"],
                model,
                "match_weighted_pre_match_rating_observations",
                observations=len(obs),
                unique_players=obs["player_id"].nunique(),
            )
        )
        rating_col = f"rating_{model}"
        rows.append(
            summarise_rating_values(
                unique_snapshot[rating_col],
                model,
                "unique_player_first_2025_appearance",
                observations=len(unique_snapshot),
                unique_players=unique_snapshot["player_id"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def build_debut_opponent_rating_distribution(debut: pd.DataFrame) -> pd.DataFrame:
    """Summarise ratings of experienced opponents faced by debut players."""

    rows = []
    for model, col in [
        ("Glicko_low", "opponent_rating_Glicko_low"),
        ("Validation_best_Elo", "opponent_rating_Validation_best_Elo"),
    ]:
        values = debut[col].dropna().astype(float)
        rows.append(
            {
                "model": model,
                "model_display": MODEL_LABELS[model],
                "count": int(len(values)),
                "unique_opponents": int(debut["experienced_opponent_id"].nunique()),
                "mean": values.mean(),
                "median": values.median(),
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "initial_rating": INITIAL_RATING,
                "initial_minus_mean": INITIAL_RATING - values.mean(),
                "initial_minus_median": INITIAL_RATING - values.median(),
                "initial_percentile_within_opponent_distribution": percentile_of_initial(values),
            }
        )
    return pd.DataFrame(rows)


def brier_decomposition(df: pd.DataFrame, model: str, bin_width: float = 0.05) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate binned Murphy decomposition using common outcome_a."""

    if df.empty:
        raise ValueError(f"Cannot decompose empty sample for {model}")
    y = df["outcome_a"].astype(float).to_numpy()
    p = df[f"p_a_{model}"].astype(float).clip(EPS, 1 - EPS).to_numpy()
    n = len(df)
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
            reliability += weight * ((mean_p - empirical) ** 2)
            resolution += weight * ((empirical - event_rate) ** 2)
        else:
            mean_p = np.nan
            empirical = np.nan
            weight = 0.0
        bin_rows.append(
            {
                "model": model,
                "model_display": MODEL_LABELS[model],
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
        "model": model,
        "model_display": MODEL_LABELS[model],
        "games": n,
        "events": int(df["event_key"].nunique()),
        "overall_event_rate": event_rate,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstructed_brier": float(reconstructed),
        "actual_brier": actual_brier,
        "reconstruction_difference": float(reconstructed - actual_brier),
        "bin_width": bin_width,
    }
    return summary, pd.DataFrame(bin_rows)


def sample_definitions(scores: pd.DataFrame) -> dict[str, pd.Series]:
    """Return canonical masks used in step 31."""

    no_debut = ~scores["either_player_debut"].astype(bool)
    exactly_one_debut = scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)
    returning_365_no_debut = no_debut & scores["either_player_inactive_365d"].astype(bool)
    both_active_no_debut = no_debut & scores["both_players_active_last_365d"].astype(bool)
    return {
        "Overall": pd.Series(True, index=scores.index),
        "No debut": no_debut,
        "Exactly one debut": exactly_one_debut,
        "Returning >=365 days, no debut": returning_365_no_debut,
        "Both active and no debut": both_active_no_debut,
    }


def calculate_standard_murphy_decomposition(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate standard player-A Murphy decomposition for key samples."""

    summary_rows = []
    bin_frames = []
    for sample, mask in sample_definitions(scores).items():
        sample_df = scores.loc[mask].copy()
        for model in DECOMPOSITION_MODELS:
            summary, bins = brier_decomposition(sample_df, model, bin_width=0.05)
            summary["sample"] = sample
            bins["sample"] = sample
            summary_rows.append(summary)
            bin_frames.append(bins)
    return pd.DataFrame(summary_rows), pd.concat(bin_frames, ignore_index=True)


def event_cluster_bootstrap_positions(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Draw one event-cluster bootstrap vector of row positions."""

    grouped = [np.asarray(pos, dtype=int) for pos in df.reset_index(drop=True).groupby(df["event_key"].to_numpy(), sort=False).indices.values()]
    draw = rng.integers(0, len(grouped), size=len(grouped))
    return np.concatenate([grouped[i] for i in draw])


def murphy_arrays(y: np.ndarray, p: np.ndarray, bin_width: float = 0.05) -> dict[str, float]:
    """Fast Murphy decomposition from NumPy arrays."""

    y = y.astype(float)
    p = np.clip(p.astype(float), EPS, 1 - EPS)
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
        reliability += weight * ((mean_p - empirical) ** 2)
        resolution += weight * ((empirical - event_rate) ** 2)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "actual_brier": actual_brier,
    }


def bootstrap_decomposition_differences(scores: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap reliability, resolution and Brier differences for key samples."""

    rows = []
    for sample in ["Overall", "No debut", "Exactly one debut", "Both active and no debut"]:
        rng = np.random.default_rng(RANDOM_SEED + len(rows) * 101)
        mask = sample_definitions(scores)[sample]
        sample_df = scores.loc[mask].reset_index(drop=True).copy()
        g_summary, _ = brier_decomposition(sample_df, "Glicko_low")
        e_summary, _ = brier_decomposition(sample_df, "Validation_best_Elo")
        y = sample_df["outcome_a"].to_numpy(dtype=float)
        p_g = sample_df["p_a_Glicko_low"].to_numpy(dtype=float)
        p_e = sample_df["p_a_Validation_best_Elo"].to_numpy(dtype=float)

        rel_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        res_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        brier_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
        for _ in range(BOOTSTRAP_REPS):
            pos = event_cluster_bootstrap_positions(sample_df, rng)
            g_boot = murphy_arrays(y[pos], p_g[pos])
            e_boot = murphy_arrays(y[pos], p_e[pos])
            rel_diff[_] = e_boot["reliability"] - g_boot["reliability"]
            res_diff[_] = g_boot["resolution"] - e_boot["resolution"]
            brier_diff[_] = e_boot["actual_brier"] - g_boot["actual_brier"]

        rows.append(
            {
                "sample": sample,
                "games": len(sample_df),
                "events": int(sample_df["event_key"].nunique()),
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
            }
        )
    return pd.DataFrame(rows)


def glicko_g(rd: pd.Series | np.ndarray | float, q: float) -> pd.Series | np.ndarray | float:
    """Glicko g(RD) scaling function."""

    return 1.0 / np.sqrt(1.0 + (3.0 * (q**2) * (np.asarray(rd, dtype=float) ** 2)) / (math.pi**2))


def glicko_expected_score(rating: pd.Series, opponent_rating: pd.Series, opponent_rd: pd.Series, q: float) -> pd.Series:
    """Expected score using the opponent's RD, matching glicko_core.py."""

    g_rd = glicko_g(opponent_rd, q)
    exponent = -g_rd * (rating.astype(float) - opponent_rating.astype(float)) / 400.0
    return pd.Series(1.0 / (1.0 + 10.0**exponent), index=rating.index)


def elo_expected_score(rating: pd.Series, opponent_rating: pd.Series, scale: float = VALIDATION_ELO_SCALE) -> pd.Series:
    """Validation-best Elo expected score with scale=300."""

    exponent = -(rating.astype(float) - opponent_rating.astype(float)) / scale
    return 1.0 / (1.0 + 10.0**exponent)


def calculate_debut_probability_mechanism(debut: pd.DataFrame, constants: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct debut-player probabilities from formula components."""

    df = debut.copy()
    q = float(constants["q"])

    for alias in ["Glicko_low", "Glicko_C0"]:
        df[f"rating_difference_{alias}"] = df[f"debut_rating_{alias}"] - df[f"opponent_rating_{alias}"]
        df[f"g_opponent_rd_{alias}"] = glicko_g(df[f"opponent_rd_{alias}"], q)
        df[f"g_debut_rd_when_debut_is_opponent_{alias}"] = glicko_g(df[f"debut_rd_{alias}"], q)
        df[f"direct_debut_perspective_p_{alias}"] = glicko_expected_score(
            df[f"debut_rating_{alias}"],
            df[f"opponent_rating_{alias}"],
            df[f"opponent_rd_{alias}"],
            q,
        )
        opponent_beats_debut = glicko_expected_score(
            df[f"opponent_rating_{alias}"],
            df[f"debut_rating_{alias}"],
            df[f"debut_rd_{alias}"],
            q,
        )
        # Meeting 5 stores actual-winner probabilities. If the debut player lost,
        # the saved debut probability is the complement of the experienced
        # opponent's expected score, not the direct debut-perspective score.
        df[f"reconstructed_p_debut_{alias}"] = np.where(
            df["debut_player_won"].astype(bool),
            df[f"direct_debut_perspective_p_{alias}"],
            1.0 - opponent_beats_debut,
        )
        df[f"abs_reconstruction_error_{alias}"] = (df[f"reconstructed_p_debut_{alias}"] - df[f"p_debut_{alias}"]).abs()

    df["rating_difference_Validation_best_Elo"] = df["debut_rating_Validation_best_Elo"] - df["opponent_rating_Validation_best_Elo"]
    df["direct_debut_perspective_p_Validation_best_Elo"] = elo_expected_score(
        df["debut_rating_Validation_best_Elo"],
        df["opponent_rating_Validation_best_Elo"],
    )
    opponent_beats_debut_elo = elo_expected_score(
        df["opponent_rating_Validation_best_Elo"],
        df["debut_rating_Validation_best_Elo"],
    )
    df["reconstructed_p_debut_Validation_best_Elo"] = np.where(
        df["debut_player_won"].astype(bool),
        df["direct_debut_perspective_p_Validation_best_Elo"],
        1.0 - opponent_beats_debut_elo,
    )
    df["abs_reconstruction_error_Validation_best_Elo"] = (
        df["reconstructed_p_debut_Validation_best_Elo"] - df["p_debut_Validation_best_Elo"]
    ).abs()
    df["glicko_low_minus_c0_probability"] = df["p_debut_Glicko_low"] - df["p_debut_Glicko_C0"]
    df["glicko_low_minus_c0_opponent_rating"] = df["opponent_rating_Glicko_low"] - df["opponent_rating_Glicko_C0"]
    df["glicko_low_minus_c0_opponent_rd"] = df["opponent_rd_Glicko_low"] - df["opponent_rd_Glicko_C0"]
    df["debut_own_rd_used_directly_in_expected_score"] = False
    df["debut_own_rd_enters_saved_probability_when_debut_loses"] = ~df["debut_player_won"].astype(bool)

    columns = [
        "match_id",
        "event_key",
        "match_date",
        "debut_player_id",
        "experienced_opponent_id",
        "debut_player_won",
        "debut_rating_Glicko_low",
        "opponent_rating_Glicko_low",
        "rating_difference_Glicko_low",
        "opponent_rd_Glicko_low",
        "g_opponent_rd_Glicko_low",
        "g_debut_rd_when_debut_is_opponent_Glicko_low",
        "direct_debut_perspective_p_Glicko_low",
        "p_debut_Glicko_low",
        "reconstructed_p_debut_Glicko_low",
        "debut_rating_Glicko_C0",
        "opponent_rating_Glicko_C0",
        "rating_difference_Glicko_C0",
        "opponent_rd_Glicko_C0",
        "g_opponent_rd_Glicko_C0",
        "g_debut_rd_when_debut_is_opponent_Glicko_C0",
        "direct_debut_perspective_p_Glicko_C0",
        "p_debut_Glicko_C0",
        "reconstructed_p_debut_Glicko_C0",
        "debut_rating_Validation_best_Elo",
        "opponent_rating_Validation_best_Elo",
        "rating_difference_Validation_best_Elo",
        "direct_debut_perspective_p_Validation_best_Elo",
        "p_debut_Validation_best_Elo",
        "reconstructed_p_debut_Validation_best_Elo",
        "glicko_low_minus_c0_probability",
        "glicko_low_minus_c0_opponent_rating",
        "glicko_low_minus_c0_opponent_rd",
        "debut_own_rd_used_directly_in_expected_score",
        "debut_own_rd_enters_saved_probability_when_debut_loses",
    ]

    summary_rows = []
    for alias in ["Glicko_low", "Glicko_C0"]:
        summary_rows.append(
            {
                "model": alias,
                "model_display": MODEL_LABELS[alias],
                "games": len(df),
                "mean_rating_difference": df[f"rating_difference_{alias}"].mean(),
                "median_rating_difference": df[f"rating_difference_{alias}"].median(),
                "mean_opponent_rd": df[f"opponent_rd_{alias}"].mean(),
                "median_opponent_rd": df[f"opponent_rd_{alias}"].median(),
                "mean_g_opponent_rd": df[f"g_opponent_rd_{alias}"].mean(),
                "median_g_opponent_rd": df[f"g_opponent_rd_{alias}"].median(),
                "mean_reconstructed_probability": df[f"reconstructed_p_debut_{alias}"].mean(),
                "median_reconstructed_probability": df[f"reconstructed_p_debut_{alias}"].median(),
                "mean_direct_debut_perspective_probability": df[f"direct_debut_perspective_p_{alias}"].mean(),
                "median_direct_debut_perspective_probability": df[f"direct_debut_perspective_p_{alias}"].median(),
                "max_abs_reconstruction_error": df[f"abs_reconstruction_error_{alias}"].max(),
                "expected_score_uses_opponent_rd": bool(constants["expected_score_uses_opponent_rd"]),
                "debut_own_rd_directly_enters_expected_score": False,
                "debut_own_rd_enters_saved_probability_for_debut_losses": bool((~df["debut_player_won"].astype(bool)).any()),
            }
        )
    summary_rows.append(
        {
            "model": "Validation_best_Elo",
            "model_display": MODEL_LABELS["Validation_best_Elo"],
            "games": len(df),
            "mean_rating_difference": df["rating_difference_Validation_best_Elo"].mean(),
            "median_rating_difference": df["rating_difference_Validation_best_Elo"].median(),
            "mean_opponent_rd": np.nan,
            "median_opponent_rd": np.nan,
            "mean_g_opponent_rd": np.nan,
            "median_g_opponent_rd": np.nan,
            "mean_reconstructed_probability": df["reconstructed_p_debut_Validation_best_Elo"].mean(),
            "median_reconstructed_probability": df["reconstructed_p_debut_Validation_best_Elo"].median(),
            "mean_direct_debut_perspective_probability": df["direct_debut_perspective_p_Validation_best_Elo"].mean(),
            "median_direct_debut_perspective_probability": df["direct_debut_perspective_p_Validation_best_Elo"].median(),
            "max_abs_reconstruction_error": df["abs_reconstruction_error_Validation_best_Elo"].max(),
            "expected_score_uses_opponent_rd": np.nan,
            "debut_own_rd_directly_enters_expected_score": np.nan,
            "debut_own_rd_enters_saved_probability_for_debut_losses": np.nan,
        }
    )
    summary_rows.append(
        {
            "model": "Glicko_low_minus_C0",
            "model_display": "Glicko low minus Glicko C0",
            "games": len(df),
            "mean_rating_difference": df["glicko_low_minus_c0_opponent_rating"].mean(),
            "median_rating_difference": df["glicko_low_minus_c0_opponent_rating"].median(),
            "mean_opponent_rd": df["glicko_low_minus_c0_opponent_rd"].mean(),
            "median_opponent_rd": df["glicko_low_minus_c0_opponent_rd"].median(),
            "mean_g_opponent_rd": (df["g_opponent_rd_Glicko_low"] - df["g_opponent_rd_Glicko_C0"]).mean(),
            "median_g_opponent_rd": (df["g_opponent_rd_Glicko_low"] - df["g_opponent_rd_Glicko_C0"]).median(),
            "mean_reconstructed_probability": df["glicko_low_minus_c0_probability"].mean(),
            "median_reconstructed_probability": df["glicko_low_minus_c0_probability"].median(),
            "mean_direct_debut_perspective_probability": (df["direct_debut_perspective_p_Glicko_low"] - df["direct_debut_perspective_p_Glicko_C0"]).mean(),
            "median_direct_debut_perspective_probability": (df["direct_debut_perspective_p_Glicko_low"] - df["direct_debut_perspective_p_Glicko_C0"]).median(),
            "max_abs_reconstruction_error": np.nan,
            "expected_score_uses_opponent_rd": True,
            "debut_own_rd_directly_enters_expected_score": False,
            "debut_own_rd_enters_saved_probability_for_debut_losses": True,
        }
    )
    return df[columns], pd.DataFrame(summary_rows)


def paired_point_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate paired Glicko-vs-Elo and inflation metrics for one subset."""

    return {
        "games": int(len(df)),
        "events": int(df["event_key"].nunique()) if len(df) else 0,
        "glicko_brier": df["brier_Glicko_low"].mean(),
        "elo_brier": df["brier_Validation_best_Elo"].mean(),
        "delta_brier": df["brier_Validation_best_Elo"].mean() - df["brier_Glicko_low"].mean(),
        "glicko_logloss": df["logloss_Glicko_low"].mean(),
        "elo_logloss": df["logloss_Validation_best_Elo"].mean(),
        "delta_logloss": df["logloss_Validation_best_Elo"].mean() - df["logloss_Glicko_low"].mean(),
        "inflation_delta_brier": df["brier_Glicko_C0"].mean() - df["brier_Glicko_low"].mean(),
        "adaptive_delta_brier": df["brier_best_AdaptiveK"].mean() - df["brier_Glicko_low"].mean(),
    }


def bootstrap_paired_metrics(df: pd.DataFrame, reps: int = BOOTSTRAP_REPS, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Event-cluster bootstrap paired score differences."""

    if df.empty:
        return {
            "delta_brier_ci_lower": np.nan,
            "delta_brier_ci_upper": np.nan,
            "delta_logloss_ci_lower": np.nan,
            "delta_logloss_ci_upper": np.nan,
            "inflation_delta_brier_ci_lower": np.nan,
            "inflation_delta_brier_ci_upper": np.nan,
            "bootstrap_replications": reps,
        }
    rng = np.random.default_rng(seed)
    work = df[["event_key"]].copy()
    work["delta_brier"] = df["brier_Validation_best_Elo"] - df["brier_Glicko_low"]
    work["delta_logloss"] = df["logloss_Validation_best_Elo"] - df["logloss_Glicko_low"]
    work["inflation_brier"] = df["brier_Glicko_C0"] - df["brier_Glicko_low"]
    event_sums = work.groupby("event_key", sort=False)[["delta_brier", "delta_logloss", "inflation_brier"]].sum().to_numpy(dtype=float)
    event_counts = work.groupby("event_key", sort=False).size().to_numpy(dtype=float)

    draws = rng.integers(0, len(event_counts), size=(reps, len(event_counts)))
    sampled_counts = event_counts[draws].sum(axis=1)
    sampled_sums = event_sums[draws].sum(axis=1)
    delta_brier = sampled_sums[:, 0] / sampled_counts
    delta_logloss = sampled_sums[:, 1] / sampled_counts
    inflation_brier = sampled_sums[:, 2] / sampled_counts
    return {
        "delta_brier_ci_lower": float(np.quantile(delta_brier, 0.025)),
        "delta_brier_ci_upper": float(np.quantile(delta_brier, 0.975)),
        "delta_logloss_ci_lower": float(np.quantile(delta_logloss, 0.025)),
        "delta_logloss_ci_upper": float(np.quantile(delta_logloss, 0.975)),
        "inflation_delta_brier_ci_lower": float(np.quantile(inflation_brier, 0.025)),
        "inflation_delta_brier_ci_upper": float(np.quantile(inflation_brier, 0.975)),
        "bootstrap_replications": reps,
    }


def build_returner_inflation_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cumulative and exclusive returner inflation tables with CIs."""

    no_debut = ~scores["either_player_debut"].astype(bool)
    valid_gap = no_debut & scores["both_players_have_history"].astype(bool) & scores["max_days_since_last_game"].notna()

    cumulative_rows = []
    for threshold in [180, 365, 540, 730, 1095]:
        mask = valid_gap & (scores["max_days_since_last_game"] >= threshold)
        df = scores.loc[mask].copy()
        point = paired_point_metrics(df)
        ci = bootstrap_paired_metrics(df, seed=RANDOM_SEED + threshold)
        cumulative_rows.append(
            {
                "subgroup": f"Returning >= {threshold} days, no debut",
                "threshold_days": threshold,
                **point,
                "delta_brier_inflation": point["inflation_delta_brier"],
                "delta_brier_inflation_ci_lower": ci["inflation_delta_brier_ci_lower"],
                "delta_brier_inflation_ci_upper": ci["inflation_delta_brier_ci_upper"],
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
        df = scores.loc[mask].copy()
        point = paired_point_metrics(df)
        ci = bootstrap_paired_metrics(df, seed=RANDOM_SEED + int(lower))
        exclusive_rows.append(
            {
                "subgroup": label,
                "lower_days": lower,
                "upper_days": upper,
                **point,
                "delta_brier_inflation": point["inflation_delta_brier"],
                "delta_brier_inflation_ci_lower": ci["inflation_delta_brier_ci_lower"],
                "delta_brier_inflation_ci_upper": ci["inflation_delta_brier_ci_upper"],
                "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
                "bootstrap_replications": BOOTSTRAP_REPS,
                "sample_type": "exclusive_inactivity_bin",
            }
        )
    return pd.DataFrame(cumulative_rows), pd.DataFrame(exclusive_rows)


def final_result_definitions(scores: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """Define final rows for the meeting results table."""

    no_debut = ~scores["either_player_debut"].astype(bool)
    exactly_one_debut = scores["a_is_debut"].astype(bool) ^ scores["b_is_debut"].astype(bool)
    return [
        ("primary", "Overall", "Overall", pd.Series(True, index=scores.index)),
        ("primary", "Overall excluding debut", "Overall excluding debut", no_debut),
        ("primary", "Exactly one debut", "Exactly one debut", exactly_one_debut),
        ("secondary", "New but not debut: 1-5 games", "New but not debut: 1-5 games", no_debut & scores["min_total_games_before"].between(1, 5, inclusive="both")),
        ("secondary", "Total previous games 21-50", "Total previous games 21-50", no_debut & scores["min_total_games_before"].between(21, 50, inclusive="both")),
        ("secondary", "Recent activity 6-15", "Recent activity 6-15", no_debut & scores["min_games_last_365_days"].between(6, 15, inclusive="both")),
        ("exploratory", "Returning >=365 days, no debut", "Returning >=365 days, no debut", no_debut & scores["either_player_inactive_365d"].astype(bool)),
        ("exploratory", "Returning >=730 days, no debut", "Returning >=730 days, no debut", no_debut & scores["either_player_inactive_730d"].astype(bool)),
        ("primary", "Both active and no debut", "Both active and no debut", no_debut & scores["both_players_active_last_365d"].astype(bool)),
        ("exploratory", "No-debut RD quartile 3", "No-debut RD quartile 3", no_debut & (scores["no_debut_rd_quartile_31"] == "quartile_3")),
        ("exploratory", "No-debut RD quartile 4", "No-debut RD quartile 4", no_debut & (scores["no_debut_rd_quartile_31"] == "quartile_4")),
    ]


def interpretation_flag(row: dict[str, Any], subgroup: str) -> str:
    """Assign the final interpretation flag for one result row."""

    if row["games"] < 50 or row["events"] < 10:
        return "small_sample"
    if subgroup == "Exactly one debut":
        return "initialisation_mismatch"
    if row["delta_brier_ci_lower"] > 0 and row["delta_logloss_ci_lower"] > 0:
        return "robust_glicko_advantage"
    if row["delta_brier_ci_upper"] < 0 and row["delta_logloss_ci_upper"] < 0:
        return "robust_elo_advantage"
    return "uncertain"


def meeting_message(row: dict[str, Any], subgroup: str) -> str:
    """Create a concise meeting-ready message for one final result."""

    if subgroup == "Exactly one debut":
        return "Validation-best Elo is clearly better in the debut subgroup; this is a stable initialisation mismatch diagnostic."
    if subgroup == "Overall":
        return "Glicko low inflation is better overall than validation-best Elo on paired Brier and log loss."
    if subgroup == "Overall excluding debut":
        return "After removing debut matches, the overall Glicko advantage becomes larger."
    if "Returning" in subgroup:
        return "RD inflation improves Glicko C0 for returners, but Glicko-vs-Elo evidence remains uncertain in this small subgroup."
    if "RD quartile" in subgroup:
        return "No-debut RD results support an uncertainty mechanism but are not monotonic."
    return "Exploratory subgroup result; use effect size and CI rather than a categorical claim."


def build_final_meeting_results(scores: pd.DataFrame) -> pd.DataFrame:
    """Build final compact Meeting 6 results table."""

    rows = []
    for i, (role, subgroup, label, mask) in enumerate(final_result_definitions(scores)):
        df = scores.loc[mask].copy()
        point = paired_point_metrics(df)
        ci = bootstrap_paired_metrics(df, seed=RANDOM_SEED + i * 17)
        row = {
            "analysis_role": role,
            "subgroup": label,
            **point,
            **ci,
            "small_sample_warning": bool(point["games"] < 50 or point["events"] < 10),
        }
        row["interpretation_flag"] = interpretation_flag(row, subgroup)
        row["meeting_message"] = meeting_message(row, subgroup)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_errorbar_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lower_col: str,
    upper_col: str,
    title: str,
    ylabel: str,
    path: Path,
    note: str | None = None,
    rotate: int = 25,
) -> Path:
    """Plot a categorical error-bar chart."""

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    x = np.arange(len(df))
    y = df[y_col].astype(float).to_numpy()
    yerr = np.vstack([y - df[lower_col].astype(float).to_numpy(), df[upper_col].astype(float).to_numpy() - y])
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, color="#2f5f8f", ecolor="#8aa8c7")
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str), rotation=rotate, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(df.itertuples(index=False)):
        games = getattr(row, "games", None)
        events = getattr(row, "events", None)
        label = f"n={int(games)}"
        if events is not None:
            label += f"\ne={int(events)}"
        ax.annotate(label, (i, y[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def create_rating_distribution_figures(unique_snapshot: pd.DataFrame, debut: pd.DataFrame) -> list[Path]:
    """Create corrected initial-rating distribution figures."""

    paths = []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(unique_snapshot["rating_Glicko_low"], bins=35, alpha=0.55, label="Glicko low inflation", color="#2f5f8f")
    ax.hist(unique_snapshot["rating_Validation_best_Elo"], bins=35, alpha=0.45, label="Validation-best Elo", color="#d07c2c")
    ax.axvline(INITIAL_RATING, color="#222222", linestyle="--", linewidth=1.4, label="Initial rating = 1500")
    ax.set_title("Initial rating vs unique established-player rating snapshot")
    ax.set_xlabel("Pre-match rating at first 2025 appearance")
    ax.set_ylabel("Unique established players")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "31_fig01_unique_player_rating_snapshot.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(debut["opponent_rating_Glicko_low"], bins=20, alpha=0.55, label="Glicko low opponent ratings", color="#2f5f8f")
    ax.hist(debut["opponent_rating_Validation_best_Elo"], bins=20, alpha=0.45, label="Validation-best Elo opponent ratings", color="#d07c2c")
    ax.axvline(INITIAL_RATING, color="#222222", linestyle="--", linewidth=1.4, label="Initial rating = 1500")
    ax.set_title("Initial rating relative to opponents faced by debut players")
    ax.set_xlabel("Experienced opponent pre-match rating")
    ax.set_ylabel("Exactly-one-debut matches")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "31_fig02_debut_opponent_rating_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def calibration_bins_for_figure(df: pd.DataFrame, model: str, bin_width: float) -> pd.DataFrame:
    """Build calibration bins for plotting with common player-A outcome."""

    _, bins = brier_decomposition(df, model, bin_width=bin_width)
    bins = bins.loc[bins["games"] > 0].copy()
    return bins


def create_calibration_figure(scores: pd.DataFrame, sample: str, mask: pd.Series, bin_width: float, path: Path, small_sample: bool = False) -> Path:
    """Create a common-outcome calibration figure."""

    sample_df = scores.loc[mask].copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {
        "Glicko_low": "#2f5f8f",
        "Validation_best_Elo": "#d07c2c",
        "best_AdaptiveK": "#5a8f5a",
        "Glicko_C0": "#8b5a9f",
    }
    for model in CALIBRATION_MODELS:
        bins = calibration_bins_for_figure(sample_df, model, bin_width)
        bins = bins.loc[bins["games"] >= (1 if small_sample else 10)].copy()
        if bins.empty:
            continue
        ax.plot(
            bins["mean_predicted_probability"],
            bins["empirical_player_a_win_rate"],
            marker="o",
            linewidth=1.4,
            label=MODEL_LABELS[model],
            color=colors[model],
        )
        if small_sample:
            for _, row in bins.iterrows():
                ax.annotate(str(int(row["games"])), (row["mean_predicted_probability"], row["empirical_player_a_win_rate"]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(sample)
    ax.set_xlabel("Mean predicted probability that player A wins")
    ax.set_ylabel("Empirical player A win rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if small_sample:
        fig.text(0.01, 0.01, "Small sample: point labels show bin games; bins use width 0.20.", fontsize=8)
    else:
        fig.text(0.01, 0.01, "Common player-A outcome; fixed probability bins.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def create_corrected_figures(
    scores: pd.DataFrame,
    unique_snapshot: pd.DataFrame,
    debut: pd.DataFrame,
    zero_recent: pd.DataFrame,
    cumulative_returners: pd.DataFrame,
    exclusive_returners: pd.DataFrame,
    exclusion: pd.DataFrame,
    rd_quartiles: pd.DataFrame,
) -> list[Path]:
    """Create all step 31 corrected figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = create_rating_distribution_figures(unique_snapshot, debut)

    masks = sample_definitions(scores)
    paths.append(
        create_calibration_figure(
            scores,
            "Overall calibration",
            masks["Overall"],
            0.05,
            FIGURE_DIR / "31_fig03_standard_calibration_overall.png",
        )
    )
    paths.append(
        create_calibration_figure(
            scores,
            "No-debut calibration",
            masks["No debut"],
            0.05,
            FIGURE_DIR / "31_fig04_standard_calibration_no_debut.png",
        )
    )
    paths.append(
        create_calibration_figure(
            scores,
            "Exactly-one-debut calibration (small sample)",
            masks["Exactly one debut"],
            0.20,
            FIGURE_DIR / "31_fig05_standard_calibration_debut.png",
            small_sample=True,
        )
    )

    zero_plot = zero_recent.copy()
    paths.append(
        plot_errorbar_table(
            zero_plot,
            "subgroup",
            "delta_brier",
            "delta_brier_ci_lower",
            "delta_brier_ci_upper",
            "No recent activity, debut status and missing-date decomposition",
            "Elo Brier - Glicko Brier",
            FIGURE_DIR / "31_fig06_zero_activity_decomposition_with_ci.png",
            note="Positive values mean Glicko low inflation has lower Brier. Missing date information is retained.",
            rotate=25,
        )
    )

    paths.append(
        plot_errorbar_table(
            cumulative_returners,
            "subgroup",
            "delta_brier_inflation",
            "delta_brier_inflation_ci_lower",
            "delta_brier_inflation_ci_upper",
            "Inactivity RD inflation contribution for cumulative returning thresholds",
            "Glicko C0 Brier - Glicko low Brier",
            FIGURE_DIR / "31_fig07_returner_inflation_with_ci.png",
            note="Nested cumulative samples, not mutually exclusive bins. Positive values mean low inflation improves on C0.",
            rotate=25,
        )
    )

    paths.append(
        plot_errorbar_table(
            exclusive_returners,
            "subgroup",
            "delta_brier_inflation",
            "delta_brier_inflation_ci_lower",
            "delta_brier_inflation_ci_upper",
            "Inactivity RD inflation contribution for exclusive returning bins",
            "Glicko C0 Brier - Glicko low Brier",
            FIGURE_DIR / "31_fig08_returner_inflation_exclusive_bins.png",
            note="Mutually exclusive inactivity bins; small samples are retained.",
            rotate=25,
        )
    )

    order = [
        "All games",
        "Excluding all debut games",
        "Excluding debut and inactive >=365 games",
        "Both players have history",
        "Both players have at least 5 previous games",
        "Both players have at least 20 previous games",
        "Both active in last 365 days and no debut",
    ]
    exclusion_plot = exclusion.set_index("subgroup").loc[order].reset_index()
    paths.append(
        plot_errorbar_table(
            exclusion_plot,
            "subgroup",
            "delta_brier",
            "delta_brier_ci_lower",
            "delta_brier_ci_upper",
            "Overall exclusion robustness with event-cluster 95% CIs",
            "Elo Brier - Glicko Brier",
            FIGURE_DIR / "31_fig09_overall_exclusion_robustness_with_ci.png",
            note="Positive values mean Glicko low inflation has lower Brier.",
            rotate=30,
        )
    )

    rd_plot = rd_quartiles.copy()
    paths.append(
        plot_errorbar_table(
            rd_plot,
            "subgroup",
            "delta_brier",
            "delta_brier_ci_lower",
            "delta_brier_ci_upper",
            "No-debut RD quartiles with event-cluster 95% CIs",
            "Elo Brier - Glicko Brier",
            FIGURE_DIR / "31_fig10_no_debut_rd_quartiles_with_ci.png",
            note="Quartiles are rebuilt within the no-debut sample; the pattern is not monotonic.",
            rotate=0,
        )
    )
    return paths


def build_figure_manifest(paths: list[Path]) -> pd.DataFrame:
    """Create a manifest of figures recommended for main meeting or appendix."""

    rows = [
        {
            "figure_number": "M5/6-1",
            "filename": "outputs/meeting6/figures/29_fig01_overall_brier_zoomed.png",
            "title": "Overall model Brier score",
            "analysis_role": "primary",
            "main_message": "Glicko low inflation has the lowest overall Brier score.",
            "sample_size": EXPECTED_GAMES,
            "small_sample_warning": False,
            "recommended_for_main_meeting": True,
            "recommended_for_appendix": False,
        },
        {
            "figure_number": "30-1",
            "filename": "outputs/meeting6/figures/30_fig01_debut_probability_vs_actual.png",
            "title": "Debut predicted probability versus actual win rate",
            "analysis_role": "primary",
            "main_message": "Glicko overpredicts debut-player wins relative to Elo.",
            "sample_size": 72,
            "small_sample_warning": False,
            "recommended_for_main_meeting": True,
            "recommended_for_appendix": False,
        },
    ]
    metadata = {
        "31_fig01_unique_player_rating_snapshot.png": ("31-1", "Initial rating vs unique established-player rating snapshot", "diagnostic", "Shows initial rating relative to unique established players.", None, False, False, True),
        "31_fig02_debut_opponent_rating_distribution.png": ("31-2", "Initial rating relative to opponents faced by debut players", "primary", "Debut opponents are much lower-rated in Glicko, explaining high debut probabilities.", 72, False, True, False),
        "31_fig03_standard_calibration_overall.png": ("31-3", "Standard player-A calibration overall", "secondary", "Common-outcome calibration replaces model-specific favourite calibration.", EXPECTED_GAMES, False, True, False),
        "31_fig04_standard_calibration_no_debut.png": ("31-4", "Standard player-A calibration excluding debut", "secondary", "Calibration pattern after removing debut matches.", 11305, False, False, True),
        "31_fig05_standard_calibration_debut.png": ("31-5", "Standard player-A calibration for debut matches", "diagnostic", "Small-sample calibration view for debut matches.", 72, True, False, True),
        "31_fig06_zero_activity_decomposition_with_ci.png": ("31-6", "No recent activity, debut and missing-date decomposition", "primary", "The negative zero-activity result is driven by debut/no-history matches.", None, False, True, False),
        "31_fig07_returner_inflation_with_ci.png": ("31-7", "Returner inflation contribution with CIs", "primary", "RD inflation improves C0 for long-inactivity samples, with uncertainty shown.", None, False, True, False),
        "31_fig08_returner_inflation_exclusive_bins.png": ("31-8", "Returner inflation contribution in exclusive bins", "exploratory", "Exclusive inactivity bins show small-sample uncertainty.", None, True, False, True),
        "31_fig09_overall_exclusion_robustness_with_ci.png": ("31-9", "Overall exclusion robustness with CIs", "primary", "Glicko advantage remains and strengthens after excluding debut matches.", EXPECTED_GAMES, False, True, False),
        "31_fig10_no_debut_rd_quartiles_with_ci.png": ("31-10", "No-debut RD quartiles with CIs", "exploratory", "No-debut RD relationship is not monotonic.", 11305, False, False, True),
    }
    for path in paths:
        meta = metadata[path.name]
        rows.append(
            {
                "figure_number": meta[0],
                "filename": str(path.relative_to(PROJECT_ROOT)),
                "title": meta[1],
                "analysis_role": meta[2],
                "main_message": meta[3],
                "sample_size": meta[4],
                "small_sample_warning": meta[5],
                "recommended_for_main_meeting": meta[6],
                "recommended_for_appendix": meta[7],
            }
        )
    return pd.DataFrame(rows)


def run_final_validation_checks(
    scores: pd.DataFrame,
    unique_snapshot: pd.DataFrame,
    rating_summary: pd.DataFrame,
    debut_opponent_summary: pd.DataFrame,
    brier_summary: pd.DataFrame,
    brier_bootstrap: pd.DataFrame,
    debut_mechanism: pd.DataFrame,
    final_results: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
    constants: dict[str, Any],
) -> pd.DataFrame:
    """Run final validation checks for step 31 outputs."""

    rows: list[dict[str, Any]] = []
    add_check(rows, "input_rows", len(scores) == EXPECTED_GAMES, len(scores), EXPECTED_GAMES)
    add_check(rows, "match_id_unique", scores["match_id"].duplicated().sum() == 0, int(scores["match_id"].duplicated().sum()), 0)
    add_check(rows, "unique_player_snapshot_one_row_per_player", unique_snapshot["player_id"].duplicated().sum() == 0, int(unique_snapshot["player_id"].duplicated().sum()), 0)

    match_weighted = rating_summary.loc[rating_summary["observation_type"] == "match_weighted_pre_match_rating_observations"]
    unique_rows = rating_summary.loc[rating_summary["observation_type"] == "unique_player_first_2025_appearance"]
    add_check(
        rows,
        "match_weighted_and_unique_not_confused",
        (match_weighted["observations"] > match_weighted["unique_players"]).all() and (unique_rows["observations"] == unique_rows["unique_players"]).all(),
        "checked",
        "match-weighted observations > unique players; unique observations = unique players",
    )
    add_check(rows, "debut_opponent_count", (debut_opponent_summary["count"] == 72).all(), debut_opponent_summary["count"].tolist(), 72)

    same_uncertainty_ok = True
    for sample, group in brier_summary.groupby("sample"):
        same_uncertainty_ok = same_uncertainty_ok and (group["uncertainty"].max() - group["uncertainty"].min() < 1e-15)
    add_check(rows, "standard_decomposition_common_uncertainty", same_uncertainty_ok, "checked", "same uncertainty within each sample")
    add_check(
        rows,
        "standard_decomposition_reconstruction_close",
        brier_summary["reconstruction_difference"].abs().max() < 0.005,
        float(brier_summary["reconstruction_difference"].abs().max()),
        "<0.005",
    )
    max_recon_error = debut_mechanism[
        [
            "reconstructed_p_debut_Glicko_low",
            "p_debut_Glicko_low",
            "reconstructed_p_debut_Glicko_C0",
            "p_debut_Glicko_C0",
            "reconstructed_p_debut_Validation_best_Elo",
            "p_debut_Validation_best_Elo",
        ]
    ]
    max_error = max(
        (max_recon_error["reconstructed_p_debut_Glicko_low"] - max_recon_error["p_debut_Glicko_low"]).abs().max(),
        (max_recon_error["reconstructed_p_debut_Glicko_C0"] - max_recon_error["p_debut_Glicko_C0"]).abs().max(),
        (max_recon_error["reconstructed_p_debut_Validation_best_Elo"] - max_recon_error["p_debut_Validation_best_Elo"]).abs().max(),
    )
    add_check(rows, "expected_score_reconstructed_probabilities_match", max_error < 1e-10, float(max_error), "<1e-10")
    add_check(rows, "glicko_formula_uses_opponent_rd", bool(constants["expected_score_uses_opponent_rd"]), constants["expected_score_uses_opponent_rd"], True)
    add_check(rows, "debut_own_rd_not_used_directly_in_expected_score", not debut_mechanism["debut_own_rd_used_directly_in_expected_score"].any(), bool(debut_mechanism["debut_own_rd_used_directly_in_expected_score"].any()), False)
    add_check(rows, "bootstrap_repetitions", (brier_bootstrap["bootstrap_replications"] == BOOTSTRAP_REPS).all(), int(brier_bootstrap["bootstrap_replications"].min()), BOOTSTRAP_REPS)
    add_check(rows, "final_results_rows", len(final_results) == 11, len(final_results), 11)
    add_check(rows, "final_results_no_zero_game_rows", (final_results["games"] > 0).all(), int((final_results["games"] == 0).sum()), 0)
    add_check(rows, "all_new_tables_generated", all(path.exists() for path in output_paths), "checked", "all paths exist")
    add_check(rows, "all_new_figures_generated", all(path.exists() for path in figure_paths), "checked", "all figures exist")
    add_check(rows, "new_outputs_use_31_prefix", all(path.name.startswith("31_") for path in output_paths + figure_paths), "checked", "31_* outputs only")
    return pd.DataFrame(rows)


def main() -> None:
    """Run the step 31 finalisation pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    constants = load_model_constants()
    scores = load_step29_scores()
    scores = merge_pre_match_ratings(scores)
    scores = add_no_debut_rd_quartiles(scores)
    step30 = load_step30_outputs()
    input_validation = validate_inputs(scores, step30)

    match_obs = build_player_match_rating_observations(scores)
    unique_snapshot = build_unique_player_first_appearance_snapshot(scores)
    rating_summary = build_rating_distribution_summary(match_obs, unique_snapshot)
    debut_opponent_summary = build_debut_opponent_rating_distribution(step30["debut"])
    brier_summary, brier_bins = calculate_standard_murphy_decomposition(scores)
    brier_bootstrap = bootstrap_decomposition_differences(scores)
    debut_mechanism, debut_mechanism_summary = calculate_debut_probability_mechanism(step30["debut"], constants)
    cumulative_returners, exclusive_returners = build_returner_inflation_tables(scores)
    final_results = build_final_meeting_results(scores)

    figure_paths = create_corrected_figures(
        scores,
        unique_snapshot,
        step30["debut"],
        step30["zero_recent"],
        cumulative_returners,
        exclusive_returners,
        step30["exclusion"],
        step30["rd_quartiles"],
    )
    figure_manifest = build_figure_manifest(figure_paths)

    output_paths = [
        INPUT_VALIDATION_PATH,
        RATING_DISTRIBUTION_SUMMARY_PATH,
        UNIQUE_PLAYER_SNAPSHOT_PATH,
        DEBUT_OPPONENT_RATING_SUMMARY_PATH,
        STANDARD_BRIER_SUMMARY_PATH,
        STANDARD_BRIER_BINS_PATH,
        BRIER_BOOTSTRAP_PATH,
        DEBUT_MECHANISM_PATH,
        DEBUT_MECHANISM_SUMMARY_PATH,
        RETURNER_CUMULATIVE_INFLATION_PATH,
        RETURNER_EXCLUSIVE_BINS_PATH,
        MEETING6_FINAL_RESULTS_PATH,
        FIGURE_MANIFEST_PATH,
        FINAL_VALIDATION_PATH,
    ]

    input_validation.to_csv(INPUT_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    rating_summary.to_csv(RATING_DISTRIBUTION_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    unique_snapshot.to_csv(UNIQUE_PLAYER_SNAPSHOT_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut_opponent_summary.to_csv(DEBUT_OPPONENT_RATING_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    brier_summary.to_csv(STANDARD_BRIER_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    brier_bins.to_csv(STANDARD_BRIER_BINS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    brier_bootstrap.to_csv(BRIER_BOOTSTRAP_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut_mechanism.to_csv(DEBUT_MECHANISM_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    debut_mechanism_summary.to_csv(DEBUT_MECHANISM_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    cumulative_returners.to_csv(RETURNER_CUMULATIVE_INFLATION_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    exclusive_returners.to_csv(RETURNER_EXCLUSIVE_BINS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    final_results.to_csv(MEETING6_FINAL_RESULTS_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    figure_manifest.to_csv(FIGURE_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    validation = run_final_validation_checks(
        scores,
        unique_snapshot,
        rating_summary,
        debut_opponent_summary,
        brier_summary,
        brier_bootstrap,
        debut_mechanism,
        final_results,
        output_paths,
        figure_paths,
        constants,
    )
    validation.to_csv(FINAL_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    validation = run_final_validation_checks(
        scores,
        unique_snapshot,
        rating_summary,
        debut_opponent_summary,
        brier_summary,
        brier_bootstrap,
        debut_mechanism,
        final_results,
        output_paths,
        figure_paths,
        constants,
    )
    validation.to_csv(FINAL_VALIDATION_PATH, index=False, encoding="utf-8-sig")

    g_unique = rating_summary.loc[
        (rating_summary["model"] == "Glicko_low")
        & (rating_summary["observation_type"] == "unique_player_first_2025_appearance")
    ].iloc[0]
    overall = final_results.loc[final_results["subgroup"] == "Overall"].iloc[0]
    excl = final_results.loc[final_results["subgroup"] == "Overall excluding debut"].iloc[0]
    print("Meeting 6 step 4 finalisation complete.")
    print(f"Rows analysed: {len(scores):,}")
    print(f"Unique established players: {int(g_unique['unique_players']):,}")
    print(f"Overall delta Brier, Elo - Glicko: {overall['delta_brier']:.6f}")
    print(f"Excluding debut delta Brier, Elo - Glicko: {excl['delta_brier']:.6f}")
    print(f"Final validation checks passed: {int(validation['passed'].sum())} / {len(validation)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
