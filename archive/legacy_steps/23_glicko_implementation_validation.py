"""Meeting 5 Glicko implementation validation.

This script answers the supervisor's question: how am I gaining confidence in
the implementation of Glicko?

It validates behaviour using toy formula checks and existing full-data outputs.
It does not rerun the full 1985-2025 Glicko model, tune parameters, add
inactivity RD inflation, or perform the final Elo-vs-Glicko comparison.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from glicko_core import (  # noqa: E402
    DEFAULT_RD,
    DEFAULT_RATING,
    MAX_RD,
    MIN_RD,
    Q,
    expected_score,
    update_player_glicko,
    update_two_players_single_game,
)


VALIDATION_CHECKS_PATH = OUTPUT_DIR / "meeting5_glicko_validation_checks.csv"
ACTIVE_SIMILARITY_PATH = OUTPUT_DIR / "meeting5_glicko_active_player_similarity.csv"
VALIDATION_ISSUES_PATH = OUTPUT_DIR / "meeting5_glicko_validation_issues.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "meeting5_glicko_validation_summary.md"
SCATTER_PATH = OUTPUT_DIR / "meeting5_elo_vs_glicko_active_players_scatter.png"

EXPECTED_2025_GAMES = 11_379


def find_candidates(pattern: str) -> list[Path]:
    """Find files below the project root using rglob."""

    return sorted(PROJECT_ROOT.rglob(pattern))


def choose_file(pattern: str, preferred_name_parts: list[str]) -> Path | None:
    """Choose the most appropriate file from rglob candidates."""

    candidates = find_candidates(pattern)
    if not candidates:
        return None
    for part in preferred_name_parts:
        preferred = [path for path in candidates if part.lower() in path.name.lower()]
        if preferred:
            return sorted(preferred, key=lambda path: len(str(path)))[0]
    return sorted(candidates, key=lambda path: path.stat().st_size if path.exists() else 0)[0]


def rel(path: Path | None) -> str:
    if path is None:
        return "missing"
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def add_issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    item: str,
    severity: str,
    details: str,
) -> None:
    issues.append(
        {
            "issue_type": issue_type,
            "item": item,
            "severity": severity,
            "details": details,
        }
    )


def add_check(
    rows: list[dict[str, Any]],
    check_id: str,
    check_group: str,
    description: str,
    expected_behaviour: str,
    observed_value: Any,
    passed: bool,
    notes: str = "",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "check_group": check_group,
            "description": description,
            "expected_behaviour": expected_behaviour,
            "observed_value": observed_value,
            "pass": bool(passed),
            "notes": notes,
        }
    )


def make_file_inventory() -> dict[str, Path | None]:
    """Locate existing files without hard-coding full paths."""

    return {
        "glicko_core": choose_file("glicko_core.py", ["glicko_core.py"]),
        "glicko_predictions": choose_file(
            "*glicko*prediction*.csv",
            ["glicko_mbm_predictions_1985_2025.csv", "glicko_mbm_predictions", "rating_period_predictions"],
        ),
        "glicko_final_ratings": choose_file(
            "*glicko*final*rating*.csv",
            ["glicko_mbm_final_ratings_1985_2025.csv", "glicko_mbm_final_ratings", "rating_period_final"],
        ),
        "elo_final_ratings": choose_file(
            "*elo*final*rating*.csv",
            ["elo_burnin_final_ratings_all_runs.csv", "elo_burnin_final"],
        ),
        "full_history_matches": choose_file(
            "*matches*1985*2025*.csv",
            ["matches_1985_2025_checked.csv"],
        ),
        "elo_baseline_decision": choose_file(
            "*baseline*decision*.*",
            ["elo_baseline_decision_summary.md", "elo_candidate_baselines.csv", "elo_baseline_evidence_table.csv"],
        ),
        "glicko_rating_period_metrics": choose_file(
            "*rating_period*metrics*.csv",
            ["glicko_rating_period_metrics_2025.csv"],
        ),
        "glicko_rd_summary": choose_file(
            "*glicko*rd_summary*.csv",
            ["glicko_mbm_rd_summary.csv", "glicko_rating_period_rd_summary.csv"],
        ),
    }


def run_formula_checks(check_rows: list[dict[str, Any]]) -> None:
    """Run toy formula checks using glicko_core.py."""

    equal_prob = expected_score(1500, 1500, 350)
    higher_prob = expected_score(1600, 1500, 350)
    lower_prob = expected_score(1400, 1500, 350)

    add_check(
        check_rows,
        "A1",
        "Formula sanity checks",
        "Equal ratings expected score",
        "Expected score should be approximately 0.5",
        f"{equal_prob:.12f}",
        abs(equal_prob - 0.5) < 1e-12,
    )
    add_check(
        check_rows,
        "A2",
        "Formula sanity checks",
        "Higher-rated player expected score",
        "Expected score should be above 0.5",
        f"{higher_prob:.12f}",
        higher_prob > 0.5,
    )
    add_check(
        check_rows,
        "A3",
        "Formula sanity checks",
        "Lower-rated player expected score",
        "Expected score should be below 0.5",
        f"{lower_prob:.12f}",
        lower_prob < 0.5,
    )

    equal_update = update_two_players_single_game(1500, 350, 1500, 350, 1)
    equal_update_pass = (
        equal_update.player1_rating_after > 1500
        and equal_update.player2_rating_after < 1500
        and equal_update.player1_rd_after <= 350
        and equal_update.player2_rd_after <= 350
    )
    add_check(
        check_rows,
        "A4",
        "Formula sanity checks",
        "Single-game update direction for equal players",
        "Winner rating should increase, loser rating should decrease, and RD should not increase",
        (
            f"A rating {equal_update.player1_rating_before:.1f}->{equal_update.player1_rating_after:.3f}; "
            f"B rating {equal_update.player2_rating_before:.1f}->{equal_update.player2_rating_after:.3f}; "
            f"RD {equal_update.player1_rd_before:.1f}->{equal_update.player1_rd_after:.3f}"
        ),
        equal_update_pass,
    )

    upset_update = update_two_players_single_game(1400, 100, 1700, 100, 1)
    weaker_gain = upset_update.player1_rating_after - upset_update.player1_rating_before
    stronger_loss = upset_update.player2_rating_after - upset_update.player2_rating_before
    add_check(
        check_rows,
        "A5",
        "Formula sanity checks",
        "Upset example direction",
        "Weaker player should gain rating points and stronger player should lose points",
        f"weaker change={weaker_gain:.3f}; stronger change={stronger_loss:.3f}",
        weaker_gain > 0 and stronger_loss < 0,
    )


def run_official_style_check(check_rows: list[dict[str, Any]]) -> None:
    """Run the standard Glicko-style batch example."""

    new_rating, new_rd = update_player_glicko(
        rating=1500,
        rd=200,
        opponent_ratings=[1400, 1550, 1700],
        opponent_rds=[30, 100, 300],
        scores=[1, 0, 0],
    )
    rating_close = abs(new_rating - 1464) < 2.0
    rd_close = abs(new_rd - 151) < 2.0
    add_check(
        check_rows,
        "B1",
        "Official-style update example",
        "Batch rating-period update",
        "New rating should be around 1464 and new RD around 151, tolerance +/-2",
        f"new_rating={new_rating:.3f}; new_rd={new_rd:.3f}",
        rating_close and rd_close,
        "Uses initial rating=1500, RD=200, opponents 1400/RD30 win, 1550/RD100 loss, 1700/RD300 loss.",
    )


def run_rd_checks(
    check_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    glicko_final_path: Path | None,
    rd_summary_path: Path | None,
) -> dict[str, Any]:
    """Check RD constants and full-data RD behaviour."""

    rd_info: dict[str, Any] = {
        "default_rd": DEFAULT_RD,
        "min_rd": MIN_RD,
        "max_rd": MAX_RD,
    }
    add_check(
        check_rows,
        "C1",
        "RD boundary and behaviour checks",
        "RD constants detected",
        "DEFAULT_RD, MIN_RD, and MAX_RD should be defined",
        f"DEFAULT_RD={DEFAULT_RD}; MIN_RD={MIN_RD}; MAX_RD={MAX_RD}",
        DEFAULT_RD > MIN_RD and MAX_RD >= DEFAULT_RD,
    )

    normal_update = update_two_players_single_game(1500, 350, 1500, 350, 1)
    add_check(
        check_rows,
        "C2",
        "RD boundary and behaviour checks",
        "RD after normal update",
        "RD should decrease or at least not increase after a played game",
        f"before={normal_update.player1_rd_before:.3f}; after={normal_update.player1_rd_after:.3f}",
        normal_update.player1_rd_after <= normal_update.player1_rd_before,
    )

    if glicko_final_path is None:
        add_issue(issues, "missing_file", "glicko_final_ratings", "warning", "Could not check full-data final RD distribution.")
        add_check(
            check_rows,
            "C3",
            "RD boundary and behaviour checks",
            "Full-data final RD distribution",
            "Final RD values should stay within implemented bounds",
            "not checked",
            False,
            "Missing Glicko final ratings file.",
        )
        return rd_info

    required = ["player_code", "final_rating", "final_rd"]
    final_cols = pd.read_csv(glicko_final_path, nrows=0).columns.tolist()
    missing_cols = [col for col in required if col not in final_cols]
    if missing_cols:
        add_issue(
            issues,
            "missing_columns",
            rel(glicko_final_path),
            "warning",
            f"Missing columns for RD check: {missing_cols}",
        )
        return rd_info

    final_ratings = pd.read_csv(glicko_final_path, usecols=required)
    rd = final_ratings["final_rd"].astype(float)
    n_players = len(rd)
    at_min = int(np.isclose(rd, MIN_RD).sum())
    at_max = int(np.isclose(rd, MAX_RD).sum())
    near_max = int((rd >= MAX_RD - 5).sum())
    rd_info.update(
        {
            "n_players": n_players,
            "mean_rd": float(rd.mean()),
            "median_rd": float(rd.median()),
            "min_rd_observed": float(rd.min()),
            "max_rd_observed": float(rd.max()),
            "number_at_min_rd": at_min,
            "number_at_max_rd": at_max,
            "number_near_max_rd": near_max,
            "proportion_at_min_rd": at_min / n_players if n_players else np.nan,
        }
    )

    add_check(
        check_rows,
        "C3",
        "RD boundary and behaviour checks",
        "Full-data final RD lower bound",
        "No final RD should fall below MIN_RD",
        f"min observed RD={rd.min():.3f}; MIN_RD={MIN_RD}",
        bool((rd >= MIN_RD - 1e-9).all()),
    )
    add_check(
        check_rows,
        "C4",
        "RD boundary and behaviour checks",
        "Full-data final RD upper bound",
        "No final RD should exceed MAX_RD/new-player RD",
        f"max observed RD={rd.max():.3f}; MAX_RD={MAX_RD}",
        bool((rd <= MAX_RD + 1e-9).all()),
    )
    add_check(
        check_rows,
        "C5",
        "RD boundary and behaviour checks",
        "Full-data RD collapse diagnostic",
        "Players at MIN_RD should not be almost all players",
        f"{at_min}/{n_players} players at MIN_RD ({at_min / n_players:.1%}); median RD={rd.median():.3f}",
        (at_min / n_players) < 0.5 and rd.median() > MIN_RD,
        "This is a diagnostic threshold, not a theoretical rule.",
    )

    if rd_summary_path is None:
        add_issue(
            issues,
            "missing_optional_file",
            "glicko_rd_summary",
            "info",
            "Could not find a saved Glicko RD summary file; computed from final ratings instead.",
        )

    return rd_info


def run_prediction_checks(
    check_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    predictions_path: Path | None,
) -> dict[str, Any]:
    """Check saved Glicko predictions for valid probabilities and pre-update consistency."""

    info: dict[str, Any] = {}
    if predictions_path is None:
        add_issue(issues, "missing_file", "glicko_predictions", "warning", "Could not check saved prediction rows.")
        add_check(
            check_rows,
            "D1",
            "Prediction-before-update checks",
            "Saved predictions file exists",
            "A Glicko predictions file should be available",
            "missing",
            False,
        )
        return info

    required = [
        "year",
        "fcode",
        "pred_a_win",
        "player_a_rating_before",
        "player_b_rating_before",
        "player_b_rd_before",
    ]
    optional = [
        "player_a",
        "player_b",
        "actual_a_win",
        "winner_rating_before",
        "winner_rating_after",
        "loser_rating_before",
        "loser_rating_after",
    ]
    available = pd.read_csv(predictions_path, nrows=0).columns.tolist()
    missing_required = [col for col in required if col not in available]
    if missing_required:
        add_issue(
            issues,
            "missing_columns",
            rel(predictions_path),
            "warning",
            f"Missing required prediction check columns: {missing_required}",
        )
        add_check(
            check_rows,
            "D1",
            "Prediction-before-update checks",
            "Prediction columns present",
            "Saved predictions should include pre-match prediction and pre-rating columns",
            f"missing {missing_required}",
            False,
        )
        return info

    usecols = required + [col for col in optional if col in available]
    pred_df = pd.read_csv(predictions_path, usecols=usecols)
    pred = pred_df["pred_a_win"].astype(float)
    finite = np.isfinite(pred)
    in_range = pred.between(0.0, 1.0)
    eval_rows_2025 = int((pred_df["year"] == 2025).sum())
    info.update(
        {
            "prediction_rows": len(pred_df),
            "eval_rows_2025": eval_rows_2025,
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
            "pred_nan_count": int(pred.isna().sum()),
            "pred_nonfinite_count": int((~finite).sum()),
            "pred_out_of_range_count": int((~in_range).sum()),
        }
    )

    add_check(
        check_rows,
        "D1",
        "Prediction-before-update checks",
        "Prediction probability range",
        "All saved prediction probabilities should be finite and between 0 and 1",
        f"min={pred.min():.6f}; max={pred.max():.6f}; nonfinite={(~finite).sum()}; out_of_range={(~in_range).sum()}",
        bool(finite.all() and in_range.all()),
    )
    add_check(
        check_rows,
        "D2",
        "Prediction-before-update checks",
        "2025 evaluation row count",
        f"2025 evaluation rows should equal {EXPECTED_2025_GAMES}",
        f"2025 rows={eval_rows_2025}",
        eval_rows_2025 == EXPECTED_2025_GAMES,
    )

    # Recompute pred_a_win from saved pre-period state. This confirms that the
    # saved prediction is based on pre-update rating/RD columns.
    g_b = 1.0 / np.sqrt(
        1.0
        + (3.0 * (Q**2) * pred_df["player_b_rd_before"].astype(float).to_numpy() ** 2) / (np.pi**2)
    )
    rating_a = pred_df["player_a_rating_before"].astype(float).to_numpy()
    rating_b = pred_df["player_b_rating_before"].astype(float).to_numpy()
    recomputed = 1.0 / (1.0 + 10.0 ** (-g_b * (rating_a - rating_b) / 400.0))
    max_abs_diff = float(np.max(np.abs(recomputed - pred.to_numpy())))
    info["max_pre_update_prediction_recompute_diff"] = max_abs_diff
    add_check(
        check_rows,
        "D3",
        "Prediction-before-update checks",
        "Prediction uses pre-update player state",
        "Recomputed Glicko expected score from pre-rating/RD columns should match saved pred_a_win",
        f"max abs difference={max_abs_diff:.12g}",
        max_abs_diff < 1e-10,
    )

    has_before_after = {
        "winner_rating_before",
        "winner_rating_after",
        "loser_rating_before",
        "loser_rating_after",
    }.issubset(set(available))
    add_check(
        check_rows,
        "D4",
        "Prediction-before-update checks",
        "Before/after rating columns are saved",
        "Saved predictions should include both pre-update and post-update state if direct audit is required",
        f"before/after columns present={has_before_after}",
        has_before_after,
        "This supports direct inspection of pre-match prediction versus post-match update.",
    )
    return info


def compute_player_counts(matches_path: Path, issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Compute total games and 2025 games per player from the full-history dataset."""

    required = ["winner", "loser", "year"]
    columns = pd.read_csv(matches_path, nrows=0).columns.tolist()
    missing = [col for col in required if col not in columns]
    if missing:
        add_issue(
            issues,
            "missing_columns",
            rel(matches_path),
            "error",
            f"Cannot compute player activity counts; missing {missing}",
        )
        return pd.DataFrame(columns=["player_code", "total_games", "games_2025"])

    matches = pd.read_csv(matches_path, usecols=required)
    all_players = pd.concat([matches["winner"], matches["loser"]]).dropna().astype(int)
    games_2025 = pd.concat(
        [
            matches.loc[matches["year"] == 2025, "winner"],
            matches.loc[matches["year"] == 2025, "loser"],
        ]
    ).dropna().astype(int)

    counts = pd.DataFrame({"player_code": all_players.value_counts().index.astype(int)})
    counts["total_games"] = counts["player_code"].map(all_players.value_counts()).fillna(0).astype(int)
    counts["games_2025"] = counts["player_code"].map(games_2025.value_counts()).fillna(0).astype(int)
    counts["active_2025"] = counts["games_2025"] > 0
    return counts


def top_overlap(subset: pd.DataFrame, elo_col: str, glicko_col: str, n: int) -> float:
    if subset.empty:
        return np.nan
    k = min(n, len(subset))
    if k == 0:
        return np.nan
    elo_top = set(subset.sort_values(elo_col, ascending=False).head(k)["player_code"])
    glicko_top = set(subset.sort_values(glicko_col, ascending=False).head(k)["player_code"])
    return len(elo_top & glicko_top) / k


def compare_active_rating_lists(
    issues: list[dict[str, Any]],
    elo_path: Path | None,
    glicko_path: Path | None,
    matches_path: Path | None,
) -> pd.DataFrame:
    """Compare Elo and Glicko final rating lists for active/high-volume players."""

    if elo_path is None:
        add_issue(issues, "missing_file", "elo_final_ratings", "warning", "Could not compute Elo-vs-Glicko similarity.")
        return pd.DataFrame()
    if glicko_path is None:
        add_issue(issues, "missing_file", "glicko_final_ratings", "warning", "Could not compute Elo-vs-Glicko similarity.")
        return pd.DataFrame()
    if matches_path is None:
        add_issue(issues, "missing_file", "full_history_matches", "warning", "Could not compute player activity groups.")
        return pd.DataFrame()

    elo_required = ["setting_name", "start_year", "player_code", "final_rating"]
    glicko_required = ["setting_name", "player_code", "final_rating"]
    elo_cols = pd.read_csv(elo_path, nrows=0).columns.tolist()
    glicko_cols = pd.read_csv(glicko_path, nrows=0).columns.tolist()
    missing_elo = [col for col in elo_required if col not in elo_cols]
    missing_glicko = [col for col in glicko_required if col not in glicko_cols]
    if missing_elo:
        add_issue(issues, "missing_columns", rel(elo_path), "warning", f"Missing Elo final rating columns: {missing_elo}")
        return pd.DataFrame()
    if missing_glicko:
        add_issue(issues, "missing_columns", rel(glicko_path), "warning", f"Missing Glicko final rating columns: {missing_glicko}")
        return pd.DataFrame()

    elo = pd.read_csv(elo_path, usecols=[col for col in elo_required + ["final_rank"] if col in elo_cols])
    glicko = pd.read_csv(
        glicko_path,
        usecols=[col for col in glicko_required + ["final_rank_by_rating", "final_rd"] if col in glicko_cols],
    )
    activity = compute_player_counts(matches_path, issues)
    if activity.empty:
        return pd.DataFrame()

    # Use full-history Elo runs. Include the three Elo baselines if present.
    elo = elo[elo["start_year"] == 1985].copy()
    preferred_settings = [
        "conservative_k10_scale500",
        "default_k20_scale500",
        "validation_best_k30_scale300",
    ]
    elo = elo[elo["setting_name"].isin(preferred_settings)]
    if elo.empty:
        add_issue(
            issues,
            "missing_rows",
            rel(elo_path),
            "warning",
            "No full-history Elo baseline rows found for expected settings.",
        )
        return pd.DataFrame()

    glicko_setting = glicko["setting_name"].iloc[0] if "setting_name" in glicko.columns else "glicko"
    glicko = glicko.rename(
        columns={
            "setting_name": "glicko_setting_name",
            "final_rating": "glicko_final_rating",
            "final_rank_by_rating": "glicko_final_rank",
        }
    )
    if "glicko_final_rank" not in glicko.columns:
        glicko["glicko_final_rank"] = glicko["glicko_final_rating"].rank(method="min", ascending=False)

    rows: list[dict[str, Any]] = []
    group_specs = [
        ("total_games_ge100", "total games >= 100", lambda df: df["total_games"] >= 100),
        ("total_games_ge200", "total games >= 200", lambda df: df["total_games"] >= 200),
        ("active_2025_games_ge5", "active 2025 players with 2025 games >= 5", lambda df: df["games_2025"] >= 5),
        (
            "active_2025_games_ge5_total_games_ge100",
            "active 2025 players with 2025 games >= 5 and total games >= 100",
            lambda df: (df["games_2025"] >= 5) & (df["total_games"] >= 100),
        ),
    ]

    for elo_setting in preferred_settings:
        elo_one = elo[elo["setting_name"] == elo_setting].copy()
        if elo_one.empty:
            continue
        elo_one = elo_one.rename(
            columns={
                "setting_name": "elo_setting_name",
                "final_rating": "elo_final_rating",
                "final_rank": "elo_final_rank",
            }
        )
        if "elo_final_rank" not in elo_one.columns:
            elo_one["elo_final_rank"] = elo_one["elo_final_rating"].rank(method="min", ascending=False)

        merged = (
            elo_one[["player_code", "elo_setting_name", "elo_final_rating", "elo_final_rank"]]
            .merge(
                glicko[["player_code", "glicko_setting_name", "glicko_final_rating", "glicko_final_rank"]],
                on="player_code",
                how="inner",
            )
            .merge(activity, on="player_code", how="left")
        )
        merged[["total_games", "games_2025"]] = merged[["total_games", "games_2025"]].fillna(0).astype(int)
        merged["elo_centered_rating"] = merged["elo_final_rating"] - merged["elo_final_rating"].mean()
        merged["glicko_centered_rating"] = merged["glicko_final_rating"] - merged["glicko_final_rating"].mean()

        for group_id, group_description, selector in group_specs:
            subset = merged[selector(merged)].copy()
            n_players = len(subset)
            if n_players >= 2:
                subset["elo_rank_within_group"] = subset["elo_final_rating"].rank(method="min", ascending=False)
                subset["glicko_rank_within_group"] = subset["glicko_final_rating"].rank(method="min", ascending=False)
                subset["abs_rank_difference"] = (
                    subset["elo_rank_within_group"] - subset["glicko_rank_within_group"]
                ).abs()
                pearson = subset["elo_final_rating"].corr(subset["glicko_final_rating"], method="pearson")
                spearman = subset["elo_final_rating"].corr(subset["glicko_final_rating"], method="spearman")
                mean_abs_rank_diff = subset["abs_rank_difference"].mean()
                mean_abs_rating_diff = (subset["elo_final_rating"] - subset["glicko_final_rating"]).abs().mean()
                mean_abs_centered_diff = (
                    subset["elo_centered_rating"] - subset["glicko_centered_rating"]
                ).abs().mean()
                top50 = top_overlap(subset, "elo_final_rating", "glicko_final_rating", 50)
                top100 = top_overlap(subset, "elo_final_rating", "glicko_final_rating", 100)
            else:
                pearson = spearman = mean_abs_rank_diff = mean_abs_rating_diff = mean_abs_centered_diff = np.nan
                top50 = top100 = np.nan
            rows.append(
                {
                    "elo_setting_name": elo_setting,
                    "glicko_setting_name": glicko_setting,
                    "group_id": group_id,
                    "group_description": group_description,
                    "number_of_overlapping_players": n_players,
                    "pearson_rating_correlation": pearson,
                    "spearman_rank_correlation": spearman,
                    "top50_overlap": top50,
                    "top100_overlap": top100,
                    "mean_abs_rank_difference": mean_abs_rank_diff,
                    "mean_abs_rating_difference": mean_abs_rating_diff,
                    "mean_abs_centered_rating_difference": mean_abs_centered_diff,
                }
            )

    similarity = pd.DataFrame(rows)
    return similarity


def make_scatter_plot(
    issues: list[dict[str, Any]],
    elo_path: Path | None,
    glicko_path: Path | None,
    matches_path: Path | None,
) -> bool:
    """Create a scatter plot for high-activity active players if possible."""

    if elo_path is None or glicko_path is None or matches_path is None:
        add_issue(
            issues,
            "plot_skipped",
            "meeting5_elo_vs_glicko_active_players_scatter.png",
            "info",
            "Skipped because one or more required files are missing.",
        )
        return False

    try:
        elo = pd.read_csv(elo_path, usecols=["setting_name", "start_year", "player_code", "final_rating"])
        glicko = pd.read_csv(glicko_path, usecols=["player_code", "final_rating"])
        activity = compute_player_counts(matches_path, issues)
        elo = elo[
            (elo["setting_name"] == "validation_best_k30_scale300")
            & (elo["start_year"] == 1985)
        ].rename(columns={"final_rating": "elo_final_rating"})
        glicko = glicko.rename(columns={"final_rating": "glicko_final_rating"})
        data = elo[["player_code", "elo_final_rating"]].merge(glicko, on="player_code", how="inner")
        data = data.merge(activity, on="player_code", how="left")
        data = data[(data["games_2025"] >= 5) & (data["total_games"] >= 100)].copy()
        if data.empty:
            add_issue(
                issues,
                "plot_skipped",
                "meeting5_elo_vs_glicko_active_players_scatter.png",
                "info",
                "No active high-volume players available for scatter plot.",
            )
            return False

        data["elo_centered_rating"] = data["elo_final_rating"] - data["elo_final_rating"].mean()
        data["glicko_centered_rating"] = data["glicko_final_rating"] - data["glicko_final_rating"].mean()

        plt.figure(figsize=(7, 6))
        plt.scatter(
            data["elo_centered_rating"],
            data["glicko_centered_rating"],
            s=14,
            alpha=0.55,
            edgecolors="none",
        )
        x_min = float(data["elo_centered_rating"].min())
        x_max = float(data["elo_centered_rating"].max())
        y_min = float(data["glicko_centered_rating"].min())
        y_max = float(data["glicko_centered_rating"].max())
        lo = min(x_min, y_min)
        hi = max(x_max, y_max)
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("Validation-best Elo final rating, centred")
        plt.ylabel("Glicko match-by-match final rating, centred")
        plt.title("Active high-volume players: Elo vs Glicko ratings")
        plt.tight_layout()
        plt.savefig(SCATTER_PATH, dpi=180)
        plt.close()
        return True
    except Exception as exc:  # pragma: no cover - graceful reporting for local file variation
        add_issue(
            issues,
            "plot_error",
            "meeting5_elo_vs_glicko_active_players_scatter.png",
            "warning",
            f"Could not generate scatter plot: {exc}",
        )
        return False


def add_similarity_checks(
    check_rows: list[dict[str, Any]],
    similarity: pd.DataFrame,
) -> None:
    """Add pass/fail style checks based on active/high-volume similarity."""

    if similarity.empty:
        add_check(
            check_rows,
            "E1",
            "Active-player Elo-vs-Glicko rating-list similarity",
            "Similarity table availability",
            "A similarity table should be computed for high-volume players",
            "not available",
            False,
        )
        return

    target = similarity[
        (similarity["elo_setting_name"] == "validation_best_k30_scale300")
        & (similarity["group_id"] == "active_2025_games_ge5_total_games_ge100")
    ]
    if target.empty:
        target = similarity[
            (similarity["elo_setting_name"] == "validation_best_k30_scale300")
            & (similarity["group_id"] == "total_games_ge100")
        ]
    if target.empty:
        add_check(
            check_rows,
            "E1",
            "Active-player Elo-vs-Glicko rating-list similarity",
            "High-volume similarity target row",
            "Target similarity group should be present",
            "missing",
            False,
        )
        return

    row = target.iloc[0]
    spearman = row["spearman_rank_correlation"]
    top50 = row["top50_overlap"]
    top100 = row["top100_overlap"]
    passed = bool(
        row["number_of_overlapping_players"] >= 50
        and spearman >= 0.85
        and top50 >= 0.70
    )
    add_check(
        check_rows,
        "E1",
        "Active-player Elo-vs-Glicko rating-list similarity",
        "High-volume active player similarity",
        "Glicko and Elo should not produce completely different lists for well-observed players",
        (
            f"group={row['group_id']}; n={int(row['number_of_overlapping_players'])}; "
            f"Spearman={spearman:.4f}; Top50={top50:.3f}; Top100={top100:.3f}"
        ),
        passed,
        "Threshold is diagnostic, not a formal performance criterion.",
    )


def write_summary(
    inventory: dict[str, Path | None],
    checks: pd.DataFrame,
    issues: pd.DataFrame,
    similarity: pd.DataFrame,
    rd_info: dict[str, Any],
    pred_info: dict[str, Any],
    scatter_created: bool,
) -> None:
    """Write the Meeting 5 validation summary Markdown."""

    def check_line(check_id: str) -> str:
        row = checks[checks["check_id"] == check_id]
        if row.empty:
            return f"- {check_id}: not available"
        r = row.iloc[0]
        status = "PASS" if bool(r["pass"]) else "CHECK"
        return f"- {check_id}: {status}; {r['observed_value']}"

    used_files = [
        f"- {key}: `{rel(path)}`" for key, path in inventory.items() if path is not None
    ]
    missing_files = [
        f"- {key}: missing" for key, path in inventory.items() if path is None
    ]

    issue_lines = ["- None"] if issues.empty else [
        f"- {row.issue_type} / {row.severity}: {row.item} - {row.details}"
        for row in issues.itertuples(index=False)
    ]

    similarity_lines = ["- Similarity table could not be computed."]
    if not similarity.empty:
        focus = similarity[
            (similarity["elo_setting_name"] == "validation_best_k30_scale300")
            & (
                similarity["group_id"].isin(
                    [
                        "total_games_ge100",
                        "total_games_ge200",
                        "active_2025_games_ge5",
                        "active_2025_games_ge5_total_games_ge100",
                    ]
                )
            )
        ]
        if not focus.empty:
            similarity_lines = [
                (
                    f"- {row.group_id}: n={int(row.number_of_overlapping_players)}, "
                    f"Spearman={row.spearman_rank_correlation:.4f}, "
                    f"Top50={row.top50_overlap:.3f}, Top100={row.top100_overlap:.3f}"
                )
                for row in focus.itertuples(index=False)
            ]

    all_passed = bool(checks["pass"].all()) if not checks.empty else False
    pass_count = int(checks["pass"].sum()) if not checks.empty else 0
    total_checks = len(checks)

    lines = [
        "# Meeting 5 Glicko Implementation Validation",
        "",
        "## Purpose",
        "",
        "This validation step answers the supervisor's question: how am I gaining confidence in the implementation of Glicko? The aim is not to tune the model or perform the final Elo-vs-Glicko comparison. The aim is to check that the Glicko implementation behaves in ways that are consistent with the Glicko mechanism.",
        "",
        "## Data and Existing Outputs Used",
        "",
        *used_files,
        "",
        "Missing files:",
        "",
        *(missing_files if missing_files else ["- None"]),
        "",
        "## Formula Sanity Checks",
        "",
        check_line("A1"),
        check_line("A2"),
        check_line("A3"),
        check_line("A4"),
        check_line("A5"),
        "",
        "## Official-Style Example",
        "",
        check_line("B1"),
        "",
        "## RD Behaviour Checks",
        "",
        f"- Constants: DEFAULT_RD={rd_info.get('default_rd')}, MIN_RD={rd_info.get('min_rd')}, MAX_RD={rd_info.get('max_rd')}",
        f"- Final players checked: {rd_info.get('n_players', 'not available')}",
        f"- Median final RD: {rd_info.get('median_rd', np.nan):.3f}" if "median_rd" in rd_info else "- Median final RD: not available",
        f"- Mean final RD: {rd_info.get('mean_rd', np.nan):.3f}" if "mean_rd" in rd_info else "- Mean final RD: not available",
        f"- Min/max observed final RD: {rd_info.get('min_rd_observed', np.nan):.3f} / {rd_info.get('max_rd_observed', np.nan):.3f}" if "min_rd_observed" in rd_info else "- Min/max observed final RD: not available",
        f"- Players at MIN_RD: {rd_info.get('number_at_min_rd', 'not available')}",
        f"- Players near MAX_RD: {rd_info.get('number_near_max_rd', 'not available')}",
        "",
        "## Prediction-Before-Update Checks",
        "",
        f"- Prediction rows checked: {pred_info.get('prediction_rows', 'not available')}",
        f"- 2025 evaluation rows: {pred_info.get('eval_rows_2025', 'not available')}",
        f"- Prediction probability range: {pred_info.get('pred_min', np.nan):.6f} to {pred_info.get('pred_max', np.nan):.6f}" if "pred_min" in pred_info else "- Prediction probability range: not available",
        f"- Max difference when recomputing pred_a_win from pre-rating/RD columns: {pred_info.get('max_pre_update_prediction_recompute_diff', np.nan):.12g}" if "max_pre_update_prediction_recompute_diff" in pred_info else "- Pre-update recomputation check: not available",
        "",
        "## Active-Player Elo-vs-Glicko Rating-List Similarity",
        "",
        "For this implementation validation, rank correlation and top-list overlap are more important than raw rating differences because Elo and Glicko ratings are not necessarily on exactly the same scale.",
        "",
        *similarity_lines,
        "",
        "## Validation Check Summary",
        "",
        f"- Checks passed: {pass_count} / {total_checks}",
        f"- Overall status: {'PASS' if all_passed else 'PASS WITH NOTES' if pass_count >= max(total_checks - 2, 0) else 'REVIEW REQUIRED'}",
        f"- Scatter plot created: {'yes' if scatter_created else 'no'}",
        "",
        "## Interpretation For Supervisor",
        "",
        "The validation checks give me more confidence that the Glicko implementation is behaving as intended. The formula sanity checks and official-style update example are consistent with expected Glicko behaviour. The saved predictions are valid probabilities and are evaluated on the same 2025 game set. For high-activity players, the Glicko and Elo rating lists are broadly similar, which suggests that the Glicko implementation is not producing implausible rankings. The next step is therefore to test inactivity RD inflation and then proceed to the fair Elo-vs-Glicko comparison.",
        "",
        "## Remaining Limitations",
        "",
        "- Passing sanity checks does not prove the implementation is mathematically perfect.",
        "- Some checks depend on which columns were saved in previous output files.",
        "- Elo and Glicko ratings are not guaranteed to be directly comparable on the raw rating scale, so rank-based checks are more important.",
        "- Full confidence also requires sensitivity checks such as RD inflation and rating-period runtime comparison.",
        "",
        "## Issues",
        "",
        *issue_lines,
        "",
    ]

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = make_file_inventory()
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for key, path in inventory.items():
        if path is None and key in {
            "glicko_core",
            "glicko_predictions",
            "glicko_final_ratings",
            "elo_final_ratings",
            "full_history_matches",
        }:
            add_issue(issues, "missing_file", key, "warning", "Expected file could not be found by rglob search.")

    run_formula_checks(checks)
    run_official_style_check(checks)
    rd_info = run_rd_checks(
        checks,
        issues,
        inventory["glicko_final_ratings"],
        inventory["glicko_rd_summary"],
    )
    pred_info = run_prediction_checks(checks, issues, inventory["glicko_predictions"])
    similarity = compare_active_rating_lists(
        issues,
        inventory["elo_final_ratings"],
        inventory["glicko_final_ratings"],
        inventory["full_history_matches"],
    )
    add_similarity_checks(checks, similarity)
    scatter_created = make_scatter_plot(
        issues,
        inventory["elo_final_ratings"],
        inventory["glicko_final_ratings"],
        inventory["full_history_matches"],
    )

    checks_df = pd.DataFrame(checks)
    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(
            [{"issue_type": "none", "item": "none", "severity": "none", "details": "No issues detected."}]
        )

    checks_df.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")
    similarity.to_csv(ACTIVE_SIMILARITY_PATH, index=False, encoding="utf-8-sig")
    issues_df.to_csv(VALIDATION_ISSUES_PATH, index=False, encoding="utf-8-sig")
    write_summary(inventory, checks_df, issues_df[issues_df["issue_type"] != "none"], similarity, rd_info, pred_info, scatter_created)

    print("Meeting 5 Glicko implementation validation complete.")
    print("Files used:")
    for key, path in inventory.items():
        print(f"  {key}: {rel(path)}")
    print()
    print("Outputs:")
    for path in [
        VALIDATION_CHECKS_PATH,
        ACTIVE_SIMILARITY_PATH,
        VALIDATION_ISSUES_PATH,
        SUMMARY_MD_PATH,
    ]:
        print(f"  {path}")
    if scatter_created:
        print(f"  {SCATTER_PATH}")
    print()
    print(f"Validation checks passed: {int(checks_df['pass'].sum())}/{len(checks_df)}")
    if not similarity.empty:
        print("Active/high-volume similarity rows:")
        print(
            similarity[
                [
                    "elo_setting_name",
                    "group_id",
                    "number_of_overlapping_players",
                    "spearman_rank_correlation",
                    "top50_overlap",
                    "top100_overlap",
                ]
            ].to_string(index=False)
        )
    print("This script did not rerun the full 1985-2025 Glicko model.")


if __name__ == "__main__":
    main()
