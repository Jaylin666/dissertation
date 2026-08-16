"""Chronological one-game-period Glicko-1 inactivity sensitivity."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.io_utils import add_event_ordering_columns as add_shared_event_ordering

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from code.models.glicko import (  # noqa: E402
    DEFAULT_RATING,
    DEFAULT_RD,
    MAX_RD,
    MIN_RD,
    expected_score,
    update_two_players_single_game,
)


START_YEAR = 1985
END_YEAR = 2025
EXPECTED_2025_GAMES = 11_379
EPS = 1e-15
NEAR_MAX_RD_THRESHOLD = MAX_RD - 5.0

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting5_glicko_rd_inflation"

METRICS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_metrics_2025.csv"
RD_SUMMARY_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_rd_summary.csv"
GAP_METRICS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_gap_metrics.csv"
PREDICTIONS_2025_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_predictions_2025.csv"
FINAL_RATINGS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_final_ratings.csv"
CALIBRATION_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_calibration_2025.csv"

BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_brier_bar.png"
LOGLOSS_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_logloss_bar.png"
RD_DIST_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_distribution_by_variant.png"
GAP_BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_gap_brier.png"


def configure_output_root(output_root: str | Path) -> Path:
    global OUTPUT_DIR
    global METRICS_PATH, RD_SUMMARY_PATH, GAP_METRICS_PATH
    global PREDICTIONS_2025_PATH, FINAL_RATINGS_PATH, CALIBRATION_PATH
    global BRIER_PLOT_PATH, LOGLOSS_PLOT_PATH
    global RD_DIST_PLOT_PATH, GAP_BRIER_PLOT_PATH

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    OUTPUT_DIR = root.resolve() / "meeting5_glicko_rd_inflation"
    METRICS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_metrics_2025.csv"
    RD_SUMMARY_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_rd_summary.csv"
    GAP_METRICS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_gap_metrics.csv"
    PREDICTIONS_2025_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_predictions_2025.csv"
    FINAL_RATINGS_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_final_ratings.csv"
    CALIBRATION_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_calibration_2025.csv"
    BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_brier_bar.png"
    LOGLOSS_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_logloss_bar.png"
    RD_DIST_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_distribution_by_variant.png"
    GAP_BRIER_PLOT_PATH = OUTPUT_DIR / "meeting5_glicko_rd_inflation_gap_brier.png"
    return OUTPUT_DIR

REQUIRED_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_COLUMNS = ["eventname", "event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

BASELINE_EXPECTED = {
    "evaluation_games_2025": 11_379,
    "log_loss": 0.571958,
    "brier": 0.195708,
    "accuracy": 0.693734,
}


def find_full_history_dataset() -> Path:
    expected = PROJECT_ROOT / "outputs" / "elo_optimization" / "matches_1985_2025_checked.csv"
    if expected.exists():
        return expected

    candidates = sorted(PROJECT_ROOT.rglob("matches_1985_2025_checked.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find matches_1985_2025_checked.csv. "
            "Run code/13_build_full_history_match_dataset.py first."
        )
    return candidates[0]


def format_code_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        value_float = float(value)
        if value_float.is_integer():
            return str(int(value_float))
    except (TypeError, ValueError):
        pass
    return str(value)


def player_code(value: Any) -> int:
    return int(float(value))


def update_player_name(player_names: dict[int, str], code: int, possible_name: Any) -> None:
    if code in player_names or pd.isna(possible_name):
        return
    name = str(possible_name).strip()
    if name:
        player_names[code] = name


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    return add_shared_event_ordering(matches)


def add_inactivity_period_index(matches: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add monthly inactivity periods with a yearly fallback."""

    matches = matches.copy()
    event_dates = pd.to_datetime(matches["event_order_date"], errors="coerce")
    if event_dates.notna().any():
        period_index = (event_dates.dt.year * 12 + event_dates.dt.month).astype("Float64")
        fallback_mask = event_dates.isna()
        # Missing dates use December for period ordering only.
        period_index.loc[fallback_mask] = matches.loc[fallback_mask, "year"].astype(int) * 12 + 12
        matches["inactivity_period_index"] = period_index.astype(int)
        matches["inactivity_period_source"] = np.where(
            fallback_mask,
            "year_fallback_december",
            "month_from_event_order_date",
        )
        return matches, "month"

    matches["inactivity_period_index"] = matches["year"].astype(int)
    matches["inactivity_period_source"] = "year_only_fallback"
    return matches, "year"


def load_matches() -> tuple[pd.DataFrame, str, Path]:
    dataset_path = find_full_history_dataset()
    matches = pd.read_csv(dataset_path, low_memory=False)

    missing_required = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{dataset_path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    missing_ids = matches[REQUIRED_COLUMNS].isna().sum()
    if int(missing_ids.sum()) > 0:
        raise ValueError(f"Required ID columns contain missing values:\n{missing_ids}")

    matches = matches[(matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)].copy()
    matches = add_event_ordering_columns(matches)
    matches["event_order_date_missing"] = matches["event_order_date"].isna()
    matches = matches.sort_values(
        ["year", "event_order_date_missing", "event_order_date", "event", "fcode"],
        na_position="last",
    ).drop(columns=["event_order_date_missing"])
    matches = matches.reset_index(drop=True)
    matches, inactivity_unit = add_inactivity_period_index(matches)

    players = pd.concat([matches["winner"], matches["loser"]]).astype(int).nunique()
    print(f"Loaded dataset: {dataset_path}")
    print(f"Matches: {len(matches):,}")
    print(f"Year range: {int(matches['year'].min())}-{int(matches['year'].max())}")
    print(f"Players: {players:,}")
    print(f"Inactivity period unit: {inactivity_unit}")
    return matches, inactivity_unit, dataset_path


def build_variants(inactivity_unit: str) -> list[dict[str, Any]]:
    """Build the frozen Glicko variants."""

    if inactivity_unit == "month":
        target_low = 240
        target_medium = 120
        target_high = 60
        unit_label = "months"
    else:
        target_low = 20
        target_medium = 10
        target_high = 5
        unit_label = "years"

    def c_from_target(target_periods: int) -> float:
        return math.sqrt(((MAX_RD**2) - (MIN_RD**2)) / target_periods)

    return [
        {
            "variant": "C0_no_inflation",
            "c_value": 0.0,
            "target_periods": 0,
            "target_label": "no inactivity inflation",
            "unit_label": unit_label,
        },
        {
            "variant": "low_inflation",
            "c_value": c_from_target(target_low),
            "target_periods": target_low,
            "target_label": f"return from MIN_RD to MAX_RD after about {target_low} {unit_label}",
            "unit_label": unit_label,
        },
        {
            "variant": "medium_inflation",
            "c_value": c_from_target(target_medium),
            "target_periods": target_medium,
            "target_label": f"return from MIN_RD to MAX_RD after about {target_medium} {unit_label}",
            "unit_label": unit_label,
        },
        {
            "variant": "high_inflation",
            "c_value": c_from_target(target_high),
            "target_periods": target_high,
            "target_label": f"return from MIN_RD to MAX_RD after about {target_high} {unit_label}",
            "unit_label": unit_label,
        },
    ]


def inflate_rd_for_inactivity(
    rd: float,
    elapsed_periods: float,
    c: float,
    min_rd: float = MIN_RD,
    max_rd: float = MAX_RD,
) -> float:
    """Inflate RD for elapsed inactivity periods."""

    if c <= 0.0 or pd.isna(elapsed_periods) or elapsed_periods <= 0:
        return float(rd)
    inflated = math.sqrt((float(rd) ** 2) + (float(c) ** 2) * float(elapsed_periods))
    return min(max_rd, max(min_rd, inflated))


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def evaluate_winner_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate metrics from stored prematch winner probabilities."""

    if predictions.empty:
        return {
            "evaluation_games_2025": 0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "accuracy": float("nan"),
            "baseline_accuracy": float("nan"),
            "mean_prediction": float("nan"),
            "actual_win_rate": float("nan"),
        }

    p = predictions["pred_winner_win"].astype(float).to_numpy()
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    higher_rating_winner = (
        predictions["pre_rating_winner"].astype(float)
        >= predictions["pre_rating_loser"].astype(float)
    ).to_numpy()
    return {
        "evaluation_games_2025": int(len(predictions)),
        "log_loss": float(-np.mean(np.log(p_clipped))),
        "brier": float(np.mean((p - 1.0) ** 2)),
        "accuracy": float(np.mean(p >= 0.5)),
        "baseline_accuracy": float(np.mean(higher_rating_winner)),
        "mean_prediction": float(np.mean(p)),
        "actual_win_rate": 1.0,
    }


def assign_gap_group(max_gap: Any, inactivity_unit: str) -> str:
    if pd.isna(max_gap):
        return "new_or_missing_gap"

    gap = float(max_gap)
    if inactivity_unit == "month":
        if gap <= 0:
            return "no_gap_or_same_period"
        if gap <= 11:
            return "gap_1_to_11_months"
        if gap <= 35:
            return "gap_12_to_35_months"
        return "gap_36plus_months"

    if gap <= 0:
        return "no_gap_or_same_year"
    if gap <= 1:
        return "gap_1_year"
    if gap <= 3:
        return "gap_2_to_3_years"
    return "gap_4plus_years"


def run_variant(matches: pd.DataFrame, variant: dict[str, Any], inactivity_unit: str) -> dict[str, Any]:
    """Run one RD-inflation variant with prediction before each update."""

    variant_name = variant["variant"]
    c_value = float(variant["c_value"])
    print(f"\nRunning variant: {variant_name} (c={c_value:.6f})")
    start_time = time.perf_counter()

    ratings: dict[int, float] = {}
    rds: dict[int, float] = {}
    last_period_index: dict[int, int] = {}
    games_played: defaultdict[int, int] = defaultdict(int)
    wins: defaultdict[int, int] = defaultdict(int)
    losses: defaultdict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}
    predictions_2025: list[dict[str, Any]] = []

    sim_cols = [
        "fcode",
        "year",
        "event",
        "eventname",
        "event_date_raw",
        "event_date_parsed",
        "event_order_date",
        "event_date_ordering_method",
        "inactivity_period_index",
        "winner",
        "loser",
        "winner_name",
        "loser_name",
    ]

    current_year = None
    year_match_count = 0
    total_matches = 0

    for row in matches[sim_cols].itertuples(index=False):
        year = int(row.year)
        if current_year is None:
            current_year = year
        if year != current_year:
            if current_year in {1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025}:
                print(f"  processed {current_year}: {year_match_count:,} matches")
            current_year = year
            year_match_count = 0

        winner = player_code(row.winner)
        loser = player_code(row.loser)
        period_index = int(row.inactivity_period_index)

        update_player_name(player_names, winner, row.winner_name)
        update_player_name(player_names, loser, row.loser_name)

        if winner not in ratings:
            ratings[winner] = DEFAULT_RATING
            rds[winner] = DEFAULT_RD
            winner_gap = np.nan
        else:
            winner_gap = period_index - last_period_index[winner]
            rds[winner] = inflate_rd_for_inactivity(rds[winner], winner_gap, c_value)

        if loser not in ratings:
            ratings[loser] = DEFAULT_RATING
            rds[loser] = DEFAULT_RD
            loser_gap = np.nan
        else:
            loser_gap = period_index - last_period_index[loser]
            rds[loser] = inflate_rd_for_inactivity(rds[loser], loser_gap, c_value)

        winner_rating_before = ratings[winner]
        winner_rd_before = rds[winner]
        loser_rating_before = ratings[loser]
        loser_rd_before = rds[loser]

        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, loser_rd_before)
        update = update_two_players_single_game(
            winner_rating_before,
            winner_rd_before,
            loser_rating_before,
            loser_rd_before,
            1.0,
        )

        ratings[winner] = update.player1_rating_after
        rds[winner] = update.player1_rd_after
        ratings[loser] = update.player2_rating_after
        rds[loser] = update.player2_rd_after

        previous_winner_period = last_period_index.get(winner, period_index)
        previous_loser_period = last_period_index.get(loser, period_index)
        last_period_index[winner] = max(previous_winner_period, period_index)
        last_period_index[loser] = max(previous_loser_period, period_index)

        games_played[winner] += 1
        games_played[loser] += 1
        wins[winner] += 1
        losses[loser] += 1

        winner_rating_change = ratings[winner] - winner_rating_before
        loser_rating_change = ratings[loser] - loser_rating_before

        if year == 2025:
            gaps = [gap for gap in (winner_gap, loser_gap) if not pd.isna(gap)]
            max_gap = max(gaps) if gaps else np.nan
            predictions_2025.append(
                {
                    "variant": variant_name,
                    "year": year,
                    "game_id": int(row.fcode),
                    "fcode": int(row.fcode),
                    "event": format_code_value(row.event),
                    "eventname": row.eventname,
                    "event_date_raw": row.event_date_raw,
                    "event_date_parsed": row.event_date_parsed,
                    "event_order_date": row.event_order_date,
                    "event_date_ordering_method": row.event_date_ordering_method,
                    "winner": winner,
                    "loser": loser,
                    "pred_winner_win": pred_winner_win,
                    "pre_rating_winner": winner_rating_before,
                    "pre_rd_winner": winner_rd_before,
                    "pre_rating_loser": loser_rating_before,
                    "pre_rd_loser": loser_rd_before,
                    "winner_gap_periods": winner_gap,
                    "loser_gap_periods": loser_gap,
                    "max_player_gap_periods": max_gap,
                    "winner_rating_change": winner_rating_change,
                    "loser_rating_change": loser_rating_change,
                    "mean_abs_player_rating_change": (
                        abs(winner_rating_change) + abs(loser_rating_change)
                    )
                    / 2.0,
                }
            )

        year_match_count += 1
        total_matches += 1

    if current_year is not None and current_year in {1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025}:
        print(f"  processed {current_year}: {year_match_count:,} matches")

    predictions_df = pd.DataFrame(predictions_2025)
    metrics = evaluate_winner_predictions(predictions_df)

    final_rows = []
    for code in sorted(ratings):
        final_rows.append(
            {
                "variant": variant_name,
                "player_id": code,
                "player_name": player_names.get(code, pd.NA),
                "rating": ratings[code],
                "rd": rds[code],
                "games_played": games_played[code],
                "wins": wins[code],
                "losses": losses[code],
                "last_period_index": last_period_index.get(code, pd.NA),
            }
        )
    final_df = pd.DataFrame(final_rows)
    final_df["final_rank_by_rating"] = final_df["rating"].rank(method="min", ascending=False).astype(int)
    final_df = final_df.sort_values(["final_rank_by_rating", "player_id"]).reset_index(drop=True)

    rds_array = final_df["rd"].astype(float).to_numpy()
    runtime_seconds = time.perf_counter() - start_time

    metrics.update(
        {
            "variant": variant_name,
            "c_value": c_value,
            "inactivity_unit": inactivity_unit,
            "target_periods": int(variant["target_periods"]),
            "target_label": variant["target_label"],
            "runtime_seconds": runtime_seconds,
            "final_players": int(len(final_df)),
            "final_mean_rd": float(np.mean(rds_array)),
            "final_median_rd": float(np.median(rds_array)),
            "final_min_rd": float(np.min(rds_array)),
            "final_max_rd": float(np.max(rds_array)),
            "players_at_min_rd": int(np.sum(rds_array <= MIN_RD + 1e-9)),
            "players_near_max_rd": int(np.sum(rds_array >= NEAR_MAX_RD_THRESHOLD)),
            "mean_abs_rating_change_2025": float(
                predictions_df["mean_abs_player_rating_change"].mean()
            )
            if not predictions_df.empty
            else float("nan"),
            "notes": "",
        }
    )

    if variant_name == "C0_no_inflation":
        deltas = {
            key: abs(metrics[key] - value)
            for key, value in BASELINE_EXPECTED.items()
            if key in metrics
        }
        if (
            deltas.get("evaluation_games_2025", 0.0) == 0
            and deltas.get("log_loss", 1.0) < 1e-4
            and deltas.get("brier", 1.0) < 1e-4
            and deltas.get("accuracy", 1.0) < 1e-4
        ):
            metrics["notes"] = "C0 matches the previous match-by-match Glicko baseline within tolerance."
        else:
            metrics["notes"] = (
                "WARNING: C0 differs from the previous match-by-match baseline; "
                f"deltas={deltas}"
            )

    print(
        "  2025 metrics: "
        f"games={metrics['evaluation_games_2025']:,}, "
        f"log_loss={metrics['log_loss']:.6f}, "
        f"brier={metrics['brier']:.6f}, "
        f"accuracy={metrics['accuracy']:.6f}"
    )
    print(
        "  final RD: "
        f"median={metrics['final_median_rd']:.3f}, "
        f"mean={metrics['final_mean_rd']:.3f}, "
        f"players_at_min={metrics['players_at_min_rd']:,}, "
        f"players_near_max={metrics['players_near_max_rd']:,}"
    )
    print(f"  runtime: {runtime_seconds:.1f}s")

    return {
        "metrics": metrics,
        "predictions_2025": predictions_df,
        "final_ratings": final_df,
        "total_matches": total_matches,
    }


def make_rd_summary(final_ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in final_ratings.groupby("variant", sort=False):
        rds = group["rd"].astype(float).to_numpy()
        rows.append(
            {
                "variant": variant,
                "mean_rd": float(np.mean(rds)),
                "median_rd": float(np.median(rds)),
                "min_rd": float(np.min(rds)),
                "q10_rd": percentile(rds, 10),
                "q25_rd": percentile(rds, 25),
                "q75_rd": percentile(rds, 75),
                "q90_rd": percentile(rds, 90),
                "max_rd": float(np.max(rds)),
                "players_at_min_rd": int(np.sum(rds <= MIN_RD + 1e-9)),
                "players_near_max_rd": int(np.sum(rds >= NEAR_MAX_RD_THRESHOLD)),
                "players": int(len(rds)),
            }
        )
    return pd.DataFrame(rows)


def make_gap_metrics(predictions: pd.DataFrame, inactivity_unit: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    data = predictions.copy()
    data["gap_group"] = data["max_player_gap_periods"].apply(
        lambda value: assign_gap_group(value, inactivity_unit)
    )

    rows = []
    for (variant, gap_group), group in data.groupby(["variant", "gap_group"], sort=False):
        metrics = evaluate_winner_predictions(group)
        rows.append(
            {
                "variant": variant,
                "gap_group": gap_group,
                "games": int(metrics["evaluation_games_2025"]),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "mean_prediction": metrics["mean_prediction"],
                "actual_win_rate": metrics["actual_win_rate"],
            }
        )
    return pd.DataFrame(rows)


def make_calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    data = predictions.copy()
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    data["probability_bin"] = pd.cut(
        data["pred_winner_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    for (variant, prob_bin), group in data.groupby(["variant", "probability_bin"], observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "probability_bin": str(prob_bin),
                "games": int(len(group)),
                "mean_prediction": float(group["pred_winner_win"].mean()),
                "actual_win_rate": 1.0,
                "brier": float(np.mean((group["pred_winner_win"].to_numpy() - 1.0) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def save_bar_plot(metrics: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_data = metrics.sort_values("variant")
    ax.bar(plot_data["variant"], plot_data[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rd_distribution_plot(final_ratings: pd.DataFrame, path: Path) -> None:
    variants = list(final_ratings["variant"].drop_duplicates())
    data = [
        final_ratings.loc[final_ratings["variant"] == variant, "rd"].astype(float).to_numpy()
        for variant in variants
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(data, labels=variants, showfliers=False)
    ax.set_title("Final RD distribution by inactivity variant")
    ax.set_ylabel("Final RD")
    ax.set_xlabel("Variant")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_gap_brier_plot(gap_metrics: pd.DataFrame, path: Path) -> None:
    if gap_metrics.empty:
        return
    pivot = gap_metrics.pivot(index="gap_group", columns="variant", values="brier")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("2025 Brier score by inactivity gap group")
    ax.set_ylabel("Brier score")
    ax.set_xlabel("Gap group")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Variant", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)




def choose_recommendation(metrics: pd.DataFrame, gap_metrics: pd.DataFrame) -> tuple[str, str]:
    metrics_sorted = metrics.sort_values(["brier", "log_loss"]).reset_index(drop=True)
    best = metrics_sorted.iloc[0]
    c0 = metrics.loc[metrics["variant"] == "C0_no_inflation"].iloc[0]
    brier_gain = float(c0["brier"] - best["brier"])
    logloss_gain = float(c0["log_loss"] - best["log_loss"])

    if best["variant"] != "C0_no_inflation" and brier_gain > 0 and logloss_gain > 0:
        headline = (
            f"{best['variant']} is the tentative candidate main Glicko variant "
            "from this sensitivity check."
        )
        reason = (
            f"It improves both 2025 Brier score by {brier_gain:.6f} and log loss by "
            f"{logloss_gain:.6f} relative to C0. This is still a sensitivity result, "
            "not a final universal best model."
        )
        return headline, reason

    if best["variant"] == "C0_no_inflation":
        headline = "C0_no_inflation remains the prediction-oriented baseline in this run."
        reason = (
            "The RD inflation variants do not improve the aggregate 2025 Brier/log-loss "
            "metrics. RD inflation remains conceptually relevant for inactive players and "
            "should still be discussed as a sensitivity check."
        )
        return headline, reason

    headline = f"{best['variant']} has the lowest Brier score, but the recommendation is cautious."
    reason = (
        "The aggregate metrics are mixed, so this should be treated as a sensitivity result "
        "rather than a final Glicko choice."
    )
    return headline, reason




def main() -> None:
    """Run the Glicko pipeline."""

    overall_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Meeting 5 Glicko inactivity RD inflation sensitivity")
    print(f"Default rating={DEFAULT_RATING}, default RD={DEFAULT_RD}, MIN_RD={MIN_RD}, MAX_RD={MAX_RD}")

    matches, inactivity_unit, dataset_path = load_matches()
    variants = build_variants(inactivity_unit)
    print("Variants:")
    for item in variants:
        print(
            f"  {item['variant']}: c={item['c_value']:.6f}, "
            f"target_periods={item['target_periods']}, {item['target_label']}"
        )

    ordering_summary = matches["event_date_ordering_method"].value_counts(dropna=False).to_dict()
    period_source_summary = matches["inactivity_period_source"].value_counts(dropna=False).to_dict()
    print(f"Event date ordering summary: {ordering_summary}")
    print(f"Inactivity period source summary: {period_source_summary}")

    all_metrics = []
    all_predictions = []
    all_final_ratings = []

    for variant in variants:
        result = run_variant(matches, variant, inactivity_unit)
        all_metrics.append(result["metrics"])
        all_predictions.append(result["predictions_2025"])
        all_final_ratings.append(result["final_ratings"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[
        [
            "variant",
            "c_value",
            "inactivity_unit",
            "target_periods",
            "evaluation_games_2025",
            "log_loss",
            "brier",
            "accuracy",
            "baseline_accuracy",
            "runtime_seconds",
            "final_players",
            "final_mean_rd",
            "final_median_rd",
            "final_min_rd",
            "final_max_rd",
            "players_at_min_rd",
            "players_near_max_rd",
            "mean_abs_rating_change_2025",
            "notes",
        ]
    ]
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    final_ratings_df = pd.concat(all_final_ratings, ignore_index=True)
    rd_summary_df = make_rd_summary(final_ratings_df)
    gap_metrics_df = make_gap_metrics(predictions_df, inactivity_unit)
    calibration_df = make_calibration_table(predictions_df)

    metrics_df.to_csv(METRICS_PATH, index=False)
    rd_summary_df.to_csv(RD_SUMMARY_PATH, index=False)
    gap_metrics_df.to_csv(GAP_METRICS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_2025_PATH, index=False)
    final_ratings_df.to_csv(FINAL_RATINGS_PATH, index=False)
    calibration_df.to_csv(CALIBRATION_PATH, index=False)

    save_bar_plot(metrics_df, "brier", BRIER_PLOT_PATH, "2025 Brier score by RD inflation variant", "Brier score")
    save_bar_plot(metrics_df, "log_loss", LOGLOSS_PLOT_PATH, "2025 log loss by RD inflation variant", "Log loss")
    save_rd_distribution_plot(final_ratings_df, RD_DIST_PLOT_PATH)
    save_gap_brier_plot(gap_metrics_df, GAP_BRIER_PLOT_PATH)

    output_paths = [
        METRICS_PATH,
        RD_SUMMARY_PATH,
        GAP_METRICS_PATH,
        PREDICTIONS_2025_PATH,
        FINAL_RATINGS_PATH,
        CALIBRATION_PATH,
        BRIER_PLOT_PATH,
        LOGLOSS_PLOT_PATH,
        RD_DIST_PLOT_PATH,
        GAP_BRIER_PLOT_PATH,
    ]

    print("\nOutput files:")
    for path in output_paths:
        print(f"  {path}")

    mismatched_games = metrics_df.loc[
        metrics_df["evaluation_games_2025"] != EXPECTED_2025_GAMES,
        ["variant", "evaluation_games_2025"],
    ]
    if not mismatched_games.empty:
        print("WARNING: Some variants do not have the expected 11,379 evaluation games:")
        print(mismatched_games.to_string(index=False))
    else:
        print("All variants have the expected 11,379 evaluation games.")

    c0_notes = metrics_df.loc[metrics_df["variant"] == "C0_no_inflation", "notes"].iloc[0]
    if c0_notes.startswith("WARNING"):
        print(c0_notes)

    print("No Elo-vs-Glicko final comparison was run.")
    print("No adaptive-K Elo was run.")
    print(f"Total runtime: {time.perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
