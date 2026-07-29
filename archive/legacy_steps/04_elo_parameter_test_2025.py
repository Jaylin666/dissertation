from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_RATING = 1500.0
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40]
SCALE_VALUES = [300, 400, 500, 600, 700, 800]


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
MATCHES_PATH = DATA_PROCESSED / "matches_2025_checked.csv"
OUTPUT_PATH = DATA_PROCESSED / "elo_parameter_results_2025.csv"


REQUIRED_MATCH_COLUMNS = ["fcode", "code", "year", "winner", "loser"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load the checked 2025 match table from stage 1."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/01_inspect_2025_data.py first."
        )

    matches = pd.read_csv(path)
    missing_required = [
        col for col in REQUIRED_MATCH_COLUMNS if col not in matches.columns
    ]
    if missing_required:
        raise ValueError(
            f"{path.name} is missing required columns: {missing_required}"
        )

    for col in REQUIRED_MATCH_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    missing_values = matches[REQUIRED_MATCH_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(
            "Required match columns contain missing/non-numeric values:\n"
            + missing_values.to_string()
        )

    matches = matches.sort_values(["year", "code"]).reset_index(drop=True)
    return matches


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def run_elo_for_params(
    matches: pd.DataFrame,
    default_rating: float,
    k: float,
    scale: float,
) -> Dict[str, object]:
    """Run simple Elo once for a single K and scale combination."""
    ratings: Dict[int, float] = {}
    pred_a_win_values: List[float] = []
    actual_a_win_values: List[int] = []
    pred_winner_win_values: List[float] = []

    for _, row in matches.iterrows():
        winner = int(row["winner"])
        loser = int(row["loser"])

        if winner not in ratings:
            ratings[winner] = default_rating
        if loser not in ratings:
            ratings[loser] = default_rating

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]

        pred_winner_win = expected_score(
            winner_rating_before, loser_rating_before, scale
        )

        # Evaluation always uses the smaller player code as player_a.
        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = expected_score(ratings[player_a], ratings[player_b], scale)

        # Update after recording the pre-match prediction.
        rating_change = k * (1.0 - pred_winner_win)
        ratings[winner] = winner_rating_before + rating_change
        ratings[loser] = loser_rating_before - rating_change

        pred_a_win_values.append(pred_a_win)
        actual_a_win_values.append(actual_a_win)
        pred_winner_win_values.append(pred_winner_win)

    return {
        "pred_a_win": np.array(pred_a_win_values, dtype=float),
        "actual_a_win": np.array(actual_a_win_values, dtype=float),
        "pred_winner_win": np.array(pred_winner_win_values, dtype=float),
        "number_of_players": len(ratings),
    }


def compute_scores(
    elo_output: Dict[str, object],
    number_of_games: int,
    default_rating: float,
    k: float,
    scale: float,
) -> Dict[str, float]:
    """Calculate evaluation metrics for one parameter combination."""
    eps = 1e-15
    pred = elo_output["pred_a_win"]
    actual = elo_output["actual_a_win"]
    pred_winner = elo_output["pred_winner_win"]

    # Clip only for log loss, because log(0) is undefined.
    clipped_pred = np.clip(pred, eps, 1.0 - eps)
    log_loss = -np.mean(
        actual * np.log(clipped_pred)
        + (1.0 - actual) * np.log(1.0 - clipped_pred)
    )
    brier_score = np.mean((pred - actual) ** 2)
    accuracy = np.mean((pred >= 0.5) == (actual == 1.0))

    return {
        "number_of_games": number_of_games,
        "number_of_players": int(elo_output["number_of_players"]),
        "K": k,
        "scale": scale,
        "default_rating": default_rating,
        "log_loss": log_loss,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "mean_pred_a_win": np.mean(pred),
        "actual_a_win_rate": np.mean(actual),
        "mean_pred_winner_win": np.mean(pred_winner),
    }


def run_parameter_grid(
    matches: pd.DataFrame,
    default_rating: float = DEFAULT_RATING,
    k_values: List[int] = K_VALUES,
    scale_values: List[int] = SCALE_VALUES,
) -> pd.DataFrame:
    """Run simple Elo for every K/scale pair and collect summary metrics."""
    results = []
    total_combinations = len(k_values) * len(scale_values)
    combination_number = 0

    for k in k_values:
        for scale in scale_values:
            combination_number += 1
            print(
                f"Running {combination_number}/{total_combinations}: "
                f"K={k}, scale={scale}"
            )
            elo_output = run_elo_for_params(matches, default_rating, k, scale)
            scores = compute_scores(
                elo_output=elo_output,
                number_of_games=len(matches),
                default_rating=default_rating,
                k=k,
                scale=scale,
            )
            results.append(scores)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["log_loss", "brier_score", "K", "scale"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return results_df


def save_results(results: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Save the parameter sensitivity results."""
    DATA_PROCESSED.mkdir(exist_ok=True)
    results.to_csv(path, index=False)


def print_best_row(title: str, row: pd.Series) -> None:
    """Print one best-parameter summary line."""
    print(
        f"{title}: K={row['K']:g}, scale={row['scale']:g}, "
        f"log_loss={row['log_loss']:.6f}, "
        f"brier_score={row['brier_score']:.6f}, "
        f"accuracy={row['accuracy']:.6f}"
    )


def print_summary(results: pd.DataFrame) -> None:
    """Print the best parameter settings under each metric."""
    best_log_loss = results.sort_values("log_loss").iloc[0]
    best_brier = results.sort_values("brier_score").iloc[0]
    best_accuracy = results.sort_values(
        ["accuracy", "log_loss"], ascending=[False, True]
    ).iloc[0]

    print("\n=== Elo Parameter Test 2025 Summary ===")
    print(f"Number of parameter combinations tested: {len(results)}")
    print_best_row("Best parameter set by log_loss", best_log_loss)
    print_best_row("Best parameter set by brier_score", best_brier)
    print_best_row("Best parameter set by accuracy", best_accuracy)
    print(f"Output path: {OUTPUT_PATH}")


def main() -> None:
    matches = load_matches()
    results = run_parameter_grid(matches)
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
