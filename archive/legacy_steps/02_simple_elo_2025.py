from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


YEAR = 2025
DEFAULT_RATING = 1500.0
K = 20.0
SCALE = 500.0


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # This fallback helps when the whole file is run in an interactive tool.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
MATCHES_PATH = DATA_PROCESSED / "matches_2025_checked.csv"
PREDICTIONS_PATH = DATA_PROCESSED / "elo_predictions_2025.csv"
FINAL_RATINGS_PATH = DATA_PROCESSED / "elo_final_ratings_2025.csv"
SCORES_PATH = DATA_PROCESSED / "elo_scores_2025.csv"


REQUIRED_MATCH_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
OPTIONAL_OUTPUT_COLUMNS = ["event", "eventname", "winner_name", "loser_name"]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load the checked 2025 matches produced by stage 1."""
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

    for col in OPTIONAL_OUTPUT_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    key_columns = ["fcode", "code", "year", "winner", "loser", "event"]
    for col in key_columns:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    missing_keys = matches[REQUIRED_MATCH_COLUMNS].isna().sum()
    missing_keys = missing_keys[missing_keys > 0]
    if not missing_keys.empty:
        raise ValueError(
            "Required match columns contain missing/non-numeric values:\n"
            + missing_keys.to_string()
        )

    matches = matches.sort_values(["year", "code"]).reset_index(drop=True)
    return matches


def expected_score(rating_a: float, rating_b: float, scale: float = SCALE) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def ensure_player(
    player_code: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    default_rating: float,
) -> None:
    """Add a player to the rating and stats dictionaries if needed."""
    if player_code not in ratings:
        ratings[player_code] = default_rating
        stats[player_code] = {"n_games": 0, "n_wins": 0, "n_losses": 0}


def update_player_name(
    player_names: Dict[int, str], player_code: int, player_name: object
) -> None:
    """Store the first non-missing name seen for a player."""
    if player_code in player_names:
        return
    if pd.notna(player_name):
        player_names[player_code] = str(player_name)


def run_elo(
    matches: pd.DataFrame,
    default_rating: float = DEFAULT_RATING,
    k: float = K,
    scale: float = SCALE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a simple Elo baseline through matches in chronological order."""
    ratings: Dict[int, float] = {}
    stats: Dict[int, Dict[str, int]] = {}
    player_names: Dict[int, str] = {}
    prediction_rows = []

    for _, row in matches.iterrows():
        winner = int(row["winner"])
        loser = int(row["loser"])

        ensure_player(winner, ratings, stats, default_rating)
        ensure_player(loser, ratings, stats, default_rating)
        update_player_name(player_names, winner, row.get("winner_name"))
        update_player_name(player_names, loser, row.get("loser_name"))

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]

        pred_winner_win = expected_score(
            winner_rating_before, loser_rating_before, scale
        )

        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = expected_score(ratings[player_a], ratings[player_b], scale)

        rating_change = k * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change

        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after

        stats[winner]["n_games"] += 1
        stats[winner]["n_wins"] += 1
        stats[loser]["n_games"] += 1
        stats[loser]["n_losses"] += 1

        prediction_rows.append(
            {
                "fcode": int(row["fcode"]),
                "code": int(row["code"]),
                "year": int(row["year"]),
                "event": row.get("event", pd.NA),
                "eventname": row.get("eventname", pd.NA),
                "winner": winner,
                "loser": loser,
                "winner_name": row.get("winner_name", pd.NA),
                "loser_name": row.get("loser_name", pd.NA),
                "player_a": player_a,
                "player_b": player_b,
                "actual_a_win": actual_a_win,
                "pred_a_win": pred_a_win,
                "pred_winner_win": pred_winner_win,
                "winner_rating_before": winner_rating_before,
                "loser_rating_before": loser_rating_before,
                "winner_rating_after": winner_rating_after,
                "loser_rating_after": loser_rating_after,
                "rating_change": rating_change,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    final_ratings = build_final_ratings(ratings, stats, player_names)
    return predictions, final_ratings


def build_final_ratings(
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    player_names: Dict[int, str],
) -> pd.DataFrame:
    """Create a player-level final rating table."""
    rows = []
    for player_code, final_rating in ratings.items():
        player_stats = stats[player_code]
        row = {
            "player_code": player_code,
            "final_rating": final_rating,
            "n_games": player_stats["n_games"],
            "n_wins": player_stats["n_wins"],
            "n_losses": player_stats["n_losses"],
        }
        if player_code in player_names:
            row["player_name"] = player_names[player_code]
        rows.append(row)

    final_ratings = pd.DataFrame(rows)
    return final_ratings.sort_values(
        ["final_rating", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_scores(
    predictions: pd.DataFrame,
    default_rating: float = DEFAULT_RATING,
    k: float = K,
    scale: float = SCALE,
) -> pd.DataFrame:
    """Compute simple prediction metrics from player_a probabilities."""
    eps = 1e-15
    actual = predictions["actual_a_win"].astype(float)
    pred = predictions["pred_a_win"].astype(float)
    clipped_pred = pred.clip(eps, 1.0 - eps)

    log_loss = -np.mean(
        actual * np.log(clipped_pred) + (1.0 - actual) * np.log(1.0 - clipped_pred)
    )
    brier_score = np.mean((pred - actual) ** 2)
    accuracy = np.mean((pred >= 0.5) == (actual == 1.0))

    return pd.DataFrame(
        [
            {
                "year": YEAR,
                "default_rating": default_rating,
                "K": k,
                "scale": scale,
                "number_of_games": len(predictions),
                "number_of_unique_players": pd.concat(
                    [predictions["player_a"], predictions["player_b"]]
                ).nunique(),
                "log_loss": log_loss,
                "brier_score": brier_score,
                "accuracy": accuracy,
            }
        ]
    )


def save_outputs(
    predictions: pd.DataFrame,
    final_ratings: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Save all second-stage output files."""
    DATA_PROCESSED.mkdir(exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    final_ratings.to_csv(FINAL_RATINGS_PATH, index=False)
    scores.to_csv(SCORES_PATH, index=False)


def print_summary(
    predictions: pd.DataFrame,
    final_ratings: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Print a concise command-line summary."""
    score_row = scores.iloc[0]

    print("\n=== Simple Elo 2025 Summary ===")
    print(f"Number of games processed: {len(predictions)}")
    print(f"Number of unique players: {len(final_ratings)}")
    print(f"Default rating: {DEFAULT_RATING:g}")
    print(f"K: {K:g}")
    print(f"Scale: {SCALE:g}")
    print(f"Log loss: {score_row['log_loss']:.6f}")
    print(f"Brier score: {score_row['brier_score']:.6f}")
    print(f"Accuracy: {score_row['accuracy']:.6f}")
    print("Output paths:")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {FINAL_RATINGS_PATH}")
    print(f"  {SCORES_PATH}")


def main() -> None:
    matches = load_matches()
    predictions, final_ratings = run_elo(matches)
    scores = compute_scores(predictions)
    save_outputs(predictions, final_ratings, scores)
    print_summary(predictions, final_ratings, scores)


if __name__ == "__main__":
    main()
