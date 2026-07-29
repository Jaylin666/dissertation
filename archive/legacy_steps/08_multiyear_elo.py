from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


START_YEAR = 2015
END_YEAR = 2025
EVALUATION_YEARS = [2025]

DEFAULT_RATING = 1500.0
K = 20.0
SCALE = 500.0


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
DATA_PROCESSED.mkdir(exist_ok=True)

YEAR_RANGE = f"{START_YEAR}_{END_YEAR}"
MATCHES_PATH = DATA_PROCESSED / f"matches_{YEAR_RANGE}_checked.csv"
PREDICTIONS_PATH = DATA_PROCESSED / f"elo_multiyear_predictions_{YEAR_RANGE}.csv"
FINAL_RATINGS_PATH = DATA_PROCESSED / f"elo_multiyear_final_ratings_{YEAR_RANGE}.csv"
RATING_HISTORY_PATH = DATA_PROCESSED / f"elo_multiyear_rating_history_{YEAR_RANGE}.csv"
RUN_SUMMARY_PATH = DATA_PROCESSED / f"elo_multiyear_run_summary_{YEAR_RANGE}.csv"


REQUIRED_COLUMNS = ["fcode", "code", "year", "winner", "loser"]
NUMERIC_ID_COLUMNS = ["fcode", "code", "year", "event", "event_fcode", "winner", "loser", "country"]
OPTIONAL_OUTPUT_COLUMNS = [
    "event",
    "event_fcode",
    "eventname",
    "event_date_raw",
    "event_date_parsed",
    "country",
    "country_name",
    "winner_name",
    "loser_name",
]


def load_matches(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Load and sort the multi-year match dataset from step 07."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/07_build_multiyear_match_dataset.py first."
        )

    matches = pd.read_csv(path)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_OUTPUT_COLUMNS:
        if col not in matches.columns:
            print(f"WARNING: optional column {col!r} not found; filling with NA.")
            matches[col] = pd.NA

    for col in NUMERIC_ID_COLUMNS:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    if "event_date_parsed" in matches.columns:
        matches["event_date_parsed"] = pd.to_datetime(
            matches["event_date_parsed"], errors="coerce"
        )
    else:
        matches["event_date_parsed"] = pd.NaT

    missing_key_counts = matches[REQUIRED_COLUMNS].isna().sum()
    missing_key_counts = missing_key_counts[missing_key_counts > 0]
    if not missing_key_counts.empty:
        print("WARNING: missing values found in required columns:")
        print(missing_key_counts.to_string())
        print("Rows with missing winner or loser will be skipped in the Elo loop.")

    sort_cols = [
        col
        for col in ["year", "event_date_parsed", "event", "code", "fcode"]
        if col in matches.columns
    ]
    matches = matches.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return matches


def expected_score(rating_a: float, rating_b: float, scale: float = SCALE) -> float:
    """Return the Elo probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def get_player_name(player_names: Dict[int, str], player_code: int) -> object:
    """Get a player's stored name, or NA if we have not seen one."""
    return player_names.get(player_code, pd.NA)


def update_player_name(
    player_names: Dict[int, str], player_code: int, possible_name: object
) -> None:
    """Store the first non-missing player name seen for a player code."""
    if player_code in player_names:
        return
    if pd.notna(possible_name) and str(possible_name).strip():
        player_names[player_code] = str(possible_name).strip()


def ensure_player(
    player_code: int,
    year: int,
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    first_year_seen: Dict[int, int],
    last_year_seen: Dict[int, int],
) -> None:
    """Initialise a player if this is their first appearance."""
    if player_code not in ratings:
        ratings[player_code] = DEFAULT_RATING
        stats[player_code] = {"n_games": 0, "n_wins": 0, "n_losses": 0}
        first_year_seen[player_code] = year
    last_year_seen[player_code] = year


def run_elo(matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Run pure simple Elo through the multi-year dataset in chronological order."""
    ratings: Dict[int, float] = {}
    stats: Dict[int, Dict[str, int]] = {}
    first_year_seen: Dict[int, int] = {}
    last_year_seen: Dict[int, int] = {}
    player_names: Dict[int, str] = {}

    prediction_rows = []
    rating_history_rows = []
    skipped_games = 0

    for _, row in matches.iterrows():
        if pd.isna(row["winner"]) or pd.isna(row["loser"]):
            skipped_games += 1
            continue

        winner = int(row["winner"])
        loser = int(row["loser"])
        year = int(row["year"])

        ensure_player(winner, year, ratings, stats, first_year_seen, last_year_seen)
        ensure_player(loser, year, ratings, stats, first_year_seen, last_year_seen)
        update_player_name(player_names, winner, row.get("winner_name"))
        update_player_name(player_names, loser, row.get("loser_name"))

        winner_rating_before = ratings[winner]
        loser_rating_before = ratings[loser]
        pred_winner_win = expected_score(winner_rating_before, loser_rating_before, SCALE)

        # Evaluation side is fixed by player code, not by winner/loser.
        player_a = min(winner, loser)
        player_b = max(winner, loser)
        actual_a_win = 1 if player_a == winner else 0
        pred_a_win = pred_winner_win if player_a == winner else 1.0 - pred_winner_win
        player_a_rating_before = ratings[player_a]
        player_b_rating_before = ratings[player_b]

        rating_change = K * (1.0 - pred_winner_win)
        winner_rating_after = winner_rating_before + rating_change
        loser_rating_after = loser_rating_before - rating_change

        ratings[winner] = winner_rating_after
        ratings[loser] = loser_rating_after

        stats[winner]["n_games"] += 1
        stats[winner]["n_wins"] += 1
        stats[loser]["n_games"] += 1
        stats[loser]["n_losses"] += 1
        last_year_seen[winner] = year
        last_year_seen[loser] = year

        is_evaluation_game = year in EVALUATION_YEARS

        prediction_rows.append(
            {
                "fcode": int(row["fcode"]),
                "code": int(row["code"]),
                "year": year,
                "event": row.get("event", pd.NA),
                "event_fcode": row.get("event_fcode", pd.NA),
                "eventname": row.get("eventname", pd.NA),
                "event_date_raw": row.get("event_date_raw", pd.NA),
                "event_date_parsed": row.get("event_date_parsed", pd.NaT),
                "country": row.get("country", pd.NA),
                "country_name": row.get("country_name", pd.NA),
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
                "player_a_rating_before": player_a_rating_before,
                "player_b_rating_before": player_b_rating_before,
                "rating_change": rating_change,
                "K": K,
                "scale": SCALE,
                "default_rating": DEFAULT_RATING,
                "is_evaluation_game": is_evaluation_game,
            }
        )

        rating_history_rows.extend(
            [
                {
                    "fcode": int(row["fcode"]),
                    "year": year,
                    "event": row.get("event", pd.NA),
                    "code": int(row["code"]),
                    "player_code": winner,
                    "player_name": get_player_name(player_names, winner),
                    "opponent_code": loser,
                    "opponent_name": get_player_name(player_names, loser),
                    "result": 1,
                    "rating_before": winner_rating_before,
                    "rating_after": winner_rating_after,
                    "rating_change_abs": abs(winner_rating_after - winner_rating_before),
                    "K": K,
                    "scale": SCALE,
                },
                {
                    "fcode": int(row["fcode"]),
                    "year": year,
                    "event": row.get("event", pd.NA),
                    "code": int(row["code"]),
                    "player_code": loser,
                    "player_name": get_player_name(player_names, loser),
                    "opponent_code": winner,
                    "opponent_name": get_player_name(player_names, winner),
                    "result": 0,
                    "rating_before": loser_rating_before,
                    "rating_after": loser_rating_after,
                    "rating_change_abs": abs(loser_rating_after - loser_rating_before),
                    "K": K,
                    "scale": SCALE,
                },
            ]
        )

    predictions = pd.DataFrame(prediction_rows)
    rating_history = pd.DataFrame(rating_history_rows)
    final_ratings = make_final_ratings_df(
        ratings=ratings,
        stats=stats,
        first_year_seen=first_year_seen,
        last_year_seen=last_year_seen,
        player_names=player_names,
    )
    return predictions, final_ratings, rating_history, skipped_games


def make_final_ratings_df(
    ratings: Dict[int, float],
    stats: Dict[int, Dict[str, int]],
    first_year_seen: Dict[int, int],
    last_year_seen: Dict[int, int],
    player_names: Dict[int, str],
) -> pd.DataFrame:
    """Build the final ratings table after the Elo run."""
    rows = []
    for player_code, final_rating in ratings.items():
        rows.append(
            {
                "player_code": player_code,
                "player_name": get_player_name(player_names, player_code),
                "final_rating": final_rating,
                "n_games": stats[player_code]["n_games"],
                "n_wins": stats[player_code]["n_wins"],
                "n_losses": stats[player_code]["n_losses"],
                "first_year_seen": first_year_seen[player_code],
                "last_year_seen": last_year_seen[player_code],
            }
        )

    final_ratings = pd.DataFrame(rows)
    if final_ratings.empty:
        return final_ratings

    return final_ratings.sort_values(
        ["final_rating", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)


def make_run_summary(
    predictions: pd.DataFrame,
    final_ratings: pd.DataFrame,
    rating_history: pd.DataFrame,
    skipped_games: int,
) -> pd.DataFrame:
    """Create a one-row summary of this multi-year Elo run."""
    if rating_history.empty:
        average_abs_rating_change = np.nan
        median_abs_rating_change = np.nan
        max_abs_rating_change = np.nan
    else:
        average_abs_rating_change = rating_history["rating_change_abs"].mean()
        median_abs_rating_change = rating_history["rating_change_abs"].median()
        max_abs_rating_change = rating_history["rating_change_abs"].max()

    if final_ratings.empty:
        final_rating_mean = np.nan
        final_rating_std = np.nan
        number_of_unique_players = 0
    else:
        final_rating_mean = final_ratings["final_rating"].mean()
        final_rating_std = final_ratings["final_rating"].std()
        number_of_unique_players = len(final_ratings)

    summary = {
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "evaluation_years": ",".join(str(year) for year in EVALUATION_YEARS),
        "number_of_games": len(predictions),
        "number_of_evaluation_games": int(predictions["is_evaluation_game"].sum())
        if "is_evaluation_game" in predictions.columns
        else 0,
        "number_of_unique_players": number_of_unique_players,
        "skipped_games_missing_winner_or_loser": skipped_games,
        "default_rating": DEFAULT_RATING,
        "K": K,
        "scale": SCALE,
        "average_abs_rating_change": average_abs_rating_change,
        "median_abs_rating_change": median_abs_rating_change,
        "max_abs_rating_change": max_abs_rating_change,
        "final_rating_mean": final_rating_mean,
        "final_rating_std": final_rating_std,
        "output_predictions_path": str(PREDICTIONS_PATH),
        "output_final_ratings_path": str(FINAL_RATINGS_PATH),
        "output_rating_history_path": str(RATING_HISTORY_PATH),
    }
    return pd.DataFrame([summary])


def save_outputs(
    predictions: pd.DataFrame,
    final_ratings: pd.DataFrame,
    rating_history: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> None:
    """Save all outputs for this step."""
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    final_ratings.to_csv(FINAL_RATINGS_PATH, index=False)
    rating_history.to_csv(RATING_HISTORY_PATH, index=False)
    run_summary.to_csv(RUN_SUMMARY_PATH, index=False)


def print_summary(run_summary: pd.DataFrame) -> None:
    """Print a simple command-line sanity check."""
    row = run_summary.iloc[0]

    print("\n=== Multi-year Elo Summary ===")
    print(f"Number of games processed: {row['number_of_games']}")
    print(f"Number of unique players: {row['number_of_unique_players']}")
    print(f"Number of evaluation games: {row['number_of_evaluation_games']}")
    print(f"Skipped games with missing winner/loser: {row['skipped_games_missing_winner_or_loser']}")
    print(f"Start year: {row['start_year']}")
    print(f"End year: {row['end_year']}")
    print(f"Default rating: {row['default_rating']}")
    print(f"K: {row['K']}")
    print(f"Scale: {row['scale']}")
    print(f"Mean final rating: {row['final_rating_mean']:.6f}")
    print(f"Standard deviation of final ratings: {row['final_rating_std']:.6f}")
    print("Output paths:")
    print(f"  predictions: {PREDICTIONS_PATH}")
    print(f"  final ratings: {FINAL_RATINGS_PATH}")
    print(f"  rating history: {RATING_HISTORY_PATH}")
    print(f"  run summary: {RUN_SUMMARY_PATH}")


def main() -> None:
    matches = load_matches()
    predictions, final_ratings, rating_history, skipped_games = run_elo(matches)
    run_summary = make_run_summary(
        predictions=predictions,
        final_ratings=final_ratings,
        rating_history=rating_history,
        skipped_games=skipped_games,
    )
    save_outputs(predictions, final_ratings, rating_history, run_summary)
    print_summary(run_summary)


if __name__ == "__main__":
    main()
