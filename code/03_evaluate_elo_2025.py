from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_NUMBER_OF_GAMES = 11379


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
PREDICTIONS_PATH = DATA_PROCESSED / "elo_predictions_2025.csv"
EVALUATION_PATH = DATA_PROCESSED / "elo_evaluation_2025.csv"
CALIBRATION_PATH = DATA_PROCESSED / "elo_calibration_2025.csv"
CONFIDENCE_PATH = DATA_PROCESSED / "elo_calibration_by_confidence_2025.csv"


REQUIRED_COLUMNS = ["actual_a_win", "pred_a_win"]
OPTIONAL_COLUMNS = ["pred_winner_win", "player_a", "player_b", "winner", "loser"]


def load_predictions(path=PREDICTIONS_PATH):
    """Load Elo prediction rows from stage 2."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/02_simple_elo_2025.py first."
        )

    predictions = pd.read_csv(path)
    print(f"Loaded predictions: {path}")
    print(f"Shape: {predictions.shape}")
    return predictions


def validate_predictions(predictions):
    """Run basic checks before calculating evaluation metrics."""
    missing_required = [
        col for col in REQUIRED_COLUMNS if col not in predictions.columns
    ]
    if missing_required:
        raise ValueError(
            f"elo_predictions_2025.csv is missing required columns: {missing_required}"
        )

    for col in OPTIONAL_COLUMNS:
        if col not in predictions.columns:
            print(f"WARNING: optional column {col!r} is missing.")

    predictions["actual_a_win"] = pd.to_numeric(
        predictions["actual_a_win"], errors="coerce"
    )
    predictions["pred_a_win"] = pd.to_numeric(
        predictions["pred_a_win"], errors="coerce"
    )

    if "pred_winner_win" in predictions.columns:
        predictions["pred_winner_win"] = pd.to_numeric(
            predictions["pred_winner_win"], errors="coerce"
        )

    actual_values = sorted(predictions["actual_a_win"].dropna().unique().tolist())
    invalid_actual = [
        value for value in actual_values if value not in [0, 1, 0.0, 1.0]
    ]
    if invalid_actual:
        raise ValueError(
            f"actual_a_win should contain only 0 and 1, but found: {invalid_actual}"
        )

    out_of_range = (
        (predictions["pred_a_win"] < 0) | (predictions["pred_a_win"] > 1)
    ).sum()
    if out_of_range > 0:
        raise ValueError(
            f"pred_a_win has {out_of_range} values outside the [0, 1] range."
        )

    missing_values = predictions[REQUIRED_COLUMNS].isna().sum()
    if missing_values.sum() > 0:
        raise ValueError(
            "Required evaluation columns contain missing values:\n"
            + missing_values.to_string()
        )

    optional_present = [
        col for col in OPTIONAL_COLUMNS if col in predictions.columns
    ]
    optional_missing = predictions[optional_present].isna().sum()

    print("\n--- Basic checks ---")
    print(f"actual_a_win values: {actual_values}")
    print(f"pred_a_win min: {predictions['pred_a_win'].min():.6f}")
    print(f"pred_a_win max: {predictions['pred_a_win'].max():.6f}")
    print("Missing values in required columns:")
    print(missing_values.to_string())
    if optional_present:
        print("Missing values in optional columns:")
        print(optional_missing.to_string())

    if len(predictions) != EXPECTED_NUMBER_OF_GAMES:
        print(
            f"WARNING: expected {EXPECTED_NUMBER_OF_GAMES} games, "
            f"but found {len(predictions)}."
        )

    return predictions


def compute_overall_metrics(predictions):
    """Compute overall log loss, Brier score, accuracy, and means."""
    eps = 1e-15
    y = predictions["actual_a_win"].astype(float)
    pred = predictions["pred_a_win"].astype(float)

    # Clip only for log loss, because log(0) is undefined.
    p = pred.clip(eps, 1.0 - eps)
    log_loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean((pred >= 0.5) == (y == 1.0))

    if "pred_winner_win" in predictions.columns:
        mean_pred_winner_win = predictions["pred_winner_win"].mean()
    else:
        mean_pred_winner_win = np.nan

    return pd.DataFrame(
        [
            {
                "number_of_games": len(predictions),
                "log_loss": log_loss,
                "brier_score": brier_score,
                "accuracy": accuracy,
                "mean_pred_a_win": pred.mean(),
                "actual_a_win_rate": y.mean(),
                "mean_pred_winner_win": mean_pred_winner_win,
            }
        ]
    )


def make_calibration_table(predictions):
    """Group games into probability bins and compare predicted vs actual rates."""
    df = predictions.copy()
    bins = np.linspace(0.0, 1.0, 11)
    labels = [
        "0.0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]

    # Empty bins are kept so the full 0.0 to 1.0 range is visible in the CSV.
    df["probability_bin"] = pd.cut(
        df["pred_a_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = df.groupby("probability_bin", observed=False)
    calibration = grouped.agg(
        n_games=("actual_a_win", "size"),
        mean_predicted_probability=("pred_a_win", "mean"),
        actual_win_rate=("actual_a_win", "mean"),
    ).reset_index()

    calibration["calibration_error"] = (
        calibration["actual_win_rate"]
        - calibration["mean_predicted_probability"]
    )
    calibration["abs_calibration_error"] = calibration[
        "calibration_error"
    ].abs()
    return calibration


def make_confidence_table(predictions):
    """Evaluate accuracy by model confidence in its chosen side."""
    df = predictions.copy()
    df["confidence"] = np.maximum(df["pred_a_win"], 1.0 - df["pred_a_win"])
    df["correct_prediction"] = (
        (df["pred_a_win"] >= 0.5) == (df["actual_a_win"] == 1)
    )

    bins = np.linspace(0.5, 1.0, 11)
    labels = [
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]

    df["confidence_bin"] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = df.groupby("confidence_bin", observed=False)
    confidence_table = grouped.agg(
        n_games=("actual_a_win", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct_prediction", "mean"),
        mean_pred_a_win=("pred_a_win", "mean"),
        actual_a_win_rate=("actual_a_win", "mean"),
    ).reset_index()
    return confidence_table


def save_outputs(evaluation, calibration, confidence_table):
    """Save all evaluation outputs as CSV files."""
    DATA_PROCESSED.mkdir(exist_ok=True)
    evaluation.to_csv(EVALUATION_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    confidence_table.to_csv(CONFIDENCE_PATH, index=False)


def print_summary(evaluation):
    """Print the most important evaluation results."""
    row = evaluation.iloc[0]

    print("\n=== Elo Evaluation 2025 Summary ===")
    print(f"Number of games: {int(row['number_of_games'])}")
    print(f"Log loss: {row['log_loss']:.6f}")
    print(f"Brier score: {row['brier_score']:.6f}")
    print(f"Accuracy: {row['accuracy']:.6f}")
    print(
        "Mean predicted probability for player_a: "
        f"{row['mean_pred_a_win']:.6f}"
    )
    print(f"Actual win rate for player_a: {row['actual_a_win_rate']:.6f}")
    if pd.notna(row["mean_pred_winner_win"]):
        print(
            "Mean predicted probability for actual winner: "
            f"{row['mean_pred_winner_win']:.6f}"
        )
    else:
        print("Mean predicted probability for actual winner: NaN")
    print(f"Calibration output path: {CALIBRATION_PATH}")
    print(f"Confidence calibration output path: {CONFIDENCE_PATH}")
    print(f"Evaluation output path: {EVALUATION_PATH}")


def main():
    predictions = load_predictions()
    predictions = validate_predictions(predictions)
    evaluation = compute_overall_metrics(predictions)
    calibration = make_calibration_table(predictions)
    confidence_table = make_confidence_table(predictions)
    save_outputs(evaluation, calibration, confidence_table)
    print_summary(evaluation)


if __name__ == "__main__":
    main()
