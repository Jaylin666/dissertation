from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


START_YEAR = 2015
END_YEAR = 2025
EVALUATION_YEARS = [2025]
EPS = 1e-15

PROBABILITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONFIDENCE_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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
PREDICTIONS_PATH = DATA_PROCESSED / f"elo_multiyear_predictions_{YEAR_RANGE}.csv"
METRICS_PATH = DATA_PROCESSED / f"elo_multiyear_metrics_{YEAR_RANGE}.csv"
CALIBRATION_PATH = DATA_PROCESSED / f"elo_multiyear_calibration_{YEAR_RANGE}.csv"
CONFIDENCE_PATH = DATA_PROCESSED / f"elo_multiyear_confidence_{YEAR_RANGE}.csv"


REQUIRED_COLUMNS = ["pred_a_win", "actual_a_win", "year"]


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load multi-year Elo predictions from step 08."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Please run code/08_multiyear_elo.py first."
        )

    predictions = pd.read_csv(path)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in predictions.columns]
    if missing_required:
        raise ValueError(
            f"{path.name} is missing required columns: {missing_required}"
        )

    predictions["pred_a_win"] = pd.to_numeric(predictions["pred_a_win"], errors="coerce")
    predictions["actual_a_win"] = pd.to_numeric(
        predictions["actual_a_win"], errors="coerce"
    )
    predictions["year"] = pd.to_numeric(predictions["year"], errors="coerce").astype("Int64")

    missing_eval_values = predictions[["pred_a_win", "actual_a_win", "year"]].isna().sum()
    if missing_eval_values.sum() > 0:
        print("WARNING: missing values found in key evaluation columns:")
        print(missing_eval_values.to_string())

    return predictions


def filter_evaluation_games(predictions: pd.DataFrame) -> pd.DataFrame:
    """Filter to the fixed evaluation set, normally 2025 games."""
    if "is_evaluation_game" in predictions.columns:
        is_eval = predictions["is_evaluation_game"]
        if is_eval.dtype == object:
            is_eval = is_eval.astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            is_eval = is_eval.astype(bool)
        eval_df = predictions[is_eval].copy()
        filter_source = "is_evaluation_game"
    else:
        print("WARNING: is_evaluation_game column not found; using year filter instead.")
        eval_df = predictions[predictions["year"].isin(EVALUATION_YEARS)].copy()
        filter_source = "year"

    if eval_df.empty:
        raise ValueError(
            "Evaluation set is empty. Check EVALUATION_YEARS or is_evaluation_game."
        )

    eval_df = eval_df.dropna(subset=["pred_a_win", "actual_a_win"]).copy()
    if eval_df.empty:
        raise ValueError("Evaluation set has no rows after dropping missing prediction values.")

    eval_df["actual_a_win"] = eval_df["actual_a_win"].astype(int)
    eval_df["predicted_a_win"] = (eval_df["pred_a_win"] >= 0.5).astype(int)
    eval_df["predicted_correct"] = (
        eval_df["predicted_a_win"] == eval_df["actual_a_win"]
    ).astype(int)
    eval_df["confidence"] = np.maximum(eval_df["pred_a_win"], 1.0 - eval_df["pred_a_win"])
    eval_df.attrs["filter_source"] = filter_source
    return eval_df


def validate_eval_df(eval_df: pd.DataFrame) -> Dict[str, object]:
    """Run sanity checks for the evaluation set."""
    actual_values = sorted(eval_df["actual_a_win"].dropna().unique().tolist())
    if set(actual_values) - {0, 1}:
        print(f"WARNING: actual_a_win has values outside 0/1: {actual_values}")
    if len(actual_values) < 2:
        print(
            "WARNING: actual_a_win does not contain both 0 and 1. "
            "The evaluation side may be incorrectly defined."
        )

    out_of_range_count = int(
        ((eval_df["pred_a_win"] < 0) | (eval_df["pred_a_win"] > 1)).sum()
    )
    if out_of_range_count > 0:
        print(f"WARNING: pred_a_win has {out_of_range_count} values outside [0, 1].")

    return {
        "actual_values": actual_values,
        "out_of_range_count": out_of_range_count,
        "pred_min": eval_df["pred_a_win"].min(),
        "pred_max": eval_df["pred_a_win"].max(),
    }


def compute_metrics(eval_df: pd.DataFrame, checks: Dict[str, object]) -> pd.DataFrame:
    """Compute log loss, Brier score, accuracy and simple baseline accuracy."""
    y = eval_df["actual_a_win"].astype(float)
    pred = eval_df["pred_a_win"].astype(float)

    # Clip only for log loss, because log(0) is undefined.
    clipped_pred = pred.clip(EPS, 1.0 - EPS)
    log_loss = -np.mean(
        y * np.log(clipped_pred) + (1.0 - y) * np.log(1.0 - clipped_pred)
    )
    brier_score = np.mean((pred - y) ** 2)
    accuracy = np.mean(eval_df["predicted_correct"])

    observed_win_rate = y.mean()
    majority_class_baseline_accuracy = max(observed_win_rate, 1.0 - observed_win_rate)

    metrics = {
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "evaluation_years": ",".join(str(year) for year in EVALUATION_YEARS),
        "evaluation_filter_source": eval_df.attrs.get("filter_source", "unknown"),
        "number_of_games": len(eval_df),
        "log_loss": log_loss,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "baseline_accuracy": majority_class_baseline_accuracy,
        "mean_predicted_probability": pred.mean(),
        "observed_win_rate": observed_win_rate,
        "pred_a_win_min": checks["pred_min"],
        "pred_a_win_max": checks["pred_max"],
        "pred_a_win_out_of_range_count": checks["out_of_range_count"],
        "actual_a_win_values": ",".join(str(value) for value in checks["actual_values"]),
        "actual_a_win_count_0": int((eval_df["actual_a_win"] == 0).sum()),
        "actual_a_win_count_1": int((eval_df["actual_a_win"] == 1).sum()),
    }
    return pd.DataFrame([metrics])


def make_bin_labels(bins) -> list:
    """Create readable interval labels such as 0.0-0.1."""
    return [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]


def make_calibration_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Create probability-bin calibration table, keeping empty bins."""
    labels = make_bin_labels(PROBABILITY_BINS)
    df = eval_df.copy()
    df["prob_bin"] = pd.cut(
        df["pred_a_win"],
        bins=PROBABILITY_BINS,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = df.groupby("prob_bin", observed=False)
    calibration = grouped.agg(
        n_games=("actual_a_win", "size"),
        mean_predicted_prob=("pred_a_win", "mean"),
        observed_win_rate=("actual_a_win", "mean"),
    ).reset_index()

    bounds = pd.DataFrame(
        {
            "prob_bin": labels,
            "bin_lower": PROBABILITY_BINS[:-1],
            "bin_upper": PROBABILITY_BINS[1:],
        }
    )
    calibration = bounds.merge(calibration, on="prob_bin", how="left")
    calibration["n_games"] = calibration["n_games"].fillna(0).astype(int)
    calibration["calibration_error"] = (
        calibration["observed_win_rate"] - calibration["mean_predicted_prob"]
    )
    return calibration


def make_confidence_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Create confidence-bin accuracy table, keeping empty bins."""
    labels = make_bin_labels(CONFIDENCE_BINS)
    df = eval_df.copy()
    df["confidence_bin"] = pd.cut(
        df["confidence"],
        bins=CONFIDENCE_BINS,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = df.groupby("confidence_bin", observed=False)
    confidence = grouped.agg(
        n_games=("actual_a_win", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("predicted_correct", "mean"),
    ).reset_index()

    bounds = pd.DataFrame(
        {
            "confidence_bin": labels,
            "bin_lower": CONFIDENCE_BINS[:-1],
            "bin_upper": CONFIDENCE_BINS[1:],
        }
    )
    confidence = bounds.merge(confidence, on="confidence_bin", how="left")
    confidence["n_games"] = confidence["n_games"].fillna(0).astype(int)
    return confidence




def save_outputs(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    confidence: pd.DataFrame,
) -> None:
    """Save evaluation CSV outputs."""
    metrics.to_csv(METRICS_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    confidence.to_csv(CONFIDENCE_PATH, index=False)


def print_command_line_summary(
    metrics: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> None:
    """Print the important results after the script runs."""
    row = metrics.iloc[0]

    print("\n=== Multi-year Elo Evaluation Summary ===")
    print(f"Input file path: {PREDICTIONS_PATH}")
    print(f"Evaluation years: {EVALUATION_YEARS}")
    print(f"Number of evaluation games: {int(row['number_of_games'])}")
    print("actual_a_win value counts:")
    print(eval_df["actual_a_win"].value_counts().sort_index().to_string())
    print(f"pred_a_win min: {row['pred_a_win_min']:.6f}")
    print(f"pred_a_win max: {row['pred_a_win_max']:.6f}")
    print(f"pred_a_win out-of-range count: {int(row['pred_a_win_out_of_range_count'])}")
    print(f"Log loss: {row['log_loss']:.6f}")
    print(f"Brier score: {row['brier_score']:.6f}")
    print(f"Accuracy: {row['accuracy']:.6f}")
    print("Output paths:")
    print(f"  metrics: {METRICS_PATH}")
    print(f"  calibration: {CALIBRATION_PATH}")
    print(f"  confidence: {CONFIDENCE_PATH}")


def main() -> None:
    predictions = load_predictions()
    eval_df = filter_evaluation_games(predictions)
    checks = validate_eval_df(eval_df)
    metrics = compute_metrics(eval_df, checks)
    calibration = make_calibration_table(eval_df)
    confidence = make_confidence_table(eval_df)
    save_outputs(metrics, calibration, confidence)
    print_command_line_summary(metrics, eval_df)


if __name__ == "__main__":
    main()
