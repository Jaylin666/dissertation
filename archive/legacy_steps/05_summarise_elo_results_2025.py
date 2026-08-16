from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_SCALE = 500.0


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run from Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
NOTES_DIR = PROJECT_ROOT / "notes"
DATA_PROCESSED.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)


MATCHES_PATH = DATA_PROCESSED / "matches_2025_checked.csv"
PREDICTIONS_PATH = DATA_PROCESSED / "elo_predictions_2025.csv"
EVALUATION_PATH = DATA_PROCESSED / "elo_evaluation_2025.csv"
CALIBRATION_PATH = DATA_PROCESSED / "elo_calibration_2025.csv"
CONFIDENCE_PATH = DATA_PROCESSED / "elo_calibration_by_confidence_2025.csv"
PARAMETER_RESULTS_PATH = DATA_PROCESSED / "elo_parameter_results_2025.csv"

DEFAULT_VS_BEST_PATH = DATA_PROCESSED / "elo_summary_default_vs_best_2025.csv"
PARAMETER_TOP10_PATH = DATA_PROCESSED / "elo_parameter_top10_2025.csv"
PARAMETER_BOTTOM10_PATH = DATA_PROCESSED / "elo_parameter_bottom10_2025.csv"
CALIBRATION_NONEMPTY_PATH = DATA_PROCESSED / "elo_calibration_nonempty_2025.csv"
CONFIDENCE_SUMMARY_PATH = DATA_PROCESSED / "elo_confidence_summary_2025.csv"
MARKDOWN_PATH = NOTES_DIR / "meeting2_elo_summary_2025.md"


HIDX_FIELDS = [
    "idx_win",
    "idx_lose",
    "grd_win",
    "grd_lose",
    "dgrd_win",
    "dgrd_lose",
    "pt_win",
    "pt_lose",
    "idx20_win",
    "idx20_lose",
    "cinc",
    "rdgrd_win",
    "rdgrd_lose",
    "rpt_win",
    "rpt_lose",
    "idgw",
    "idgl",
]


def load_csv_if_exists(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists; otherwise warn and continue."""
    if not path.exists():
        print(f"WARNING: {label} file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"Loaded {label}: {path} shape={df.shape}")
        return df
    except Exception as exc:
        print(f"WARNING: could not read {label} at {path}: {exc}")
        return None


def count_missing(series: pd.Series) -> int:
    """Count both NaN values and blank strings as missing."""
    cleaned = series.fillna("").astype(str).str.strip()
    return int(cleaned.eq("").sum())


def as_display(value, digits: int = 6) -> str:
    """Format numbers for markdown while keeping missing values readable."""
    if value is None:
        return "not available"
    try:
        if pd.isna(value):
            return "not available"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def get_data_check_summary(matches: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Recalculate the main data checks from the checked matches table."""
    summary = {
        "number_of_matches": "not available",
        "number_of_columns": "not available",
        "missing_event_rows": "not available",
        "missing_hidx_rows": "not available",
        "missing_winner_names": "not available",
        "missing_loser_names": "not available",
    }
    if matches is None:
        return summary

    summary["number_of_matches"] = len(matches)
    summary["number_of_columns"] = matches.shape[1]

    if "eventname" in matches.columns:
        summary["missing_event_rows"] = count_missing(matches["eventname"])
    else:
        print("WARNING: eventname column not found in matches.")
        print(f"Available matches columns: {list(matches.columns)}")

    available_hidx = [col for col in HIDX_FIELDS if col in matches.columns]
    if available_hidx:
        summary["missing_hidx_rows"] = int(matches[available_hidx].isna().all(axis=1).sum())
    else:
        print("WARNING: no expected hidx fields found in matches.")
        print(f"Available matches columns: {list(matches.columns)}")

    if "winner_name" in matches.columns:
        summary["missing_winner_names"] = count_missing(matches["winner_name"])
    else:
        print("WARNING: winner_name column not found in matches.")

    if "loser_name" in matches.columns:
        summary["missing_loser_names"] = count_missing(matches["loser_name"])
    else:
        print("WARNING: loser_name column not found in matches.")

    return summary


def compute_metrics_from_predictions(predictions: pd.DataFrame) -> Dict[str, object]:
    """Fallback metric calculation if elo_evaluation_2025.csv is missing."""
    required = ["actual_a_win", "pred_a_win"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        print(f"WARNING: cannot compute default metrics; missing columns: {missing}")
        print(f"Available prediction columns: {list(predictions.columns)}")
        return {}

    y = pd.to_numeric(predictions["actual_a_win"], errors="coerce")
    pred = pd.to_numeric(predictions["pred_a_win"], errors="coerce")
    eps = 1e-15
    clipped = pred.clip(eps, 1.0 - eps)

    metrics = {
        "number_of_games": len(predictions),
        "log_loss": -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)),
        "brier_score": np.mean((pred - y) ** 2),
        "accuracy": np.mean((pred >= 0.5) == (y == 1.0)),
        "mean_pred_a_win": pred.mean(),
        "actual_a_win_rate": y.mean(),
        "mean_pred_winner_win": np.nan,
    }

    if "pred_winner_win" in predictions.columns:
        metrics["mean_pred_winner_win"] = pd.to_numeric(
            predictions["pred_winner_win"], errors="coerce"
        ).mean()
    return metrics


def get_default_metrics(
    evaluation: Optional[pd.DataFrame], predictions: Optional[pd.DataFrame]
) -> Dict[str, object]:
    """Get default Elo metrics from evaluation output, or recalculate if needed."""
    if evaluation is not None and not evaluation.empty:
        row = evaluation.iloc[0].to_dict()
    elif predictions is not None:
        print("WARNING: evaluation file missing; recalculating metrics from predictions.")
        row = compute_metrics_from_predictions(predictions)
    else:
        row = {}

    row["model_name"] = "Default simple Elo"
    row["K"] = DEFAULT_K
    row["scale"] = DEFAULT_SCALE
    row["default_rating"] = DEFAULT_RATING
    return row


def get_best_parameter_result(parameter_results: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Find the best grid result by minimum log loss."""
    if parameter_results is None or parameter_results.empty:
        return {}
    if "log_loss" not in parameter_results.columns:
        print("WARNING: log_loss column not found in parameter results.")
        print(f"Available parameter columns: {list(parameter_results.columns)}")
        return {}

    best = parameter_results.sort_values("log_loss", ascending=True).iloc[0].to_dict()
    best["model_name"] = "Best grid simple Elo"
    return best


def get_metric(row: Dict[str, object], name: str):
    return row.get(name, np.nan)


def make_default_vs_best_table(
    default_metrics: Dict[str, object], best_metrics: Dict[str, object]
) -> pd.DataFrame:
    """Compare the default Elo result with the best parameter-grid result."""
    rows = []
    default_log_loss = get_metric(default_metrics, "log_loss")
    default_brier = get_metric(default_metrics, "brier_score")
    default_accuracy = get_metric(default_metrics, "accuracy")

    for metrics in [default_metrics, best_metrics]:
        if not metrics:
            continue
        log_loss = get_metric(metrics, "log_loss")
        brier = get_metric(metrics, "brier_score")
        accuracy = get_metric(metrics, "accuracy")

        rows.append(
            {
                "model_name": metrics.get("model_name", "not available"),
                "K": metrics.get("K", np.nan),
                "scale": metrics.get("scale", np.nan),
                "default_rating": metrics.get("default_rating", DEFAULT_RATING),
                "log_loss": log_loss,
                "brier_score": brier,
                "accuracy": accuracy,
                "log_loss_improvement_vs_default": default_log_loss - log_loss,
                "brier_improvement_vs_default": default_brier - brier,
                "accuracy_improvement_vs_default": accuracy - default_accuracy,
            }
        )
    return pd.DataFrame(rows)


def make_parameter_top_bottom_tables(
    parameter_results: Optional[pd.DataFrame],
) -> tuple:
    """Return the top 10 and bottom 10 parameter settings by log loss."""
    if parameter_results is None or parameter_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "log_loss" not in parameter_results.columns:
        print("WARNING: cannot make top/bottom tables because log_loss is missing.")
        print(f"Available parameter columns: {list(parameter_results.columns)}")
        return pd.DataFrame(), pd.DataFrame()

    top10 = parameter_results.sort_values("log_loss", ascending=True).head(10)
    bottom10 = parameter_results.sort_values("log_loss", ascending=False).head(10)
    return top10.reset_index(drop=True), bottom10.reset_index(drop=True)


def select_columns_if_available(
    df: pd.DataFrame, columns: list, label: str
) -> pd.DataFrame:
    """Select expected columns and print available names if any are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"WARNING: {label} missing expected columns: {missing}")
        print(f"Available {label} columns: {list(df.columns)}")
    available = [col for col in columns if col in df.columns]
    return df[available].copy()


def make_calibration_nonempty(calibration: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep only probability bins with at least one game."""
    if calibration is None or calibration.empty:
        return pd.DataFrame()
    if "n_games" not in calibration.columns:
        print("WARNING: calibration table has no n_games column.")
        print(f"Available calibration columns: {list(calibration.columns)}")
        return pd.DataFrame()

    expected = [
        "probability_bin",
        "n_games",
        "mean_predicted_probability",
        "actual_win_rate",
        "calibration_error",
        "abs_calibration_error",
    ]
    nonempty = calibration[pd.to_numeric(calibration["n_games"], errors="coerce") > 0]
    return select_columns_if_available(nonempty, expected, "calibration")


def make_confidence_summary(confidence: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create a compact confidence-bin table for reporting."""
    if confidence is None or confidence.empty:
        return pd.DataFrame()
    expected = ["confidence_bin", "n_games", "mean_confidence", "accuracy"]
    return select_columns_if_available(confidence, expected, "confidence")


def get_high_confidence_note(confidence_summary: pd.DataFrame) -> str:
    """Warn if the highest non-empty confidence bin contains very few games."""
    if confidence_summary.empty or "n_games" not in confidence_summary.columns:
        return ""
    nonempty = confidence_summary[
        pd.to_numeric(confidence_summary["n_games"], errors="coerce") > 0
    ]
    if nonempty.empty:
        return ""
    highest = nonempty.iloc[-1]
    n_games = int(highest["n_games"])
    confidence_bin = highest.get("confidence_bin", "highest non-empty bin")
    if n_games < 20:
        game_word = "game" if n_games == 1 else "games"
        return (
            f" The highest non-empty confidence bin ({confidence_bin}) has only "
            f"{n_games} {game_word}."
        )
    return ""


def make_markdown_summary(
    data_summary: Dict[str, object],
    default_metrics: Dict[str, object],
    best_metrics: Dict[str, object],
    default_vs_best: pd.DataFrame,
    confidence_summary: pd.DataFrame,
) -> str:
    """Build the meeting-ready markdown summary."""
    high_confidence_note = get_high_confidence_note(confidence_summary)

    best_k = as_display(best_metrics.get("K", np.nan), digits=0)
    best_scale = as_display(best_metrics.get("scale", np.nan), digits=0)
    best_log_loss = as_display(best_metrics.get("log_loss"))
    best_brier = as_display(best_metrics.get("brier_score"))
    best_accuracy = as_display(best_metrics.get("accuracy"))

    default_log_loss = as_display(default_metrics.get("log_loss"))
    default_brier = as_display(default_metrics.get("brier_score"))
    default_accuracy = as_display(default_metrics.get("accuracy"))

    improvement_text = "not available"
    if not default_vs_best.empty and len(default_vs_best) >= 2:
        best_row = default_vs_best[default_vs_best["model_name"].eq("Best grid simple Elo")]
        if not best_row.empty:
            row = best_row.iloc[0]
            improvement_text = (
                f"log loss improved by {as_display(row['log_loss_improvement_vs_default'])}, "
                f"Brier score improved by {as_display(row['brier_improvement_vs_default'])}, "
                f"and accuracy changed by {as_display(row['accuracy_improvement_vs_default'])}."
            )

    markdown = f"""# 2025 Simple Elo Prototype Summary

## 1. Current coding progress

* Downloaded raw croquet data for 1985-2025.
* Built a 2025 checked match dataset from games, events, hidx and names.
* Implemented a simple Elo baseline using 2025 data.
* Evaluated predictions using log loss, Brier score and accuracy.
* Built calibration and confidence tables.
* Tested sensitivity to K and scale.

## 2. Data check summary

* Number of matches: {data_summary['number_of_matches']}
* Number of columns in matches table: {data_summary['number_of_columns']}
* Missing event rows: {data_summary['missing_event_rows']}
* Missing hidx rows: {data_summary['missing_hidx_rows']}
* Missing winner names: {data_summary['missing_winner_names']}
* Missing loser names: {data_summary['missing_loser_names']}

## 3. Default simple Elo result

Default parameters:

* default_rating = {DEFAULT_RATING:g}
* K = {DEFAULT_K:g}
* scale = {DEFAULT_SCALE:g}

Default result:

* log_loss = {default_log_loss}
* brier_score = {default_brier}
* accuracy = {default_accuracy}

## 4. Parameter sensitivity result

Best tested parameter setting by log loss:

* best K = {best_k}
* best scale = {best_scale}
* best log_loss = {best_log_loss}
* best brier_score = {best_brier}
* best accuracy = {best_accuracy}

Compared with the default setting, {improvement_text}

Because all players currently start from the same default rating of 1500, more aggressive updates may perform better in this one-year prototype setting. This should not be interpreted as the final optimal parameter choice for the whole project.

## 5. Calibration and confidence summary

* Calibration table was generated by probability bins.
* The extreme bins 0.0-0.1 and 0.9-1.0 may be empty because the model predictions do not reach those ranges.
* Confidence-based accuracy generally increases as confidence increases.
* High-confidence bins with very few games should not be over-interpreted.{high_confidence_note}

## 6. Main interpretation for meeting

I have now built a small prototype pipeline using the 2025 data. The main purpose was not to reproduce the official croquet system yet, but to make sure I can read the raw data, construct a match-level dataset, run a simple Elo baseline, and evaluate probabilistic predictions.

The default simple Elo baseline gives a log loss of about {default_log_loss}, Brier score of about {default_brier}, and accuracy of about {default_accuracy}. A small parameter grid search suggests that more aggressive updates perform better in this one-year prototype, with the best tested setting being K = {best_k} and scale = {best_scale}.

However, I would treat this as an exploratory result rather than a final conclusion, because all players currently start from 1500 and I have not yet used historical ratings or official starting grades.

## 7. Questions for next meeting

1. Should I next use out2025.csv or historical data to initialise ratings more realistically?
2. Should the next comparison be simple Elo vs official historical DG predictions before moving to Glicko or TrueSkill?
3. Is it reasonable to use log loss, Brier score and calibration together as the main evaluation tools?
4. Should I continue with Association Croquet only for now?
5. How much of the official croquet system should I try to reproduce before comparing alternative systems?

## 8. Suggested next coding steps

1. Confirm how to use out2025.csv, especially dgrd_start, as starting ratings.
2. Build an improved Elo baseline using better initial ratings.
3. Compare simple Elo predictions with official DG-based predictions from hidx.
4. Later expand from 2025 only to multiple years or train/test split.
5. Only after that, consider Glicko or TrueSkill.
"""
    return markdown


def save_outputs(
    default_vs_best: pd.DataFrame,
    top10: pd.DataFrame,
    bottom10: pd.DataFrame,
    calibration_nonempty: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    markdown: str,
) -> None:
    """Write all stage 5 summary outputs."""
    default_vs_best.to_csv(DEFAULT_VS_BEST_PATH, index=False)
    top10.to_csv(PARAMETER_TOP10_PATH, index=False)
    bottom10.to_csv(PARAMETER_BOTTOM10_PATH, index=False)
    calibration_nonempty.to_csv(CALIBRATION_NONEMPTY_PATH, index=False)
    confidence_summary.to_csv(CONFIDENCE_SUMMARY_PATH, index=False)
    MARKDOWN_PATH.write_text(markdown, encoding="utf-8")


def print_created_summary() -> None:
    print("\nCreated:")
    print(f"* {DEFAULT_VS_BEST_PATH}")
    print(f"* {PARAMETER_TOP10_PATH}")
    print(f"* {PARAMETER_BOTTOM10_PATH}")
    print(f"* {CALIBRATION_NONEMPTY_PATH}")
    print(f"* {CONFIDENCE_SUMMARY_PATH}")
    print(f"* {MARKDOWN_PATH}")


def main() -> None:
    matches = load_csv_if_exists(MATCHES_PATH, "matches")
    predictions = load_csv_if_exists(PREDICTIONS_PATH, "predictions")
    evaluation = load_csv_if_exists(EVALUATION_PATH, "evaluation")
    calibration = load_csv_if_exists(CALIBRATION_PATH, "calibration")
    confidence = load_csv_if_exists(CONFIDENCE_PATH, "confidence calibration")
    parameter_results = load_csv_if_exists(PARAMETER_RESULTS_PATH, "parameter results")

    data_summary = get_data_check_summary(matches)
    default_metrics = get_default_metrics(evaluation, predictions)
    best_metrics = get_best_parameter_result(parameter_results)

    default_vs_best = make_default_vs_best_table(default_metrics, best_metrics)
    top10, bottom10 = make_parameter_top_bottom_tables(parameter_results)
    calibration_nonempty = make_calibration_nonempty(calibration)
    confidence_summary = make_confidence_summary(confidence)
    markdown = make_markdown_summary(
        data_summary=data_summary,
        default_metrics=default_metrics,
        best_metrics=best_metrics,
        default_vs_best=default_vs_best,
        confidence_summary=confidence_summary,
    )

    save_outputs(
        default_vs_best=default_vs_best,
        top10=top10,
        bottom10=bottom10,
        calibration_nonempty=calibration_nonempty,
        confidence_summary=confidence_summary,
        markdown=markdown,
    )
    print_created_summary()


if __name__ == "__main__":
    main()
