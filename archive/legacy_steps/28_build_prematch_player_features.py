"""Build leakage-free prematch player features."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

START_YEAR = 1985
END_YEAR = 2025
EXPECTED_FULL_HISTORY_MATCHES = 456_382
EXPECTED_2025_GAMES = 11_379
EXPECTED_LONG_ROWS = EXPECTED_2025_GAMES * 2
RANDOM_SEED = 20260713

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meeting6"

LONG_FEATURES_PATH = OUTPUT_DIR / "28_prematch_player_features_2025_long.csv"
MATCH_FEATURES_PATH = OUTPUT_DIR / "28_prematch_match_features_2025.csv"
VALIDATION_CHECKS_PATH = OUTPUT_DIR / "28_feature_validation_checks.csv"
SPOT_CHECKS_PATH = OUTPUT_DIR / "28_feature_spot_checks.csv"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "28_feature_summary.csv"
GROUP_COUNTS_PATH = OUTPUT_DIR / "28_feature_group_counts.csv"

REQUIRED_MATCH_COLUMNS = ["fcode", "year", "event", "winner", "loser"]
OPTIONAL_MATCH_COLUMNS = ["event_date_raw", "event_date_parsed", "winner_name", "loser_name"]

SUMMARY_VARIABLES = [
    "total_games_before",
    "games_last_90_days",
    "games_last_365_days",
    "games_previous_calendar_year",
    "days_since_last_game",
    "career_days_before",
    "min_total_games_before",
    "min_games_last_365_days",
    "max_days_since_last_game",
]


def find_file(filename: str, preferred_relative: str | None = None) -> Path | None:
    """Find an input file, trying the expected project-relative path first."""

    if preferred_relative:
        preferred = PROJECT_ROOT / preferred_relative
        if preferred.exists():
            return preferred
    matches = sorted(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def player_code(value: Any) -> int:
    """Convert player IDs from CSV numeric values to stable integer IDs."""

    return int(float(value))


def add_validation_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any = "",
    severity: str = "error",
    detail: str = "",
) -> None:
    """Append one validation check row."""

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


def add_event_ordering_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add the same ordering helper columns used by the meeting 5 scripts."""

    matches = matches.copy()
    if "event_date_raw" not in matches.columns:
        matches["event_date_raw"] = pd.NA
    if "event_date_parsed" not in matches.columns:
        matches["event_date_parsed"] = pd.NA

    if "event_order_date" in matches.columns and "event_date_ordering_method" in matches.columns:
        matches["event_order_date"] = pd.to_datetime(matches["event_order_date"], errors="coerce")
        return matches

    parsed = pd.to_datetime(matches["event_date_parsed"], errors="coerce")
    matches["event_order_date"] = parsed
    matches["event_date_ordering_method"] = np.where(
        parsed.notna(),
        "parsed_full_date",
        "fallback_no_date",
    )

    missing_parsed = parsed.isna()
    raw = matches.loc[missing_parsed, "event_date_raw"].astype("string").str.strip()
    extracted = raw.str.extract(r"^(?P<month>\d{1,2})\.(?P<year>\d{2}|\d{4})$")
    valid_month_year = extracted["month"].notna()

    if valid_month_year.any():
        months = pd.to_numeric(extracted.loc[valid_month_year, "month"], errors="coerce")
        raw_years = extracted.loc[valid_month_year, "year"].astype(str)
        years_numeric = raw_years.astype(int)
        years = np.where(
            raw_years.str.len().eq(2),
            np.where(years_numeric >= 85, 1900 + years_numeric, 2000 + years_numeric),
            years_numeric,
        )
        valid_month = months.between(1, 12).fillna(False)
        valid_mask = valid_month.to_numpy(dtype=bool)
        valid_index = extracted.loc[valid_month_year].index[valid_mask]
        imputed_dates = pd.to_datetime(
            {
                "year": np.asarray(years)[valid_mask],
                "month": months.loc[valid_index].astype(int).to_numpy(),
                "day": np.repeat(15, len(valid_index)),
            },
            errors="coerce",
        )
        matches.loc[valid_index, "event_order_date"] = imputed_dates.to_numpy()
        matches.loc[valid_index, "event_date_ordering_method"] = "month_year_imputed"

    return matches


def date_quality_from_method(method: Any) -> str:
    """Map project ordering method names to compact feature-quality labels."""

    if method == "parsed_full_date":
        return "exact"
    if method == "month_year_imputed":
        return "project_fallback"
    return "missing"


def load_canonical_matches(validation_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, Path]:
    """Load the full-history checked dataset and add stable match sequences."""

    dataset_path = find_file(
        "matches_1985_2025_checked.csv",
        "outputs/elo_optimization/matches_1985_2025_checked.csv",
    )
    if dataset_path is None:
        raise FileNotFoundError("Could not find matches_1985_2025_checked.csv")

    matches = pd.read_csv(dataset_path, low_memory=False)
    missing_required = [col for col in REQUIRED_MATCH_COLUMNS if col not in matches.columns]
    if missing_required:
        raise ValueError(f"{dataset_path.name} is missing required columns: {missing_required}")

    for col in OPTIONAL_MATCH_COLUMNS:
        if col not in matches.columns:
            matches[col] = pd.NA

    for col in REQUIRED_MATCH_COLUMNS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    missing_ids = matches[REQUIRED_MATCH_COLUMNS].isna().sum()
    if int(missing_ids.sum()) > 0:
        raise ValueError(f"Required ID columns contain missing values:\n{missing_ids}")

    matches = matches[(matches["year"] >= START_YEAR) & (matches["year"] <= END_YEAR)].copy()
    matches = add_event_ordering_columns(matches)
    matches["date_quality"] = matches["event_date_ordering_method"].map(date_quality_from_method)
    matches["event_order_date_missing"] = matches["event_order_date"].isna()
    matches = (
        matches.sort_values(
            ["year", "event_order_date_missing", "event_order_date", "event", "fcode"],
            na_position="last",
        )
        .drop(columns=["event_order_date_missing"])
        .reset_index(drop=True)
    )
    matches["match_sequence"] = np.arange(1, len(matches) + 1, dtype=int)
    matches["match_id"] = matches["fcode"].astype(int)
    matches["year"] = matches["year"].astype(int)
    matches["event"] = matches["event"].astype(int)
    matches["winner"] = matches["winner"].astype(int)
    matches["loser"] = matches["loser"].astype(int)

    add_validation_check(
        validation_rows,
        "full_history_matches",
        len(matches) == EXPECTED_FULL_HISTORY_MATCHES,
        len(matches),
        EXPECTED_FULL_HISTORY_MATCHES,
        detail="Canonical full-history match rows after year filtering.",
    )
    return matches, dataset_path


def validate_chronological_order(matches: pd.DataFrame, validation_rows: list[dict[str, Any]]) -> None:
    """Check uniqueness and monotonicity of the match ordering."""

    add_validation_check(
        validation_rows,
        "match_id_unique_full_history",
        not matches["match_id"].duplicated().any(),
        int(matches["match_id"].duplicated().sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "match_sequence_unique_full_history",
        not matches["match_sequence"].duplicated().any(),
        int(matches["match_sequence"].duplicated().sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "match_sequence_monotone",
        bool(matches["match_sequence"].is_monotonic_increasing),
        bool(matches["match_sequence"].is_monotonic_increasing),
        True,
    )
    add_validation_check(
        validation_rows,
        "date_quality_values",
        set(matches["date_quality"].dropna().unique()).issubset({"exact", "project_fallback", "missing"}),
        ", ".join(sorted(matches["date_quality"].dropna().unique())),
        "exact/project_fallback/missing",
    )


def make_initial_state() -> dict[str, Any]:
    """Create the mutable pre-match history state for one player."""

    return {
        "total_games": 0,
        "first_game_date": None,
        "last_game_date": None,
        "recent_game_dates": deque(),
        "games_by_year": defaultdict(int),
        "missing_dated_games": 0,
    }


def count_recent_games(recent_dates: deque[pd.Timestamp], current_date: pd.Timestamp, days: int) -> int:
    """Count prior games in a rolling window, including earlier same-day games."""

    window_start = current_date - pd.Timedelta(days=days)
    return int(sum(1 for game_date in recent_dates if game_date >= window_start))


def get_prematch_player_features(
    state: dict[str, Any],
    player_id: int,
    opponent_id: int,
    side: str,
    row: Any,
) -> dict[str, Any]:
    """Return one player's features before the current match is applied."""

    current_date = row.event_order_date
    has_current_date = pd.notna(current_date)
    current_date = pd.Timestamp(current_date) if has_current_date else pd.NaT
    total_games_before = int(state["total_games"])
    has_previous_history = total_games_before > 0
    date_features_available = bool(
        has_current_date
        and state["missing_dated_games"] == 0
        and (not has_previous_history or state["last_game_date"] is not None)
    )

    if date_features_available:
        games_last_90_days = count_recent_games(state["recent_game_dates"], current_date, 90)
        games_last_365_days = count_recent_games(state["recent_game_dates"], current_date, 365)
        if has_previous_history:
            last_game_date = pd.Timestamp(state["last_game_date"])
            first_game_date = pd.Timestamp(state["first_game_date"])
            days_since_last_game = int((current_date - last_game_date).days)
            career_days_before = int((current_date - first_game_date).days)
        else:
            days_since_last_game = np.nan
            career_days_before = np.nan
    else:
        games_last_90_days = np.nan
        games_last_365_days = np.nan
        days_since_last_game = np.nan
        career_days_before = np.nan

    return {
        "match_id": int(row.match_id),
        "match_sequence": int(row.match_sequence),
        "year": int(row.year),
        "event_id": int(row.event),
        "match_date": current_date.date().isoformat() if has_current_date else pd.NA,
        "player_id": int(player_id),
        "opponent_id": int(opponent_id),
        "player_side": side,
        "total_games_before": total_games_before,
        "games_last_90_days": games_last_90_days,
        "games_last_365_days": games_last_365_days,
        "games_previous_calendar_year": int(state["games_by_year"].get(int(row.year) - 1, 0)),
        "days_since_last_game": days_since_last_game,
        "career_days_before": career_days_before,
        "is_debut": bool(total_games_before == 0),
        "has_previous_history": bool(has_previous_history),
        "date_features_available": bool(date_features_available),
        "date_quality": row.date_quality,
    }


def update_player_state(state: dict[str, Any], year: int, match_date: Any) -> None:
    """Apply the current match to one player's history state."""

    state["total_games"] += 1
    state["games_by_year"][int(year)] += 1
    if pd.notna(match_date):
        current_date = pd.Timestamp(match_date)
        if state["first_game_date"] is None:
            state["first_game_date"] = current_date
        state["last_game_date"] = current_date
        state["recent_game_dates"].append(current_date)
        cutoff = current_date - pd.Timedelta(days=365)
        while state["recent_game_dates"] and state["recent_game_dates"][0] < cutoff:
            state["recent_game_dates"].popleft()
    else:
        state["missing_dated_games"] += 1


def build_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Build two pre-match feature rows per 2025 evaluation match."""

    player_state: defaultdict[int, dict[str, Any]] = defaultdict(make_initial_state)
    rows: list[dict[str, Any]] = []
    needed_cols = [
        "match_id",
        "match_sequence",
        "year",
        "event",
        "event_order_date",
        "date_quality",
        "winner",
        "loser",
    ]

    for row in matches[needed_cols].itertuples(index=False):
        winner = player_code(row.winner)
        loser = player_code(row.loser)
        player_a = min(winner, loser)
        player_b = max(winner, loser)

        if int(row.year) == END_YEAR:
            rows.append(
                get_prematch_player_features(
                    player_state[player_a],
                    player_a,
                    player_b,
                    "A",
                    row,
                )
            )
            rows.append(
                get_prematch_player_features(
                    player_state[player_b],
                    player_b,
                    player_a,
                    "B",
                    row,
                )
            )

        update_player_state(player_state[winner], int(row.year), row.event_order_date)
        update_player_state(player_state[loser], int(row.year), row.event_order_date)

    return pd.DataFrame(rows)


def bool_value(value: Any) -> bool:
    """Convert pandas/numpy scalar booleans to plain bool."""

    if pd.isna(value):
        return False
    return bool(value)


def numeric_min(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(min(float(a), float(b)))


def numeric_max(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(max(float(a), float(b)))


def numeric_abs_diff(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(abs(float(a) - float(b)))


def build_match_level_features(matches: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse A/B player rows into one symmetric feature row per match."""

    eval_matches = matches.loc[matches["year"] == END_YEAR].copy()
    a_rows = long_df.loc[long_df["player_side"] == "A"].set_index("match_id")
    b_rows = long_df.loc[long_df["player_side"] == "B"].set_index("match_id")
    rows: list[dict[str, Any]] = []

    for match in eval_matches.itertuples(index=False):
        match_id = int(match.match_id)
        a = a_rows.loc[match_id]
        b = b_rows.loc[match_id]
        player_a_id = int(a["player_id"])
        player_b_id = int(b["player_id"])
        winner_id = int(match.winner)
        loser_id = int(match.loser)
        player_a_is_winner = player_a_id == winner_id

        row = {
            "match_id": match_id,
            "match_sequence": int(match.match_sequence),
            "year": int(match.year),
            "event_id": int(match.event),
            "match_date": a["match_date"],
            "player_a_id": player_a_id,
            "player_b_id": player_b_id,
            "winner_id": winner_id,
            "loser_id": loser_id,
            "player_a_is_winner": bool(player_a_is_winner),
            "outcome_a": int(player_a_is_winner),
        }

        for prefix, values in [("a", a), ("b", b)]:
            for col in [
                "total_games_before",
                "games_last_90_days",
                "games_last_365_days",
                "games_previous_calendar_year",
                "days_since_last_game",
                "career_days_before",
                "is_debut",
                "has_previous_history",
                "date_features_available",
                "date_quality",
            ]:
                row[f"{prefix}_{col}"] = values[col]

        paired_numeric = [
            ("total_games_before", "total_games_before"),
            ("games_last_90_days", "games_last_90_days"),
            ("games_last_365_days", "games_last_365_days"),
            ("previous_year_games", "games_previous_calendar_year"),
        ]
        for out_name, source_name in paired_numeric:
            a_value = row[f"a_{source_name}"]
            b_value = row[f"b_{source_name}"]
            row[f"min_{out_name}"] = numeric_min(a_value, b_value)
            row[f"max_{out_name}"] = numeric_max(a_value, b_value)
            row[f"abs_diff_{out_name}"] = numeric_abs_diff(a_value, b_value)

        if bool_value(row["a_has_previous_history"]) and bool_value(row["b_has_previous_history"]):
            row["min_days_since_last_game"] = numeric_min(
                row["a_days_since_last_game"],
                row["b_days_since_last_game"],
            )
            row["max_days_since_last_game"] = numeric_max(
                row["a_days_since_last_game"],
                row["b_days_since_last_game"],
            )
        else:
            row["min_days_since_last_game"] = np.nan
            row["max_days_since_last_game"] = np.nan

        row["either_player_debut"] = bool_value(row["a_is_debut"]) or bool_value(row["b_is_debut"])
        row["both_players_have_history"] = bool_value(row["a_has_previous_history"]) and bool_value(
            row["b_has_previous_history"]
        )
        # Debut matches are handled as "No previous history" rather than
        # inactive/returning matches, even if the opponent has a long gap.
        if row["either_player_debut"]:
            row["either_player_inactive_365d"] = False
            row["either_player_inactive_730d"] = False
        else:
            row["either_player_inactive_365d"] = bool(
                (
                    bool_value(row["a_has_previous_history"])
                    and pd.notna(row["a_days_since_last_game"])
                    and row["a_days_since_last_game"] >= 365
                )
                or (
                    bool_value(row["b_has_previous_history"])
                    and pd.notna(row["b_days_since_last_game"])
                    and row["b_days_since_last_game"] >= 365
                )
            )
            row["either_player_inactive_730d"] = bool(
                (
                    bool_value(row["a_has_previous_history"])
                    and pd.notna(row["a_days_since_last_game"])
                    and row["a_days_since_last_game"] >= 730
                )
                or (
                    bool_value(row["b_has_previous_history"])
                    and pd.notna(row["b_days_since_last_game"])
                    and row["b_days_since_last_game"] >= 730
                )
            )
        row["both_players_active_last_365d"] = bool(
            pd.notna(row["a_games_last_365_days"])
            and pd.notna(row["b_games_last_365_days"])
            and row["a_games_last_365_days"] > 0
            and row["b_games_last_365_days"] > 0
        )
        row["either_player_low_recent_activity"] = bool(
            pd.notna(row["min_games_last_365_days"]) and row["min_games_last_365_days"] <= 5
        )
        rows.append(row)

    return pd.DataFrame(rows)


def identify_fixed_2025_evaluation_set(
    match_df: pd.DataFrame,
    validation_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, Path | None]:
    """Load the fair-comparison prediction set used as the 2025 reference."""

    pred_path = find_file(
        "meeting5_fair_elo_vs_glicko_predictions_2025.csv",
        "outputs/meeting5_fair_elo_vs_glicko/meeting5_fair_elo_vs_glicko_predictions_2025.csv",
    )
    if pred_path is None:
        add_validation_check(
            validation_rows,
            "prediction_file_alignment",
            False,
            "missing",
            "meeting5 fair comparison predictions",
            severity="warning",
            detail="The feature rows were still built from year==2025 in the canonical dataset.",
        )
        return None, None

    pred = pd.read_csv(pred_path, low_memory=False)
    required = {"model", "game_id", "fcode", "winner", "loser"}
    missing = required - set(pred.columns)
    if missing:
        add_validation_check(
            validation_rows,
            "prediction_file_has_stable_match_id",
            False,
            ", ".join(sorted(missing)),
            "model/game_id/fcode/winner/loser columns present",
        )
        return pred, pred_path

    reference_model = "Validation_best_Elo"
    if reference_model not in set(pred["model"].unique()):
        reference_model = str(pred["model"].iloc[0])
    pred_ref = pred.loc[pred["model"] == reference_model].copy()
    pred_ref["game_id"] = pred_ref["game_id"].astype(int)
    pred_ref["fcode"] = pred_ref["fcode"].astype(int)
    pred_ref["winner"] = pred_ref["winner"].astype(int)
    pred_ref["loser"] = pred_ref["loser"].astype(int)

    feature_ids = match_df["match_id"].astype(int).tolist()
    pred_ids = pred_ref["game_id"].astype(int).tolist()
    feature_set = set(feature_ids)
    pred_set = set(pred_ids)
    missing_ids = sorted(pred_set - feature_set)
    extra_ids = sorted(feature_set - pred_set)

    add_validation_check(
        validation_rows,
        "prediction_file_has_stable_match_id",
        pred_ref["game_id"].is_unique and pred_ref["game_id"].equals(pred_ref["fcode"]),
        f"unique={pred_ref['game_id'].is_unique}; game_id_equals_fcode={pred_ref['game_id'].equals(pred_ref['fcode'])}",
        "unique game_id and game_id==fcode",
        detail=f"Reference model: {reference_model}",
    )
    add_validation_check(
        validation_rows,
        "prediction_evaluation_match_count",
        len(pred_ref) == EXPECTED_2025_GAMES,
        len(pred_ref),
        EXPECTED_2025_GAMES,
        detail=f"Reference model: {reference_model}",
    )
    add_validation_check(
        validation_rows,
        "feature_ids_match_prediction_ids",
        not missing_ids and not extra_ids,
        f"missing={len(missing_ids)}; extra={len(extra_ids)}",
        "missing=0; extra=0",
    )
    add_validation_check(
        validation_rows,
        "feature_order_matches_prediction_order",
        feature_ids == pred_ids,
        bool(feature_ids == pred_ids),
        True,
    )

    merged = match_df[["match_id", "winner_id", "loser_id"]].merge(
        pred_ref[["game_id", "winner", "loser"]],
        left_on="match_id",
        right_on="game_id",
        how="inner",
    )
    same_players = (
        (merged["winner_id"].astype(int) == merged["winner"].astype(int))
        & (merged["loser_id"].astype(int) == merged["loser"].astype(int))
    )
    add_validation_check(
        validation_rows,
        "prediction_winner_loser_alignment",
        bool(same_players.all()) and len(merged) == len(match_df),
        f"aligned={int(same_players.sum())}; compared={len(merged)}",
        len(match_df),
    )
    return pred_ref, pred_path


def compare_values(expected: Any, calculated: Any) -> tuple[float, bool]:
    """Return numeric difference and pass/fail, treating NaN values as equal."""

    if pd.isna(expected) and pd.isna(calculated):
        return 0.0, True
    if pd.isna(expected) or pd.isna(calculated):
        return np.nan, False
    difference = float(calculated) - float(expected)
    return difference, abs(difference) < 1e-9


def recompute_spot_features(matches: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    """Recompute selected pre-match features by direct dataframe filtering."""

    player_id = int(row["player_id"])
    current_sequence = int(row["match_sequence"])
    current_year = int(row["year"])
    current_date = pd.to_datetime(row["match_date"], errors="coerce")
    previous = matches.loc[
        (matches["match_sequence"] < current_sequence)
        & ((matches["winner"] == player_id) | (matches["loser"] == player_id))
    ].copy()

    expected: dict[str, Any] = {
        "total_games_before": int(len(previous)),
        "games_previous_calendar_year": int((previous["year"] == current_year - 1).sum()),
    }
    has_missing_previous_dates = bool(previous["event_order_date"].isna().any())
    if pd.isna(current_date) or has_missing_previous_dates:
        expected["games_last_90_days"] = np.nan
        expected["games_last_365_days"] = np.nan
        expected["days_since_last_game"] = np.nan
    else:
        previous_dates = pd.to_datetime(previous["event_order_date"], errors="coerce")
        expected["games_last_90_days"] = int(
            ((previous_dates >= current_date - pd.Timedelta(days=90)) & (previous_dates <= current_date)).sum()
        )
        expected["games_last_365_days"] = int(
            ((previous_dates >= current_date - pd.Timedelta(days=365)) & (previous_dates <= current_date)).sum()
        )
        if len(previous_dates) == 0:
            expected["days_since_last_game"] = np.nan
        else:
            expected["days_since_last_game"] = int((current_date - previous_dates.max()).days)
    return expected


def run_spot_checks(matches: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Run fixed-seed leakage spot checks on sampled player-match rows."""

    sample_size = min(20, len(long_df))
    sample = long_df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    variables = [
        "total_games_before",
        "games_last_90_days",
        "games_last_365_days",
        "games_previous_calendar_year",
        "days_since_last_game",
    ]
    for sample_index, feature_row in sample.iterrows():
        expected = recompute_spot_features(matches, feature_row)
        for variable in variables:
            expected_value = expected[variable]
            calculated_value = feature_row[variable]
            difference, passed = compare_values(expected_value, calculated_value)
            rows.append(
                {
                    "sample_index": sample_index,
                    "match_id": int(feature_row["match_id"]),
                    "match_sequence": int(feature_row["match_sequence"]),
                    "player_id": int(feature_row["player_id"]),
                    "variable": variable,
                    "expected_value": expected_value,
                    "calculated_value": calculated_value,
                    "difference": difference,
                    "passed": bool(passed),
                }
            )
    return pd.DataFrame(rows)


def run_validation_checks(
    matches: pd.DataFrame,
    long_df: pd.DataFrame,
    match_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    validation_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Run feature integrity checks requested for meeting 6 step 1."""

    add_validation_check(
        validation_rows,
        "2025_evaluation_matches",
        len(match_df) == EXPECTED_2025_GAMES,
        len(match_df),
        EXPECTED_2025_GAMES,
    )
    add_validation_check(
        validation_rows,
        "match_level_rows",
        len(match_df) == EXPECTED_2025_GAMES,
        len(match_df),
        EXPECTED_2025_GAMES,
    )
    add_validation_check(
        validation_rows,
        "long_format_rows",
        len(long_df) == EXPECTED_LONG_ROWS,
        len(long_df),
        EXPECTED_LONG_ROWS,
    )
    add_validation_check(
        validation_rows,
        "match_level_match_id_unique",
        not match_df["match_id"].duplicated().any(),
        int(match_df["match_id"].duplicated().sum()),
        0,
    )
    rows_per_match = long_df.groupby("match_id").size()
    add_validation_check(
        validation_rows,
        "long_format_two_rows_per_match",
        bool((rows_per_match == 2).all()) and len(rows_per_match) == len(match_df),
        f"bad_matches={int((rows_per_match != 2).sum())}; represented_matches={len(rows_per_match)}",
        f"bad_matches=0; represented_matches={len(match_df)}",
    )
    distinct_players = long_df.groupby("match_id")["player_id"].nunique()
    add_validation_check(
        validation_rows,
        "two_distinct_players_per_match",
        bool((distinct_players == 2).all()),
        int((distinct_players != 2).sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "no_duplicate_match_player_rows",
        not long_df[["match_id", "player_id"]].duplicated().any(),
        int(long_df[["match_id", "player_id"]].duplicated().sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "match_sequence_unique_2025",
        not match_df["match_sequence"].duplicated().any(),
        int(match_df["match_sequence"].duplicated().sum()),
        0,
    )

    numeric_nonnegative = [
        "total_games_before",
        "games_last_90_days",
        "games_last_365_days",
        "games_previous_calendar_year",
    ]
    for col in numeric_nonnegative:
        values = pd.to_numeric(long_df[col], errors="coerce")
        add_validation_check(
            validation_rows,
            f"{col}_nonnegative",
            bool((values.dropna() >= 0).all()),
            int((values.dropna() < 0).sum()),
            0,
        )
    add_validation_check(
        validation_rows,
        "games_365_ge_games_90",
        bool(
            (
                long_df.loc[
                    long_df["games_last_365_days"].notna() & long_df["games_last_90_days"].notna(),
                    "games_last_365_days",
                ]
                >= long_df.loc[
                    long_df["games_last_365_days"].notna() & long_df["games_last_90_days"].notna(),
                    "games_last_90_days",
                ]
            ).all()
        ),
        "checked",
        "games_last_365_days >= games_last_90_days",
    )
    add_validation_check(
        validation_rows,
        "debut_total_games_zero",
        bool((long_df.loc[long_df["is_debut"], "total_games_before"] == 0).all()),
        int((long_df.loc[long_df["is_debut"], "total_games_before"] != 0).sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "debut_days_since_last_missing",
        bool(long_df.loc[long_df["is_debut"], "days_since_last_game"].isna().all()),
        int(long_df.loc[long_df["is_debut"], "days_since_last_game"].notna().sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "has_history_total_games_positive",
        bool((long_df.loc[long_df["has_previous_history"], "total_games_before"] > 0).all()),
        int((long_df.loc[long_df["has_previous_history"], "total_games_before"] <= 0).sum()),
        0,
    )
    non_debut_valid_dates = long_df.loc[
        long_df["has_previous_history"] & long_df["date_features_available"] & long_df["days_since_last_game"].notna()
    ]
    add_validation_check(
        validation_rows,
        "history_days_since_last_nonnegative",
        bool((non_debut_valid_dates["days_since_last_game"] >= 0).all()),
        int((non_debut_valid_dates["days_since_last_game"] < 0).sum()),
        0,
    )
    add_validation_check(
        validation_rows,
        "inactive_365_requires_history",
        bool(
            (
                ~match_df["either_player_inactive_365d"]
                | (
                    (match_df["a_has_previous_history"] & (match_df["a_days_since_last_game"] >= 365))
                    | (match_df["b_has_previous_history"] & (match_df["b_days_since_last_game"] >= 365))
                )
            ).all()
        ),
        "checked",
        "inactive flag only from players with history and gap >= 365",
    )
    add_validation_check(
        validation_rows,
        "inactive_730_requires_history",
        bool(
            (
                ~match_df["either_player_inactive_730d"]
                | (
                    (match_df["a_has_previous_history"] & (match_df["a_days_since_last_game"] >= 730))
                    | (match_df["b_has_previous_history"] & (match_df["b_days_since_last_game"] >= 730))
                )
            ).all()
        ),
        "checked",
        "inactive flag only from players with history and gap >= 730",
    )
    add_validation_check(
        validation_rows,
        "inactive_flags_exclude_debut_matches",
        bool(
            (
                ~match_df["either_player_debut"]
                | (~match_df["either_player_inactive_365d"] & ~match_df["either_player_inactive_730d"])
            ).all()
        ),
        "checked",
        "debut matches are grouped as No previous history, not inactive/returning",
    )
    add_validation_check(
        validation_rows,
        "match_level_inactivity_gaps_nonnegative",
        bool(
            (match_df["min_days_since_last_game"].dropna() >= 0).all()
            and (match_df["max_days_since_last_game"].dropna() >= 0).all()
        ),
        "checked",
        "min/max days_since_last_game >= 0 when defined",
    )
    for name in ["total_games_before", "games_last_90_days", "games_last_365_days", "previous_year_games"]:
        add_validation_check(
            validation_rows,
            f"min_le_max_{name}",
            bool((match_df[f"min_{name}"].dropna() <= match_df[f"max_{name}"].dropna()).all()),
            "checked",
            f"min_{name} <= max_{name}",
        )
        add_validation_check(
            validation_rows,
            f"abs_diff_nonnegative_{name}",
            bool((match_df[f"abs_diff_{name}"].dropna() >= 0).all()),
            "checked",
            f"abs_diff_{name} >= 0",
        )
    add_validation_check(
        validation_rows,
        "outcome_a_binary",
        set(match_df["outcome_a"].dropna().unique()).issubset({0, 1}),
        ", ".join(map(str, sorted(match_df["outcome_a"].dropna().unique()))),
        "0/1",
    )
    add_validation_check(
        validation_rows,
        "player_a_is_winner_consistent",
        bool((match_df["player_a_is_winner"] == (match_df["player_a_id"] == match_df["winner_id"])).all()),
        "checked",
        "player_a_is_winner == (player_a_id == winner_id)",
    )
    add_validation_check(
        validation_rows,
        "outcome_a_consistent",
        bool((match_df["outcome_a"] == match_df["player_a_is_winner"].astype(int)).all()),
        "checked",
        "outcome_a == int(player_a_is_winner)",
    )
    for col in ["date_quality", "a_date_quality", "b_date_quality"]:
        source = long_df if col == "date_quality" else match_df
        add_validation_check(
            validation_rows,
            f"{col}_allowed_values",
            set(source[col].dropna().unique()).issubset({"exact", "project_fallback", "missing"}),
            ", ".join(sorted(source[col].dropna().unique())),
            "exact/project_fallback/missing",
        )
    add_validation_check(
        validation_rows,
        "spot_checks_no_leakage",
        bool(spot_df["passed"].all()),
        f"passed={int(spot_df['passed'].sum())}; total={len(spot_df)}",
        f"passed={len(spot_df)}; total={len(spot_df)}",
        detail="Spot checks recompute features using match_sequence < current_match_sequence.",
    )

    return pd.DataFrame(validation_rows)


def build_feature_summary(long_df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise feature distributions for long and match-level outputs."""

    rows: list[dict[str, Any]] = []
    for variable in SUMMARY_VARIABLES:
        if variable in long_df.columns:
            source_name = "long"
            series = pd.to_numeric(long_df[variable], errors="coerce")
        elif variable in match_df.columns:
            source_name = "match"
            series = pd.to_numeric(match_df[variable], errors="coerce")
        else:
            continue
        rows.append(
            {
                "source": source_name,
                "variable": variable,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "std": float(series.std()) if series.notna().sum() > 1 else np.nan,
                "min": float(series.min()) if series.notna().any() else np.nan,
                "p10": float(series.quantile(0.10)) if series.notna().any() else np.nan,
                "p25": float(series.quantile(0.25)) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "p75": float(series.quantile(0.75)) if series.notna().any() else np.nan,
                "p90": float(series.quantile(0.90)) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
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


def bin_max_days_since_last_game(row: pd.Series) -> str:
    if bool(row["either_player_debut"]):
        return "No previous history"
    value = row["max_days_since_last_game"]
    if pd.isna(value):
        return "Missing date information"
    value = float(value)
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


def add_group_counts(rows: list[dict[str, Any]], group_type: str, labels: pd.Series, total: int) -> None:
    counts = labels.value_counts(dropna=False)
    for group, games in counts.items():
        rows.append(
            {
                "group_type": group_type,
                "group": str(group),
                "games": int(games),
                "percentage": float(games / total) if total else np.nan,
            }
        )


def build_group_count_preview(match_df: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive group counts without fitting performance subgroups."""

    total = len(match_df)
    rows: list[dict[str, Any]] = []
    add_group_counts(
        rows,
        "min_total_games_before",
        match_df["min_total_games_before"].map(bin_min_total_games),
        total,
    )
    add_group_counts(
        rows,
        "min_games_last_365_days",
        match_df["min_games_last_365_days"].map(bin_min_games_last_365),
        total,
    )
    add_group_counts(
        rows,
        "max_days_since_last_game",
        match_df.apply(bin_max_days_since_last_game, axis=1),
        total,
    )

    flag_specs = [
        ("either_player_debut", "either player debut"),
        ("either_player_inactive_365d", "either player inactive >= 365 days"),
        ("either_player_inactive_730d", "either player inactive >= 730 days"),
        ("both_players_active_last_365d", "both players active in last 365 days"),
        ("either_player_low_recent_activity", "either player has <= 5 games in last 365 days"),
    ]
    for col, label in flag_specs:
        games = int(match_df[col].sum())
        rows.append(
            {
                "group_type": "flag",
                "group": label,
                "games": games,
                "percentage": float(games / total) if total else np.nan,
            }
        )
    low_total_games = int((match_df["min_total_games_before"] <= 5).sum())
    rows.append(
        {
            "group_type": "flag",
            "group": "either player has <= 5 total previous games",
            "games": low_total_games,
            "percentage": float(low_total_games / total) if total else np.nan,
        }
    )
    return pd.DataFrame(rows)


def format_int(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"




def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, Any]] = []

    matches, dataset_path = load_canonical_matches(validation_rows)
    validate_chronological_order(matches, validation_rows)
    long_df = build_long_format(matches)
    match_df = build_match_level_features(matches, long_df)
    _, prediction_path = identify_fixed_2025_evaluation_set(match_df, validation_rows)
    spot_df = run_spot_checks(matches, long_df)
    validation_checks = run_validation_checks(matches, long_df, match_df, spot_df, validation_rows)
    feature_summary = build_feature_summary(long_df, match_df)
    group_counts = build_group_count_preview(match_df)

    long_df.to_csv(LONG_FEATURES_PATH, index=False, encoding="utf-8-sig")
    match_df.to_csv(MATCH_FEATURES_PATH, index=False, encoding="utf-8-sig")
    validation_checks.to_csv(VALIDATION_CHECKS_PATH, index=False, encoding="utf-8-sig")
    spot_df.to_csv(SPOT_CHECKS_PATH, index=False, encoding="utf-8-sig")
    feature_summary.to_csv(FEATURE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    group_counts.to_csv(GROUP_COUNTS_PATH, index=False, encoding="utf-8-sig")

    print("Meeting 6 step 1 pre-match features complete.")
    print(f"Full-history matches scanned: {len(matches):,}")
    print(f"2025 match-level rows: {len(match_df):,}")
    print(f"2025 long-format rows: {len(long_df):,}")
    print(f"Validation checks passed: {int(validation_checks['passed'].sum())} / {len(validation_checks)}")
    print(f"Leakage spot checks passed: {int(spot_df['passed'].sum())} / {len(spot_df)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
