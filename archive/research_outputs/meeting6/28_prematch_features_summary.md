# Meeting 6 Step 1 Prematch Player Features

## Purpose

This step builds leakage-safe pre-match player-history features for the fixed 2025 evaluation set. It does not rerun Elo or Glicko and does not evaluate subgroup model performance.

## Inputs

- Canonical match dataset: `outputs\elo_optimization\matches_1985_2025_checked.csv`
- Fair-comparison prediction file: `outputs\meeting5_fair_elo_vs_glicko\meeting5_fair_elo_vs_glicko_predictions_2025.csv`
- Ordering rule: year, date availability, event order date, event ID, then fcode, matching the meeting 5 scripts.
- Same-day earlier matches are included through `match_sequence`; the current match is recorded before its players' states are updated.

## Output Size

- Full-history matches scanned: 456,382
- Fixed 2025 evaluation matches: 11,379
- Long-format player-match rows: 22,758

## Date Quality In 2025 Features

- exact: 11,379 matches
- project_fallback: 0 matches
- missing: 0 matches

## Main Group Count Preview

- either player debut: 74 games (0.7%)
- either player inactive >= 365 days: 198 games (1.7%)
- either player inactive >= 730 days: 83 games (0.7%)
- both players active in last 365 days: 11,036 games (97.0%)
- either player has <= 5 games in last 365 days: 1,635 games (14.4%)
- either player has <= 5 total previous games: 465 games (4.1%)

## Feature Distribution Highlights

- total_games_before: count=22,758, missing=0, median=344.000, p90=1788.000
- games_last_365_days: count=22,685, missing=73, median=33.000, p90=91.000
- days_since_last_game: count=22,609, missing=149, median=0.000, p90=28.000
- min_total_games_before: count=11,379, missing=0, median=168.000, p90=874.000
- min_games_last_365_days: count=11,306, missing=73, median=22.000, p90=58.000
- max_days_since_last_game: count=11,232, missing=147, median=0.000, p90=56.000

## Validation

- Validation checks passed: 46 / 46
- Leakage spot checks passed: 100 / 100
- Spot checks recomputed features using only rows with `match_sequence < current_match_sequence`.
- Debut matches are kept separate from inactive/returning flags as `No previous history`.
- `days_since_last_game` and `career_days_before` are missing for debut rows or rows with unreliable date history.
- Issues: none

## Files Written

- `outputs\meeting6\28_prematch_player_features_2025_long.csv`
- `outputs\meeting6\28_prematch_match_features_2025.csv`
- `outputs\meeting6\28_feature_validation_checks.csv`
- `outputs\meeting6\28_feature_spot_checks.csv`
- `outputs\meeting6\28_feature_summary.csv`
- `outputs\meeting6\28_feature_group_counts.csv`

## Next Step

The next script can merge `28_prematch_match_features_2025.csv` with model predictions and compare Glicko vs Elo performance across low-experience, low-activity, and returning-player groups.