# Meeting 8 Technical Diagnostic: Burn-in, Recorded Entry, and Rating-Scale Drift

## Scope

This is a targeted diagnostic, not a new model search. It reuses the frozen low-inflation Glicko configuration and adds the two checks requested after Meeting 7: a defensible recorded-entry definition and an audit of the evolving rating scale.

## Reused model and data

- Full-history data: `outputs\elo_optimization\matches_1985_2025_checked.csv`
- Years: 1985-2025
- Low-inflation variant: `low_inflation`
- Inactivity unit: month
- C value: 22.509257354846
- Fixed new-player state: rating 1500, RD 350
- Probability convention: canonical Player A probability, complemented for Player B, matching the frozen Meeting 7 reporting convention.

## Operational definitions

- **System-start / left-censored player:** first observed in 1985. The data cannot establish that this was the player's true career debut.
- **Within-burn-in recorded entry:** first observed during 1986-1989. These rows are excluded from the primary post-burn-in definition.
- **Post-burn-in recorded entry:** first observed from 1990 onward after a five-calendar-year burn-in.
- **2025 test-year recorded entry:** first observed in 2025. This is a subset of post-burn-in recorded entries and is the source of the frozen first_1 test result.
- The phrase *recorded entry* is deliberate: the available data do not prove true career debut status.

## Cohort sizes

- System-start players: 314
- Within five-year burn-in entries: 456
- Post-burn-in entries before the test year: 4,297
- 2025 test-year entries: 76
- Total classified players: 5,143

## 2025 first-appearance regression

- First appearances: 76
- Unique matches: 74
- Mean predicted win probability: 0.743448
- Empirical win rate: 0.407895
- Brier score: 0.322316
- Mean opponent rating: 1180.755
- Maximum absolute probability difference from Step 34: 5.285e-13

All 2025 first appearances occur after the five-year burn-in. Therefore, the 2025 first_1 weakness is not an artefact of treating the 1985 model-start population as genuinely new. It remains a limitation for players newly entering the recorded system in the held-out test year.

## Rating-scale audit

- Established-player scale is undefined in 1985, because every observed player is in the system-start cohort.
- First defined established-active median (1986): 1499.614
- Established active median rating in 2025: 1334.544
- Fixed anchor minus established-active median in 2025: 165.456
- Mean rating across all 5,143 known players in 2025: 1257.881
- Largest annual mean anchor-minus-debut-opponent gap after burn-in: 324.334 in 2024
- Cumulative net two-player Glicko update change by 2025: -1245217.700

The absolute rating level has no standalone substantive meaning because predictions depend on relative ratings. These scale summaries are therefore diagnostic: they show whether the fixed 1500 entry anchor stays aligned with the contemporaneous established-player scale. The direct quantity for the new-player mechanism is the focal-minus-opponent prematch rating gap.

## What should go into the dissertation

1. State the four operational definitions above in the methodology or limitations section.
2. Report that the held-out 2025 first_1 sample consists entirely of post-burn-in recorded entrants.
3. Use the annual scale figure only to explain the mechanism; do not interpret 1500 as an absolute skill level.
4. Present the adaptive-K work separately as a short negative result. No additional adaptive-K experiment is required here.

## Validation

- Checks passed: 19/19
- Failed checks: 0
