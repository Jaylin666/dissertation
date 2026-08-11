# Step 39 Orientation Sensitivity Wording for Meeting 7

## 1. Short robustness statement

Because the two direct Glicko expected scores are not exactly complementary when player RDs differ, I tested current, reversed and midpoint outcome-independent orientation conventions. The numerical magnitude changes slightly, but the overall Glicko Brier advantage, the first-appearance Elo advantage and the first-appearance Glicko over-prediction result remain unchanged.

## 2. Overall performance wording

The overall Brier advantage of low-inflation Glicko is statistically clear under all three conventions. Log-loss point estimates also favour Glicko under all conventions, although the reversed-convention confidence interval crosses zero.

## 3. Early-game wording

The first-appearance result is highly robust: Validation-best Elo performs better and Glicko substantially over-predicts new players under all three conventions. For the first_5 to first_20 windows, the performance gap becomes smaller, but the strength of statistical evidence varies across conventions and analysis units.

## 4. Reliability wording

The predictive disadvantage of low-inflation Glicko relative to Validation-best Elo is concentrated in the earliest recorded appearances. By approximately 10 to 20 appearances, the Brier-score difference is small and is generally not clearly distinguishable from zero under the main convention.

## 5. Initialisation wording

New players are anchored above many established opponents on the evolved Glicko rating scale. A common shift of the entire rating scale cannot resolve this relative new-player initialisation mismatch.

## 6. Adaptive-K wording

Step 38 confirmed that the existing previous-year-activity adaptive-K Elo already uses player-specific K values. It did not introduce a new rule that explicitly assigns a larger K to the lower-rated player.
