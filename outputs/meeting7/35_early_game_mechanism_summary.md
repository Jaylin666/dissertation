# Step 35 Early-Game Mechanism Analysis

## 1. Research question
This analysis investigates why Glicko low inflation performs poorly for players' earliest recorded appearances and why the gap relative to validation-best Elo shrinks as players gain recorded games.

## 2. Data and analysis units
The primary unit is the focal-player appearance from Step 34. The dataset contains 22,758 appearances from 11,379 matches. All probabilities are focal-player win probabilities already saved in Step 34.

## 3. Does Glicko over-predict debut players?
For first appearances, Glicko low fixed has mean predicted probability 0.743, empirical win rate 0.408, and prediction bias 0.336. The player-cluster bootstrap CI is [0.233, 0.435]. This supports the statement that Glicko over-predicts debut players in this evaluation set.
Validation-best Elo also has positive debut bias (0.131) but it is much smaller.

## 4. Bias over the first 20 appearances
Glicko first_5 bias is 0.143. The Elo-minus-Glicko Delta Brier changes from -0.112 at appearance 1 to 0.006 at appearance 20, indicating that the early gap narrows as recorded experience accumulates.

## 5. Prediction extremity
In first appearances, the extreme-prediction share (p < 0.10 or p > 0.90) is 0.105 for Glicko low fixed and 0.026 for validation-best Elo. This supports an overconfidence/extremity mechanism in the debut group.

## 6. Glicko RD mechanism
Mean focal Glicko RD for first appearances is 350.00. The first_20 correlation between focal RD and Brier loss is 0.104, so high uncertainty is associated with prediction error only to the extent shown in the RD association table.

## 7. Opponent strength
Opponent strength is measured using rating_opponent_Glicko_low. The strongest negative debut Delta Brier occurs in the third_quartile group, but the full quartile table should be used cautiously because the debut sample is small.

## 8. Glicko low versus Glicko C0
In first appearances, C0 minus low Delta Brier is 0.026. Positive values mean the low-inflation variant performs better than C0.

## 9. Cautious mechanism interpretation
The evidence points to a debut-specific mechanism: Glicko assigns high probabilities to some newly recorded players, producing larger early Brier/log-loss penalties. The gap then shrinks as player histories accumulate and Glicko/Elo predictions become more similar.

## 10. Limitations
These are first recorded appearances in the available dataset, not necessarily true career debuts. Some exact-appearance and probability-band groups are small. This analysis diagnoses mechanisms using existing model outputs only and does not retune any rating system.
