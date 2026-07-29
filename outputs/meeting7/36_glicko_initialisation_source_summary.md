# Step 36 Glicko Initialisation Source Diagnostic

## 1. Purpose
This diagnostic investigates why first recorded appearances have a high saved low-inflation Glicko predicted win probability. It uses stored Step 33/34 probabilities and pre-match states only.

## 2. Data and sample definition
The diagnostic sample contains 76 first_1 focal appearances from 74 unique matches and 76 unique focal players.

## 3. Verification of the 0.743 first-appearance probability
The reproduced mean Glicko probability is 0.743448, the empirical win rate is 0.407895, and the prediction bias is 0.335554.

## 4. Orientation and formula reconstruction
The maximum focal-orientation audit error is 4.999e-13. The maximum reconstruction error from the Step 33 formula is 4.498e-12.
The saved probability definition is the Step 33 canonical-A direct probability converted to focal orientation.

## 5. Debut focal rating and opponent rating comparison
First-appearance focal ratings have mean 1500.000; opponent ratings have mean 1180.755. The mean focal-minus-opponent rating difference is 319.245.

## 6. Role of focal RD in the implemented expected score formula
Focal debut RD is 350.000 on average and is constant at the initial value in this sample. Because most first_1 focal players are canonical player B, the saved focal probability uses RD_B in 72 rows as the focal player's RD and in 4 rows as the focal player's opponent RD.

## 7. Role of opponent RD
Mean focal-opponent RD is 139.246. The Pearson association between focal-opponent RD and Glicko Brier loss is 0.030840067090995855.

## 8. Extreme observations
The observed mean probability is 0.743448, while the 5 percent trimmed mean is 0.757357. This checks whether the mean is driven by only a few extreme probabilities.

## 9. One-debut versus both-debut matches
Exactly-one-debut matches contribute 72 appearances from 72 matches. Both-debut matches contribute 4 appearances from 2 matches.

## 10. Low-inflation Glicko versus Glicko C0 at debut
C0 minus low Brier at first appearance is 0.026269. Positive values mean the low-inflation variant has lower Brier loss than C0.

## 11. Counterfactual diagnostic results
Setting the focal rating equal to the opponent rating gives mean probability 0.500000. These are formula-only diagnostics and are not fitted models.

## 12. Most defensible explanation
The diagnostic conclusion code is `INITIAL_RATING_LEVEL_DOMINANT`. The evidence should be interpreted as diagnostic rather than causal model refitting.

## 13. What cannot yet be concluded
This step does not prove what the optimal initial rating or RD should be. It also does not establish true career debut status; these are first recorded appearances in the available data.

## 14. Initial-rating sensitivity experiment
Initial-rating sensitivity experiment recommended next: True. This recommendation is for a future diagnostic/sensitivity step only.
