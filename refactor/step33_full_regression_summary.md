# Complete Step 33 Refactor Regression

## Execution

- Command: `python -m code.cli compare-models --full-run --output-root outputs/refactor_validation`
- Runtime: 89.900688 seconds
- Reference root: `outputs/meeting6`
- Regenerated root: `outputs/refactor_validation/meeting6`
- Bootstrap repetitions: 2000
- Random seed: 20260715
- Probability convention: direct player-A probability, `expected_score(rating_A, rating_B, RD_B)`.

## Coverage

- Outputs expected: 36
- Outputs regenerated: 36
- Outputs compared: 36
- Exact passes: 25
- Numeric-tolerance passes: 0
- Normalised-formatting or figure passes: 11
- Historical-only outputs: 0
- Failed or missing outputs: 0

## Numerical Equivalence

- Maximum absolute numerical difference: 0
- Maximum relative numerical difference: 0
- All required key sets match: True
- All bootstrap outputs match: True
- All explicit headline checks pass: True
- All Markdown headline checks pass: True
- All figures pass automated and manual checks: True

## Conclusion

The refactored comparison pipeline is scientifically equivalent to the original Step 33 workflow.
