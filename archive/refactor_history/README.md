# Refactor validation

This directory contains the code audit, migration plan, and numerical
regression evidence for `refactor/clean-code-layout`.

## Complete Step 33 regression

The full regression utility can regenerate every output supported by the
active Step 33 comparison pipeline into the ignored validation directory:

```bash
python refactor/validate_step33_refactor.py --run --manual-visual-check pending
```

After visually inspecting all ten regenerated figures, record the review and
repeat the comparison without rerunning the pipeline:

```bash
python refactor/validate_step33_refactor.py --manual-visual-check pass
```

Custom roots are supported:

```bash
python refactor/validate_step33_refactor.py \
  --reference-root outputs/meeting6 \
  --new-root outputs/refactor_validation/meeting6
```

The utility refuses to run the expensive pipeline outside
`outputs/refactor_validation`, never overwrites the reference outputs, performs
full keyed comparisons of large row-level files, validates fixed-seed
bootstrap outputs, extracts Markdown headline values, checks figures, writes
the complete reports, and exits non-zero on any substantive mismatch.

