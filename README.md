# Croquet Rating Systems Dissertation

This repository contains the supported scientific implementation and compact
direct evidence for the Association Croquet Elo-Glicko MSc dissertation. The
model equations, parameters, temporal splits, probability convention,
bootstrap design, and reported empirical evidence are frozen.

## Requirements

Python 3.10 or later is required. Install the dependencies in an isolated
environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Repository structure

- `code/data/` downloads and constructs the checked 1985-2025 game history.
- `code/models/` contains the shared Elo and Glicko-1 equations.
- `code/pipelines/` contains parameter validation, rating runs, prematch input
  construction, comparison, calibration, and bootstrap workflows.
- `code/analysis/` contains orientation, early-game, initial-rating, entry,
  and rating-drift analyses.
- `tests/` contains unit, synthetic, and compact-output regression tests.
- `outputs/dissertation_evidence/chapter4/` and `chapter5/` contain the tracked
  direct evidence used by the dissertation.

Raw data, full histories, row-level predictions, caches, and reproduction
artifacts are deliberately untracked. Historical numbered scripts and meeting
outputs are available through Git history but are not supported dependencies.

## Supported workflows

Commands are previews unless `--full-run` is supplied. Full runs accept
`--output-root` and default to the ignored `outputs/reproduction/` directory.
They never overwrite tracked dissertation evidence.

| Thesis component | Current command | Direct evidence directory |
| --- | --- | --- |
| Checked historical data | `python -m code.cli build-data` | Input to Chapters 4-5 |
| Elo validation grid | `python -m code.cli select-elo` | `outputs/dissertation_evidence/chapter4/` |
| Fixed Elo histories | `python -m code.cli run-elo` | `outputs/dissertation_evidence/chapter4/` |
| Glicko inactivity variants | `python -m code.cli run-glicko` | `outputs/dissertation_evidence/chapter4/` |
| Glicko rating periods | `python -m code.cli glicko-periods` | `outputs/dissertation_evidence/chapter4/` |
| Adaptive-K candidates | `python -m code.cli run-adaptive-elo` | `outputs/dissertation_evidence/chapter4/` |
| Prematch comparison input | `python -m code.cli build-comparison-inputs` | Input to Chapters 4-5 |
| Comparison, calibration, bootstrap | `python -m code.cli compare-models` | `outputs/dissertation_evidence/chapter4/` |
| Early-game analysis | `python -m code.cli early-game` | `outputs/dissertation_evidence/chapter5/` |
| Common initial-rating sensitivity | `python -m code.cli initial-rating-sensitivity` | `outputs/dissertation_evidence/chapter5/` |
| Recorded-entry diagnostics | `python -m code.cli entry-diagnostics` | `outputs/dissertation_evidence/chapter5/` |

For an individual full run, provide its required checked-history or unified
comparison input where necessary:

```bash
python -m code.cli select-elo --full-run \
  --matches-path outputs/reproduction/elo_optimization/matches_1985_2025_checked.csv \
  --output-root outputs/reproduction
```

## Full reproduction

Place the original annual Association Croquet data in `data_raw/` using the
layout expected by `code/data/download.py` and `code/data/build_matches.py`.
Then run the supported sequence:

```bash
python -m code.cli reproduce-dissertation \
  --full-run \
  --output-root outputs/reproduction
```

The command builds the checked history, selects the frozen Elo specification,
runs the fixed Elo and Glicko variants, runs rating-period and Adaptive-K
sensitivity workflows, constructs leakage-free prematch inputs, and reproduces
the Chapter 4-5 analyses and figures. It finishes by comparing 19 compact CSV
outputs with the tracked evidence. Large intermediate files remain in the
ignored output root and are not committed.

## Lightweight validation

These checks inspect the tracked evidence and code without rebuilding the full
historical analysis:

```bash
python -m code.cli validate
python -m unittest discover -s tests -v
python -m compileall -f code
git diff --check
```

## Repository states

The current `main` branch contains the complete supported implementation. The
annotated tag `dissertation-final-deliverable-2026` remains fixed to commit
`6da10b616596c15e7122799d24165f609d1dc87a`, the earlier evidence-freeze
snapshot. Completing the active implementation does not move or replace that
tag.
