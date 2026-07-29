# Croquet Rating Systems Dissertation

This repository contains the technical analysis for an MSc dissertation
comparing Elo and Glicko-type rating systems for Association Croquet match
data.

## Project overview

The project studies probabilistic match prediction and rating-system behaviour,
including:

- baseline and validation-selected Elo models;
- classic Glicko-1;
- inactivity-based rating-deviation inflation;
- adaptive-K Elo;
- a fixed 2025 test evaluation;
- early-game and first-recorded-appearance diagnostics;
- burn-in definitions and rating-scale alignment.

Scripts in `code/` are numbered in the order in which the analysis was
developed. Later scripts reuse and audit earlier validated components rather
than silently replacing them.

## Current main findings

- Low-inflation Glicko performs slightly better overall than the strongest Elo
  baseline on the fixed 2025 test set.
- First recorded appearance is the clearest limitation of the selected Glicko
  specification.
- The 2025 first-appearance sample consists entirely of players entering the
  recorded system after a long burn-in, rather than players created at the
  1985 model start.
- The fixed entry rating of 1500 is high relative to both the contemporaneous
  prematch established-player scale and the ratings of actual first opponents.
- A first recorded appearance does not necessarily represent a true career
  debut.
- Historical entry-cohort results are descriptive mechanism evidence; 2025 is
  the formal held-out test.

These findings support a relative new-player initialisation mismatch
interpretation. They do not establish that rating-scale movement alone causes
the first-appearance prediction error.

## Repository structure

- `code/`: numbered data-processing, Elo, Glicko, evaluation, robustness, and
  diagnostic scripts.
- `outputs/`: selected compact result tables, validation checks, summaries,
  manifests, and figures.
- `outputs/meeting6/`: orientation correction and supporting diagnostics.
- `outputs/meeting7/`: early-game, first-appearance, initialisation, adaptive-K,
  and probability-orientation results.
- `outputs/meeting8_technical/`: burn-in, rating-scale, strict prematch, and
  cross-file audits from Steps 41 and 42.

## Reproducibility

Install the external Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The numbered scripts document the analysis sequence and use project-relative
paths. However, this repository is not fully self-contained: raw match data
and several large generated intermediate tables are intentionally not tracked.
To reproduce the full workflow, obtain the annual source data separately,
place it under `data_raw/`, and run the required numbered construction scripts
before downstream analyses.

The repository retains compact summaries and validation artifacts rather than
large per-match predictions, player-appearance tables, complete rating
histories, or repeated parameter-search traces. In particular, raw data,
processed full-history match tables, multi-gigabyte burn-in histories, and
other reproducible row-level outputs are excluded from Git.

## Data

The analysis uses historical Association Croquet match records organised as
annual match, event, and player data files. Raw data are not published in this
repository. Users must obtain permission and access to the underlying source
data independently.

## Status

The technical analysis is provisionally frozen. The main project focus has
moved to dissertation writing, with additional computation limited to
targeted checks required by the written argument.
