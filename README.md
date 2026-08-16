# Croquet Rating Systems Dissertation

## Overview

This repository contains the technical implementation and controlled evidence
for an MSc dissertation comparing Elo and Glicko-1 rating systems for
Association Croquet. The empirical analysis and dissertation parameters are
frozen. The repository is organised to make the current implementation,
automated validation, Chapter 4 and Chapter 5 evidence, and scientific
provenance easy to distinguish.

The repository is not fully self-contained. Raw source data and large
row-level intermediates are intentionally excluded from Git and are required
for a complete reconstruction.

## Repository structure

- `code/`: supported modular implementation and command-line interface.
  - `models/` contains the canonical Elo and Glicko-1 equations.
  - `data/` validates and constructs the checked historical game table.
  - `pipelines/` contains the supported Elo, Glicko, and comparison workflows.
  - `analysis/` contains orientation, early-game, rating-drift, and
    recorded-entry diagnostics.
  - `config.py` records frozen dissertation parameters and expected values.
- `tests/`: equation, probability-orientation, recorded-entry, and frozen
  output regression tests.
- `outputs/dissertation_evidence/`: controlled compact evidence for the
  dissertation.
- `archive/`: scientifically relevant legacy scripts plus the compact source
  and validation artefacts needed for provenance and regression checks.
- `CODE_MAP.md`: map from the chronological research scripts to the current
  modules or archive.

`code/glicko_core.py` is a compatibility layer for historical imports. New
code should import the canonical implementation from `code/models/glicko.py`.

## Reproducing the analysis

Python 3.10 or later is required. Install the external dependencies in an
isolated environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the lightweight checks that do not rebuild the historical analysis:

```bash
python -m code.cli validate
python -m unittest discover -s tests -v
python -m compileall -f code
```

The first command checks tracked validation records and controlled evidence;
the unit tests cover the rating equations, orientation rules, recorded-entry
definitions, and frozen headline values. These checks do not rerun the rating
models or generate new dissertation results.

## Main workflows

Inspect the supported commands and their inputs before attempting a full run:

```bash
python -m code.cli --help
python -m code.cli build-data --help
python -m code.cli run-elo --help
python -m code.cli run-glicko --help
python -m code.cli compare-models --help
python -m code.cli early-game --help
python -m code.cli entry-diagnostics --help
```

Computational workflows run only when `--full-run` is supplied and should be
directed to an ignored local output root, for example:

```bash
python -m code.cli run-elo --full-run --output-root outputs/reproduction
```

The commands represent the supported data build, Elo, Glicko,
orientation-corrected comparison, early-game, and recorded-entry workflows.
They are not a one-command reproduction pipeline: each full run requires its
documented upstream source or intermediate files. The CLI prints the exact
paths and frozen parameter configuration before execution.

## Dissertation evidence

The controlled evidence packages are:

- `outputs/dissertation_evidence/chapter4/`: Elo selection, overall model
  comparison, calibration, Glicko sensitivity, and adaptive-K evidence.
- `outputs/dissertation_evidence/chapter5/`: early-game performance,
  first-recorded-appearance mechanisms, entry-scale alignment, and robustness
  evidence.

The primary Elo specification was selected using chronological validation on
2023-2024 data; 2025 outcomes were not used for that selection. The Glicko
inactivity settings and adaptive-K variants were instead explored using the
2025 evaluation results, so they are exploratory comparisons rather than
independently validation-selected final models.

Each package contains its own README and manifest. The manifests document
scope, probability orientation, provenance, reporting status, and hashes.
Chapter 4 and Chapter 5 files are the compact evidence sources to use when
checking the dissertation; exploratory or superseded outputs must not replace
the controlled files.

## Data availability

The analysis uses annual Association Croquet game, event, player, and index
files for 1985-2025. These raw files are not published in this repository.
They must be obtained separately from the source provider under the applicable
access and use conditions, then placed under `data_raw/` using the filenames
expected by `code/data/download.py` and `code/data/build_matches.py`.

The checked full-history table, per-game predictions, rating histories,
player-appearance tables, caches, and repeated experiment dumps are also
excluded from Git. After obtaining the source data, reconstruct the checked
historical table before running downstream workflows. Generated files under
`outputs/` are ignored by default; only the reviewed dissertation evidence
packages are tracked there.

## Scientific provenance

`archive/legacy_steps/` preserves chronological scientific analysis and
validation scripts that remain useful for provenance but are not supported
entry points. Writing-only exporters and experiment-planning helpers are not
part of the technical deliverable. `archive/research_outputs/` retains only
compact source artefacts referenced by the controlled-evidence manifests and
the validation tables read by the current checks.

Use the tagged historical snapshots only when an exact earlier layout is
needed. For current use, start with `code/`, `tests/`, and the controlled
evidence packages.
