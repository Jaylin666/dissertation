# Croquet Rating Systems Dissertation

This repository contains the supported implementation and compact controlled
outputs for the Association Croquet Elo-Glicko MSc dissertation. The model
definitions, parameters and empirical results are frozen.

## Requirements

Python 3.10 or later is required. Install dependencies in an isolated
environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Repository structure

- `code/` contains the supported data, model, pipeline and analysis modules.
- `tests/` contains equation, orientation and frozen-output regression tests,
  plus the compact validation fixtures required by those checks.
- `outputs/dissertation_evidence/chapter4/` contains the direct Chapter 4
  tables and figures.
- `outputs/dissertation_evidence/chapter5/` contains the direct Chapter 5
  tables and figures.

Historical scripts and intermediate research outputs remain available through
Git history and the frozen repository tags; they are not part of the current
execution tree.

## Lightweight validation

These commands validate tracked records without rebuilding the historical
rating analysis or generating dissertation evidence:

```bash
python -m code.cli validate
python -m unittest discover -s tests -v
python -m compileall -f code
git diff --check
```

## Supported commands

Inspect the command interface and individual workflows with:

```bash
python -m code.cli --help
python -m code.cli build-data --help
python -m code.cli run-elo --help
python -m code.cli run-glicko --help
python -m code.cli compare-models --help
python -m code.cli early-game --help
python -m code.cli entry-diagnostics --help
```

Computational workflows run only when `--full-run` is supplied. They require
the appropriate untracked source data and intermediates and should write to an
ignored local output directory.

## Data and repository states

Raw Association Croquet data and large derived datasets are not published in
this repository. The expected input layout is defined by
`code/data/download.py` and `code/data/build_matches.py`.

The current `main` commit identifies the minimal final technical deliverable.
The existing `dissertation-final-deliverable-2026` tag remains fixed to the
earlier evidence-frozen repository snapshot; it is not moved when `main`
receives documentation or repository-layout cleanup commits.
