# Active analysis code

The active code is organised by responsibility rather than meeting number:

- `models/`: canonical Elo and Glicko equations;
- `data/`: source download and checked historical game construction;
- `pipelines/`: Elo, Glicko, and orientation-corrected comparison workflows;
- `analysis/`: orientation, early-game, burn-in, drift, and entry diagnostics;
- `config.py`: frozen dissertation parameters and expected sample sizes;
- `io_utils.py`: checked input loading and the shared chronological sort;
- `validation_utils.py`: reusable validation records and result checks;
- `cli.py`: discoverable command-line entry points.

The small `glicko_core.py` module is retained only as an import compatibility
layer. New code should import from `code.models.glicko`.

## Commands

Run lightweight regression checks:

```bash
python -m code.cli validate
python -m unittest discover -s tests -v
python -m compileall -f code
```

Inspect available commands:

```bash
python -m code.cli --help
python -m code.cli run-elo --help
python -m code.cli entry-diagnostics --help
```

Run a full workflow below a protected output root:

```bash
python -m code.cli run-elo --full-run --output-root outputs/reproduction
python -m code.cli run-glicko --full-run --output-root outputs/reproduction
python -m code.cli compare-models --full-run --output-root outputs/reproduction
python -m code.cli early-game --full-run --output-root outputs/reproduction
python -m code.cli entry-diagnostics --full-run --output-root outputs/reproduction
```

The default CLI mode is validation-only. A computational workflow runs only
when `--full-run` is supplied. The Elo and Glicko core workflows read the
tracked name-free dataset in `data/processed/`. Rebuilding that dataset needs
private raw inputs, while downstream workflows may still need the untracked
intermediates described in the root `README.md`. These examples are individual
entry points, not a claim that all workflows can be reproduced with one
command.

