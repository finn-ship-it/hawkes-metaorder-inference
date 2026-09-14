# Reproducibility

All commands below start at the repository root. Python 3.12 and the exact
package versions in `requirements_lock.txt` form the tested environment.

## Verify the code

```bash
make setup
make verify
```

This runs the self-contained simulator, estimator, recovery, quoting, and
evaluation tests, followed by one short command-line smoke run. Smoke output
is written under the ignored `.workbench/` directory.

## Verify the twelve-type adapter

The adapter uses a separate checkout of `konqr/lobSimulations`:

```bash
git clone https://github.com/konqr/lobSimulations.git /path/to/lobSimulations
git -C /path/to/lobSimulations checkout f0d5b22a69d9cd0b7d9b3e881514c571c7189e39
export KONARK_REPO_ROOT=/path/to/lobSimulations
make test-d2
```

The compatibility revision and adapted-code notices are recorded in
`THIRD_PARTY_NOTICES.md`.

## Run a controlled simulation

```bash
PYTHONPATH=src .venv/bin/python -m simulator --config configs/meta_order_smoke.json --observe --summarise
```

The command prints the output directory. Use `--help` to inspect seed, horizon,
experiment, and comparison options.

## Empirical EUR/USD inputs

Raw quote files and local preprocessing scripts are not distributed in this
public repository. Empirical fit builders under
`dissertation_artifacts/build_eurusd_*.py` require those external inputs or
the stored fit inputs referenced in each file. They are not part of
`make verify`.

## Frozen dissertation evidence

The current nine plots can be rebuilt independently from their saved inputs:

```bash
python -m pip install -r dissertation_artifacts/figures/requirements-plots.txt
python dissertation_artifacts/figures/render_submission.py
python dissertation_artifacts/figures/render_submission.py --check
```

This plot-only route does not run the experiments or change tables.
See [`SUBMISSION_NOTES.md`](SUBMISSION_NOTES.md) for method provenance,
the empirical clock convention, and the withdrawn historical estimate.

The repository includes the numerical results used by the dissertation:

- `dissertation_artifacts/data/` contains stored JSON and CSV results;
- `dissertation_artifacts/figures/` contains rendered figures;
- `dissertation_artifacts/tables/` contains generated LaTeX fragments;
- `dissertation_artifacts/ARTIFACTS_MANIFEST.md` maps outputs to builders and source results;
- `runs/` at repository root contains preserved run summaries.

The default verification path does not overwrite these files or rerun research
computations.
