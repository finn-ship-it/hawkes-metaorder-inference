# Hawkes Meta-Order Inference

Research code for an MSc dissertation on recovering hidden directional
pressure from FX top-of-book events.

## MSc submission: `msc-submission-2026`

This release corrects the controlled simulator's next-event bound so that it
includes excitation from the event just accepted. The affected controlled
results and submission figures have been regenerated; empirical EM/CLS fits,
fitted-model forward checks and twelve-type projection results are unchanged.
The [submission tag](https://github.com/finn-ship-it/hawkes-metaorder-inference/tree/msc-submission-2026)
is a single-commit snapshot of the submitted code and supporting results.
Earlier development history is retained separately, outside this public repository.

See [the correction and Overleaf handoff](dissertation_artifacts/CONTROLLED_SIMULATOR_CORRECTION_V2.md)
for the changed results, replacement assets and reproducible rerun command.

The repository contains:

- four-type Hawkes simulation with known directional-pressure periods;
- top-of-book projection and minimum-spread filtering;
- parametric EM and non-parametric conditional-least-squares estimation;
- HMM and rate-and-imbalance recovery methods;
- posterior-informed quoting and markout evaluation;
- an optional twelve-type latent-book adapter based on Konark Jain and
  contributors' `lobSimulations` project.

## Quick start

Python 3.12 is the tested runtime.

```bash
make setup
make verify
```

`make verify` runs the complete self-contained test suite and one five-second
CLI smoke run. Its output is written under the ignored `.workbench/` directory.

For a single run with a supplied configuration:

```bash
PYTHONPATH=src .venv/bin/python -m simulator --config configs/meta_order_smoke.json --observe --summarise
```

## Code map

Appendix B uses package paths such as `simulator/hawkes_core.py`. In this
repository those files are stored under `src/simulator/`.

| Task | File |
|---|---|
| Controlled Hawkes simulation | `src/simulator/hawkes_core.py` |
| Directional-pressure schedule | `src/simulator/latent.py` |
| Top-of-book projection | `src/simulator/observation.py` |
| Twelve-type latent-book adapter | `src/simulator/d2_konark_latent_book.py` |
| HMM recovery | `src/simulator/recovery_hmm.py` |
| Recovery evaluation | `src/simulator/recovery_eval.py` |
| Quote rule | `src/simulator/skew_agent.py` |
| Markout calculation | `src/simulator/markout.py` |

Start parameter changes in `configs/meta_order_smoke.json`. Configuration
validation is in `src/simulator/config.py`.

```text
src/simulator/              Simulation, estimation, recovery, and evaluation
src/simulator/tests/        Unit and integration tests
configs/                    Controlled-simulation configurations
targets/                    Stored empirical comparison targets
dissertation_artifacts/     Frozen data, figures, tables, and builders
runs/                       Preserved run summaries
```

## Optional twelve-type integration

The twelve-type adapter uses a separate checkout of
[`konqr/lobSimulations`](https://github.com/konqr/lobSimulations). Point the
test harness at that checkout:

```bash
export KONARK_REPO_ROOT=/absolute/path/to/lobSimulations
make test-d2
```

The audited compatibility revision and code attribution are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Data and stored results

Raw EUR/USD quote files and local preprocessing scripts are not included.
Frozen dissertation results are under `dissertation_artifacts/data/`; the
corresponding figures and tables sit in the adjacent directories.

The nine current submission plots, their frozen inputs, and plot-only
rendering instructions are listed in
[`SUBMISSION_NOTES.md`](dissertation_artifacts/SUBMISSION_NOTES.md).
The historical one-sided kernel experiment is excluded from the submission
comparison; Figure 4.1 uses the saved CLS and converged EM estimates.

The standard verification commands do not rerun empirical fitting or research
experiments. Their external inputs and provenance are recorded in
`dissertation_artifacts/REPRODUCIBILITY.md` and
`dissertation_artifacts/ARTIFACTS_MANIFEST.md`.

## Citation and reuse

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

No open-source licence is attached to the original code at this stage. Reuse
requires permission from the author. Third-party notices apply to the adapted
material identified in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
