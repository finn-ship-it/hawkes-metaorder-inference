# Dissertation experiment runner

This document accompanies `simulator.experiment` and `simulator.runner`'s `--experiment` flag. It describes the output directory layout, the manifest schema, and the recommended way to add new experiments to the dissertation.

## What the runner does

`run_experiment(cfg)` executes the following steps in order:

1. Load the `SimulatorConfig` from `cfg.config_path`.
2. Create the experiment directory `<output_root>/<experiment_label>_<timestamp>/`.
3. Persist `resolved_config.json` (the full round-tripped config).
4. Run `simulator.ensemble.run_ensemble` with the supplied seeds, observation enabled, optional envelope comparison, and `timestamp_prefix = experiment_label`.
5. If `envelope_path` is supplied, write `ensemble_comparison.json` via `simulator.ensemble_comparison.save_ensemble_comparison(mode='mean')`.
6. For each member: optionally run the regime baseline; optionally run the meta-order baseline; if the configuration has `MetaOrderWindow`s, evaluate the meta-order recovery against ground truth.
7. Persist one `recovery/member_<seed>.json` per member.
8. Aggregate per-member meta-order metrics into `evaluation_summary.json` (when applicable).
9. Persist `experiment_manifest.json`.

## Output directory layout

```
<experiment_dir>/
    resolved_config.json
    experiment_manifest.json
    evaluation_summary.json
    ensemble_<label>/
        ensemble_summary.json
        ensemble_comparison.json     (when envelope_path is supplied)
        <label>_<seed_1>/            (full member run artefact)
        <label>_<seed_2>/
        ...
    recovery/
        member_<seed_1>.json
        member_<seed_2>.json
        ...
```

## Manifest schema

`experiment_manifest.json` keys:

- `experiment_schema_version` — version of the manifest contract.
- `experiment_label`, `experiment_timestamp_utc`, `experiment_dir`.
- `config_path` — absolute path of the input config.
- `envelope_path` — absolute path of the envelope, or `null`.
- `seeds` — the integer seed list.
- `ensemble_dir`, `ensemble_summary_file`, `ensemble_comparison_file` — the ensemble artefacts; `ensemble_comparison_file` is `null` when no envelope was supplied.
- `recovery_dir`, `evaluation_summary_file` — the recovery artefacts.
- `true_meta_order_windows` — verbatim copy of the simulator's true meta-order windows (start, end, alpha, target event types) so that the manifest is self-contained for downstream reading.
- `regime_baseline_used`, `meta_order_baseline_used` — booleans.

## CLI

The shipped CLI exposes the runner via `python -m simulator.runner --experiment`:

```bash
PYTHONPATH=src python -m simulator.runner \
    --config configs/meta_order_smoke.json \
    --experiment \
    --experiment-seeds 1,2,3,4,5 \
    --experiment-label dissertation_meta_order \
    --runs-root /tmp/dissertation_runs \
    --compare targets/real_fx_envelope.json
```

The CLI does not currently expose `regime_baseline_cfg` or `meta_order_baseline_cfg`. For experiments that need recovery and evaluation, call `run_experiment` from Python:

```python
from simulator.experiment import ExperimentConfig, run_experiment
from simulator.recovery import (
    RegimeBaselineConfig, MetaOrderBaselineConfig,
)

exp = ExperimentConfig(
    config_path="configs/meta_order_smoke.json",
    seeds=(1, 2, 3, 4, 5),
    output_root="/tmp/dissertation_runs",
    envelope_path="targets/real_fx_envelope.json",
    regime_baseline_cfg=None,
    meta_order_baseline_cfg=MetaOrderBaselineConfig(
        window_seconds=5.0, score_threshold=0.5,
        direction="up", min_active_seconds=5.0,
    ),
    experiment_label="dissertation_meta_order",
)
print(run_experiment(exp))
```

## Adding a new experiment

1. Add a new SimulatorConfig under `My repo/configs/`. The validators on the existing module enforce the FX-native shape; nothing else is required.
2. Choose recovery configurations that match the expected dynamics (see `My repo/docs md/layer2_baseline.md` for tuning advice).
3. Wrap the experiment in a small Python script under `My repo/scripts/experiments/` (folder may need creating). Keep the script under 50 lines: a single `run_experiment` call with explicit dataclass arguments. The run produces a self-contained directory; downstream analysis should read from that directory rather than re-running.

## Reproducibility

- The runner uses the seeds verbatim. Re-running with the same seed list and the same config produces identical artefacts modulo the timestamp.
- The `resolved_config.json` is the round-tripped config; it captures the exact parameter set the simulator saw, regardless of any future edits to the source config file.
- The `experiment_manifest.json` records absolute paths to every input. Move-only renaming of the experiment directory is safe; any other path edit will require regenerating the manifest.

## Cross-references

- `My repo/src/simulator/experiment.py` — implementation.
- `My repo/src/simulator/tests/test_experiment.py` — 4 tests (smoke on first_milestone, full pipeline on meta_order_smoke, argument validation).
- `My repo/docs md/layer2_baseline.md` — how to tune the recovery baselines.
- `My repo/docs md/recovery_evaluation.md` — what the metrics mean.
