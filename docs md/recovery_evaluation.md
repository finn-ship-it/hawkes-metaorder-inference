# Recovery evaluation harness

This document accompanies `simulator.recovery_eval` and explains the metric definitions and the recommended way to use the harness on the shipped configurations.

## Metric definitions

The harness discretises the simulation horizon `[0, horizon)` into a grid of `n_steps = ceil(horizon / time_resolution)` time bins. Two boolean masks are constructed: `true_mask[i]` is `True` iff the `i`-th bin lies inside any true `MetaOrderWindow`; `inferred_mask[i]` is `True` iff the `i`-th bin lies inside any inferred interval. Per-bin counts then yield:

- `precision = TP / (TP + FP)` — fraction of inferred-active time that overlaps with truth.
- `recall = TP / (TP + FN)` — fraction of true-window time that the inference covers.
- `f1 = 2 * precision * recall / (precision + recall)`.
- `mean_detection_delay_seconds` — for each true window, the offset from the window's start to the first inferred-active bin inside it; averaged across windows. `NaN` when no true window is detected.

Edge cases:

- Empty truth and empty inference: precision and recall are vacuously `1.0`.
- Inference present but no truth: precision is `0.0`, recall is `1.0` (vacuous), `f1 = 0.0`.
- Truth present but no inference: precision is `1.0` (no FPs), recall is `0.0`, `f1 = 0.0`.

The `false_positive_intervals` and `false_negative_intervals` lists are derived from the per-bin difference masks and report contiguous runs of disagreement.

## Worked example: `meta_order_smoke.json` over `seeds = 1..5`

The `meta_order_smoke.json` configuration uplifts `BID_UP` and `ASK_UP` by `alpha = 1.0` for `[20, 80)` seconds at horizon `T = 120 s`. The recommended detector configuration is `MetaOrderBaselineConfig(window_seconds=5.0, score_threshold=0.5, direction="up", min_active_seconds=5.0)`.

Run:

```python
from simulator.config import config_from_json
from simulator.runner import run_once
from simulator.observation import default_observation_config
from simulator.artifact import load_run
from simulator.recovery import (
    MetaOrderBaselineConfig, score_meta_order_activity,
)
from simulator.recovery_eval import evaluate_meta_order_recovery
import dataclasses

cfg = config_from_json("configs/meta_order_smoke.json")
detector = MetaOrderBaselineConfig(
    window_seconds=5.0, score_threshold=0.5,
    direction="up", min_active_seconds=5.0,
)

for seed in range(1, 6):
    run_dir = run_once(
        cfg=dataclasses.replace(cfg, seed=seed),
        runs_root="/tmp/meta_eval", observation_config=default_observation_config(),
        write_summary=True,
    )
    run = load_run(run_dir)
    intervals = score_meta_order_activity(run.observed, detector, horizon=cfg.horizon)
    ev = evaluate_meta_order_recovery(
        cfg.meta_order.windows, intervals, horizon=cfg.horizon,
    )
    print(seed, ev.precision, ev.recall, ev.f1, ev.mean_detection_delay_seconds)
```

The numerical output is reported in the ledger's prompt 09 entry.

## How to read the metrics

- A precision near 1.0 with low recall means the detector fires only inside true windows but misses most of them. Lower the `score_threshold` or shorten the `window_seconds`.
- A recall near 1.0 with low precision means the detector fires far outside true windows. Raise the `score_threshold` or restrict `direction`.
- A high `mean_detection_delay_seconds` relative to the true-window length signals that the detector requires too many windows of evidence before triggering. Reduce `min_active_seconds` or `hop_seconds`.

## Limits

- The harness compares time intervals, not event lists. A detector that catches a few correct events inside a window but does not span enough of it will look poor by this metric. This is a deliberate choice: the dissertation's claim is about *interval recovery*, not per-event classification.
- The harness does not currently weight bins by the simulator's instantaneous intensity. A future refinement could give greater weight to bins inside the meta-order's most-active interior.
- The metrics are deterministic functions of the inputs but the inputs themselves depend on the seed; ensemble-averaging is the user's responsibility.

## Cross-references

- `My repo/src/simulator/recovery_eval.py` — implementation.
- `My repo/src/simulator/tests/test_recovery_eval.py` — 8 tests.
- `My repo/docs md/layer2_baseline.md` — the upstream feature/score module.
- `10_dissertation_experiment_runner.md` — the experiment runner that orchestrates the recovery-evaluation pipeline.
