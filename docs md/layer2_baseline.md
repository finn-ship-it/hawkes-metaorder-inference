# Layer-2 baseline — regime and meta-order recovery

> **Status update — 2026-05-08 (restored_dissertation_pack Phase A):** The probabilistic recovery layer that this document calls "future work" has shipped. **HMM is now the primary Layer-2 recovery method** (`simulator.recovery_hmm`, default for `ExperimentConfig.recovery_method`); the threshold detector documented below is retained as the **comparison baseline**. Headline ensemble result on the 50-seed `meta_order_smoke` benchmark: HMM mean F1 = 0.863 (std 0.090) vs threshold-detector F1 = 0.727 (std 0.084). The rest of this document is preserved as the threshold-baseline reference; treat any "future work / Hook 2 / probabilistic estimator" framing here as historical and superseded.

This document accompanies `simulator.recovery`. It explains the scope of the threshold baseline (now a comparison baseline rather than the primary recovery method), the rationale for the simple thresholding rule, and the limits of what that baseline can be expected to recover. Treat HMM, not the threshold detector, as the dissertation's primary Layer-2 method.

## Scope

The baseline answers two questions from observed simulator output:

1. **Regime label per time window.** Two-regime configurations switch the baseline intensity vector between calm and active states; the recovery problem is to label each window with the inferred regime.
2. **Meta-order activity intervals.** Configurations with one or more `MetaOrderWindow` apply a multiplicative uplift to the baseline of selected event types over a finite interval. The recovery problem is to detect that interval from the observed stream.

Both problems are solved with sliding-window features and threshold rules. There is no probabilistic model and no smoothing beyond optional dwell-time hysteresis.

## Features

`compute_window_features` produces, per non-overlapping window:

- `num_events` — total observed events.
- `directional_imbalance` — `(n_up - n_down) / num_events` where `n_up = count(BID_UP) + count(ASK_UP)` and `n_down = count(BID_DOWN) + count(ASK_DOWN)`. Range `[-1, +1]`.
- `spread_change` — post-event spread at window end minus post-event spread at window start, in ticks.
- `recent_event_rate` — `num_events / window_seconds`.

Windows containing zero events are emitted with all-zero features. The window edges are derived from `numpy.arange(0, horizon - 1e-12, hop_seconds)`; the inclusive-exclusive convention matches `numpy`'s default and is exposed to the caller via `horizon`.

## Regime baseline

`recover_regimes_baseline(observed, cfg)`:

- A window is labelled regime 1 when `recent_event_rate > rate_threshold`, else regime 0.
- An optional `hysteresis_seconds` enforces a minimum dwell time between successive label changes. The hysteresis is one-sided (forward in time); it does not retroactively re-label.
- The output is a list of `(start_time, label)` pairs marking each label transition, beginning at `0.0`.

Tuning advice: set `rate_threshold` to the geometric mean of the two regime-conditional total rates, multiplied by the simulator's stationary amplification factor `1 / (1 - rho)` where `rho` is the spectral radius. For `two_regime_smoke.json` (calm `mu = 0.3`, active `mu = 1.5`, four components, `rho = 0.2`), the rule of thumb gives `threshold ~ sqrt(0.3 * 1.5) * 4 / 0.8 ~ 3.4 events/s`. The dissertation experiment runner uses this as the default.

## Meta-order baseline

`score_meta_order_activity(observed, cfg)`:

- A per-window score is computed. For `direction = "up"` the score is `recent_event_rate * max(0, directional_imbalance)`; for `direction = "down"`, `recent_event_rate * max(0, -directional_imbalance)`; for `direction = "any"`, `recent_event_rate * abs(directional_imbalance)`.
- Windows with score `> score_threshold` are merged into contiguous intervals. Intervals shorter than `min_active_seconds` are dropped.
- The output is a list of `(start, end, mean_score)` tuples per surviving interval.

Tuning advice: the `meta_order_smoke.json` configuration uplifts `BID_UP` and `ASK_UP` by `alpha = 1.0` for `[20, 80)` seconds. With the simulator's default `mu = 0.5` per component the active-window rate is roughly twice the base rate, and the imbalance during the window is biased toward `+1` because two of the four components fire faster. A threshold near the un-uplifted total rate (`~ 2.5 events/s` × imbalance `~ 0.3` = `0.75`) is a sensible starting point.

## Limits

- The threshold baseline is *not* an estimator. It does not identify Hawkes parameters; it only labels windows.
- Sample noise on short horizons (`first_milestone.json`, `T = 60 s`) makes the baseline unreliable for any configuration with fewer than ~50 observed events per window.
- The detector cannot disentangle a regime change from a meta-order onset using rate alone. Combine `recover_regimes_baseline` with `score_meta_order_activity` and consult `directional_imbalance` to triangulate.
- The hysteresis is a debouncing convenience; it is not a Bayesian prior on dwell-time. Sensible values are 1–3 times the window length.

## Future work

- Compare the threshold rule with the implemented two-state HMM in `src/simulator/recovery_hmm.py`.
- Add a likelihood-ratio score that incorporates the kernel's `spectral_radius` and the configuration-known `mu_calm` / `mu_active`, returning a posterior probability rather than a hard label.
- Generalise the meta-order score to a bank of templates (sigmoidal onsets, exponential decays) for the multi-window case.

## Cross-references

- `My repo/src/simulator/recovery.py` — the implementation.
- `My repo/src/simulator/tests/test_recovery.py` — 14 tests covering features, regime baseline, meta-order baseline, and `true_regime_at`.
- `09_meta_order_recovery_evaluation.md` — the evaluation harness that consumes `score_meta_order_activity`'s output.
- `DECISION_LOG.md` D-005 — the decision to ship a threshold baseline rather than a probabilistic estimator.
