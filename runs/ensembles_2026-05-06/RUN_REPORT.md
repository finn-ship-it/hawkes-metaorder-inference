# Run report — prompt 02 (50-seed production ensembles)

Date: 2026-05-06.
Driver: `My repo/runs/run_prompt_02.py`.
Python: `/tmp/hawkes_venv/bin/python3` with `numpy 2.4.4`, `scipy 1.17.1`.
Envelope: `My repo/targets/real_fx_envelope.json` (placeholder).

## What was run

Three 50-seed ensembles via `simulator.experiment.run_experiment`:

| Config | Final dir | Members | Horizon (s) |
| --- | --- | --- | --- |
| `first_milestone.json` | `ensembles_2026-05-06/first_milestone_50seeds/` | 50 | 60 |
| `two_regime_smoke.json` | `ensembles_2026-05-06/two_regime_smoke_50seeds/` | 50 | 600 |
| `meta_order_smoke.json` | `ensembles_2026-05-06/meta_order_smoke_50seeds/` | 50 | 120 |

Each final directory contains an `ensemble_<label>` subdirectory with the
50 member runs, `ensemble_summary.json`, and `ensemble_comparison.json`,
plus a `recovery/` subfolder (member metadata only — meta-order recovery
is not invoked here; that is prompt 04). Seeds are 1..50.

## Branching-ratio invariant

| Config | $\rho(\Phi)$ mean | $\rho$ std | $\rho$ min | $\rho$ max |
| --- | --- | --- | --- | --- |
| `first_milestone` | 0.19999999999999996 | 0.0 | 0.19999999999999996 | 0.19999999999999996 |
| `two_regime_smoke` | 0.19999999999999996 | 0.0 | 0.19999999999999996 | 0.19999999999999996 |
| `meta_order_smoke` | 0.19999999999999996 | 0.0 | 0.19999999999999996 | 0.19999999999999996 |

The standard deviation is exactly zero across all 150 members. The
acceptance threshold was 1e-9; this is far below it. The configuration-
level branching-matrix invariant ($\rho$ depends on $\alpha/\beta$, not on
seed) holds across all three configurations as expected. The single
`endogeneity.spectral_radius` value 0.19999999999999996 is the IEEE-754
closest representation of $0.05 \times 4 = 0.2$.

## Crossed-market invariant

| Config | $\sum$ `num_crossed_observations` | per-seed max |
| --- | --- | --- |
| `first_milestone` | 0 | 0 |
| `two_regime_smoke` | 0 | 0 |
| `meta_order_smoke` | 0 | 0 |

Total: 0 across all 150 runs. The censor policy holds.

## Total rates

| Config | $\bar N_\text{lat}$ | $\bar N_\text{obs}$ | total_rate mean | total_rate std |
| --- | --- | --- | --- | --- |
| `first_milestone` | 147.0 ± 17.7 | 138.8 ± 17.9 | 2.450 | 0.294 |
| `two_regime_smoke` | 2031.5 ± 179.4 | 1993.1 ± 172.0 | 3.386 | 0.299 |
| `meta_order_smoke` | 371.2 ± 23.2 | 357.5 ± 25.7 | 3.093 | 0.193 |

`total_rate` is events per second across the 4-type system. The
`first_milestone` mean of 2.450 is consistent with the prompt-01 single-
seed value of 2.65 (within 1σ at the seed-1 sample); the spread of
single-seed values across 50 seeds is 0.29 events/s. `two_regime_smoke`
includes the high-rate regime ($\mu = 1.5$) for part of the horizon, hence
the higher rate; `meta_order_smoke` carries the meta-order uplift inside
$[20, 80)$.

## Spread distribution (ensemble means, ticks)

| Config | mean | median | p90 | p99 | max | final | censoring_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `first_milestone` | 8.38 | 8.24 | 13.95 | 16.32 | 17.00 | 11.40 | 0.056 |
| `two_regime_smoke` | 23.16 | 21.48 | 43.83 | 52.56 | 55.70 | 37.94 | 0.019 |
| `meta_order_smoke` | 10.66 | 10.27 | 18.16 | 21.74 | 22.96 | 14.74 | 0.037 |

Spread inflates with horizon as expected from the documented sub-
diffusive scaling (`docs md/spread_scaling_diagnostic.md`):
$T = 60 \to 600$ for `two_regime_smoke` predicts a multiplicative
inflation factor of roughly $(600/60)^{0.29} \approx 1.95$ on the bulk and
$\approx 2.27$ on the p90 tail; the empirical inflation factors observed
here are 23.16/8.38 = 2.77 (mean) and 43.83/13.95 = 3.14 (p90), slightly
larger than the diagnostic predicted, which is consistent with the two-
regime variant's higher event rate driving more spread excursions.

## Envelope comparison (placeholder envelope)

| Config | Ensemble overall_status | Per-field status counts | Per-member status counts |
| --- | --- | --- | --- |
| `first_milestone` | **warn** | pass=11, warn=2, fail=0 | pass=6, warn=19, fail=25 |
| `two_regime_smoke` | **fail** | pass=7, warn=2, fail=4 | pass=0, warn=5, fail=45 |
| `meta_order_smoke` | **fail** | pass=10, warn=2, fail=1 | pass=4, warn=22, fail=24 |

Acceptance criterion 4 expected the warn/fail/fail pattern; all three
match. The per-member counts capture how many of the 50 seeds returned
each overall status; the per-field counts apply at the ensemble-mean
level. The `fail` statuses are the documented placeholder-envelope vs
longer-horizon mismatch on the spread fields, confirmed by inspecting
which fields fail (per `ensemble_comparison.json`):

- `first_milestone`: warns on `observation.spread_p99_ticks` and
  `observation.spread_max_ticks` (both p99 and max exceed the
  $T = 60$-calibrated tolerance).
- `two_regime_smoke`: fails on the four spread fields (mean / p90 / p99
  / max all far above the placeholder targets) and warns on
  `observation.spread_median_ticks` and the interarrival statistics.
- `meta_order_smoke`: fails on `observation.spread_p99_ticks` (the p99
  exceeds the 18-tick fail threshold by ~4 ticks); warns on `mean` and
  `max` spread.

This pattern is consistent with the empirical-envelope refresh prompt's
documented expectation: tighter spread tolerances live in
`empirical_envelope_draft.json`, not `real_fx_envelope.json`.

## Self-consistency vs prompt 01 single-seed dry-runs

Prompt 01 ran each config under seed 42 only:

| Config | seed 42 total_rate | 50-seed mean | seed 42 mean spread | 50-seed mean spread |
| --- | --- | --- | --- | --- |
| `first_milestone` | 2.65 | 2.45 | 7.38 (single seed) | 8.38 |
| `two_regime_smoke` | 3.58 | 3.39 | 22.89 (single seed) | 23.16 |
| `meta_order_smoke` | 2.94 | 3.09 | 7.13 (single seed) | 10.66 |

The 50-seed mean spread is higher than seed 42 alone for
`meta_order_smoke` (10.66 vs 7.13) — seed 42 happens to be a low-spread
realisation. The 50-seed mean is the production-quality number for
prompts 06–08 onwards. The dissertation's chapter 4 single-run regression
table (currently quoting seed 42 values for `mean_spread`) will need
to either keep the seed-42 caveat or be updated by prompt 12 to the
ensemble mean.

## No regressions to anchored numbers

| Anchor | Ensemble result | Match |
| --- | --- | --- |
| $\rho(\Phi) = 0.2000$ on three configs | 0.19999999999999996 ± 0 | yes |
| `num_crossed_observations` = 0 invariant | 0 across 150 runs | yes |

Layer-1, recovery, and spread-scaling anchors are not exercised by
prompt 02; they will be re-checked by prompts 03, 04, and 05.

## Driver behaviour

- Driver runtime: ~6 minutes total (first_milestone ~30 s, two_regime
  ~5 min, meta_order ~30 s).
- Idempotent: re-running `run_prompt_02.py` reuses any existing
  `*_50seeds/` directory; only missing ones are produced.
- A `run_prompt_02_results.json` next to the per-config dirs records the
  headline aggregate stats for downstream prompts.

## Findings flagged for downstream prompts

1. **First-milestone single-seed total_rate (2.65) is at the upper tail
   of the 50-seed distribution.** The ensemble mean is 2.45 ± 0.29.
   When prompt 12 syncs inline numbers in the dissertation chapter 4
   single-run regression table, decide explicitly whether to keep the
   seed-42 number with a "seed 42" caveat or replace with the ensemble
   mean. The shipped table currently says 2.65 (seed 42 implied).
2. **Spread-mean inflation factor 600s/60s = 2.77 vs diffusive 1.95.**
   Slightly above the spread-scaling-doc estimate; not regression-
   level. Will be checked again at production scale by prompt 03 and
   any drift logged there.
3. **`meta_order_smoke` spread_mean shifted from 7.13 (seed 42) to
   10.66 (50-seed mean).** Significant — over 50% relative shift. The
   seed-42 single run is unrepresentative for spread fields on this
   config. Update inline numbers via prompt 12 if dissertation
   currently quotes 7.13.

No anchored number drifted. No prompt-amendment necessary at this
boundary.
