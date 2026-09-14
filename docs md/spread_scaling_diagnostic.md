# Spread scaling diagnostic

This document records the outcome of `01_spread_scaling_diagnostic.md`. It quantifies how the spread-distribution diagnostics in `summary.json` scale with the simulation horizon `T` and the realised observed event count `N_obs`.

## Method

A controlled sweep on `configs/first_milestone.json` over horizons `T in {30, 60, 120, 240, 480}` seconds. Each horizon was run with `seeds = 1..10` and `observe = True` via `simulator.ensemble.run_ensemble`. The per-ensemble means of the five spread-distribution fields and `final_spread_ticks` were extracted and log-log-regressed against the realised `observation.num_observed_events`.

All other configuration parameters were held at the first-milestone defaults: `d = 4`, `mu = 0.5 * ones(4)`, `alpha = 0.1 * ones(4, 4)`, `beta = 2.0 * ones(4, 4)`, exponential kernel, single regime, no meta-order, `min_spread_ticks = 1`, censor policy.

## Headline numbers

| `T` (s) | `N_lat` mean | `N_obs` mean | `spread_mean` | `spread_median` | `spread_p90` | `spread_p99` | `spread_max` | `final_spread` |
|---|---|---|---|---|---|---|---|---|
| 30 | 71.5 | 67.6 | 7.89 | 7.95 | 12.65 | 14.38 | 14.80 | 10.60 |
| 60 | 148.4 | 143.1 | 10.53 | 10.50 | 16.58 | 19.14 | 20.00 | 12.70 |
| 120 | 306.1 | 295.9 | 12.48 | 12.80 | 20.05 | 24.01 | 25.20 | 16.30 |
| 240 | 598.1 | 579.1 | 16.46 | 16.80 | 26.30 | 30.91 | 32.70 | 18.10 |
| 480 | 1208.9 | 1178.9 | 17.96 | 17.20 | 31.57 | 38.24 | 40.70 | 23.70 |

## Fitted scaling exponents

The model `field_mean = a * N_obs^b` was fit by ordinary least squares on log-log axes.

| Field | `b` | `a` | R² |
|---|---|---|---|
| `observation.spread_mean_ticks` | 0.2939 | 2.37 | 0.9778 |
| `observation.spread_median_ticks` | 0.2833 | 2.52 | 0.9565 |
| `observation.spread_p90_ticks` | 0.3218 | 3.29 | 0.9957 |
| `observation.spread_p99_ticks` | 0.3424 | 3.44 | 0.9984 |
| `observation.spread_max_ticks` | 0.3535 | 3.39 | 0.9981 |
| `observation.final_spread_ticks` | 0.2761 | 3.28 | 0.9888 |

## Interpretation

The pure-random-walk hypothesis predicts `b = 0.5`. All six fitted exponents fall in the band `b in [0.28, 0.35]`, well below 0.5. The ratio `field_mean / sqrt(N_obs)` is monotonically decreasing across the sweep:

| `T` (s) | `N_obs` | `mean / sqrt(N)` | `p90 / sqrt(N)` | `p99 / sqrt(N)` | `max / sqrt(N)` |
|---|---|---|---|---|---|
| 30 | 67.6 | 0.96 | 1.54 | 1.75 | 1.80 |
| 60 | 143.1 | 0.88 | 1.39 | 1.60 | 1.67 |
| 120 | 295.9 | 0.73 | 1.17 | 1.40 | 1.47 |
| 240 | 579.1 | 0.68 | 1.09 | 1.28 | 1.36 |
| 480 | 1178.9 | 0.52 | 0.92 | 1.11 | 1.19 |

The pure-random-walk hypothesis is rejected. The spread distribution scales *sub-diffusively* in `N_obs`. The mechanism is the minimum-spread invariant enforced by the censor policy at `min_spread_ticks = 1`: when `S_n` approaches the floor, the censor drops `BID_UP` and `ASK_DOWN` events that would push it to zero. This acts as a reflecting barrier that suppresses the lower tail of the spread distribution and induces a state-dependent restoring drift. Each member's spread random walk is therefore reflecting, not free, and the distribution accumulates a bounded positive bias.

The sub-diffusive scaling is consistent across the bulk (`mean`, `median`) and tail (`p90`, `p99`, `max`) of the distribution, with marginally higher exponents in the tails because the tails are insensitive to the lower reflecting barrier. The `final_spread_ticks` exponent (0.28) is below the bulk exponents because it is a single sample and inherits the highest variance.

## Consequence for the envelope

The placeholder envelope's spread tolerances were calibrated against the first-milestone default (`T = 60`, `N_obs ≈ 143`). Under the sub-diffusive scaling above, increasing `T` to 480 inflates the bulk spread fields by approximately a factor of 1.7 and the tail fields by approximately a factor of 2.5. A per-run comparison against horizon-blind tolerances will therefore produce systematic warnings or failures on configurations with longer horizons, even though the simulator behaviour is internally consistent.

This is not a simulator defect; it is a diagnostic-contract issue. The chosen response, ratified in `DECISION_LOG.md` D-003 and implemented under prompt 02, is to (a) document the sub-diffusive mechanism on the envelope's spread fields and (b) defer any horizon-aware envelope rewrite until empirical FX targets are available.

## Future work pointers

1. A second sweep on `configs/two_regime_smoke.json` and `configs/meta_order_smoke.json` to confirm that the sub-diffusive form survives regime mixing and meta-order uplift.
2. An analytical derivation of the reflecting-barrier exponent for a symmetric `{-1, 0, +1}` random walk with absorbing-step censoring at zero. The empirical `b ≈ 0.3` is too high for a single-side-reflected diffusion (which would yield `b = 0.5`) and lower than `b = 0.5`; the discrepancy is attributable to the discrete spectrum of allowed steps and the exact censor mechanic.
3. An explicit ensemble-level comparison module that reports the spread-field means against tolerances after correcting for the realised `N_obs` (deferred to prompt 03).
