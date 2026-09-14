# Horizon-aware spread contract

This document records the decision and the action taken under `02_horizon_aware_spread_contract.md`. It is the operational counterpart to `spread_scaling_diagnostic.md` (the supporting evidence) and `DECISION_LOG.md` D-003 (the rationale).

## Decision

Branch 2: the spread-distribution diagnostics in `summary.json` exhibit *sub-diffusive* scaling in the realised observed event count `N_obs`. The mechanism is the censor policy at `min_spread_ticks = 1`, which acts as a reflecting barrier on the spread random walk and induces a state-dependent restoring drift. Mean reversion is therefore present, but it is finite and partial; the spread distribution still grows with horizon, only more slowly than `sqrt(N_obs)`.

The chosen response is documentation only. The placeholder envelope's spread targets and tolerances are not changed in their numerical content; only the `notes` strings on the affected fields are revised to make the horizon dependence explicit and to point at `spread_scaling_diagnostic.md`. The simulator core, the observation operator, the evaluation harness, and the comparison layer are untouched.

## What changed

- `targets/real_fx_envelope.json`: the `notes` strings on `observation.final_spread_ticks`, `observation.spread_mean_ticks`, `observation.spread_median_ticks`, `observation.spread_p90_ticks`, `observation.spread_p99_ticks`, and `observation.spread_max_ticks` were rewritten to record the prompt-01 fitted exponents and the reflecting-barrier mechanism. The numerical `target` and `tolerance` fields are unchanged.
- `My repo/docs md/spread_scaling_diagnostic.md`: written in the same prompt to record the underlying evidence.
- `My repo/docs md/horizon_aware_spread_contract.md`: this file.

## What did not change

- `My repo/src/simulator/observation.py` — observation operator unchanged.
- `My repo/src/simulator/evaluation.py` — `summary.json` schema unchanged.
- `My repo/src/simulator/comparison.py` — `comparison.json` contract unchanged.
- `My repo/src/simulator/hawkes_core.py` and the rest of the simulator core.
- The `target` and `tolerance` numbers in `targets/real_fx_envelope.json`.

## How a single-run comparison should now be read

For any run with `T` materially different from 60 seconds (the placeholder calibration horizon for the spread fields), expect systematic warnings or failures on the spread fields. These are *not* simulator regressions; they are an artefact of placeholder tolerances calibrated against a single `(T, N_obs)` point. The correct interpretation is:

1. `observation.num_crossed_observations` is the only spread-related hard invariant. It must remain zero by construction; any non-zero value indicates a regression in the censor policy.
2. `endogeneity.spectral_radius` is configuration-level and deterministic. It is the cleanest single-run check.
3. `observation.spread_mean_ticks` and `observation.spread_median_ticks` are the cleanest path-integrated diagnostics. Treat them as descriptive rather than gating until the envelope is empirically calibrated.
4. `observation.spread_p90_ticks`, `observation.spread_p99_ticks`, and `observation.spread_max_ticks` are tail diagnostics; expect the highest horizon sensitivity here.
5. `observation.final_spread_ticks` is a single-sample summary of a reflecting random walk. Its placeholder warn/fail outcomes carry no microstructural information until empirical FX targets are integrated.

## Deferred to later prompts

- An optional ensemble-level comparison module is the natural next step (prompt 03). The ensemble mean is more stable than the per-member spread fields and can be matched against a single tolerance per field with materially less seed-driven noise.
- The empirical envelope bridge (prompt 04) is the path for replacing the placeholder spread targets with measured FX values. Until that prompt is executed and the empirical numbers are ratified, the placeholder targets remain in place.
- A fully horizon-aware envelope, with target functions of `T` or `N_obs` rather than scalars, remains future work.
