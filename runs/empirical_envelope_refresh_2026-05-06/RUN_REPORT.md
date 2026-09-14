# Run report — prompt 06 (empirical-envelope refresh, EURUSD-only)

Date: 2026-05-06.
Driver: `My repo/runs/run_prompt_06.py`.

## What changed in `empirical_envelope_draft.json`

- `eurusd` block added with: `total_rate_per_s`, `rates_by_event_type_per_s`,
  `branching_ratio_primary_day` (= 0.988, primary-day single-UTC fit on
  2021-07-20), `branching_ratio_secondary_day` (= 0.943, single-UTC robustness
  fit on 2021-07-19), `n_events_primary_day` (= 97,952),
  `n_events_secondary_day` (= 104,734), `window_seconds_primary_day` (=
  86,400 s), `primary_day_label`, `secondary_day_label`,
  `spread_p50/p90/p99_ticks`, `spread_scaling_exponents` (six per-field
  $b$ values from prompt 03), and `window` (the 13-month source data window
  from which the two single-day fits are drawn).
- `meta` block added with: `last_refresh_date`, `eurusd_window`,
  `n_seeds_per_ensemble = 50`, `crossed_obs_invariant_holds = true`,
  and the per-config ensemble summary statistics (ρ, total rate, mean
  spread, max crossed obs).
- The original `fields` dictionary is **unchanged** (it carries the
  per-field tolerance schema consumed by `simulator.envelope_bridge.merge_envelopes`).

(Updated 2026-05-06 by terminology audit: the previous keys
`branching_ratio_full_sample`, `n_events_full_sample`,
`window_seconds_full_sample`, `branching_ratio_best_day`,
`n_events_best_day`, `best_day_label` were renamed to
`*_primary_day` / `*_secondary_day` with explicit date labels because
the EURUSD Tier-3 fit is a single UTC day, not a multi-day pool.)

## Anchor verification

| Anchor | Source-of-truth value | What this prompt wrote | Match |
| --- | --- | --- | --- |
| EURUSD primary-day single-UTC $\rho$ (2021-07-20) | 0.987665 | 0.987665 | yes |
| EURUSD secondary-day single-UTC $\rho$ (2021-07-19) | 0.943209 | 0.943209 | yes |
| EURUSD $N$ events on primary day | 97,952 | 97,952 | yes |
| Spread-scaling exponents (new) | $b \in [0.36, 0.45]$ | as provided | yes |
| `crossed_obs_invariant_holds` | 0 across 150 ensemble runs | true | yes |

## Regression check on `first_milestone` (seed 42)

Run dir: `runs/empirical_envelope_refresh_2026-05-06/20260506T140554Z_42/`.

Status counts against the **empirical** draft envelope (after refresh):

```
overall_status: fail
status_counts: pass=2, warn=5, fail=2, missing=0, unchecked=0
```

This is the documented placeholder-vs-empirical distinction: the
placeholder envelope (`real_fx_envelope.json`) was calibrated against
the simulator's first-milestone default (and reports `warn` overall on
seed 42); the empirical draft is calibrated against measured EURUSD
microstructure (median spread 3 ticks vs simulator's ~8 ticks) and
correctly fails the simulator's spread fields. This separation is the
whole point of the empirical-envelope-bridge architecture.

The simulator did not regress — it produces the same $\rho = 0.2$ and
zero crossed observations as before; what changed is the comparison
target.

## Files produced

```
runs/empirical_envelope_refresh_2026-05-06/
├── 20260506T140554Z_42/   # first_milestone seed-42 regression run
│   ├── config.json, events.npz, metadata.json, observation.npz
│   ├── summary.json
│   └── comparison.json
├── regression_status.json # {overall_status, status_counts}
└── RUN_REPORT.md          # this file

targets/empirical_envelope_draft.json   # +eurusd block, +meta block; fields unchanged
```

## Post-run amendments

- EURUSD `branching_ratio_primary_day` reproduces 0.988 exactly — no
  regression. Anchor preserved.
- `crossed_obs_invariant_holds` is `true` (no per-seed violation) —
  censor policy intact.

No prompt amendments triggered.
