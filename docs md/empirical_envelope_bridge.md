# Empirical envelope bridge

This document records the design and the source-to-field mapping of the empirical envelope draft, and explains how the placeholder envelope and the empirical draft are intended to evolve once additional empirical artefacts are available.

## Files

- `targets/real_fx_envelope.json` — the placeholder envelope. **Never overwritten by this bridge.** Every field is marked `placeholder: true` and is calibrated against the simulator's first-milestone default rather than against measured FX microstructure.
- `targets/empirical_envelope_draft.json` — the empirical draft. Each field carries `placeholder: false` and a `source` key pointing at the empirical artefact under `archive_empirical/Data output/EURUSD output/` from which the target was derived.
- `src/simulator/envelope_bridge.py` — the bridge module. Public functions are `load_empirical_envelope_draft` (validates that every draft field has a `source`) and `merge_envelopes` (returns a new envelope dict; never mutates inputs).

## Source-to-field mapping

| Envelope field | Source artefact | Source key | Conversion |
|---|---|---|---|
| `latent.total_rate` | `eurusd_event_audit.csv` | sum of `rate_per_s_global` over rows `bid_up`, `bid_down`, `ask_up`, `ask_down` | none |
| `latent.rates_by_event_type` | `eurusd_event_audit.csv` | per-row `rate_per_s_global` for the four 4-type rows in canonical order | none |
| `endogeneity.spectral_radius` | `layer1/layer1_params_exp_4type_jitter_random_ms.json` | `"spectral_radius"` | none |
| `observation.spread_mean_ticks` | `eurusd_event_audit.csv` | `_spread_summary_` row, `spread_pips_mean` (= 0.380) | × 10 (pips → ticks at `tick_size = 1e-5`) |
| `observation.spread_median_ticks` | `eurusd_event_audit.csv` | `spread_pips_median` (= 0.300) | × 10 |
| `observation.spread_p90_ticks` | `eurusd_event_audit.csv` | `spread_pips_p90` (= 0.500) | × 10 |
| `observation.spread_p99_ticks` | `eurusd_event_audit.csv` | `spread_pips_p99` (= 1.700) | × 10 |
| `observation.spread_max_ticks` | `eurusd_event_audit.csv` | `spread_pips_max` (= 15.400) | × 10 |
| `observation.num_crossed_observations` | structural | — | — |

## What was deliberately not promoted

- **Per-component inter-arrival statistics.** The empirical event-audit reports per-event-type inter-event-time medians and percentiles (`iet_ms_*`), but not the same population-variance moment that the simulator's `summary.json` emits via `_interarrival_moments`. Promoting these without a matching definition would invite spurious comparisons. Deferred.
- **Branching matrix entries (`Phi`).** The empirical 4-type fit produces a full `4 × 4` `branching_matrix`. The simulator's envelope schema does not currently expose a per-element `Phi` field, so nothing maps to it directly. The empirical scalar `spectral_radius` is the single scalar contraction that the existing schema already supports.
- **Sum-of-exponential kernel parameters.** The Layer-1 recommendation document promotes the sum-of-exponentials fit to Tier-3, but the simulator currently supports only single-exponential kernels. Promoting these parameters would require the kernel extension envisaged in `06_kernel_extension_plan.md`.
- **Session-stratified rates.** The audit splits rates into Asia, London, Overlap, NY, and Late sessions. Including these as envelope fields would require either session-aware comparison (out of scope) or a session-conditioned simulator configuration (also out of scope).

## How to merge

Merging is by design non-destructive:

```python
from simulator.envelope_bridge import (
    load_empirical_envelope_draft, merge_envelopes,
)
from simulator.comparison import load_envelope, compare_to_envelope

placeholder = load_envelope("targets/real_fx_envelope.json")
draft = load_empirical_envelope_draft("targets/empirical_envelope_draft.json")
merged = merge_envelopes(placeholder, draft, prefer="draft")
# `merged` is a fresh dict; the on-disk placeholder is unchanged.
report = compare_to_envelope(summary_dict, merged)
```

The `prefer` argument controls the field-level winner when both envelopes provide a target for the same dotted path. The merged envelope carries a top-level `merge_provenance` mapping that records which envelope each field originated from, and a `merge_mode` key recording the choice of `prefer`.

## How to extend the draft

When a new empirical artefact arrives:

1. Add the artefact under `archive_empirical/Data output/` (or an appropriately named subdirectory). Do not move it into `My repo/`.
2. Open `targets/empirical_envelope_draft.json` and add or update a field. Each new field must carry: `target`, `tolerance`, `placeholder: false`, `source` (the path under `archive_empirical/`), and `notes`.
3. Run `pytest src/simulator/tests/test_envelope_bridge.py -q` and `python -c "from simulator.envelope_bridge import load_empirical_envelope_draft; load_empirical_envelope_draft('targets/empirical_envelope_draft.json')"` to validate provenance.
4. Do not edit `targets/real_fx_envelope.json` to remove placeholder flags. The placeholder envelope and the empirical draft are deliberately separate files so that running the comparison against either one is a one-line change.

## Open work

- Once the kernel extension prompt (`06_kernel_extension_plan.md`) is executed, the draft envelope can grow per-element `Phi_ij` fields and (if implemented) sum-of-exponential decay-grid checks.
- Once Layer-1 fits exist for additional days, the draft can be promoted to a multi-day median, with `tolerance` widened to reflect day-to-day variation.
- Once the empirical event audit is extended with per-event-type post-event spread distributions, the spread fields above can be tightened.
