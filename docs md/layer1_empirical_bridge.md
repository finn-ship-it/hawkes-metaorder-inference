# Layer-1 empirical bridge

This document explains how the existing FX-native Layer-1 work in `archive_empirical/` relates to the simulator package and how a fitted empirical Layer-1 artefact can be lifted into a `SimulatorConfig`.

## Two streams, one event-type space

Both streams operate on the four-component directional event-type space:

| Index | Simulator `EventType` | Empirical alphabet name |
|---|---|---|
| 0 | `EventType.BID_UP` | `bid_up` |
| 1 | `EventType.BID_DOWN` | `bid_down` |
| 2 | `EventType.ASK_UP` | `ask_up` |
| 3 | `EventType.ASK_DOWN` | `ask_down` |

The mapping is canonical and stable. The empirical `eurusd_event_audit.csv` rows are in the same order. Any rate or branching-matrix vector emitted by the empirical pipeline therefore aligns with the simulator's `mu`, `alpha`, and `beta` arrays index-for-index.

The empirical `mid_up`, `mid_down`, `spread_widen`, `spread_tighten` rows are *projections* of the directional alphabet (see `archive_empirical/Data output/EURUSD output/alphabet_note.md`). They are not modelled directly by the simulator; they are derived quantities you could compute from any simulated observed stream by combining BID and ASK moves. The Layer-1 bivariate mid fit (Tier-2 in the empirical recommendation) corresponds to a 2-component projection that the simulator does not produce natively.

## Two streams, two responsibilities

| Concern | Stream | Where it lives |
|---|---|---|
| Empirical event-stream construction | offline | `archive_empirical/build_event_stream.py` |
| Empirical exponential-Hawkes MLE | offline | `archive_empirical/layer1_hawkes_fit.py` |
| Empirical fit artefacts | offline output | `archive_empirical/Data output/EURUSD output/layer1/*.json` |
| Synthetic event-stream generation | runtime | `simulator.hawkes_core.HawkesSimulator` |
| Synthetic observation projection | runtime | `simulator.observation.ObservationOperator` |
| Empirical envelope draft (Layer-1 → comparison) | one-shot bridge | `My repo/targets/empirical_envelope_draft.json` |

The bridge is one-way: empirical work writes JSON; the simulator reads JSON. Neither pipeline imports the other at runtime.

## Mapping a fitted Layer-1 artefact to a `SimulatorConfig`

A 4-type single-exponential fit JSON (e.g. `layer1_params_exp_4type_jitter_random_ms.json`) carries:

- `mu`: shape `(4,)` — empirical baseline rate per component, in events per second.
- `alpha`: shape `(4, 4)` — empirical excitation magnitudes.
- `beta`: scalar (shared across all pairs in the empirical fit).
- `branching_matrix`: shape `(4, 4)`, equal to `alpha / beta`.
- `spectral_radius`: float, dominant eigenvalue of the branching matrix.

The corresponding `SimulatorConfig` fields are:

- `cfg.baseline.mu = np.asarray(fit['mu'])` — drop in directly.
- `cfg.kernel_params.alpha = np.asarray(fit['alpha'])` — drop in directly.
- `cfg.kernel_params.beta = np.full((4, 4), fit['beta'])` — broadcast the empirical scalar to a `4 x 4` matrix because the simulator stores per-pair betas.
- `cfg.dimension = 4` (constant for the FX-native first milestone).
- `cfg.kernel_family = KernelFamily.EXPONENTIAL`.
- `cfg.latent_regime = LatentRegimeSpec(num_regimes=1, ...)` (or two-regime if the empirical work later supplies regime-conditioned fits).
- `cfg.meta_order = MetaOrderSpec(windows=())` unless the experiment requires uplift.
- `cfg.numerics.spectral_radius_cap = 0.95` (keep the existing cap; the empirical 4-type fit's `rho = 0.50` is well within it).
- `cfg.horizon` and `cfg.seed` — chosen by the experiment.

A worked example: take the empirical `mu = [0.139, 0.142, 0.138, 0.147]`, `alpha` from the JSON, and `beta = 25.7`. The result is a `SimulatorConfig` whose simulated total rate is approximately `2 * sum(mu) / (1 - rho) = 2 * 0.567 / 0.50 ~ 2.27 events/s`, which is the expected order of magnitude given the empirical aggregate `total_rate = 1.016 events/s` (the discrepancy comes from the Hawkes branching-driven amplification in the simulator's stationary regime).

## Optional helper

A future iteration may add `My repo/src/simulator/empirical_bridge.py` exposing:

```python
def config_from_layer1_fit(fit_dict: dict, *, horizon: float, seed: int) -> SimulatorConfig:
    ...
```

This helper is not added under prompt 07 because:

1. The Layer-1 fit JSONs are already structured; a five-line snippet in any experiment script can map the fields by hand.
2. Until the simulator surface gains an `EXPONENTIAL_SUM` kernel (deferred under prompt 06), the helper would only support the single-`beta` 4-type case. A premature interface would lock the bridge into a shape that needs revision when the kernel surface expands.
3. The empirical envelope draft (`targets/empirical_envelope_draft.json`) already provides the principal empirical quantities (`total_rate`, per-component rates, spectral radius, spread distribution) that a comparison would need.

If the dissertation experiments grow to the point where the helper would save real time, add it under a follow-on prompt with a paired test file `test_empirical_bridge.py`.

## What this bridge does not do

- It does not re-fit empirical parameters at simulator runtime.
- It does not pipe real EURUSD ticks into the simulator.
- It does not validate that the empirical fit was produced under any particular jitter scheme; the simulator user is responsible for selecting a fit produced under `random_ms` jitter (see Layer-1 recommendation §4 for why deterministic priority schemes must be avoided).
- It does not bridge the empirical sum-of-exp Tier-3 fit into the simulator. That requires the deferred `EXPONENTIAL_SUM` kernel.

## Cross-references

- `archive_empirical/Data output/EURUSD output/layer1/layer1_recommendation.md` — Tier-1 / Tier-2 / Tier-3 model selection.
- `My repo/docs md/empirical_envelope_bridge.md` — the envelope-side bridge (this is the comparison-side counterpart).
- `My repo/docs md/kernel_extension_decision.md` — why the simulator kernel surface remains exponential-only.
