# Kernel extension — decision document

> **Status update — 2026-05-08 (restored_dissertation_pack prompt 04, run):** The deferral below is **reversed**. `EXPONENTIAL_SUM` and `POWERLAW_CUTOFF` are now both **implemented** in `simulator/kernels.py` and the `simulator.config.validate*` chain. Tests live under `simulator/tests/test_kernels_sumexp.py` and `simulator/tests/test_kernels_power_law.py`. Wall-clock benchmark (10 s d=4 simulation, median of 20 runs):
>
> | family | median wall (ms) | events |
> |---|---:|---:|
> | EXPONENTIAL | 1.93 | 26 |
> | EXPONENTIAL_SUM (R=2) | 6.44 | 34 |
> | POWERLAW_CUTOFF | 2.40 | 26 |
>
> Source: `dissertation_artifacts/data/kernel_benchmark.csv`; benchmark runner at `simulator/tests/benchmarks/bench_kernels.py`. The deferred-rationale text below is preserved verbatim for historical context.
>
> Power-law form is mathematically attributed to `Konarks-github repo/src/simulation/functions.py::powerLawCutoff`; code reimplemented here, not vendored.

This document records the outcome of `06_kernel_extension_plan.md`. The relevant code-side artefact is the existing `simulator.kernels.build_kernel` factory, which already raises `NotImplementedError` for any non-exponential `KernelFamily` enum value. The decision below explains why this stub is the deliberate end state for the MSc and what would change if the dissertation later expanded into multi-scale kernel territory.

## Decision

**Branch 3 of prompt 06 — defer.** *(Reversed 2026-05-08 by restored_dissertation_pack prompt 04 — see status update at the top.)* The MSc simulator surface remains exponential-only. The `KernelFamily` enum already names `EXPONENTIAL_SUM` and `POWERLAW_CUTOFF`; the factory raises `NotImplementedError` with a clear message. No code is added under this prompt; documentation is.

## Empirical evidence reviewed

From `archive_empirical/Data output/EURUSD output/layer1/layer1_recommendation.md`:

- The 4-type single-exponential fit on EURUSD 20 July 2021 gives `rho(Phi) = 0.50` with `beta = 25.7 / s` (kernel lifetime ~ 39 ms) under `random_ms` jitter. Time-rescaling diagnostic: `Var[tau] = 1.92` — over-dispersed, indicating that a single decay rate cannot simultaneously match fast bid/ask co-activation and slower cross-side dynamics.
- The 4-type sum-of-exponential fit with `R = 4` and `beta in {0.1, 1, 10, 100} / s` gives `rho(Phi) = 0.988` with a log-likelihood improvement of +41,263 units over the matched single-exponential 4-type. Time-rescaling: `Var[tau] = 0.68` — **under-dispersed**, which the recommendation flags as the kernel having "too much flexibility" and fitting non-Hawkesian residual structure.
- The bivariate single-exponential fit on `{mid_up, mid_down}` gives the cleanest of the three: `rho(Phi) = 0.597`, `Var[tau] = 1.48`, `KS = 0.069`. This is the recommended Tier-2 model in the empirical work and is single-exponential.

The empirical evidence therefore supports two contradictory readings: multi-scale structure is detectable by likelihood, but the multi-scale fit is itself flagged as over-flexible. The cleanest single empirical fit is single-exponential. The MSc simulator's exponential default is consistent with the cleanest empirical fit.

## Why a stub rather than a full implementation

1. **Simulator-core surface.** A correct `EXPONENTIAL_SUM` kernel changes the simulator core's intensity-bookkeeping cost from `O(N)` per accepted event (a single recursive sum per pair) to `O(R * N)` (one recursive sum per decay rate per pair). The thinning upper bound also expands to a sum of per-rate envelopes. The simulator core is currently frozen by the protocol; implementing the kernel correctly requires either touching `hawkes_core.py` or relying on the existing intensity-recomputation path, which would require careful re-validation against the existing tests.
2. **Empirical ambiguity.** The Layer-1 recommendation explicitly warns that the `R = 4` sum-of-exp fit is under-dispersed and should be treated as an upper bound on kernel complexity rather than as a target. Building a simulator that outputs distributions matching an under-dispersed fit will make the comparison layer harder to interpret, not easier.
3. **Dissertation focus.** The MSc's central claims are about regime / meta-order recovery from observed simulator output (prompts 08, 09). Multi-scale kernel fidelity is not on the critical path for those claims; it is a downstream refinement that would appear in a "future work" chapter regardless.

## What would change under a future implementation

If a later prompt approves a full `EXPONENTIAL_SUM` implementation, the minimum surface is:

- A new dataclass `ExponentialSumKernelParams` with shape `(R, d, d)` for `alpha` and `(R,)` or `(R, d, d)` for `beta`.
- A new `ExponentialSumKernel` class implementing `evaluate`, `integral`, `upper_bound`. The branching matrix is `sum_r alpha_r / beta_r` (per-element division when `beta` is `(R, d, d)`).
- An update to `build_kernel` to dispatch on `KernelFamily.EXPONENTIAL_SUM`.
- Tests covering the single-`R` limit, the additivity of the branching matrix, and a `config_to_json` / `config_from_json` round trip.
- Verification that `hawkes_core.py` correctly handles the new kernel via its existing protocol-level interface — and, if not, a documented and tested core change.

That extension remains future work.

## What does *not* change today

- `simulator.kernels.build_kernel` continues to raise `NotImplementedError("kernel family EXPONENTIAL_SUM is not implemented in the first coding milestone")` for `EXPONENTIAL_SUM` and `POWERLAW_CUTOFF`. The error message is the operational stub.
- The three shipped configurations all use `KernelFamily.EXPONENTIAL` and continue to validate and run.
- The single-exponential default remains the only kernel exercised by the test suite.

## Cross-references

- `DECISION_LOG.md` D-004 — the formal decision record.
- `targets/empirical_envelope_draft.json` — single-exponential `endogeneity.spectral_radius = 0.4999` field is the empirical target derived from the 4-type single-exponential fit, which is the model the simulator's exponential surface most directly compares to.
- `KONARK_REUSE_AUDIT.md` row for `src/simulation/functions.py` — class A* qualifier ("only when paired with a dedicated test file under `My repo/src/simulator/tests/test_kernels_power_law.py` and only if prompt 06 authorises power-law work; otherwise class D"). Prompt 06 has not authorised power-law work; the qualifier resolves to class D.
