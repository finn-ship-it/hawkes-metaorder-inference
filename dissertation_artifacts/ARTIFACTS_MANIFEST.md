# Artefacts manifest — dissertation cross-reference

**MSc submission:** the controlled-simulator correction, regenerated results and
current Overleaf replacements are listed in
[`CONTROLLED_SIMULATOR_CORRECTION_V2.md`](CONTROLLED_SIMULATOR_CORRECTION_V2.md).
That record and the current figure manifest supersede the older controlled
numbers in the historical inventory below. Unlisted legacy plots, baseline
ensemble summaries and envelope comparisons remain historical; their earlier
code and inputs are retained in a separate development archive.

The current nine submission plots are indexed in
[`figures/figure_manifest.json`](figures/figure_manifest.json), including
input and output hashes. See [`SUBMISSION_NOTES.md`](SUBMISSION_NOTES.md)
for the current methods and plot-only rendering instructions.
The inventory below also contains historical artifacts and superseded figures.

This file records the historical inventory of what the dissertation
`\input{}`s, `\includegraphics{}`-es, and quotes inline. Every artefact
in `My repo/dissertation_artifacts/` and every backing run under `My
repo/runs/` is listed here with its source data, the prompt that
produced it, and the dissertation site (chapter, section, label) it
serves.

Last refresh: 2026-05-06 (Phase A). Maintenance catch-up 2026-06-04 (RUN A) —
restoration-pack artefacts (prompts 01–07c) catalogued in §12; §1 and §3 below
remain the Phase-A snapshot. Final residue close-out 2026-06-17: the two
Phase-A figure-title PDFs were redrawn from the committed `data/` summaries
with plain-text titles only, and the Layer-3 markout artifact gained explicit
`plumbing_reconstructions` metadata for the mid-trace reconstruction.
Dissertation source: `Work on Hawkes/overleaf/my overleaf/Work_on_Hawkes/main.tex`.
Repository: `Work on Hawkes/My repo/`.

## 1. Figures (5, Phase A; +7 restoration-pack figures catalogued in §12.1)

| File | Source data | Source prompt | Dissertation site | Label |
| --- | --- | --- | --- | --- |
| [`figures/branching_ratio_invariant.pdf`](figures/branching_ratio_invariant.pdf) | `data/ensemble_summary_*.json` (3 configs, prompt 02) | 07 | Chapter 4 §4.2 | `fig:rho-invariant` |
| [`figures/spread_scaling_loglog.pdf`](figures/spread_scaling_loglog.pdf) | `data/spread_scaling_summary.csv`, `data/spread_scaling_fit.json` (prompt 03) | 07 | Chapter 4 §4.4 | `fig:spread-scaling` |
| [`figures/recovery_f1_distribution.pdf`](figures/recovery_f1_distribution.pdf) | `data/recovery_summary.csv` (prompt 04) | 07 | Chapter 5 §5.6 | `fig:recovery-f1` |
| [`figures/recovery_delay_distribution.pdf`](figures/recovery_delay_distribution.pdf) | `data/recovery_summary.csv` (prompt 04) | 07 | Chapter 5 §5.6 | `fig:recovery-delay` |
| [`figures/kernel_decay_eurusd_sumexp.pdf`](figures/kernel_decay_eurusd_sumexp.pdf) | `Data/Data output/EURUSD output/layer1/layer1_params_sumexp_4type.json` (existing, read-only) | 07 | Chapter 4 §4.6 | `fig:kernel-decay` |

Captions are recorded in `figures/figure_manifest.json` and are the
authoritative copy. The dissertation prose cites `\autoref{<label>}`
or `\Cref{<label>}` — the labels above are the source of truth for
those calls.

## 2. Tables (6)

| File | Source data | Source prompt | Dissertation site | Label |
| --- | --- | --- | --- | --- |
| [`tables/tab_sim_params.tex`](tables/tab_sim_params.tex) | `My repo/configs/{first_milestone, two_regime_smoke, meta_order_smoke}.json` | 08 | Chapter 3 §3.4 | `tab:sim-params` |
| [`tables/tab_module_layout.tex`](tables/tab_module_layout.tex) | enumeration of `My repo/src/simulator/*.py` | 08 | Chapter 3 §3.7 | `tab:module-layout` |
| [`tables/tab_single_run_regression.tex`](tables/tab_single_run_regression.tex) | seed-42 member of each prompt-02 ensemble (`runs/ensembles_2026-05-06/<config>_50seeds/.../<config>_50seeds_42/summary.json`) | 08 | Chapter 4 §4.2 | `tab:single-run-regression` |
| [`tables/tab_spread_scaling.tex`](tables/tab_spread_scaling.tex) | `data/spread_scaling_summary.csv` (prompt 03) | 08 | Chapter 4 §4.4 | `tab:spread-scaling` |
| [`tables/tab_recovery_example.tex`](tables/tab_recovery_example.tex) | `data/recovery_summary.csv` (prompt 04) | 08 | Chapter 5 §5.6 | `tab:recovery-example` |
| [`tables/tab_baseline_parameters.tex`](tables/tab_baseline_parameters.tex) | `My repo/configs/first_milestone.json` + observation defaults | 08 | Appendix B §B.2 | `tab:baseline_parameters` |

Each fragment is a bare `tabular` environment; the dissertation wraps
it in `\begin{table}[H] \caption{...} \label{...}`. Phase B prompt 10
performs the wiring.

## 3. Data files (8, Phase A; +12 restoration-pack data files catalogued in §12.2)

These are the CSVs and JSONs the regenerate.py scripts read.

| File | Source artefact (under `runs/`) | Source prompt | Consumed by |
| --- | --- | --- | --- |
| [`data/ensemble_summary_first_milestone.json`](data/ensemble_summary_first_milestone.json) | `runs/ensembles_2026-05-06/first_milestone_50seeds/ensemble_first_milestone_50seeds/ensemble_summary.json` | 02 | `fig:rho-invariant`, `tab:single-run-regression` |
| [`data/ensemble_summary_two_regime_smoke.json`](data/ensemble_summary_two_regime_smoke.json) | `runs/ensembles_2026-05-06/two_regime_smoke_50seeds/ensemble_two_regime_smoke_50seeds/ensemble_summary.json` | 02 | `fig:rho-invariant`, `tab:single-run-regression` |
| [`data/ensemble_summary_meta_order_smoke.json`](data/ensemble_summary_meta_order_smoke.json) | `runs/ensembles_2026-05-06/meta_order_smoke_50seeds/ensemble_meta_order_smoke_50seeds/ensemble_summary.json` | 02 | `fig:rho-invariant`, `tab:single-run-regression` |
| [`data/ensembles_aggregate.json`](data/ensembles_aggregate.json) | `runs/ensembles_2026-05-06/run_prompt_02_results.json` | 02 | RUN_REPORT.md (prompt 02) |
| [`data/spread_scaling_summary.csv`](data/spread_scaling_summary.csv) | `runs/spread_scaling_2026-05-06/spread_scaling_summary.csv` | 03 | `tab:spread-scaling`, `fig:spread-scaling` |
| [`data/spread_scaling_fit.json`](data/spread_scaling_fit.json) | `runs/spread_scaling_2026-05-06/spread_scaling_fit.json` | 03 | `fig:spread-scaling`, prompt 06 envelope refresh, prompt 12 inline numbers |
| [`data/recovery_summary.csv`](data/recovery_summary.csv) | `runs/recovery_meta_order_smoke_2026-05-06/recovery_summary.csv` | 04 | `tab:recovery-example`, `fig:recovery-f1`, `fig:recovery-delay` |
| [`data/recovery_aggregate.json`](data/recovery_aggregate.json) | `runs/recovery_meta_order_smoke_2026-05-06/recovery_aggregate.json` | 04 | RUN_REPORT.md (prompt 04), prompt 12 inline numbers |
| [`data/eurusd_layer1_summary.json`](data/eurusd_layer1_summary.json) | derived from `Data/Data output/EURUSD output/layer1/layer1_models_summary.csv` and the corresponding `layer1_params_sumexp_4type*.json` files | 06, 09 | `fig:kernel-decay`, prompt 06 envelope refresh, prompt 12 inline numbers |

## 4. Run directories (Phase A production runs)

These are the on-disk back-up of every Phase A computation. Each
`RUN_REPORT.md` summarises the run; each member subdirectory carries
a full artefact set (`config.json`, `events.npz`, `metadata.json`,
`observation.npz`, `summary.json`, `comparison.json`).

| Directory | Purpose | Prompt | Members | Backs |
| --- | --- | --- | --- | --- |
| `runs/ensembles_2026-05-06/first_milestone_50seeds/` | 50-seed ensemble on `first_milestone.json` ($T=60$ s) | 02 | 50 | `fig:rho-invariant`, `tab:single-run-regression` (seed 42) |
| `runs/ensembles_2026-05-06/two_regime_smoke_50seeds/` | 50-seed ensemble on `two_regime_smoke.json` ($T=600$ s) | 02 | 50 | as above |
| `runs/ensembles_2026-05-06/meta_order_smoke_50seeds/` | 50-seed ensemble on `meta_order_smoke.json` ($T=120$ s) | 02 | 50 | as above |
| `runs/spread_scaling_2026-05-06/T030/.../T960/` | spread sweep over $T \in \{30,60,120,240,480,960\}$ s, 30 seeds each | 03 | 180 | `tab:spread-scaling`, `fig:spread-scaling`; spread-scaling exponent claim in §4.4 prose |
| `runs/recovery_meta_order_smoke_2026-05-06/recovery_meta_order_smoke_50seeds_<ts>/` | 50-seed recovery sweep with the dissertation's anchored detector config | 04 | 50 | `tab:recovery-example`, `fig:recovery-f1`, `fig:recovery-delay`; §5.6 worked example |
| `runs/empirical_envelope_refresh_2026-05-06/20260506T140554Z_42/` | seed-42 first_milestone regression vs the refreshed empirical envelope | 06 | 1 | §4.6 envelope-comparison narrative; `targets/empirical_envelope_draft.json` regression check |

## 5. Configurations and targets (read-only inputs)

| File | Role | Dissertation site |
| --- | --- | --- |
| `configs/first_milestone.json` | shipped configuration #1 ($d=4$, single regime, $T=60$ s) | `tab:sim-params`, `tab:baseline_parameters` |
| `configs/two_regime_smoke.json` | shipped configuration #2 (two-regime, $T=600$ s) | `tab:sim-params` |
| `configs/meta_order_smoke.json` | shipped configuration #3 (single regime + meta-order window, $T=120$ s) | `tab:sim-params`, §5.6 worked example |
| `targets/real_fx_envelope.json` | placeholder envelope (calibrated to first-milestone defaults) | §4.5 envelope diagnostics; never overwritten |
| `targets/empirical_envelope_draft.json` | empirical draft envelope (EURUSD-derived); refreshed by prompt 06 with new `eurusd` and `meta` blocks | §4.6 empirical-envelope-bridge narrative |

## 6. Source-code modules (`src/simulator/*.py`)

Cross-referenced once in `tables/tab_module_layout.tex`. The
methodology chapter uses these module names verbatim.

| Module | Role | Dissertation reference |
| --- | --- | --- |
| `artifact.py` | run-artefact persistence | Chapter 3 §3.7, Appendix B §B.4 |
| `comparison.py` | single-run vs envelope comparison | Chapter 4 §4.5 |
| `config.py` | configuration dataclasses + JSON I/O | Chapter 3 §3.4 |
| `ensemble.py` | multi-seed ensemble harness | Chapter 4 §4.5 |
| `ensemble_comparison.py` | ensemble-level comparison | Chapter 4 §4.5 |
| `envelope_bridge.py` | placeholder/empirical envelope merge | Chapter 4 §4.6 |
| `evaluation.py` | per-run summary computation | Chapter 3 §3.6, Chapter 4 §4.5 |
| `experiment.py` | reproducible experiment runner | Chapter 3 §3.7, Chapter 5 §5.6 |
| `hawkes_core.py` | multivariate Hawkes thinning sampler | Chapter 3 §3.1, Appendix A |
| `kernels.py` | exponential kernel; `EXPONENTIAL_SUM`, `POWERLAW_CUTOFF` deferred | Chapter 3 §3.1 |
| `latent.py` | latent regime + meta-order generator | Chapter 3 §3.2, §3.3 |
| `observation.py` | observation operator $\Phi_\beta$ + censor policy | Chapter 3 §3.3, Chapter 4 §4.3 |
| `recovery.py` | threshold regime / meta-order detectors | Chapter 5 §5.3, §5.4 |
| `recovery_eval.py` | precision/recall/F1 evaluation harness | Chapter 5 §5.5 |
| `runner.py` | single-run orchestration | Chapter 3 §3.7 |

## 7. Source-of-truth dissertation claims (numeric inline)

Every numerical claim in the dissertation prose with a specific
backing artefact. Use this table when prompt 12 syncs inline numbers.

| Number | Site in main.tex | Source artefact | Last verified |
| --- | --- | --- | --- |
| $\rho(\Phi)=0.2000$ | §4.2 prose, abstract para 3, Chapter 8 contributions | `data/ensemble_summary_*.json` ($\rho$ mean 0.19999999999999996, std 0) | prompt 02 |
| `num_crossed_observations = 0` invariant | §4.3, §7.1 | $\sum$ across 150 ensemble members + 180 spread-sweep members = 0 | prompt 02, 03 |
| Spread-scaling exponent range $b \in [0.36, 0.45]$ (was $[0.28,\,0.35]$) | §4.4 prose | `data/spread_scaling_fit.json` (mean 0.382, median 0.360, p90 0.417, p99 0.436, max 0.446, final 0.396; all $R^2 > 0.97$) | prompt 03 |
| Per-seed recovery seeds 1–5 (P=0.770, R=0.633, F1=0.688, delay=5.0 s) | §5.6 worked example sanity check (no longer the headline) | `data/recovery_summary.csv` rows 1..5; `data/recovery_aggregate.json` 5-seed footer | prompt 04 — exact reproduction |
| **50-seed HMM recovery (mean F1=0.863, std 0.090; primary recovery method)** | abstract, §5 chapter intro, §8 conclusion | `data/recovery_hmm_meta_order_smoke.json` (`hmm_aggregate.f1_mean` = 0.863) | restored_dissertation_pack prompt 03 |
| 50-seed threshold-detector recovery (F1=0.727±0.084, delay=1.60±3.38 s; comparison baseline) | §5.6 comparison baseline | `data/recovery_aggregate.json` and `data/recovery_hmm_meta_order_smoke.json.threshold_aggregate` | prompt 04 |
| **EURUSD Layer-1 EM-converged fit (primary day 2021-07-20) $\rho=0.9373$, $\ell=-64{,}922.72$, $n_{\mathrm{iter}}=516$, `converged=True`, $N=97{,}952$** | abstract, §4.5 chapter 4, §8 conclusion | `data/eurusd_layer1_em_summary.json` (`primary_day_2021_07_20.em_result.rho_spec` = 0.937308, `.log_likelihood` = -64922.718) | restored_dissertation_pack prompt 01 |
| EURUSD Layer-1 EM single-basin verdict (16 random inits → $\rho=0.937217 \pm 7.5\times 10^{-5}$) | §4.5 prose | `data/eurusd_layer1_em_robustness.json.summary.single_basin_verdict` (true) | restored_dissertation_pack prompt 01 |
| EURUSD Layer-1 sum-exp 4-type **L-BFGS-B** comparator $\rho=0.988$, $\ell=-66{,}715.38$, `converged=False` (corrective footnote, *not* the headline) | §4.5 corrective footnote | `Data/Data output/EURUSD output/layer1/layer1_params_sumexp_4type.json` | prompt 01, 06 (historical) |
| EURUSD Layer-1 EM-converged fit (secondary day 2021-07-19) $\rho=0.9486$, $\ell=-61{,}388.58$ | §4.5 prose | `data/eurusd_layer1_em_summary.json.day_2021_07_19.em_result` | restored_dissertation_pack prompt 01 |
| Withdrawn one-sided kernel experiment; excluded from current submission evidence | historical only | `data/eurusd_bacry_muzy_kernel.json` | retained for provenance |
| Kernel benchmark wall-times (1.93/6.44/2.40 ms for single-exp/sum-exp/power-law) | §3.1 prose, §A.4 footer | `data/kernel_benchmark.csv` | restored_dissertation_pack prompt 04 |
| $10^5$ events/day at $d=4$ | §4.6, Appendix B §B.4 | $N$=97,952 events (rounded to $10^5$) — `data/eurusd_layer1_em_summary.json.primary_day_2021_07_20.n_events` | existing (read-only) |

## 8. Reverse cross-reference: every dissertation site

Use this section to answer "Chapter X §Y references what?".

### Abstract paragraph 3
- $\rho(\Phi) = 0.2000$ → `data/ensemble_summary_*.json`.
- F1 = 0.688 (seeds 1–5) → `data/recovery_summary.csv` rows 1..5.

### Chapter 3 §3.4 Simulation parameters
- `tab:sim-params` → `tables/tab_sim_params.tex` ← `configs/{first_milestone, two_regime_smoke, meta_order_smoke}.json`.

### Chapter 3 §3.7 Implementation and repository layout
- `tab:module-layout` → `tables/tab_module_layout.tex` ← enumeration of `src/simulator/*.py`.

### Chapter 4 §4.2 Configuration-level endogeneity
- `tab:single-run-regression` → `tables/tab_single_run_regression.tex` ← seed-42 member of each prompt-02 ensemble.
- `fig:rho-invariant` → `figures/branching_ratio_invariant.pdf` ← `data/ensemble_summary_*.json`.

### Chapter 4 §4.3 Observation and censor invariant
- `num_crossed_observations = 0` claim → 150 + 180 = 330 production-scale runs.

### Chapter 4 §4.4 Spread scaling diagnostic
- `tab:spread-scaling` → `tables/tab_spread_scaling.tex` ← `data/spread_scaling_summary.csv`.
- `fig:spread-scaling` → `figures/spread_scaling_loglog.pdf` ← `data/spread_scaling_summary.csv` + `data/spread_scaling_fit.json`.
- Exponent range $b \in [0.36, 0.45]$ → `data/spread_scaling_fit.json`. Six per-field values: mean 0.382, median 0.360, p90 0.417, p99 0.436, max 0.446, final 0.396.

### Chapter 4 §4.5 Envelope comparison and ensemble diagnostics
- `targets/real_fx_envelope.json` (placeholder) and `targets/empirical_envelope_draft.json` (empirical) → both in `My repo/targets/`.
- 50-seed ensemble overall statuses warn / fail / fail → `runs/ensembles_2026-05-06/<config>_50seeds/.../ensemble_comparison.json` × 3.

### Chapter 4 §4.5 Empirical-envelope bridge (post-restoration)
- **EM-converged anchor $\rho=0.9373$, $\ell=-64{,}922.72$, $N=97{,}952$ on primary day 2021-07-20** → `data/eurusd_layer1_em_summary.json`.
- L-BFGS-B comparator $\rho=0.988$, $\ell=-66{,}715.38$, `converged=False` (corrective footnote, not headline) → `Data/Data output/EURUSD output/layer1/layer1_params_sumexp_4type.json`.
- 16-init single-basin verdict → `data/eurusd_layer1_em_robustness.json`.
- Historical one-sided kernel output → `data/eurusd_bacry_muzy_kernel.json`; withdrawn from the submission comparison.
- `fig:nonparametric-kernel` → `figures/eurusd_nonparametric_kernel.pdf` ← full stored CLS and converged EM branching matrices from `data/eurusd_konark_cls_kernel.json` and `data/plot_inputs/eurusd_em_parameters.json`, with two selected kernel profiles over the first ten seconds.
- `targets/empirical_envelope_draft.json` post-refresh: 9 sourced fields + new `eurusd` + `meta` blocks.

### Chapter 5 §5.6 Worked example on `meta_order_smoke.json`
- `tab:recovery-example` → `tables/tab_recovery_example.tex` ← `data/recovery_summary.csv`.
- `fig:recovery-f1` → `figures/recovery_f1_distribution.pdf` ← `data/recovery_summary.csv`.
- `fig:recovery-delay` → `figures/recovery_delay_distribution.pdf` ← `data/recovery_summary.csv`.
- Per-seed numbers (seeds 1–5) → `data/recovery_summary.csv` rows 1..5 (reproduced exactly).
- 50-seed footer numbers → `data/recovery_aggregate.json`.

### Appendix B §B.2 Baseline parameters
- `tab:baseline_parameters` → `tables/tab_baseline_parameters.tex` ← `configs/first_milestone.json` + observation defaults (`tick_size=1e-5`, `min_spread_ticks=1`).

## 9. Repository scope

This public snapshot contains the EUR/USD dissertation implementation and its
supporting simulation results. Raw market-data archives, unrelated research
drivers and future research notes are retained separately.

## 10. Path-handling decision (prompt 10)

The dissertation source `main.tex` lives at `Work on Hawkes/overleaf/my
overleaf/Work_on_Hawkes/main.tex` — both the `Work on Hawkes` and `my
overleaf` segments contain spaces, and `\input{}` traversal across
`..` segments with spaces is fragile under different LaTeX engines.

**Decision:** the table fragments are *copied* into
`overleaf/my overleaf/Work_on_Hawkes/tables/` and `\input{tables/<name>}`-ed
from `main.tex`. The canonical sources live under
`My repo/dissertation_artifacts/tables/` and the regenerate.py driver
overwrites both locations whenever it runs. The same convention will be
applied for figures (prompt 11) — copy into
`overleaf/.../figures/` and `\includegraphics{figures/<name>}`.

**Reproducibility implication:** `dissertation_artifacts/tables/regenerate.py`
must `cp` the fragments into the Overleaf folder after writing them, or
the operator must do so manually. The end-to-end recipe (prompt 13)
will encapsulate this.

## 11. Reproducibility envelope

- All artefacts under `dissertation_artifacts/` are regenerable from the
  source data + the regenerate.py drivers; running `regenerate.py` is
  byte-deterministic except for `last_generated_utc` timestamps in
  `figure_manifest.json`.
- The figure factory and table factory rebuild from `data/` under the
  same directory; you do **not** need to traverse to `runs/` to
  re-produce the dissertation outputs once `data/` is populated.
- The `data/` files are themselves regenerable from `runs/` via the
  prompt-02/03/04/06 drivers in `runs/run_prompt_*.py`.
- See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact command
  ladder.

## 12. Restoration-pack artefacts (prompts 01–07c) — manifest catch-up (2026-06-04)

Added by the maintenance run of 2026-06-04 (RUN A). Sections §1 (Figures)
and §3 (Data files) above are the Phase-A snapshot (last refreshed
2026-05-06) and do **not** list the artefacts produced by the restoration
pack (prompts 01, 02, 02b, 03, 04, 10msc, 07, 07b, 07c). Those are
catalogued here. Unlike the Phase-A `runs/ → data/` copy pattern, each of
these is written directly by a `build_*.py` driver in
`dissertation_artifacts/` (run via `rebuild.py`), so this table carries a
**builder** column in place of the §3 "source artefact under `runs/`"
column. Verified against the live `data/` and `figures/` directories on
2026-06-04 (figures: 5 Phase-A + 7 here = 12 on disk; data: the Phase-A
set + 12 here = 21 on disk).

### 12.1 Figures (7)

| File | Source prompt | Builder | Description |
| --- | --- | --- | --- |
| [`figures/recovery_hmm_posterior.pdf`](figures/recovery_hmm_posterior.pdf) | 03 | `build_recovery_hmm_meta_order_smoke.py` | Smoothed Baum–Welch HMM posterior trace for the median-F1 seed (Layer-2 recovery; ensemble F1=0.863). |
| [`figures/eurusd_nonparametric_kernel.pdf`](figures/eurusd_nonparametric_kernel.pdf) | current submission | `figures/render_eurusd_kernel_comparison.py` | Hawkes excitation kernels for EUR/USD: full fitted branching matrices with spectral radii, above two aligned kernel profiles illustrating bid reversal and same-direction ask movement. |
| [`figures/eurusd_konark_cls_kernel.pdf`](figures/eurusd_konark_cls_kernel.pdf) | 02b | `build_eurusd_konark_cls.py` | Konark CLS-LogLin long-memory kernel estimate on EURUSD (ρ≈0.978). |
| [`figures/layer3_d1_markout_distribution.pdf`](figures/layer3_d1_markout_distribution.pdf) | 10msc | `build_layer3_d1_skew_markout.py` | Per-side markout \|r−r*\| KDE, Episode A (Hawkes-blind) vs B (HMM-aware) on D1. |
| [`figures/d2_vs_d1_branching_ratio.pdf`](figures/d2_vs_d1_branching_ratio.pdf) | 07 | `build_d2_konark_inference.py` | Per-FX-type composition-gap diagnostic, D2 legacy "match-D1" framing vs D1 (CONFIG_INCOMPLETE_B). |
| [`figures/d2_latent_book_inference.pdf`](figures/d2_latent_book_inference.pdf) | 07b | `build_d2_latent_book_calibration.py` | D2 latent full-book inference summary under explicit C_β with injected meta-order truth. |
| [`figures/d2_regime_detectability_sweep.pdf`](figures/d2_regime_detectability_sweep.pdf) | 07c | `build_d2_regime_detectability.py` | 3-panel regime-detectability sweep: F1_latent & F1_observed vs intensity, censoring-cost heatmap. |

### 12.2 Data files (12)

| File | Source prompt | Builder | Description |
| --- | --- | --- | --- |
| [`data/eurusd_layer1_em_summary.json`](data/eurusd_layer1_em_summary.json) | 01 | `build_eurusd_layer1_em.py` | **The dissertation anchor.** EM-converged sum-exp 4-type fit; primary day 2021-07-20 ρ=0.937308, ℓ=−64,922.72, N=97,952, n_iter=516, `converged=True`; secondary day ρ=0.9486. |
| [`data/eurusd_layer1_em_robustness.json`](data/eurusd_layer1_em_robustness.json) | 01 | `build_eurusd_layer1_em_robustness.py` | 16-init single-basin robustness check (ρ_med=0.937217, cluster range 7.5×10⁻⁵; verdict YES). |
| [`data/eurusd_bacry_muzy_kernel.json`](data/eurusd_bacry_muzy_kernel.json) | historical | `build_eurusd_bacry_muzy.py` | Withdrawn one-sided kernel experiment; its stored values are preserved but are not submission evidence. |
| [`data/recovery_hmm_meta_order_smoke.json`](data/recovery_hmm_meta_order_smoke.json) | 03 | `build_recovery_hmm_meta_order_smoke.py` | 50-seed HMM recovery ensemble; `hmm_aggregate.f1_mean`=0.863 (primary) vs `threshold_aggregate` 0.727. |
| [`data/kernel_benchmark.csv`](data/kernel_benchmark.csv) | 04 | `src/simulator/tests/benchmarks/bench_kernels.py` | Kernel wall-clock benchmark (EXP / SUM-EXP / POWERLAW = 1.93 / 6.44 / 2.40 ms; d=4, T=10 s, median of 20). |
| [`data/eurusd_konark_cls_kernel.json`](data/eurusd_konark_cls_kernel.json) | 02b | `build_eurusd_konark_cls.py` | Saved non-parametric CLS fit with log-linear lag bins and OSQP; ρ=0.9784, compared with converged EM at 0.9373. |
| [`data/konark_cls_low_rho_bias_diagnostic.json`](data/konark_cls_low_rho_bias_diagnostic.json) | 02b (forensics) | `build_konark_cls_low_rho_forensics.py` | Bias-forensics matrix isolating the OSQP non-negativity "ghost" off-diagonal mass at low ρ (LIMITATION_EXPLAINED). |
| [`data/layer3_d1_skew_markout_comparison.json`](data/layer3_d1_skew_markout_comparison.json) | 10msc | `build_layer3_d1_skew_markout.py` | Full per-seed Layer-3 SkewAgent + markout comparison, 30-seed matched pairs A vs B (NEGATIVE_RESULT: rule too coarse), with `plumbing_reconstructions.mid_trace` documenting the right-continuous mid reconstruction used by the markout harness. ~3.9 MB. |
| [`data/d2_konark_inference_summary.json`](data/d2_konark_inference_summary.json) | 07 | `build_d2_konark_inference.py` | D2 legacy "match-D1" integration summary; 8,602 FX events, asymmetric composition (BID_UP 62.3%); CONFIG_INCOMPLETE_B (frozen). |
| [`data/eurusd_observed_composition.json`](data/eurusd_observed_composition.json) | 07b | `build_eurusd_observed_composition.py` | Real-EURUSD observed 4-type composition baseline (calibration target for the D2 latent book). |
| [`data/d2_latent_book_calibration_summary.json`](data/d2_latent_book_calibration_summary.json) | 07b | `build_d2_latent_book_calibration.py` | D2 latent full-book calibration; holds both the default and the threshold-cell-14 headlines (ρ_D2=0.836, F1_obs=0.716, F1_lat=0.749, censoring_cost=−0.032) + the gate-resolution block. |
| [`data/d2_regime_detectability_sweep.json`](data/d2_regime_detectability_sweep.json) | 07c | `build_d2_regime_detectability.py` | 30-cell regime-detectability grid + threshold (cell 14) + Pareto frontier + `gate_resolution_07b_censoring_sanity` block; F1_latent 0.254→0.984. |

*Scope note (this run): §12 catalogues the `data/` and `figures/` artefacts
only, per RUN A Task 4. The §6 source-module table (15 modules) and §4 run
directories are also stale relative to the 24-module package but are outside
this maintenance run's stated scope; they remain tracked via
`tables/tab_module_layout.tex` and `hawkes_tree.txt`.*

### 12.3 Evidence-pack additions (P6b, P6c)

| File | Source run | Builder | Description |
| --- | --- | --- | --- |
| [`data/eurusd_layer1_em_residual_diagnostics.json`](data/eurusd_layer1_em_residual_diagnostics.json) | P6b | `build_eurusd_layer1_em_residual_diagnostics.py` | Time-rescaling residual diagnostics for the EM anchor fit (alpha_all reconstructed, anchor reproduced exactly); per-type means 1.000, pooled variance 0.684, KS 0.183 — diagnostic check, departures stated honestly. |
| [`figures/eurusd_layer1_em_residual_qq.pdf`](figures/eurusd_layer1_em_residual_qq.pdf) | P6b | `build_eurusd_layer1_em_residual_diagnostics.py` | QQ + tail-sensitive survival of the rescaled residuals vs Exp(1). |
| [`data/eurusd_layer1_likelihood_ladder.json`](data/eurusd_layer1_likelihood_ladder.json) | P6b | `build_eurusd_likelihood_ladder.py` | Poisson / single-exp / sum-exp-R4 L-BFGS-B (superseded) / sum-exp-R4 EM ladder; EM lowest AIC 129,981 / BIC 130,627 (k=68). |
| [`tables/tab_eurusd_likelihood_ladder.tex`](tables/tab_eurusd_likelihood_ladder.tex) | P6b | `build_eurusd_likelihood_ladder.py` | LaTeX ladder table (ch.04 Table 4.4). |
| [`data/layer3_d1_skew_frontier.json`](data/layer3_d1_skew_frontier.json) | P6c | `build_layer3_d1_skew_frontier.py` | Posterior-sensitivity skew frontier (D1, 30 seeds); reuses the 10msc protocol, sweeps s∈{0,0.10,0.20,0.35,0.50,0.75,1.00}. Feasible region s∈{0.10,0.20,0.35,0.50}; best s=0.50 (+6.33% toxicity reduction at 0.601 fill retention). Endpoint cross-check vs `layer3_d1_skew_markout_comparison.json` passes. |
| [`figures/layer3_d1_skew_frontier.pdf`](figures/layer3_d1_skew_frontier.pdf) | P6c | `build_layer3_d1_skew_frontier.py` | Frontier plot: fill-rate retention (x) vs 5 s toxicity reduction (y), labelled by sensitivity; 0.50 retention + 0 reduction reference lines; feasible quadrant shaded. |
| [`tables/tab_layer3_d1_skew_frontier.tex`](tables/tab_layer3_d1_skew_frontier.tex) | P6c | `build_layer3_d1_skew_frontier.py` | LaTeX frontier table (ch.06 Table; sensitivity / toxicity reduction / fill retention / sign-stable / mean fills). |

*Helper: `rebuild_figure_cosmetics_p6b.py` (P6b) regenerates two figures from committed summary JSONs without recomputing headline artefacts.*
