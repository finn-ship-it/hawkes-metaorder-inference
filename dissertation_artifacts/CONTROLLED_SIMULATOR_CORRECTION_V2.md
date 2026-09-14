# Controlled-simulator correction and Overleaf hand-off

The controlled Hawkes simulator now includes the excitation from an event just
accepted at the current time when proposing the next arrival. Candidate acceptance
still uses the pre-event intensity. Invalid proposal bounds now stop the run instead
of silently giving an acceptance probability above one.

The corrected release uses **`msc-submission-2026`**, a single-commit public
snapshot. Earlier development history is archived separately.
Use this release's code, results and figure manifest together; do not mix old controlled
figures with the regenerated tables.

The main numerical changes are HMM mean F1 **0.831 versus 0.728** for the
threshold method, and **all six nonzero quoting settings meeting the aggregate
fill floor**. The largest tested reduction is now **8.57% at s=1**, with
**57.2% median fill retention**. The old conclusion that s=1 fails the fill
criterion must be changed; this is not just a percentage update.

## Verification and reproducibility

- The private and public-mirror test runs each passed **379 tests**, including
  35 deterministic regressions
  for post-event proposal bounds, pre-event acceptance, retained history, all three
  supported kernel families, both simulation entry points, exogenous boundaries,
  and invalid bounds/intensities.
- The original implementation fails the post-event regression for each kernel
  family; the corrected implementation passes.
- Both runs emitted the same 36 OSQP deprecation warnings and no test failures.
- The regenerated recovery, quoting and spread-scaling experiments retain their
  seeds, horizons, pressure schedules, HMM settings, threshold settings and fill
  calibration. Recovery HMM initialisation remains at zero, matching the
  historical builder. The threshold comparator is selected explicitly.
- The [source and protocol record](data/controlled_simulator_correction_v2.json)
  contains source hashes, runtime details, before/after results and file mappings.

From the repository root, with the project dependencies installed:

```bash
python dissertation_artifacts/rebuild_controlled_v2.py --output-root .workbench/controlled-v2-reproduction
```

The output directory must not already exist. This command recomputes recovery,
quoting, spread scaling, seed-42 structural smoke checks for the three controlled
configurations, and kernel timings into that new directory; it does not overwrite
existing research outputs or edit the thesis.
Plot rendering remains separate from the numerical rerun. The final figure sources,
hashes and captions are listed in [figure_manifest.json](figures/figure_manifest.json).

## What remains unchanged

This correction concerns the controlled four-type simulator, not the separate
empirical or twelve-type implementations. The following results and definitions are
unchanged:

- Table 3.1 configurations; the analytical controlled branching ratio
  `rho = 0.20`; kernel equations, pressure schedules and minimum-spread rule.
- Empirical EUR/USD data, EM and CLS estimates: Tables 4.2-4.3 and Figures 4.1-4.2.
- Fitted-model forward validation: Table 4.4, Figure 4.3, G0-G5 and Appendix B.9.
- Twelve-type projection/recovery: Tables 4.5-4.6 and Figures 4.4-4.5.
- Separate fitted near-critical recovery: Table 5.2 and its 0.482/0.245 comparison.
- HMM recursions, quote-rule equations, logistic fill calibration and markout
  definitions. Their realised controlled results have been regenerated.

The unchanged historical slides in Appendix C remain a record of the earlier plan.
Its present-day introductory summary on printed page 76 must, however, use the
corrected results.

## Files to replace in Overleaf

Printed pages below refer to the September 14, 19:50 PDF; later edits can move them.
Replace these assets using their existing Overleaf filenames, or update the
corresponding `includegraphics`/`input` paths if those names differ locally.

| Dissertation item | Corrected repository asset |
|---|---|
| Figure 5.1, p.34 | [recovery_hmm_posterior.pdf](figures/recovery_hmm_posterior.pdf) |
| Figure 5.2, p.36 | [recovery_method_comparison.pdf](figures/recovery_method_comparison.pdf) |
| Figure 6.1, p.42 | [layer3_d1_markout_distribution.pdf](figures/layer3_d1_markout_distribution.pdf) |
| Figure 6.2, p.43 | [layer3_d1_skew_frontier.pdf](figures/layer3_d1_skew_frontier.pdf) |
| Table 4.1, p.18 | [tab_spread_scaling.tex](tables/tab_spread_scaling.tex) |
| Table 5.1, p.36 | [tab_recovery_example.tex](tables/tab_recovery_example.tex) |
| Table 6.1, p.44 | [tab_layer3_d1_skew_frontier.tex](tables/tab_layer3_d1_skew_frontier.tex) |

The table files contain tabular content; retain the dissertation's surrounding
table environment, label and caption. Update the Table 5.1 note as specified below.
Use the current captions in the figure manifest, not captions copied from the old
PDF. Any older plot or table outside this current figure/table replacement inventory
is historical unless explicitly marked as refreshed in the correction record.
Earlier outcomes are preserved in a separate development archive. In particular,
`ensemble_summary_first_milestone.json`, `ensemble_summary_two_regime_smoke.json`,
`ensembles_aggregate.json`, `tab_single_run_regression.tex`, and older empirical-
envelope comparisons are historical outputs, not newly regenerated 50-seed
baseline studies. The corrected recovery ensemble does contain 50 seeds, and the
new smoke checks separately confirm the configured branching ratio and zero
crossed observed books for each controlled configuration.

The replacement passages below give the prose and mathematical notation. When
pasting a percentage into LaTeX, write the percent sign as `\%`.

## Exact Appendix B changes

### Repository link, p.64

Change the appendix's repository link to:

```latex
\url{https://github.com/finn-ship-it/hawkes-metaorder-inference/tree/msc-submission-2026}
```

### B.2: proposal and acceptance, pp.65-66

Keep the opening explanation of Ogata thinning and the separate twelve-type
Poisson additions. Replace the paragraph beginning "At the current time, the
simulator evaluates..." with the following, and delete the final paragraph
beginning "The proposal rate is calculated...". This consolidates the explanation
without retaining the incorrect subcriticality justification:

> In the controlled four-type simulator, each proposal uses the right-limit
> intensity at the current time, including any event just accepted there. For the
> non-negative, decreasing kernels used here, the intensity cannot rise between events
> while the regime and pressure factors stay fixed. Multiplying the total by the
> safety factor therefore gives a proposal rate until the next regime change or
> pressure-period boundary, when it is recalculated. At a candidate event,
> acceptance uses the intensity immediately before that event. The code stops if
> this intensity exceeds the proposal rate.

In the proposal excerpt, replace its first assignment with the current code:

```python
lam_now = self._current_intensity(
    self._state.current_time, include_current=True
)
```

The acceptance call remains `_current_intensity(t_candidate)`, whose default
excludes an event at the candidate time. If retaining the full proposal/acceptance
listing rather than a shortened excerpt, copy it directly from this release's
`src/simulator/hawkes_core.py`, including its finite-bound and bound-violation
checks. Do not leave an old full listing beneath the new explanation.

### B.7: kernel evaluation and computational cost, p.71

Replace only the sentence claiming that the family-specific `upper_bound` supplies
the proposal rate with:

> During simulation, kernel values are summed over the retained event history,
> and the resulting right-limit total intensity supplies the proposal bound
> described in Appendix B.2.

Replace the sentence claiming maintained excitation components and `O(RN)` versus
`O(N)` updates with:

> The implementation recomputes excitation from the events retained within the
> truncation window. For fixed dimension, each evaluation costs \(O(H)\) for the
> single-exponential and power-law families, and \(O(RH)\) for \(R\) exponential
> components, where \(H\) is the number of retained events.

Retain the branching-matrix equations, power-law integral listing and statement
that changing the power-law cutoff changes the process. Refresh the local runtime
comparison in Section 3.1 and the matching qualitative sentence in B.7 from the new
benchmark, rather than carrying forward timings from the old implementation.

## Recovery results and exact repeated-number locations

The recovered-state comparison still favours the HMM, but its magnitude is smaller.
The original reporting conventions are preserved: the HMM summary uses a population
standard deviation (`numpy.std`, default `ddof=0`), while the threshold summary and
Table 5.1 use sample standard deviations. The correction does not change these
aggregation definitions.

| Quantity | Previous PDF | Corrected result |
|---|---:|---:|
| HMM mean F1 | 0.863 | **0.831** |
| HMM F1 standard deviation (population) | 0.090 | **0.128** |
| HMM mean precision | 0.855 | **0.838** |
| HMM mean recall | 0.895 | **0.852** |
| HMM mean start offset | 2.30 s | **2.40 s** |
| Threshold mean F1 | 0.727 | **0.728** |
| Threshold F1 standard deviation (sample) | 0.084 | **0.102** |
| Threshold mean start offset | 1.60 s | **2.10 s** |
| Absolute mean-F1 improvement | 0.136 | **0.103** |
| Relative mean-F1 improvement | 18.7% | **14.1%** |
| HMM wins / losses / ties | 45 / 5 / 0 | **39 / 10 / 1** |
| HMM convergence | 50/50 | **50/50** |

Replace Section 5.2's headline-results paragraph on p.34 with:

> On the fifty-seed controlled directional-pressure ensemble, the hidden Markov
> recovery model converged on every seed and returned a mean \(F_1\) of 0.831
> (standard deviation 0.128), mean precision 0.838, mean recall 0.852, and a mean
> start-time difference of 2.40 s. The rate-and-imbalance threshold method was
> evaluated on the same fifty simulation seeds and returned a mean \(F_1\) of
> 0.728 (standard deviation 0.102) and a mean start-time difference of 2.10 s. The
> hidden Markov model improves mean \(F_1\) by 0.103, or 14.1%, and remains above
> the specified benchmark of \(F_1\geq0.78\). Its score is higher on 39 seeds,
> lower on ten and equal on one. Figure 5.2 shows this paired comparison, while
> Figure 5.1 illustrates the smoothed posterior on a median-\(F_1\) example.

Update the Table 5.1 note to:

> Five-second windows put start-time differences on a five-second grid. Across
> 50 seeds, the threshold method has 35, 10, 4 and 1 cases at 0, 5, 10 and
> 15 seconds; the HMM has 38, 5, 3, 3 and 1 cases at 0, 5, 10, 15 and
> 20 seconds. In the final row, \(\pm\) gives the sample standard deviation
> across seeds.

Apply the same corrected numbers only to the controlled-result repetitions:

| Location | Required change |
|---|---|
| Abstract, p.ii | Rounded HMM/threshold means **0.83 / 0.73**, not 0.86 / 0.73. |
| Section 1.4, p.4 | HMM mean/SD and threshold mean. |
| Section 5.2, p.34 | Full paragraph above; representative figure label and caption. |
| Section 5.4, p.35 | Comparator identifier "reported mean F1 = 0.727" becomes **0.728**. |
| Section 5.7, p.36 | Table 5.1, onset note, Figure 5.2 and its win/loss/tie description. |
| Section 5.8 opening, p.37 | Controlled reference **0.831**; retain the separate fitted-background results. |
| Section 7.1, p.46 | Means, SD and start-time difference. |
| Section 7.5, p.49 | Controlled reference **0.831**. |
| Section 8.1, p.50 | Means and SD. |
| Section 8.2, p.51 | **0.831 against 0.728**. |

The longer numerical values are available in the saved JSON. A global replacement
of short strings such as `0.73` is inappropriate: other studies are unchanged.

## Quoting: matched comparison in Section 6.4

The corrected matched comparison changes a substantive conclusion: **the maximum
tested sensitivity now passes the aggregate fill-retention criterion**. The
0.50 criterion and the experiment settings have not been changed.

| Quantity at s=1 | Previous PDF | Corrected result |
|---|---:|---:|
| Equal-seed baseline five-second movement | 1.563681 ticks | **1.561867 ticks** |
| Equal-seed signal-aware movement | 1.423170 ticks | **1.427949 ticks** |
| Equal-seed reduction | 0.140511 ticks | **0.133918 ticks** |
| Seeds with lower movement | 28/30 | **26/30** |
| Median fill retention | 0.427 | **0.572** |
| Mean active-period posterior | 0.886 | **0.852** |
| Valid baseline / signal-aware five-second records | 25,991 / 14,083 | **26,236 / 15,303** |
| Pooled baseline / signal-aware means | 1.565253 / 1.422921 ticks | **1.563520 / 1.427498 ticks** |
| Seeds with lower fill proportion | 30/30 | **30/30** |

Replace the three result bullets in Section 6.4, p.41, with:

> - Criterion 1 (absolute-markout reduction): passes. The equal-seed mean
>   absolute five-second movement falls from 1.562 to 1.428 ticks, a reduction
>   of 0.134 tick (8.6%), and is lower on 26 of 30 seeds (86.7%).
> - Criterion 2 (fill-proportion retention): passes. The median
>   signal-aware/baseline fill-proportion ratio is 0.572, above the 0.50
>   threshold.
> - Criterion 3 (signal precondition): passes. The mean posterior over true
>   active windows is 0.852.

In the following Figure 6.1 explanation and its caption, use the updated record
counts and pooled/equal-seed means above. The distinction between pooled fill
weighting and equal-seed weighting remains correct, as does the statement that
fill proportions fall in all thirty simulations.

## Quoting frontier and narrative replacements

| s | Movement reduction | Median fill retention | Seeds with lower movement | Seeds retaining at least half |
|---:|---:|---:|---:|---:|
| 0.10 | 1.30% | 92.8% | 25/30 | 30/30 |
| 0.20 | 2.94% | 86.2% | 29/30 | 30/30 |
| 0.35 | 4.49% | 77.6% | 28/30 | 30/30 |
| 0.50 | 6.05% | 70.6% | 28/30 | 27/30 |
| 0.75 | 8.16% | 61.8% | 28/30 | 21/30 |
| 1.00 | 8.57% | 57.2% | 26/30 | 18/30 |

The aggregate curve remains monotone across the tested settings: greater
sensitivity gives a larger mean movement reduction and lower median retention.
**The 50% aggregate floor is not crossed anywhere in the tested grid.** This does
not mean that every seed retains half its fills, nor that an optimum beyond the
tested grid has been identified.

### Section 6.1, p.39

Replace the two sentences beginning "Third, that curve contains..." and ending
"falls below the fill-retention criterion" with:

> Third, all six nonzero tested sensitivities reduce mean absolute movement while
> meeting the aggregate fill-retention criterion. The largest tested setting
> produces the largest reduction, while retaining a median 57.2% of baseline
> fills.

### Section 6.5, p.43

Replace the result discussion from "Across the tested settings the trade-off is..."
through "the absolute-movement reduction levels off" with:

> Across the tested settings, stronger posterior widening reduces mean absolute
> five-second movement and retains fewer fills. All six nonzero sensitivities meet
> the aggregate 0.50 fill-retention criterion. At \(s=0.50\), mean absolute
> movement falls by 6.05% with median fill retention of 70.6%; the more
> conservative \(s=0.20\) gives a 2.94% reduction with 86.2% retention. The
> largest tested reduction occurs at \(s=1\): 8.57%, with median retention of
> 57.2%. At that setting, 26 of 30 seeds have lower mean absolute movement and
> 18 retain at least half their baseline fill proportion. The movement-fill
> trade-off therefore remains, but the aggregate fill floor is not crossed within
> the tested range.

### Section 6.6, p.44

Replace its opening paragraph, which currently says s=1 lies outside the feasible
region, with:

> The execution-value frontier organises the matched-seed results under the
> criteria fixed before evaluation. All six nonzero tested sensitivities reduce
> mean absolute movement while meeting the aggregate fill-retention criterion.
> The largest tested setting, \(s=1\), reduces mean absolute five-second movement
> by 8.57% and retains a median 57.2% of baseline fills. It therefore satisfies
> all three criteria, although fill retention varies across seeds. The frontier
> measures the cost of stronger posterior widening within this quoting family;
> its aggregate fill floor remains unbroken over the tested range.

### Section 7.3, p.47

Replace the sentence beginning "Its execution-modelling implication..." with:

> Its execution-modelling implication is the frontier in Section 6.5: stronger
> posterior widening exchanges fill retention for lower absolute post-fill
> movement, while all six nonzero tested settings remain above the aggregate
> fill floor.

### Short result statement for the repeated summaries

Use this in place of each claim that only four settings are feasible or that
s=0.50 is the best feasible setting:

> All six nonzero tested sensitivities reduce mean absolute five-second movement
> while meeting the aggregate fill-retention criterion. The largest tested
> reduction occurs at \(s=1\), at 8.57% with median fill retention of 57.2%.

The corresponding set is
\(s\in\{0.10,0.20,0.35,0.50,0.75,1.00\}\). Apply the replacement at:

| Location | Existing statement requiring correction |
|---|---|
| Section 1.5, p.5 | Four nonzero settings pass; s=0.75 and s=1 fail. |
| Section 6.1, p.39 | Largest setting falls below the fill criterion. |
| Section 6.4, p.41 | Criterion 2 fails; old counts, means and probabilities. |
| Figure 6.1 caption, p.42 | Old record counts and pooled/equal-seed means. |
| Section 6.5, p.43 | Four feasible settings; s=0.50 best; the fill constraint binds. |
| Table 6.1 and Section 6.6, p.44 | Old table values; s=1 outside the feasible region. |
| Section 7.1, p.46 | Four settings pass; 6.33% reduction with 60.1% retention is best. |
| Section 7.3, p.47 | Fill retention is the binding constraint over this grid. |
| Section 8.1, p.50 | Four settings pass; s=0.50 is the best feasible setting. |
| Section 8.3, p.52 | Only four nonzero settings meet the criterion. |
| Section 8.4, p.52, first future direction | The repeated four-setting set and s=0.75/s=1 failures. |
| Appendix C current summary, p.76 | The repeated four-setting set and excluded s=0.75/s=1. |

The future control programme can remain in place. What changes here is the
measured location of the present curve relative to its threshold, not the
motivation for studying that control problem. This hand-off concerns the
simulator-dependent corrections only.

## Corrected figure captions

These are the captions in the refreshed
[figure manifest](figures/figure_manifest.json). They avoid carrying forward a
description of the old representative trajectory. Figure 5.1 now shows seed 19
with F1 = 0.846, not seed 1.

### Figure 5.1, p.34

> Full-sequence probability of the fitted higher-rate activity state on the
> deterministic median-F1 example (seed 19), with the known directional-pressure
> period shaded. The state is identified by fitted activity rate; the displayed
> probability uses the complete sequence.

### Figure 5.2, p.36

> Paired F1 scores for the rate-and-imbalance threshold method and HMM on the same
> 50 simulation seeds. The HMM scores higher on 39 seeds, lower on 10, and ties
> on 1; the means are 0.728 and 0.831.

### Figure 6.1, p.42

> The cumulative curves pool 26,236 and 15,303 valid post-fill records at s=0
> and s=1. At movement x, the curve gives the percentage of valid fills with
> absolute five-second movement no greater than x. The right panel shows fills
> as a percentage of bid- and ask-side quote opportunities over each complete
> 600-second simulation. Thirty matched simulations are shown. Absolute movement
> measures magnitude; signed markout retains economic direction.

### Figure 6.2, p.43

> Thirty matched simulations at seven posterior-sensitivity settings. The upper
> panel shows fill proportions relative to each seed's s=0 baseline, with
> medians; the lower shows changes from s=0 in mean absolute five-second
> movement in ticks, with equal-seed means. The horizontal references mark half
> of baseline fills and zero movement change.

## Spread scaling: Table 4.1 and Section 4.4, p.18

Replace the entire table from [tab_spread_scaling.tex](tables/tab_spread_scaling.tex).
For checking the imported values, the corrected table is:

| T (s) | Mean observed events | Mean spread | Median | p90 | p99 | Maximum | Final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 68.4 | 6.10 | 5.87 | 10.00 | 11.39 | 11.70 | 8.83 |
| 60 | 140.4 | 8.47 | 8.22 | 14.24 | 17.13 | 17.83 | 12.33 |
| 120 | 285.8 | 11.82 | 12.17 | 19.50 | 22.56 | 23.47 | 16.23 |
| 240 | 579.1 | 15.15 | 15.53 | 24.22 | 28.19 | 29.77 | 18.90 |
| 480 | 1173.8 | 18.22 | 17.63 | 31.49 | 37.96 | 40.27 | 27.80 |
| 960 | 2366.6 | 26.74 | 25.40 | 48.33 | 57.80 | 61.23 | 36.83 |

Spread quantities are in ticks. The six horizons and thirty seeds per horizon
are unchanged. Replace the first two sentences below Table 4.1 with:

> Log-log fits of the form \(\mathrm{field}=aN_{\mathrm{obs}}^b\) give
> exponents \(b=0.401\) for the mean spread, 0.398 for the median, 0.422 for
> the 90th percentile, 0.433 for the 99th, 0.442 for the maximum, and 0.392
> for the final spread. All six estimates lie below the unrestricted
> random-walk value of 0.5.

The unrounded exponents range from 0.392350 to 0.441844. This preserves the
reported horizon dependence under the minimum-spread filter; it does not turn the
controlled spread distribution into an empirical calibration.

## Runtime benchmark

The refreshed benchmark retains the original kernel fixtures: dimension four,
a ten-second horizon, and twenty runs with seeds 42-61 for each family. Runtime
is a local wall-clock measurement, not a speed guarantee on another machine.

| Kernel family | Previous PDF median | Corrected median |
|---|---:|---:|
| Single exponential | 1.93 ms | **2.08 ms** |
| Sum of two exponentials | 6.44 ms | **6.65 ms** |
| Power law with cutoff | 2.40 ms | **2.74 ms** |

In Section 3.1, p.13, replace the numerical benchmark passage with:

> On the fixed benchmark (median wall-clock time for a 10 s, four-dimensional
> simulation over 20 runs), the families cost 2.08 ms for a single exponential,
> 6.65 ms for two exponentials, and 2.74 ms for the power law with cutoff. The
> power-law configuration runs at a similar scale to the single-exponential
> configuration and is about 2.4 times faster than the local two-term
> sum-of-exponentials configuration. These timings compare the three local
> configurations.

Retain the following existing clarification that this is not a benchmark of the
Rambaldi implementation. In B.7, p.71, replace the two sentences beginning
"These implementations support the kernel comparison..." with:

> The fixed benchmark in Chapter 3 gives median runtimes of 2.08 ms for a
> single exponential, 6.65 ms for two exponentials, and 2.74 ms for the power
> law with cutoff. The sum-of-exponentials evaluation carries the additional
> cost of its decay components.

The precise medians are 2.079646103, 6.651459029 and 2.741603996 ms; the
sum-exponential/power-law ratio is approximately 2.43. Use the rounded values
consistently in prose. The three separate seed-42 structural checks report
branching ratio 0.20 and zero crossed observed books for each configuration;
these are smoke checks, not newly generated fifty-seed baseline ensembles.

## Final Overleaf check

1. Import the four corrected figure PDFs and three corrected table bodies above.
2. Update B.2, B.7 and the public tag URL; retain the unchanged analytical formulas.
3. Reconcile all recovery repetitions and the six-setting quoting conclusion using
   the page inventories in this document. In particular, remove the old statements
   that the s=1 point fails the aggregate fill criterion or lies outside the feasible
   region.
4. Refresh the Table 5.1 note, four figure captions, spread exponents and local
   runtime comparison. Preserve the original seed grids and evaluation criteria.
5. Compile and check that the imported assets, captions and surrounding prose all
   describe the same corrected results. These changes do not require replacing empirical
   EM/CLS, fitted-model validation or twelve-type results.

No Overleaf source or dissertation PDF was edited as part of this code release.
