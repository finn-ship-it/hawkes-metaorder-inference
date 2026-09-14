# Submission figures and method provenance

The `msc-submission-2026` controlled-simulator correction and the required
dissertation updates are recorded in
[`CONTROLLED_SIMULATOR_CORRECTION_V2.md`](CONTROLLED_SIMULATOR_CORRECTION_V2.md).
Its regenerated controlled results supersede the corresponding older numbers
in the historical artifact inventory. Earlier development history is archived separately.

The nine current figures are listed in `figures/figure_manifest.json`, including
their titles, suggested captions, input hashes and renderer hashes. Existing PDF
filenames are retained so that Overleaf references continue to work.

## Plot-only reproduction

From the repository root, using Python 3.12:

```sh
python -m pip install -r dissertation_artifacts/figures/requirements-plots.txt
python dissertation_artifacts/figures/render_submission.py
python dissertation_artifacts/figures/render_submission.py --check
```

This route reads stored results and regenerates figures only. It leaves tables,
fitted parameters, simulations and recovery results unchanged. The three files
in `data/plot_inputs/` contain only the EM coefficients and Fano summaries needed
by the plots, with hashes identifying the original frozen sources. The residual
input contains sorted Q-Q coordinates, not raw quote records. The additional
latent-book input contains previously stored simulation probabilities.

## Empirical clock and sample days

HistData specifies fixed Eastern Standard Time (UTC-05:00), without daylight
saving: https://www.histdata.com/f-a-q/ (checked 10 September 2026).
The original import parses the provider timestamp with
`pd.to_datetime(..., format="%Y%m%d %H%M%S%f")`, preserving its clock. The event
builder carries that timestamp through, and both EM and CLS select midnight-to-
midnight samples on that clock. The stored primary sample is therefore the
provider day 20 July 2021, with 97,952 events; the secondary sample is the provider
day 19 July 2021. Each spans 86,400 seconds. Expressed in UTC, the primary window
is 05:00 on 20 July to 05:00 on 21 July. The samples and fitted results are retained.
Earlier descriptions of these samples as UTC calendar days should read
"provider-time days (fixed EST)". Session labels based on the unconverted clock
also require this convention; they are not used to fit the reported full-day models.

## Conditional least squares

The empirical comparator is `src/simulator/layer1_konark_cls.py`, based on the
Kirchner conditional-least-squares approach and informed by Konark Jain's
`src/fit/ConditionalLeastSquaresLogLin.py` in `konqr/lobSimulations`.
The upstream file constructs a combined linear/logarithmic lag grid and regresses
binned event counts. The local adaptation uses all fine time bins, four quote-
event types, explicit bin-width weighting and a constrained quadratic solve.
These adaptations should be attributed separately from the general method.

The saved empirical result uses 0.5-second fine bins, 42 lag bins spanning
0-500 seconds, and the OSQP solver. Its spectral radius is 0.9783592019942144.
The appendix excerpt should follow this implementation and configuration, not
an older equal-bin or unconstrained least-squares example. The upstream code file
is the verified implementation reference; no unverified thesis page is supplied.

Figure 4.1 compares this saved CLS result with converged EM. Its full stored
branching matrices integrate CLS over its fitted 0-500 second support and EM to
infinity. They reproduce the reported spectral radii, 0.9784 and 0.9373.
Two aligned curve panels show the first ten seconds of bid-up to bid-down and
bid-up to ask-up excitation. They are selected from the full matrices as the
strongest shared off-diagonal interaction and the largest remaining difference.
The near-zero third example has been removed from the display.

## Retired one-sided estimate

`src/simulator/layer1_nonparametric.py` is a historical one-sided Volterra
experiment, not the stationary Bacry-Muzy Wiener-Hopf estimator. Its saved output
is excluded from the submission figure set and empirical conclusions. Historical
identifiers are retained for compatibility; their presence is not a method claim.
The previous 1.445 result and proposed histogram-bias correction are not used.
Frozen historical files remain unchanged as records of earlier work.

## Runtime comparison

`src/simulator/tests/benchmarks/bench_kernels.py` compares three local simulator
configurations over 20 runs, each with a 10-second horizon and four event types.
The sum-of-exponentials configuration has two terms. The corrected medians in
`data/kernel_benchmark.csv` are 0.0020796461030840874 seconds for the single
exponential, 0.006651459028944373 seconds for the two-term sum, and
0.0027416039956733584 seconds for the power-law-with-cutoff configuration.
The sum/power-law ratio is about 2.43. These were rerun under Python 3.12.13 and
NumPy 2.4.6 on macOS arm64; timings are machine-dependent.
This is a comparison of these local configurations, not a
benchmark of Rambaldi and colleagues' implementation or their power-law approximation.

## Overleaf handoff

Figure 4.1's caption and surrounding comparison must refer to CLS and converged
EM. Its short title is "Hawkes excitation kernels for EUR/USD"; the caption in the figure
manifest now explains the full branching matrices and the two curve panels.
The agreed titles in `FIGURE_TABLE_TITLES.md` are applied to all nine plot assets
and the figure manifest. Use the same titles for Overleaf captions/list entries;
table-title recommendations remain for manual Overleaf use.
Figure 4.5 is titled "Stronger pressure is easier to detect". Its pressure
targets are native market orders and in-spread limit orders on alternating book
sides; the second panel also includes top-of-book limit orders and cancellations.
The horizontal rate multiplier is m=1+delta_p, with existing values 2, 3, 5, 8, 10.
Tables and dissertation source are maintained separately in Overleaf.
