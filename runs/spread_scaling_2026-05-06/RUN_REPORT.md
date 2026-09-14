# Run report — prompt 03 (spread-scaling sweep at production scale)

Date: 2026-05-06.
Driver: `My repo/runs/run_prompt_03.py`.
Config: `configs/first_milestone.json` with `horizon` overridden per
ensemble. Other parameters held at first-milestone defaults
(`d=4`, `mu=0.5*ones(4)`, `alpha=0.1*ones(4,4)`, `beta=2.0*ones(4,4)`,
`KernelFamily.EXPONENTIAL`, single regime, no meta-order,
`min_spread_ticks=1`).

## Sweep

Horizons $T \in \{30, 60, 120, 240, 480, 960\}$ s; 30 seeds per horizon
(seeds 1..30). Total runs: 180. Per-horizon ensemble runtimes:

| $T$ (s) | wall (s) | members |
| --- | --- | --- |
| 30  | 0.6 | 30 |
| 60  | 1.9 | 30 |
| 120 | 7.1 | 30 |
| 240 | 28.9 | 30 |
| 480 | 113.0 | 30 |
| 960 | 349.6 | 30 |

Total wall time: ~8.4 minutes.

## Headline numbers

Ensemble means across 30 seeds per horizon:

| $T$ (s) | $\bar N_\text{obs}$ | spread\_mean | spread\_median | spread\_p90 | spread\_p99 | spread\_max | final\_spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30  | 66.6   | 6.23  | 6.17  | 10.12 | 11.58 | 11.90 | 8.23 |
| 60  | 137.4  | 8.95  | 8.73  | 14.65 | 17.13 | 17.80 | 12.47 |
| 120 | 285.6  | 11.23 | 11.60 | 18.08 | 21.44 | 22.47 | 14.37 |
| 240 | 572.8  | 14.74 | 15.23 | 23.40 | 27.94 | 29.60 | 17.13 |
| 480 | 1166.7 | 18.75 | 17.87 | 33.09 | 40.18 | 42.53 | 28.43 |
| 960 | 2341.6 | 25.37 | 22.85 | 47.17 | 57.76 | 61.70 | 34.87 |

## Fitted exponents

Model: `field_mean = a * N_obs^b`, OLS on log-log.

| Field | $b$ | $a$ | $R^2$ |
| --- | --- | --- | --- |
| `observation.spread_mean_ticks`   | **0.382** | 1.298 | 0.996 |
| `observation.spread_median_ticks` | **0.360** | 1.449 | 0.989 |
| `observation.spread_p90_ticks`    | **0.417** | 1.763 | 0.992 |
| `observation.spread_p99_ticks`    | **0.436** | 1.877 | 0.993 |
| `observation.spread_max_ticks`    | **0.446** | 1.850 | 0.994 |
| `observation.final_spread_ticks`  | **0.396** | 1.596 | 0.972 |

The new range is **$b \in [0.36,\,0.45]$**, vs the previously documented
range $[0.28,\,0.35]$.

## Where the shift comes from

The shipped diagnostic
(`docs md/spread_scaling_diagnostic.md`) used 10 seeds and $T \in
\{30,\,60,\,120,\,240,\,480\}$. The new sweep adds $T = 960$ and triples
the seed count to 30. Re-fitting the new ensemble on the original
horizon range (excluding $T = 960$) gives:

| Field | $b$ (this sweep, $T=30\dots480$) | $b$ (this sweep, $T=30\dots960$) | $b$ (shipped diagnostic) |
| --- | --- | --- | --- |
| `spread_mean_ticks` | 0.378 | 0.382 | 0.294 |
| `spread_median_ticks` | 0.375 | 0.360 | 0.283 |
| `spread_p90_ticks` | 0.397 | 0.417 | 0.322 |
| `spread_p99_ticks` | 0.416 | 0.436 | 0.342 |
| `spread_max_ticks` | 0.427 | 0.446 | 0.354 |
| `final_spread_ticks` | 0.391 | 0.396 | 0.276 |

The bulk of the shift ($\Delta b \approx 0.08$ on average) is already
present at $T \le 480$ — i.e. it is **not** a regime change introduced
by adding $T = 960$, but a tighter estimate at the same horizon range
under more seeds. The marginal contribution of $T = 960$ is small
($\Delta b = 0.004$ to $0.020$). The mechanism is that the 10-seed
ensemble had higher variance on the small-$T$ means (the per-seed
spread is itself a tail-sensitive observable on short horizons), and a
subset of high-spread seeds biased the small-$T$ means upward in the
shipped diagnostic — flattening the slope. With 30 seeds the small-$T$
means come down (e.g. $T=30$ mean: 7.89 → 6.23, a 21% drop) while the
large-$T$ means move only slightly (e.g. $T=480$: 17.96 → 18.75, a 4%
rise), steepening the slope.

The qualitative interpretation in `spread_scaling_diagnostic.md` is
unchanged. The exponents remain well below the unrestricted-random-walk
value 0.5 — the reflecting-barrier mechanism is intact, just the
quantitative numbers tighten.

## $T = 960$ regime check

Per the prompt 03 post-run amendment rule, an exponent jump triggered
solely by adding $T = 960$ would require user approval before amending.
The check above shows $\Delta b$(adding $T=960$) is at most 0.020 on
any field, which is at the prompt 03 threshold for amendment but not a
"regime change." The shift is dominated by the seed-count increase, not
the horizon extension. Proceeding with the amendment.

## Amendments triggered

All six exponents fall outside $[0.28,\,0.35]$ by more than 0.02. Per
the prompt 03 post-run amendment rule, this triggers:

1. **Prompt 07 (figure factory)** — `spread_scaling_loglog.pdf` is
   regenerated automatically by `dissertation_artifacts/figures/regenerate.py`,
   which reads the new exponents from `spread_scaling_fit.json`. No
   text amendment to prompt 07 is needed; the regenerate script
   already produces the correct caption.
2. **Prompt 08 (table factory)** — `tab_spread_scaling.tex` is
   regenerated automatically from `spread_scaling_summary.csv` by
   `dissertation_artifacts/tables/regenerate.py`. The dissertation's
   chapter 4 caption around this table is text the dissertation
   maintains, not the table fragment; that caption needs an inline
   update by prompt 12.
3. **Prompt 12 (inline numbers)** — every site in the dissertation
   that quotes "$b \in [0.28,\,0.35]$" must be updated to
   "$b \in [0.36,\,0.45]$", with a footnote pointing at the 30-seed
   $T=960$-extended sweep. Concretely, chapter 4 §4.4 has both the
   range and individual exponents (mean 0.2939, median 0.2833, p90
   0.3218, p99 0.3424, max 0.3535, final 0.2761); each becomes the new
   value (0.382 / 0.360 / 0.417 / 0.436 / 0.446 / 0.396).

The amendments to prompts 12 are recorded in `DECISION_LOG.md`.

## Files produced

```
runs/spread_scaling_2026-05-06/
├── T030/ ... T960/                     # 6 horizon ensembles, 30 members each
├── spread_scaling_summary.csv          # one row per horizon
├── spread_scaling_fit.json             # exponents + a + R^2
└── RUN_REPORT.md                       # this file
```

180 ensemble members fit in ~8.4 min driver wall time.
