# Figure and table titles

Use a short title to name what the reader sees, followed by a caption explaining
how to read it. Keep the experimental detail, units and sample sizes in that
explanation. Useful result-led titles, such as Figure 4.5, can remain.

All nine figure titles below are applied to the current plot assets and figure
manifest. Use the same short titles for captions/list entries in Overleaf.
The table titles remain recommendations for manual use; no tables are changed.

## Figures

| Figure | Applied title |
| --- | --- |
| 4.1 | Hawkes excitation kernels for EUR/USD |
| 4.2 | Time-rescaling residuals of the EUR/USD EM fit |
| 4.3 | Empirical and simulated count dispersion |
| 4.4 | Latent-book projection and recovery under weak pressure |
| 4.5 | Stronger pressure is easier to detect |
| 5.1 | Recovering a known directional-pressure period |
| 5.2 | Recovery accuracy across 50 controlled simulations |
| 6.1 | Post-fill movement and fill proportions |
| 6.2 | Execution-value frontier |

Figure 4.3's caption identifies the Fano factor. Figure 6.1's caption and axes
retain the absolute-movement definition and five-second horizon. Fill proportions
in Figure 6.1 and retention relative to baseline in Figure 6.2 remain distinct.

## Tables

| Table | Recommended short title |
| --- | --- |
| 3.1 | Controlled simulator configurations |
| 4.1 | Controlled spread-scaling sweep |
| 4.2 | EUR/USD data summary |
| 4.3 | Model comparison for EUR/USD quote events |
| 4.4 | Forward simulation of EUR/USD quote activity |
| 4.5 | Twelve-to-four projection |
| 4.6 | Selected points on the recoverable frontier |
| 5.1 | Rate-and-imbalance threshold recovery |
| 5.2 | Recovery on the fitted EUR/USD background |
| 6.1 | Execution-value frontier |

The existing explanatory captions and numerical contents stay in place.
In particular, Table 5.1 identifies seeds 1-5 and the 50-seed summary; Table 5.2
identifies the paired pressure/control study. Figure 6.2 and Table 6.1 can share
a title because they are graphical and numerical views of the same frontier.

For a short list entry and a fuller caption, use the usual LaTeX form:

```tex
\caption[Short title]{Short title. Explanation of the figure or table.}
```

Keep the existing `\label{...}` and PDF filenames so cross-references and
figure inclusion paths continue to work.
