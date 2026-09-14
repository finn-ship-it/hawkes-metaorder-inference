"""Regenerate dissertation figures under dissertation_artifacts/figures/.

Each figure is saved as PDF (vector) plus a one-line JSON entry in
figure_manifest.json. Idempotent: running twice produces byte-identical
PDFs (with deterministic figure_manifest entries except last_generated_utc).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO / "dissertation_artifacts" / "data"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

# Match the dissertation's lmodern preamble where reasonable, but stay with matplotlib defaults.
plt.rcParams["font.family"] = "serif"
plt.rcParams["pdf.fonttype"] = 42  # avoid Type-3 fonts
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGS_DIR / name
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _load_ensemble_summary(cfg_name: str) -> dict:
    path = (
        REPO / "runs" / "ensembles_2026-05-06" / f"{cfg_name}_50seeds"
        / f"ensemble_{cfg_name}_50seeds" / "ensemble_summary.json"
    )
    return json.load(open(path))


# ---------------------------------------------------------------------------
# 1. branching_ratio_invariant.pdf
# ---------------------------------------------------------------------------


def fig_branching_ratio_invariant() -> tuple[Path, dict]:
    cfgs = ["first_milestone", "two_regime_smoke", "meta_order_smoke"]
    rho_means = []
    rho_stds = []
    sources = []
    for c in cfgs:
        es = _load_ensemble_summary(c)
        rec = es["aggregate"]["endogeneity.spectral_radius"]
        rho_means.append(rec["mean"])
        rho_stds.append(rec["std"])
        sources.append(str(REPO / "runs" / "ensembles_2026-05-06" / f"{c}_50seeds" /
                           f"ensemble_{c}_50seeds" / "ensemble_summary.json"))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(cfgs))
    # Plot error bars even though std is exactly 0 — the visualization records
    # the invariance claim. Use a tiny vertical band for visibility.
    bars = ax.bar(x, rho_means, yerr=[max(s, 1e-6) for s in rho_stds], capsize=4,
                  color="#4C72B0", edgecolor="black", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cfgs])
    ax.set_ylabel(r"Spectral radius $\rho(\Phi)$")
    ax.set_ylim(0, 0.3)
    ax.axhline(0.2, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
               label=r"Configuration $\rho=0.2$")
    ax.legend(loc="upper right")
    for xi, m, s in zip(x, rho_means, rho_stds):
        ax.text(xi, m + 0.005, f"{m:.4f}\n$\\sigma={s:.0e}$",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Branching-ratio invariant across the three configurations\n"
                 "(50-seed ensembles; $\\rho$ depends on $\\alpha/\\beta$, not on seed)")
    path = _save(fig, "branching_ratio_invariant.pdf")
    return path, {
        "path": str(path.relative_to(REPO)),
        "source_data": sources,
        "caption": "Spectral radius $\\rho(\\Phi)$ across the three shipped configurations, ensemble of 50 seeds. The radius depends only on $\\alpha_{ij}/\\beta_{ij}$ and is invariant under reseeding.",
    }


# ---------------------------------------------------------------------------
# 2. spread_scaling_loglog.pdf
# ---------------------------------------------------------------------------


def fig_spread_scaling_loglog() -> tuple[Path, dict] | None:
    fit_path = DATA_DIR / "spread_scaling_fit.json"
    csv_path = DATA_DIR / "spread_scaling_summary.csv"
    if not (fit_path.exists() and csv_path.exists()):
        return None
    fit = json.load(open(fit_path))
    rows = list(csv.DictReader(open(csv_path)))
    fields = (
        "observation.spread_mean_ticks",
        "observation.spread_median_ticks",
        "observation.spread_p90_ticks",
        "observation.spread_p99_ticks",
        "observation.spread_max_ticks",
        "observation.final_spread_ticks",
    )
    labels = ("mean", "median", "p90", "p99", "max", "final")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    Ns = np.array([float(r["N_obs_mean"]) for r in rows])
    for f, lab in zip(fields, labels):
        ys = np.array([float(r[f + ".mean"]) for r in rows])
        b = fit["fits"][f]["b"]
        ax.loglog(Ns, ys, "o-", label=f"{lab}  ($b={b:.3f}$)", markersize=5)
    ax.set_xlabel(r"$N_\mathrm{obs}$ (ensemble mean)")
    ax.set_ylabel("Spread field (ticks)")
    ax.set_title("Spread-scaling diagnostic on the first-milestone configuration\n"
                 "(30 seeds per horizon; $T \\in \\{30,60,120,240,480,960\\}$)")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    path = _save(fig, "spread_scaling_loglog.pdf")
    return path, {
        "path": str(path.relative_to(REPO)),
        "source_data": [str(fit_path.relative_to(REPO)), str(csv_path.relative_to(REPO))],
        "caption": (
            "Log-log spread-distribution fields against ensemble-mean observed events $N_\\mathrm{obs}$, "
            "30 seeds per horizon, $T \\in \\{30,60,120,240,480,960\\}$ s on \\texttt{first\\_milestone}. "
            "Fitted exponents are reported in the legend."
        ),
    }


# ---------------------------------------------------------------------------
# 3. recovery_f1_distribution.pdf
# ---------------------------------------------------------------------------


def _recovery_rows() -> list[dict] | None:
    p = DATA_DIR / "recovery_summary.csv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p)))


def fig_recovery_f1_distribution() -> tuple[Path, dict] | None:
    rows = _recovery_rows()
    if not rows:
        return None
    f1s = np.array([float(r["f1"]) for r in rows])
    mean = float(np.mean(f1s))
    std = float(np.std(f1s, ddof=0))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(f1s, bins=12, color="#55A868", edgecolor="black", alpha=0.85)
    ax.axvline(mean, color="black", linestyle="--", label=f"mean $\\bar{{F1}}={mean:.3f}$")
    ax.set_xlabel("F1")
    ax.set_ylabel("Number of seeds")
    ax.set_title("Per-seed F1 on the meta-order smoke benchmark, 50 seeds\n"
                 f"$\\bar{{F1}}={mean:.3f}$, $\\sigma={std:.3f}$")
    ax.legend()
    path = _save(fig, "recovery_f1_distribution.pdf")
    return path, {
        "path": str(path.relative_to(REPO)),
        "source_data": [str((DATA_DIR / "recovery_summary.csv").relative_to(REPO))],
        "caption": f"Per-seed F1 on the shipped meta-order smoke configuration, 50 seeds; mean $\\bar{{F1}}={mean:.3f}$, std ${std:.3f}$.",
    }


# ---------------------------------------------------------------------------
# 4. recovery_delay_distribution.pdf
# ---------------------------------------------------------------------------


def fig_recovery_delay_distribution() -> tuple[Path, dict] | None:
    rows = _recovery_rows()
    if not rows:
        return None
    delays = np.array([float(r["mean_detection_delay_seconds"]) for r in rows])
    mean = float(np.mean(delays))
    std = float(np.std(delays, ddof=0))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bin_edges = np.arange(-2.5, max(delays) + 5.0, 2.5)
    ax.hist(delays, bins=bin_edges, color="#C44E52", edgecolor="black", alpha=0.85)
    ax.axvline(mean, color="black", linestyle="--", label=f"mean ${mean:.2f}$ s")
    ax.set_xlabel("Mean detection delay (s)")
    ax.set_ylabel("Number of seeds")
    ax.set_title("Per-seed mean detection delay, 50 seeds\n"
                 f"$\\bar{{\\mathrm{{delay}}}}={mean:.2f}$ s, $\\sigma={std:.2f}$ s")
    ax.legend()
    path = _save(fig, "recovery_delay_distribution.pdf")
    return path, {
        "path": str(path.relative_to(REPO)),
        "source_data": [str((REPO / "runs" / "recovery_meta_order_smoke_2026-05-06" / "recovery_summary.csv").relative_to(REPO))],
        "caption": f"Per-seed mean detection delay (s) on the shipped meta-order smoke configuration, 50 seeds; mean ${mean:.2f}$ s, std ${std:.2f}$ s.",
    }


# ---------------------------------------------------------------------------
# 5. kernel_decay_eurusd_sumexp.pdf
# ---------------------------------------------------------------------------


def fig_kernel_decay_eurusd_sumexp() -> tuple[Path, dict] | None:
    p = REPO.parent / "Data" / "Data output" / "EURUSD output" / "layer1" / "layer1_params_sumexp_4type.json"
    if not p.exists():
        return None
    fit = json.load(open(p))
    # alpha_all has shape (R, M, M) per the sum-of-exp fitter
    alpha_all = np.asarray(fit["alpha_all"], dtype=float)
    betas = np.asarray(fit["betas"], dtype=float)
    R, M, _ = alpha_all.shape
    type_names = ["bid_up", "bid_down", "ask_up", "ask_down"][:M]

    t_grid = np.linspace(0.001, 5.0, 400)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:M]
    for i in range(M):
        # Self-excitation kernel: phi_ii(t) = sum_r alpha_iir * exp(-beta_r * t)
        a_i = alpha_all[:, i, i]  # length R
        phi = np.zeros_like(t_grid)
        for r, beta_r in enumerate(betas):
            phi += a_i[r] * np.exp(-beta_r * t_grid)
        ax.semilogy(t_grid, phi, label=type_names[i], color=colors[i])
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel(r"$\phi_{ii}(t)$")
    ax.set_title("EURUSD sum-of-exp 4-type self-excitation kernel\n"
                 "(primary-day single-UTC fit, 2021-07-20; $\\beta$-grid $=\\{0.1, 1, 10, 100\\}$ /s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = _save(fig, "kernel_decay_eurusd_sumexp.pdf")
    return path, {
        "path": str(path.relative_to(REPO)),
        "source_data": [str(p.relative_to(REPO.parent))],
        "caption": "Fitted self-excitation kernel $\\phi_{ii}(t)$ for the EURUSD 4-type sum-of-exp Hawkes model on the primary-day single-UTC fit (2021-07-20), $t \\in [0, 5]$ s, log-y. Four lines (one per event type).",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    manifest = []
    generators = (
        fig_branching_ratio_invariant,
        fig_spread_scaling_loglog,
        fig_recovery_f1_distribution,
        fig_recovery_delay_distribution,
        fig_kernel_decay_eurusd_sumexp,
    )
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for gen in generators:
        result = gen()
        if result is None:
            print(f"  skipped {gen.__name__} (source data missing)")
            continue
        path, entry = result
        entry["last_generated_utc"] = now
        manifest.append(entry)
        print(f"  wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")

    manifest_path = FIGS_DIR / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"figures": manifest}, f, indent=2)
    print(f"  wrote {manifest_path.relative_to(REPO)} ({len(manifest)} figures)")


if __name__ == "__main__":
    # The default CLI must not restore the historical five-figure programme.
    from render_submission import main as render_submission
    render_submission()
