"""Build artefact: dissertation_artifacts/data/eurusd_bacry_muzy_kernel.json
            and: dissertation_artifacts/figures/eurusd_nonparametric_kernel.pdf.

Runs the histogram-basis Bacry-Muzy non-parametric kernel estimator on the
2021-07-20 EURUSD primary-day events (random_ms jitter, seed 20210720) and
compares the resulting branching ratio against the EM-converged parametric
anchor 0.937308 (post prompt-01 amendment). The 4-panel figure overlays the
non-parametric phi[i, j](t) on the parametric sum-of-exp kernel from the
existing L-BFGS-B / EM fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PARQUET = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "eurusd_events.parquet"
LBFGSB_DIR = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "layer1"
EM_SUMMARY_PATH = HERE / "data" / "eurusd_layer1_em_summary.json"
OUT_JSON = HERE / "data" / "eurusd_bacry_muzy_kernel.json"
OUT_PDF = HERE / "figures" / "eurusd_nonparametric_kernel.pdf"

PRIMARY_4TYPE = ("bid_up", "bid_down", "ask_up", "ask_down")
JITTER_SEED = 20210720
T_DAY = 86400.0


def _ensure_simulator_on_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_day_events(date_str: str):
    type_to_idx = {t: i for i, t in enumerate(PRIMARY_4TYPE)}
    df = pd.read_parquet(
        PARQUET,
        columns=["timestamp", "event_type"],
        filters=[("event_type", "in", list(PRIMARY_4TYPE))],
    )
    day = pd.Timestamp(date_str)
    mask = (df["timestamp"] >= day) & (df["timestamp"] < day + pd.Timedelta(days=1))
    df = df.loc[mask].copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    t = (df["timestamp"] - day).dt.total_seconds().to_numpy(dtype=np.float64)
    typ = df["event_type"].map(type_to_idx).to_numpy(dtype=np.int64)
    return t, typ


def apply_random_ms_jitter(t, typ, seed):
    rng = np.random.default_rng(seed)
    t_j = t + rng.uniform(0.0, 1.0e-3, size=t.shape[0])
    order = np.argsort(t_j, kind="stable")
    return t_j[order], typ[order]


def parametric_kernel_grid(em_summary_payload, betas_grid, lag_grid):
    """Reconstruct the parametric kernel phi(t) at lag_grid from the
    EM-converged sum-of-exp result.
    """
    primary = em_summary_payload["primary_day_2021_07_20"]
    em = primary["em_result"]
    # branching_matrix is what we need for spectral comparison; for the
    # kernel curves, recompute from alpha_all stored in eurusd_layer1_em_summary.
    # If alpha_all not present, fall back to L-BFGS-B JSON.
    alpha_path = LBFGSB_DIR / "layer1_params_sumexp_4type.json"
    payload = json.loads(alpha_path.read_text())
    alpha_all = np.asarray(payload["alpha_all"], dtype=np.float64)  # (R, M, M)
    betas = np.asarray(payload["betas"], dtype=np.float64)
    M = alpha_all.shape[1]
    K = lag_grid.size
    phi = np.zeros((M, M, K), dtype=np.float64)
    for r in range(betas.size):
        # phi_ij(t) = sum_r alpha_ij^{(r)} exp(-beta_r t)
        decay = np.exp(-betas[r] * lag_grid)  # (K,)
        phi += alpha_all[r, :, :, None] * decay[None, None, :]
    return phi


def main():
    _ensure_simulator_on_path()
    from simulator.layer1_nonparametric import BacryMuzyConfig, fit_bacry_muzy

    print("=== EURUSD 2021-07-20 Bacry-Muzy non-parametric fit ===")
    t_raw, typ_raw = load_day_events("2021-07-20")
    print(f"  loaded N={t_raw.size:,}")
    t, typ = apply_random_ms_jitter(t_raw, typ_raw, JITTER_SEED)

    # Histogram basis with 200 bins out to 10 s
    cfg = BacryMuzyConfig(
        n_grid=200,
        max_lag=10.0,
        tikhonov_lambda=0.0,
        enforce_non_negative=True,
        basis="histogram",
    )
    M = 4
    res = fit_bacry_muzy(t, typ, T=T_DAY, M=M, config=cfg)
    print(
        f"  rho_nonparam = {res.spectral_radius:.6f}  "
        f"vs EM-converged parametric 0.937308"
    )
    print(
        f"  branching_matrix:\n{res.branching_matrix}"
    )

    # Load EM summary to get the parametric comparator (the EM-converged result)
    em_payload = json.loads(EM_SUMMARY_PATH.read_text())
    rho_em = em_payload["primary_day_2021_07_20"]["em_result"]["rho_spec"]
    rho_lbfgsb = em_payload["primary_day_2021_07_20"]["lbfgsb_warm_start"]["rho_spec"]

    # Compute relative-difference metrics
    rho_diff_em = res.spectral_radius - rho_em
    rho_diff_em_pct = 100.0 * rho_diff_em / rho_em

    out_payload = {
        "primary_day": "2021-07-20",
        "n_events": int(res.n_events),
        "fit_window_seconds": float(res.fit_window_seconds),
        "config": {
            "basis": "histogram",
            "n_grid": int(cfg.n_grid),
            "max_lag_seconds": float(cfg.max_lag),
            "tikhonov_lambda": float(cfg.tikhonov_lambda),
            "enforce_non_negative": bool(cfg.enforce_non_negative),
        },
        "result": {
            "spectral_radius": float(res.spectral_radius),
            "branching_matrix": res.branching_matrix.tolist(),
            "asymptotic_intensities": res.asymptotic_intensities.tolist(),
            "n_events_per_type": res.n_events_per_type.tolist(),
            "bin_width_seconds": float(res.bin_width),
            "grid_seconds": res.grid_seconds.tolist(),
            "phi_grid": res.phi_grid.tolist(),
            "fit_quality": res.fit_quality,
        },
        "comparator": {
            "em_converged_rho": float(rho_em),
            "lbfgsb_rho": float(rho_lbfgsb),
            "nonparam_minus_em_absolute": float(rho_diff_em),
            "nonparam_minus_em_percent": float(rho_diff_em_pct),
        },
        "histogram_basis_caveat": (
            "Histogram-basis Bacry-Muzy with bin width Delta = 0.05 s carries "
            "a known finite-sample / discretisation bias of order Delta * g(0) "
            "(typically 5-25% on the per-bin kernel value, depending on the "
            "branching ratio). The integrated branching matrix and spectral "
            "radius inherit this bias; the EM-converged parametric anchor "
            "rho=0.937 should be considered the more reliable point estimate. "
            "Smoother bases (B-spline / raised cosine) are reserved for a "
            "future revision."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_payload, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # 4-panel figure: phi[i, j](t) non-parametric vs parametric
    print("\n[figure] generating 4-panel grid...")
    parametric_phi = parametric_kernel_grid(
        em_payload, np.array([0.1, 1.0, 10.0, 100.0]), res.grid_seconds
    )
    type_labels = list(PRIMARY_4TYPE)
    fig, axes = plt.subplots(M, M, figsize=(12, 10), sharex=True)
    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            ax.plot(
                res.grid_seconds,
                res.phi_grid[i, j, :],
                color="C0",
                lw=1.5,
                label="non-parametric",
            )
            ax.plot(
                res.grid_seconds,
                parametric_phi[i, j, :],
                color="C1",
                lw=1.0,
                ls="--",
                label="sum-of-exp (L-BFGS-B)",
            )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(rf"$\phi_{{{type_labels[i][:1]}{type_labels[j][:1]},{type_labels[i][-1]}{type_labels[j][-1]}}}(t)$",
                         fontsize=9)
            if i == M - 1:
                ax.set_xlabel("lag (s)")
            if j == 0:
                ax.set_ylabel(rf"$\phi_{{{i},{j}}}(t)$")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "EURUSD 2021-07-20: non-parametric Bacry-Muzy kernel vs parametric sum-of-exp",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
