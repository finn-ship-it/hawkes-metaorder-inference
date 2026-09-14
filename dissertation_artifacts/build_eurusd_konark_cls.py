"""Build artefacts:
   dissertation_artifacts/data/eurusd_konark_cls_kernel.json
   dissertation_artifacts/figures/eurusd_konark_cls_kernel.pdf

Runs the Konark CLS-LogLin long-memory non-parametric estimator on the
2021-07-20 EURUSD primary-day events (`random_ms` jitter, seed 20210720,
N = 97,952) and emits the dissertation-quotable JSON + a 4-panel figure
overlaying the parametric EM-fitted sum-of-exp kernel on the
non-parametric CLS-LogLin coefficients.

Also runs the synthetic-recovery battery requested by the prompt 02b
spec at rho_true in {0.4, 0.7, 0.9}, persists the per-rho diagnostics
inline (under `synthetic_recovery_battery` in the JSON) so the V1-V6
audit can trace every quoted number to a single source-of-truth file.
"""

from __future__ import annotations

import json
import sys
import time
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
OUT_JSON = HERE / "data" / "eurusd_konark_cls_kernel.json"
OUT_PDF = HERE / "figures" / "eurusd_konark_cls_kernel.pdf"
SRC_PATH = str(REPO_ROOT / "src")

PRIMARY_4TYPE = ("bid_up", "bid_down", "ask_up", "ask_down")
JITTER_SEED = 20210720
T_DAY = 86400.0


def _ensure_simulator_on_path() -> None:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


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


def _simulate_power_law_cluster(
    mu, alpha, beta_pl, gamma_pl, t_max_pl, T, rng_seed, max_generations=64
):
    """Cluster-process simulator for power-law-cutoff Hawkes.
    Exact for any sub-critical kernel.

    Mirrors the helper in tests/test_layer1_konark_cls.py.
    """
    rng = np.random.default_rng(int(rng_seed))
    M = mu.shape[0]
    K_factor = (
        (1.0 - (1.0 + gamma_pl * t_max_pl) ** (1.0 - beta_pl))
        / (gamma_pl * (beta_pl - 1.0))
    )
    K_endpoint = 1.0 - (1.0 + gamma_pl * t_max_pl) ** (1.0 - beta_pl)
    inv_exp = 1.0 / (1.0 - beta_pl)
    times_list = []
    types_list = []
    for i in range(M):
        n_bg = int(rng.poisson(mu[i] * T))
        if n_bg > 0:
            times_list.append(rng.uniform(0.0, T, n_bg))
            types_list.append(np.full(n_bg, i, dtype=np.int64))
    if not times_list:
        return np.empty(0), np.empty(0, dtype=np.int64), 0
    parent_t = np.concatenate(times_list)
    parent_k = np.concatenate(types_list)
    all_t = [parent_t.copy()]
    all_k = [parent_k.copy()]
    n_generations = 0
    for _ in range(max_generations):
        if parent_t.size == 0:
            break
        new_t_segs = []
        new_k_segs = []
        for j in range(M):
            mask_j = (parent_k == j)
            if not np.any(mask_j):
                continue
            tp_j = parent_t[mask_j]
            n_par = tp_j.size
            for i in range(M):
                rate = float(alpha[i, j]) * K_factor
                if rate <= 0.0:
                    continue
                n_children = rng.poisson(rate, size=n_par)
                total = int(n_children.sum())
                if total == 0:
                    continue
                tp_rep = np.repeat(tp_j, n_children)
                u = rng.uniform(0.0, 1.0, total)
                tau = ((1.0 - u * K_endpoint) ** inv_exp - 1.0) / gamma_pl
                ct = tp_rep + tau
                keep = ct < T
                if np.any(keep):
                    new_t_segs.append(ct[keep])
                    new_k_segs.append(np.full(int(np.sum(keep)), i, dtype=np.int64))
        if not new_t_segs:
            break
        parent_t = np.concatenate(new_t_segs)
        parent_k = np.concatenate(new_k_segs)
        all_t.append(parent_t)
        all_k.append(parent_k)
        n_generations += 1
    times = np.concatenate(all_t)
    types = np.concatenate(all_k)
    if times.size == 0:
        return times, types, n_generations
    o = np.argsort(times, kind="stable")
    return times[o], types[o], n_generations


def _simulate_sumexp_for_synth(mu, alpha_all, betas, T, rng, max_generations=64):
    """Cluster-process Hawkes simulator (single-exp) used by the synthetic
    recovery battery. Identical to the one in the test file."""
    M = mu.shape[0]
    R = betas.shape[0]
    times: list = []
    types: list = []
    for i in range(M):
        n_bg = int(rng.poisson(mu[i] * T))
        if n_bg > 0:
            tt = rng.uniform(0.0, T, n_bg)
            times.extend(tt.tolist())
            types.extend([i] * n_bg)
    parent_t = list(times)
    parent_k = list(types)
    for _ in range(max_generations):
        new_t: list = []
        new_k: list = []
        for t_p, k_p in zip(parent_t, parent_k):
            for r in range(R):
                inv_beta = 1.0 / float(betas[r])
                for i in range(M):
                    rate = float(alpha_all[r, i, k_p]) / float(betas[r])
                    n = int(rng.poisson(rate))
                    if n == 0:
                        continue
                    d = rng.exponential(inv_beta, n)
                    tt = t_p + d
                    keep = tt < T
                    if not np.any(keep):
                        continue
                    new_t.extend(tt[keep].tolist())
                    new_k.extend([i] * int(np.sum(keep)))
        if not new_t:
            break
        times.extend(new_t)
        types.extend(new_k)
        parent_t = new_t
        parent_k = new_k
    arr = np.asarray(times, dtype=np.float64)
    typs = np.asarray(types, dtype=np.int64)
    if arr.size == 0:
        return arr, typs
    o = np.argsort(arr, kind="stable")
    return arr[o], typs[o]


def synthetic_recovery_battery():
    _ensure_simulator_on_path()
    from simulator.layer1_konark_cls import KonarkCLSConfig, fit_konark_cls

    print("\n=== Synthetic recovery battery (power-law-cutoff; cluster-process simulator) ===")
    print("    Cluster-process simulator: exact for any sub-critical Hawkes; runs in")
    print("    O(N * M * M) wall-time regardless of kernel form. Replaces the prior")
    print("    Ogata-thinning attempt (which exceeded 10+ minute wall at near-critical rho).")
    print("    Per the 02b spec, power-law-cutoff is the kernel form the long-memory")
    print("    estimator validates against; this is the methodologically correct synthetic.")
    print()
    print("    Spec event-count targets:")
    print("      rho=0.4: >= 25,000 events.")
    print("      rho=0.7: >= 80,000 events.")
    print("      rho=0.9: >= 250,000 events.")
    results = {}
    M = 2  # diagonal-dominant symmetric; M=4 has high off-diagonal collinearity at low rho
    beta_pl = 2.0
    gamma_pl = 1.0
    t_max_pl = 500.0
    K_factor = (
        (1.0 - (1.0 + gamma_pl * t_max_pl) ** (1.0 - beta_pl))
        / (gamma_pl * (beta_pl - 1.0))
    )
    for rho_target, T_sim in [(0.4, 16000.0), (0.7, 16000.0), (0.9, 16000.0)]:
        print(f"\n  rho_target = {rho_target}")
        # Diagonal-dominant: rho = d + o for M=2 symmetric; pick o = 0.05.
        Phi_off = 0.05
        Phi_diag = rho_target - Phi_off
        alpha = np.full((M, M), Phi_off / K_factor, dtype=np.float64)
        np.fill_diagonal(alpha, Phi_diag / K_factor)
        Phi_true = alpha * K_factor
        rho_true_actual = float(np.max(np.abs(np.linalg.eigvals(Phi_true))))
        mu = np.full(M, 1.0)
        t_sim_start = time.time()
        times, types, n_gen = _simulate_power_law_cluster(
            mu, alpha, beta_pl, gamma_pl, t_max_pl, T_sim,
            rng_seed=int(round(rho_target * 1000)),
        )
        sim_wall = time.time() - t_sim_start
        # Slope-fit window is in the asymptotic gamma*t >> 1 regime, so
        # the analytical log-log slope is close to -beta. min_lag=0.5,
        # max_lag=500 -> slope window [2.5, 50]. num_datapoints=18 ->
        # 17 log bins -> ~7 in window.
        cfg = KonarkCLSConfig(
            min_lag=0.5, max_lag=500.0, num_datapoints=18,
            fine_bin_seconds=0.05, solver="osqp", subcritical_cap=0.999,
            chunk_size=50000,
        )
        t0 = time.time()
        res = fit_konark_cls(times, types, T=T_sim, M=M, config=cfg)
        fit_wall = time.time() - t0
        rho_fit = res.spectral_radius
        gap = rho_fit - rho_true_actual
        gap_pct = 100.0 * gap / rho_true_actual

        # Composite kernel-shape metrics
        diffs_diag = []
        for i in range(M):
            diffs_diag.extend(np.diff(res.theta[i, i, :]).tolist())
        nonpos_diag = sum(1 for d in diffs_diag if d <= 1e-12)
        frac_nonpos_diag = nonpos_diag / max(len(diffs_diag), 1)
        diffs_off = []
        for i in range(M):
            for j in range(M):
                if i != j:
                    diffs_off.extend(np.diff(res.theta[i, j, :]).tolist())
        nonpos_off = sum(1 for d in diffs_off if d <= 1e-12)
        frac_nonpos_off = nonpos_off / max(len(diffs_off), 1)

        # Slope-fit window
        win_lo, win_hi = 5.0 * cfg.min_lag, 0.1 * cfg.max_lag
        bin_centres = 0.5 * (res.lag_grid[:-1] + res.lag_grid[1:])
        in_win = (bin_centres >= win_lo) & (bin_centres <= win_hi)
        slope_window_bins = int(np.sum(in_win))
        # Pooled-(i, j) log-log slope on the slope-fit window
        pooled_theta = np.sum(res.theta, axis=(0, 1))
        x = bin_centres[in_win]
        y = pooled_theta[in_win]
        pos_mask = y > 1e-12
        if int(np.sum(pos_mask)) >= 3:
            slope, _intercept = np.polyfit(np.log(x[pos_mask]), np.log(y[pos_mask]), 1)
        else:
            slope = float("nan")
        slope_gap = slope + beta_pl  # gap to -beta

        results[f"rho_{rho_target:.1f}"] = {
            "rho_target": float(rho_target),
            "rho_true": float(rho_true_actual),
            "rho_fit": float(rho_fit),
            "gap_absolute": float(gap),
            "gap_percent": float(gap_pct),
            "n_events_simulated": int(times.size),
            "T_seconds": float(T_sim),
            "n_generations_completed": int(n_gen),
            "simulation_wall_seconds": float(sim_wall),
            "fit_wall_seconds": float(fit_wall),
            "osqp_status": res.osqp_status,
            "osqp_n_iter": int(res.osqp_iter),
            "osqp_pri_res": float(res.osqp_pri_res),
            "osqp_dua_res": float(res.osqp_dua_res),
            "osqp_polish_status": res.osqp_polish_status,
            "subcrit_margin_min": float(res.fit_quality["subcriticality_margin_min"]),
            "subcrit_margin_max": float(res.fit_quality["subcriticality_margin_max"]),
            "kernel_nonpos_diff_fraction_diagonal": float(frac_nonpos_diag),
            "kernel_nonpos_diff_fraction_off_diagonal": float(frac_nonpos_off),
            "slope_window_bin_count": slope_window_bins,
            "pooled_log_log_slope": float(slope) if not np.isnan(slope) else None,
            "slope_gap_vs_minus_beta": float(slope_gap) if not np.isnan(slope) else None,
            "branching_matrix": res.branching_matrix.tolist(),
            # Acceptance per the spec
            "acceptance_pass_rho_within_10pct": bool(abs(gap) / rho_true_actual < 0.10),
            "acceptance_pass_slope_within_0p5_of_minus_beta": bool(
                not np.isnan(slope) and abs(slope_gap) < 0.5
            ),
            "acceptance_event_count_meets_spec_threshold": bool(
                times.size >= {0.4: 25000, 0.7: 80000, 0.9: 250000}[rho_target]
            ),
        }
        print(
            f"    N={times.size}  sim_wall={sim_wall:.2f}s  fit_wall={fit_wall:.2f}s  "
            f"generations={n_gen}"
        )
        print(
            f"    rho_true={rho_true_actual:.4f}  rho_fit={rho_fit:.4f}  "
            f"gap={gap_pct:+.2f}%  spec_acceptance(+-10%)="
            f"{'PASS' if abs(gap)/rho_true_actual < 0.10 else 'FAIL'}"
        )
        print(
            f"    pooled-slope={slope:.3f}  -beta={-beta_pl:.1f}  "
            f"slope_gap={slope_gap:+.3f}  spec_acceptance(+-0.5)="
            f"{'PASS' if abs(slope_gap) < 0.5 else 'FAIL'}"
        )
        print(
            f"    OSQP status={res.osqp_status} iter={res.osqp_iter} "
            f"pri={res.osqp_pri_res:.2e} dua={res.osqp_dua_res:.2e}  "
            f"subcrit_margin_min={res.fit_quality['subcriticality_margin_min']:.3f}"
        )
        print(
            f"    nonpos-diff frac: diag={frac_nonpos_diag:.2f} off={frac_nonpos_off:.2f}  "
            f"slope-fit window bins={slope_window_bins}"
        )
    return results


def fit_eurusd_primary_day(max_lag_s: float = 500.0):
    _ensure_simulator_on_path()
    from simulator.layer1_konark_cls import KonarkCLSConfig, fit_konark_cls

    print("\n=== EURUSD primary day 2021-07-20 ===")
    t_raw, typ_raw = load_day_events("2021-07-20")
    print(f"  loaded N = {t_raw.size}")
    t, typ = apply_random_ms_jitter(t_raw, typ_raw, JITTER_SEED)

    # FX-tuned defaults for prompt 02b. The fine-bin width is set
    # explicitly (0.5 s for FX at ~1.1 events/s; this gives an active-
    # bin fraction of ~0.4, which is much higher than Konark's equity
    # default at 0.111 ms). num_datapoints=22 ensures the slope-fit
    # window [5*min_lag, 0.1*max_lag] = [2.5, 50] contains >= 5 bins
    # (the per-spec configuration sanity check).
    cfg = KonarkCLSConfig(
        min_lag=0.5,
        max_lag=max_lag_s,
        num_datapoints=22,
        fine_bin_seconds=0.5,
        solver="osqp",
        subcritical_cap=0.999,
        chunk_size=50000,
    )
    M = 4
    t0 = time.time()
    res = fit_konark_cls(t, typ, T=T_DAY, M=M, config=cfg)
    dt = time.time() - t0
    print(
        f"  fit dt={dt:.1f}s  rho={res.spectral_radius:.4f}  "
        f"status={res.osqp_status}  iter={res.osqp_iter}  "
        f"pri={res.osqp_pri_res:.2e}  dua={res.osqp_dua_res:.2e}  "
        f"polish={res.osqp_polish_status}"
    )
    print(f"  branching matrix:")
    print(res.branching_matrix)

    # Three-way comparison
    em_payload = json.loads(EM_SUMMARY_PATH.read_text())
    rho_em = em_payload["primary_day_2021_07_20"]["em_result"]["rho_spec"]
    rho_lbfgsb = em_payload["primary_day_2021_07_20"]["lbfgsb_warm_start"]["rho_spec"]
    bm_path = HERE / "data" / "eurusd_bacry_muzy_kernel.json"
    if bm_path.exists():
        bm_payload = json.loads(bm_path.read_text())
        rho_bm = bm_payload["result"]["spectral_radius"]
    else:
        rho_bm = None

    konark_em_gap = res.spectral_radius - rho_em
    konark_em_gap_pct = 100.0 * konark_em_gap / rho_em
    band_pass = abs(konark_em_gap) / rho_em < 0.10

    return res, {
        "konark_cls_rho": float(res.spectral_radius),
        "em_converged_rho": float(rho_em),
        "lbfgsb_rho": float(rho_lbfgsb),
        "bacry_muzy_rho": float(rho_bm) if rho_bm is not None else None,
        "konark_minus_em_absolute": float(konark_em_gap),
        "konark_minus_em_percent": float(konark_em_gap_pct),
        "band_pass_within_10pct_of_em": band_pass,
    }


def make_figure(res, rho_em: float, rho_lbfgsb: float):
    """Four-panel kernel figure: phi[i, j](t) on log-log axes for
    selected (i, j) pairs (we render the four diagonal entries as the
    headline panel set; full 4x4 grid produced as the alternative)."""
    grid_centres = 0.5 * (res.lag_grid[:-1] + res.lag_grid[1:])

    # Reconstruct the parametric kernel on the same grid for overlay
    parametric_phi = None
    try:
        lbfgsb_path = LBFGSB_DIR / "layer1_params_sumexp_4type.json"
        payload = json.loads(lbfgsb_path.read_text())
        alpha_all = np.asarray(payload["alpha_all"], dtype=np.float64)
        betas_par = np.asarray(payload["betas"], dtype=np.float64)
        M_par = alpha_all.shape[1]
        K = grid_centres.size
        parametric_phi = np.zeros((M_par, M_par, K), dtype=np.float64)
        for r in range(betas_par.size):
            decay = np.exp(-betas_par[r] * grid_centres)
            parametric_phi += alpha_all[r, :, :, None] * decay[None, None, :]
    except Exception as exc:
        print(f"  (could not load parametric overlay: {exc})")

    type_labels = list(PRIMARY_4TYPE)
    M = res.theta.shape[0]
    fig, axes = plt.subplots(M, M, figsize=(13, 11), sharex=True)
    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            ax.plot(
                grid_centres, res.theta[i, j, :],
                color="C0", lw=1.6, label="Konark CLS-LogLin",
            )
            if parametric_phi is not None:
                ax.plot(
                    grid_centres, parametric_phi[i, j, :],
                    color="C1", ls="--", lw=1.0, label="parametric (L-BFGS-B)",
                )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(rf"$\phi_{{{i}{j}}}(t)$  ({type_labels[i][:1]}{type_labels[j][:1]})", fontsize=8)
            if i == M - 1:
                ax.set_xlabel("lag (s)")
            if j == 0:
                ax.set_ylabel("kernel value")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"EUR/USD 2021-07-20: Konark CLS-LogLin kernel (4x4 grid)\n"
        f"rho={res.spectral_radius:.4f} (vs EM 0.937, L-BFGS-B 0.988)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


def main():
    _ensure_simulator_on_path()

    synth_results = synthetic_recovery_battery()

    res, comparison = fit_eurusd_primary_day(max_lag_s=500.0)

    out_payload = {
        "primary_day": "2021-07-20",
        "synthetic_simulator": "cluster_process_power_law_cutoff",
        "synthetic_simulator_kernel_form": (
            "phi_ij(t) = alpha_ij * (1 + gamma * t)^(-beta) for t in [0, t_max]; "
            "delays sampled by inverse-CDF on truncated power-law density; "
            "exact for any sub-critical Hawkes regardless of kernel form."
        ),
        "n_events": int(res.n_events),
        "fit_window_seconds": float(res.fit_window_seconds),
        "config": {
            "min_lag": 0.5,
            "max_lag": 500.0,
            "num_datapoints": 22,
            "n_bins": int(res.n_bins),
            "fine_bin_seconds": 0.5,
            "solver": "osqp",
            "subcritical_cap": 0.999,
            "chunk_size": 50000,
        },
        "result": {
            "spectral_radius": float(res.spectral_radius),
            "branching_matrix": res.branching_matrix.tolist(),
            "intercept_mu_per_second": res.intercept.tolist(),
            "lag_grid": res.lag_grid.tolist(),
            "bin_widths": res.bin_widths.tolist(),
            "delta_fine": float(res.delta_fine),
            "n_events_per_type": res.n_events_per_type.tolist(),
            "theta": res.theta.tolist(),
            "fit_quality": res.fit_quality,
            "osqp_status": res.osqp_status,
            "osqp_iter": int(res.osqp_iter),
            "osqp_pri_res": float(res.osqp_pri_res),
            "osqp_dua_res": float(res.osqp_dua_res),
            "osqp_polish_status": res.osqp_polish_status,
        },
        "three_way_comparison": comparison,
        "synthetic_recovery_battery": synth_results,
        "honest_claims_caveat": (
            "Konark CLS-LogLin is the long-memory non-parametric Layer-1 "
            "estimator. The lag-grid follows the Konark default convention "
            "(num_datapoints=10 -> 19 lag points -> 18 non-overlapping bins, "
            "with bin b spanning [lag_grid[b], lag_grid[b+1])). The deviations "
            "from Konark's pipeline are: (i) all-fine-bins iteration in "
            "place of active-bins-only (corrects active-bins-only intercept "
            "bias); (ii) bin-width-weighted subcriticality constraint instead "
            "of Konark's uniform-bin shortcut (correct on log-spaced grids); "
            "(iii) FX 4-type alphabet, not 12-type LOB; (iv) no equity "
            "TOD / spread-power-law / special-date calendars; (v) NPHC "
            "graph-sign hints accepted as input but not computed (off by "
            "default). All deviations named in the docstring of "
            "simulator/layer1_konark_cls.py."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_payload, indent=2))
    print(f"\nWrote {OUT_JSON}")

    rho_em = comparison["em_converged_rho"]
    rho_lbfgsb = comparison["lbfgsb_rho"]
    make_figure(res, rho_em, rho_lbfgsb)


if __name__ == "__main__":
    main()
