"""Build artefact: EUR/USD Layer-1 EM residual (goodness-of-fit) diagnostics.

Computes multivariate time-rescaling-theorem residuals for the 2021-07-20
EUR/USD four-type EM-converged sum-of-exponentials Hawkes fit, the dissertation's
empirical anchor.

Method (Ogata time-rescaling / random time change):
  For each event type i with conditional intensity
      lambda_i(t) = mu_i + sum_j sum_r alpha_ij^(r) * A_j^(r)(t),
      A_j^(r)(t) = sum_{t_m < t, type=j} exp(-beta_r (t - t_m)),
  the compensator is
      Lambda_i(t) = mu_i t + sum_j sum_r (alpha_ij^(r)/beta_r) *
                    sum_{t_m < t, type=j} (1 - exp(-beta_r (t - t_m))).
  Between consecutive type-i events the rescaled increments
      tau_k = Lambda_i(t_{k+1}^{(i)}) - Lambda_i(t_k^{(i)})
  are i.i.d. Exp(1) iff the fitted model is the true data-generating process.
  We test each type's residuals (and the pooled set) against Exp(1) by KS.

This run computes NEW diagnostics on the EXISTING fit and EXISTING data. The
EM summary JSON stores mu and the branching matrix but not alpha_all, so the
deterministic EM fit is re-run (identical event loading, jitter seed, beta grid,
T, warm start, config) ONLY to recover the per-decay kernel coefficients
alpha_all. The reconstruction is hard-gated against the pinned anchor:
  rho=0.937308200001031, ll=-64922.718170694534, N=97952, n_iter=516, tol.
If it does not reproduce to tight tolerance, the script raises SystemExit and
writes nothing. eurusd_layer1_em_summary.json is never modified.

Left-tail caveat: HistData millisecond timestamp aggregation plus the random
sub-millisecond tie-break (random_ms jitter) perturb the very shortest
inter-event times, which map onto the smallest rescaled residuals; departures
in the extreme left tail are therefore partly a data-resolution artefact, in
the spirit of the FX Hawkes residual diagnostics of Rambaldi, Pennesi and
Lillo (2015). The diagnostic is reported as a diagnostic check, not a pass/fail
certification.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Reuse the exact event-loading / jitter / fit configuration of the anchor build.
import build_eurusd_layer1_em as embuild

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "data" / "eurusd_layer1_em_residual_diagnostics.json"
OUT_FIG = HERE / "figures" / "eurusd_layer1_em_residual_qq.pdf"

PRIMARY_DATE = "2021-07-20"
PRIMARY_TYPES = embuild.PRIMARY_4TYPE  # (bid_up, bid_down, ask_up, ask_down)
T_WINDOW = 86400.0

# Pinned anchor (artifact wins over memory). Reconstruction must reproduce these.
ANCHOR = {
    "rho_spec": 0.937308200001031,
    "log_likelihood": -64922.718170694534,
    "n_events": 97952,
    "n_iter": 516,
    "converged_reason": "tol",
}
RHO_TOL = 1e-9
LL_TOL = 1e-4


def reconstruct_fit():
    """Re-run the deterministic primary-day EM fit to recover alpha_all.

    Identical to build_eurusd_layer1_em.fit_one_day for the primary day, but
    returns the full Layer1EMResult (which carries alpha_all) instead of the
    JSON-trimmed dict.
    """
    embuild._ensure_simulator_on_path()
    from simulator.layer1_em import Layer1EMConfig, fit_layer1_em

    t_raw, typ_raw = embuild.load_day_events(PRIMARY_DATE)
    t, typ = embuild.apply_random_ms_jitter(t_raw, typ_raw, embuild.JITTER_SEED)

    mu0, alpha0, rho0, ll0, _conv = embuild.load_warm_start(
        embuild.LBFGSB_DIR / "layer1_params_sumexp_4type.json"
    )
    cfg = Layer1EMConfig(
        betas=embuild.BETAS,
        max_iter=600,
        tol_relative_ll=1e-8,
        rho_cap=0.99,
        project_spectral=True,
        initial_mu=mu0,
        initial_alpha_all=alpha0,
        record_trace=True,
    )
    t0 = time.time()
    res = fit_layer1_em(t, typ, T=T_WINDOW, M=4, config=cfg)
    dt = time.time() - t0
    print(
        f"  EM re-run: rho={res.spectral_radius:.12f} ll={res.log_likelihood:.6f} "
        f"n_iter={res.n_iter} converged={res.converged} reason={res.converged_reason} "
        f"({dt:.1f}s)"
    )
    return t, typ, res


def assert_anchor(res):
    drift_rho = abs(res.spectral_radius - ANCHOR["rho_spec"])
    drift_ll = abs(res.log_likelihood - ANCHOR["log_likelihood"])
    checks = {
        "rho_drift": drift_rho,
        "ll_drift": drift_ll,
        "n_events_match": int(res.n_events) == ANCHOR["n_events"],
        "n_iter_match": int(res.n_iter) == ANCHOR["n_iter"],
        "converged": bool(res.converged) and res.converged_reason == ANCHOR["converged_reason"],
    }
    print(f"  anchor check: {checks}")
    ok = (
        drift_rho <= RHO_TOL
        and drift_ll <= LL_TOL
        and checks["n_events_match"]
        and checks["n_iter_match"]
        and checks["converged"]
    )
    if not ok:
        raise SystemExit(
            "STOP: reconstructed EM fit does not reproduce the pinned anchor "
            f"within tolerance: {checks}"
        )
    return checks


def compensator_residuals(t, typ, mu, alpha_all, betas, M):
    """Time-rescaling residuals per type via a single O(N R M) forward pass.

    Returns dict type_idx -> np.ndarray of consecutive-event Exp(1) residuals,
    plus the per-type compensator value at each of its own event times.
    """
    R = betas.shape[0]
    betas = np.asarray(betas, dtype=np.float64)
    A = np.zeros((R, M), dtype=np.float64)  # decayed counts A_j^(r)(t) at current t
    n = np.zeros(M, dtype=np.float64)  # raw counts of type-j events strictly before t
    t_prev = 0.0
    # per-type list of compensator values at that type's own event times
    lam_at_event = {i: [] for i in range(M)}
    N = t.shape[0]
    for k in range(N):
        tk = float(t[k])
        ik = int(typ[k])
        dt = tk - t_prev
        if dt > 0.0:
            A *= np.exp(-betas[:, None] * dt)
        # H_j^(r)(tk) = (n_j - A_j^(r)) / beta_r  ; integral of A over [0, tk]
        H = (n[None, :] - A) / betas[:, None]  # (R, M)
        # Lambda_i(tk) = mu_i * tk + sum_{r,j} alpha_all[r,i,j] * H[r,j]
        # we only need it for i = ik (this event's own type)
        comp_excite = float(np.einsum("rj,rj->", alpha_all[:, ik, :], H))
        Lam_ik = mu[ik] * tk + comp_excite
        lam_at_event[ik].append(Lam_ik)
        # register event k
        A[:, ik] += 1.0
        n[ik] += 1.0
        t_prev = tk

    residuals = {}
    comp_vals = {}
    for i in range(M):
        v = np.asarray(lam_at_event[i], dtype=np.float64)
        comp_vals[i] = v
        residuals[i] = np.diff(v) if v.size >= 2 else np.array([], dtype=np.float64)
    return residuals, comp_vals


def summarise(tau: np.ndarray) -> dict:
    tau = np.asarray(tau, dtype=np.float64)
    out = {
        "count": int(tau.size),
        "mean": float(np.mean(tau)) if tau.size else None,
        "variance": float(np.var(tau, ddof=1)) if tau.size > 1 else None,
        "median": float(np.median(tau)) if tau.size else None,
    }
    if tau.size >= 2:
        ks = stats.kstest(tau, "expon", args=(0.0, 1.0))
        out["ks_statistic"] = float(ks.statistic)
        out["ks_pvalue"] = float(ks.pvalue)
        # left-tail diagnostic: under Exp(1), P(tau < 0.1) = 1 - e^{-0.1} = 0.09516
        thr = 0.1
        out["left_tail_threshold"] = thr
        out["left_tail_observed_frac"] = float(np.mean(tau < thr))
        out["left_tail_expected_frac"] = float(1.0 - np.exp(-thr))
    return out


def make_figure(residuals, type_names, out_path):
    """Two-panel QQ: linear Exp(1) QQ (per type) + tail-sensitive log-survival."""
    pooled = np.concatenate([residuals[i] for i in range(len(type_names))])
    pooled_sorted = np.sort(pooled)
    n = pooled_sorted.size
    # theoretical Exp(1) quantiles at plotting positions (m-0.5)/n
    pp = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log1p(-pp)  # = -ln(1-pp), Exp(1) quantile

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel 1: per-type QQ against Exp(1), linear axes.
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(type_names)))
    qmax = 0.0
    for i, name in enumerate(type_names):
        ti = np.sort(residuals[i])
        if ti.size < 2:
            continue
        ppi = (np.arange(1, ti.size + 1) - 0.5) / ti.size
        theoi = -np.log1p(-ppi)
        ax1.plot(theoi, ti, ".", ms=2.5, color=colors[i], label=name.replace("_", r"\_") if False else name)
        qmax = max(qmax, float(theoi.max()), float(ti.max()))
    lim = min(qmax, 12.0)
    ax1.plot([0, lim], [0, lim], "k--", lw=1.0, label="y = x")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Theoretical Exp(1) quantile")
    ax1.set_ylabel("Empirical rescaled residual")
    ax1.set_title("Time-rescaling QQ by event type")
    ax1.legend(fontsize=8, markerscale=3, loc="upper left", framealpha=0.9)

    # Panel 2: tail-sensitive empirical log-survival vs Exp(1) line.
    surv = 1.0 - (np.arange(1, n + 1) - 0.5) / n
    ax2.semilogy(pooled_sorted, surv, ".", ms=2.0, color="#1f4e79", label="pooled residuals")
    xline = np.linspace(0, float(pooled_sorted.max()), 200)
    ax2.semilogy(xline, np.exp(-xline), "r-", lw=1.2, label=r"Exp(1): $e^{-\tau}$")
    ax2.set_xlabel("Rescaled residual " + r"$\tau$")
    ax2.set_ylabel("Survival  P(T > " + r"$\tau$)")
    ax2.set_title("Tail-sensitive survival (pooled)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(max(1.0 / n / 5.0, 1e-6), 1.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print(f"=== EURUSD {PRIMARY_DATE} EM residual diagnostics ===")
    t, typ, res = reconstruct_fit()
    anchor_checks = assert_anchor(res)

    mu = np.asarray(res.mu, dtype=np.float64)
    alpha_all = np.asarray(res.alpha_all, dtype=np.float64)
    betas = np.asarray(res.betas, dtype=np.float64)
    M = mu.shape[0]

    residuals, comp_vals = compensator_residuals(t, typ, mu, alpha_all, betas, M)

    per_type = {}
    for i, name in enumerate(PRIMARY_TYPES):
        s = summarise(residuals[i])
        s["compensator_total"] = float(comp_vals[i][-1]) if comp_vals[i].size else None
        s["n_events_type"] = int(comp_vals[i].size)
        per_type[name] = s
        print(
            f"  {name:9s}: n={s['count']:6d} mean={s['mean']:.4f} "
            f"var={s['variance']:.4f} KS={s['ks_statistic']:.4f} p={s['ks_pvalue']:.3e}"
        )

    pooled = np.concatenate([residuals[i] for i in range(M)])
    pooled_summary = summarise(pooled)
    print(
        f"  POOLED   : n={pooled_summary['count']} mean={pooled_summary['mean']:.4f} "
        f"var={pooled_summary['variance']:.4f} KS={pooled_summary['ks_statistic']:.4f} "
        f"p={pooled_summary['ks_pvalue']:.3e}"
    )

    make_figure(residuals, list(PRIMARY_TYPES), OUT_FIG)

    payload = {
        "artifact": "eurusd_layer1_em_residual_diagnostics",
        "description": (
            "Multivariate time-rescaling-theorem residual diagnostics for the "
            "2021-07-20 EUR/USD four-type EM-converged sum-of-exponentials Hawkes "
            "fit (the dissertation's empirical anchor). NEW diagnostics on the "
            "EXISTING fit/data; alpha_all reconstructed by re-running the "
            "deterministic EM fit and gated against the pinned anchor."
        ),
        "method": "time_rescaling_theorem_consecutive_same_type_increments_vs_Exp1",
        "primary_date": PRIMARY_DATE,
        "fit_window_seconds": T_WINDOW,
        "betas_grid_per_s": list(embuild.BETAS),
        "event_types": list(PRIMARY_TYPES),
        "jitter_seed": embuild.JITTER_SEED,
        "reconstruction_anchor_check": {
            "pinned": ANCHOR,
            "reconstructed": {
                "rho_spec": float(res.spectral_radius),
                "log_likelihood": float(res.log_likelihood),
                "n_events": int(res.n_events),
                "n_iter": int(res.n_iter),
                "converged_reason": res.converged_reason,
            },
            "rho_drift": float(abs(res.spectral_radius - ANCHOR["rho_spec"])),
            "ll_drift": float(abs(res.log_likelihood - ANCHOR["log_likelihood"])),
            "passed": True,
        },
        "per_type": per_type,
        "pooled": pooled_summary,
        "interpretation_caveats": (
            "Residuals are consecutive same-type compensator increments; the "
            "first interval [0, first event] and the final interval [last event, T] "
            "are excluded, so each type contributes (count_i - 1) residuals. KS is "
            "against Exp(1) with fixed loc=0, scale=1 (no parameters estimated from "
            "the residuals). HistData millisecond timestamp aggregation and the "
            "random sub-millisecond tie-break distort the extreme left tail "
            "(shortest inter-event times), so small-residual departures are partly "
            "a data-resolution artefact, consistent with Rambaldi-Pennesi-Lillo "
            "(2015) FX Hawkes diagnostics. Reported as a diagnostic check, not a "
            "pass/fail certification."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
