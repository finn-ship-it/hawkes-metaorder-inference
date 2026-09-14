"""Multi-init EM robustness check for the Layer-1 sum-of-exp anchor.

Runs 16 EM fits on the 2021-07-20 EURUSD primary day with random
alpha/mu initialisations sampled to give rho_init in {0.3, 0.5, 0.7, 0.9}
(4 starts each). Reports the distribution of converged rho and ll.

Also performs an LL sanity check: scores the existing L-BFGS-B JSON's
(mu, alpha_all) with the EM module's intensity calculator. The expected
LL is -66715.38; deviation > 1e-3 indicates the comparison is mis-aligned
(e.g. different jitter, different event ordering, code bug).

Output: dissertation_artifacts/data/eurusd_layer1_em_robustness.json.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent  # "Work on Hawkes"
PARQUET = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "eurusd_events.parquet"
LBFGSB_DIR = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "layer1"
OUT_PATH = HERE / "data" / "eurusd_layer1_em_robustness.json"
SRC_PATH = str(REPO_ROOT / "src")

PRIMARY_4TYPE = ("bid_up", "bid_down", "ask_up", "ask_down")
JITTER_SEED = 20210720
BETAS = (0.1, 1.0, 10.0, 100.0)
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


def random_init_for_rho(rho_init: float, M: int, betas: np.ndarray, seed: int):
    """Sample a random (mu, alpha_all) with branching ratio rho(Phi) == rho_init."""
    from simulator.layer1_em import (
        branching_matrix_sumexp,
        spectral_radius_sumexp,
    )

    rng = np.random.default_rng(seed)
    R = betas.shape[0]
    # Random non-negative alpha_all of order beta_r (so per-component
    # branching mass alpha/beta is order 1 before scaling).
    alpha = rng.uniform(low=0.0, high=1.0, size=(R, M, M))
    for r in range(R):
        # Diagonal slightly larger than off-diagonal to mimic self-excitation
        alpha[r] *= float(betas[r]) * 0.5
        for i in range(M):
            alpha[r, i, i] *= 1.5
    rho_now = spectral_radius_sumexp(alpha, betas)
    if rho_now > 0.0:
        alpha = alpha * (rho_init / rho_now)
    rho_check = spectral_radius_sumexp(alpha, betas)
    assert abs(rho_check - rho_init) < 1e-9, (rho_check, rho_init)
    # mu sampled around base rate * (1 - rho_init)
    base_rate = 1.0  # arbitrary placeholder; caller overrides via empirical N_i / T
    mu = rng.uniform(0.5, 1.5, size=M) * (1.0 - rho_init) * base_rate
    mu = np.maximum(mu, 1e-6)
    return mu, alpha


def _fit_one(args):
    """Worker: fit EM from a given init. Returns dict with diagnostics."""
    _ensure_simulator_on_path()
    from simulator.layer1_em import Layer1EMConfig, fit_layer1_em

    (
        idx,
        rho_init,
        seed,
        max_iter,
        rho_cap,
        tol,
        t_path,
        ty_path,
        T,
        M,
        betas_tuple,
        base_rate_per_type,
    ) = args

    times = np.load(t_path)
    types = np.load(ty_path)
    betas = np.asarray(betas_tuple, dtype=np.float64)

    mu0, alpha0 = random_init_for_rho(rho_init, M, betas, seed)
    # Rescale mu to roughly match empirical base-rate * (1 - rho_init)
    mu0 = np.asarray(base_rate_per_type, dtype=np.float64) * (1.0 - rho_init)
    mu0 = np.maximum(mu0, 1e-6)

    cfg = Layer1EMConfig(
        betas=betas_tuple,
        max_iter=max_iter,
        tol_relative_ll=tol,
        rho_cap=rho_cap,
        project_spectral=True,
        initial_mu=mu0,
        initial_alpha_all=alpha0,
        record_trace=False,
    )
    t0 = time.time()
    res = fit_layer1_em(times, types, T=T, M=M, config=cfg)
    dt = time.time() - t0

    return {
        "idx": idx,
        "rho_init": float(rho_init),
        "seed": int(seed),
        "rho_final": float(res.spectral_radius),
        "log_likelihood": float(res.log_likelihood),
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        "converged_reason": str(res.converged_reason),
        "projection_count": int(res.projection_count),
        "wall_clock_seconds": float(dt),
    }


def ll_sanity_check(times, types, T, M, betas):
    _ensure_simulator_on_path()
    from simulator.layer1_em import log_likelihood_sumexp

    payload = json.loads(
        (LBFGSB_DIR / "layer1_params_sumexp_4type.json").read_text()
    )
    mu = np.asarray(payload["mu"], dtype=np.float64)
    alpha_all = np.asarray(payload["alpha_all"], dtype=np.float64)
    expected_ll = float(payload["ll"])
    em_ll = log_likelihood_sumexp(
        times, types, T, M, mu, alpha_all, np.asarray(betas, dtype=np.float64)
    )
    delta = em_ll - expected_ll
    return {
        "expected_ll": expected_ll,
        "em_module_ll": em_ll,
        "delta": delta,
        "agree_within_1e-3": abs(delta) < 1e-3,
    }


def main():
    _ensure_simulator_on_path()
    print("=== Layer-1 EM robustness on EURUSD primary day 2021-07-20 ===")
    t_raw, typ_raw = load_day_events("2021-07-20")
    print(f"  loaded N={t_raw.size:,}")
    t, typ = apply_random_ms_jitter(t_raw, typ_raw, JITTER_SEED)

    # Persist jittered events to disk so worker processes share memory cheaply.
    tmp_dir = HERE / ".robustness_tmp"
    tmp_dir.mkdir(exist_ok=True)
    t_path = tmp_dir / "times.npy"
    ty_path = tmp_dir / "types.npy"
    np.save(t_path, t)
    np.save(ty_path, typ)

    M = 4
    base_rate_per_type = np.bincount(typ, minlength=M) / T_DAY

    # 1) LL sanity check
    print("\n[sanity] scoring L-BFGS-B JSON's (mu, alpha) with EM intensity calc...")
    sanity = ll_sanity_check(t, typ, T_DAY, M, BETAS)
    print(
        f"  expected ll = {sanity['expected_ll']:.4f}  "
        f"em_module ll = {sanity['em_module_ll']:.4f}  "
        f"delta = {sanity['delta']:.6e}  "
        f"agree(<1e-3): {sanity['agree_within_1e-3']}"
    )

    # 2) Multi-init EM
    rho_levels = (0.3, 0.5, 0.7, 0.9)
    starts_per_level = 4
    args_list = []
    base_seed = 4242
    idx = 0
    for rho_init in rho_levels:
        for k in range(starts_per_level):
            args_list.append(
                (
                    idx,
                    rho_init,
                    base_seed + idx * 17,
                    600,           # max_iter
                    0.99,          # rho_cap
                    1e-7,          # tol
                    str(t_path),
                    str(ty_path),
                    T_DAY,
                    M,
                    BETAS,
                    base_rate_per_type.tolist(),
                )
            )
            idx += 1
    print(
        f"\n[multi-init] running {len(args_list)} EM fits "
        f"({starts_per_level} per rho_init level) in parallel..."
    )

    runs = []
    n_workers = min(8, len(args_list))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_fit_one, a) for a in args_list]
        for fut in as_completed(futures):
            r = fut.result()
            runs.append(r)
            print(
                f"  idx={r['idx']:>2}  rho_init={r['rho_init']:.1f}  "
                f"-> rho={r['rho_final']:.6f}  ll={r['log_likelihood']:.2f}  "
                f"n_iter={r['n_iter']}  conv={r['converged']}  "
                f"({r['wall_clock_seconds']:.1f}s)"
            )
    total_dt = time.time() - t0
    print(f"\n  total wall time: {total_dt:.1f}s ({total_dt/60:.1f} min)")

    # Sort by idx for stable reporting
    runs = sorted(runs, key=lambda r: r["idx"])

    # 3) Cluster analysis
    rhos = np.array([r["rho_final"] for r in runs])
    lls = np.array([r["log_likelihood"] for r in runs])
    converged = np.array([r["converged"] for r in runs], dtype=bool)
    n_conv = int(np.sum(converged))

    # Identify whether all converged runs cluster around a single rho.
    # Define a cluster as runs within +-0.02 of the median rho among converged.
    if n_conv > 0:
        rho_med = float(np.median(rhos[converged]))
        ll_med = float(np.median(lls[converged]))
        rho_range = float(rhos[converged].max() - rhos[converged].min())
        ll_range = float(lls[converged].max() - lls[converged].min())
    else:
        rho_med = float("nan")
        ll_med = float("nan")
        rho_range = float("nan")
        ll_range = float("nan")

    # Look for a second basin: any run that converged with rho > 0.97 and
    # ll within 10 of the median.
    second_basin_candidates = [
        r for r in runs
        if r["converged"]
        and r["rho_final"] > 0.97
        and abs(r["log_likelihood"] - ll_med) <= 10.0
    ]

    # Also detect a more lenient second basin signature: any cluster whose
    # rho is at least 0.05 from the median converged rho.
    distinct_basin_runs = [
        r for r in runs
        if r["converged"] and abs(r["rho_final"] - rho_med) > 0.05
    ]

    # All converged runs near the median (single-basin verdict)
    single_basin = (
        n_conv == len(runs)
        and rho_range < 0.02
        and ll_range < 10.0
        and len(second_basin_candidates) == 0
    )

    payload = {
        "primary_day": "2021-07-20",
        "n_events": int(t.size),
        "fit_window_seconds": float(T_DAY),
        "betas_grid_per_s": list(BETAS),
        "jitter": {"scheme": "random_ms", "seed": JITTER_SEED},
        "ll_sanity_check": sanity,
        "rho_init_grid": list(rho_levels),
        "starts_per_level": int(starts_per_level),
        "max_iter": 600,
        "tol_relative_ll": 1e-7,
        "rho_cap": 0.99,
        "runs": runs,
        "summary": {
            "n_runs": len(runs),
            "n_converged": n_conv,
            "rho_median_converged": rho_med,
            "ll_median_converged": ll_med,
            "rho_range_converged": rho_range,
            "ll_range_converged": ll_range,
            "single_basin_verdict": single_basin,
            "second_basin_candidates": [r["idx"] for r in second_basin_candidates],
            "distinct_basin_runs": [r["idx"] for r in distinct_basin_runs],
            "total_wall_clock_seconds": float(total_dt),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}")

    print("\n--- Verdict ---")
    print(f"  LL sanity: {'PASS' if sanity['agree_within_1e-3'] else 'FAIL'}")
    print(f"  single-basin verdict: {'YES' if single_basin else 'NO'}")
    print(f"  rho median (converged): {rho_med:.6f}  range: {rho_range:.6f}")
    print(f"  ll median (converged):  {ll_med:.4f}    range: {ll_range:.4f}")
    print(f"  second-basin candidates (rho>0.97 within 10 LL of median): "
          f"{len(second_basin_candidates)}")
    print(f"  distinct-basin runs (|rho - median| > 0.05): "
          f"{len(distinct_basin_runs)}")


if __name__ == "__main__":
    main()
