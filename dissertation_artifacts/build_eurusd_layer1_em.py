"""Build artefact: dissertation_artifacts/data/eurusd_layer1_em_summary.json.

Runs the sum-of-exp Hawkes EM (simulator.layer1_em.fit_layer1_em) on the
2021-07-20 EURUSD primary-day events and the 2021-07-19 robustness-day
events, warm-starting from the L-BFGS-B sum-of-exp anchor in
Data/Data output/EURUSD output/layer1/layer1_params_sumexp_4type{,_day_2021-07-19}.json.

Acceptance bands (from restored_dissertation_pack/01_layer1_em_calibration.md):
- 2021-07-20: rho in [0.985, 0.992], N == 97,952, EM converged.
- 2021-07-19: rho in [0.940, 0.946].

Both fits use the same `random_ms` jitter (uniform [0, 1 ms], seed 20210720)
as Data/layer1_hawkes_fit.py, so the event multisets are identical to the
historical anchor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent  # "Work on Hawkes"
PARQUET = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "eurusd_events.parquet"
LBFGSB_DIR = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "layer1"
OUT_PATH = HERE / "data" / "eurusd_layer1_em_summary.json"

PRIMARY_4TYPE = ("bid_up", "bid_down", "ask_up", "ask_down")
JITTER_SEED = 20210720
BETAS = (0.1, 1.0, 10.0, 100.0)


def _ensure_simulator_on_path() -> None:
    import sys

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_day_events(date_str: str):
    """Load a HistData provider-time day (fixed EST); return elapsed seconds/types."""
    type_to_idx = {t: i for i, t in enumerate(PRIMARY_4TYPE)}
    df = pd.read_parquet(
        PARQUET,
        columns=["timestamp", "event_type"],
        filters=[("event_type", "in", list(PRIMARY_4TYPE))],
    )
    day = pd.Timestamp(date_str)
    mask = (df["timestamp"] >= day) & (df["timestamp"] < day + pd.Timedelta(days=1))
    df = df.loc[mask].copy()
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    t = (df["timestamp"] - day).dt.total_seconds().to_numpy(dtype=np.float64)
    typ = df["event_type"].map(type_to_idx).to_numpy(dtype=np.int64)
    return t, typ


def apply_random_ms_jitter(t: np.ndarray, typ: np.ndarray, seed: int):
    """Match Data/layer1_hawkes_fit.py:apply_jitter scheme='random_ms'."""
    rng = np.random.default_rng(seed)
    t_j = t + rng.uniform(0.0, 1.0e-3, size=t.shape[0])
    order = np.argsort(t_j, kind="stable")
    return t_j[order], typ[order]


def load_warm_start(json_path: Path):
    payload = json.loads(json_path.read_text())
    mu = np.asarray(payload["mu"], dtype=np.float64)
    alpha_all = np.asarray(payload["alpha_all"], dtype=np.float64)
    rho = float(payload["spectral_radius"])
    ll = float(payload["ll"])
    converged = bool(payload["converged"])
    return mu, alpha_all, rho, ll, converged


def fit_one_day(date_str: str, lbfgsb_json_path: Path, max_iter: int = 200) -> dict:
    _ensure_simulator_on_path()
    from simulator.layer1_em import Layer1EMConfig, fit_layer1_em

    print(f"\n=== EURUSD {date_str} ===")
    t_raw, typ_raw = load_day_events(date_str)
    print(f"  loaded N={t_raw.size:,}")
    t, typ = apply_random_ms_jitter(t_raw, typ_raw, JITTER_SEED)
    print(f"  jittered (random_ms, seed={JITTER_SEED})")

    mu0, alpha0, rho0, ll0, lbfgsb_conv = load_warm_start(lbfgsb_json_path)
    print(
        f"  L-BFGS-B warm start: rho={rho0:.6f} ll={ll0:.2f} "
        f"converged={lbfgsb_conv}"
    )

    cfg = Layer1EMConfig(
        betas=BETAS,
        max_iter=max_iter,
        tol_relative_ll=1e-8,
        rho_cap=0.99,
        project_spectral=True,
        initial_mu=mu0,
        initial_alpha_all=alpha0,
        record_trace=True,
    )
    t0 = time.time()
    res = fit_one_day_em(t, typ, cfg)
    dt = time.time() - t0
    print(
        f"  EM finished in {dt:.1f}s: rho={res.spectral_radius:.6f} "
        f"ll={res.log_likelihood:.2f} n_iter={res.n_iter} "
        f"converged={res.converged} reason={res.converged_reason}"
    )

    return {
        "date": date_str,
        "n_events": int(res.n_events),
        "fit_window_seconds": float(res.fit_window_seconds),
        "betas": list(res.betas.tolist()),
        "lbfgsb_warm_start": {
            "rho_spec": rho0,
            "log_likelihood": ll0,
            "converged": lbfgsb_conv,
            "source_file": str(lbfgsb_json_path),
        },
        "em_result": {
            "rho_spec": float(res.spectral_radius),
            "log_likelihood": float(res.log_likelihood),
            "n_iter": int(res.n_iter),
            "converged": bool(res.converged),
            "converged_reason": str(res.converged_reason),
            "rho_cap": float(res.rho_cap),
            "projection_count": int(res.projection_count),
            "ll_improvement_over_lbfgsb": float(res.log_likelihood - ll0),
            "wall_clock_seconds": float(dt),
            "mu": res.mu.tolist(),
            "branching_matrix": res.branching_matrix.tolist(),
        },
    }


def fit_one_day_em(t, typ, cfg):
    from simulator.layer1_em import fit_layer1_em

    T = 86400.0
    return fit_layer1_em(t, typ, T=T, M=4, config=cfg)


def main():
    primary = fit_one_day(
        "2021-07-20", LBFGSB_DIR / "layer1_params_sumexp_4type.json", max_iter=600
    )
    secondary = fit_one_day(
        "2021-07-19",
        LBFGSB_DIR / "layer1_params_sumexp_4type_day_2021-07-19.json",
        max_iter=600,
    )

    payload = {
        "preferred_model": "Layer1EM_SumExp_4type_R4",
        "kernel_family": "EXPONENTIAL_SUM",
        "betas_grid_per_s": list(BETAS),
        "data_window": (
            "2021-07-20 (primary), 2021-07-19 (robustness); EURUSD HistData; "
            "single provider-time day each (fixed EST), T=86,400 s; random_ms jitter, seed 20210720"
        ),
        "primary_date": "2021-07-20",
        "secondary_date": "2021-07-19",
        "anchor_lbfgsb_csv": (
            "Data/Data output/EURUSD output/layer1/layer1_models_summary.csv"
        ),
        "primary_day_2021_07_20": primary,
        "day_2021_07_19": secondary,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # Acceptance bands
    rho1 = primary["em_result"]["rho_spec"]
    rho2 = secondary["em_result"]["rho_spec"]
    n1 = primary["n_events"]
    n2 = secondary["n_events"]
    print("\n--- Acceptance check ---")
    p1_ok = (0.985 <= rho1 <= 0.992) and (n1 == 97_952) and primary["em_result"]["converged"]
    p2_ok = (0.940 <= rho2 <= 0.946) and (n2 == 104_734)
    print(f"  primary 2021-07-20: rho={rho1:.6f} N={n1}  -> {'PASS' if p1_ok else 'FAIL'}")
    print(f"  secondary 2021-07-19: rho={rho2:.6f} N={n2} -> {'PASS' if p2_ok else 'FAIL'}")
    if not (p1_ok and p2_ok):
        raise SystemExit("Acceptance bands not met.")


if __name__ == "__main__":
    main()
