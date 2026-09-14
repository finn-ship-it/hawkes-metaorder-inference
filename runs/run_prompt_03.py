"""Driver for prompt 03 — spread-scaling sweep at production scale.

Sweep horizons T in {30, 60, 120, 240, 480, 960} on first_milestone.json,
30 seeds per horizon. Idempotent: skips horizons whose ensemble dir
already exists.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import statistics
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

import numpy as np

from simulator.config import config_from_json
from simulator.ensemble import run_ensemble

DATE = "2026-05-06"
SWEEP_ROOT = os.path.join(REPO, "runs", f"spread_scaling_{DATE}")
CONFIG = os.path.join(REPO, "configs", "first_milestone.json")
HORIZONS = (30.0, 60.0, 120.0, 240.0, 480.0, 960.0)
SEEDS = tuple(range(1, 31))

# Spread fields whose horizon-mean should be log-log fit against N_obs_mean.
SPREAD_FIELDS = (
    "observation.spread_mean_ticks",
    "observation.spread_median_ticks",
    "observation.spread_p90_ticks",
    "observation.spread_p99_ticks",
    "observation.spread_max_ticks",
    "observation.final_spread_ticks",
)


def _final_dir_for_horizon(T: float) -> str:
    return os.path.join(SWEEP_ROOT, f"T{int(T):03d}")


def _aggregate_for_horizon(T: float) -> dict:
    final_dir = _final_dir_for_horizon(T)
    inner = next(
        os.path.join(final_dir, d)
        for d in os.listdir(final_dir)
        if os.path.isdir(os.path.join(final_dir, d)) and d.startswith("ensemble_")
    )
    es = json.load(open(os.path.join(inner, "ensemble_summary.json")))
    agg = es["aggregate"]
    out = {"T": T, "inner_ensemble_dir": inner, "n_members": es["num_members"]}
    out["N_obs_mean"] = agg["observation.num_observed_events"]["mean"]
    out["N_obs_std"] = agg["observation.num_observed_events"]["std"]
    for field in SPREAD_FIELDS:
        rec = agg.get(field, {})
        if isinstance(rec, dict) and rec.get("kind") == "scalar":
            out[field + ".mean"] = rec["mean"]
            out[field + ".std"] = rec["std"]
        else:
            out[field + ".mean"] = None
            out[field + ".std"] = None
    return out


def run_one(T: float) -> str:
    final_dir = _final_dir_for_horizon(T)
    if os.path.isdir(final_dir):
        print(f"[T={T}] already complete — reusing {final_dir}")
        return final_dir
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    base_cfg = config_from_json(CONFIG)
    cfg = dataclasses.replace(base_cfg, horizon=float(T))
    os.makedirs(final_dir, exist_ok=False)
    t0 = time.time()
    inner = run_ensemble(
        cfg,
        seeds=SEEDS,
        runs_root=final_dir,
        observe=True,
        envelope_path=None,
        timestamp_prefix=f"spread_T{int(T):03d}",
    )
    elapsed = time.time() - t0
    print(f"[T={T}] inner={inner} ({elapsed:.1f} s)")
    return final_dir


def loglog_fit(N: list[float], y: list[float]) -> dict:
    """OLS fit of log(y) = log(a) + b*log(N). Returns a, b, R²."""
    if any(v is None or v <= 0 for v in y) or any(n is None or n <= 0 for n in N):
        return {"a": None, "b": None, "r2": None, "n": 0}
    xs = np.log(np.asarray(N, dtype=float))
    ys = np.log(np.asarray(y, dtype=float))
    if len(xs) < 2:
        return {"a": None, "b": None, "r2": None, "n": int(len(xs))}
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "a": float(math.exp(intercept)),
        "b": float(slope),
        "r2": (None if r2 is None else float(r2)),
        "n": int(len(xs)),
    }


def main():
    print(f"SWEEP_ROOT = {SWEEP_ROOT}")
    print(f"horizons = {HORIZONS}")
    print(f"seeds = 1..30")

    for T in HORIZONS:
        run_one(T)

    rows = [_aggregate_for_horizon(T) for T in HORIZONS]

    csv_path = os.path.join(SWEEP_ROOT, "spread_scaling_summary.csv")
    headers = ["T", "n_members", "N_obs_mean", "N_obs_std"]
    for f in SPREAD_FIELDS:
        headers.extend([f + ".mean", f + ".std"])
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h, "") for h in headers])

    fits = {}
    Ns = [r["N_obs_mean"] for r in rows]
    for f in SPREAD_FIELDS:
        ys = [r[f + ".mean"] for r in rows]
        fits[f] = loglog_fit(Ns, ys)
    fit_path = os.path.join(SWEEP_ROOT, "spread_scaling_fit.json")
    with open(fit_path, "w") as fh:
        json.dump(
            {
                "model": "field_mean = a * N_obs^b",
                "horizons": list(HORIZONS),
                "n_seeds_per_horizon": len(SEEDS),
                "fits": fits,
                "rows": rows,
            },
            fh,
            indent=2,
        )

    print(f"summary CSV: {csv_path}")
    print(f"fit JSON:    {fit_path}")
    print(json.dumps({"fits": fits}, indent=2))


if __name__ == "__main__":
    main()
