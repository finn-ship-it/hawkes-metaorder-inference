"""Wall-clock benchmark across the three implemented kernel families.

Times a 10-second d=4 Hawkes simulation under each kernel family on a
fixed seed grid; reports the median wall-clock time across 20 runs and
writes `dissertation_artifacts/data/kernel_benchmark.csv`.

Usage:
    PYTHONPATH=src python3 -m simulator.tests.benchmarks.bench_kernels
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from simulator.config import (
    BaselineParams,
    ExponentialKernelParams,
    ExponentialSumKernelParams,
    KernelFamily,
    NumericalSettings,
    PowerLawCutoffKernelParams,
    SimulatorConfig,
)
from simulator.hawkes_core import HawkesSimulator


HORIZON = 10.0
DIMENSION = 4
N_RUNS = 20


def _exp_cfg(seed: int) -> SimulatorConfig:
    d = DIMENSION
    alpha = 0.05 * np.ones((d, d))
    np.fill_diagonal(alpha, 0.20)
    beta = 2.0 * np.ones((d, d))
    return SimulatorConfig(
        dimension=d,
        horizon=HORIZON,
        seed=int(seed),
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=np.full(d, 0.5)),
        numerics=NumericalSettings(),
    )


def _sumexp_cfg(seed: int) -> SimulatorConfig:
    d = DIMENSION
    R = 2
    alphas = np.zeros((d, d, R))
    betas = np.zeros((d, d, R))
    alphas[..., 0] = 0.04
    betas[..., 0] = 0.5
    alphas[..., 1] = 0.04
    betas[..., 1] = 5.0
    np.fill_diagonal(alphas[..., 0], 0.10)
    np.fill_diagonal(alphas[..., 1], 0.10)
    return SimulatorConfig(
        dimension=d,
        horizon=HORIZON,
        seed=int(seed),
        kernel_family=KernelFamily.EXPONENTIAL_SUM,
        kernel_params=ExponentialSumKernelParams(alphas=alphas, betas=betas),
        baseline=BaselineParams(mu=np.full(d, 0.5)),
        numerics=NumericalSettings(),
    )


def _powerlaw_cfg(seed: int) -> SimulatorConfig:
    d = DIMENSION
    alpha = 0.05 * np.ones((d, d))
    np.fill_diagonal(alpha, 0.18)
    beta = 2.5 * np.ones((d, d))
    gamma = 1.0 * np.ones((d, d))
    return SimulatorConfig(
        dimension=d,
        horizon=HORIZON,
        seed=int(seed),
        kernel_family=KernelFamily.POWERLAW_CUTOFF,
        kernel_params=PowerLawCutoffKernelParams(
            alpha=alpha, beta=beta, gamma=gamma, t_max=8.0
        ),
        baseline=BaselineParams(mu=np.full(d, 0.5)),
        numerics=NumericalSettings(),
    )


def _bench_one(cfg: SimulatorConfig) -> tuple:
    sim = HawkesSimulator(cfg)
    sim.reset(seed=cfg.seed)
    t0 = time.perf_counter()
    trace = sim.simulate()
    elapsed = time.perf_counter() - t0
    return elapsed, trace.num_events


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[4]
        / "dissertation_artifacts"
        / "data"
        / "kernel_benchmark.csv"
    )
    rows = []
    for family_name, builder in (
        ("EXPONENTIAL", _exp_cfg),
        ("EXPONENTIAL_SUM", _sumexp_cfg),
        ("POWERLAW_CUTOFF", _powerlaw_cfg),
    ):
        timings = []
        counts = []
        for r in range(N_RUNS):
            elapsed, n = _bench_one(builder(seed=42 + r))
            timings.append(elapsed)
            counts.append(n)
        rows.append(
            {
                "kernel_family": family_name,
                "horizon_seconds": HORIZON,
                "dimension": DIMENSION,
                "n_runs": N_RUNS,
                "median_wall_clock_seconds": float(np.median(timings)),
                "mean_wall_clock_seconds": float(np.mean(timings)),
                "min_wall_clock_seconds": float(np.min(timings)),
                "max_wall_clock_seconds": float(np.max(timings)),
                "median_event_count": int(np.median(counts)),
                "mean_event_count": float(np.mean(counts)),
            }
        )
        print(
            f"{family_name:>16}: median {rows[-1]['median_wall_clock_seconds']*1000:.2f} ms  "
            f"events {rows[-1]['median_event_count']}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
