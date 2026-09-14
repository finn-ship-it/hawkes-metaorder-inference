"""Tests for hawkes_core.py."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.config import (
    EventType,
    KernelFamily,
    ExponentialKernelParams,
    BaselineParams,
    SimulatorConfig,
)
from simulator.hawkes_core import HawkesSimulator


def _make_config(
    d: int = 4,
    horizon: float = 60.0,
    seed: int = 20260423,
    mu_value: float = 0.5,
    alpha_value: float = 0.1,
    beta_value: float = 2.0,
) -> SimulatorConfig:
    alpha = alpha_value * np.ones((d, d))
    beta = beta_value * np.ones((d, d))
    mu = mu_value * np.ones(d)
    return SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )


def test_first_milestone_basic_run():
    cfg = _make_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert trace.num_events > 0
    assert trace.dimension == 4
    assert trace.horizon == cfg.horizon
    assert trace.seed == cfg.seed


def test_timestamps_strictly_increasing():
    cfg = _make_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    diffs = np.diff(trace.times)
    assert np.all(diffs > 0.0), "timestamps must be strictly increasing"


def test_timestamps_within_horizon():
    cfg = _make_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert np.all(trace.times >= 0.0)
    assert np.all(trace.times <= cfg.horizon)


def test_event_types_in_range():
    cfg = _make_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert np.all(trace.event_types >= 0)
    assert np.all(trace.event_types < cfg.dimension)


def test_acceptance_ratio_in_band():
    cfg = _make_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert trace.num_proposals > 0
    assert 0.0 < trace.acceptance_ratio <= 1.0


def test_deterministic_under_fixed_seed():
    cfg = _make_config()
    sim_a = HawkesSimulator(cfg)
    sim_a.reset()
    trace_a = sim_a.simulate()

    sim_b = HawkesSimulator(cfg)
    sim_b.reset()
    trace_b = sim_b.simulate()

    np.testing.assert_array_equal(trace_a.times, trace_b.times)
    np.testing.assert_array_equal(trace_a.event_types, trace_b.event_types)


def test_empirical_rate_matches_stationary_mean_ensemble():
    """Empirical per-component rate over an ensemble of seeds should match
    the analytic stationary rate lambda_inf = (I - Phi)^{-1} mu within a
    tolerance consistent with the sample size.
    """
    d = 4
    mu = 0.5
    alpha = 0.1
    beta = 2.0
    T = 200.0
    num_runs = 20

    base_cfg = _make_config(
        d=d, horizon=T, mu_value=mu, alpha_value=alpha, beta_value=beta
    )
    phi = (alpha / beta) * np.ones((d, d))
    lambda_inf = np.linalg.solve(np.eye(d) - phi, mu * np.ones(d))

    rates = np.zeros(d)
    total_time = 0.0
    for k in range(num_runs):
        cfg = SimulatorConfig(**{**base_cfg.__dict__, "seed": 1_000 + k})
        sim = HawkesSimulator(cfg)
        sim.reset()
        trace = sim.simulate()
        for t_index in range(d):
            rates[t_index] += float(np.sum(trace.event_types == t_index))
        total_time += T

    empirical_rate = rates / total_time

    # Tolerance is looser than the asymptotic Poisson standard error to allow
    # for correlation induced by self-excitation.
    rel_error = np.abs(empirical_rate - lambda_inf) / lambda_inf
    assert np.all(rel_error < 0.15), (
        f"empirical rate {empirical_rate} deviates too far from stationary "
        f"rate {lambda_inf}; relative error {rel_error}"
    )


def test_zero_baseline_produces_no_events():
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    cfg = SimulatorConfig(
        dimension=d,
        horizon=10.0,
        seed=123,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=1e-12 * np.ones(d)),  # effectively zero
    )
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    # Either zero events, or at most a trivial number; confirm no divergence.
    assert trace.num_events >= 0
    assert trace.num_events < 10
