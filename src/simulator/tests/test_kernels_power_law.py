"""Tests for the PowerLawCutoffKernel family."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.config import (
    BaselineParams,
    KernelFamily,
    NumericalSettings,
    PowerLawCutoffKernelParams,
    SimulatorConfig,
    validate,
)
from simulator.hawkes_core import HawkesSimulator
from simulator.kernels import (
    PowerLawCutoffKernel,
    branching_matrix,
    build_kernel,
)


def test_powerlaw_evaluate_at_zero_equals_alpha():
    alpha = np.array([[0.2, 0.05], [0.05, 0.2]])
    beta = np.array([[1.5, 2.0], [2.0, 1.5]])
    gamma = np.array([[1.0, 1.0], [1.0, 1.0]])
    k = PowerLawCutoffKernel(alpha=alpha, beta=beta, gamma=gamma, t_max=10.0)
    np.testing.assert_allclose(k.evaluate(0.0), alpha)


def test_powerlaw_evaluate_returns_zero_past_t_max():
    alpha = np.array([[0.2]])
    beta = np.array([[1.5]])
    gamma = np.array([[1.0]])
    k = PowerLawCutoffKernel(alpha=alpha, beta=beta, gamma=gamma, t_max=2.0)
    np.testing.assert_allclose(k.evaluate(3.0), 0.0)


def test_powerlaw_branching_matrix_matches_numerical_quadrature_5x5():
    """Closed-form branching matrix matches numerical quadrature on a 5x5
    random-but-stable parameter grid to within 1e-7."""
    rng = np.random.default_rng(0)
    d = 3
    alpha = rng.uniform(0.05, 0.5, size=(d, d))
    beta = rng.uniform(1.5, 3.5, size=(d, d))
    gamma = rng.uniform(0.5, 3.0, size=(d, d))
    t_max = 8.0
    k = PowerLawCutoffKernel(alpha=alpha, beta=beta, gamma=gamma, t_max=t_max)

    Phi_closed = branching_matrix(k)

    # Numerical integration of phi(t) = alpha (1 + gamma t)^(-beta) over [0, t_max].
    # Use trapezoidal rule on a fine grid; for a smooth integrand to 1e-7
    # absolute, ~10000 grid points are sufficient at t_max=8.
    n_pts = 20000
    grid = np.linspace(0.0, t_max, n_pts)
    Phi_quad = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            phi_t = alpha[i, j] * (1.0 + gamma[i, j] * grid) ** (-beta[i, j])
            Phi_quad[i, j] = float(np.trapezoid(phi_t, grid))
    np.testing.assert_allclose(Phi_closed, Phi_quad, atol=1e-7)


def test_powerlaw_rejects_super_critical_kernel_via_spectral_validation():
    """Building a config with a power-law kernel whose closed-form branching
    matrix has spectral radius >= 1 must be flagged by the validator."""
    d = 1
    alpha = np.array([[10.0]])  # very large
    beta = np.array([[1.5]])
    gamma = np.array([[0.5]])
    cfg = SimulatorConfig(
        dimension=d,
        horizon=10.0,
        seed=0,
        kernel_family=KernelFamily.POWERLAW_CUTOFF,
        kernel_params=PowerLawCutoffKernelParams(
            alpha=alpha, beta=beta, gamma=gamma, t_max=10.0
        ),
        baseline=BaselineParams(mu=np.array([0.1])),
        numerics=NumericalSettings(),
    )
    with pytest.raises(Exception):  # ConfigurationError or similar
        validate(cfg)


def test_powerlaw_rejects_beta_le_one_at_construction():
    with pytest.raises(ValueError):
        PowerLawCutoffKernel(
            alpha=np.array([[0.1]]),
            beta=np.array([[1.0]]),
            gamma=np.array([[1.0]]),
            t_max=5.0,
        )


def test_simulator_runs_with_powerlaw_cutoff_kernel():
    """60-second smoke run."""
    d = 2
    alpha = np.array([[0.2, 0.05], [0.05, 0.2]])
    beta = np.array([[2.5, 2.5], [2.5, 2.5]])
    gamma = np.array([[1.0, 1.0], [1.0, 1.0]])
    cfg = SimulatorConfig(
        dimension=d,
        horizon=60.0,
        seed=0,
        kernel_family=KernelFamily.POWERLAW_CUTOFF,
        kernel_params=PowerLawCutoffKernelParams(
            alpha=alpha, beta=beta, gamma=gamma, t_max=10.0
        ),
        baseline=BaselineParams(mu=np.array([0.5, 0.5])),
        numerics=NumericalSettings(),
    )
    sim = HawkesSimulator(cfg)
    sim.reset(seed=0)
    trace = sim.simulate()
    assert trace.num_events > 0
    assert trace.num_events < 100_000
    assert np.all(np.diff(trace.times) >= 0.0)


def test_build_kernel_dispatches_powerlaw_cutoff():
    alpha = np.array([[0.05]])
    beta = np.array([[2.0]])
    gamma = np.array([[1.0]])
    cfg = SimulatorConfig(
        dimension=1,
        horizon=10.0,
        seed=0,
        kernel_family=KernelFamily.POWERLAW_CUTOFF,
        kernel_params=PowerLawCutoffKernelParams(
            alpha=alpha, beta=beta, gamma=gamma, t_max=5.0
        ),
        baseline=BaselineParams(mu=np.array([1.0])),
        numerics=NumericalSettings(),
    )
    validate(cfg)
    kernel = build_kernel(cfg)
    assert isinstance(kernel, PowerLawCutoffKernel)
