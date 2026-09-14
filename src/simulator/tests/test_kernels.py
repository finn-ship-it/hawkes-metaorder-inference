"""Tests for kernels.py."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.config import (
    ExponentialKernelParams,
    BaselineParams,
    KernelFamily,
    SimulatorConfig,
)
from simulator.kernels import (
    ExponentialKernel,
    build_kernel,
    branching_matrix,
    spectral_radius,
)


def test_exponential_evaluate_at_zero():
    alpha = np.array([[0.2, 0.1], [0.3, 0.4]])
    beta = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    np.testing.assert_allclose(k.evaluate(0.0), alpha)


def test_exponential_evaluate_monotone_decreasing():
    alpha = np.array([[0.5]])
    beta = np.array([[2.0]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    v0 = float(k.evaluate(0.0).item())
    v1 = float(k.evaluate(1.0).item())
    v2 = float(k.evaluate(2.0).item())
    assert v0 > v1 > v2
    np.testing.assert_allclose(v1, 0.5 * np.exp(-2.0 * 1.0))


def test_exponential_integral_analytic():
    alpha = np.array([[0.4, 0.2], [0.1, 0.3]])
    beta = np.array([[1.0, 2.0], [3.0, 0.5]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    t = 1.5
    expected = (alpha / beta) * (1.0 - np.exp(-beta * t))
    np.testing.assert_allclose(k.integral(t), expected)


def test_exponential_integral_limit_equals_branching():
    alpha = np.array([[0.4, 0.2], [0.1, 0.3]])
    beta = np.array([[1.0, 2.0], [3.0, 0.5]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    # large t
    np.testing.assert_allclose(k.integral(1e6), alpha / beta, atol=1e-9)
    np.testing.assert_allclose(branching_matrix(k), alpha / beta)


def test_upper_bound_equals_alpha_for_exponential():
    alpha = np.array([[0.1, 0.5], [0.2, 0.3]])
    beta = np.array([[1.0, 2.0], [0.5, 4.0]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    np.testing.assert_allclose(k.upper_bound(10.0), alpha)


def test_negative_t_rejected():
    k = ExponentialKernel(alpha=np.ones((2, 2)), beta=np.ones((2, 2)))
    with pytest.raises(ValueError):
        k.evaluate(-0.1)
    with pytest.raises(ValueError):
        k.integral(-0.1)


def test_build_kernel_exponential():
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 1.0 * np.ones((d, d))
    cfg = SimulatorConfig(
        dimension=d,
        horizon=10.0,
        seed=0,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=np.ones(d)),
    )
    k = build_kernel(cfg)
    assert isinstance(k, ExponentialKernel)
    assert k.dimension == d


def test_spectral_radius_matches_eigvals():
    alpha = np.array([[0.2, 0.1], [0.05, 0.3]])
    beta = np.array([[1.0, 1.0], [1.0, 1.0]])
    k = ExponentialKernel(alpha=alpha, beta=beta)
    rho = spectral_radius(k)
    expected = float(np.max(np.abs(np.linalg.eigvals(alpha / beta))))
    assert abs(rho - expected) < 1e-12
