"""Tests for the ExponentialSumKernel family.

Acceptance criteria from restored_dissertation_pack/04_kernel_surface_expansion.md:
- single-R limit reproduces ExponentialKernel under fixed seed (parity).
- branching-matrix additivity per (i, j, r).
- HawkesSimulator accepts the new kernel via the existing protocol;
  60-second smoke run returns reasonable counts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.config import (
    BaselineParams,
    ExponentialKernelParams,
    ExponentialSumKernelParams,
    KernelFamily,
    NumericalSettings,
    SimulatorConfig,
    validate,
)
from simulator.hawkes_core import HawkesSimulator
from simulator.kernels import (
    ExponentialKernel,
    ExponentialSumKernel,
    branching_matrix,
    build_kernel,
)


def test_exponential_sum_evaluate_at_zero():
    alphas = np.array([[[0.2, 0.3]]])  # (1, 1, 2)
    betas = np.array([[[1.0, 5.0]]])
    k = ExponentialSumKernel(alphas, betas)
    np.testing.assert_allclose(k.evaluate(0.0), np.array([[0.5]]))


def test_exponential_sum_branching_matrix_additivity():
    alphas = np.array([
        [[0.2, 0.3], [0.05, 0.1]],
        [[0.05, 0.05], [0.2, 0.4]],
    ])
    betas = np.array([
        [[1.0, 5.0], [2.0, 10.0]],
        [[1.0, 5.0], [2.0, 10.0]],
    ])
    k = ExponentialSumKernel(alphas, betas)
    Phi = branching_matrix(k)
    expected = np.sum(alphas / betas, axis=2)
    np.testing.assert_allclose(Phi, expected)


def test_exponential_sum_R1_matches_exponential_kernel_exactly():
    """ExponentialSumKernel with R=1 produces the same evaluate, integral,
    upper_bound, and branching matrix as ExponentialKernel for the same
    (alpha, beta).
    """
    alpha = np.array([[0.2, 0.05], [0.1, 0.3]])
    beta = np.array([[1.0, 2.0], [3.0, 4.0]])
    k_exp = ExponentialKernel(alpha=alpha, beta=beta)
    k_sum = ExponentialSumKernel(
        alphas=alpha[..., None], betas=beta[..., None]
    )
    for t in (0.0, 0.5, 1.0, 2.5, 10.0):
        np.testing.assert_allclose(k_exp.evaluate(t), k_sum.evaluate(t))
        np.testing.assert_allclose(k_exp.integral(t), k_sum.integral(t))
    np.testing.assert_allclose(k_exp.upper_bound(5.0), k_sum.upper_bound(5.0))
    np.testing.assert_allclose(branching_matrix(k_exp), branching_matrix(k_sum))


def test_exponential_sum_R1_simulator_parity_under_fixed_seed():
    """Running HawkesSimulator with ExponentialKernel and the equivalent
    ExponentialSumKernel (R=1) under the same seed produces identical
    event sequences."""
    d = 2
    alpha = np.array([[0.2, 0.05], [0.05, 0.2]])
    beta = np.array([[1.5, 2.0], [2.0, 1.5]])
    cfg_exp = SimulatorConfig(
        dimension=d,
        horizon=20.0,
        seed=11,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=np.array([0.5, 0.5])),
        numerics=NumericalSettings(),
    )
    cfg_sum = SimulatorConfig(
        dimension=d,
        horizon=20.0,
        seed=11,
        kernel_family=KernelFamily.EXPONENTIAL_SUM,
        kernel_params=ExponentialSumKernelParams(
            alphas=alpha[..., None],
            betas=beta[..., None],
        ),
        baseline=BaselineParams(mu=np.array([0.5, 0.5])),
        numerics=NumericalSettings(),
    )
    sim_exp = HawkesSimulator(cfg_exp)
    sim_sum = HawkesSimulator(cfg_sum)
    sim_exp.reset(seed=11)
    sim_sum.reset(seed=11)
    trace_exp = sim_exp.simulate()
    trace_sum = sim_sum.simulate()
    np.testing.assert_allclose(trace_exp.times, trace_sum.times, atol=1e-12)
    np.testing.assert_array_equal(trace_exp.event_types, trace_sum.event_types)


def test_build_kernel_dispatches_exponential_sum():
    alphas = np.array([[[0.1]]])
    betas = np.array([[[1.0]]])
    cfg = SimulatorConfig(
        dimension=1,
        horizon=10.0,
        seed=0,
        kernel_family=KernelFamily.EXPONENTIAL_SUM,
        kernel_params=ExponentialSumKernelParams(alphas=alphas, betas=betas),
        baseline=BaselineParams(mu=np.array([1.0])),
        numerics=NumericalSettings(),
    )
    validate(cfg)
    kernel = build_kernel(cfg)
    assert isinstance(kernel, ExponentialSumKernel)


def test_simulator_runs_with_two_component_sumexp_kernel():
    """60-second smoke run with R=2 sum-of-exp returns sensible counts."""
    d = 2
    alphas = np.zeros((d, d, 2))
    betas = np.zeros((d, d, 2))
    alphas[..., 0] = 0.05
    betas[..., 0] = 0.5
    alphas[..., 1] = 0.05
    betas[..., 1] = 5.0
    np.fill_diagonal(alphas[..., 0], 0.15)
    np.fill_diagonal(alphas[..., 1], 0.15)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=60.0,
        seed=0,
        kernel_family=KernelFamily.EXPONENTIAL_SUM,
        kernel_params=ExponentialSumKernelParams(alphas=alphas, betas=betas),
        baseline=BaselineParams(mu=np.array([0.5, 0.5])),
        numerics=NumericalSettings(),
    )
    sim = HawkesSimulator(cfg)
    sim.reset(seed=0)
    trace = sim.simulate()
    assert trace.num_events > 0
    assert trace.num_events < 100_000  # sanity: not blowing up
    assert np.all(np.diff(trace.times) >= 0.0)
