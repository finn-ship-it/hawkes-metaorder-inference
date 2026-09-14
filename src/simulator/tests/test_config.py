"""Tests for config.py."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from simulator.config import (
    EventType,
    KernelFamily,
    ExponentialKernelParams,
    BaselineParams,
    LatentRegimeSpec,
    MetaOrderSpec,
    NumericalSettings,
    SimulatorConfig,
    ConfigurationError,
    validate,
    validate_first_milestone,
    config_to_json,
    config_from_json,
)


def _make_valid_config(dimension: int = 4, seed: int = 20260423) -> SimulatorConfig:
    d = dimension
    alpha = 0.1 * np.ones((d, d))
    beta = 1.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    return SimulatorConfig(
        dimension=d,
        horizon=60.0,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )


def test_event_type_values():
    assert EventType.BID_UP == 0
    assert EventType.BID_DOWN == 1
    assert EventType.ASK_UP == 2
    assert EventType.ASK_DOWN == 3


def test_valid_config_passes_validation():
    cfg = _make_valid_config()
    validate(cfg)
    validate_first_milestone(cfg)


def test_shape_mismatch_baseline_rejected():
    cfg = _make_valid_config()
    bad = SimulatorConfig(**{**cfg.__dict__, "baseline": BaselineParams(mu=np.ones(3))})
    with pytest.raises(ConfigurationError):
        validate(bad)


def test_shape_mismatch_alpha_rejected():
    cfg = _make_valid_config()
    bad_kp = ExponentialKernelParams(alpha=np.ones((3, 3)), beta=np.ones((4, 4)))
    with pytest.raises(ConfigurationError):
        validate(SimulatorConfig(**{**cfg.__dict__, "kernel_params": bad_kp}))


def test_negative_mu_rejected():
    cfg = _make_valid_config()
    bad = BaselineParams(mu=-np.ones(4))
    with pytest.raises(ConfigurationError):
        validate(SimulatorConfig(**{**cfg.__dict__, "baseline": bad}))


def test_negative_alpha_rejected():
    cfg = _make_valid_config()
    bad_alpha = -np.ones((4, 4))
    bad_kp = ExponentialKernelParams(alpha=bad_alpha, beta=np.ones((4, 4)))
    with pytest.raises(ConfigurationError):
        validate(SimulatorConfig(**{**cfg.__dict__, "kernel_params": bad_kp}))


def test_nonpositive_beta_rejected():
    cfg = _make_valid_config()
    bad_kp = ExponentialKernelParams(alpha=np.ones((4, 4)), beta=np.zeros((4, 4)))
    with pytest.raises(ConfigurationError):
        validate(SimulatorConfig(**{**cfg.__dict__, "kernel_params": bad_kp}))


def test_spectral_radius_cap_rejected():
    # Phi_ij = alpha / beta = 1.0 for every entry; rho(Phi) = d = 4 -> rejected.
    cfg = _make_valid_config()
    bad_kp = ExponentialKernelParams(alpha=np.ones((4, 4)), beta=np.ones((4, 4)))
    with pytest.raises(ConfigurationError):
        validate(SimulatorConfig(**{**cfg.__dict__, "kernel_params": bad_kp}))


def test_json_roundtrip(tmp_path):
    cfg = _make_valid_config()
    path = os.path.join(tmp_path, "cfg.json")
    config_to_json(cfg, path)
    cfg2 = config_from_json(path)
    assert cfg2.dimension == cfg.dimension
    assert cfg2.horizon == cfg.horizon
    assert cfg2.seed == cfg.seed
    assert cfg2.kernel_family is cfg.kernel_family
    np.testing.assert_array_equal(cfg2.kernel_params.alpha, cfg.kernel_params.alpha)
    np.testing.assert_array_equal(cfg2.kernel_params.beta, cfg.kernel_params.beta)
    np.testing.assert_array_equal(cfg2.baseline.mu, cfg.baseline.mu)
    assert cfg2.latent_regime.num_regimes == cfg.latent_regime.num_regimes
    assert cfg2.meta_order.enabled == cfg.meta_order.enabled
    assert cfg2.numerics.truncation_window == cfg.numerics.truncation_window


def test_first_milestone_rejects_wrong_dimension():
    cfg = _make_valid_config(dimension=3)
    # validate_shapes passes for d=3, but first-milestone gate rejects it.
    # Build a d=3 config with sub-unit spectral radius for a clean test.
    with pytest.raises(ConfigurationError):
        validate_first_milestone(cfg)
