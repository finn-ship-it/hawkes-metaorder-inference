"""Tests for the latent-regime channel."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from simulator.config import (
    SimulatorConfig,
    ExponentialKernelParams,
    BaselineParams,
    LatentRegimeSpec,
    KernelFamily,
    ConfigurationError,
    validate,
)
from simulator.latent import RegimeProcess, baseline_for_regime
from simulator.hawkes_core import HawkesSimulator
from simulator.artifact import save_run, load_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_regime_config(
    horizon: float = 60.0,
    seed: int = 20260423,
    mu_calm: float = 0.3,
    mu_active: float = 1.5,
    q_calm_to_active: float = 0.10,
    q_active_to_calm: float = 0.20,
    initial_regime: int = 0,
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = np.array(
        [
            [mu_calm] * d,
            [mu_active] * d,
        ],
        dtype=float,
    )
    Q = np.array(
        [
            [-q_calm_to_active, q_calm_to_active],
            [q_active_to_calm, -q_active_to_calm],
        ],
        dtype=float,
    )
    return SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
        latent_regime=LatentRegimeSpec(
            num_regimes=2, transition_rates=Q, initial_regime=initial_regime
        ),
    )


def _single_regime_config(seed: int = 20260423) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    return SimulatorConfig(
        dimension=d,
        horizon=60.0,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_two_regime_config_passes_validate():
    validate(_two_regime_config())


def test_reject_mu_wrong_shape_for_two_regimes():
    cfg = _two_regime_config()
    bad_mu = np.array([0.3, 0.3, 0.3, 0.3])  # shape (d,), invalid for K=2
    bad = SimulatorConfig(
        **{**cfg.__dict__, "baseline": BaselineParams(mu=bad_mu)}
    )
    with pytest.raises(ConfigurationError):
        validate(bad)


def test_reject_transition_rates_row_sum_nonzero():
    cfg = _two_regime_config()
    Q_bad = np.array([[-0.1, 0.2], [0.2, -0.2]])  # row 0 sums to 0.1
    bad = SimulatorConfig(
        **{
            **cfg.__dict__,
            "latent_regime": LatentRegimeSpec(
                num_regimes=2, transition_rates=Q_bad, initial_regime=0
            ),
        }
    )
    with pytest.raises(ConfigurationError):
        validate(bad)


def test_reject_negative_off_diagonal_rate():
    cfg = _two_regime_config()
    Q_bad = np.array([[0.0, -0.1], [0.2, -0.2]])
    bad = SimulatorConfig(
        **{
            **cfg.__dict__,
            "latent_regime": LatentRegimeSpec(
                num_regimes=2, transition_rates=Q_bad, initial_regime=0
            ),
        }
    )
    with pytest.raises(ConfigurationError):
        validate(bad)


def test_reject_initial_regime_out_of_range():
    cfg = _two_regime_config()
    with pytest.raises(ConfigurationError):
        validate(
            SimulatorConfig(
                **{
                    **cfg.__dict__,
                    "latent_regime": LatentRegimeSpec(
                        num_regimes=2,
                        transition_rates=cfg.latent_regime.transition_rates,
                        initial_regime=5,
                    ),
                }
            )
        )


def test_reject_transition_rates_for_single_regime():
    d = 4
    cfg = SimulatorConfig(
        dimension=d,
        horizon=10.0,
        seed=0,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(
            alpha=0.1 * np.ones((d, d)), beta=2.0 * np.ones((d, d))
        ),
        baseline=BaselineParams(mu=0.5 * np.ones(d)),
        latent_regime=LatentRegimeSpec(
            num_regimes=1,
            transition_rates=np.array([[0.0]]),  # must be None for K=1
            initial_regime=0,
        ),
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


# ---------------------------------------------------------------------------
# Baseline lookup
# ---------------------------------------------------------------------------


def test_baseline_for_regime_multi():
    cfg = _two_regime_config(mu_calm=0.3, mu_active=1.5)
    mu0 = baseline_for_regime(cfg, 0)
    mu1 = baseline_for_regime(cfg, 1)
    np.testing.assert_allclose(mu0, 0.3 * np.ones(4))
    np.testing.assert_allclose(mu1, 1.5 * np.ones(4))


def test_baseline_for_regime_single():
    cfg = _single_regime_config()
    mu0 = baseline_for_regime(cfg, 0)
    np.testing.assert_allclose(mu0, 0.5 * np.ones(4))


# ---------------------------------------------------------------------------
# RegimeProcess
# ---------------------------------------------------------------------------


def test_regime_process_single_regime_never_jumps():
    cfg = _single_regime_config()
    rp = RegimeProcess(cfg)
    rng = np.random.default_rng(0)
    wait, nxt = rp.sample_transition(current_regime=0, rng=rng)
    assert wait == float("inf")
    assert nxt == 0


def test_regime_process_mean_sojourn_time():
    """In regime 0 the exit rate is 0.10 so the mean sojourn is 10.0. Sample
    many waits and check the sample mean is within 5% of the analytic mean.
    """
    cfg = _two_regime_config()
    rp = RegimeProcess(cfg)
    rng = np.random.default_rng(42)
    n = 5000
    waits = np.empty(n)
    for i in range(n):
        w, _ = rp.sample_transition(current_regime=0, rng=rng)
        waits[i] = w
    mean = float(waits.mean())
    assert abs(mean - 10.0) / 10.0 < 0.05, f"sample mean {mean} too far from 10.0"


def test_regime_process_two_state_transitions_to_other():
    cfg = _two_regime_config()
    rp = RegimeProcess(cfg)
    rng = np.random.default_rng(1)
    # From regime 0 the only possible next regime is 1.
    for _ in range(100):
        _, nxt = rp.sample_transition(current_regime=0, rng=rng)
        assert nxt == 1
    # From regime 1 the only possible next regime is 0.
    for _ in range(100):
        _, nxt = rp.sample_transition(current_regime=1, rng=rng)
        assert nxt == 0


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


def test_single_regime_backward_compatibility():
    """With K=1 the trajectory is a single record at t=0, regime=0, and
    behaviour is unchanged from the first milestone."""
    cfg = _single_regime_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert trace.num_regimes == 1
    assert trace.regime_times.shape == (1,)
    assert trace.regime_values.shape == (1,)
    assert trace.regime_times[0] == 0.0
    assert trace.regime_values[0] == 0
    # basic point-process sanity
    assert np.all(np.diff(trace.times) > 0.0)


def test_two_regime_trajectory_recorded():
    cfg = _two_regime_config(horizon=200.0, seed=101)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    # With horizon 200 and mean sojourn ~10 s we expect many regime jumps.
    assert trace.num_regimes == 2
    assert trace.regime_times.shape == trace.regime_values.shape
    assert trace.regime_times[0] == 0.0
    assert trace.regime_values[0] == cfg.latent_regime.initial_regime
    # Regime jump times must be strictly increasing and bounded by horizon.
    if trace.regime_times.shape[0] > 1:
        assert np.all(np.diff(trace.regime_times) > 0.0)
        assert trace.regime_times[-1] <= cfg.horizon
    # Regime values must alternate between 0 and 1.
    if trace.regime_values.shape[0] > 1:
        diffs = np.diff(trace.regime_values.astype(int))
        assert np.all(np.abs(diffs) == 1)


def test_two_regime_event_rate_tracks_regime():
    """Active regime has baseline 1.5; calm regime has baseline 0.3. Event
    rate during active intervals must substantially exceed the calm rate.
    """
    cfg = _two_regime_config(horizon=600.0, seed=303)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()

    if trace.regime_times.shape[0] < 2 or trace.times.shape[0] == 0:
        pytest.skip("run produced no regime jumps or no events")

    # Build regime-interval edges.
    edges = np.concatenate(
        [trace.regime_times, np.array([cfg.horizon], dtype=np.float64)]
    )
    regime_labels = trace.regime_values

    calm_time = 0.0
    active_time = 0.0
    calm_events = 0
    active_events = 0
    for i in range(regime_labels.shape[0]):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        dur = t1 - t0
        count = int(np.sum((trace.times >= t0) & (trace.times < t1)))
        if int(regime_labels[i]) == 0:
            calm_time += dur
            calm_events += count
        else:
            active_time += dur
            active_events += count

    assert calm_time > 0.0 and active_time > 0.0
    calm_rate = calm_events / calm_time
    active_rate = active_events / active_time
    # Active baseline is 5x the calm baseline; the same kernel sum applies to
    # both, so the active rate must be clearly above the calm rate.
    assert active_rate > 1.5 * calm_rate, (
        f"calm_rate={calm_rate:.3f}, active_rate={active_rate:.3f} "
        f"(expected active >> calm)"
    )


def test_two_regime_determinism_under_fixed_seed():
    cfg = _two_regime_config(horizon=200.0, seed=7)
    a = HawkesSimulator(cfg)
    a.reset()
    ta = a.simulate()
    b = HawkesSimulator(cfg)
    b.reset()
    tb = b.simulate()
    np.testing.assert_array_equal(ta.times, tb.times)
    np.testing.assert_array_equal(ta.event_types, tb.event_types)
    np.testing.assert_array_equal(ta.regime_times, tb.regime_times)
    np.testing.assert_array_equal(ta.regime_values, tb.regime_values)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_run_preserves_regime_trajectory():
    cfg = _two_regime_config(horizon=120.0, seed=555)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260423T000000Z"
        )
        loaded = load_run(run_dir)
        np.testing.assert_array_equal(loaded.regime_times, trace.regime_times)
        np.testing.assert_array_equal(loaded.regime_values, trace.regime_values)
        assert loaded.metadata["num_regimes"] == 2
        assert loaded.metadata["num_regime_jumps"] == int(
            max(0, trace.regime_times.shape[0] - 1)
        )
