"""Tests for the observation operator and its persistence layer."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from simulator.config import (
    SimulatorConfig,
    ExponentialKernelParams,
    BaselineParams,
    KernelFamily,
    validate_first_milestone,
)
from simulator.hawkes_core import HawkesSimulator, RunTrace
from simulator.observation import (
    ALLOWED_CROSSING_POLICIES,
    ObservationConfig,
    ObservedTrace,
    ObservationOperator,
    default_observation_config,
)
from simulator.artifact import save_run, load_run, OBSERVATION_FILENAME
from simulator.runner import run_once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_regime_config(
    seed: int = 20260423, horizon: float = 30.0
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )
    validate_first_milestone(cfg)
    return cfg


def _synthetic_trace(event_types: list, times: list) -> RunTrace:
    return RunTrace(
        times=np.asarray(times, dtype=np.float64),
        event_types=np.asarray(event_types, dtype=np.int64),
        final_intensity=np.zeros(4, dtype=np.float64),
        num_proposals=len(times),
        num_acceptances=len(times),
        wall_clock_seconds=0.0,
        dimension=4,
        horizon=float(times[-1] + 1.0) if times else 1.0,
        seed=0,
        num_regimes=1,
        regime_times=np.zeros(1, dtype=np.float64),
        regime_values=np.zeros(1, dtype=np.int64),
        meta_order_windows=(),
    )


# ---------------------------------------------------------------------------
# ObservationConfig
# ---------------------------------------------------------------------------


def test_default_observation_config_values():
    cfg = default_observation_config()
    assert cfg.initial_bid_ticks == 0
    assert cfg.initial_ask_ticks == 1
    assert cfg.min_spread_ticks == 1
    assert cfg.crossing_policy == "censor"
    assert cfg.tick_size == 1e-5


def test_reject_initial_spread_below_min_spread():
    with pytest.raises(ValueError):
        ObservationConfig(
            initial_bid_ticks=5, initial_ask_ticks=5, min_spread_ticks=1
        )
    with pytest.raises(ValueError):
        ObservationConfig(
            initial_bid_ticks=0, initial_ask_ticks=1, min_spread_ticks=2
        )


def test_reject_non_positive_min_spread_ticks():
    with pytest.raises(ValueError):
        ObservationConfig(min_spread_ticks=0)
    with pytest.raises(ValueError):
        ObservationConfig(min_spread_ticks=-1)


def test_reject_unknown_crossing_policy():
    with pytest.raises(ValueError):
        ObservationConfig(crossing_policy="ignore")
    # Sanity: all allowed policies construct cleanly.
    for p in ALLOWED_CROSSING_POLICIES:
        ObservationConfig(crossing_policy=p)


def test_reject_non_positive_tick_size():
    with pytest.raises(ValueError):
        ObservationConfig(tick_size=0.0)
    with pytest.raises(ValueError):
        ObservationConfig(tick_size=-1e-5)


# ---------------------------------------------------------------------------
# Deterministic projection on synthetic traces
# ---------------------------------------------------------------------------


def test_project_empty_trace_returns_empty_observed():
    trace = _synthetic_trace([], [])
    op = ObservationOperator(default_observation_config())
    observed = op.project(trace)
    assert observed.num_observed_events == 0
    assert observed.num_latent_events == 0
    assert observed.num_censored_events == 0
    assert observed.censoring_fraction == 0.0
    assert observed.final_bid_ticks == 0
    assert observed.final_ask_ticks == 1
    assert observed.num_crossed_observations == 0
    assert observed.spread_ticks.shape == (0,)


def test_project_censors_min_spread_violations():
    # Sequence: BID_UP, ASK_DOWN, BID_UP, ASK_UP, BID_DOWN
    # Initial (bid, ask) = (0, 1), min_spread = 1.
    # Event 0 BID_UP: proposal (1, 1), spread 0 -> CENSORED, state (0, 1)
    # Event 1 ASK_DOWN: proposal (0, 0), spread 0 -> CENSORED, state (0, 1)
    # Event 2 BID_UP: proposal (1, 1), spread 0 -> CENSORED, state (0, 1)
    # Event 3 ASK_UP: proposal (0, 2), spread 2 -> ADMIT, state (0, 2)
    # Event 4 BID_DOWN: proposal (-1, 2), spread 3 -> ADMIT, state (-1, 2)
    trace = _synthetic_trace(
        event_types=[0, 3, 0, 2, 1],
        times=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    op = ObservationOperator(default_observation_config())
    observed = op.project(trace)
    assert observed.num_latent_events == 5
    assert observed.num_censored_events == 3
    assert observed.num_observed_events == 2
    assert observed.censoring_fraction == pytest.approx(3.0 / 5.0)
    np.testing.assert_array_equal(observed.times, np.array([4.0, 5.0]))
    np.testing.assert_array_equal(observed.event_types, np.array([2, 1]))
    np.testing.assert_array_equal(observed.bid_ticks, np.array([0, -1]))
    np.testing.assert_array_equal(observed.ask_ticks, np.array([2, 2]))
    np.testing.assert_array_equal(observed.spread_ticks, np.array([2, 3]))


def test_no_observed_quote_is_crossed():
    # Construct a worst-case aggressive sequence and verify no crossed output.
    rng = np.random.default_rng(20260424)
    trace = _synthetic_trace(
        event_types=list(rng.integers(0, 4, size=2000).astype(np.int64)),
        times=list(np.arange(1, 2001, dtype=np.float64)),
    )
    op = ObservationOperator(default_observation_config())
    observed = op.project(trace)
    assert observed.num_crossed_observations == 0
    assert int(np.min(observed.spread_ticks)) >= observed.min_spread_ticks


def test_spread_always_at_least_min_spread_ticks_non_default():
    cfg = ObservationConfig(
        initial_bid_ticks=0, initial_ask_ticks=5, min_spread_ticks=3
    )
    rng = np.random.default_rng(7)
    trace = _synthetic_trace(
        event_types=list(rng.integers(0, 4, size=500).astype(np.int64)),
        times=list(np.arange(1, 501, dtype=np.float64)),
    )
    observed = ObservationOperator(cfg).project(trace)
    assert int(np.min(observed.spread_ticks)) >= 3


def test_project_preserves_time_order_of_admitted_events():
    trace = _synthetic_trace(
        event_types=[0, 3, 2, 2, 1, 1, 0, 3],
        times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    observed = ObservationOperator(default_observation_config()).project(trace)
    if observed.num_observed_events > 1:
        assert np.all(np.diff(observed.times) > 0.0)


def test_project_non_default_initial_ticks():
    trace = _synthetic_trace(event_types=[0, 2], times=[1.0, 2.0])
    cfg = ObservationConfig(
        initial_bid_ticks=100,
        initial_ask_ticks=102,
        min_spread_ticks=1,
        tick_size=1e-4,
    )
    observed = ObservationOperator(cfg).project(trace)
    np.testing.assert_array_equal(observed.bid_ticks, np.array([101, 101]))
    np.testing.assert_array_equal(observed.ask_ticks, np.array([102, 103]))
    np.testing.assert_array_equal(observed.spread_ticks, np.array([1, 2]))
    assert observed.num_censored_events == 0
    assert observed.tick_size == 1e-4


def test_project_reject_out_of_range_event_type():
    trace = _synthetic_trace(event_types=[0, 7], times=[1.0, 2.0])
    with pytest.raises(ValueError):
        ObservationOperator(default_observation_config()).project(trace)


def test_project_is_deterministic():
    trace = _synthetic_trace(
        event_types=[0, 1, 2, 3, 0, 2, 3, 0, 1, 2],
        times=list(np.arange(1, 11, dtype=np.float64)),
    )
    op = ObservationOperator(default_observation_config())
    a = op.project(trace)
    b = op.project(trace)
    np.testing.assert_array_equal(a.times, b.times)
    np.testing.assert_array_equal(a.event_types, b.event_types)
    np.testing.assert_array_equal(a.bid_ticks, b.bid_ticks)
    np.testing.assert_array_equal(a.ask_ticks, b.ask_ticks)
    np.testing.assert_array_equal(a.spread_ticks, b.spread_ticks)
    assert a.num_latent_events == b.num_latent_events
    assert a.num_censored_events == b.num_censored_events


def test_no_censoring_when_min_spread_accommodates_proposal():
    # Start with a wide book (spread 10), min_spread 1. No latent event
    # sequence of finite length can drive the spread below 1 quickly, so we
    # expect low but non-zero censoring on average for very long sequences.
    # Here we check that a short sequence never triggers censoring.
    cfg = ObservationConfig(
        initial_bid_ticks=0, initial_ask_ticks=10, min_spread_ticks=1
    )
    trace = _synthetic_trace(
        event_types=[0, 2, 1, 3], times=[1.0, 2.0, 3.0, 4.0]
    )
    observed = ObservationOperator(cfg).project(trace)
    assert observed.num_censored_events == 0
    assert observed.num_observed_events == 4


# ---------------------------------------------------------------------------
# Integration with simulator
# ---------------------------------------------------------------------------


def test_project_simulator_trace_obeys_spread_invariant():
    cfg = _single_regime_config(seed=1234, horizon=30.0)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    observed = ObservationOperator(default_observation_config()).project(trace)
    assert observed.num_latent_events == trace.num_events
    assert observed.num_observed_events + observed.num_censored_events == (
        trace.num_events
    )
    if observed.num_observed_events > 0:
        assert int(np.min(observed.spread_ticks)) >= observed.min_spread_ticks
        assert observed.num_crossed_observations == 0


def test_project_admitted_counters_match_admitted_event_counts():
    """Under the censor policy the final tick counters match the net counts
    of *admitted* events, not the raw latent counts."""
    cfg = _single_regime_config(seed=2026, horizon=20.0)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    obs_cfg = default_observation_config()
    observed = ObservationOperator(obs_cfg).project(trace)
    if observed.num_observed_events == 0:
        pytest.skip("run produced no admitted events")
    counts = np.bincount(observed.event_types.astype(np.int64), minlength=4)
    expected_bid_net = int(counts[0]) - int(counts[1])
    expected_ask_net = int(counts[2]) - int(counts[3])
    assert observed.final_bid_ticks == obs_cfg.initial_bid_ticks + expected_bid_net
    assert observed.final_ask_ticks == obs_cfg.initial_ask_ticks + expected_ask_net


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_run_round_trips_observed_trace():
    cfg = _single_regime_config(seed=42, horizon=15.0)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    obs_cfg = ObservationConfig(
        initial_bid_ticks=10,
        initial_ask_ticks=12,
        min_spread_ticks=1,
        tick_size=1e-4,
    )
    observed = ObservationOperator(obs_cfg).project(trace)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg,
            trace=trace,
            runs_root=tmp,
            timestamp="20260424T000000Z",
            observed=observed,
        )
        assert os.path.isfile(os.path.join(run_dir, OBSERVATION_FILENAME))
        loaded = load_run(run_dir)
        assert loaded.observed is not None
        np.testing.assert_array_equal(loaded.observed.times, observed.times)
        np.testing.assert_array_equal(
            loaded.observed.event_types, observed.event_types
        )
        np.testing.assert_array_equal(
            loaded.observed.bid_ticks, observed.bid_ticks
        )
        np.testing.assert_array_equal(
            loaded.observed.ask_ticks, observed.ask_ticks
        )
        np.testing.assert_array_equal(
            loaded.observed.spread_ticks, observed.spread_ticks
        )
        assert loaded.observed.initial_bid_ticks == 10
        assert loaded.observed.initial_ask_ticks == 12
        assert loaded.observed.min_spread_ticks == 1
        assert loaded.observed.crossing_policy == "censor"
        assert loaded.observed.tick_size == 1e-4
        assert loaded.observed.num_latent_events == observed.num_latent_events
        assert loaded.observed.num_censored_events == observed.num_censored_events

        assert loaded.metadata["has_observation"] is True
        block = loaded.metadata["observation"]
        assert block["min_spread_ticks"] == 1
        assert block["crossing_policy"] == "censor"
        assert block["num_latent_events"] == int(observed.num_latent_events)
        assert block["num_observed_events"] == int(observed.num_observed_events)
        assert block["num_censored_events"] == int(observed.num_censored_events)
        assert block["censoring_fraction"] == pytest.approx(
            observed.censoring_fraction
        )
        assert block["num_crossed_observations"] == 0


def test_save_run_without_observed_is_backward_compatible():
    cfg = _single_regime_config(seed=42, horizon=15.0)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260424T000000Z"
        )
        assert not os.path.isfile(os.path.join(run_dir, OBSERVATION_FILENAME))
        loaded = load_run(run_dir)
        assert loaded.observed is None
        assert loaded.metadata["has_observation"] is False


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_run_once_with_observation_config_persists_observed_stream():
    cfg = _single_regime_config(seed=7, horizon=15.0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(
            cfg=cfg,
            runs_root=tmp,
            observation_config=default_observation_config(),
        )
        assert os.path.isfile(os.path.join(run_dir, OBSERVATION_FILENAME))
        loaded = load_run(run_dir)
        assert loaded.observed is not None
        # Spread invariant must hold end-to-end through the runner.
        if loaded.observed.num_observed_events > 0:
            assert int(np.min(loaded.observed.spread_ticks)) >= 1
            assert loaded.observed.num_crossed_observations == 0
        # Diagnostic counts are consistent.
        assert loaded.observed.num_latent_events == int(loaded.times.shape[0])
        assert (
            loaded.observed.num_observed_events
            + loaded.observed.num_censored_events
            == loaded.observed.num_latent_events
        )


def test_run_once_without_observation_config_persists_latent_only():
    cfg = _single_regime_config(seed=7, horizon=15.0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(cfg=cfg, runs_root=tmp)
        assert not os.path.isfile(os.path.join(run_dir, OBSERVATION_FILENAME))
        loaded = load_run(run_dir)
        assert loaded.observed is None
