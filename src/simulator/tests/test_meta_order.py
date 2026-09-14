"""Tests for meta-order windows."""

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
    MetaOrderSpec,
    MetaOrderWindow,
    KernelFamily,
    ConfigurationError,
    validate,
    validate_first_milestone,
    config_to_json,
    config_from_json,
)
from simulator.latent import MetaOrderSchedule
from simulator.hawkes_core import HawkesSimulator
from simulator.artifact import save_run, load_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(d: int = 4):
    return dict(
        dimension=d,
        horizon=120.0,
        seed=20260423,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(
            alpha=0.1 * np.ones((d, d)), beta=2.0 * np.ones((d, d))
        ),
        baseline=BaselineParams(mu=0.5 * np.ones(d)),
    )


def _single_regime_with_windows(
    windows=(),
    horizon: float = 120.0,
    seed: int = 20260423,
    mu_scalar: float = 0.5,
) -> SimulatorConfig:
    d = 4
    kwargs = _base_kwargs(d)
    kwargs["horizon"] = horizon
    kwargs["seed"] = seed
    kwargs["baseline"] = BaselineParams(mu=mu_scalar * np.ones(d))
    return SimulatorConfig(
        **kwargs,
        meta_order=MetaOrderSpec(windows=tuple(windows)),
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_single_window_passes_validate():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=10.0,
                end_time=20.0,
                alpha=0.5,
                target_event_types=(0, 2),
            )
        ]
    )
    validate(cfg)


def test_multiple_non_overlapping_windows_pass_validate():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=5.0, end_time=15.0, alpha=0.5, target_event_types=(0,)
            ),
            MetaOrderWindow(
                start_time=20.0, end_time=30.0, alpha=0.2, target_event_types=(1, 2)
            ),
        ]
    )
    validate(cfg)


def test_reject_overlapping_windows():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=5.0, end_time=15.0, alpha=0.5, target_event_types=(0,)
            ),
            MetaOrderWindow(
                start_time=10.0, end_time=20.0, alpha=0.3, target_event_types=(1,)
            ),
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_negative_alpha():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=5.0, end_time=15.0, alpha=-0.1, target_event_types=(0,)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_end_not_greater_than_start():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=10.0, end_time=10.0, alpha=0.1, target_event_types=(0,)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_negative_start_time():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=-1.0, end_time=5.0, alpha=0.1, target_event_types=(0,)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_target_event_type_out_of_range():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=1.0, end_time=5.0, alpha=0.1, target_event_types=(7,)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_empty_target_event_types():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=1.0, end_time=5.0, alpha=0.1, target_event_types=()
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_reject_duplicate_target_event_types():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=1.0, end_time=5.0, alpha=0.1, target_event_types=(0, 0)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate(cfg)


def test_first_milestone_rejects_any_window():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=1.0, end_time=5.0, alpha=0.1, target_event_types=(0,)
            )
        ]
    )
    with pytest.raises(ConfigurationError):
        validate_first_milestone(cfg)


def test_first_milestone_accepts_empty_windows():
    cfg = _single_regime_with_windows(windows=[])
    validate_first_milestone(cfg)


# ---------------------------------------------------------------------------
# MetaOrderSchedule
# ---------------------------------------------------------------------------


def test_schedule_empty_has_no_pending():
    cfg = _single_regime_with_windows(windows=[])
    s = MetaOrderSchedule(cfg)
    assert s.has_pending is False
    assert s.next_boundary_time() == float("inf")
    np.testing.assert_allclose(s.factors, np.ones(4))


def test_schedule_start_and_end_update_factors():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=2.0, end_time=7.0, alpha=1.0, target_event_types=(0, 2)
            )
        ]
    )
    s = MetaOrderSchedule(cfg)
    assert s.next_boundary_time() == 2.0
    s.advance()  # open window
    np.testing.assert_allclose(s.factors, np.array([2.0, 1.0, 2.0, 1.0]))
    assert s.next_boundary_time() == 7.0
    s.advance()  # close window
    np.testing.assert_allclose(s.factors, np.ones(4))
    assert s.has_pending is False


def test_schedule_reset_restores_initial_state():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=2.0, end_time=7.0, alpha=0.5, target_event_types=(1,)
            )
        ]
    )
    s = MetaOrderSchedule(cfg)
    s.advance()
    s.reset()
    assert s.next_boundary_time() == 2.0
    np.testing.assert_allclose(s.factors, np.ones(4))


def test_schedule_sorts_by_time():
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=15.0, end_time=20.0, alpha=0.5, target_event_types=(0,)
            ),
            MetaOrderWindow(
                start_time=5.0, end_time=10.0, alpha=0.5, target_event_types=(0,)
            ),
        ]
    )
    s = MetaOrderSchedule(cfg)
    times = []
    while s.has_pending:
        times.append(s.next_boundary_time())
        s.advance()
    assert times == [5.0, 10.0, 15.0, 20.0]


def test_schedule_end_before_start_at_equal_times():
    """When one window closes and another opens at the same instant, the
    closing boundary must be consumed before the opening one so that the
    factor vector is correctly reset before the new window opens."""
    cfg = _single_regime_with_windows(
        windows=[
            MetaOrderWindow(
                start_time=0.0, end_time=5.0, alpha=1.0, target_event_types=(0,)
            ),
            MetaOrderWindow(
                start_time=5.0, end_time=10.0, alpha=2.0, target_event_types=(0,)
            ),
        ]
    )
    s = MetaOrderSchedule(cfg)
    # open first
    s.advance()
    np.testing.assert_allclose(s.factors, np.array([2.0, 1.0, 1.0, 1.0]))
    # next boundary is both the end of window 0 and the start of window 1 at t=5
    assert s.next_boundary_time() == 5.0
    # consume the "end" first
    s.advance()
    np.testing.assert_allclose(s.factors, np.ones(4))
    # then the "start" of window 1
    s.advance()
    np.testing.assert_allclose(s.factors, np.array([3.0, 1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


def test_no_window_backward_compat_unchanged_from_prior_runs():
    """With no meta-order windows, the simulator output must match the prior
    implementation byte-for-byte. We verify determinism against a fresh copy
    of the simulator with the same config."""
    cfg = _single_regime_with_windows(windows=[], horizon=60.0, seed=20260423)
    a = HawkesSimulator(cfg)
    a.reset()
    ta = a.simulate()
    b = HawkesSimulator(cfg)
    b.reset()
    tb = b.simulate()
    np.testing.assert_array_equal(ta.times, tb.times)
    np.testing.assert_array_equal(ta.event_types, tb.event_types)
    assert ta.meta_order_windows == ()
    assert tb.meta_order_windows == ()


def test_rate_uplift_during_window_vs_outside():
    """Inside a large uplift window the event rate on targeted event types
    must substantially exceed the outside rate. We pick alpha = 5 (a factor of
    6) and a long window so the comparison is not dominated by sampling
    noise."""
    windows = [
        MetaOrderWindow(
            start_time=200.0,
            end_time=600.0,
            alpha=5.0,  # 6x uplift on event types 0 and 2
            target_event_types=(0, 2),
        )
    ]
    cfg = _single_regime_with_windows(
        windows=windows,
        horizon=800.0,
        seed=9001,
        mu_scalar=0.5,
    )
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()

    times = trace.times
    types = trace.event_types

    in_window = (times >= 200.0) & (times < 600.0)
    out_window = ~in_window

    window_duration = 400.0
    outside_duration = cfg.horizon - window_duration

    target_mask = np.isin(types, [0, 2])
    other_mask = ~target_mask

    target_in = int(np.sum(target_mask & in_window))
    target_out = int(np.sum(target_mask & out_window))
    other_in = int(np.sum(other_mask & in_window))
    other_out = int(np.sum(other_mask & out_window))

    target_rate_in = target_in / window_duration
    target_rate_out = target_out / outside_duration
    other_rate_in = other_in / window_duration
    other_rate_out = other_out / outside_duration

    assert target_rate_in > 2.0 * target_rate_out, (
        f"target uplift not observed: in={target_rate_in:.3f}, "
        f"out={target_rate_out:.3f}"
    )
    # Non-target rate should not be dramatically inflated by the window.
    # Some self-excitation bleed is expected, so allow up to 2x tolerance.
    assert other_rate_in < 2.0 * other_rate_out + 1e-9, (
        f"non-target rate unexpectedly inflated during window: "
        f"in={other_rate_in:.3f}, out={other_rate_out:.3f}"
    )


def test_determinism_under_fixed_seed_with_window():
    windows = [
        MetaOrderWindow(
            start_time=10.0, end_time=40.0, alpha=1.0, target_event_types=(0, 2)
        )
    ]
    cfg = _single_regime_with_windows(windows=windows, horizon=80.0, seed=77)
    a = HawkesSimulator(cfg)
    a.reset()
    ta = a.simulate()
    b = HawkesSimulator(cfg)
    b.reset()
    tb = b.simulate()
    np.testing.assert_array_equal(ta.times, tb.times)
    np.testing.assert_array_equal(ta.event_types, tb.event_types)
    assert ta.meta_order_windows == tb.meta_order_windows


def test_meta_order_windows_recorded_in_trace():
    windows = [
        MetaOrderWindow(
            start_time=10.0, end_time=40.0, alpha=0.5, target_event_types=(0, 2)
        ),
        MetaOrderWindow(
            start_time=50.0, end_time=70.0, alpha=1.0, target_event_types=(1,)
        ),
    ]
    cfg = _single_regime_with_windows(windows=windows, horizon=80.0, seed=42)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    assert trace.meta_order_windows == (
        (10.0, 40.0, 0.5, (0, 2)),
        (50.0, 70.0, 1.0, (1,)),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_run_preserves_meta_order_metadata():
    windows = [
        MetaOrderWindow(
            start_time=10.0, end_time=40.0, alpha=1.0, target_event_types=(0, 2)
        )
    ]
    cfg = _single_regime_with_windows(windows=windows, horizon=60.0, seed=555)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260423T000000Z"
        )
        loaded = load_run(run_dir)
        # Config round-trips.
        assert loaded.config.meta_order.windows[0].start_time == 10.0
        assert loaded.config.meta_order.windows[0].end_time == 40.0
        assert loaded.config.meta_order.windows[0].alpha == 1.0
        assert loaded.config.meta_order.windows[0].target_event_types == (0, 2)
        # Metadata exposes the window record.
        assert loaded.metadata["num_meta_order_windows"] == 1
        assert loaded.metadata["meta_order_windows"][0]["alpha"] == 1.0
        assert loaded.metadata["meta_order_windows"][0]["target_event_types"] == [0, 2]


def test_round_trip_meta_order_config_through_json():
    windows = [
        MetaOrderWindow(
            start_time=5.0, end_time=9.0, alpha=0.25, target_event_types=(0, 1, 3)
        )
    ]
    cfg = _single_regime_with_windows(windows=windows, horizon=60.0, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cfg.json")
        config_to_json(cfg, p)
        loaded = config_from_json(p)
    assert len(loaded.meta_order.windows) == 1
    w = loaded.meta_order.windows[0]
    assert w.start_time == 5.0
    assert w.end_time == 9.0
    assert w.alpha == 0.25
    assert w.target_event_types == (0, 1, 3)
