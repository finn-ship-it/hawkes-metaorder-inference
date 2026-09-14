"""Deterministic regressions for predictable intensities and thinning bounds.

The next proposal must include the jump caused by an event just accepted at
the current time. Acceptance, in contrast, uses the pre-event intensity.
These tests check those two limits directly rather than relying on noisy
ensemble means to reveal the original missing-jump error.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulator.config import (
    BaselineParams,
    ExponentialKernelParams,
    ExponentialSumKernelParams,
    KernelFamily,
    LatentRegimeSpec,
    MetaOrderSpec,
    MetaOrderWindow,
    NumericalSettings,
    PowerLawCutoffKernelParams,
    SimulatorConfig,
)
from simulator.hawkes_core import HawkesSimulator


FAMILIES = (
    KernelFamily.EXPONENTIAL,
    KernelFamily.EXPONENTIAL_SUM,
    KernelFamily.POWERLAW_CUTOFF,
)


class _ScriptedRng:
    """Make event times/decisions deterministic and record proposal scales."""

    def __init__(self, wait: float = 0.25, uniforms=()):
        self.wait = wait
        self.scales = []
        self.uniforms = iter(uniforms)
        self.uniform_calls = 0

    def exponential(self, scale: float) -> float:
        self.scales.append(float(scale))
        return self.wait

    def uniform(self, low: float, high: float) -> float:
        self.uniform_calls += 1
        return next(self.uniforms, 0.0)


def _config(family: KernelFamily, *, safety: float = 1.1) -> SimulatorConfig:
    # Different columns check that a type-j event excites column j, not row j.
    alpha = np.tile(np.array([0.10, 0.08, 0.06, 0.04]), (4, 1))
    if family is KernelFamily.EXPONENTIAL:
        params = ExponentialKernelParams(alpha=alpha, beta=np.full((4, 4), 2.0))
    elif family is KernelFamily.EXPONENTIAL_SUM:
        params = ExponentialSumKernelParams(
            alphas=np.stack((0.3 * alpha, 0.7 * alpha), axis=-1),
            betas=np.broadcast_to(np.array([0.5, 2.0]), (4, 4, 2)).copy(),
        )
    else:
        params = PowerLawCutoffKernelParams(
            alpha=alpha,
            beta=np.full((4, 4), 2.0),
            gamma=np.full((4, 4), 2.0),
            t_max=7.0,
        )
    return SimulatorConfig(
        dimension=4,
        horizon=20.0,
        seed=7,
        kernel_family=family,
        kernel_params=params,
        baseline=BaselineParams(mu=np.full(4, 0.5)),
        numerics=NumericalSettings(truncation_window=10.0, upper_bound_safety=safety),
    )


def _column(family: KernelFamily, lag: float, event_type: int) -> np.ndarray:
    """Independent, explicit kernel values for the small fixture above."""
    amplitude = (0.10, 0.08, 0.06, 0.04)[event_type]
    if family is KernelFamily.EXPONENTIAL:
        value = amplitude * np.exp(-2.0 * lag)
    elif family is KernelFamily.EXPONENTIAL_SUM:
        value = amplitude * (0.3 * np.exp(-0.5 * lag) + 0.7 * np.exp(-2.0 * lag))
    else:
        value = amplitude / (1.0 + 2.0 * lag) ** 2 if lag <= 7.0 else 0.0
    return np.full(4, value)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("safety", (1.0, 1.1))
def test_proposal_includes_just_accepted_event(family, safety):
    sim = HawkesSimulator(_config(family, safety=safety))
    sim.reset()
    sim._update_history(1.0, 0)
    sim._state.current_time = 1.0

    # The predictable intensity at the event still excludes its own jump.
    np.testing.assert_allclose(sim._current_intensity(1.0), np.full(4, 0.5))
    _, bound = sim._propose_candidate()
    expected_right_limit = np.full(4, 0.5) + _column(family, 0.0, 0)
    np.testing.assert_allclose(sim._state.intensity, expected_right_limit)
    assert bound == pytest.approx(safety * expected_right_limit.sum())
    # The old code returned 2.2 with safety=1.1, below the 2.4 right limit.
    assert bound >= expected_right_limit.sum()


@pytest.mark.parametrize("family", FAMILIES)
def test_bound_contains_retained_history_and_dominates_between_jumps(family):
    sim = HawkesSimulator(_config(family))
    sim.reset()
    sim._state.history_times[:] = [0.0, 9.0, 10.0, 12.0]
    sim._state.history_types[:] = [1, 2, 3, 0]
    sim._state.current_time = 12.0
    _, bound = sim._propose_candidate()
    expected = np.full(4, 0.5)
    for lag, event_type in ((3.0, 2), (2.0, 3), (0.0, 0)):
        expected += _column(family, lag, event_type)
    assert bound == pytest.approx(1.1 * expected.sum())
    for candidate in (np.nextafter(12.0, np.inf), 12.01, 13.0, 19.01, 23.0):
        assert sim._current_intensity(candidate).sum() <= bound


@pytest.mark.parametrize("family", FAMILIES)
def test_acceptance_uses_predictable_left_limit(family):
    sim = HawkesSimulator(_config(family))
    sim.reset()
    sim._state.history_times[:] = [1.0]
    sim._state.history_types[:] = [0]
    sim._state.rng = _ScriptedRng(uniforms=(0.55,))
    # At t=1, 0.55*4=2.2 exceeds the left intensity 2.0 but not the
    # right intensity 2.4. This must reject, not use the event's own jump.
    assert sim._accept_or_reject(1.0, upper_bound=4.0) is None


@pytest.mark.parametrize("family", FAMILIES)
def test_underestimated_bound_fails_before_acceptance_draw(family):
    sim = HawkesSimulator(_config(family))
    sim.reset()
    sim._state.history_times[:] = [0.0]
    sim._state.history_types[:] = [0]
    rng = _ScriptedRng()
    sim._state.rng = rng
    with pytest.raises(RuntimeError):
        sim._accept_or_reject(0.01, upper_bound=2.0)
    assert rng.uniform_calls == 0


@pytest.mark.parametrize("bound", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_bound_fails_before_acceptance_draw(bound):
    sim = HawkesSimulator(_config(KernelFamily.EXPONENTIAL))
    sim.reset()
    rng = _ScriptedRng()
    sim._state.rng = rng
    with pytest.raises(RuntimeError):
        sim._accept_or_reject(1.0, upper_bound=bound)
    assert rng.uniform_calls == 0


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0))
def test_invalid_candidate_intensity_fails_before_acceptance_draw(value, monkeypatch):
    sim = HawkesSimulator(_config(KernelFamily.EXPONENTIAL))
    sim.reset()
    rng = _ScriptedRng()
    sim._state.rng = rng
    monkeypatch.setattr(sim, "_current_intensity", lambda t: np.full(4, value))
    with pytest.raises(RuntimeError):
        sim._accept_or_reject(1.0, upper_bound=3.0)
    assert rng.uniform_calls == 0


def test_nonfinite_proposal_bound_fails_before_drawing_wait(monkeypatch):
    sim = HawkesSimulator(_config(KernelFamily.EXPONENTIAL))
    sim.reset()
    rng = _ScriptedRng()
    sim._state.rng = rng
    monkeypatch.setattr(
        sim, "_current_intensity", lambda t, **kwargs: np.full(4, float("inf"))
    )
    with pytest.raises(RuntimeError):
        sim._propose_candidate()
    assert rng.scales == []


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("entry_point", ("step", "simulate"))
def test_every_accepted_jump_enters_the_next_bound(family, entry_point):
    sim = HawkesSimulator(replace(_config(family), horizon=1.1))
    sim.reset()
    rng = _ScriptedRng(wait=0.25)
    sim._state.rng = rng
    if entry_point == "simulate":
        trace = sim.simulate()
        np.testing.assert_allclose(trace.times, [0.25, 0.5, 0.75, 1.0])
    else:
        for expected_time in (0.25, 0.5, 0.75, 1.0):
            event = sim.step()
            assert event.time == pytest.approx(expected_time)
            assert event.event_type == 0
        assert sim.step() is None  # the next candidate exceeds the horizon

    assert len(rng.scales) == 5
    history = []
    for index, scale in enumerate(rng.scales):
        now = index * 0.25
        expected = np.full(4, 0.5)
        for event_time in history:
            expected += _column(family, now - event_time, 0)
        assert 1.0 / scale == pytest.approx(1.1 * expected.sum())
        history.append(now + 0.25)


@pytest.mark.parametrize("family", FAMILIES)
def test_window_start_and_end_restart_proposal_before_acceptance(family):
    cfg = replace(
        _config(family),
        meta_order=MetaOrderSpec(windows=(
            MetaOrderWindow(1.0, 2.0, 2.0, (0, 2)),
        )),
    )
    sim = HawkesSimulator(cfg)
    sim.reset()
    rng = _ScriptedRng(wait=10.0)
    sim._state.rng = rng

    assert sim.step() is None
    assert sim._state.current_time == 1.0
    assert sim._state.meta_order_active
    assert sim._state.history_times == []
    assert sim.step() is None
    assert sim._state.current_time == 2.0
    assert not sim._state.meta_order_active
    assert sim._state.history_times == []
    _, after_end = sim._propose_candidate()
    np.testing.assert_allclose(1.0 / np.asarray(rng.scales), [2.2, 4.4, 2.2])
    assert after_end == pytest.approx(2.2)


@pytest.mark.parametrize("family", FAMILIES)
def test_regime_jump_restarts_proposal_before_acceptance(family, monkeypatch):
    cfg = replace(
        _config(family),
        baseline=BaselineParams(mu=np.array([[0.5] * 4, [1.5] * 4])),
        latent_regime=LatentRegimeSpec(
            num_regimes=2,
            transition_rates=np.array([[-0.1, 0.1], [0.2, -0.2]]),
        ),
    )
    sim = HawkesSimulator(cfg)
    sim.reset()
    sim._next_regime_jump_time = 1.0
    sim._pending_next_regime = 1
    monkeypatch.setattr(sim, "_sample_next_regime_jump", lambda: (float("inf"), 1))
    rng = _ScriptedRng(wait=10.0)
    sim._state.rng = rng

    assert sim.step() is None
    assert sim._state.current_time == 1.0
    assert sim._state.regime == 1
    assert sim._state.history_times == []
    _, after_jump = sim._propose_candidate()
    np.testing.assert_allclose(1.0 / np.asarray(rng.scales), [2.2, 6.6])
    assert after_jump == pytest.approx(6.6)
