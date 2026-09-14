"""
test_recovery.py — tests for the Layer-2 baseline recovery module.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.observation import ObservedTrace
from simulator.recovery import (
    WindowFeatures,
    compute_window_features,
    RegimeBaselineConfig,
    recover_regimes_baseline,
    MetaOrderBaselineConfig,
    score_meta_order_activity,
    true_regime_at,
)


# ---------------------------------------------------------------------------
# Synthetic ObservedTrace helpers
# ---------------------------------------------------------------------------


def _make_observed(
    times,
    types,
    *,
    initial_bid=0,
    initial_ask=1,
    min_spread=1,
    crossing_policy="censor",
    tick_size=1e-5,
):
    times = np.asarray(times, dtype=float)
    types = np.asarray(types, dtype=int)
    bid = np.full(times.shape, initial_bid, dtype=int)
    ask = np.full(times.shape, initial_ask, dtype=int)
    spread = ask - bid
    return ObservedTrace(
        times=times,
        event_types=types,
        bid_ticks=bid,
        ask_ticks=ask,
        spread_ticks=spread,
        initial_bid_ticks=initial_bid,
        initial_ask_ticks=initial_ask,
        min_spread_ticks=min_spread,
        crossing_policy=crossing_policy,
        tick_size=tick_size,
        num_latent_events=int(times.size),
        num_censored_events=0,
    )


# ---------------------------------------------------------------------------
# compute_window_features
# ---------------------------------------------------------------------------


def test_compute_window_features_rejects_invalid_window_seconds():
    obs = _make_observed([], [])
    with pytest.raises(ValueError):
        compute_window_features(obs, window_seconds=0.0)


def test_compute_window_features_rejects_invalid_hop_seconds():
    obs = _make_observed([], [])
    with pytest.raises(ValueError):
        compute_window_features(obs, window_seconds=1.0, hop_seconds=-1.0)


def test_compute_window_features_empty_trace_emits_zero_window():
    obs = _make_observed([], [])
    feats = compute_window_features(obs, window_seconds=2.0, horizon=2.0)
    assert len(feats) == 1
    assert feats[0].num_events == 0
    assert feats[0].recent_event_rate == 0.0


def test_compute_window_features_counts_events_per_window():
    times = [0.5, 1.5, 2.5, 3.5, 4.5]
    types = [0, 1, 2, 3, 0]
    obs = _make_observed(times, types)
    feats = compute_window_features(obs, window_seconds=2.0, horizon=4.5)
    # Windows starts span [0, 2, 4]; widths 2.0; event 4.5 falls into [4, 6).
    assert len(feats) == 3
    assert feats[0].num_events == 2
    assert feats[1].num_events == 2
    assert feats[2].num_events == 1
    assert feats[0].recent_event_rate == pytest.approx(1.0)
    # Imbalance in [0,2): types 0,1 -> n_up=1, n_down=1 -> 0
    assert feats[0].directional_imbalance == pytest.approx(0.0)


def test_compute_window_features_imbalance_sign():
    # Three BID_UP plus one ASK_DOWN in a single window -> n_up=3, n_down=1 -> +0.5
    obs = _make_observed([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 3])
    feats = compute_window_features(obs, window_seconds=1.0, horizon=1.0)
    assert len(feats) == 1
    assert feats[0].directional_imbalance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# recover_regimes_baseline
# ---------------------------------------------------------------------------


def test_recover_regimes_calm_run_stays_calm():
    # 0.5 events/sec is well below threshold 2.0
    obs = _make_observed([1.0, 3.0, 5.0, 7.0], [0, 1, 2, 3])
    cfg = RegimeBaselineConfig(window_seconds=2.0, rate_threshold=2.0)
    transitions = recover_regimes_baseline(obs, cfg, horizon=8.0)
    # Expect a single "calm" label at start.
    assert transitions[0] == (0.0, 0)
    assert all(label == 0 for _, label in transitions)


def test_recover_regimes_detects_transition_to_active():
    # First 4 seconds: 2 events (rate 1/s, below 3.0 threshold)
    # Next 4 seconds: 16 events (rate 4/s, above threshold)
    times = [1.0, 3.0] + list(np.linspace(4.1, 7.9, 16))
    types = [0, 1] + [2] * 16
    obs = _make_observed(times, types)
    cfg = RegimeBaselineConfig(window_seconds=2.0, rate_threshold=3.0)
    transitions = recover_regimes_baseline(obs, cfg, horizon=8.0)
    labels = [t[1] for t in transitions]
    # Must include at least one calm and one active label.
    assert 0 in labels and 1 in labels
    # The first label is calm.
    assert transitions[0][1] == 0
    # The first transition to active must occur within 2 * window_seconds of t=4.
    active_starts = [t[0] for t in transitions if t[1] == 1]
    assert any(4.0 <= s <= 8.0 for s in active_starts)


def test_recover_regimes_hysteresis_suppresses_short_flips():
    # Window-by-window oscillation between rates straddling the threshold.
    # With hysteresis_seconds = 4.0, only the first crossing is recorded.
    times = []
    types = []
    # Pattern: 10 events in each odd-indexed 1-second window, 0 in even windows.
    for w in range(8):
        if w % 2 == 1:
            for i in range(10):
                times.append(w + (i + 1) * 0.05)
                types.append(0)
    obs = _make_observed(times, types)
    cfg_no_hysteresis = RegimeBaselineConfig(
        window_seconds=1.0, rate_threshold=5.0, hysteresis_seconds=0.0
    )
    transitions_no_hys = recover_regimes_baseline(
        obs, cfg_no_hysteresis, horizon=8.0
    )
    cfg_hysteresis = RegimeBaselineConfig(
        window_seconds=1.0, rate_threshold=5.0, hysteresis_seconds=4.0
    )
    transitions_hys = recover_regimes_baseline(
        obs, cfg_hysteresis, horizon=8.0
    )
    # Hysteresis must produce no more transitions than the un-hysteresis case.
    assert len(transitions_hys) <= len(transitions_no_hys)


# ---------------------------------------------------------------------------
# score_meta_order_activity
# ---------------------------------------------------------------------------


def test_score_meta_order_inactive_when_no_imbalance():
    # Many events but balanced direction.
    times = list(np.linspace(0.05, 4.95, 50))
    types = [0, 1, 2, 3] * 12 + [0, 1]
    obs = _make_observed(times, types[:50])
    cfg = MetaOrderBaselineConfig(
        window_seconds=1.0, score_threshold=2.0, direction="any"
    )
    intervals = score_meta_order_activity(obs, cfg, horizon=5.0)
    assert intervals == []


def test_score_meta_order_active_when_directional_burst():
    # First half quiet, second half a burst of mostly BID_UP and ASK_UP.
    quiet_times = [0.5, 1.0, 1.5, 2.0]
    quiet_types = [1, 3, 1, 3]  # all DOWN events
    burst_times = list(np.linspace(2.05, 4.95, 30))
    burst_types = [0, 2] * 15  # all UP events
    obs = _make_observed(quiet_times + burst_times, quiet_types + burst_types)
    cfg = MetaOrderBaselineConfig(
        window_seconds=1.0,
        score_threshold=2.0,
        direction="up",
        min_active_seconds=0.5,
    )
    intervals = score_meta_order_activity(obs, cfg, horizon=5.0)
    assert len(intervals) >= 1
    start, end, score = intervals[0]
    # The burst lies entirely in [2, 5); the inferred interval must overlap there.
    assert end > 2.0
    assert score > cfg.score_threshold


def test_score_meta_order_rejects_invalid_direction():
    obs = _make_observed([], [])
    cfg = MetaOrderBaselineConfig(
        window_seconds=1.0, score_threshold=1.0, direction="left"
    )
    with pytest.raises(ValueError):
        score_meta_order_activity(obs, cfg, horizon=1.0)


# ---------------------------------------------------------------------------
# true_regime_at
# ---------------------------------------------------------------------------


def test_true_regime_at_step_function():
    rt = [0.0, 5.0, 10.0]
    rv = [0, 1, 0]
    qt = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
    out = true_regime_at(qt, rt, rv)
    assert out.tolist() == [0, 0, 1, 1, 0, 0]


def test_true_regime_at_query_before_first_jump_uses_initial_label():
    rt = [3.0, 6.0]
    rv = [0, 1]
    out = true_regime_at([1.0], rt, rv)
    assert out.tolist() == [0]


def test_true_regime_at_handles_single_regime_with_empty_jumps():
    rt = []
    rv = [0]
    out = true_regime_at([0.0, 1.0, 2.0], rt, rv)
    assert out.tolist() == [0, 0, 0]
