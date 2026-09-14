"""Tests for `simulator.markout` — Layer-3 markout harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.markout import MarkoutHarness, MarkoutResult, MarkoutPerFill
from simulator.skew_agent import Fill


def _make_fill(t, side, mid_at_fill, posterior=0.5):
    return Fill(
        time=float(t),
        side=str(side),
        skew_ticks=1.0,
        fill_probability=0.4,
        mid_at_fill=float(mid_at_fill),
        posterior=float(posterior),
        inventory_before=0.0,
        inventory_after=0.0,
    )


def test_harness_validates_inputs():
    with pytest.raises(ValueError):
        MarkoutHarness(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        MarkoutHarness(np.array([0.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        # Non-monotone times
        MarkoutHarness(np.array([0.0, 1.0, 0.5]), np.array([1.0, 1.1, 1.2]))


def test_mid_at_lookup_step_function():
    """Right-continuous step lookup returns the most recent mid <= t."""
    harness = MarkoutHarness(
        mid_times=np.array([0.0, 1.0, 5.0, 10.0]),
        mid_values=np.array([1.10, 1.11, 1.12, 1.13]),
    )
    assert harness.mid_at(-1.0) == pytest.approx(1.10)
    assert harness.mid_at(0.0) == pytest.approx(1.10)
    assert harness.mid_at(0.5) == pytest.approx(1.10)
    assert harness.mid_at(1.0) == pytest.approx(1.11)
    assert harness.mid_at(4.999) == pytest.approx(1.11)
    assert harness.mid_at(5.0) == pytest.approx(1.12)
    assert harness.mid_at(10.0) == pytest.approx(1.13)


def test_markout_horizon_correctness():
    """Hand-constructed mid trace + a single bid fill: mid at t_fill+5
    must be picked correctly."""
    times = np.linspace(0.0, 100.0, 1001)
    mids = 1.10 + 0.001 * times  # mid drifts +0.001 per second
    harness = MarkoutHarness(times, mids)
    # Bid fill at t=20.0, mid at fill = 1.12 (approx 1.10 + 0.001*20)
    f = _make_fill(t=20.0, side="bid", mid_at_fill=1.12)
    res = harness.compute([f], horizons_seconds=(5.0,))
    assert len(res.per_fill) == 1
    rec = res.per_fill[0]
    assert rec.horizon_seconds == pytest.approx(5.0)
    # mid at 25.0 ~ 1.125
    assert rec.mid_at_horizon == pytest.approx(1.10 + 0.001 * 25.0, abs=1e-6)
    # bid-side: signed = + (mid_25 - mid_20) ~ +0.005
    assert rec.signed_markout == pytest.approx(0.005, abs=1e-6)
    assert rec.toxicity == pytest.approx(0.005, abs=1e-6)


def test_per_side_aggregation_sign_convention():
    """Buy-side fills get + delta, sell-side fills get - delta."""
    times = np.array([0.0, 10.0, 20.0])
    mids = np.array([1.10, 1.20, 1.30])
    harness = MarkoutHarness(times, mids)
    # Bid fill at t=0 with mid=1.10; horizon 10s -> mid_at_h=1.20
    bid_fill = _make_fill(t=0.0, side="bid", mid_at_fill=1.10)
    # Ask fill at t=0 with mid=1.10; horizon 10s -> mid_at_h=1.20
    ask_fill = _make_fill(t=0.0, side="ask", mid_at_fill=1.10)
    res = harness.compute([bid_fill, ask_fill], horizons_seconds=(10.0,))
    assert len(res.per_fill) == 2
    bid_rec = next(r for r in res.per_fill if r.side == "bid")
    ask_rec = next(r for r in res.per_fill if r.side == "ask")
    # Bid side: mid moved +0.10, LP bought -> +0.10 (adverse)
    assert bid_rec.signed_markout == pytest.approx(+0.10, abs=1e-9)
    # Ask side: mid moved +0.10, LP sold -> -0.10 (favourable)
    assert ask_rec.signed_markout == pytest.approx(-0.10, abs=1e-9)
    # Both have toxicity 0.10
    assert bid_rec.toxicity == pytest.approx(0.10)
    assert ask_rec.toxicity == pytest.approx(0.10)


def test_multi_horizon_coherence():
    """1, 5, 30 s horizons computed in one call: independent and
    correctly indexed."""
    times = np.linspace(0.0, 60.0, 6001)
    # Linear drift: mid = 1.10 + 0.0001 * t
    mids = 1.10 + 0.0001 * times
    harness = MarkoutHarness(times, mids)
    f = _make_fill(t=10.0, side="bid", mid_at_fill=1.10 + 0.0001 * 10.0)
    res = harness.compute(
        [f], horizons_seconds=(1.0, 5.0, 30.0)
    )
    assert len(res.per_fill) == 3
    # Per-horizon aggregates exist
    for h in (1.0, 5.0, 30.0):
        agg = res.aggregate_for(h)
        assert agg["n_fills"] == 1
        assert agg["n_bid_fills"] == 1
    # Per-horizon mids: 1.10+0.0001*11, *15, *40
    rec_1 = next(r for r in res.per_fill if r.horizon_seconds == 1.0)
    rec_5 = next(r for r in res.per_fill if r.horizon_seconds == 5.0)
    rec_30 = next(r for r in res.per_fill if r.horizon_seconds == 30.0)
    assert rec_1.mid_at_horizon == pytest.approx(1.10 + 0.0001 * 11.0, abs=1e-7)
    assert rec_5.mid_at_horizon == pytest.approx(1.10 + 0.0001 * 15.0, abs=1e-7)
    assert rec_30.mid_at_horizon == pytest.approx(1.10 + 0.0001 * 40.0, abs=1e-7)


def test_horizon_past_trace_right_edge_dropped():
    """Fills whose horizon falls past the trace end are excluded for that
    horizon's aggregates."""
    times = np.array([0.0, 10.0])
    mids = np.array([1.0, 1.0])
    harness = MarkoutHarness(times, mids)
    # Fill at t=8.0; horizon 5s = 13.0 > 10.0 (trace end)
    f = _make_fill(t=8.0, side="bid", mid_at_fill=1.0)
    res = harness.compute([f], horizons_seconds=(5.0,))
    assert len(res.per_fill) == 0
    assert res.aggregate_for(5.0)["n_fills"] == 0


def test_per_horizon_aggregates_independent():
    """Two fills, two horizons. Per-horizon aggregates must use only the
    records belonging to that horizon."""
    times = np.linspace(0.0, 60.0, 601)
    mids = 1.10 + 0.0001 * times
    harness = MarkoutHarness(times, mids)
    f1 = _make_fill(t=5.0, side="bid", mid_at_fill=1.10 + 0.0001 * 5.0)
    f2 = _make_fill(t=20.0, side="ask", mid_at_fill=1.10 + 0.0001 * 20.0)
    res = harness.compute([f1, f2], horizons_seconds=(1.0, 30.0))
    # Each horizon has 2 records
    for h in (1.0, 30.0):
        agg = res.aggregate_for(h)
        assert agg["n_fills"] == 2
        assert agg["n_bid_fills"] == 1
        assert agg["n_ask_fills"] == 1


def test_invalid_side_raises():
    times = np.array([0.0, 10.0])
    mids = np.array([1.0, 1.0])
    harness = MarkoutHarness(times, mids)
    bogus = Fill(
        time=1.0,
        side="middle",
        skew_ticks=0.0,
        fill_probability=0.5,
        mid_at_fill=1.0,
        posterior=0.5,
        inventory_before=0.0,
        inventory_after=0.0,
    )
    with pytest.raises(ValueError):
        harness.compute([bogus], horizons_seconds=(1.0,))


def test_invalid_horizon_raises():
    harness = MarkoutHarness(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    f = _make_fill(t=0.0, side="bid", mid_at_fill=1.0)
    with pytest.raises(ValueError):
        harness.compute([f], horizons_seconds=())
    with pytest.raises(ValueError):
        harness.compute([f], horizons_seconds=(0.0,))
    with pytest.raises(ValueError):
        harness.compute([f], horizons_seconds=(-1.0,))


def test_json_round_trip():
    times = np.linspace(0.0, 60.0, 601)
    mids = 1.10 + 0.0001 * times
    harness = MarkoutHarness(times, mids)
    fills = [
        _make_fill(t=5.0, side="bid", mid_at_fill=1.10 + 0.0001 * 5.0),
        _make_fill(t=20.0, side="ask", mid_at_fill=1.10 + 0.0001 * 20.0),
    ]
    res = harness.compute(fills, horizons_seconds=(1.0, 5.0, 30.0))
    s = res.to_json()
    res2 = MarkoutResult.from_json(s)
    assert tuple(res2.horizons_seconds) == res.horizons_seconds
    assert len(res2.per_fill) == len(res.per_fill)
    for r1, r2 in zip(res.per_fill, res2.per_fill):
        assert r1.fill_time == r2.fill_time
        assert r1.side == r2.side
        assert r1.horizon_seconds == r2.horizon_seconds
        assert r1.signed_markout == pytest.approx(r2.signed_markout, abs=1e-12)
        assert r1.toxicity == pytest.approx(r2.toxicity, abs=1e-12)
    for h in res.horizons_seconds:
        a1 = res.aggregate_for(h)
        a2 = res2.aggregate_for(h)
        assert a1["n_fills"] == a2["n_fills"]
        if a1["n_fills"] > 0:
            assert a1["mean_signed_markout"] == pytest.approx(
                a2["mean_signed_markout"], abs=1e-12
            )
