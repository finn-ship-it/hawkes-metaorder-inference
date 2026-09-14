"""Tests for `simulator.skew_agent` — Layer-3 closed-form skew rule."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from simulator.skew_agent import (
    Fill,
    SkewAgent,
    SkewAgentConfig,
    compute_fill_logistic_params,
)


def test_default_config_validates():
    cfg = SkewAgentConfig()
    assert cfg.base_spread_ticks == 2.0
    assert cfg.max_skew_ticks == 5.0
    assert cfg.posterior_sensitivity == 1.0
    assert cfg.inventory_penalty == 0.05
    # Logistic targets
    assert cfg.fill_probability_p_at_zero == pytest.approx(0.4)
    assert cfg.fill_probability_p_at_max == pytest.approx(0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_spread_ticks": 0.0},
        {"max_skew_ticks": -1.0},
        {"posterior_sensitivity": -0.1},
        {"inventory_penalty": -0.05},
        {"fill_probability_p_at_zero": 0.0},
        {"fill_probability_p_at_zero": 1.0},
        {"fill_probability_p_at_max": 0.0},
        {"fill_probability_p_at_max": 1.0},
        {
            "fill_probability_p_at_zero": 0.05,
            "fill_probability_p_at_max": 0.4,
        },
        {"tick_size_in_pips": 0.0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SkewAgentConfig(**kwargs)


def test_logistic_params_match_targets():
    alpha, beta = compute_fill_logistic_params(
        p_at_zero=0.4, p_at_max=0.05, max_skew_ticks=5.0
    )
    p0 = 1.0 / (1.0 + math.exp(alpha))
    p_max = 1.0 / (1.0 + math.exp(alpha + beta * 5.0))
    assert p0 == pytest.approx(0.4, abs=1e-12)
    assert p_max == pytest.approx(0.05, abs=1e-12)


def test_fill_probability_bounds():
    """`P(fill | skew)` is in [0, 1] for all skew values in the valid range."""
    cfg = SkewAgentConfig()
    agent = SkewAgent(cfg)
    skews = np.linspace(-1.0, cfg.max_skew_ticks + 1.0, 31)
    for s in skews:
        p = agent.fill_probability(float(s))
        assert 0.0 <= p <= 1.0, f"P(fill | skew={s}) = {p}"


def test_flat_posterior_zero_inventory_emits_symmetric_base_spread():
    """At p = 0.5 with sensitivity = 0 (baseline) and zero inventory, the
    per-side skew is the symmetric base half-spread."""
    cfg = SkewAgentConfig(posterior_sensitivity=0.0)
    agent = SkewAgent(cfg)
    bid, ask = agent.quote(t=42.0, posterior_at_t=0.5, inventory=0.0)
    assert bid == pytest.approx(cfg.base_spread_ticks / 2.0)
    assert ask == pytest.approx(cfg.base_spread_ticks / 2.0)


def test_monotone_response_in_posterior_increases_skew():
    """Increasing `p(z=informed)` increases per-side skew (and therefore
    decreases fill probability) monotonically when posterior_sensitivity > 0.

    The sensitivity is chosen such that the max-posterior skew sits below
    `max_skew_ticks`, so the strict-monotonicity property holds without
    being broken by the saturation clip.
    """
    cfg = SkewAgentConfig(
        posterior_sensitivity=0.6, inventory_penalty=0.0
    )
    agent = SkewAgent(cfg)
    posteriors = np.linspace(0.0, 1.0, 11)
    bids = []
    asks = []
    p_bids = []
    p_asks = []
    for p in posteriors:
        bid, ask = agent.quote(t=0.0, posterior_at_t=float(p), inventory=0.0)
        bids.append(bid)
        asks.append(ask)
        p_bids.append(agent.fill_probability(bid))
        p_asks.append(agent.fill_probability(ask))
    # Strictly increasing skews
    assert all(b2 > b1 for b1, b2 in zip(bids[:-1], bids[1:]))
    assert all(a2 > a1 for a1, a2 in zip(asks[:-1], asks[1:]))
    # Strictly decreasing fill probabilities
    assert all(p2 < p1 for p1, p2 in zip(p_bids[:-1], p_bids[1:]))
    assert all(p2 < p1 for p1, p2 in zip(p_asks[:-1], p_asks[1:]))


def test_inventory_consistency_positive_inventory_widens_bid_tightens_ask():
    """Positive inventory should encourage selling: ask shrinks, bid grows."""
    cfg = SkewAgentConfig(
        posterior_sensitivity=0.0, inventory_penalty=0.05
    )
    agent = SkewAgent(cfg)
    # Reference: zero inventory, p=0.5
    bid0, ask0 = agent.quote(t=0.0, posterior_at_t=0.5, inventory=0.0)
    # Long inventory: bid widens, ask tightens
    bid_long, ask_long = agent.quote(t=0.0, posterior_at_t=0.5, inventory=10.0)
    assert bid_long > bid0
    assert ask_long < ask0
    # Short inventory: bid tightens, ask widens
    bid_short, ask_short = agent.quote(
        t=0.0, posterior_at_t=0.5, inventory=-10.0
    )
    assert bid_short < bid0
    assert ask_short > ask0


def test_quote_clipped_to_max_skew():
    """At the maximum posterior + extreme inventory the per-side skew is
    clipped to `max_skew_ticks` (and never goes below zero)."""
    cfg = SkewAgentConfig(
        posterior_sensitivity=2.0, inventory_penalty=1.0, max_skew_ticks=5.0
    )
    agent = SkewAgent(cfg)
    # Long inventory, p=1.0: bid would otherwise be huge
    bid, ask = agent.quote(t=0.0, posterior_at_t=1.0, inventory=100.0)
    assert bid == pytest.approx(5.0)
    # Ask would otherwise go negative
    assert ask == pytest.approx(0.0)


def test_quote_rejects_invalid_posterior():
    cfg = SkewAgentConfig()
    agent = SkewAgent(cfg)
    with pytest.raises(ValueError):
        agent.quote(t=0.0, posterior_at_t=-0.1, inventory=0.0)
    with pytest.raises(ValueError):
        agent.quote(t=0.0, posterior_at_t=1.1, inventory=0.0)


def test_step_emits_fills_under_low_skew_high_u():
    """Step-level fill resolution: low skew + low U -> fills emitted."""
    cfg = SkewAgentConfig(posterior_sensitivity=0.0, inventory_penalty=0.0)
    agent = SkewAgent(cfg)
    # At p=0.0 inventory=0 the skew is base_half = 1.0 ticks for both sides.
    # P(fill | 1.0 tick) > P(fill | 5.0 tick); should fire on small u.
    bid_fill, ask_fill, inv_after = agent.step(
        t=10.0,
        posterior_at_t=0.0,
        inventory=0.0,
        mid_at_t=1.10000,
        u_bid=0.01,
        u_ask=0.01,
    )
    assert bid_fill is not None
    assert ask_fill is not None
    assert bid_fill.side == "bid"
    assert ask_fill.side == "ask"
    # Inventory: +1 then -1 -> 0
    assert inv_after == pytest.approx(0.0)


def test_step_no_fill_under_high_u():
    cfg = SkewAgentConfig()
    agent = SkewAgent(cfg)
    bid_fill, ask_fill, inv_after = agent.step(
        t=0.0,
        posterior_at_t=0.5,
        inventory=0.0,
        mid_at_t=1.0,
        u_bid=0.999,
        u_ask=0.999,
    )
    assert bid_fill is None
    assert ask_fill is None
    assert inv_after == pytest.approx(0.0)


def test_json_round_trip():
    cfg = SkewAgentConfig(
        base_spread_ticks=3.0,
        max_skew_ticks=4.0,
        posterior_sensitivity=0.7,
        inventory_penalty=0.025,
        fill_probability_p_at_zero=0.5,
        fill_probability_p_at_max=0.1,
        tick_size_in_pips=0.05,
    )
    s = cfg.to_json()
    cfg2 = SkewAgentConfig.from_json(s)
    assert cfg2.base_spread_ticks == cfg.base_spread_ticks
    assert cfg2.max_skew_ticks == cfg.max_skew_ticks
    assert cfg2.posterior_sensitivity == cfg.posterior_sensitivity
    assert cfg2.inventory_penalty == cfg.inventory_penalty
    assert cfg2.fill_probability_p_at_zero == cfg.fill_probability_p_at_zero
    assert cfg2.fill_probability_p_at_max == cfg.fill_probability_p_at_max
    assert cfg2.tick_size_in_pips == cfg.tick_size_in_pips


def test_baseline_episode_a_collapses_to_no_posterior_sensitivity():
    """Episode A baseline: posterior_sensitivity = 0 means the rule does NOT
    consume the posterior; quotes are independent of `p`."""
    cfg = SkewAgentConfig(
        posterior_sensitivity=0.0, inventory_penalty=0.0
    )
    agent = SkewAgent(cfg)
    quotes = [agent.quote(0.0, p, 0.0) for p in (0.0, 0.3, 0.7, 1.0)]
    base_half = cfg.base_spread_ticks / 2.0
    for bid, ask in quotes:
        assert bid == pytest.approx(base_half)
        assert ask == pytest.approx(base_half)
