"""
skew_agent.py — Layer-3 toxicity-aware closed-form skew rule (D1, MSc scope).

Implements the deterministic parameterised skew rule consumed by the
Layer-3 markout experiment in `dissertation_artifacts/build_layer3_d1_skew_markout.py`.
The skew rule is a closed-form mapping from
`(p(z = informed | O_{0:T}), inventory)` to per-side skew offsets in
ticks; no HJB, no value-function approximation, no RL. A logistic fill
model converts per-side skew into a fill probability, calibrated so that
`P(fill | skew=0) ~= 0.4` and `P(fill | skew=max) ~= 0.05` under the
spec defaults.

Mathematical rule
-----------------
With posterior `p` in [0, 1] and signed inventory `q`, per-side skew is

    bid_skew(p, q) =  base_half + s * p * max_skew + lambda * q
    ask_skew(p, q) =  base_half + s * p * max_skew - lambda * q

where `base_half = base_spread_ticks / 2`, `s = posterior_sensitivity`,
`max_skew = max_skew_ticks`, `lambda = inventory_penalty`. Both per-side
skews are clipped into `[0, max_skew_ticks]` after the linear map. A
positive inventory tightens the ask (`-lambda * q`) and widens the bid
(`+lambda * q`), encouraging the inventory to be sold back. The mapping
is symmetric across the two sides under high `p`: the rule does NOT use
side-of-meta-order information because the MSc HMM is two-state
(calm / informed) and does not expose direction; widening both sides
under high informed-flow inference is the appropriate conservative
response.

Fill model
----------

    P(fill | skew) = 1 / (1 + exp(alpha + beta * skew_ticks)),

with `alpha`, `beta` set such that `P(fill | 0) = 0.4` and
`P(fill | max_skew) = 0.05` under the default `max_skew_ticks = 5`. The
fill probability is per-side and applied independently at each market
event under the simulator's matched-RNG protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import json

import numpy as np


__all__ = [
    "SkewAgentConfig",
    "SkewAgent",
    "Fill",
    "compute_fill_logistic_params",
]


# ---------------------------------------------------------------------------
# Logistic fill-probability calibration
# ---------------------------------------------------------------------------


def compute_fill_logistic_params(
    *,
    p_at_zero: float = 0.4,
    p_at_max: float = 0.05,
    max_skew_ticks: float = 5.0,
) -> Tuple[float, float]:
    """Solve `(alpha, beta)` such that
    `1 / (1 + exp(alpha))         = p_at_zero` and
    `1 / (1 + exp(alpha + beta*M)) = p_at_max`.

    Closed-form: `alpha = log(1/p_at_zero - 1)`,
    `beta = (log(1/p_at_max - 1) - alpha) / M`.
    """
    if not (0.0 < p_at_zero < 1.0 and 0.0 < p_at_max < 1.0):
        raise ValueError(
            f"p_at_zero and p_at_max must be in (0, 1); got {p_at_zero}, {p_at_max}"
        )
    if p_at_max >= p_at_zero:
        raise ValueError(
            f"p_at_max ({p_at_max}) must be < p_at_zero ({p_at_zero}) for the "
            "logistic to be monotone-decreasing in skew"
        )
    if max_skew_ticks <= 0.0:
        raise ValueError(f"max_skew_ticks must be > 0; got {max_skew_ticks}")
    alpha = float(np.log(1.0 / p_at_zero - 1.0))
    beta = float((np.log(1.0 / p_at_max - 1.0) - alpha) / max_skew_ticks)
    return alpha, beta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SkewAgentConfig:
    """Configuration for the closed-form Layer-3 skew rule.

    Defaults match the prompt-10msc spec.

    Fields
    ------
    base_spread_ticks : float
        Symmetric quote spread when the posterior is flat. Default 2.0.
    max_skew_ticks : float
        Saturation cap on per-side skew. Default 5.0.
    posterior_sensitivity : float
        Multiplier mapping `p(z = informed)` to per-side skew offset. The
        spec defaults to 1.0 so that at `p = 0.5` the per-side offset
        contribution is `0.5 * max_skew_ticks`. Set to 0.0 for the
        Hawkes-blind baseline (Episode A in the headline experiment).
    inventory_penalty : float
        Coefficient on the (anti-)symmetric inventory adjustment, in
        ticks per unit inventory. Default 0.05.
    fill_probability_p_at_zero : float
        Target `P(fill | skew = 0)` for the logistic fill model.
        Default 0.4.
    fill_probability_p_at_max : float
        Target `P(fill | skew = max_skew_ticks)`. Default 0.05.
    tick_size_in_pips : float
        Conversion factor between ticks and pips for the markout harness
        and downstream P&L. Default 0.1 (sub-pip FX tick).
    """

    base_spread_ticks: float = 2.0
    max_skew_ticks: float = 5.0
    posterior_sensitivity: float = 1.0
    inventory_penalty: float = 0.05
    fill_probability_p_at_zero: float = 0.4
    fill_probability_p_at_max: float = 0.05
    tick_size_in_pips: float = 0.1

    def __post_init__(self) -> None:
        if self.base_spread_ticks <= 0.0:
            raise ValueError(
                f"base_spread_ticks must be > 0; got {self.base_spread_ticks}"
            )
        if self.max_skew_ticks <= 0.0:
            raise ValueError(
                f"max_skew_ticks must be > 0; got {self.max_skew_ticks}"
            )
        if self.posterior_sensitivity < 0.0:
            raise ValueError(
                f"posterior_sensitivity must be >= 0; "
                f"got {self.posterior_sensitivity}"
            )
        if self.inventory_penalty < 0.0:
            raise ValueError(
                f"inventory_penalty must be >= 0; got {self.inventory_penalty}"
            )
        if not (0.0 < self.fill_probability_p_at_zero < 1.0):
            raise ValueError(
                f"fill_probability_p_at_zero must be in (0, 1); "
                f"got {self.fill_probability_p_at_zero}"
            )
        if not (0.0 < self.fill_probability_p_at_max < 1.0):
            raise ValueError(
                f"fill_probability_p_at_max must be in (0, 1); "
                f"got {self.fill_probability_p_at_max}"
            )
        if self.fill_probability_p_at_max >= self.fill_probability_p_at_zero:
            raise ValueError(
                f"fill_probability_p_at_max ({self.fill_probability_p_at_max}) "
                f"must be < fill_probability_p_at_zero "
                f"({self.fill_probability_p_at_zero})"
            )
        if self.tick_size_in_pips <= 0.0:
            raise ValueError(
                f"tick_size_in_pips must be > 0; got {self.tick_size_in_pips}"
            )

    @property
    def fill_probability_alpha(self) -> float:
        a, _ = compute_fill_logistic_params(
            p_at_zero=self.fill_probability_p_at_zero,
            p_at_max=self.fill_probability_p_at_max,
            max_skew_ticks=self.max_skew_ticks,
        )
        return a

    @property
    def fill_probability_beta(self) -> float:
        _, b = compute_fill_logistic_params(
            p_at_zero=self.fill_probability_p_at_zero,
            p_at_max=self.fill_probability_p_at_max,
            max_skew_ticks=self.max_skew_ticks,
        )
        return b

    def to_json(self) -> str:
        return json.dumps(
            {
                "base_spread_ticks": float(self.base_spread_ticks),
                "max_skew_ticks": float(self.max_skew_ticks),
                "posterior_sensitivity": float(self.posterior_sensitivity),
                "inventory_penalty": float(self.inventory_penalty),
                "fill_probability_p_at_zero": float(
                    self.fill_probability_p_at_zero
                ),
                "fill_probability_p_at_max": float(
                    self.fill_probability_p_at_max
                ),
                "tick_size_in_pips": float(self.tick_size_in_pips),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "SkewAgentConfig":
        d = json.loads(s)
        return cls(
            base_spread_ticks=float(d["base_spread_ticks"]),
            max_skew_ticks=float(d["max_skew_ticks"]),
            posterior_sensitivity=float(d["posterior_sensitivity"]),
            inventory_penalty=float(d["inventory_penalty"]),
            fill_probability_p_at_zero=float(d["fill_probability_p_at_zero"]),
            fill_probability_p_at_max=float(d["fill_probability_p_at_max"]),
            tick_size_in_pips=float(d["tick_size_in_pips"]),
        )


# ---------------------------------------------------------------------------
# Fill record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """One realised fill emitted by `SkewAgent.step`.

    Fields
    ------
    time : float
        Time of the market event at which the fill occurred (s).
    side : str
        "bid" if the LP's bid was hit (LP bought), "ask" if the LP's ask
        was lifted (LP sold).
    skew_ticks : float
        Per-side skew applied at this fill (>= 0).
    fill_probability : float
        `P(fill | skew_ticks)` evaluated at the fill.
    mid_at_fill : float
        Mid quote at the fill time, in price units (`tick_size`-scaled).
    posterior : float
        HMM posterior `p(z = informed | O_{0:t})` consumed for this skew.
    inventory_before : float
        Signed LP inventory in units immediately before this fill.
    inventory_after : float
        Signed LP inventory in units immediately after this fill.
    """

    time: float
    side: str
    skew_ticks: float
    fill_probability: float
    mid_at_fill: float
    posterior: float
    inventory_before: float
    inventory_after: float


# ---------------------------------------------------------------------------
# SkewAgent
# ---------------------------------------------------------------------------


class SkewAgent:
    """Deterministic Layer-3 skew agent.

    The agent is constructed from a `SkewAgentConfig` and exposes:
        - `quote(t, posterior_at_t, inventory)`: the closed-form skew rule.
        - `fill_probability(skew_ticks)`: logistic acceptance probability.
        - `step(t, posterior_at_t, inventory, mid_at_t, u_bid, u_ask)`:
          per-event fill resolution. Each side's fill is decided by the
          uniform draw `u_side < P(fill | skew_side)`. If both sides
          fill in the same step the symmetric tie-breaker emits the bid
          fill first (the LP buys then sells) — this collapses to two
          back-to-back fills under the matched-RNG protocol.

    No optimal control. No HJB. No value-function approximation. The
    rule is a pure function of `(p(z=informed), inventory)`.
    """

    def __init__(self, cfg: SkewAgentConfig) -> None:
        self._cfg = cfg
        self._alpha, self._beta = compute_fill_logistic_params(
            p_at_zero=cfg.fill_probability_p_at_zero,
            p_at_max=cfg.fill_probability_p_at_max,
            max_skew_ticks=cfg.max_skew_ticks,
        )

    @property
    def config(self) -> SkewAgentConfig:
        return self._cfg

    def quote(
        self, t: float, posterior_at_t: float, inventory: float
    ) -> Tuple[float, float]:
        """Closed-form skew rule.

        Returns
        -------
        (bid_skew_ticks, ask_skew_ticks) — both >= 0 and <= max_skew_ticks.
        """
        if not (0.0 <= posterior_at_t <= 1.0):
            raise ValueError(
                f"posterior_at_t must be in [0, 1]; got {posterior_at_t}"
            )
        cfg = self._cfg
        base_half = 0.5 * cfg.base_spread_ticks
        posterior_skew = (
            cfg.posterior_sensitivity * posterior_at_t * cfg.max_skew_ticks
        )
        inv_skew = cfg.inventory_penalty * inventory
        bid = base_half + posterior_skew + inv_skew
        ask = base_half + posterior_skew - inv_skew
        bid = float(np.clip(bid, 0.0, cfg.max_skew_ticks))
        ask = float(np.clip(ask, 0.0, cfg.max_skew_ticks))
        return bid, ask

    def fill_probability(self, skew_ticks: float) -> float:
        """Logistic fill probability `1 / (1 + exp(alpha + beta * skew))`."""
        z = self._alpha + self._beta * float(skew_ticks)
        return float(1.0 / (1.0 + np.exp(z)))

    def step(
        self,
        t: float,
        posterior_at_t: float,
        inventory: float,
        mid_at_t: float,
        u_bid: float,
        u_ask: float,
    ) -> Tuple[Optional[Fill], Optional[Fill], float]:
        """Resolve one market event into zero, one, or two `Fill`s.

        Parameters
        ----------
        t : float
            Market event time.
        posterior_at_t : float
            HMM posterior `p(z = informed | O_{0:t})` at the event time.
        inventory : float
            Signed LP inventory in units immediately before the event.
        mid_at_t : float
            Mid quote at the event time (price units).
        u_bid, u_ask : float
            U(0, 1) draws for the bid and ask side fills. The build
            runner draws these from a matched RNG so that Episode A and
            Episode B see identical noise.

        Returns
        -------
        (bid_fill | None, ask_fill | None, inventory_after)
            A bid fill represents the LP buying (inventory increases by
            +1 unit); an ask fill represents the LP selling
            (inventory decreases by 1 unit). When both sides fill, the
            bid is emitted first.
        """
        bid_skew, ask_skew = self.quote(t, posterior_at_t, inventory)
        p_bid = self.fill_probability(bid_skew)
        p_ask = self.fill_probability(ask_skew)
        bid_fill: Optional[Fill] = None
        ask_fill: Optional[Fill] = None

        inv = float(inventory)
        if u_bid < p_bid:
            bid_fill = Fill(
                time=float(t),
                side="bid",
                skew_ticks=float(bid_skew),
                fill_probability=float(p_bid),
                mid_at_fill=float(mid_at_t),
                posterior=float(posterior_at_t),
                inventory_before=float(inv),
                inventory_after=float(inv + 1.0),
            )
            inv += 1.0
        if u_ask < p_ask:
            ask_fill = Fill(
                time=float(t),
                side="ask",
                skew_ticks=float(ask_skew),
                fill_probability=float(p_ask),
                mid_at_fill=float(mid_at_t),
                posterior=float(posterior_at_t),
                inventory_before=float(inv),
                inventory_after=float(inv - 1.0),
            )
            inv -= 1.0
        return bid_fill, ask_fill, inv
