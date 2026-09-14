"""
markout.py — Layer-3 markout harness for D1 fills.

Computes per-fill toxicity at horizons {1, 5, 30} s. For each `Fill`,
reads the realised mid quote at `t_fill + horizon` from a mid trace and
emits a signed markout per the convention

    buy-side  (LP bought a unit): markout = + (mid_at_horizon - mid_at_fill)
    sell-side (LP sold a unit):   markout = - (mid_at_horizon - mid_at_fill)

A positive markout is **adverse** for the LP: under a bid fill the LP is
left long an instrument whose mid moved away from the LP, and the LP's
mark-to-market loss is proportional to the markout. Realised toxicity
is `|markout|`; signed mean markout is the directional bias.

Pricing convention
------------------
The mid trace is queried at `t_fill + horizon` by **right-continuous
step interpolation**: the most recent mid quote at or before the query
time is used. This matches the discrete-event semantics of the D1
simulator: between two market events the mid is constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import json

import numpy as np


__all__ = [
    "MarkoutHarness",
    "MarkoutResult",
    "MarkoutPerFill",
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkoutPerFill:
    """One fill-level markout record at a single horizon.

    Fields
    ------
    fill_time : float
    side : str
        "bid" or "ask".
    horizon_seconds : float
    mid_at_fill : float
    mid_at_horizon : float
    signed_markout : float
        Per-side signed markout (see module docstring).
    toxicity : float
        `abs(signed_markout)`.
    posterior : float
        HMM posterior at the fill time (kept for downstream stratification).
    """

    fill_time: float
    side: str
    horizon_seconds: float
    mid_at_fill: float
    mid_at_horizon: float
    signed_markout: float
    toxicity: float
    posterior: float


@dataclass
class MarkoutResult:
    """Aggregated output of `MarkoutHarness.compute`.

    Per-horizon aggregates are computed on the pooled per-fill records
    using the signed-markout sign convention. `n_fills` counts fills
    whose horizon end-time falls inside the mid trace; fills with
    `t_fill + horizon` past the trace right edge are dropped from the
    aggregates for that horizon.
    """

    horizons_seconds: Tuple[float, ...]
    per_fill: List[MarkoutPerFill] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)

    def aggregate_for(self, horizon_seconds: float) -> dict:
        return self.aggregates[float(horizon_seconds)]

    def to_json(self) -> str:
        payload = {
            "horizons_seconds": list(self.horizons_seconds),
            "per_fill": [
                {
                    "fill_time": float(r.fill_time),
                    "side": str(r.side),
                    "horizon_seconds": float(r.horizon_seconds),
                    "mid_at_fill": float(r.mid_at_fill),
                    "mid_at_horizon": float(r.mid_at_horizon),
                    "signed_markout": float(r.signed_markout),
                    "toxicity": float(r.toxicity),
                    "posterior": float(r.posterior),
                }
                for r in self.per_fill
            ],
            "aggregates": {
                f"{float(k)}": v for k, v in self.aggregates.items()
            },
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "MarkoutResult":
        d = json.loads(s)
        horizons = tuple(float(h) for h in d["horizons_seconds"])
        per_fill = [
            MarkoutPerFill(
                fill_time=float(r["fill_time"]),
                side=str(r["side"]),
                horizon_seconds=float(r["horizon_seconds"]),
                mid_at_fill=float(r["mid_at_fill"]),
                mid_at_horizon=float(r["mid_at_horizon"]),
                signed_markout=float(r["signed_markout"]),
                toxicity=float(r["toxicity"]),
                posterior=float(r["posterior"]),
            )
            for r in d["per_fill"]
        ]
        aggregates = {float(k): v for k, v in d["aggregates"].items()}
        return cls(
            horizons_seconds=horizons,
            per_fill=per_fill,
            aggregates=aggregates,
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class MarkoutHarness:
    """Compute per-fill markouts on a (sorted) mid trace.

    The mid trace is supplied as `(times, mids)` arrays, where
    `times[i]` is the time at which `mids[i]` becomes the active mid.
    `times` must be strictly increasing. Both arrays must have equal
    length. The trace is right-continuous: at any query time `t`, the
    mid is `mids[k]` where `k` is the largest index with `times[k] <= t`.
    For `t < times[0]` the harness uses the initial mid `mids[0]`.
    """

    def __init__(
        self,
        mid_times: np.ndarray,
        mid_values: np.ndarray,
    ) -> None:
        times = np.asarray(mid_times, dtype=np.float64)
        mids = np.asarray(mid_values, dtype=np.float64)
        if times.shape != mids.shape:
            raise ValueError(
                f"mid_times and mid_values shape mismatch: "
                f"{times.shape} vs {mids.shape}"
            )
        if times.ndim != 1:
            raise ValueError("mid_times must be 1-D")
        if times.size == 0:
            raise ValueError("mid trace must contain at least one point")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("mid_times must be non-decreasing")
        self._times = times
        self._mids = mids

    @property
    def trace_t_max(self) -> float:
        return float(self._times[-1])

    def mid_at(self, t: float) -> float:
        """Right-continuous step lookup of the mid at time `t`.

        For `t < times[0]` the initial mid is returned. For
        `t >= times[-1]` the last mid is returned (the caller is
        responsible for filtering fills whose horizon falls past the
        trace right edge).
        """
        if t < self._times[0]:
            return float(self._mids[0])
        idx = int(np.searchsorted(self._times, t, side="right") - 1)
        idx = max(0, min(idx, self._times.size - 1))
        return float(self._mids[idx])

    def compute(
        self,
        fills: Sequence,
        horizons_seconds: Tuple[float, ...] = (1.0, 5.0, 30.0),
    ) -> MarkoutResult:
        """Compute per-fill, per-horizon markouts and aggregate.

        Parameters
        ----------
        fills : sequence of `simulator.skew_agent.Fill`
            Each must expose `.time`, `.side`, `.mid_at_fill`,
            `.posterior`. Other fields are ignored here.
        horizons_seconds : tuple of float
            Horizons at which to compute markouts. Default
            `(1.0, 5.0, 30.0)`.

        Returns
        -------
        MarkoutResult.
        """
        if len(horizons_seconds) == 0:
            raise ValueError("at least one horizon is required")
        for h in horizons_seconds:
            if float(h) <= 0.0:
                raise ValueError(f"horizon must be > 0; got {h}")

        per_fill: List[MarkoutPerFill] = []
        for h in horizons_seconds:
            for f in fills:
                t_query = float(f.time) + float(h)
                if t_query > self.trace_t_max:
                    continue
                mid_h = self.mid_at(t_query)
                delta = mid_h - float(f.mid_at_fill)
                if str(f.side) == "bid":
                    signed = +float(delta)
                elif str(f.side) == "ask":
                    signed = -float(delta)
                else:
                    raise ValueError(
                        f"unrecognised side {f.side!r}; "
                        "expected 'bid' or 'ask'"
                    )
                per_fill.append(
                    MarkoutPerFill(
                        fill_time=float(f.time),
                        side=str(f.side),
                        horizon_seconds=float(h),
                        mid_at_fill=float(f.mid_at_fill),
                        mid_at_horizon=float(mid_h),
                        signed_markout=float(signed),
                        toxicity=float(abs(signed)),
                        posterior=float(getattr(f, "posterior", 0.0)),
                    )
                )

        aggregates: dict = {}
        for h in horizons_seconds:
            recs = [r for r in per_fill if r.horizon_seconds == float(h)]
            sides_bid = [r for r in recs if r.side == "bid"]
            sides_ask = [r for r in recs if r.side == "ask"]
            aggregates[float(h)] = {
                "n_fills": int(len(recs)),
                "n_bid_fills": int(len(sides_bid)),
                "n_ask_fills": int(len(sides_ask)),
                "mean_signed_markout": _safe_mean([r.signed_markout for r in recs]),
                "mean_toxicity": _safe_mean([r.toxicity for r in recs]),
                "median_toxicity": _safe_median([r.toxicity for r in recs]),
                "p10_signed_markout": _safe_pct([r.signed_markout for r in recs], 10.0),
                "p50_signed_markout": _safe_pct([r.signed_markout for r in recs], 50.0),
                "p90_signed_markout": _safe_pct([r.signed_markout for r in recs], 90.0),
                "mean_signed_markout_bid": _safe_mean(
                    [r.signed_markout for r in sides_bid]
                ),
                "mean_signed_markout_ask": _safe_mean(
                    [r.signed_markout for r in sides_ask]
                ),
                "mean_toxicity_bid": _safe_mean(
                    [r.toxicity for r in sides_bid]
                ),
                "mean_toxicity_ask": _safe_mean(
                    [r.toxicity for r in sides_ask]
                ),
            }

        return MarkoutResult(
            horizons_seconds=tuple(float(h) for h in horizons_seconds),
            per_fill=per_fill,
            aggregates=aggregates,
        )


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(values))


def _safe_pct(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, pct))
