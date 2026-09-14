"""
observation.py — deterministic observation operator.

Implements the minimal projection from the latent event stream produced by
HawkesSimulator to an observable FX-style event stream. The operator is
deliberately side-effect-free and depends only on a RunTrace and a static
ObservationConfig; it performs no sampling, no noise injection, no latency
modelling, and no micro-structural filtering beyond a minimum-spread
invariant.

Semantics
---------
Each latent event is classified by its component index k in {0, 1, 2, 3}
corresponding to {BID_UP, BID_DOWN, ASK_UP, ASK_DOWN}. The operator maintains
a pair of integer tick counters (bid_ticks, ask_ticks) initialised from the
configuration and evaluates the proposed post-event quote state for each
latent event:

    k = BID_UP   -> new_bid = bid + 1
    k = BID_DOWN -> new_bid = bid - 1
    k = ASK_UP   -> new_ask = ask + 1
    k = ASK_DOWN -> new_ask = ask - 1

The proposal is admitted iff

    new_ask - new_bid >= min_spread_ticks

When the proposal is admitted the operator updates the counters and appends a
single record to the observed stream containing the event time, the event
type, the post-event (bid_ticks, ask_ticks), and the resulting spread.

When the proposal would violate the minimum-spread invariant the operator
applies the configured crossing_policy. The only policy implemented in this
milestone is "censor": the offending event is dropped, the counters are left
unchanged, and the event is counted toward num_censored_events. The latent
stream is not modified and the simulator core is untouched.

The censor policy yields an observed stream in which every record satisfies
the minimum-spread invariant and therefore num_crossed_observations is
identically zero. The diagnostic is retained to guard against regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from .config import EventType
from .hawkes_core import RunTrace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Crossing policies recognised by ObservationOperator. Additional policies
#: (for example, clipping or book rebuilding) would be added here.
ALLOWED_CROSSING_POLICIES: Tuple[str, ...] = ("censor",)


@dataclass(frozen=True)
class ObservationConfig:
    """Static configuration for the observation operator.

    Fields
    ------
    initial_bid_ticks : int
        Bid quote at t = 0 in integer tick units. Reference frame only;
        downstream consumers may add a constant offset without affecting
        estimator input.
    initial_ask_ticks : int
        Ask quote at t = 0 in integer tick units. Required to satisfy
        initial_ask_ticks - initial_bid_ticks >= min_spread_ticks.
    min_spread_ticks : int
        Minimum admissible spread in integer tick units. Required to be
        >= 1. Every observed record r satisfies r.ask_ticks - r.bid_ticks
        >= min_spread_ticks by construction under the "censor" policy.
    crossing_policy : str
        Policy applied when a latent event would violate the minimum-spread
        invariant. Must be one of ALLOWED_CROSSING_POLICIES. The default
        "censor" discards the offending event without updating the quote
        counters.
    tick_size : float
        Currency-unit size of one tick. Recorded for downstream use; the
        operator itself does not multiply ticks by this factor.
    """

    initial_bid_ticks: int = 0
    initial_ask_ticks: int = 1
    min_spread_ticks: int = 1
    crossing_policy: str = "censor"
    tick_size: float = 1e-5

    def __post_init__(self) -> None:
        if int(self.min_spread_ticks) < 1:
            raise ValueError(
                f"min_spread_ticks must be >= 1, got {self.min_spread_ticks}"
            )
        if self.crossing_policy not in ALLOWED_CROSSING_POLICIES:
            raise ValueError(
                f"crossing_policy must be one of "
                f"{ALLOWED_CROSSING_POLICIES}, got {self.crossing_policy!r}"
            )
        if (
            int(self.initial_ask_ticks) - int(self.initial_bid_ticks)
            < int(self.min_spread_ticks)
        ):
            raise ValueError(
                f"initial spread "
                f"({int(self.initial_ask_ticks) - int(self.initial_bid_ticks)}) "
                f"must be >= min_spread_ticks "
                f"({int(self.min_spread_ticks)})"
            )
        if float(self.tick_size) <= 0.0:
            raise ValueError(
                f"tick_size must be strictly positive, got {self.tick_size}"
            )


def default_observation_config() -> ObservationConfig:
    """Return the default observation configuration.

    initial_bid_ticks = 0, initial_ask_ticks = 1, min_spread_ticks = 1,
    crossing_policy = "censor", tick_size = 1e-5. These values define a
    one-tick opening spread at an arbitrary reference price with the sub-pip
    FX tick size used by most major-pair venues.
    """
    return ObservationConfig()


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------


@dataclass
class ObservedTrace:
    """Result of projecting a RunTrace through the observation operator.

    Only latent events that pass the minimum-spread test appear in the
    arrays; the counts num_latent_events and num_censored_events preserve
    the bookkeeping required to relate observable and latent streams.

    Fields
    ------
    times : np.ndarray, shape (num_observed_events,), float64
        Times of admitted latent events, strictly increasing.
    event_types : np.ndarray, shape (num_observed_events,), int64
        Event-type component indices of admitted events.
    bid_ticks : np.ndarray, shape (num_observed_events,), int64
        Integer bid quote after each admitted event.
    ask_ticks : np.ndarray, shape (num_observed_events,), int64
        Integer ask quote after each admitted event.
    spread_ticks : np.ndarray, shape (num_observed_events,), int64
        Post-event spread (ask_ticks - bid_ticks), guaranteed to be
        >= min_spread_ticks under the "censor" policy.
    initial_bid_ticks : int
    initial_ask_ticks : int
    min_spread_ticks : int
    crossing_policy : str
    tick_size : float
        Configuration values under which the projection was produced.
    num_latent_events : int
        Number of events in the source RunTrace.
    num_censored_events : int
        Number of latent events dropped by the crossing policy.
    """

    times: np.ndarray
    event_types: np.ndarray
    bid_ticks: np.ndarray
    ask_ticks: np.ndarray
    spread_ticks: np.ndarray
    initial_bid_ticks: int
    initial_ask_ticks: int
    min_spread_ticks: int
    crossing_policy: str
    tick_size: float
    num_latent_events: int
    num_censored_events: int

    @property
    def num_observed_events(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_observations(self) -> int:
        """Alias retained for backward compatibility."""
        return self.num_observed_events

    @property
    def censoring_fraction(self) -> float:
        if self.num_latent_events == 0:
            return 0.0
        return float(self.num_censored_events) / float(self.num_latent_events)

    @property
    def final_bid_ticks(self) -> int:
        if self.num_observed_events == 0:
            return int(self.initial_bid_ticks)
        return int(self.bid_ticks[-1])

    @property
    def final_ask_ticks(self) -> int:
        if self.num_observed_events == 0:
            return int(self.initial_ask_ticks)
        return int(self.ask_ticks[-1])

    @property
    def num_crossed_observations(self) -> int:
        """Number of records for which ask_ticks - bid_ticks < min_spread_ticks.

        Under the "censor" policy this count is identically zero by
        construction; the property is retained as a regression diagnostic.
        """
        if self.num_observed_events == 0:
            return 0
        return int(
            np.sum((self.ask_ticks - self.bid_ticks) < int(self.min_spread_ticks))
        )


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class ObservationOperator:
    """Deterministic projection from a latent RunTrace to an ObservedTrace.

    The operator has no hidden state: each call to project() is a pure
    function of the RunTrace and the stored ObservationConfig. Two calls with
    identical inputs yield arrays that are bit-for-bit equal.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg = cfg
        # Canonical mapping check: the class assumes EventType values are
        # exactly {BID_UP=0, BID_DOWN=1, ASK_UP=2, ASK_DOWN=3}. Guard against
        # silent drift in the enum.
        expected = {
            EventType.BID_UP: 0,
            EventType.BID_DOWN: 1,
            EventType.ASK_UP: 2,
            EventType.ASK_DOWN: 3,
        }
        for k, v in expected.items():
            if int(k) != v:
                raise RuntimeError(
                    f"ObservationOperator assumes {k.name} == {v}, got "
                    f"{int(k)}"
                )

    @property
    def config(self) -> ObservationConfig:
        return self._cfg

    def _empty_observed(self, n_latent: int) -> ObservedTrace:
        return ObservedTrace(
            times=np.zeros(0, dtype=np.float64),
            event_types=np.zeros(0, dtype=np.int64),
            bid_ticks=np.zeros(0, dtype=np.int64),
            ask_ticks=np.zeros(0, dtype=np.int64),
            spread_ticks=np.zeros(0, dtype=np.int64),
            initial_bid_ticks=int(self._cfg.initial_bid_ticks),
            initial_ask_ticks=int(self._cfg.initial_ask_ticks),
            min_spread_ticks=int(self._cfg.min_spread_ticks),
            crossing_policy=str(self._cfg.crossing_policy),
            tick_size=float(self._cfg.tick_size),
            num_latent_events=int(n_latent),
            num_censored_events=0,
        )

    def project(self, trace: RunTrace) -> ObservedTrace:
        """Project trace to an ObservedTrace under the configured policy.

        Raises ValueError if an event-type index falls outside {0, 1, 2, 3}
        or if times and event_types have mismatched shapes.
        """
        types = np.asarray(trace.event_types, dtype=np.int64)
        times = np.asarray(trace.times, dtype=np.float64)
        if types.shape != times.shape:
            raise ValueError(
                f"trace.times shape {times.shape} inconsistent with "
                f"trace.event_types shape {types.shape}"
            )

        n_latent = int(types.shape[0])
        if n_latent == 0:
            return self._empty_observed(n_latent)

        if int(types.min()) < 0 or int(types.max()) > 3:
            raise ValueError(
                f"event_types must lie in [0, 3], got range "
                f"[{int(types.min())}, {int(types.max())}]"
            )

        min_spread = int(self._cfg.min_spread_ticks)
        bid = int(self._cfg.initial_bid_ticks)
        ask = int(self._cfg.initial_ask_ticks)

        # Pre-allocate output buffers at the latent-stream length; trim at the
        # end once the admitted count is known. Per-event logic is required
        # because admission depends on running state.
        out_times = np.empty(n_latent, dtype=np.float64)
        out_types = np.empty(n_latent, dtype=np.int64)
        out_bid = np.empty(n_latent, dtype=np.int64)
        out_ask = np.empty(n_latent, dtype=np.int64)
        out_spread = np.empty(n_latent, dtype=np.int64)

        n_observed = 0
        n_censored = 0
        for i in range(n_latent):
            k = int(types[i])
            new_bid = bid
            new_ask = ask
            if k == 0:
                new_bid = bid + 1
            elif k == 1:
                new_bid = bid - 1
            elif k == 2:
                new_ask = ask + 1
            else:
                new_ask = ask - 1

            if (new_ask - new_bid) >= min_spread:
                bid = new_bid
                ask = new_ask
                out_times[n_observed] = float(times[i])
                out_types[n_observed] = k
                out_bid[n_observed] = bid
                out_ask[n_observed] = ask
                out_spread[n_observed] = ask - bid
                n_observed += 1
            else:
                # Crossing policy. Only "censor" is implemented; other values
                # are rejected at configuration time.
                n_censored += 1

        return ObservedTrace(
            times=out_times[:n_observed].copy(),
            event_types=out_types[:n_observed].copy(),
            bid_ticks=out_bid[:n_observed].copy(),
            ask_ticks=out_ask[:n_observed].copy(),
            spread_ticks=out_spread[:n_observed].copy(),
            initial_bid_ticks=int(self._cfg.initial_bid_ticks),
            initial_ask_ticks=int(self._cfg.initial_ask_ticks),
            min_spread_ticks=int(self._cfg.min_spread_ticks),
            crossing_policy=str(self._cfg.crossing_policy),
            tick_size=float(self._cfg.tick_size),
            num_latent_events=int(n_latent),
            num_censored_events=int(n_censored),
        )
