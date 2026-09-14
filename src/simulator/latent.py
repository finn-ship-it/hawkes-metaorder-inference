"""
latent.py — latent regime channel.

Implements the minimal latent-regime infrastructure required by the two-regime
extension:

    * `RegimeProcess`: a continuous-time Markov chain (CTMC) on the state set
      {0, 1, ..., K - 1} with generator matrix Q of shape (K, K).
    * `baseline_for_regime`: resolve the per-regime baseline intensity vector
      mu(z) from a SimulatorConfig.

The regime channel evolves independently of the observed point process. The
point-process intensity depends on the current regime only through the
baseline term mu(z_t); kernels and history contributions are regime-
independent in this milestone.

Degenerate single-regime case
-----------------------------
When num_regimes == 1 the process is trivial: the regime is always 0 and no
transition ever occurs. `RegimeProcess.sample_transition` returns
(+inf, 0), which the simulator core uses to bypass the regime-jump branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .config import SimulatorConfig, MetaOrderWindow


# ---------------------------------------------------------------------------
# Baseline lookup
# ---------------------------------------------------------------------------


def baseline_for_regime(cfg: SimulatorConfig, regime: int) -> np.ndarray:
    """Return the d-vector mu(regime) selected from cfg.baseline.mu.

    Accepted shapes:
        (d,)         - single regime (K == 1)
        (1, d)       - single regime, row-shaped
        (K, d)       - multi-regime (K >= 1)
    """
    mu = np.asarray(cfg.baseline.mu, dtype=float)
    k = cfg.latent_regime.num_regimes
    d = cfg.dimension

    if mu.ndim == 1:
        if k != 1 or mu.shape[0] != d:
            raise ValueError(
                f"mu shape {mu.shape} inconsistent with (d={d}, K={k})"
            )
        return mu.copy()

    if mu.ndim == 2:
        if mu.shape != (k, d):
            raise ValueError(
                f"mu shape {mu.shape} inconsistent with (K={k}, d={d})"
            )
        if not (0 <= regime < k):
            raise IndexError(
                f"regime {regime} out of range [0, {k})"
            )
        return mu[regime].copy()

    raise ValueError(f"mu must be 1-d or 2-d, got ndim={mu.ndim}")


# ---------------------------------------------------------------------------
# Regime process (CTMC)
# ---------------------------------------------------------------------------


class RegimeProcess:
    """Continuous-time Markov chain on {0, ..., K - 1} with generator Q.

    For K == 1 the process is degenerate: sample_transition returns
    (+inf, 0) regardless of the current regime.
    """

    def __init__(self, cfg: SimulatorConfig) -> None:
        self._k = int(cfg.latent_regime.num_regimes)
        if self._k == 1:
            self._q = None
            self._exit_rates = None
            self._stoch_kernel = None
        else:
            q = np.asarray(cfg.latent_regime.transition_rates, dtype=float)
            if q.shape != (self._k, self._k):
                raise ValueError(
                    f"transition_rates shape {q.shape} inconsistent with "
                    f"num_regimes {self._k}"
                )
            self._q = q
            # Exit rate from regime z equals -Q[z, z] (which is the sum of
            # off-diagonal entries in row z).
            self._exit_rates = -np.diag(q)

            # Stochastic kernel p(z' | z) = Q[z, z'] / exit_rate[z] for z' != z.
            # Rows are set to delta_z when exit_rate[z] == 0 (absorbing state).
            p = np.zeros_like(q)
            for z in range(self._k):
                rate = self._exit_rates[z]
                if rate <= 0.0:
                    p[z, z] = 1.0
                else:
                    for zp in range(self._k):
                        if zp == z:
                            continue
                        p[z, zp] = q[z, zp] / rate
            self._stoch_kernel = p

    @property
    def num_regimes(self) -> int:
        return self._k

    def sample_transition(
        self, current_regime: int, rng: np.random.Generator
    ) -> Tuple[float, int]:
        """Return (wait_time, next_regime) for one CTMC transition.

        For K == 1 or for absorbing states (exit_rate == 0) the returned
        wait_time is +inf and next_regime equals current_regime.
        """
        if self._k == 1:
            return float("inf"), 0

        z = int(current_regime)
        if not (0 <= z < self._k):
            raise IndexError(
                f"current_regime {z} out of range [0, {self._k})"
            )

        rate = float(self._exit_rates[z])
        if rate <= 0.0:
            return float("inf"), z

        wait = float(rng.exponential(1.0 / rate))
        probs = self._stoch_kernel[z]
        u = float(rng.uniform(0.0, 1.0))
        cumulative = 0.0
        next_regime = z
        for zp in range(self._k):
            cumulative += float(probs[zp])
            if u <= cumulative:
                next_regime = zp
                break
        return wait, next_regime


# ---------------------------------------------------------------------------
# Meta-order schedule
# ---------------------------------------------------------------------------


@dataclass
class _BoundaryEvent:
    """Internal record of a single meta-order window boundary."""

    time: float
    kind: str  # either "start" or "end"
    window_index: int


class MetaOrderSchedule:
    """Pre-computed schedule of meta-order window boundaries.

    Precomputes a sorted list of (time, kind, window_index) boundary records
    from the meta-order specification and maintains a pointer to the next
    unprocessed boundary. Also exposes a per-event-type multiplicative
    baseline factor vector of shape (d,) that is updated as boundaries are
    consumed by the simulator core.

    Because the configuration validator forbids overlapping windows, each
    event type is inside at most one active window at any instant and the
    factor vector can be computed by direct assignment rather than by
    multiplicative accumulation.
    """

    def __init__(self, cfg: SimulatorConfig) -> None:
        self._d = int(cfg.dimension)
        self._windows: Tuple[MetaOrderWindow, ...] = tuple(cfg.meta_order.windows)
        # Build the sorted boundary schedule.
        schedule: List[_BoundaryEvent] = []
        for idx, w in enumerate(self._windows):
            schedule.append(_BoundaryEvent(time=float(w.start_time), kind="start", window_index=idx))
            schedule.append(_BoundaryEvent(time=float(w.end_time), kind="end", window_index=idx))
        # Sort primarily by time; "end" before "start" at equal times so that
        # a window that closes at t=T0 does not conflict with one that opens
        # at t=T0 (non-overlap is already enforced by validate_meta_order, so
        # this branch only handles the degenerate equality case).
        schedule.sort(key=lambda b: (b.time, 0 if b.kind == "end" else 1))
        self._schedule: Tuple[_BoundaryEvent, ...] = tuple(schedule)
        self._pointer: int = 0
        self._factors: np.ndarray = np.ones(self._d, dtype=float)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def windows(self) -> Tuple[MetaOrderWindow, ...]:
        return self._windows

    @property
    def factors(self) -> np.ndarray:
        """Current per-event-type multiplicative baseline factor, shape (d,)."""
        return self._factors

    @property
    def has_pending(self) -> bool:
        return self._pointer < len(self._schedule)

    def next_boundary_time(self) -> float:
        """Absolute time of the next unprocessed boundary; +inf if none remain."""
        if not self.has_pending:
            return float("inf")
        return self._schedule[self._pointer].time

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the schedule pointer and factor vector to their initial state."""
        self._pointer = 0
        self._factors = np.ones(self._d, dtype=float)

    def advance(self) -> Optional[_BoundaryEvent]:
        """Consume the next boundary: update the factor vector and return the
        consumed boundary record. Returns None if no boundaries remain.
        """
        if not self.has_pending:
            return None
        b = self._schedule[self._pointer]
        w = self._windows[b.window_index]
        factor = 1.0 + float(w.alpha)
        for i in w.target_event_types:
            if b.kind == "start":
                self._factors[int(i)] = factor
            else:
                self._factors[int(i)] = 1.0
        self._pointer += 1
        return b
