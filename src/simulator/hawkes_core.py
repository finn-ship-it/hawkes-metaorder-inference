"""
hawkes_core.py — Ogata thinning simulator for a multivariate Hawkes process.

Implements the simulator core required by the first coding milestone. The
model sequence consists of timestamped events over the four FX event types
{BID_UP, BID_DOWN, ASK_UP, ASK_DOWN}; the kernel family is exponential; the
latent regime is degenerate (one regime); no meta-order window is applied.

Structural reference: Konark's
`HawkesRLTrading/src/Stochastic_Processes/Arrival_Models.py`,
class `HawkesArrival`, method `thinningOgataIS2`.

Mathematical reference: Konark's
`src/backup/hawkes/simulate_optimized.py`, function `thinningOgataIS2`.

Only the decomposition (proposal, acceptance, event-type sampling, state
update, truncation window) is borrowed; all equity-specific branches
(coltolevel map, in-spread scaling, queue/size generation, exchange hooks)
are omitted.

Algorithm summary
-----------------
Let lambda_i(t) = mu_i + sum_{j} sum_{t_k < t, type_k = j} phi_ij(t - t_k),
bar_lambda(t) = sum_i lambda_i(t).

At each step:
    1. Compute M from the right-limit intensity, including any event just
       accepted at t. For the supported non-negative, non-increasing kernels,
       this bounds future intensity until the next exogenous boundary.
    2. Draw candidate increment dt ~ Exp(M).
    3. If a regime or pressure boundary occurs first, apply it and re-propose.
       Otherwise, if t + dt > horizon, terminate.
    4. Draw U ~ Uniform(0, 1). Accept if U * M <= bar_lambda(t + dt).
    5. If accepted, sample event type i with probability lambda_i / bar_lambda.
    6. Update t <- t + dt and record the event; intensities are recomputed
       from the event history on demand via the truncation window.

Proposal, acceptance and event-type draws are successive draws from the
seeded generator. A violated proposal bound raises an error rather than
silently accepting from an invalid envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import numpy as np

from .config import SimulatorConfig, validate
from .kernels import Kernel, ExponentialKernel, build_kernel
from .latent import RegimeProcess, baseline_for_regime, MetaOrderSchedule


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A single recorded event."""

    time: float
    event_type: int


@dataclass
class RunTrace:
    """In-memory representation of a completed simulation."""

    times: np.ndarray  # shape (N,), float64, strictly increasing
    event_types: np.ndarray  # shape (N,), int64, values in {0, ..., d-1}
    final_intensity: np.ndarray  # shape (d,), float64
    num_proposals: int
    num_acceptances: int
    wall_clock_seconds: float
    dimension: int
    horizon: float
    seed: int
    num_regimes: int = 1
    # Regime trajectory as a piecewise-constant step function.
    # regime_times[0] == 0.0 and regime_values[0] is the initial regime.
    # Subsequent entries record each CTMC jump. Both arrays have equal length.
    regime_times: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64)
    )
    regime_values: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int64)
    )
    # Meta-order windows as recorded plain-Python tuples of
    # (start_time, end_time, alpha, target_event_types). An empty tuple means
    # no meta-order windows were configured for this run.
    meta_order_windows: Tuple[Tuple[float, float, float, Tuple[int, ...]], ...] = ()

    @property
    def num_events(self) -> int:
        return int(self.times.shape[0])

    @property
    def acceptance_ratio(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_acceptances / self.num_proposals

    def events(self) -> List[Event]:
        return [
            Event(time=float(t), event_type=int(k))
            for t, k in zip(self.times, self.event_types)
        ]


# ---------------------------------------------------------------------------
# Simulator state
# ---------------------------------------------------------------------------


@dataclass
class SimulatorState:
    """Mutable state held by the simulator between step() calls."""

    current_time: float
    intensity: np.ndarray  # shape (d,), the instantaneous intensity vector
    history_times: List[float] = field(default_factory=list)
    history_types: List[int] = field(default_factory=list)
    rng: Optional[np.random.Generator] = None
    regime: int = 0  # degenerate in the first coding milestone
    meta_order_active: bool = False  # degenerate in the first coding milestone


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class HawkesSimulator:
    """Ogata thinning simulator for a multivariate Hawkes process."""

    def __init__(self, cfg: SimulatorConfig) -> None:
        validate(cfg)
        self._cfg = cfg
        self._kernel: Kernel = build_kernel(cfg)
        self._d = int(cfg.dimension)
        self._k = int(cfg.latent_regime.num_regimes)
        # Per-regime baseline cache: shape (K, d)
        self._mu_by_regime = np.stack(
            [baseline_for_regime(cfg, z) for z in range(self._k)], axis=0
        )
        self._initial_regime = int(cfg.latent_regime.initial_regime)
        self._regime_process = RegimeProcess(cfg)
        self._meta_order_schedule = MetaOrderSchedule(cfg)
        self._tau = float(cfg.numerics.truncation_window)
        self._safety = float(cfg.numerics.upper_bound_safety)
        self._state: Optional[SimulatorState] = None

    # ------------------------------------------------------------------
    # Public control surface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the simulator to t = 0 with empty history and a fresh RNG."""
        if seed is None:
            seed = self._cfg.seed
        rng = np.random.default_rng(int(seed))
        z0 = self._initial_regime
        self._state = SimulatorState(
            current_time=0.0,
            intensity=self._mu_by_regime[z0].copy(),
            history_times=[],
            history_types=[],
            rng=rng,
            regime=z0,
            meta_order_active=False,
        )
        self._regime_times: List[float] = [0.0]
        self._regime_values: List[int] = [z0]
        # Precomputed time of next regime jump, stored absolutely.
        self._next_regime_jump_time, self._pending_next_regime = self._sample_next_regime_jump()
        # Reset the meta-order schedule to its initial state and cache the
        # time of the next pending window boundary.
        self._meta_order_schedule.reset()
        self._next_boundary_time: float = self._meta_order_schedule.next_boundary_time()

    def simulate(self, horizon: Optional[float] = None) -> RunTrace:
        """Run the simulator until the configured horizon and return a RunTrace.

        If horizon is None the configured horizon is used.
        """
        if self._state is None:
            self.reset(seed=self._cfg.seed)

        assert self._state is not None
        T = float(self._cfg.horizon if horizon is None else horizon)
        if T <= 0.0:
            raise ValueError(f"horizon must be strictly positive, got {T}")

        num_proposals = 0
        num_acceptances = 0
        wall_start = time.perf_counter()

        while True:
            # Stream 1: next regime jump (absolute time).
            t_regime = self._next_regime_jump_time
            # Stream 2: next meta-order window boundary (absolute time).
            t_boundary = self._next_boundary_time

            # Stream 3: next event proposal.
            candidate = self._propose_candidate()
            if candidate is None:
                # No further events possible; but a regime jump or a
                # meta-order window boundary may still fire inside [t, T].
                t_next_exog = min(t_regime, t_boundary)
                if t_next_exog <= T and t_next_exog < float("inf"):
                    if t_regime <= t_boundary:
                        self._apply_regime_jump(t_regime)
                    else:
                        self._apply_window_boundary(t_boundary)
                    continue
                break
            dt, upper_bound = candidate
            t_candidate = self._state.current_time + dt

            # Whichever stream fires first wins this iteration.
            t_exog = min(t_regime, t_boundary)
            if t_exog <= t_candidate:
                if t_exog > T:
                    self._state.current_time = T
                    break
                if t_regime <= t_boundary:
                    self._apply_regime_jump(t_regime)
                else:
                    self._apply_window_boundary(t_boundary)
                continue

            num_proposals += 1
            if t_candidate > T:
                self._state.current_time = T
                break

            accepted_type = self._accept_or_reject(t_candidate, upper_bound)
            if accepted_type is not None:
                num_acceptances += 1
                self._update_history(t_candidate, accepted_type)

            self._state.current_time = t_candidate

        wall_elapsed = time.perf_counter() - wall_start
        times = np.asarray(self._state.history_times, dtype=float)
        types = np.asarray(self._state.history_types, dtype=np.int64)
        final_intensity = self._current_intensity(self._state.current_time)

        meta_windows_tuple: Tuple[
            Tuple[float, float, float, Tuple[int, ...]], ...
        ] = tuple(
            (
                float(w.start_time),
                float(w.end_time),
                float(w.alpha),
                tuple(int(i) for i in w.target_event_types),
            )
            for w in self._cfg.meta_order.windows
        )

        return RunTrace(
            times=times,
            event_types=types,
            final_intensity=final_intensity,
            num_proposals=num_proposals,
            num_acceptances=num_acceptances,
            wall_clock_seconds=float(wall_elapsed),
            dimension=self._d,
            horizon=T,
            seed=int(self._cfg.seed),
            num_regimes=self._k,
            regime_times=np.asarray(self._regime_times, dtype=np.float64),
            regime_values=np.asarray(self._regime_values, dtype=np.int64),
            meta_order_windows=meta_windows_tuple,
        )

    def step(self) -> Optional[Event]:
        """Advance the simulator by one thinning iteration.

        Returns the accepted event if one fired at this step, otherwise None.
        A regime jump occurring before the next event candidate is applied
        silently (it is recorded in the regime trajectory but yields None).
        Raises StopIteration if the current_time has exceeded the horizon.
        """
        if self._state is None:
            self.reset(seed=self._cfg.seed)
        assert self._state is not None

        T = self._cfg.horizon
        if self._state.current_time >= T:
            raise StopIteration

        t_regime = self._next_regime_jump_time
        t_boundary = self._next_boundary_time
        candidate = self._propose_candidate()
        if candidate is None:
            t_next_exog = min(t_regime, t_boundary)
            if t_next_exog <= T and t_next_exog < float("inf"):
                if t_regime <= t_boundary:
                    self._apply_regime_jump(t_regime)
                else:
                    self._apply_window_boundary(t_boundary)
                return None
            self._state.current_time = T
            return None
        dt, upper_bound = candidate
        t_candidate = self._state.current_time + dt

        t_exog = min(t_regime, t_boundary)
        if t_exog <= t_candidate:
            if t_exog > T:
                self._state.current_time = T
                return None
            if t_regime <= t_boundary:
                self._apply_regime_jump(t_regime)
            else:
                self._apply_window_boundary(t_boundary)
            return None

        if t_candidate > T:
            self._state.current_time = T
            return None

        accepted_type = self._accept_or_reject(t_candidate, upper_bound)
        self._state.current_time = t_candidate
        if accepted_type is None:
            return None
        self._update_history(t_candidate, accepted_type)
        return Event(time=t_candidate, event_type=accepted_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_intensity(
        self, t: float, *, include_current: bool = False
    ) -> np.ndarray:
        """Evaluate lambda_i(t) = mu_i(z_t) * f_i(t) + sum_k phi_ij(t - t_k).

        The baseline is selected by the current regime and multiplied element-
        wise by the meta-order factor vector f(t), whose entries equal
        (1 + alpha) for event types covered by an active window and 1.0
        otherwise. By default, only events within (t - TAU, t) contribute:
        this is the predictable intensity used to accept a candidate.
        With include_current=True, accepted events at t also contribute,
        giving the right limit required to bound the next proposal. Events
        at or before t - TAU are truncated in both cases.
        """
        assert self._state is not None
        lam = np.asarray(
            self._mu_by_regime[self._state.regime]
            * self._meta_order_schedule.factors,
            dtype=float,
        ).copy()
        if not self._state.history_times:
            return lam

        cutoff = t - self._tau
        # Iterate backwards because events are appended in time order.
        for k in range(len(self._state.history_times) - 1, -1, -1):
            t_k = self._state.history_times[k]
            if t_k <= cutoff:
                break
            if t_k > t or (t_k == t and not include_current):
                continue
            j = self._state.history_types[k]
            # phi[:, j] evaluated at (t - t_k)
            lam = lam + self._kernel.evaluate(t - t_k)[:, j]
        return lam

    def _sample_next_regime_jump(self) -> Tuple[float, int]:
        """Sample the absolute time and target regime of the next CTMC jump."""
        assert self._state is not None
        wait, next_regime = self._regime_process.sample_transition(
            current_regime=self._state.regime, rng=self._state.rng
        )
        if wait == float("inf"):
            return float("inf"), int(next_regime)
        return float(self._state.current_time + wait), int(next_regime)

    def _apply_regime_jump(self, t_jump: float) -> None:
        """Advance simulator time to t_jump, switch regime, record the jump,
        and sample the next regime jump time."""
        assert self._state is not None
        self._state.current_time = t_jump
        self._state.regime = self._pending_next_regime
        self._regime_times.append(float(t_jump))
        self._regime_values.append(int(self._state.regime))
        (
            self._next_regime_jump_time,
            self._pending_next_regime,
        ) = self._sample_next_regime_jump()

    def _apply_window_boundary(self, t_boundary: float) -> None:
        """Advance simulator time to t_boundary and consume the next meta-order
        window boundary from the schedule. Updates the factor vector and
        refreshes the cached next-boundary time. Also synchronises the
        meta_order_active flag on the simulator state (True iff any event
        type currently has a non-unit factor)."""
        assert self._state is not None
        self._state.current_time = t_boundary
        self._meta_order_schedule.advance()
        self._next_boundary_time = self._meta_order_schedule.next_boundary_time()
        self._state.meta_order_active = bool(
            np.any(self._meta_order_schedule.factors != 1.0)
        )

    def _propose_candidate(self) -> Optional[Tuple[float, float]]:
        """Generate a candidate inter-arrival dt and its upper bound M.

        Returns (dt, M), where dt ~ Exp(M). M is the safety-scaled right-limit
        total intensity. The supported kernels are non-negative and
        non-increasing, so M bounds intensity until the next regime or
        pressure boundary; simulate()/step() intercept that boundary and
        recompute the proposal. Returns None if the current total is zero.
        """
        assert self._state is not None
        lam_now = self._current_intensity(
            self._state.current_time, include_current=True
        )
        upper_bound = float(np.sum(lam_now)) * self._safety
        if not np.isfinite(upper_bound):
            raise RuntimeError("Non-finite Hawkes proposal bound")
        if upper_bound <= 0.0:
            return None
        dt = float(self._state.rng.exponential(1.0 / upper_bound))
        return dt, upper_bound

    def _accept_or_reject(
        self, t_candidate: float, upper_bound: float
    ) -> Optional[int]:
        """Thin the candidate and, if accepted, sample an event type."""
        assert self._state is not None
        lam_candidate = self._current_intensity(t_candidate)
        total = float(np.sum(lam_candidate))
        if not np.isfinite(upper_bound) or upper_bound <= 0.0:
            raise RuntimeError("Hawkes proposal bound must be finite and positive")
        if not np.isfinite(total) or total < 0.0:
            raise RuntimeError("Hawkes candidate intensity must be finite and non-negative")
        # Allow only round-off at equality; never silently clip an invalid
        # acceptance ratio. A violation indicates a proposal/boundary bug.
        if total > upper_bound * (1.0 + 1e-12):
            raise RuntimeError(
                "Hawkes thinning bound violated: "
                f"candidate intensity {total:.17g} exceeds bound {upper_bound:.17g} "
                f"at t={t_candidate:.17g}"
            )
        u = float(self._state.rng.uniform(0.0, 1.0))
        if u * upper_bound > total:
            return None
        # Sample event type proportional to lam_candidate.
        if total <= 0.0:
            return None
        probabilities = lam_candidate / total
        v = float(self._state.rng.uniform(0.0, 1.0))
        cumulative = 0.0
        for i in range(self._d):
            cumulative += float(probabilities[i])
            if v <= cumulative:
                return i
        return self._d - 1  # numerical safeguard

    def _update_history(self, t: float, event_type: int) -> None:
        assert self._state is not None
        self._state.history_times.append(float(t))
        self._state.history_types.append(int(event_type))
        self._state.intensity = self._current_intensity(t, include_current=True)
