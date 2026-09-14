"""
recovery.py — Layer-2 threshold detector (comparison baseline).

This module implements a transparent threshold detector for regime labels
and meta-order activity intervals on the observed event stream. The
detector uses sliding-window features and a rate threshold with optional
hysteresis.

**HMM is now the primary Layer-2 recovery method (see `recovery_hmm.py`
and `experiment.ExperimentConfig.recovery_method`, which defaults to
"hmm"). The threshold detector documented here is retained as the
comparison baseline.** Headline ensemble result on the 50-seed
`meta_order_smoke` benchmark: HMM mean F1 = 0.863 (std 0.090) vs
threshold-detector F1 = 0.727 (std 0.084).

Public surface
--------------
- WindowFeatures dataclass
- compute_window_features(observed, *, window_seconds, hop_seconds)
- RegimeBaselineConfig dataclass
- recover_regimes_baseline(observed, cfg)
- MetaOrderBaselineConfig dataclass
- score_meta_order_activity(observed, cfg)
- true_regime_at(times, regime_times, regime_values)

The threshold detector operates on `simulator.observation.ObservedTrace`
instances and on the regime trajectory arrays already persisted in
`metadata.json` and `events.npz`. It does not depend on `hawkes_core.py`,
`latent.py`, `observation.py`, or any other simulator-core module beyond
the ObservedTrace type for type-hinting purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .observation import ObservedTrace


# ---------------------------------------------------------------------------
# Window features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowFeatures:
    """Per-window summary of the observed event stream.

    Fields
    ------
    window_start, window_end : float
        Window edges in seconds.
    num_events : int
        Total observed events in [window_start, window_end).
    directional_imbalance : float
        (n_up - n_down) / max(num_events, 1) where n_up = count(BID_UP) +
        count(ASK_UP) and n_down = count(BID_DOWN) + count(ASK_DOWN).
        Interpreted as the net upward pressure on the mid quote during the
        window. Range [-1, +1].
    spread_change : float
        (spread at window end - spread at window start) in ticks. Computed
        from the post-event spread of the last event preceding window_end
        and the post-event spread of the last event preceding window_start.
        Zero when no events bracket the window.
    recent_event_rate : float
        num_events / (window_end - window_start), in events per second.
    """
    window_start: float
    window_end: float
    num_events: int
    directional_imbalance: float
    spread_change: float
    recent_event_rate: float


def compute_window_features(
    observed: ObservedTrace,
    *,
    window_seconds: float,
    hop_seconds: Optional[float] = None,
    horizon: Optional[float] = None,
) -> List[WindowFeatures]:
    """Compute sliding-window features over an observed trace.

    Parameters
    ----------
    observed : ObservedTrace
        Output of ObservationOperator.project. Times are in seconds.
    window_seconds : float
        Window length, > 0.
    hop_seconds : float or None
        Window hop, > 0. Defaults to window_seconds (non-overlapping).
    horizon : float or None
        Maximum time to sweep windows over. Defaults to the maximum observed
        time (or window_seconds if the trace is empty).

    Returns
    -------
    list[WindowFeatures]
        One WindowFeatures per window, in window-start order. Includes
        windows that contain zero observed events; these report zero rates,
        zero imbalance, and zero spread change.
    """
    if window_seconds <= 0.0:
        raise ValueError(
            f"window_seconds must be positive, got {window_seconds}"
        )
    if hop_seconds is None:
        hop_seconds = window_seconds
    if hop_seconds <= 0.0:
        raise ValueError(
            f"hop_seconds must be positive, got {hop_seconds}"
        )

    times = np.asarray(observed.times, dtype=float)
    types = np.asarray(observed.event_types, dtype=int)
    spreads = np.asarray(observed.spread_ticks, dtype=float)

    if horizon is None:
        horizon = float(times[-1]) if times.size else window_seconds

    starts = np.arange(0.0, max(0.0, horizon - 1e-12), hop_seconds, dtype=float)
    if starts.size == 0:
        starts = np.asarray([0.0])

    features: List[WindowFeatures] = []
    initial_spread = float(observed.initial_ask_ticks - observed.initial_bid_ticks)

    for start in starts:
        end = start + window_seconds
        if times.size == 0:
            features.append(
                WindowFeatures(
                    window_start=float(start),
                    window_end=float(end),
                    num_events=0,
                    directional_imbalance=0.0,
                    spread_change=0.0,
                    recent_event_rate=0.0,
                )
            )
            continue

        in_window = (times >= start) & (times < end)
        idx_in = np.where(in_window)[0]
        n = int(idx_in.size)

        if n == 0:
            n_up = 0
            n_down = 0
        else:
            wt = types[idx_in]
            # 0 = BID_UP, 1 = BID_DOWN, 2 = ASK_UP, 3 = ASK_DOWN
            n_up = int(np.sum((wt == 0) | (wt == 2)))
            n_down = int(np.sum((wt == 1) | (wt == 3)))

        denom_imb = max(n, 1)
        imb = (n_up - n_down) / denom_imb

        # Spread at the latest event strictly before each edge.
        before_start = np.where(times < start)[0]
        before_end = np.where(times < end)[0]
        spread_at_start = (
            spreads[before_start[-1]] if before_start.size else initial_spread
        )
        spread_at_end = (
            spreads[before_end[-1]] if before_end.size else initial_spread
        )
        spread_change = float(spread_at_end - spread_at_start)

        rate = float(n) / float(window_seconds)

        features.append(
            WindowFeatures(
                window_start=float(start),
                window_end=float(end),
                num_events=n,
                directional_imbalance=float(imb),
                spread_change=spread_change,
                recent_event_rate=rate,
            )
        )

    return features


# ---------------------------------------------------------------------------
# Regime baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBaselineConfig:
    """Threshold-based regime detector configuration.

    Fields
    ------
    window_seconds : float
        Sliding window length.
    rate_threshold : float
        A window's `recent_event_rate` strictly greater than this value is
        labelled regime 1 ("active"); otherwise regime 0 ("calm"). Tune
        against the configuration's regime-conditional baseline rates.
    hop_seconds : float or None
        Hop between windows. Defaults to window_seconds.
    hysteresis_seconds : float
        Minimum dwell time in a label before another change is permitted.
        Suppresses single-window flips. Defaults to 0.
    """
    window_seconds: float
    rate_threshold: float
    hop_seconds: Optional[float] = None
    hysteresis_seconds: float = 0.0


def recover_regimes_baseline(
    observed: ObservedTrace,
    cfg: RegimeBaselineConfig,
    *,
    horizon: Optional[float] = None,
) -> List[Tuple[float, int]]:
    """Run the threshold baseline on an observed trace.

    Returns
    -------
    list[tuple[float, int]]
        Inferred (start_time, regime_label) pairs, ordered by start_time.
        The first pair always begins at 0.0; subsequent pairs mark label
        changes. Labels are 0 (calm) or 1 (active).
    """
    features = compute_window_features(
        observed,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        horizon=horizon,
    )
    transitions: List[Tuple[float, int]] = []
    last_label = -1
    last_change_time = -np.inf
    for f in features:
        label = 1 if f.recent_event_rate > cfg.rate_threshold else 0
        # Hysteresis: do not switch within hysteresis_seconds of the previous switch.
        if label != last_label and (f.window_start - last_change_time) >= cfg.hysteresis_seconds:
            transitions.append((float(f.window_start), int(label)))
            last_label = label
            last_change_time = f.window_start
    return transitions


# ---------------------------------------------------------------------------
# Meta-order baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaOrderBaselineConfig:
    """Threshold-based meta-order activity detector configuration.

    Fields
    ------
    window_seconds : float
        Sliding window length.
    score_threshold : float
        Threshold on the directional-imbalance-weighted rate. A window scores
        positive when it shows both elevated rate and biased direction.
    hop_seconds : float or None
        Hop between windows.
    min_active_seconds : float
        Minimum duration for an inferred active interval. Shorter detections
        are dropped as noise.
    direction : str
        "up" or "down" or "any". Restricts the detector to one side of the
        market or accepts either.
    """
    window_seconds: float
    score_threshold: float
    hop_seconds: Optional[float] = None
    min_active_seconds: float = 0.0
    direction: str = "any"


def score_meta_order_activity(
    observed: ObservedTrace,
    cfg: MetaOrderBaselineConfig,
    *,
    horizon: Optional[float] = None,
) -> List[Tuple[float, float, float]]:
    """Score windows for meta-order activity and merge contiguous active windows.

    Returns
    -------
    list[tuple[float, float, float]]
        (start, end, mean_score) per inferred active interval. Filtered by
        `min_active_seconds`.
    """
    if cfg.direction not in ("up", "down", "any"):
        raise ValueError(
            f"direction must be 'up', 'down', or 'any', got {cfg.direction!r}"
        )
    features = compute_window_features(
        observed,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        horizon=horizon,
    )
    intervals: List[Tuple[float, float, float]] = []
    current_start: Optional[float] = None
    current_scores: List[float] = []
    current_end: float = 0.0

    for f in features:
        if cfg.direction == "up":
            score = f.recent_event_rate * max(0.0, f.directional_imbalance)
        elif cfg.direction == "down":
            score = f.recent_event_rate * max(0.0, -f.directional_imbalance)
        else:
            score = f.recent_event_rate * abs(f.directional_imbalance)

        if score > cfg.score_threshold:
            if current_start is None:
                current_start = f.window_start
            current_end = f.window_end
            current_scores.append(score)
        else:
            if current_start is not None:
                duration = current_end - current_start
                if duration >= cfg.min_active_seconds:
                    intervals.append(
                        (
                            float(current_start),
                            float(current_end),
                            float(np.mean(current_scores)),
                        )
                    )
                current_start = None
                current_scores = []

    if current_start is not None:
        duration = current_end - current_start
        if duration >= cfg.min_active_seconds:
            intervals.append(
                (
                    float(current_start),
                    float(current_end),
                    float(np.mean(current_scores)),
                )
            )
    return intervals


# ---------------------------------------------------------------------------
# True-regime utility
# ---------------------------------------------------------------------------


def true_regime_at(
    times: Sequence[float],
    regime_times: Sequence[float],
    regime_values: Sequence[int],
) -> np.ndarray:
    """Return the true regime label for each query time.

    The regime trajectory is described by step-function arrays: at
    `regime_times[k]` the regime becomes `regime_values[k]`. The label
    returned for a query time `t` is `regime_values[k]` where `k` is the
    largest index with `regime_times[k] <= t`. If `t` precedes
    `regime_times[0]`, the label `regime_values[0]` is used (interpreted
    as the initial regime).
    """
    rt = np.asarray(regime_times, dtype=float)
    rv = np.asarray(regime_values, dtype=int)
    qt = np.asarray(times, dtype=float)
    if rt.size == 0:
        if rv.size == 1:
            return np.full(qt.shape, int(rv[0]), dtype=int)
        return np.zeros(qt.shape, dtype=int)
    idx = np.searchsorted(rt, qt, side="right") - 1
    idx = np.clip(idx, 0, rv.size - 1)
    return rv[idx]


__all__ = [
    "WindowFeatures",
    "compute_window_features",
    "RegimeBaselineConfig",
    "recover_regimes_baseline",
    "MetaOrderBaselineConfig",
    "score_meta_order_activity",
    "true_regime_at",
]
