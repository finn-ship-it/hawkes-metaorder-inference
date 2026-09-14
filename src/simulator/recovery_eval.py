"""
recovery_eval.py — evaluation of Layer-2 recovery against simulator ground truth.

This module turns the inferred-interval output of `recovery.score_meta_order_activity`
into precision / recall / detection-delay metrics against the true MetaOrderWindow
schedule that the simulator was configured with.

Public surface
--------------
- RecoveryEvaluation dataclass
- evaluate_meta_order_recovery(true_windows, inferred_intervals, horizon, *, time_resolution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .config import MetaOrderWindow


@dataclass(frozen=True)
class RecoveryEvaluation:
    """Aggregate metrics for one (true_windows, inferred_intervals) pair.

    Fields
    ------
    precision : float
        Fraction of inferred-active time that overlaps with any true window.
        Range [0, 1]. Equal to 1.0 when the inference contains no false
        positives (no inferred time outside true windows).
    recall : float
        Fraction of true-window time that is covered by an inferred interval.
        Range [0, 1]. Equal to 1.0 when every second of every true window is
        labelled inferred-active.
    f1 : float
        Harmonic mean of precision and recall. Zero when either is zero.
    mean_detection_delay_seconds : float or None
        For each true window, the time from the window's start to the first
        inferred-active sample within it. Averaged across windows. None when
        no true window is detected at all.
    false_positive_intervals : list[tuple[float, float]]
        Inferred intervals (or sub-intervals) that lie entirely outside any
        true window.
    false_negative_intervals : list[tuple[float, float]]
        Sub-intervals of true windows that are not covered by any inferred
        interval.
    confusion_table : dict[str, int]
        Per-time-resolution-step counts: {true_pos, false_pos, true_neg,
        false_neg}.
    """
    precision: float
    recall: float
    f1: float
    mean_detection_delay_seconds: float
    false_positive_intervals: List[Tuple[float, float]]
    false_negative_intervals: List[Tuple[float, float]]
    confusion_table: dict


def _runs_of_true(mask: np.ndarray, time_resolution: float) -> List[Tuple[float, float]]:
    """Convert a boolean mask into a list of (start, end) intervals where mask is True."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(float(s * time_resolution), float(e * time_resolution)) for s, e in zip(starts, ends)]


def evaluate_meta_order_recovery(
    true_windows: Sequence[MetaOrderWindow],
    inferred_intervals: Sequence[Tuple[float, float, float]],
    horizon: float,
    *,
    time_resolution: float = 0.1,
) -> RecoveryEvaluation:
    """Compare inferred meta-order intervals against the simulator's true windows.

    Parameters
    ----------
    true_windows : sequence of MetaOrderWindow
        The simulator's ground-truth meta-order schedule.
    inferred_intervals : sequence of (start, end, score)
        Output of `recovery.score_meta_order_activity`.
    horizon : float
        Total simulation horizon in seconds.
    time_resolution : float
        Discretisation step for the per-sample comparison. Smaller values give
        finer-grained metrics at the cost of memory.

    Returns
    -------
    RecoveryEvaluation
    """
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if time_resolution <= 0.0:
        raise ValueError(
            f"time_resolution must be positive, got {time_resolution}"
        )

    n_steps = int(np.ceil(horizon / time_resolution))
    grid = np.arange(n_steps) * time_resolution

    true_mask = np.zeros(n_steps, dtype=bool)
    for w in true_windows:
        s_idx = int(np.floor(w.start_time / time_resolution))
        e_idx = int(np.ceil(w.end_time / time_resolution))
        true_mask[max(0, s_idx) : min(n_steps, e_idx)] = True

    inferred_mask = np.zeros(n_steps, dtype=bool)
    for start, end, _score in inferred_intervals:
        s_idx = int(np.floor(start / time_resolution))
        e_idx = int(np.ceil(end / time_resolution))
        inferred_mask[max(0, s_idx) : min(n_steps, e_idx)] = True

    tp = int(np.sum(true_mask & inferred_mask))
    fp = int(np.sum(inferred_mask & ~true_mask))
    fn = int(np.sum(true_mask & ~inferred_mask))
    tn = int(np.sum(~true_mask & ~inferred_mask))

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fp == 0 else 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Detection delay per true window.
    delays: List[float] = []
    for w in true_windows:
        s_idx = max(0, int(np.floor(w.start_time / time_resolution)))
        e_idx = min(n_steps, int(np.ceil(w.end_time / time_resolution)))
        if s_idx >= e_idx:
            continue
        local = inferred_mask[s_idx:e_idx]
        first_hit = np.argmax(local) if local.any() else -1
        if first_hit >= 0 and local.any():
            delay = first_hit * time_resolution
            delays.append(float(delay))

    mean_delay = float(np.mean(delays)) if delays else float("nan")

    fp_runs = _runs_of_true(inferred_mask & ~true_mask, time_resolution)
    fn_runs = _runs_of_true(true_mask & ~inferred_mask, time_resolution)

    return RecoveryEvaluation(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_detection_delay_seconds=mean_delay,
        false_positive_intervals=fp_runs,
        false_negative_intervals=fn_runs,
        confusion_table={
            "true_pos": tp,
            "false_pos": fp,
            "true_neg": tn,
            "false_neg": fn,
        },
    )


__all__ = [
    "RecoveryEvaluation",
    "evaluate_meta_order_recovery",
]
