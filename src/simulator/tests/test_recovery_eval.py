"""
test_recovery_eval.py — coverage for the recovery evaluation harness.
"""

from __future__ import annotations

import math

import pytest

from simulator.config import MetaOrderWindow
from simulator.recovery_eval import (
    RecoveryEvaluation,
    evaluate_meta_order_recovery,
)


def _w(start, end, alpha=1.0, targets=(0, 2)):
    return MetaOrderWindow(
        start_time=float(start),
        end_time=float(end),
        alpha=float(alpha),
        target_event_types=tuple(targets),
    )


def test_evaluate_perfect_recovery():
    true_windows = [_w(20.0, 80.0)]
    inferred = [(20.0, 80.0, 5.0)]
    ev = evaluate_meta_order_recovery(true_windows, inferred, horizon=120.0)
    assert ev.precision == pytest.approx(1.0)
    assert ev.recall == pytest.approx(1.0)
    assert ev.f1 == pytest.approx(1.0)
    assert ev.mean_detection_delay_seconds == pytest.approx(0.0)
    assert ev.false_positive_intervals == []
    assert ev.false_negative_intervals == []


def test_evaluate_perfect_miss():
    true_windows = [_w(20.0, 80.0)]
    inferred = []
    ev = evaluate_meta_order_recovery(true_windows, inferred, horizon=120.0)
    assert ev.recall == pytest.approx(0.0)
    # No inferred-active time at all -> precision is conventionally 1.0 (no FPs).
    assert ev.precision == pytest.approx(1.0)
    assert ev.f1 == pytest.approx(0.0)
    assert math.isnan(ev.mean_detection_delay_seconds)
    assert len(ev.false_negative_intervals) == 1
    assert ev.false_negative_intervals[0][0] == pytest.approx(20.0, abs=0.2)
    assert ev.false_negative_intervals[0][1] == pytest.approx(80.0, abs=0.2)


def test_evaluate_half_overlap():
    """Inferred interval covers the second half of the true window."""
    true_windows = [_w(20.0, 80.0)]
    inferred = [(50.0, 80.0, 5.0)]
    ev = evaluate_meta_order_recovery(
        true_windows, inferred, horizon=120.0, time_resolution=0.1
    )
    # Recall: 30 / 60 = 0.5
    assert ev.recall == pytest.approx(0.5, abs=0.01)
    # Precision: all 30 inferred seconds are inside truth -> 1.0
    assert ev.precision == pytest.approx(1.0, abs=0.01)
    assert ev.f1 == pytest.approx(2 / 3, abs=0.01)
    # Detection delay: first inferred-active sample inside truth is at t=50.
    # Window starts at 20 -> delay = 30.
    assert ev.mean_detection_delay_seconds == pytest.approx(30.0, abs=0.2)
    assert len(ev.false_negative_intervals) == 1
    assert ev.false_negative_intervals[0][0] == pytest.approx(20.0, abs=0.2)
    assert ev.false_negative_intervals[0][1] == pytest.approx(50.0, abs=0.2)


def test_evaluate_inferred_outside_truth_counts_as_false_positive():
    true_windows = [_w(20.0, 30.0)]
    inferred = [(60.0, 80.0, 3.0)]
    ev = evaluate_meta_order_recovery(true_windows, inferred, horizon=120.0)
    assert ev.precision == pytest.approx(0.0)
    assert ev.recall == pytest.approx(0.0)
    assert ev.f1 == pytest.approx(0.0)
    assert math.isnan(ev.mean_detection_delay_seconds)
    assert len(ev.false_positive_intervals) == 1


def test_evaluate_multiple_true_windows():
    true_windows = [_w(10.0, 20.0), _w(50.0, 60.0)]
    # Detect the first window perfectly, miss the second.
    inferred = [(10.0, 20.0, 4.0)]
    ev = evaluate_meta_order_recovery(true_windows, inferred, horizon=80.0)
    assert ev.recall == pytest.approx(0.5, abs=0.01)
    assert ev.precision == pytest.approx(1.0, abs=0.01)
    # Mean delay: only the first window contributes a delay (= 0).
    assert ev.mean_detection_delay_seconds == pytest.approx(0.0, abs=0.2)
    # False-negative coverage spans the second window.
    assert any(
        abs(fn[0] - 50.0) < 0.2 and abs(fn[1] - 60.0) < 0.2
        for fn in ev.false_negative_intervals
    )


def test_evaluate_rejects_invalid_horizon():
    with pytest.raises(ValueError):
        evaluate_meta_order_recovery([], [], horizon=0.0)


def test_evaluate_rejects_invalid_time_resolution():
    with pytest.raises(ValueError):
        evaluate_meta_order_recovery([], [], horizon=10.0, time_resolution=0.0)


def test_evaluate_no_true_windows_no_inferred():
    ev = evaluate_meta_order_recovery([], [], horizon=10.0)
    # Empty truth, empty inference -> precision and recall are vacuously 1.0.
    assert ev.precision == pytest.approx(1.0)
    assert ev.recall == pytest.approx(1.0)
    assert ev.f1 == pytest.approx(1.0)
