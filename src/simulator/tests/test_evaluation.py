"""Tests for the deterministic evaluation harness."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simulator.config import (
    SimulatorConfig,
    ExponentialKernelParams,
    BaselineParams,
    LatentRegimeSpec,
    MetaOrderSpec,
    MetaOrderWindow,
    KernelFamily,
    validate,
    validate_first_milestone,
)
from simulator.hawkes_core import HawkesSimulator
from simulator.observation import (
    ObservationConfig,
    ObservationOperator,
    default_observation_config,
)
from simulator.artifact import save_run, load_run
from simulator.runner import run_once
from simulator.evaluation import (
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA_VERSION,
    projection_cost_f1,
    summarise,
    save_summary,
    _counts_by_event_type,
    _interarrival_moments,
    _time_in_each_regime,
    _time_in_each_window,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _single_regime_config(
    seed: int = 20260423, horizon: float = 30.0
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )
    validate_first_milestone(cfg)
    return cfg


def _two_regime_config(
    seed: int = 101, horizon: float = 200.0
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = np.array([[0.3] * d, [1.5] * d], dtype=float)
    Q = np.array([[-0.10, 0.10], [0.20, -0.20]], dtype=float)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
        latent_regime=LatentRegimeSpec(
            num_regimes=2, transition_rates=Q, initial_regime=0
        ),
    )
    validate(cfg)
    return cfg


def _meta_order_config(
    seed: int = 42, horizon: float = 120.0
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    windows = (
        MetaOrderWindow(
            start_time=20.0,
            end_time=80.0,
            alpha=1.0,
            target_event_types=(0, 2),
        ),
    )
    cfg = SimulatorConfig(
        dimension=d,
        horizon=horizon,
        seed=seed,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
        meta_order=MetaOrderSpec(windows=windows),
    )
    validate(cfg)
    return cfg


def _loaded_run_from_config(
    cfg: SimulatorConfig, tmp: str, with_observation: bool = False
):
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    observed = None
    if with_observation:
        observed = ObservationOperator(default_observation_config()).project(trace)
    run_dir = save_run(
        cfg=cfg,
        trace=trace,
        runs_root=tmp,
        timestamp="20260424T000000Z",
        observed=observed,
    )
    return load_run(run_dir), run_dir


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_projection_cost_f1_uses_projected_minus_latent():
    assert projection_cost_f1(
        0.29069767441860467, 0.25414364640883974
    ) == pytest.approx(0.03655402800976493)
    assert projection_cost_f1(
        0.7164179104477612, 0.7486631016042781
    ) == pytest.approx(-0.03224519115651692)


def test_counts_by_event_type_handles_empty_and_full():
    assert _counts_by_event_type(np.zeros(0, dtype=np.int64), 4) == [0, 0, 0, 0]
    arr = np.array([0, 0, 1, 2, 3, 3, 3], dtype=np.int64)
    assert _counts_by_event_type(arr, 4) == [2, 1, 1, 3]


def test_interarrival_moments_requires_two_or_more_timestamps():
    assert _interarrival_moments(np.zeros(0, dtype=np.float64)) == (None, None)
    assert _interarrival_moments(np.array([5.0])) == (None, None)
    m, v = _interarrival_moments(np.array([1.0, 3.0, 9.0]))
    # gaps = [2.0, 6.0], mean=4.0, population variance = 4.0
    assert m == pytest.approx(4.0)
    assert v == pytest.approx(4.0)


def test_time_in_each_regime_partitions_horizon():
    rt = np.array([0.0, 10.0, 25.0, 40.0], dtype=np.float64)
    rv = np.array([0, 1, 0, 1], dtype=np.int64)
    totals = _time_in_each_regime(rt, rv, horizon=60.0, num_regimes=2)
    # regime 0: [0,10) + [25,40) -> 10 + 15 = 25
    # regime 1: [10,25) + [40,60] -> 15 + 20 = 35
    np.testing.assert_allclose(totals, np.array([25.0, 35.0]))


def test_time_in_each_window_clips_to_horizon():
    windows = [
        {"start_time": 10.0, "end_time": 40.0,
         "alpha": 0.0, "target_event_types": [0]},
        {"start_time": 100.0, "end_time": 150.0,
         "alpha": 0.0, "target_event_types": [1]},
    ]
    t_in = _time_in_each_window(windows, horizon=120.0)
    # First window fully inside -> 30; second clipped to [100, 120] -> 20.
    assert t_in == [30.0, 20.0]


# ---------------------------------------------------------------------------
# Top-level summary structure
# ---------------------------------------------------------------------------


def test_summary_has_required_top_level_fields():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_single_regime_config(), tmp)
        s = summarise(run)
    for key in ("summary_schema_version", "horizon", "seed", "dimension",
                "latent", "regime", "meta_order", "observation"):
        assert key in s
    assert s["summary_schema_version"] == SUMMARY_SCHEMA_VERSION


def test_summary_without_observation_reports_present_false():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(
            _single_regime_config(), tmp, with_observation=False
        )
        s = summarise(run)
    assert s["observation"]["present"] is False


def test_summary_with_observation_reports_present_true_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(
            _single_regime_config(), tmp, with_observation=True
        )
        s = summarise(run)
    obs = s["observation"]
    assert obs["present"] is True
    assert obs["num_latent_events"] == int(run.times.shape[0])
    assert obs["num_observed_events"] + obs["num_censored_events"] == (
        obs["num_latent_events"]
    )
    # Under the censor policy the spread invariant holds.
    assert obs["num_crossed_observations"] == 0
    # Each per-type counter is a non-negative integer.
    assert all(c >= 0 for c in obs["counts_by_event_type"])
    # Final mid = 0.5 * (bid + ask), spread = ask - bid.
    assert obs["final_spread_ticks"] == (
        obs["final_ask_ticks"] - obs["final_bid_ticks"]
    )
    assert obs["final_mid_ticks"] == pytest.approx(
        0.5 * (obs["final_bid_ticks"] + obs["final_ask_ticks"])
    )


def test_latent_counts_and_rates_consistent_with_horizon():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_single_regime_config(), tmp)
        s = summarise(run)
    latent = s["latent"]
    assert latent["num_events"] == int(run.times.shape[0])
    assert sum(latent["counts_by_event_type"]) == latent["num_events"]
    horizon = s["horizon"]
    for c, r in zip(latent["counts_by_event_type"], latent["rates_by_event_type"]):
        assert r == pytest.approx(c / horizon)
    assert latent["total_rate"] == pytest.approx(
        sum(latent["rates_by_event_type"])
    )


# ---------------------------------------------------------------------------
# Regime block
# ---------------------------------------------------------------------------


def test_summary_single_regime_fraction_is_full():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_single_regime_config(), tmp)
        s = summarise(run)
    reg = s["regime"]
    assert reg["num_regimes"] == 1
    assert reg["num_regime_jumps"] == 0
    np.testing.assert_allclose(reg["time_fraction_by_regime"], [1.0])


def test_summary_two_regime_fractions_sum_to_one():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_two_regime_config(), tmp)
        s = summarise(run)
    reg = s["regime"]
    assert reg["num_regimes"] == 2
    total = sum(reg["time_fraction_by_regime"])
    assert total == pytest.approx(1.0, abs=1e-12)
    assert reg["num_regime_jumps"] >= 0


# ---------------------------------------------------------------------------
# Meta-order block
# ---------------------------------------------------------------------------


def test_summary_no_meta_order_reports_zero_fraction():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_single_regime_config(), tmp)
        s = summarise(run)
    mo = s["meta_order"]
    assert mo["num_windows"] == 0
    assert mo["time_fraction_inside_windows"] == 0.0
    assert mo["time_fraction_by_window"] == []


def test_summary_meta_order_window_fraction_matches_config():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _loaded_run_from_config(_meta_order_config(), tmp)
        s = summarise(run)
    mo = s["meta_order"]
    assert mo["num_windows"] == 1
    # window is [20, 80) with horizon 120 -> fraction 60/120 = 0.5
    assert mo["time_fraction_inside_windows"] == pytest.approx(0.5)
    assert mo["time_fraction_by_window"] == [pytest.approx(0.5)]
    assert mo["time_inside_windows"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_summary_writes_parseable_json_and_returns_path():
    with tempfile.TemporaryDirectory() as tmp:
        _, run_dir = _loaded_run_from_config(
            _single_regime_config(), tmp, with_observation=True
        )
        path = save_summary(run_dir)
        assert isinstance(path, Path)
        assert path.name == SUMMARY_FILENAME
        assert path.is_file()
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        # Round-trip: summarise(load_run(run_dir)) == parsed.
        fresh = summarise(load_run(run_dir))
        assert parsed == fresh


def test_save_summary_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        _, run_dir = _loaded_run_from_config(
            _single_regime_config(seed=303), tmp, with_observation=True
        )
        path_a = save_summary(run_dir)
        payload_a = json.load(open(path_a))
        path_b = save_summary(run_dir)
        payload_b = json.load(open(path_b))
        assert payload_a == payload_b


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_run_once_with_write_summary_persists_summary_json():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(
            cfg=_single_regime_config(seed=7, horizon=15.0),
            runs_root=tmp,
            observation_config=default_observation_config(),
            write_summary=True,
        )
        summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
        assert os.path.isfile(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["summary_schema_version"] == SUMMARY_SCHEMA_VERSION
        assert parsed["observation"]["present"] is True


def test_run_once_without_write_summary_does_not_write_summary_json():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(
            cfg=_single_regime_config(seed=7, horizon=15.0),
            runs_root=tmp,
        )
        assert not os.path.isfile(os.path.join(run_dir, SUMMARY_FILENAME))


# ---------------------------------------------------------------------------
# Spread-distribution diagnostics in the observation block
# ---------------------------------------------------------------------------


from simulator.artifact import LoadedRun  # noqa: E402
from simulator.observation import ObservedTrace  # noqa: E402
from simulator.comparison import compare_to_envelope  # noqa: E402


def _loaded_run_with_observed(
    cfg: SimulatorConfig,
    spread_ticks: np.ndarray,
    times: np.ndarray = None,
    event_types: np.ndarray = None,
) -> LoadedRun:
    """Assemble a LoadedRun in memory with a hand-crafted ObservedTrace.

    The bid/ask arrays are constructed so that ask - bid == spread_ticks;
    bid_ticks is held at 0, ask_ticks equals spread_ticks. The times and
    event_types default to a simple admissible sequence of ASK_UP events.
    """
    n = int(np.asarray(spread_ticks).shape[0])
    if times is None:
        times = np.arange(1, n + 1, dtype=np.float64)
    if event_types is None:
        event_types = 2 * np.ones(n, dtype=np.int64)  # ASK_UP
    spread_arr = np.asarray(spread_ticks, dtype=np.int64)
    bid_arr = np.zeros(n, dtype=np.int64)
    ask_arr = spread_arr.copy()
    observed = ObservedTrace(
        times=np.asarray(times, dtype=np.float64),
        event_types=np.asarray(event_types, dtype=np.int64),
        bid_ticks=bid_arr,
        ask_ticks=ask_arr,
        spread_ticks=spread_arr,
        initial_bid_ticks=0,
        initial_ask_ticks=1,
        min_spread_ticks=1,
        crossing_policy="censor",
        tick_size=1e-5,
        num_latent_events=n,
        num_censored_events=0,
    )
    return LoadedRun(
        config=cfg,
        times=np.asarray(times, dtype=np.float64),
        event_types=np.asarray(event_types, dtype=np.int64),
        regime_times=np.zeros(1, dtype=np.float64),
        regime_values=np.zeros(1, dtype=np.int64),
        metadata={
            "num_proposals": n,
            "num_acceptances": n,
            "acceptance_ratio": 1.0,
        },
        run_dir="<in-memory>",
        observed=observed,
    )


def test_spread_distribution_on_synthetic_trace():
    """Known spread_ticks vector with known percentiles."""
    cfg = _single_regime_config(horizon=60.0)
    # Strictly increasing 1..10. mean 5.5, median 5.5, p90 = 9.1, p99 ≈ 9.91,
    # max 10 (numpy linear interpolation convention).
    spread = np.arange(1, 11, dtype=np.int64)
    run = _loaded_run_with_observed(cfg, spread_ticks=spread)
    s = summarise(run)
    obs = s["observation"]
    assert obs["spread_mean_ticks"] == pytest.approx(5.5)
    assert obs["spread_median_ticks"] == pytest.approx(5.5)
    assert obs["spread_p90_ticks"] == pytest.approx(9.1)
    assert obs["spread_p99_ticks"] == pytest.approx(9.91)
    assert obs["spread_max_ticks"] == 10.0


def test_spread_distribution_single_observation_is_constant():
    cfg = _single_regime_config(horizon=60.0)
    run = _loaded_run_with_observed(cfg, spread_ticks=np.array([7], dtype=np.int64))
    s = summarise(run)
    obs = s["observation"]
    for key in (
        "spread_mean_ticks",
        "spread_median_ticks",
        "spread_p90_ticks",
        "spread_p99_ticks",
        "spread_max_ticks",
    ):
        assert obs[key] == 7.0


def test_spread_distribution_null_for_empty_observed_stream():
    """An observed trace with zero admitted events must return null for all
    five spread-distribution fields."""
    cfg = _single_regime_config(horizon=60.0)
    empty = ObservedTrace(
        times=np.zeros(0, dtype=np.float64),
        event_types=np.zeros(0, dtype=np.int64),
        bid_ticks=np.zeros(0, dtype=np.int64),
        ask_ticks=np.zeros(0, dtype=np.int64),
        spread_ticks=np.zeros(0, dtype=np.int64),
        initial_bid_ticks=0,
        initial_ask_ticks=1,
        min_spread_ticks=1,
        crossing_policy="censor",
        tick_size=1e-5,
        num_latent_events=0,
        num_censored_events=0,
    )
    run = LoadedRun(
        config=cfg,
        times=np.zeros(0, dtype=np.float64),
        event_types=np.zeros(0, dtype=np.int64),
        regime_times=np.zeros(1, dtype=np.float64),
        regime_values=np.zeros(1, dtype=np.int64),
        metadata={
            "num_proposals": 0,
            "num_acceptances": 0,
            "acceptance_ratio": 0.0,
        },
        run_dir="<in-memory>",
        observed=empty,
    )
    s = summarise(run)
    obs = s["observation"]
    assert obs["spread_mean_ticks"] is None
    assert obs["spread_median_ticks"] is None
    assert obs["spread_p90_ticks"] is None
    assert obs["spread_p99_ticks"] is None
    assert obs["spread_max_ticks"] is None


def test_final_spread_fields_unchanged_by_new_diagnostics():
    """Adding spread-distribution fields must not perturb the existing
    final_* diagnostics or min_spread_ticks/crossing_policy."""
    cfg = _single_regime_config(horizon=60.0)
    spread = np.array([1, 2, 3, 2, 1, 4], dtype=np.int64)
    run = _loaded_run_with_observed(cfg, spread_ticks=spread)
    s = summarise(run)
    obs = s["observation"]
    # Bid is 0 across the trajectory; ask follows spread. Final values come
    # from the LAST entry of each array.
    assert obs["final_bid_ticks"] == 0
    assert obs["final_ask_ticks"] == 4
    assert obs["final_spread_ticks"] == 4
    assert obs["final_mid_ticks"] == pytest.approx(2.0)
    assert obs["min_spread_ticks"] == 1
    assert obs["crossing_policy"] == "censor"


def test_comparison_resolves_new_spread_distribution_fields():
    """The comparison layer resolves dotted paths; verify that each of the
    new fields is reachable and that pass/warn/fail routing works."""
    cfg = _single_regime_config(horizon=60.0)
    # spread_mean = 5.5
    spread = np.arange(1, 11, dtype=np.int64)
    run = _loaded_run_with_observed(cfg, spread_ticks=spread)
    summary = summarise(run)
    envelope = {
        "envelope_schema_version": "1",
        "fields": {
            "observation.spread_mean_ticks": {
                "target": 5.5,
                "tolerance": {"warn_abs": 0.5, "fail_abs": 2.0},
            },  # pass
            "observation.spread_median_ticks": {
                "target": 5.0,
                "tolerance": {"warn_abs": 0.25, "fail_abs": 5.0},
            },  # actual 5.5 -> warn
            "observation.spread_p90_ticks": {
                "target": 2.0,
                "tolerance": {"warn_abs": 1.0, "fail_abs": 3.0},
            },  # actual 9.1 -> fail
        },
    }
    report = compare_to_envelope(summary, envelope)
    assert (
        report["per_field"]["observation.spread_mean_ticks"]["status"] == "pass"
    )
    assert (
        report["per_field"]["observation.spread_median_ticks"]["status"] == "warn"
    )
    assert (
        report["per_field"]["observation.spread_p90_ticks"]["status"] == "fail"
    )
    assert report["overall_status"] == "fail"


def test_backward_compatibility_summarise_on_existing_artefact():
    """A run artefact saved on disk (no embedded summary) continues to load
    and produce a summary; the new spread diagnostics are populated in
    place."""
    cfg = _single_regime_config(seed=42, horizon=15.0)
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    observed = ObservationOperator(default_observation_config()).project(trace)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg,
            trace=trace,
            runs_root=tmp,
            timestamp="20260424T000000Z",
            observed=observed,
        )
        loaded = load_run(run_dir)
        s = summarise(loaded)
    obs = s["observation"]
    for key in (
        "spread_mean_ticks",
        "spread_median_ticks",
        "spread_p90_ticks",
        "spread_p99_ticks",
        "spread_max_ticks",
    ):
        assert key in obs
        # If the run produced any observed events, values are non-null.
        if obs["num_observed_events"] > 0:
            assert obs[key] is not None
    # Ordering invariant: min <= median <= p90 <= p99 <= max.
    if obs["num_observed_events"] > 0:
        chain = [
            int(obs["min_spread_ticks"]),
            obs["spread_median_ticks"],
            obs["spread_p90_ticks"],
            obs["spread_p99_ticks"],
            obs["spread_max_ticks"],
        ]
        for a, b in zip(chain, chain[1:]):
            assert a <= b + 1e-12
