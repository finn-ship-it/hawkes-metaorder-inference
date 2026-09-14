"""Tests for the endogeneity diagnostics in the summary and envelope."""

from __future__ import annotations

import dataclasses
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
    KernelFamily,
    validate,
    validate_first_milestone,
)
from simulator.hawkes_core import HawkesSimulator, RunTrace
from simulator.artifact import LoadedRun, save_run, load_run
from simulator.evaluation import (
    save_summary,
    summarise,
    _endogeneity_block,
    _total_baseline_rate,
    SUMMARY_FILENAME,
)
from simulator.comparison import compare_to_envelope, save_comparison


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _single_regime_config(
    seed: int = 20260423,
    horizon: float = 15.0,
    mu_scalar: float = 0.5,
) -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = mu_scalar * np.ones(d)
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


def _synthetic_loaded_run(
    cfg: SimulatorConfig,
    times: np.ndarray,
    event_types: np.ndarray,
    regime_times: np.ndarray,
    regime_values: np.ndarray,
    metadata: dict = None,
) -> LoadedRun:
    """Construct a LoadedRun in-memory without touching disk."""
    return LoadedRun(
        config=cfg,
        times=times,
        event_types=event_types,
        regime_times=regime_times,
        regime_values=regime_values,
        metadata=metadata or {
            "num_proposals": 0,
            "num_acceptances": 0,
            "acceptance_ratio": 0.0,
        },
        run_dir="<in-memory>",
        observed=None,
    )


# ---------------------------------------------------------------------------
# Spectral radius calculation
# ---------------------------------------------------------------------------


def test_spectral_radius_uniform_exponential():
    """For a 4x4 matrix with all entries equal to c, the branching matrix is
    rank-1 with spectral radius 4 * c. Our default cfg has alpha=0.1 and
    beta=2.0 so Phi_ij = 0.05 uniformly and rho = 0.2."""
    cfg = _single_regime_config()
    run = _synthetic_loaded_run(
        cfg=cfg,
        times=np.array([1.0, 2.0]),
        event_types=np.array([0, 1]),
        regime_times=np.zeros(1),
        regime_values=np.zeros(1, dtype=np.int64),
    )
    s = summarise(run)
    assert "endogeneity" in s
    assert s["endogeneity"]["kernel_family"] == "EXPONENTIAL"
    assert s["endogeneity"]["spectral_radius"] == pytest.approx(0.2, abs=1e-12)
    assert s["endogeneity"]["notes"] == [] or all(
        "spectral_radius" not in n for n in s["endogeneity"]["notes"]
    )


def test_spectral_radius_asymmetric_exponential():
    """Sanity check against a known asymmetric branching matrix.

    alpha = [[0.4, 0.0], [0.0, 0.8]]
    beta  = [[2.0, 2.0], [2.0, 2.0]]
    Phi   = [[0.2, 0.0], [0.0, 0.4]]
    rho(Phi) = 0.4.
    """
    d = 2
    alpha = np.array([[0.4, 0.0], [0.0, 0.8]])
    beta = 2.0 * np.ones((d, d))
    mu = 0.1 * np.ones(d)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=5.0,
        seed=1,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )
    validate(cfg)
    run = _synthetic_loaded_run(
        cfg=cfg,
        times=np.array([0.5]),
        event_types=np.array([0]),
        regime_times=np.zeros(1),
        regime_values=np.zeros(1, dtype=np.int64),
    )
    s = summarise(run)
    assert s["endogeneity"]["spectral_radius"] == pytest.approx(0.4, abs=1e-12)


# ---------------------------------------------------------------------------
# Rate-based proxy
# ---------------------------------------------------------------------------


def test_rate_based_proxy_single_regime_known_values():
    """For the default single-regime config mu = 0.5 * ones(4), the total
    baseline rate is 2.0. We inject a synthetic latent trace of 180 events
    over 60 seconds so total_rate = 3.0 and the proxy = 1 - 2/3 = 0.3333."""
    cfg = _single_regime_config(horizon=60.0, mu_scalar=0.5)
    times = np.linspace(0.1, 59.9, 180)
    types = np.tile(np.array([0, 1, 2, 3]), 45)
    run = _synthetic_loaded_run(
        cfg=cfg,
        times=times,
        event_types=types.astype(np.int64),
        regime_times=np.zeros(1),
        regime_values=np.zeros(1, dtype=np.int64),
    )
    s = summarise(run)
    end = s["endogeneity"]
    assert end["total_baseline_rate"] == pytest.approx(2.0)
    assert s["latent"]["total_rate"] == pytest.approx(180.0 / 60.0)
    assert end["rate_based_proxy"] == pytest.approx(1.0 - 2.0 / 3.0)


def test_rate_based_proxy_two_regime_time_weighted():
    """Two-regime config with mu_calm = 0.3 and mu_active = 1.5 and known
    time fractions. Baseline rate should be regime-time-weighted."""
    cfg = _two_regime_config(horizon=100.0)
    # Construct a regime trajectory that spends exactly 70% of [0, 100] in
    # regime 0 and 30% in regime 1. regime_times[0] = 0 by construction.
    rt = np.array([0.0, 70.0], dtype=np.float64)
    rv = np.array([0, 1], dtype=np.int64)
    # 60 events over the horizon, arbitrary types.
    times = np.linspace(0.5, 99.5, 60)
    types = np.tile(np.array([0, 1, 2, 3]), 15).astype(np.int64)
    run = _synthetic_loaded_run(
        cfg=cfg,
        times=times,
        event_types=types,
        regime_times=rt,
        regime_values=rv,
    )
    s = summarise(run)
    end = s["endogeneity"]
    # mu_calm sum = 4 * 0.3 = 1.2; mu_active sum = 4 * 1.5 = 6.0
    # Weighted: 0.7 * 1.2 + 0.3 * 6.0 = 0.84 + 1.8 = 2.64
    assert end["total_baseline_rate"] == pytest.approx(2.64, abs=1e-12)
    total_rate = s["latent"]["total_rate"]
    assert total_rate == pytest.approx(60.0 / 100.0)
    assert end["rate_based_proxy"] == pytest.approx(
        1.0 - 2.64 / total_rate, abs=1e-12
    )


def test_rate_based_proxy_null_for_zero_latent_rate():
    """Empty latent trace -> total_rate is zero -> proxy is null with an
    explanatory note."""
    cfg = _single_regime_config()
    run = _synthetic_loaded_run(
        cfg=cfg,
        times=np.zeros(0, dtype=np.float64),
        event_types=np.zeros(0, dtype=np.int64),
        regime_times=np.zeros(1),
        regime_values=np.zeros(1, dtype=np.int64),
    )
    s = summarise(run)
    end = s["endogeneity"]
    assert end["rate_based_proxy"] is None
    assert any("latent total rate is zero" in n for n in end["notes"])
    # Spectral radius is still defined in this case.
    assert end["spectral_radius"] == pytest.approx(0.2, abs=1e-12)
    # total_baseline_rate is still computable.
    assert end["total_baseline_rate"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Unsupported kernel family
# ---------------------------------------------------------------------------


def test_unsupported_kernel_family_returns_null_with_note():
    """A configuration whose kernel family is not implemented must produce a
    null spectral_radius with an explanatory note, not an unhandled
    exception. The config is constructed via dataclasses.replace so that the
    validator is not invoked."""
    cfg = _single_regime_config()
    bad_cfg = dataclasses.replace(
        cfg, kernel_family=KernelFamily.POWERLAW_CUTOFF
    )
    run = _synthetic_loaded_run(
        cfg=bad_cfg,
        times=np.array([1.0, 2.0]),
        event_types=np.array([0, 1]),
        regime_times=np.zeros(1),
        regime_values=np.zeros(1, dtype=np.int64),
    )
    s = summarise(run)
    end = s["endogeneity"]
    assert end["kernel_family"] == "POWERLAW_CUTOFF"
    assert end["spectral_radius"] is None
    assert any(
        "spectral_radius not computed for kernel family POWERLAW_CUTOFF" in n
        for n in end["notes"]
    )
    # The rate-based proxy is independent of the kernel family and should
    # still be computed.
    assert end["total_baseline_rate"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Comparison against the new envelope field
# ---------------------------------------------------------------------------


def test_comparison_passes_against_placeholder_spectral_radius_field():
    cfg = _single_regime_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260424T000000Z"
        )
        save_summary(run_dir)
        envelope = {
            "envelope_schema_version": "1",
            "fields": {
                "endogeneity.spectral_radius": {
                    "target": 0.2,
                    "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
                }
            },
        }
        env_path = os.path.join(tmp, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
        save_comparison(run_dir, env_path)
        report = json.load(
            open(os.path.join(run_dir, "comparison.json"), "r")
        )
    assert report["overall_status"] == "pass"
    block = report["per_field"]["endogeneity.spectral_radius"]
    assert block["status"] == "pass"
    assert block["actual"] == pytest.approx(0.2, abs=1e-12)


def test_comparison_fails_for_spectral_radius_far_from_target():
    """Calibrate a config whose spectral radius is 0.8 and check that the
    envelope flags it as fail."""
    d = 4
    alpha = 0.4 * np.ones((d, d))  # 4 * 0.4/2.0 = 0.8
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    cfg = SimulatorConfig(
        dimension=d,
        horizon=5.0,
        seed=1,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )
    validate(cfg)  # passes; spectral radius 0.8 < cap 0.95
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260424T000000Z"
        )
        save_summary(run_dir)
        envelope = {
            "envelope_schema_version": "1",
            "fields": {
                "endogeneity.spectral_radius": {
                    "target": 0.2,
                    "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
                }
            },
        }
        env_path = os.path.join(tmp, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
        save_comparison(run_dir, env_path)
        report = json.load(open(os.path.join(run_dir, "comparison.json")))
    assert report["per_field"]["endogeneity.spectral_radius"]["status"] == (
        "fail"
    )
    assert report["overall_status"] == "fail"


# ---------------------------------------------------------------------------
# Backward compatibility of existing run artefacts
# ---------------------------------------------------------------------------


def test_summarise_of_existing_run_artefact_populates_endogeneity_block():
    """A persisted run artefact (no embedded summary) continues to load and
    summarise correctly, and the new endogeneity block is populated freshly
    from its config."""
    cfg = _single_regime_config()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(
            cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260424T000000Z"
        )
        loaded = load_run(run_dir)
        s = summarise(loaded)
    assert "endogeneity" in s
    assert s["endogeneity"]["spectral_radius"] == pytest.approx(0.2, abs=1e-12)


# ---------------------------------------------------------------------------
# Direct helper behaviour
# ---------------------------------------------------------------------------


def test_total_baseline_rate_single_regime():
    cfg = _single_regime_config(mu_scalar=0.25)  # sum = 1.0
    rate, notes = _total_baseline_rate(cfg, [1.0])
    assert rate == pytest.approx(1.0)
    assert notes == []


def test_total_baseline_rate_two_regime_matches_weighting():
    cfg = _two_regime_config()
    rate, notes = _total_baseline_rate(cfg, [0.4, 0.6])
    # mu_calm sum = 1.2; mu_active sum = 6.0; weighted = 0.48 + 3.6 = 4.08
    assert rate == pytest.approx(4.08, abs=1e-12)
    assert notes == []


def test_total_baseline_rate_shape_mismatch_flags_note():
    cfg = _two_regime_config()
    rate, notes = _total_baseline_rate(cfg, [1.0])  # expects length 2
    assert rate is None
    assert any("regime time-fraction" in n for n in notes)
