"""Tests for artifact.py and runner.py."""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np

from simulator.config import (
    KernelFamily,
    ExponentialKernelParams,
    BaselineParams,
    SimulatorConfig,
)
from simulator.hawkes_core import HawkesSimulator
from simulator.artifact import (
    ARTEFACT_FORMAT_VERSION,
    CONFIG_FILENAME,
    EVENTS_FILENAME,
    METADATA_FILENAME,
    save_run,
    load_run,
    run_directory_name,
)
from simulator.runner import (
    build_default_first_milestone_config,
    run_once,
)


def _make_cfg() -> SimulatorConfig:
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    return SimulatorConfig(
        dimension=d,
        horizon=30.0,
        seed=20260423,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )


def test_save_run_creates_three_files():
    cfg = _make_cfg()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260423T000000Z")
        assert os.path.isdir(run_dir)
        for name in (CONFIG_FILENAME, EVENTS_FILENAME, METADATA_FILENAME):
            assert os.path.isfile(os.path.join(run_dir, name))


def test_run_directory_name_format():
    name = run_directory_name(seed=42, timestamp="20260423T000000Z")
    assert name == "20260423T000000Z_42"


def test_load_run_roundtrip():
    cfg = _make_cfg()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260423T000000Z")
        loaded = load_run(run_dir)
        np.testing.assert_array_equal(loaded.times, trace.times)
        np.testing.assert_array_equal(loaded.event_types, trace.event_types)
        assert loaded.config.dimension == cfg.dimension
        assert loaded.config.seed == cfg.seed
        assert loaded.metadata["artefact_format_version"] == ARTEFACT_FORMAT_VERSION
        assert loaded.metadata["num_events"] == trace.num_events
        assert loaded.metadata["seed"] == trace.seed


def test_metadata_contents_consistent_with_trace():
    cfg = _make_cfg()
    sim = HawkesSimulator(cfg)
    sim.reset()
    trace = sim.simulate()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = save_run(cfg=cfg, trace=trace, runs_root=tmp, timestamp="20260423T000000Z")
        with open(os.path.join(run_dir, METADATA_FILENAME), "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dimension"] == 4
        assert meta["horizon"] == cfg.horizon
        assert meta["num_proposals"] == trace.num_proposals
        assert meta["num_acceptances"] == trace.num_acceptances
        assert abs(meta["acceptance_ratio"] - trace.acceptance_ratio) < 1e-12
        assert len(meta["final_intensity"]) == cfg.dimension


def test_runner_default_config_passes_first_milestone_gate():
    cfg = build_default_first_milestone_config()
    # build_default_first_milestone_config calls validate_first_milestone
    assert cfg.dimension == 4
    assert cfg.kernel_family is KernelFamily.EXPONENTIAL
    assert cfg.latent_regime.num_regimes == 1
    assert cfg.meta_order.enabled is False


def test_run_once_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(runs_root=tmp)
        assert os.path.isdir(run_dir)
        loaded = load_run(run_dir)
        assert loaded.metadata["dimension"] == 4
        # Under the default fixed seed the run is non-trivial.
        assert loaded.metadata["num_events"] >= 1
