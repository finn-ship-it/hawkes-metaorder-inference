"""Tests for the ensemble harness."""

from __future__ import annotations

import json
import os
import tempfile
from typing import List

import numpy as np
import pytest

from simulator.config import (
    SimulatorConfig,
    ExponentialKernelParams,
    BaselineParams,
    KernelFamily,
    validate_first_milestone,
)
from simulator.ensemble import (
    ENSEMBLE_SCHEMA_VERSION,
    ENSEMBLE_SUMMARY_FILENAME,
    run_ensemble,
    _aggregate_summaries,
    _walk_dotted_fields,
)
from simulator.evaluation import SUMMARY_FILENAME
from simulator.comparison import COMPARISON_FILENAME


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _small_config(seed: int = 0, horizon: float = 5.0) -> SimulatorConfig:
    """A deliberately small configuration so the suite stays fast."""
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


def _load_ensemble_summary(ensemble_dir: str) -> dict:
    with open(
        os.path.join(ensemble_dir, ENSEMBLE_SUMMARY_FILENAME),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregation helpers (pure)
# ---------------------------------------------------------------------------


def test_walk_dotted_fields_stops_at_lists_and_scalars():
    tree = {"a": 1, "b": {"c": [1, 2], "d": "x"}, "e": None}
    out = dict(_walk_dotted_fields(tree))
    assert out == {"a": 1, "b.c": [1, 2], "b.d": "x", "e": None}


def test_aggregate_scalar_mean_std_min_max():
    summaries = [
        {"latent": {"total_rate": 1.0}},
        {"latent": {"total_rate": 2.0}},
        {"latent": {"total_rate": 3.0}},
    ]
    agg, skipped = _aggregate_summaries(summaries)
    r = agg["latent.total_rate"]
    assert r["kind"] == "scalar"
    assert r["n"] == 3
    assert r["mean"] == pytest.approx(2.0)
    assert r["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert r["min"] == 1.0
    assert r["max"] == 3.0
    assert "latent.total_rate" not in skipped


def test_aggregate_list_element_wise_when_lengths_match():
    summaries = [
        {"latent": {"counts_by_event_type": [10, 20, 30, 40]}},
        {"latent": {"counts_by_event_type": [20, 20, 30, 40]}},
        {"latent": {"counts_by_event_type": [30, 20, 30, 40]}},
    ]
    agg, _ = _aggregate_summaries(summaries)
    r = agg["latent.counts_by_event_type"]
    assert r["kind"] == "list"
    assert r["length"] == 4
    assert r["elements"][0]["mean"] == pytest.approx(20.0)
    assert r["elements"][0]["min"] == 10.0
    assert r["elements"][0]["max"] == 30.0
    assert r["elements"][1]["mean"] == pytest.approx(20.0)
    assert r["elements"][1]["std"] == 0.0


def test_aggregate_list_with_null_elements_is_handled():
    summaries = [
        {"obs": {"ia_mean": [1.0, 2.0, None]}},
        {"obs": {"ia_mean": [3.0, None, None]}},
        {"obs": {"ia_mean": [5.0, 6.0, 7.0]}},
    ]
    agg, _ = _aggregate_summaries(summaries)
    r = agg["obs.ia_mean"]
    e0, e1, e2 = r["elements"]
    assert e0["n"] == 3 and e0["mean"] == pytest.approx(3.0)
    assert e1["n"] == 2 and e1.get("n_null") == 1
    assert e2["n"] == 1 and e2.get("n_null") == 2


def test_aggregate_rejects_variable_list_length():
    summaries = [
        {"meta_order": {"time_fraction_by_window": [0.5]}},
        {"meta_order": {"time_fraction_by_window": [0.5, 0.1]}},
    ]
    _, skipped = _aggregate_summaries(summaries)
    assert "meta_order.time_fraction_by_window" in skipped
    assert "list lengths vary" in skipped[
        "meta_order.time_fraction_by_window"
    ]


def test_aggregate_skips_string_fields():
    summaries = [
        {"summary_schema_version": "1", "endogeneity": {"kernel_family": "EXPONENTIAL"}},
        {"summary_schema_version": "1", "endogeneity": {"kernel_family": "EXPONENTIAL"}},
    ]
    agg, skipped = _aggregate_summaries(summaries)
    assert "summary_schema_version" in skipped
    assert "endogeneity.kernel_family" in skipped
    assert "string" in skipped["summary_schema_version"]
    assert agg == {}


def test_aggregate_skips_bool_fields():
    summaries = [
        {"observation": {"present": True}},
        {"observation": {"present": True}},
    ]
    _, skipped = _aggregate_summaries(summaries)
    assert "observation.present" in skipped
    assert "bool" in skipped["observation.present"]


def test_aggregate_flags_field_absent_from_some_members():
    summaries = [
        {"latent": {"total_rate": 1.0}, "obs": {"n": 5}},
        {"latent": {"total_rate": 2.0}},  # obs.n missing
    ]
    _, skipped = _aggregate_summaries(summaries)
    assert "obs.n" in skipped
    assert "field absent" in skipped["obs.n"]


def test_aggregate_handles_nulls_in_scalar_with_n_null_record():
    summaries = [
        {"x": 1.0},
        {"x": None},
        {"x": 3.0},
    ]
    agg, _ = _aggregate_summaries(summaries)
    assert agg["x"]["n"] == 2
    assert agg["x"].get("n_null") == 1
    assert agg["x"]["mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# run_ensemble — filesystem layout
# ---------------------------------------------------------------------------


def test_run_ensemble_creates_expected_member_directories():
    cfg = _small_config()
    seeds = [1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=seeds,
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
        assert summary["num_members"] == 3
        assert summary["seeds"] == seeds
        assert len(summary["member_run_dirs"]) == 3
        for name in summary["member_run_dirs"]:
            member_dir = os.path.join(ens_dir, name)
            assert os.path.isdir(member_dir)


def test_each_member_has_summary_json():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
        for name in summary["member_run_dirs"]:
            assert os.path.isfile(
                os.path.join(ens_dir, name, SUMMARY_FILENAME)
            )


def test_ensemble_summary_has_required_fields():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
    assert summary["ensemble_schema_version"] == ENSEMBLE_SCHEMA_VERSION
    for key in (
        "num_members",
        "seeds",
        "member_run_dirs",
        "observation_enabled",
        "comparison_requested",
        "aggregate",
        "skipped_fields",
    ):
        assert key in summary


def test_ensemble_observation_disabled_skips_observed_artefacts():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
            observe=False,
        )
        summary = _load_ensemble_summary(ens_dir)
        assert summary["observation_enabled"] is False
        for name in summary["member_run_dirs"]:
            assert not os.path.isfile(
                os.path.join(ens_dir, name, "observation.npz")
            )


# ---------------------------------------------------------------------------
# run_ensemble — aggregate content
# ---------------------------------------------------------------------------


def test_ensemble_aggregate_contains_required_scalar_fields():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2, 3],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
    agg = summary["aggregate"]
    for key in (
        "latent.total_rate",
        "latent.num_events",
        "observation.censoring_fraction",
        "observation.final_spread_ticks",
        "endogeneity.spectral_radius",
        "endogeneity.rate_based_proxy",
    ):
        assert key in agg, f"{key} missing from aggregate"
        record = agg[key]
        assert record["kind"] == "scalar"
        assert record["n"] >= 1
        assert "mean" in record and "std" in record
        assert "min" in record and "max" in record


def test_ensemble_aggregate_list_fields_element_wise_when_consistent():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[11, 12, 13],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
    agg = summary["aggregate"]
    assert "latent.counts_by_event_type" in agg
    record = agg["latent.counts_by_event_type"]
    assert record["kind"] == "list"
    assert record["length"] == 4


def test_ensemble_skipped_fields_includes_non_numeric_entries():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
    skipped = summary["skipped_fields"]
    # At minimum the schema version and kernel family are strings and must
    # be skipped with a string-category reason.
    assert "summary_schema_version" in skipped
    assert "endogeneity.kernel_family" in skipped
    # A boolean example: observation.present
    assert "observation.present" in skipped


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fixed_seeds_give_deterministic_aggregate():
    cfg = _small_config()
    seeds = [17, 18, 19]
    summaries: List[dict] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            ens_dir = run_ensemble(
                cfg=cfg,
                seeds=seeds,
                runs_root=tmp,
                timestamp_prefix="20260424T000000Z",
            )
            summaries.append(_load_ensemble_summary(ens_dir))
    # Aggregate payload is a deterministic function of the seeds.
    assert summaries[0]["aggregate"] == summaries[1]["aggregate"]
    assert summaries[0]["skipped_fields"] == summaries[1]["skipped_fields"]


# ---------------------------------------------------------------------------
# Optional per-member comparison
# ---------------------------------------------------------------------------


def test_ensemble_with_envelope_writes_per_member_comparison():
    cfg = _small_config()
    envelope = {
        "envelope_schema_version": "1",
        "fields": {
            "endogeneity.spectral_radius": {
                "target": 0.2,
                "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        env_path = os.path.join(tmp, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2, 3],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
            envelope_path=env_path,
        )
        summary = _load_ensemble_summary(ens_dir)
        assert summary["comparison_requested"] is True
        assert len(summary["comparison_statuses"]) == 3
        for entry in summary["comparison_statuses"]:
            assert entry["overall_status"] == "pass"
            # Spot-check that the comparison file really exists.
            member = os.path.join(ens_dir, entry["run_dir"])
            assert os.path.isfile(os.path.join(member, COMPARISON_FILENAME))
        assert summary["comparison_summary"]["pass"] == 3


def test_ensemble_without_envelope_skips_comparison_files():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        ens_dir = run_ensemble(
            cfg=cfg,
            seeds=[1, 2],
            runs_root=tmp,
            timestamp_prefix="20260424T000000Z",
        )
        summary = _load_ensemble_summary(ens_dir)
        assert summary["comparison_requested"] is False
        assert summary["comparison_statuses"] == []
        for name in summary["member_run_dirs"]:
            assert not os.path.isfile(
                os.path.join(ens_dir, name, COMPARISON_FILENAME)
            )


# ---------------------------------------------------------------------------
# Seed hygiene
# ---------------------------------------------------------------------------


def test_empty_seed_list_raises():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            run_ensemble(
                cfg=cfg,
                seeds=[],
                runs_root=tmp,
                timestamp_prefix="20260424T000000Z",
            )


def test_duplicate_seeds_raise():
    cfg = _small_config()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            run_ensemble(
                cfg=cfg,
                seeds=[1, 2, 1],
                runs_root=tmp,
                timestamp_prefix="20260424T000000Z",
            )
