"""Tests for the envelope comparison layer."""

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
    KernelFamily,
    validate_first_milestone,
)
from simulator.hawkes_core import HawkesSimulator
from simulator.observation import (
    ObservationOperator,
    default_observation_config,
)
from simulator.artifact import save_run
from simulator.evaluation import save_summary, SUMMARY_FILENAME
from simulator.runner import run_once
from simulator.comparison import (
    COMPARISON_FILENAME,
    COMPARISON_SCHEMA_VERSION,
    EnvelopeError,
    compare_to_envelope,
    load_envelope,
    save_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_summary() -> dict:
    """A minimal summary dict with representative fields for testing."""
    return {
        "summary_schema_version": "1",
        "horizon": 60.0,
        "seed": 1,
        "dimension": 4,
        "latent": {
            "num_events": 120,
            "counts_by_event_type": [30, 30, 30, 30],
            "rates_by_event_type": [0.5, 0.5, 0.5, 0.5],
            "total_rate": 2.0,
            "acceptance_ratio": 0.9,
            "num_proposals": 133,
            "num_acceptances": 120,
        },
        "regime": {
            "num_regimes": 1,
            "num_regime_jumps": 0,
            "time_in_regime": [60.0],
            "time_fraction_by_regime": [1.0],
        },
        "meta_order": {
            "num_windows": 0,
            "time_inside_windows": 0.0,
            "time_fraction_inside_windows": 0.0,
            "time_fraction_by_window": [],
        },
        "observation": {
            "present": True,
            "num_latent_events": 120,
            "num_observed_events": 115,
            "num_censored_events": 5,
            "censoring_fraction": 5.0 / 120.0,
            "num_crossed_observations": 0,
            "counts_by_event_type": [29, 28, 29, 29],
            "interarrival_mean_by_event_type": [2.0, 2.0, 2.0, 2.0],
            "interarrival_variance_by_event_type": [1.0, 1.0, 1.0, 1.0],
            "final_bid_ticks": 0,
            "final_ask_ticks": 2,
            "final_mid_ticks": 1.0,
            "final_spread_ticks": 2,
            "min_spread_ticks": 1,
            "crossing_policy": "censor",
        },
    }


def _envelope(fields: dict, version: str = "1") -> dict:
    return {
        "envelope_schema_version": version,
        "description": "test envelope",
        "notes": [],
        "fields": fields,
    }


def _first_milestone_config(seed: int = 20260423, horizon: float = 15.0):
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


# ---------------------------------------------------------------------------
# Envelope loader
# ---------------------------------------------------------------------------


def test_load_envelope_rejects_missing_fields_key():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "e.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"description": "no fields"}, f)
        with pytest.raises(EnvelopeError):
            load_envelope(p)


def test_load_envelope_rejects_field_without_target():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "e.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {"fields": {"latent.total_rate": {"tolerance": {"warn_abs": 1}}}}, f
            )
        with pytest.raises(EnvelopeError):
            load_envelope(p)


def test_load_envelope_accepts_minimal_valid_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "e.json")
        payload = {
            "envelope_schema_version": "1",
            "fields": {"latent.total_rate": {"target": 2.0}},
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        loaded = load_envelope(p)
        assert loaded["envelope_schema_version"] == "1"
        assert "latent.total_rate" in loaded["fields"]


# ---------------------------------------------------------------------------
# Per-field comparison
# ---------------------------------------------------------------------------


def test_exact_match_passes():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.0,
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.3},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert report["overall_status"] == "pass"
    assert report["per_field"]["latent.total_rate"]["status"] == "pass"
    assert report["per_field"]["latent.total_rate"]["abs_diff"] == 0.0
    assert report["per_field"]["latent.total_rate"]["rel_diff"] == 0.0


def test_small_relative_deviation_warns():
    summary = _minimal_summary()
    # target 2.2 vs actual 2.0 -> rel 0.0909; warn_rel 0.05 triggers warn
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.2,
                "tolerance": {"warn_rel": 0.05, "fail_rel": 0.3},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert report["overall_status"] == "warn"
    assert report["per_field"]["latent.total_rate"]["status"] == "warn"


def test_large_deviation_fails_on_relative_tolerance():
    summary = _minimal_summary()
    # target 5.0 vs actual 2.0 -> rel 0.6; fail_rel 0.5 triggers fail
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 5.0,
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.5},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert report["overall_status"] == "fail"
    assert report["per_field"]["latent.total_rate"]["status"] == "fail"


def test_absolute_tolerance_fail_takes_precedence_over_warn():
    summary = _minimal_summary()
    # target 10 vs actual 2 -> abs_diff 8; fail_abs 5 triggers fail, not warn
    env = _envelope(
        {
            "observation.final_spread_ticks": {
                "target": 10,
                "tolerance": {"warn_abs": 1, "fail_abs": 5},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert (
        report["per_field"]["observation.final_spread_ticks"]["status"] == "fail"
    )


def test_no_tolerance_is_reported_as_unchecked():
    summary = _minimal_summary()
    env = _envelope({"latent.total_rate": {"target": 2.0}})
    report = compare_to_envelope(summary, env)
    field = report["per_field"]["latent.total_rate"]
    assert field["status"] == "unchecked"
    assert field["abs_diff"] == 0.0
    # overall degrades to unchecked in the absence of passes/warns/fails
    assert report["overall_status"] == "unchecked"


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def test_array_element_wise_pass_all():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.counts_by_event_type": {
                "target": [30, 30, 30, 30],
                "tolerance": {"warn_abs": 1, "fail_abs": 5},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert (
        report["per_field"]["latent.counts_by_event_type"]["status"] == "pass"
    )
    assert len(
        report["per_field"]["latent.counts_by_event_type"]["per_element"]
    ) == 4


def test_array_element_wise_worst_element_drives_status():
    summary = _minimal_summary()
    # counts are [30, 30, 30, 30]; target [30, 30, 30, 20] -> last element
    # abs_diff 10; warn_abs 1 triggers warn on that element; fail_abs 15
    # does not trigger. Overall "warn".
    env = _envelope(
        {
            "latent.counts_by_event_type": {
                "target": [30, 30, 30, 20],
                "tolerance": {"warn_abs": 1, "fail_abs": 15},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    block = report["per_field"]["latent.counts_by_event_type"]
    assert block["status"] == "warn"
    assert block["per_element"][0]["status"] == "pass"
    assert block["per_element"][3]["status"] == "warn"


def test_array_length_mismatch_fails():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.counts_by_event_type": {
                "target": [30, 30, 30],  # only 3 entries
                "tolerance": {"warn_abs": 1, "fail_abs": 5},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert (
        report["per_field"]["latent.counts_by_event_type"]["status"] == "fail"
    )
    assert "length" in report["per_field"][
        "latent.counts_by_event_type"
    ]["reason"]


def test_array_element_with_null_actual_is_unchecked():
    summary = _minimal_summary()
    # Inter-arrival mean array with a null (simulating <2 observations of one
    # event type).
    summary["observation"]["interarrival_mean_by_event_type"] = [
        2.0, 2.0, None, 2.0,
    ]
    env = _envelope(
        {
            "observation.interarrival_mean_by_event_type": {
                "target": [2.0, 2.0, 2.0, 2.0],
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.3},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    block = report["per_field"]["observation.interarrival_mean_by_event_type"]
    assert block["per_element"][2]["status"] == "unchecked"
    # Overall degrades to unchecked (the three numeric elements pass).
    assert block["status"] == "unchecked"


# ---------------------------------------------------------------------------
# Missing summary fields
# ---------------------------------------------------------------------------


def test_missing_summary_field_is_flagged_explicitly():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.0,
                "tolerance": {"warn_rel": 0.1},
            },
            "latent.nonexistent_field": {
                "target": 1.0,
                "tolerance": {"warn_rel": 0.1},
            },
        }
    )
    report = compare_to_envelope(summary, env)
    assert (
        report["per_field"]["latent.nonexistent_field"]["status"] == "missing"
    )
    assert (
        "nonexistent_field"
        in report["per_field"]["latent.nonexistent_field"]["reason"]
    )
    assert report["overall_status"] == "missing"


def test_missing_observation_fields_flagged_when_observation_absent():
    summary = _minimal_summary()
    summary["observation"] = {"present": False}
    env = _envelope(
        {
            "observation.final_spread_ticks": {
                "target": 2,
                "tolerance": {"warn_abs": 1},
            }
        }
    )
    report = compare_to_envelope(summary, env)
    assert (
        report["per_field"]["observation.final_spread_ticks"]["status"]
        == "missing"
    )


# ---------------------------------------------------------------------------
# Aggregation rules
# ---------------------------------------------------------------------------


def test_overall_status_aggregation_precedence():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.0,
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.3},
            },  # pass
            "latent.acceptance_ratio": {
                "target": 0.8,
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.5},
            },  # rel 0.125 -> warn
            "observation.final_spread_ticks": {
                "target": 10,
                "tolerance": {"warn_abs": 1, "fail_abs": 2},
            },  # abs 8 -> fail
        }
    )
    report = compare_to_envelope(summary, env)
    counts = report["status_counts"]
    assert counts["pass"] >= 1
    assert counts["warn"] >= 1
    assert counts["fail"] >= 1
    assert report["overall_status"] == "fail"


# ---------------------------------------------------------------------------
# Persistence and runner integration
# ---------------------------------------------------------------------------


def test_save_comparison_round_trip():
    summary = _minimal_summary()
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.0,
                "tolerance": {"warn_rel": 0.1, "fail_rel": 0.3},
            }
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = os.path.join(tmp, SUMMARY_FILENAME)
        env_path = os.path.join(tmp, "envelope.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f)
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env, f)
        out = save_comparison(tmp, env_path)
        assert isinstance(out, Path)
        assert out.name == COMPARISON_FILENAME
        with open(out, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["comparison_schema_version"] == COMPARISON_SCHEMA_VERSION
        assert parsed["overall_status"] == "pass"


def test_save_comparison_without_summary_raises():
    env = _envelope(
        {"latent.total_rate": {"target": 2.0, "tolerance": {"warn_abs": 0.1}}}
    )
    with tempfile.TemporaryDirectory() as tmp:
        env_path = os.path.join(tmp, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env, f)
        with pytest.raises(FileNotFoundError):
            save_comparison(tmp, env_path)


def test_runner_compare_flag_writes_comparison_json():
    cfg = _first_milestone_config()
    env = _envelope(
        {
            "latent.total_rate": {
                "target": 2.5,
                "tolerance": {"warn_rel": 1.0, "fail_rel": 2.0},
            }
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        env_path = os.path.join(tmp, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env, f)
        run_dir = run_once(
            cfg=cfg,
            runs_root=tmp,
            observation_config=default_observation_config(),
            compare_envelope_path=env_path,
        )
        assert os.path.isfile(os.path.join(run_dir, SUMMARY_FILENAME))
        assert os.path.isfile(os.path.join(run_dir, COMPARISON_FILENAME))
        with open(
            os.path.join(run_dir, COMPARISON_FILENAME), "r", encoding="utf-8"
        ) as f:
            report = json.load(f)
        assert report["overall_status"] in {"pass", "warn", "fail", "missing",
                                            "unchecked"}


def test_runner_without_compare_flag_does_not_write_comparison():
    cfg = _first_milestone_config()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_once(cfg=cfg, runs_root=tmp)
        assert not os.path.isfile(os.path.join(run_dir, COMPARISON_FILENAME))
