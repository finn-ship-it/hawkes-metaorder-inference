"""
test_ensemble_comparison.py — coverage for the optional ensemble-level
comparison module.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from simulator.ensemble_comparison import (
    compare_ensemble_to_envelope,
    save_ensemble_comparison,
    _build_synthetic_summary,
    ENSEMBLE_COMPARISON_FILENAME,
    ENSEMBLE_COMPARISON_SCHEMA_VERSION,
    ALLOWED_MODES,
    DEFAULT_K,
)
from simulator.comparison import EnvelopeError, load_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_ensemble_summary() -> dict:
    """A minimal ensemble summary with one scalar and one list aggregate."""
    return {
        "ensemble_schema_version": "1",
        "summary_schema_version": "1",
        "num_members": 5,
        "seeds": [1, 2, 3, 4, 5],
        "aggregate": {
            "latent.total_rate": {
                "kind": "scalar",
                "n": 5,
                "mean": 2.5,
                "std": 0.10,
                "min": 2.4,
                "max": 2.6,
            },
            "latent.rates_by_event_type": {
                "kind": "list",
                "length": 4,
                "elements": [
                    {"n": 5, "mean": 0.6, "std": 0.05, "min": 0.55, "max": 0.65},
                    {"n": 5, "mean": 0.6, "std": 0.05, "min": 0.55, "max": 0.65},
                    {"n": 5, "mean": 0.6, "std": 0.05, "min": 0.55, "max": 0.65},
                    {"n": 5, "mean": 0.6, "std": 0.05, "min": 0.55, "max": 0.65},
                ],
            },
            "endogeneity.spectral_radius": {
                "kind": "scalar",
                "n": 5,
                "mean": 0.20,
                "std": 0.0,
                "min": 0.20,
                "max": 0.20,
            },
        },
        "skipped_fields": {"meta_order.windows_active": "string field"},
    }


def _toy_envelope() -> dict:
    return {
        "envelope_schema_version": "1",
        "fields": {
            "latent.total_rate": {
                "target": 2.7,
                "tolerance": {"warn_rel": 0.3, "fail_rel": 0.6},
            },
            "latent.rates_by_event_type": {
                "target": [0.7, 0.7, 0.7, 0.7],
                "tolerance": {"warn_rel": 0.4, "fail_rel": 0.7},
            },
            "endogeneity.spectral_radius": {
                "target": 0.20,
                "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
            },
        },
    }


# ---------------------------------------------------------------------------
# Synthetic-summary construction
# ---------------------------------------------------------------------------


def test_build_synthetic_summary_mean_mode_uses_aggregate_means():
    es = _toy_ensemble_summary()
    syn = _build_synthetic_summary(es, mode="mean", k=2.0)
    assert syn["latent"]["total_rate"] == pytest.approx(2.5)
    assert syn["latent"]["rates_by_event_type"] == [0.6, 0.6, 0.6, 0.6]
    assert syn["endogeneity"]["spectral_radius"] == pytest.approx(0.20)


def test_build_synthetic_summary_mean_plus_std_widens():
    es = _toy_ensemble_summary()
    syn = _build_synthetic_summary(es, mode="mean_plus_std", k=2.0)
    assert syn["latent"]["total_rate"] == pytest.approx(2.5 + 2.0 * 0.10)
    assert syn["latent"]["rates_by_event_type"] == [
        pytest.approx(0.7),
        pytest.approx(0.7),
        pytest.approx(0.7),
        pytest.approx(0.7),
    ]


def test_build_synthetic_summary_rejects_invalid_mode():
    es = _toy_ensemble_summary()
    with pytest.raises(ValueError):
        _build_synthetic_summary(es, mode="not_a_mode", k=2.0)


# ---------------------------------------------------------------------------
# End-to-end comparison
# ---------------------------------------------------------------------------


def test_compare_ensemble_to_envelope_mean_mode():
    es = _toy_ensemble_summary()
    env = _toy_envelope()
    report = compare_ensemble_to_envelope(es, env, mode="mean")

    assert report["mode"] == "mean"
    assert report["k"] is None
    assert report["ensemble_num_members"] == 5
    assert report["ensemble_seeds"] == [1, 2, 3, 4, 5]
    assert (
        report["ensemble_comparison_schema_version"]
        == ENSEMBLE_COMPARISON_SCHEMA_VERSION
    )

    pf = report["per_field"]
    # Total rate: |2.5 - 2.7| / 2.7 = 0.074, well within warn_rel=0.3
    assert pf["latent.total_rate"]["status"] == "pass"
    # Rates: 0.6 vs 0.7, rel diff = 0.143; within 0.4
    assert pf["latent.rates_by_event_type"]["status"] == "pass"
    # Spectral radius: exact target
    assert pf["endogeneity.spectral_radius"]["status"] == "pass"
    assert report["overall_status"] == "pass"


def test_compare_ensemble_to_envelope_mean_plus_std_can_widen_to_warn():
    """A mean inside tolerances may push to warn under mean+k*std widening."""
    es = _toy_ensemble_summary()
    # Inflate the std on total_rate so mean+2*std crosses warn_rel=0.3.
    es["aggregate"]["latent.total_rate"]["std"] = 0.50  # mean+2*std = 3.5
    env = _toy_envelope()
    report = compare_ensemble_to_envelope(es, env, mode="mean_plus_std", k=2.0)
    assert report["mode"] == "mean_plus_std"
    assert report["k"] == 2.0
    # 3.5 vs 2.7: rel_diff = 0.296 < 0.3 (warn_rel) — still pass
    assert report["per_field"]["latent.total_rate"]["status"] == "pass"
    # Push k higher to force a warn
    report3 = compare_ensemble_to_envelope(es, env, mode="mean_plus_std", k=3.0)
    # mean + 3*std = 4.0; rel_diff = 1.296/2.7 = 0.48 > 0.3
    # but < 0.6 fail_rel, so status = warn
    assert report3["per_field"]["latent.total_rate"]["status"] == "warn"


def test_compare_ensemble_to_envelope_missing_path():
    es = _toy_ensemble_summary()
    env = _toy_envelope()
    env["fields"]["nonexistent.path"] = {
        "target": 1.0,
        "tolerance": {"warn_abs": 0.1, "fail_abs": 0.5},
    }
    report = compare_ensemble_to_envelope(es, env, mode="mean")
    assert report["per_field"]["nonexistent.path"]["status"] == "missing"
    assert report["overall_status"] in ("missing", "fail")


def test_compare_ensemble_skipped_fields_preserved():
    es = _toy_ensemble_summary()
    env = _toy_envelope()
    report = compare_ensemble_to_envelope(es, env, mode="mean")
    assert report["ensemble_skipped_fields"] == {
        "meta_order.windows_active": "string field"
    }


def test_compare_ensemble_rejects_invalid_mode():
    es = _toy_ensemble_summary()
    env = _toy_envelope()
    with pytest.raises(ValueError):
        compare_ensemble_to_envelope(es, env, mode="silly")


# ---------------------------------------------------------------------------
# save_ensemble_comparison disk round-trip
# ---------------------------------------------------------------------------


def test_save_ensemble_comparison_round_trip():
    es = _toy_ensemble_summary()
    env = _toy_envelope()
    with tempfile.TemporaryDirectory() as td:
        es_path = os.path.join(td, "ensemble_summary.json")
        env_path = os.path.join(td, "envelope.json")
        with open(es_path, "w", encoding="utf-8") as f:
            json.dump(es, f)
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env, f)
        out_path = save_ensemble_comparison(td, env_path, mode="mean")
        assert os.path.basename(str(out_path)) == ENSEMBLE_COMPARISON_FILENAME
        with open(out_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["overall_status"] == "pass"
        assert on_disk["mode"] == "mean"
        assert on_disk["ensemble_num_members"] == 5


def test_save_ensemble_comparison_missing_file_raises():
    with tempfile.TemporaryDirectory() as td:
        env_path = os.path.join(td, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(_toy_envelope(), f)
        with pytest.raises(FileNotFoundError):
            save_ensemble_comparison(td, env_path, mode="mean")


# ---------------------------------------------------------------------------
# Single-run comparison contract is preserved
# ---------------------------------------------------------------------------


def test_single_run_compare_to_envelope_unchanged():
    """Smoke test that we did not break the single-run contract by importing."""
    from simulator.comparison import compare_to_envelope
    summary = {
        "summary_schema_version": "1",
        "latent": {"total_rate": 2.5},
        "endogeneity": {"spectral_radius": 0.2},
    }
    env = {
        "envelope_schema_version": "1",
        "fields": {
            "latent.total_rate": {
                "target": 2.7,
                "tolerance": {"warn_rel": 0.3, "fail_rel": 0.6},
            },
            "endogeneity.spectral_radius": {
                "target": 0.2,
                "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
            },
        },
    }
    report = compare_to_envelope(summary, env)
    # The single-run path returns no 'mode' or 'ensemble_*' keys.
    assert "mode" not in report
    assert "ensemble_num_members" not in report
    assert report["overall_status"] == "pass"
