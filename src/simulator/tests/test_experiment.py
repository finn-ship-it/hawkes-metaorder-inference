"""
test_experiment.py — end-to-end coverage for the dissertation experiment runner.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from simulator.experiment import (
    ExperimentConfig,
    run_experiment,
    EXPERIMENT_MANIFEST_FILENAME,
    EXPERIMENT_EVALUATION_FILENAME,
    EXPERIMENT_RECOVERY_DIRNAME,
    EXPERIMENT_SCHEMA_VERSION,
)
from simulator.recovery import (
    MetaOrderBaselineConfig,
    RegimeBaselineConfig,
)
from simulator.runner import default_first_milestone_config_path


def _meta_order_smoke_path() -> str:
    here = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    return os.path.join(here, "configs", "meta_order_smoke.json")


def _placeholder_envelope_path() -> str:
    here = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    return os.path.join(here, "targets", "real_fx_envelope.json")


def test_run_experiment_minimal_first_milestone():
    """Smoke test on first_milestone with seeds (1, 2)."""
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig(
            config_path=default_first_milestone_config_path(),
            seeds=(1, 2),
            output_root=td,
            envelope_path=None,
            regime_baseline_cfg=None,
            meta_order_baseline_cfg=None,
            experiment_label="fm_smoke",
        )
        out = run_experiment(cfg)
        assert os.path.isdir(out)
        assert os.path.isfile(os.path.join(out, EXPERIMENT_MANIFEST_FILENAME))
        assert os.path.isfile(os.path.join(out, EXPERIMENT_EVALUATION_FILENAME))
        assert os.path.isdir(os.path.join(out, EXPERIMENT_RECOVERY_DIRNAME))
        assert os.path.isfile(os.path.join(out, "resolved_config.json"))
        with open(os.path.join(out, EXPERIMENT_MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        assert manifest["experiment_schema_version"] == EXPERIMENT_SCHEMA_VERSION
        assert manifest["seeds"] == [1, 2]
        # Ensemble-comparison file is None when no envelope was supplied.
        assert manifest["ensemble_comparison_file"] is None
        # No regime/meta-order baselines used.
        assert manifest["regime_baseline_used"] is False
        assert manifest["meta_order_baseline_used"] is False


def test_run_experiment_meta_order_smoke_with_recovery():
    """End-to-end with envelope, regime baseline, and meta-order baseline.

    Explicit `recovery_method="threshold"` preserves the original test intent
    after the default flipped to "hmm" (Task 1, restored_dissertation_pack
    Phase-A polish 2026-05-08).
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig(
            config_path=_meta_order_smoke_path(),
            seeds=(1, 2),
            output_root=td,
            envelope_path=_placeholder_envelope_path(),
            regime_baseline_cfg=RegimeBaselineConfig(
                window_seconds=10.0, rate_threshold=3.0
            ),
            meta_order_baseline_cfg=MetaOrderBaselineConfig(
                window_seconds=5.0, score_threshold=0.5,
                direction="up", min_active_seconds=5.0,
            ),
            recovery_method="threshold",
            experiment_label="mo_smoke",
        )
        out = run_experiment(cfg)
        with open(os.path.join(out, EXPERIMENT_MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        assert manifest["meta_order_baseline_used"] is True
        assert manifest["regime_baseline_used"] is True
        assert manifest["ensemble_comparison_file"] is not None
        assert os.path.isfile(manifest["ensemble_comparison_file"])
        with open(os.path.join(out, EXPERIMENT_EVALUATION_FILENAME)) as f:
            ev = json.load(f)
        assert ev["num_members"] == 2
        # Per-member evaluations should include a meta_order_evaluation block.
        assert "meta_order_evaluation" in ev["members"][0]
        # Aggregate block must be present and contain numeric means.
        assert "meta_order_aggregate" in ev
        agg = ev["meta_order_aggregate"]
        assert isinstance(agg["precision_mean"], float)
        assert 0.0 <= agg["precision_mean"] <= 1.0
        assert isinstance(agg["recall_mean"], float)
        assert 0.0 <= agg["recall_mean"] <= 1.0
        # True meta-order windows are recorded in the manifest.
        assert len(manifest["true_meta_order_windows"]) == 1
        assert manifest["true_meta_order_windows"][0]["start_time"] == 20.0


def test_run_experiment_rejects_empty_seeds():
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig(
            config_path=default_first_milestone_config_path(),
            seeds=(),
            output_root=td,
            experiment_label="bad",
        )
        with pytest.raises(ValueError):
            run_experiment(cfg)


def test_run_experiment_rejects_duplicate_seeds():
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig(
            config_path=default_first_milestone_config_path(),
            seeds=(1, 1, 2),
            output_root=td,
            experiment_label="bad",
        )
        with pytest.raises(ValueError):
            run_experiment(cfg)
