"""Tests for the post-restoration recovery_method default flip.

After Phase-A polish (restored_dissertation_pack 2026-05-08, Task 1) the
`ExperimentConfig.recovery_method` field defaults to "hmm". The
threshold detector remains callable as an explicit baseline. These three
tests verify:

    (1) default-is-hmm:           a fresh ExperimentConfig has recovery_method == "hmm".
    (2) threshold-still-callable: explicit "threshold" runs and returns
                                   the documented baseline structure.
    (3) hmm-and-threshold-coexist: running the same seed under each method
                                   produces results with the expected
                                   per-method shape and no shared state.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from simulator.experiment import (
    EXPERIMENT_EVALUATION_FILENAME,
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentConfig,
    run_experiment,
)
from simulator.recovery import MetaOrderBaselineConfig


def _meta_order_smoke_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "meta_order_smoke.json"
    )


def _placeholder_envelope_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "targets",
        "real_fx_envelope.json",
    )


# ---------------------------------------------------------------------------
# Test 1: default is HMM
# ---------------------------------------------------------------------------


def test_default_recovery_method_is_hmm():
    """A freshly-constructed ExperimentConfig has recovery_method == 'hmm'."""
    cfg = ExperimentConfig(
        config_path=_meta_order_smoke_path(),
        seeds=(1,),
        output_root="/tmp",  # never used; this is a pure introspection test
    )
    assert cfg.recovery_method == "hmm"
    # And the HMM cfg defaulting field is None unless explicitly overridden.
    assert cfg.hmm_recovery_cfg is None


# ---------------------------------------------------------------------------
# Test 2: threshold remains callable with documented baseline structure
# ---------------------------------------------------------------------------


def test_threshold_recovery_still_callable_with_baseline_structure():
    """`recovery_method='threshold'` runs to completion and produces a
    manifest + evaluation with the documented threshold-baseline shape.
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = ExperimentConfig(
            config_path=_meta_order_smoke_path(),
            seeds=(1, 2),
            output_root=td,
            envelope_path=_placeholder_envelope_path(),
            meta_order_baseline_cfg=MetaOrderBaselineConfig(
                window_seconds=5.0,
                score_threshold=0.5,
                direction="up",
                min_active_seconds=5.0,
            ),
            recovery_method="threshold",
            experiment_label="thresh_explicit",
        )
        out = run_experiment(cfg)
        with open(os.path.join(out, EXPERIMENT_MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        assert manifest["recovery_method"] == "threshold"
        # `meta_order_baseline_used` is True iff some recovery produced an evaluation.
        assert manifest["meta_order_baseline_used"] is True
        with open(os.path.join(out, EXPERIMENT_EVALUATION_FILENAME)) as f:
            ev = json.load(f)
        assert ev["recovery_method"] == "threshold"
        assert ev["num_members"] == 2
        # Each member has the threshold-shape evaluation (precision / recall / f1).
        for m in ev["members"]:
            assert "meta_order_evaluation" in m
            assert {"precision", "recall", "f1"}.issubset(
                m["meta_order_evaluation"].keys()
            )
            # Threshold path does NOT emit `hmm_summary`.
            assert "hmm_summary" not in m


# ---------------------------------------------------------------------------
# Test 3: HMM and threshold coexist (same seed, different methods, no leak)
# ---------------------------------------------------------------------------


def test_hmm_and_threshold_coexist_on_same_seed():
    """Running the same simulator config + seed under each recovery method
    produces results with the expected per-method shape (no exception, no
    shared state leak)."""
    with tempfile.TemporaryDirectory() as td:
        seeds = (3,)
        # HMM run (default settings)
        cfg_hmm = ExperimentConfig(
            config_path=_meta_order_smoke_path(),
            seeds=seeds,
            output_root=td,
            recovery_method="hmm",
            experiment_label="coexist_hmm",
        )
        out_hmm = run_experiment(cfg_hmm)
        # Threshold run (explicit)
        cfg_threshold = ExperimentConfig(
            config_path=_meta_order_smoke_path(),
            seeds=seeds,
            output_root=td,
            meta_order_baseline_cfg=MetaOrderBaselineConfig(
                window_seconds=5.0,
                score_threshold=0.5,
                direction="up",
                min_active_seconds=5.0,
            ),
            recovery_method="threshold",
            experiment_label="coexist_threshold",
        )
        out_threshold = run_experiment(cfg_threshold)

        with open(os.path.join(out_hmm, EXPERIMENT_EVALUATION_FILENAME)) as f:
            ev_hmm = json.load(f)
        with open(os.path.join(out_threshold, EXPERIMENT_EVALUATION_FILENAME)) as f:
            ev_threshold = json.load(f)

        assert ev_hmm["recovery_method"] == "hmm"
        assert ev_threshold["recovery_method"] == "threshold"

        # HMM members carry an `hmm_summary` block; threshold members do not.
        member_hmm = ev_hmm["members"][0]
        member_threshold = ev_threshold["members"][0]
        assert "hmm_summary" in member_hmm
        assert "hmm_summary" not in member_threshold
        # Both produce a meta_order_evaluation block on the meta_order_smoke config.
        assert "meta_order_evaluation" in member_hmm
        assert "meta_order_evaluation" in member_threshold
        # The two output directories are distinct (no shared state leak).
        assert out_hmm != out_threshold
        assert os.path.isdir(out_hmm) and os.path.isdir(out_threshold)
