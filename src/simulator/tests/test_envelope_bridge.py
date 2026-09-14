"""
test_envelope_bridge.py — coverage for the empirical-envelope bridge.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

import pytest

from simulator.envelope_bridge import (
    load_empirical_envelope_draft,
    merge_envelopes,
    ALLOWED_PREFER,
)
from simulator.comparison import EnvelopeError, load_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_placeholder() -> dict:
    return {
        "envelope_schema_version": "1",
        "fields": {
            "latent.total_rate": {
                "target": 2.7,
                "tolerance": {"warn_rel": 0.3, "fail_rel": 0.6},
                "placeholder": True,
                "notes": "placeholder",
            },
            "endogeneity.spectral_radius": {
                "target": 0.20,
                "tolerance": {"warn_abs": 0.15, "fail_abs": 0.5},
                "placeholder": True,
                "notes": "placeholder",
            },
        },
    }


def _toy_draft() -> dict:
    return {
        "envelope_schema_version": "1",
        "fields": {
            "latent.total_rate": {
                "target": 1.0,
                "tolerance": {"warn_rel": 0.4, "fail_rel": 0.7},
                "placeholder": False,
                "source": "audit_csv",
                "notes": "empirical",
            },
            "observation.spread_mean_ticks": {
                "target": 3.8,
                "tolerance": {"warn_abs": 3.0, "fail_abs": 8.0},
                "placeholder": False,
                "source": "audit_csv",
                "notes": "empirical",
            },
        },
    }


# ---------------------------------------------------------------------------
# load_empirical_envelope_draft
# ---------------------------------------------------------------------------


def test_load_empirical_envelope_draft_validates_source():
    bad = _toy_draft()
    del bad["fields"]["latent.total_rate"]["source"]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "draft.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with pytest.raises(ValueError):
            load_empirical_envelope_draft(path)


def test_load_empirical_envelope_draft_succeeds_when_all_fields_have_source():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "draft.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_toy_draft(), f)
        env = load_empirical_envelope_draft(path)
        assert "latent.total_rate" in env["fields"]


def test_load_empirical_envelope_draft_loads_repo_draft():
    """Smoke test against the actual draft file under My repo/targets/."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    draft_path = os.path.join(here, "targets", "empirical_envelope_draft.json")
    if not os.path.isfile(draft_path):
        pytest.skip("empirical_envelope_draft.json not present in repo")
    env = load_empirical_envelope_draft(draft_path)
    assert "fields" in env
    for spec in env["fields"].values():
        assert "source" in spec


# ---------------------------------------------------------------------------
# merge_envelopes — prefer='draft'
# ---------------------------------------------------------------------------


def test_merge_envelopes_prefer_draft_overrides_shared_fields():
    placeholder = _toy_placeholder()
    draft = _toy_draft()
    merged = merge_envelopes(placeholder, draft, prefer="draft")

    # Shared field: draft wins.
    assert merged["fields"]["latent.total_rate"]["target"] == 1.0
    assert merged["fields"]["latent.total_rate"]["placeholder"] is False
    # Placeholder-only field: pass through.
    assert merged["fields"]["endogeneity.spectral_radius"]["target"] == 0.20
    # Draft-only field: pass through.
    assert merged["fields"]["observation.spread_mean_ticks"]["target"] == 3.8
    # Provenance recorded.
    assert merged["merge_provenance"]["latent.total_rate"].startswith("draft")
    assert (
        merged["merge_provenance"]["endogeneity.spectral_radius"]
        == "placeholder"
    )
    assert (
        "placeholder has no entry"
        in merged["merge_provenance"]["observation.spread_mean_ticks"]
    )
    assert merged["merge_mode"] == "draft"


def test_merge_envelopes_prefer_placeholder_keeps_placeholder_fields():
    placeholder = _toy_placeholder()
    draft = _toy_draft()
    merged = merge_envelopes(placeholder, draft, prefer="placeholder")
    # Shared field: placeholder wins.
    assert merged["fields"]["latent.total_rate"]["target"] == 2.7
    # Draft-only field still added.
    assert merged["fields"]["observation.spread_mean_ticks"]["target"] == 3.8
    assert merged["merge_mode"] == "placeholder"


def test_merge_envelopes_does_not_mutate_inputs():
    placeholder = _toy_placeholder()
    draft = _toy_draft()
    placeholder_before = copy.deepcopy(placeholder)
    draft_before = copy.deepcopy(draft)
    _ = merge_envelopes(placeholder, draft, prefer="draft")
    assert placeholder == placeholder_before
    assert draft == draft_before


def test_merge_envelopes_round_trips_to_placeholder_when_preferred():
    placeholder = _toy_placeholder()
    draft = _toy_draft()
    merged = merge_envelopes(placeholder, draft, prefer="placeholder")
    # Each placeholder field's spec is identical to the original.
    for name, spec in placeholder["fields"].items():
        for key, value in spec.items():
            assert merged["fields"][name][key] == value


def test_merge_envelopes_passes_load_envelope_validation():
    placeholder = _toy_placeholder()
    draft = _toy_draft()
    merged = merge_envelopes(placeholder, draft, prefer="draft")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "merged.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        env = load_envelope(path)
        assert "fields" in env


def test_merge_envelopes_rejects_invalid_prefer():
    with pytest.raises(ValueError):
        merge_envelopes(_toy_placeholder(), _toy_draft(), prefer="other")


# ---------------------------------------------------------------------------
# Round trip with the actual repo placeholder envelope
# ---------------------------------------------------------------------------


def test_merge_with_repo_placeholder_does_not_overwrite_placeholder_file():
    """Confirm that targets/real_fx_envelope.json is byte-identical after a
    merge — i.e. the placeholder file is never touched by this module."""
    here = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    placeholder_path = os.path.join(here, "targets", "real_fx_envelope.json")
    draft_path = os.path.join(here, "targets", "empirical_envelope_draft.json")
    if not (os.path.isfile(placeholder_path) and os.path.isfile(draft_path)):
        pytest.skip("repo envelope files not present")

    with open(placeholder_path, "rb") as f:
        before = f.read()
    placeholder = load_envelope(placeholder_path)
    draft = load_empirical_envelope_draft(draft_path)
    _ = merge_envelopes(placeholder, draft, prefer="draft")
    with open(placeholder_path, "rb") as f:
        after = f.read()
    assert before == after, (
        "envelope_bridge must not modify the on-disk placeholder envelope"
    )
