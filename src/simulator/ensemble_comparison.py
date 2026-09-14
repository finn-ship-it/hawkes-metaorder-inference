"""
ensemble_comparison.py — optional ensemble-level comparison of an
ensemble_summary.json against an empirical envelope.

This module is *additive* with respect to comparison.py and ensemble.py.
The single-run comparison contract in comparison.py is preserved unchanged.
ensemble.py is not modified by this module; the ensemble harness still
produces the same artefacts.

Why an ensemble-level comparison?
---------------------------------
A single-run comparison evaluates one realisation of the simulator against
envelope tolerances. For fields that are sample-noisy (such as the spread
distribution diagnostics, which are sub-diffusive in N_obs as quantified
under prompt 01) the ensemble mean is materially more stable than the
per-member value. Comparing the ensemble mean to the envelope target makes
plausibility checks less seed-dependent.

Public surface
--------------
- compare_ensemble_to_envelope(ensemble_summary, envelope, *, mode='mean')
- save_ensemble_comparison(ensemble_dir, envelope_path, *, mode='mean')

Both call into comparison.compare_to_envelope after constructing a
"synthetic" summary dict from the ensemble's aggregate block. The synthetic
summary uses ensemble-mean (mode='mean') or mean-plus-k-std (mode='mean_plus_std')
values for each leaf path that exists in the aggregate; non-aggregated paths
are passed through as-is from the ensemble report's metadata.

Output schema
-------------
The returned dict has the same shape as comparison.compare_to_envelope's
output, augmented with:
- 'mode': the aggregation mode used ('mean' or 'mean_plus_std')
- 'ensemble_num_members': number of ensemble members
- 'ensemble_seeds': the seed list
- 'ensemble_skipped_fields': dotted paths the ensemble harness could not
  aggregate (passed through verbatim)
- 'k': the std multiplier used when mode='mean_plus_std' (else None)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comparison import (
    compare_to_envelope,
    load_envelope,
    EnvelopeError,
)
from .ensemble import ENSEMBLE_SUMMARY_FILENAME

ENSEMBLE_COMPARISON_FILENAME = "ensemble_comparison.json"
ENSEMBLE_COMPARISON_SCHEMA_VERSION = "1"
ALLOWED_MODES = ("mean", "mean_plus_std")
DEFAULT_K = 2.0


def _build_synthetic_summary(
    ensemble_summary: Dict[str, Any], mode: str, k: float
) -> Dict[str, Any]:
    """Reconstruct a nested summary-shaped dict from the ensemble's aggregate.

    The aggregate dict is keyed by dotted paths; we walk the keys and rebuild
    a nested dict so that comparison._lookup can resolve the same dotted
    paths the envelope uses.
    """
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"mode must be one of {ALLOWED_MODES!r}, got {mode!r}"
        )

    aggregate = ensemble_summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        raise EnvelopeError(
            "ensemble_summary['aggregate'] must be a dict"
        )

    synthetic: Dict[str, Any] = {}

    for dotted_path, record in aggregate.items():
        if not isinstance(record, dict):
            continue
        kind = record.get("kind", "scalar")
        if kind == "scalar":
            mean = record.get("mean")
            std = record.get("std")
            if mean is None:
                value: Any = None
            elif mode == "mean":
                value = mean
            else:  # mean_plus_std
                if std is None:
                    value = mean
                else:
                    # Use the upper edge of the band: mean + k * std. The
                    # comparison machinery is symmetric about the target,
                    # so a single representative value suffices for the
                    # plausibility check.
                    value = mean + k * std
        elif kind == "list":
            elements = record.get("elements", [])
            value = []
            for el in elements:
                el_mean = el.get("mean")
                el_std = el.get("std")
                if el_mean is None:
                    value.append(None)
                elif mode == "mean":
                    value.append(el_mean)
                else:
                    if el_std is None:
                        value.append(el_mean)
                    else:
                        value.append(el_mean + k * el_std)
        else:
            value = record.get("mean")

        # Walk the dotted path and insert.
        node = synthetic
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value

    # Forward the schema versions from the ensemble report when available so
    # the comparison report's auditing fields read sensibly.
    synthetic.setdefault(
        "summary_schema_version",
        ensemble_summary.get("summary_schema_version", "ensemble"),
    )
    return synthetic


def compare_ensemble_to_envelope(
    ensemble_summary: Dict[str, Any],
    envelope: Dict[str, Any],
    *,
    mode: str = "mean",
    k: float = DEFAULT_K,
) -> Dict[str, Any]:
    """Compare an ensemble's aggregate block against an empirical envelope.

    Parameters
    ----------
    ensemble_summary : dict
        Parsed ensemble_summary.json, as produced by simulator.ensemble.run_ensemble.
    envelope : dict
        Parsed envelope dict, as produced by comparison.load_envelope.
    mode : str
        'mean' (default) compares the ensemble-mean to the target.
        'mean_plus_std' compares mean + k * std to the target, providing a
        looser plausibility check for noisy fields.
    k : float
        Multiplier for the std band when mode='mean_plus_std'. Defaults to 2.

    Returns
    -------
    dict
        A comparison report shaped like comparison.compare_to_envelope's
        return value, with additional ensemble-aware metadata keys.
    """
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"mode must be one of {ALLOWED_MODES!r}, got {mode!r}"
        )

    synthetic_summary = _build_synthetic_summary(ensemble_summary, mode, k)
    base_report = compare_to_envelope(synthetic_summary, envelope)

    base_report["mode"] = mode
    base_report["k"] = k if mode == "mean_plus_std" else None
    base_report["ensemble_num_members"] = ensemble_summary.get("num_members")
    base_report["ensemble_seeds"] = ensemble_summary.get("seeds")
    base_report["ensemble_skipped_fields"] = ensemble_summary.get(
        "skipped_fields", {}
    )
    base_report["ensemble_comparison_schema_version"] = (
        ENSEMBLE_COMPARISON_SCHEMA_VERSION
    )
    return base_report


def save_ensemble_comparison(
    ensemble_dir: str,
    envelope_path: str,
    *,
    mode: str = "mean",
    k: float = DEFAULT_K,
    out_filename: str = ENSEMBLE_COMPARISON_FILENAME,
) -> Path:
    """Read ensemble_summary.json from ensemble_dir, compare against the
    envelope at envelope_path, and write the result into ensemble_dir.

    Returns the absolute Path of the written file.
    """
    es_path = os.path.join(ensemble_dir, ENSEMBLE_SUMMARY_FILENAME)
    if not os.path.isfile(es_path):
        raise FileNotFoundError(
            f"expected {ENSEMBLE_SUMMARY_FILENAME} inside {ensemble_dir}"
        )
    with open(es_path, "r", encoding="utf-8") as f:
        ensemble_summary = json.load(f)
    envelope = load_envelope(envelope_path)
    report = compare_ensemble_to_envelope(
        ensemble_summary, envelope, mode=mode, k=k
    )
    out_path = Path(ensemble_dir) / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    return out_path.resolve()


__all__: List[str] = [
    "ENSEMBLE_COMPARISON_FILENAME",
    "ENSEMBLE_COMPARISON_SCHEMA_VERSION",
    "ALLOWED_MODES",
    "DEFAULT_K",
    "compare_ensemble_to_envelope",
    "save_ensemble_comparison",
]
