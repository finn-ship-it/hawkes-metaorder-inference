"""
comparison.py — deterministic comparison of summary.json against an empirical
envelope specification.

The comparison layer answers one question only: is the current simulator
output broadly plausible with respect to a declared envelope of empirical
targets, or is it mis-specified? It does not perform inference, does not fit
Hawkes parameters, and does not touch real market data directly. Its input
is the summary.json produced by the evaluation harness and an envelope JSON
file whose keys are dotted paths into that summary.

Per-field status values
-----------------------
    pass       actual value is within all specified tolerances
    warn       actual value exceeds at least one warn tolerance but no fail
               tolerance
    fail       actual value exceeds at least one fail tolerance, or an array
               field has a length mismatch against the target
    missing    the envelope references a summary path that does not resolve
    unchecked  the envelope field declares a target but no tolerances

Overall status is aggregated across fields with the precedence:
    missing or fail  > warn > unchecked > pass

A field whose actual value is None (for example, inter-arrival statistics
for a component with fewer than two observations) is reported with
status="unchecked" and an explanatory message; missing tolerances on that
element likewise yield "unchecked".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evaluation import SUMMARY_FILENAME


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


COMPARISON_FILENAME = "comparison.json"
COMPARISON_SCHEMA_VERSION = "1"

_MISSING = object()  # sentinel for failed dotted-path lookups


# ---------------------------------------------------------------------------
# Envelope I/O
# ---------------------------------------------------------------------------


class EnvelopeError(ValueError):
    """Raised when an envelope payload is structurally invalid."""


def load_envelope(path: str) -> Dict[str, Any]:
    """Read an envelope JSON file and validate its structural contract.

    The envelope must be a JSON object containing at least a "fields" dict.
    Each field value must be a dict with a "target" key; tolerances, notes,
    and placeholder flags are optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        envelope = json.load(f)
    if not isinstance(envelope, dict):
        raise EnvelopeError(
            f"envelope at {path} must be a JSON object at top level"
        )
    fields = envelope.get("fields")
    if not isinstance(fields, dict):
        raise EnvelopeError(
            f"envelope at {path} must contain a 'fields' object; got "
            f"{type(fields).__name__}"
        )
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            raise EnvelopeError(
                f"envelope field '{name}' must be an object, got "
                f"{type(spec).__name__}"
            )
        if "target" not in spec:
            raise EnvelopeError(
                f"envelope field '{name}' is missing the 'target' key"
            )
    return envelope


# ---------------------------------------------------------------------------
# Dotted-path lookup
# ---------------------------------------------------------------------------


def _lookup(summary: Dict[str, Any], dotted_path: str) -> Any:
    """Return summary[path] for a dotted path, or the _MISSING sentinel."""
    node: Any = summary
    for part in dotted_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return _MISSING
    return node


# ---------------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------------


_STATUS_RANK: Dict[str, int] = {
    "pass": 0,
    "unchecked": 1,
    "warn": 2,
    "fail": 3,
    "missing": 4,
}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def _compare_scalar(
    actual: Any, target: Any, tolerance: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare a scalar actual/target pair under the supplied tolerances."""
    if actual is None:
        return {
            "status": "unchecked",
            "target": target,
            "actual": None,
            "abs_diff": None,
            "rel_diff": None,
            "reason": "actual value is null",
        }

    try:
        diff = float(actual) - float(target)
    except (TypeError, ValueError):
        return {
            "status": "fail",
            "target": target,
            "actual": actual,
            "abs_diff": None,
            "rel_diff": None,
            "reason": (
                f"actual value {actual!r} is not numerically comparable to "
                f"target {target!r}"
            ),
        }

    abs_diff = abs(diff)
    try:
        target_magnitude = abs(float(target))
    except (TypeError, ValueError):
        target_magnitude = 0.0
    rel_diff: Optional[float] = (
        abs_diff / target_magnitude if target_magnitude > 0.0 else None
    )

    warn_abs = tolerance.get("warn_abs")
    fail_abs = tolerance.get("fail_abs")
    warn_rel = tolerance.get("warn_rel")
    fail_rel = tolerance.get("fail_rel")

    if all(t is None for t in (warn_abs, fail_abs, warn_rel, fail_rel)):
        return {
            "status": "unchecked",
            "target": target,
            "actual": actual,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "reason": "no tolerances specified",
        }

    status = "pass"
    reason: Optional[str] = None

    # Evaluate in fail > warn order so that a fail trigger is never hidden.
    if fail_abs is not None and abs_diff > float(fail_abs):
        status = "fail"
        reason = f"abs_diff {abs_diff:.6g} > fail_abs {float(fail_abs):.6g}"
    elif (
        fail_rel is not None
        and rel_diff is not None
        and rel_diff > float(fail_rel)
    ):
        status = "fail"
        reason = f"rel_diff {rel_diff:.6g} > fail_rel {float(fail_rel):.6g}"
    elif warn_abs is not None and abs_diff > float(warn_abs):
        status = "warn"
        reason = f"abs_diff {abs_diff:.6g} > warn_abs {float(warn_abs):.6g}"
    elif (
        warn_rel is not None
        and rel_diff is not None
        and rel_diff > float(warn_rel)
    ):
        status = "warn"
        reason = f"rel_diff {rel_diff:.6g} > warn_rel {float(warn_rel):.6g}"

    return {
        "status": status,
        "target": target,
        "actual": actual,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "reason": reason,
    }


def _compare_array(
    actual: Any, target: List[Any], tolerance: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(actual, list):
        return {
            "status": "fail",
            "target": target,
            "actual": actual,
            "per_element": [],
            "reason": (
                f"actual value is {type(actual).__name__}, expected list "
                f"of length {len(target)}"
            ),
        }
    if len(actual) != len(target):
        return {
            "status": "fail",
            "target": target,
            "actual": actual,
            "per_element": [],
            "reason": (
                f"actual length {len(actual)} does not match target length "
                f"{len(target)}"
            ),
        }

    per_element: List[Dict[str, Any]] = []
    worst = "pass"
    for i in range(len(target)):
        result = _compare_scalar(actual[i], target[i], tolerance)
        per_element.append(result)
        worst = _worse(worst, result["status"])

    return {
        "status": worst,
        "target": target,
        "actual": actual,
        "per_element": per_element,
        "reason": None,
    }


def _compare_field(
    actual: Any, spec: Dict[str, Any]
) -> Dict[str, Any]:
    target = spec["target"]
    tolerance = spec.get("tolerance", {}) or {}
    if isinstance(target, list):
        result = _compare_array(actual, target, tolerance)
    else:
        result = _compare_scalar(actual, target, tolerance)
    # Attach audit fields verbatim from the envelope specification so that
    # the comparison artefact is self-describing.
    if "placeholder" in spec:
        result["placeholder"] = bool(spec["placeholder"])
    if "notes" in spec:
        result["notes"] = spec["notes"]
    return result


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def compare_to_envelope(
    summary: Dict[str, Any], envelope: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare a summary dict against an envelope dict.

    Returns a JSON-serialisable comparison report with a per-field breakdown
    and an overall aggregated status.
    """
    if "fields" not in envelope or not isinstance(envelope["fields"], dict):
        raise EnvelopeError("envelope must contain a 'fields' object")

    per_field: Dict[str, Dict[str, Any]] = {}
    overall = "pass"
    counts = {"pass": 0, "warn": 0, "fail": 0, "missing": 0, "unchecked": 0}

    for field_name, spec in envelope["fields"].items():
        if not isinstance(spec, dict) or "target" not in spec:
            raise EnvelopeError(
                f"envelope field '{field_name}' is malformed (missing 'target')"
            )
        actual = _lookup(summary, field_name)
        if actual is _MISSING:
            result = {
                "status": "missing",
                "target": spec["target"],
                "actual": None,
                "abs_diff": None,
                "rel_diff": None,
                "reason": f"summary does not contain path '{field_name}'",
            }
            if "placeholder" in spec:
                result["placeholder"] = bool(spec["placeholder"])
            if "notes" in spec:
                result["notes"] = spec["notes"]
        else:
            result = _compare_field(actual, spec)
        per_field[field_name] = result
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        overall = _worse(overall, result["status"])

    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "envelope_schema_version": envelope.get(
            "envelope_schema_version", "unknown"
        ),
        "summary_schema_version": summary.get(
            "summary_schema_version", "unknown"
        ),
        "overall_status": overall,
        "status_counts": counts,
        "per_field": per_field,
    }


def save_comparison(run_dir: str, envelope_path: str) -> Path:
    """Compute comparison.json for the run at run_dir against the envelope
    at envelope_path and write it into the run directory.

    Requires summary.json to already exist inside run_dir. Returns the
    absolute Path of the written comparison file.
    """
    summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"expected {SUMMARY_FILENAME} inside {run_dir}; run with "
            f"--summarise before --compare"
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    envelope = load_envelope(envelope_path)
    report = compare_to_envelope(summary, envelope)
    path = Path(run_dir) / COMPARISON_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    return path.resolve()
