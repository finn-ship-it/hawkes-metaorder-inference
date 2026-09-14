"""
ensemble.py — minimal harness for executing a configuration under multiple
distinct seeds and aggregating the resulting summaries.

The ensemble harness is a thin wrapper around the existing single-run
pipeline in runner.run_once. It does not alter the simulator core, the
observation operator, the comparison layer, or the summary schema. Its sole
responsibilities are:

    1. Construct a member SimulatorConfig for each requested seed by cloning
       the supplied configuration with seed overridden (dataclasses.replace).
    2. Invoke run_once for each member under a shared ensemble directory so
       that every member produces a full run artefact (config, events,
       metadata, summary, and optionally observation and comparison).
    3. Aggregate the per-member summary.json payloads into a single
       ensemble_summary.json that reports the mean, population standard
       deviation, minimum, and maximum of each numeric scalar field (or
       element-wise for consistent-length numeric list fields).

Per-member envelope comparison remains a per-run operation. The ensemble
summary records each member's overall status for auditing but does not
compute a cross-member comparison; that is deferred to a later milestone.

Aggregation rules
-----------------
Each member's summary.json is flattened into a set of (dotted_path, value)
pairs; walking descends into dicts and stops at lists. For each dotted path
observed in any member:

    - If the field is absent from one or more members it is recorded in
      skipped_fields with the reason "field absent from N members".
    - Otherwise, if every value is a numeric scalar (bool excluded) — or
      numeric with at most a finite number of nulls — the field is
      aggregated as a scalar with n, mean, std, min, max.
    - Otherwise, if every value is a list of identical length whose
      elements are numeric or null, the field is aggregated element-wise.
    - Otherwise the field is recorded in skipped_fields with a concrete
      reason (non-numeric scalar, variable list length, non-numeric list
      elements, or mixed types across members).
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .artifact import _utc_timestamp_compact, default_runs_root
from .comparison import COMPARISON_FILENAME
from .config import SimulatorConfig
from .evaluation import SUMMARY_FILENAME
from .observation import ObservationConfig, default_observation_config
from .runner import run_once


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


ENSEMBLE_SCHEMA_VERSION = "1"
ENSEMBLE_SUMMARY_FILENAME = "ensemble_summary.json"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    """True iff x is a Python int or float that is not a bool."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _value_kind(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, str):
        return "string"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    if isinstance(x, (int, float)):
        return type(x).__name__
    return type(x).__name__


def _walk_dotted_fields(node: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (dotted_path, value) pairs. Descends into dicts; stops at
    lists, scalars, and other leaf types."""
    if isinstance(node, dict):
        for key, child in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_dotted_fields(child, new_prefix)
    else:
        yield prefix, node


def _aggregate_scalar(
    path: str, values: List[Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Aggregate a list of scalar values. Returns (record, skip_reason).
    Numeric values are aggregated; nulls are counted separately. Returns
    (None, reason) if the values are not aggregatable."""
    numeric = [v for v in values if _is_number(v)]
    null_count = sum(1 for v in values if v is None)
    if not numeric:
        return None, None  # caller decides the reason
    if len(numeric) + null_count != len(values):
        return None, None
    arr = np.asarray(numeric, dtype=np.float64)
    record: Dict[str, Any] = {
        "kind": "scalar",
        "n": int(len(numeric)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if null_count > 0:
        record["n_null"] = int(null_count)
    return record, None


def _aggregate_list(
    path: str, values: List[Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Aggregate a list of list-valued fields element-wise."""
    lengths = {len(v) for v in values}
    if len(lengths) != 1:
        return None, (
            f"list lengths vary across members: {sorted(lengths)}"
        )
    length = next(iter(lengths))
    if length == 0:
        return None, "list fields are empty"
    for v in values:
        for x in v:
            if not (_is_number(x) or x is None):
                return None, "non-numeric elements in list field"
    element_stats: List[Dict[str, Any]] = []
    for j in range(length):
        column = [v[j] for v in values]
        numeric = [x for x in column if _is_number(x)]
        null_count = sum(1 for x in column if x is None)
        if not numeric:
            element_stats.append(
                {
                    "n": 0,
                    "n_null": int(null_count),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
            )
            continue
        arr = np.asarray(numeric, dtype=np.float64)
        entry = {
            "n": int(len(numeric)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        if null_count > 0:
            entry["n_null"] = int(null_count)
        element_stats.append(entry)
    return {
        "kind": "list",
        "length": int(length),
        "elements": element_stats,
    }, None


def _aggregate_summaries(
    summaries: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return (aggregate_by_path, skipped_by_path)."""
    if not summaries:
        return {}, {}

    # Collect per-path value lists.
    paths: Dict[str, List[Tuple[int, Any]]] = {}
    n_members = len(summaries)
    for i, s in enumerate(summaries):
        for path, value in _walk_dotted_fields(s):
            paths.setdefault(path, []).append((i, value))

    aggregate: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    for path, indexed in paths.items():
        if len(indexed) != n_members:
            skipped[path] = (
                f"field absent from {n_members - len(indexed)} of "
                f"{n_members} members"
            )
            continue
        values = [v for _, v in indexed]

        # Scalar aggregation first.
        record, _ = _aggregate_scalar(path, values)
        if record is not None:
            aggregate[path] = record
            continue

        # List aggregation.
        if all(isinstance(v, list) for v in values):
            record, reason = _aggregate_list(path, values)
            if record is not None:
                aggregate[path] = record
                continue
            if reason is not None:
                skipped[path] = reason
                continue

        # Fall-through: classify the reason precisely for the skipped map.
        kinds = sorted({_value_kind(v) for v in values})
        if len(kinds) == 1:
            k = kinds[0]
            if k == "string":
                skipped[path] = "non-numeric scalar (string)"
            elif k == "bool":
                skipped[path] = "non-numeric scalar (bool)"
            elif k == "null":
                skipped[path] = "all values are null"
            elif k == "list":
                skipped[path] = "non-numeric list field"
            elif k == "dict":
                skipped[path] = "unsupported nested dict leaf"
            else:
                skipped[path] = f"unsupported type ({k})"
        else:
            skipped[path] = f"mixed types across members: {kinds}"

    return aggregate, skipped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ensemble(
    cfg: SimulatorConfig,
    seeds: Iterable[int],
    runs_root: Optional[str] = None,
    observe: bool = True,
    envelope_path: Optional[str] = None,
    timestamp_prefix: Optional[str] = None,
    simulator_version: str = "0.1.0",
) -> str:
    """Run cfg under each supplied seed and persist an aggregated summary.

    Parameters
    ----------
    cfg : SimulatorConfig
        Base configuration. The seed field is overridden per member via
        dataclasses.replace; all other fields are shared across members.
    seeds : iterable of int
        Seeds to run. Must be non-empty and pairwise distinct.
    runs_root : str, optional
        Parent directory for the ensemble. Defaults to the repository's
        default runs root.
    observe : bool, default True
        When True, every member is projected through the default
        ObservationOperator.
    envelope_path : str, optional
        When supplied, every member is compared against the envelope JSON
        and its comparison.json is persisted alongside summary.json.
    timestamp_prefix : str, optional
        Deterministic timestamp for the ensemble and all its member
        directories. When None, a UTC timestamp is generated.
    simulator_version : str
        Version string forwarded to save_run.

    Returns
    -------
    str
        Absolute path of the ensemble directory.
    """
    seed_list = [int(s) for s in seeds]
    if len(seed_list) == 0:
        raise ValueError("at least one seed is required")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError(
            f"seeds must be pairwise distinct, got {seed_list}"
        )

    if runs_root is None:
        runs_root = default_runs_root()
    os.makedirs(runs_root, exist_ok=True)

    ensemble_ts = (
        timestamp_prefix
        if timestamp_prefix is not None
        else _utc_timestamp_compact()
    )
    ensemble_dir = os.path.join(runs_root, f"ensemble_{ensemble_ts}")
    os.makedirs(ensemble_dir, exist_ok=False)

    observation_config: Optional[ObservationConfig] = (
        default_observation_config() if observe else None
    )

    member_run_dirs: List[str] = []
    member_summaries: List[Dict[str, Any]] = []
    comparison_statuses: List[Dict[str, Any]] = []

    for seed in seed_list:
        member_cfg = dataclasses.replace(cfg, seed=int(seed))
        member_run_dir = run_once(
            cfg=member_cfg,
            runs_root=ensemble_dir,
            simulator_version=simulator_version,
            observation_config=observation_config,
            write_summary=True,
            compare_envelope_path=envelope_path,
            timestamp=ensemble_ts,
        )
        member_run_dirs.append(member_run_dir)
        with open(
            os.path.join(member_run_dir, SUMMARY_FILENAME), "r", encoding="utf-8"
        ) as f:
            member_summaries.append(json.load(f))
        if envelope_path is not None:
            with open(
                os.path.join(member_run_dir, COMPARISON_FILENAME),
                "r",
                encoding="utf-8",
            ) as f:
                comp = json.load(f)
            comparison_statuses.append(
                {
                    "seed": int(seed),
                    "run_dir": os.path.basename(member_run_dir),
                    "overall_status": comp["overall_status"],
                    "status_counts": comp["status_counts"],
                }
            )

    aggregate_fields, skipped_fields = _aggregate_summaries(member_summaries)

    comparison_summary: Dict[str, int] = {}
    if comparison_statuses:
        for status_name in ("pass", "warn", "fail", "missing", "unchecked"):
            comparison_summary[status_name] = sum(
                1
                for entry in comparison_statuses
                if entry["overall_status"] == status_name
            )

    report: Dict[str, Any] = {
        "ensemble_schema_version": ENSEMBLE_SCHEMA_VERSION,
        "ensemble_timestamp_utc": ensemble_ts,
        "num_members": len(seed_list),
        "seeds": seed_list,
        "member_run_dirs": [os.path.basename(d) for d in member_run_dirs],
        "observation_enabled": bool(observe),
        "comparison_requested": envelope_path is not None,
        "envelope_path": (
            os.path.abspath(envelope_path) if envelope_path is not None else None
        ),
        "comparison_summary": comparison_summary,
        "comparison_statuses": comparison_statuses,
        "aggregate": aggregate_fields,
        "skipped_fields": skipped_fields,
    }

    path = os.path.join(ensemble_dir, ENSEMBLE_SUMMARY_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)

    return ensemble_dir
