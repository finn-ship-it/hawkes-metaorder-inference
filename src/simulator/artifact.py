"""
artifact.py — persistence layer for the simulator core.

One completed simulation is persisted as a single run artefact directory:

    <root>/runs/<YYYYMMDDTHHMMSSZ>_<seed>/
        config.json       JSON sidecar describing the SimulatorConfig
        events.npz        compressed NPZ with the event times and types
        metadata.json     JSON file with run-level metadata

The format is deliberately simple and robust:
    - NPZ for the tabular arrays (binary, compressed, ubiquitous).
    - JSON for the configuration and metadata (human-readable, diffable).
    - No external dependencies beyond NumPy.

This module is intentionally agnostic of latent regime, meta-order windows,
and non-exponential kernels; those fields are already accommodated by the
configuration serializer in config.py and require no schema change here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from typing import Optional
import numpy as np

from .config import SimulatorConfig, config_to_json, config_from_json
from .hawkes_core import RunTrace
from .observation import ObservedTrace, ObservationConfig


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------

CONFIG_FILENAME = "config.json"
EVENTS_FILENAME = "events.npz"
METADATA_FILENAME = "metadata.json"
OBSERVATION_FILENAME = "observation.npz"

ARTEFACT_FORMAT_VERSION = "1"


# ---------------------------------------------------------------------------
# Metadata record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunMetadata:
    """Run-level metadata persisted alongside the event arrays."""

    artefact_format_version: str
    simulator_version: str
    run_timestamp_utc: str
    seed: int
    dimension: int
    horizon: float
    num_events: int
    num_proposals: int
    num_acceptances: int
    acceptance_ratio: float
    wall_clock_seconds: float
    final_intensity: list  # length d, serialisable as JSON
    num_regimes: int = 1
    num_regime_jumps: int = 0
    num_meta_order_windows: int = 0
    meta_order_windows: list = None  # list of dicts, serialisable as JSON
    # Observation layer. Populated only when an ObservedTrace is supplied to
    # save_run; otherwise the observation block is absent.
    has_observation: bool = False
    observation: dict = None  # observation config + summary counts

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("meta_order_windows") is None:
            d["meta_order_windows"] = []
        if d.get("observation") is None:
            d["observation"] = {}
        return d


# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------


def _utc_timestamp_compact() -> str:
    """Return a compact UTC timestamp of the form YYYYMMDDTHHMMSSZ."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y%m%dT%H%M%SZ")


def run_directory_name(seed: int, timestamp: Optional[str] = None) -> str:
    ts = timestamp if timestamp is not None else _utc_timestamp_compact()
    return f"{ts}_{int(seed)}"


def default_runs_root() -> str:
    """Return the default runs root directory relative to the repository.

    Resolution rule: `<repo_root>/runs/`, where repo_root is the directory
    containing the `src` folder that contains this module.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # here = .../src/simulator
    src_dir = os.path.dirname(here)
    repo_root = os.path.dirname(src_dir)
    return os.path.join(repo_root, "runs")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def save_run(
    cfg: SimulatorConfig,
    trace: RunTrace,
    runs_root: Optional[str] = None,
    simulator_version: str = "0.1.0",
    timestamp: Optional[str] = None,
    observed: Optional[ObservedTrace] = None,
) -> str:
    """Persist a completed run as a single directory.

    Parameters
    ----------
    cfg : SimulatorConfig
        The configuration that produced the run.
    trace : RunTrace
        The completed RunTrace returned by HawkesSimulator.simulate().
    runs_root : str, optional
        Base directory for run artefacts. Defaults to `<repo_root>/runs`.
    simulator_version : str
        Version string recorded in the metadata file.
    timestamp : str, optional
        Override for the UTC timestamp used in the directory name. Intended
        for tests and reproducibility checks.

    Returns
    -------
    str
        Absolute path of the created run directory.
    """
    if runs_root is None:
        runs_root = default_runs_root()

    dir_name = run_directory_name(seed=trace.seed, timestamp=timestamp)
    run_dir = os.path.join(runs_root, dir_name)
    os.makedirs(run_dir, exist_ok=False)

    # Configuration sidecar
    config_to_json(cfg, os.path.join(run_dir, CONFIG_FILENAME))

    # Events table and regime trajectory
    np.savez_compressed(
        os.path.join(run_dir, EVENTS_FILENAME),
        times=np.asarray(trace.times, dtype=np.float64),
        event_types=np.asarray(trace.event_types, dtype=np.int64),
        regime_times=np.asarray(trace.regime_times, dtype=np.float64),
        regime_values=np.asarray(trace.regime_values, dtype=np.int64),
    )

    # Observation layer. When an ObservedTrace is supplied, persist the
    # post-event quote arrays in a sibling NPZ file and record the
    # configuration and summary counts in the metadata.
    observation_dict: Optional[dict] = None
    has_observation = observed is not None
    if observed is not None:
        np.savez_compressed(
            os.path.join(run_dir, OBSERVATION_FILENAME),
            times=np.asarray(observed.times, dtype=np.float64),
            event_types=np.asarray(observed.event_types, dtype=np.int64),
            bid_ticks=np.asarray(observed.bid_ticks, dtype=np.int64),
            ask_ticks=np.asarray(observed.ask_ticks, dtype=np.int64),
            spread_ticks=np.asarray(observed.spread_ticks, dtype=np.int64),
        )
        observation_dict = {
            "initial_bid_ticks": int(observed.initial_bid_ticks),
            "initial_ask_ticks": int(observed.initial_ask_ticks),
            "min_spread_ticks": int(observed.min_spread_ticks),
            "crossing_policy": str(observed.crossing_policy),
            "tick_size": float(observed.tick_size),
            "num_latent_events": int(observed.num_latent_events),
            "num_observed_events": int(observed.num_observed_events),
            "num_censored_events": int(observed.num_censored_events),
            "censoring_fraction": float(observed.censoring_fraction),
            "num_crossed_observations": int(observed.num_crossed_observations),
            "final_bid_ticks": int(observed.final_bid_ticks),
            "final_ask_ticks": int(observed.final_ask_ticks),
        }

    # Meta-order window records serialised as a list of dicts so the
    # metadata file is both JSON-compatible and human-readable.
    meta_order_windows_serialised = [
        {
            "start_time": float(w[0]),
            "end_time": float(w[1]),
            "alpha": float(w[2]),
            "target_event_types": [int(i) for i in w[3]],
        }
        for w in getattr(trace, "meta_order_windows", ())
    ]

    # Metadata
    metadata = RunMetadata(
        artefact_format_version=ARTEFACT_FORMAT_VERSION,
        simulator_version=simulator_version,
        run_timestamp_utc=timestamp if timestamp is not None else _utc_timestamp_compact(),
        seed=int(trace.seed),
        dimension=int(trace.dimension),
        horizon=float(trace.horizon),
        num_events=int(trace.num_events),
        num_proposals=int(trace.num_proposals),
        num_acceptances=int(trace.num_acceptances),
        acceptance_ratio=float(trace.acceptance_ratio),
        wall_clock_seconds=float(trace.wall_clock_seconds),
        final_intensity=[float(x) for x in np.asarray(trace.final_intensity).tolist()],
        num_regimes=int(trace.num_regimes),
        num_regime_jumps=int(max(0, len(np.asarray(trace.regime_times)) - 1)),
        num_meta_order_windows=len(meta_order_windows_serialised),
        meta_order_windows=meta_order_windows_serialised,
        has_observation=has_observation,
        observation=observation_dict,
    )
    with open(os.path.join(run_dir, METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, sort_keys=False)

    return run_dir


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedRun:
    """In-memory representation of a persisted run, as returned by load_run.

    The observed_* fields are populated only when the run directory contains
    an observation.npz file. For runs saved without an observation layer the
    fields are None and the metadata dict does not contain a populated
    observation block.
    """

    config: SimulatorConfig
    times: np.ndarray
    event_types: np.ndarray
    regime_times: np.ndarray
    regime_values: np.ndarray
    metadata: dict
    run_dir: str
    observed: Optional[ObservedTrace] = None


def load_run(run_dir: str) -> LoadedRun:
    """Load a persisted run from a directory produced by save_run."""
    cfg_path = os.path.join(run_dir, CONFIG_FILENAME)
    events_path = os.path.join(run_dir, EVENTS_FILENAME)
    meta_path = os.path.join(run_dir, METADATA_FILENAME)

    for p in (cfg_path, events_path, meta_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"expected file not found: {p}")

    cfg = config_from_json(cfg_path)
    with np.load(events_path) as z:
        times = np.asarray(z["times"], dtype=np.float64)
        event_types = np.asarray(z["event_types"], dtype=np.int64)
        # Backward compatibility: older artefacts do not include regime arrays.
        if "regime_times" in z.files:
            regime_times = np.asarray(z["regime_times"], dtype=np.float64)
            regime_values = np.asarray(z["regime_values"], dtype=np.int64)
        else:
            regime_times = np.zeros(1, dtype=np.float64)
            regime_values = np.zeros(1, dtype=np.int64)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Observation layer is optional. Present iff observation.npz exists.
    observed: Optional[ObservedTrace] = None
    obs_path = os.path.join(run_dir, OBSERVATION_FILENAME)
    if os.path.isfile(obs_path):
        obs_block = metadata.get("observation", {}) or {}
        with np.load(obs_path) as z:
            bid_arr = np.asarray(z["bid_ticks"], dtype=np.int64)
            ask_arr = np.asarray(z["ask_ticks"], dtype=np.int64)
            # Backward compatibility: artefacts written before the spread
            # invariant was introduced do not store a spread_ticks array.
            if "spread_ticks" in z.files:
                spread_arr = np.asarray(z["spread_ticks"], dtype=np.int64)
            else:
                spread_arr = (ask_arr - bid_arr).astype(np.int64)
            n_obs = int(bid_arr.shape[0])
            observed = ObservedTrace(
                times=np.asarray(z["times"], dtype=np.float64),
                event_types=np.asarray(z["event_types"], dtype=np.int64),
                bid_ticks=bid_arr,
                ask_ticks=ask_arr,
                spread_ticks=spread_arr,
                initial_bid_ticks=int(obs_block.get("initial_bid_ticks", 0)),
                initial_ask_ticks=int(obs_block.get("initial_ask_ticks", 1)),
                min_spread_ticks=int(obs_block.get("min_spread_ticks", 1)),
                crossing_policy=str(obs_block.get("crossing_policy", "censor")),
                tick_size=float(obs_block.get("tick_size", 1e-5)),
                num_latent_events=int(
                    obs_block.get("num_latent_events", n_obs)
                ),
                num_censored_events=int(
                    obs_block.get("num_censored_events", 0)
                ),
            )

    return LoadedRun(
        config=cfg,
        times=times,
        event_types=event_types,
        regime_times=regime_times,
        regime_values=regime_values,
        metadata=metadata,
        run_dir=os.path.abspath(run_dir),
        observed=observed,
    )
