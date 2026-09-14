"""
evaluation.py — deterministic summary harness for persisted run artefacts.

Given a LoadedRun returned by artifact.load_run, the harness computes a fixed
panel of summary statistics covering the latent event stream, the latent
regime trajectory, any meta-order windows, and (when present) the observable
event stream produced by the observation operator. The output is a
JSON-serialisable dict and may be persisted as summary.json inside the run
directory.

The harness is pure and deterministic: the same input produces the same
output bit-for-bit. It performs no inference, makes no comparison against
external data, and does not invoke the simulator core. Its purpose is to
expose a stable numerical surface against which later estimator work, kernel
changes, or observation-layer refinements can be measured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .artifact import LoadedRun, load_run
from .config import KernelFamily
from .kernels import build_kernel, spectral_radius as _kernel_spectral_radius


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SUMMARY_FILENAME = "summary.json"
SUMMARY_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def projection_cost_f1(f1_projected: float, f1_latent: float) -> float:
    """Return the repository's signed F1 change after projection.

    The canonical convention is ``F1_projected - F1_latent``.  A negative
    value therefore records a loss after projection, while a positive value
    means that projected-stream recovery scored higher.  Historical artefacts
    retain the JSON key ``censoring_cost_f1`` for compatibility.
    """
    return float(f1_projected) - float(f1_latent)


def _counts_by_event_type(event_types: np.ndarray, dimension: int) -> List[int]:
    """Return a length-`dimension` list of per-component event counts."""
    d = int(dimension)
    if event_types is None or np.asarray(event_types).shape[0] == 0:
        return [0] * d
    c = np.bincount(np.asarray(event_types, dtype=np.int64), minlength=d)
    return [int(x) for x in c[:d]]


def _interarrival_moments(
    times: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, population variance) of successive inter-arrival gaps.

    Requires at least two timestamps to produce a defined mean, and the
    population variance is reported with `ddof=0`. Both values are None when
    fewer than two timestamps are supplied.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.shape[0] < 2:
        return None, None
    gaps = np.diff(t)
    return float(np.mean(gaps)), float(np.var(gaps, ddof=0))


def _time_in_each_regime(
    regime_times: np.ndarray,
    regime_values: np.ndarray,
    horizon: float,
    num_regimes: int,
) -> np.ndarray:
    """Return a length-`num_regimes` array of total sojourn time per regime.

    `regime_times[0] == 0` by construction. For each i the simulator was in
    `regime_values[i]` on the interval [regime_times[i], regime_times[i+1]);
    the trailing interval is [regime_times[-1], horizon]. Times beyond the
    horizon are clipped.
    """
    k = int(num_regimes)
    totals = np.zeros(k, dtype=np.float64)
    rt = np.asarray(regime_times, dtype=np.float64)
    rv = np.asarray(regime_values, dtype=np.int64)
    n = rt.shape[0]
    if n == 0:
        return totals
    H = float(horizon)
    for i in range(n):
        t0 = float(rt[i])
        t1 = float(rt[i + 1]) if i + 1 < n else H
        t0 = max(0.0, min(H, t0))
        t1 = max(0.0, min(H, t1))
        dur = max(0.0, t1 - t0)
        z = int(rv[i])
        if 0 <= z < k:
            totals[z] += dur
    return totals


def _time_in_each_window(windows: List[dict], horizon: float) -> List[float]:
    """Return per-window sojourn time clipped to [0, horizon].

    Assumes windows are non-overlapping (enforced by validate_meta_order).
    """
    H = float(horizon)
    out: List[float] = []
    for w in windows:
        start = max(0.0, float(w["start_time"]))
        end = min(H, float(w["end_time"]))
        out.append(max(0.0, end - start))
    return out


# ---------------------------------------------------------------------------
# Per-section builders
# ---------------------------------------------------------------------------


def _latent_block(run: LoadedRun) -> Dict[str, Any]:
    cfg = run.config
    d = int(cfg.dimension)
    horizon = float(cfg.horizon)
    times = np.asarray(run.times, dtype=np.float64)
    types = np.asarray(run.event_types, dtype=np.int64)
    num_events = int(times.shape[0])
    counts = _counts_by_event_type(types, d)
    rates = (
        [float(c) / horizon for c in counts] if horizon > 0.0 else [0.0] * d
    )
    md = run.metadata
    return {
        "num_events": num_events,
        "counts_by_event_type": counts,
        "rates_by_event_type": rates,
        "total_rate": (float(num_events) / horizon) if horizon > 0.0 else 0.0,
        "acceptance_ratio": float(md.get("acceptance_ratio", 0.0)),
        "num_proposals": int(md.get("num_proposals", 0)),
        "num_acceptances": int(md.get("num_acceptances", 0)),
    }


def _regime_block(run: LoadedRun) -> Dict[str, Any]:
    cfg = run.config
    k = int(cfg.latent_regime.num_regimes)
    horizon = float(cfg.horizon)
    totals = _time_in_each_regime(
        run.regime_times, run.regime_values, horizon, k
    )
    if horizon > 0.0:
        frac = (totals / horizon).tolist()
    else:
        frac = [0.0] * k
    num_jumps = int(max(0, len(np.asarray(run.regime_times)) - 1))
    return {
        "num_regimes": k,
        "num_regime_jumps": num_jumps,
        "time_in_regime": [float(x) for x in totals],
        "time_fraction_by_regime": [float(x) for x in frac],
    }


def _meta_order_window_records(run: LoadedRun) -> List[dict]:
    """Return the window records for this run.

    Prefer the list already serialised into metadata. Fall back to the in
    memory SimulatorConfig.meta_order.windows tuple when the metadata list is
    empty or absent, so that summaries remain faithful even for runs that
    somehow lack the metadata block.
    """
    md_windows = run.metadata.get("meta_order_windows", []) or []
    if md_windows:
        return list(md_windows)
    reconstructed: List[dict] = []
    for w in run.config.meta_order.windows:
        reconstructed.append(
            {
                "start_time": float(w.start_time),
                "end_time": float(w.end_time),
                "alpha": float(w.alpha),
                "target_event_types": [int(i) for i in w.target_event_types],
            }
        )
    return reconstructed


def _meta_order_block(run: LoadedRun) -> Dict[str, Any]:
    horizon = float(run.config.horizon)
    windows = _meta_order_window_records(run)
    times_in = _time_in_each_window(windows, horizon)
    total_in = float(sum(times_in))
    frac_by = (
        [float(t) / horizon for t in times_in]
        if horizon > 0.0
        else [0.0 for _ in times_in]
    )
    return {
        "num_windows": int(len(windows)),
        "time_inside_windows": total_in,
        "time_fraction_inside_windows": (
            (total_in / horizon) if horizon > 0.0 else 0.0
        ),
        "time_fraction_by_window": frac_by,
    }


def _observation_block(run: LoadedRun) -> Dict[str, Any]:
    observed = run.observed
    if observed is None:
        return {"present": False}
    d = int(run.config.dimension)
    times = np.asarray(observed.times, dtype=np.float64)
    types = np.asarray(observed.event_types, dtype=np.int64)
    counts = _counts_by_event_type(types, d)
    ia_mean: List[Optional[float]] = []
    ia_var: List[Optional[float]] = []
    for k in range(d):
        mask = types == k
        m, v = _interarrival_moments(times[mask])
        ia_mean.append(m)
        ia_var.append(v)
    final_bid = int(observed.final_bid_ticks)
    final_ask = int(observed.final_ask_ticks)
    final_mid_ticks = 0.5 * (final_bid + final_ask)
    final_spread = int(final_ask - final_bid)

    # Spread-distribution diagnostics. These summarise the entire
    # post-event spread trajectory rather than the terminal value, so they
    # are less sensitive to the random-walk drift that dominates
    # final_spread_ticks under the current observation operator.
    spread_mean_ticks: Optional[float] = None
    spread_median_ticks: Optional[float] = None
    spread_p90_ticks: Optional[float] = None
    spread_p99_ticks: Optional[float] = None
    spread_max_ticks: Optional[float] = None
    if observed.num_observed_events > 0:
        spread = np.asarray(observed.spread_ticks, dtype=np.float64)
        spread_mean_ticks = float(np.mean(spread))
        spread_median_ticks = float(np.median(spread))
        spread_p90_ticks = float(np.percentile(spread, 90))
        spread_p99_ticks = float(np.percentile(spread, 99))
        spread_max_ticks = float(np.max(spread))

    return {
        "present": True,
        "num_latent_events": int(observed.num_latent_events),
        "num_observed_events": int(observed.num_observed_events),
        "num_censored_events": int(observed.num_censored_events),
        "censoring_fraction": float(observed.censoring_fraction),
        "num_crossed_observations": int(observed.num_crossed_observations),
        "counts_by_event_type": counts,
        "interarrival_mean_by_event_type": ia_mean,
        "interarrival_variance_by_event_type": ia_var,
        "final_bid_ticks": final_bid,
        "final_ask_ticks": final_ask,
        "final_mid_ticks": float(final_mid_ticks),
        "final_spread_ticks": final_spread,
        "spread_mean_ticks": spread_mean_ticks,
        "spread_median_ticks": spread_median_ticks,
        "spread_p90_ticks": spread_p90_ticks,
        "spread_p99_ticks": spread_p99_ticks,
        "spread_max_ticks": spread_max_ticks,
        "min_spread_ticks": int(observed.min_spread_ticks),
        "crossing_policy": str(observed.crossing_policy),
    }


# ---------------------------------------------------------------------------
# Endogeneity diagnostics
# ---------------------------------------------------------------------------


def _total_baseline_rate(
    cfg, time_fraction_by_regime: List[float]
) -> Tuple[Optional[float], List[str]]:
    """Return a regime-time-weighted total baseline rate and any advisory
    notes. The rate is computed as

        sum_z p(z) * sum_i mu_i(z)

    where p(z) is the fraction of simulation time spent in regime z and
    mu_i(z) is the per-component baseline in that regime. For a single
    regime the weighting reduces to sum_i mu_i.
    """
    notes: List[str] = []
    mu = np.asarray(cfg.baseline.mu, dtype=np.float64)
    if mu.ndim == 1:
        return float(np.sum(mu)), notes
    if mu.ndim == 2:
        fracs = np.asarray(time_fraction_by_regime, dtype=np.float64)
        per_regime_sums = mu.sum(axis=1)
        if fracs.shape[0] != per_regime_sums.shape[0]:
            notes.append(
                "total_baseline_rate not computed: regime time-fraction "
                f"length {fracs.shape[0]} does not match baseline.mu "
                f"leading dimension {per_regime_sums.shape[0]}"
            )
            return None, notes
        return float(np.sum(fracs * per_regime_sums)), notes
    notes.append(
        f"total_baseline_rate not computed: baseline.mu has unsupported "
        f"ndim={mu.ndim}"
    )
    return None, notes


def _endogeneity_block(
    run: LoadedRun,
    regime_block: Dict[str, Any],
    latent_block: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the endogeneity diagnostics for a run.

    The block contains:
        kernel_family         canonical name of the kernel family
        spectral_radius       rho(Phi) where Phi is the branching matrix;
                              null when the kernel family is not supported
        total_baseline_rate   regime-time-weighted baseline rate sum
        rate_based_proxy      1 - total_baseline_rate / latent.total_rate;
                              null when the latent total rate is zero or the
                              baseline rate could not be computed
        notes                 list of strings explaining any null values
    """
    cfg = run.config
    notes: List[str] = []

    kernel_family_name = cfg.kernel_family.name

    spectral_radius_value: Optional[float] = None
    try:
        kernel = build_kernel(cfg)
        spectral_radius_value = float(_kernel_spectral_radius(kernel))
    except (NotImplementedError, AttributeError, TypeError) as exc:
        notes.append(
            f"spectral_radius not computed for kernel family "
            f"{kernel_family_name}: {exc}"
        )

    total_baseline_rate, rate_notes = _total_baseline_rate(
        cfg, regime_block.get("time_fraction_by_regime", [])
    )
    notes.extend(rate_notes)

    total_rate = float(latent_block.get("total_rate", 0.0))
    rate_based_proxy: Optional[float] = None
    if total_baseline_rate is None:
        notes.append(
            "rate_based_proxy not computed: total_baseline_rate is null"
        )
    elif total_rate <= 0.0:
        notes.append(
            "rate_based_proxy not computed: latent total rate is zero"
        )
    else:
        rate_based_proxy = 1.0 - float(total_baseline_rate) / total_rate

    return {
        "kernel_family": kernel_family_name,
        "spectral_radius": spectral_radius_value,
        "total_baseline_rate": (
            float(total_baseline_rate)
            if total_baseline_rate is not None
            else None
        ),
        "rate_based_proxy": rate_based_proxy,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def summarise(run: LoadedRun) -> Dict[str, Any]:
    """Return the summary dict for a LoadedRun.

    The output is JSON-serialisable without modification. All fields are
    deterministic functions of the input.
    """
    cfg = run.config
    latent = _latent_block(run)
    regime = _regime_block(run)
    meta_order = _meta_order_block(run)
    observation = _observation_block(run)
    endogeneity = _endogeneity_block(run, regime, latent)
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "horizon": float(cfg.horizon),
        "seed": int(cfg.seed),
        "dimension": int(cfg.dimension),
        "latent": latent,
        "regime": regime,
        "meta_order": meta_order,
        "observation": observation,
        "endogeneity": endogeneity,
    }


def save_summary(run_dir: str) -> Path:
    """Compute summary.json for the run at run_dir and write it into that
    directory. Returns the absolute Path of the written file.

    The run artefact is re-loaded from disk via load_run; callers that
    already hold a LoadedRun can use summarise(run) directly.
    """
    run = load_run(run_dir)
    summary = summarise(run)
    path = Path(run_dir) / SUMMARY_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    return path.resolve()
