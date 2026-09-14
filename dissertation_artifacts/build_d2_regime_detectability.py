"""Build artefacts:
    dissertation_artifacts/data/d2_regime_detectability_sweep.json
    dissertation_artifacts/figures/d2_regime_detectability_sweep.pdf

30-cell regime-detectability sweep on the prompt-07b D2 latent
full-book, with 07b calibration AND HMM configuration FROZEN.

Per the prompt-07c spec, this sweep disambiguates three mechanisms
underlying 07b's `F1_observed = 0.291`:

    (1) weak regime design;
    (2) HMM transferability failure;
    (3) true censoring cost.

Sweep dimensions (5 x 3 x 2 = 30 cells; full grid mandated):
    intensity_multiplier in {2.0, 3.0, 5.0, 8.0, 10.0}
    duty cycle in {0.20, 0.33, 0.50}
    affected_channels in {DEFAULT, BROADER}

For each cell:
    - construct D2LatentBookConfig with FROZEN 07b calibration
      AND the cell's regime parameters;
    - run simulate_d2_latent_book(config, T=1800, seed=1);
    - run HMM Layer-2 on the LATENT stream -> F1_latent;
    - run HMM Layer-2 on the OBSERVED stream -> F1_observed;
    - record per-cell censoring_cost = F1_observed - F1_latent.

Threshold = lowest-intensity cell with F1_latent >= 0.7. If found,
the promote-07b clause re-runs 07b's headline at the threshold cell
and updates `d2_latent_book_calibration_summary.json` in place.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_PATH = str(REPO_ROOT / "src")
INPUT_07B_JSON = HERE / "data" / "d2_latent_book_calibration_summary.json"
OUT_JSON = HERE / "data" / "d2_regime_detectability_sweep.json"
OUT_PDF = HERE / "figures" / "d2_regime_detectability_sweep.pdf"

T_SIM_SECONDS = 1800.0
SIM_SEED = 1
SEED_REGIME = 42
WINDOW_WIDTH_SECONDS = 30.0
HMM_F1_THRESHOLD = 0.70
F1_VARIATION_FLOOR_FOR_CONFIG_INCOMPLETE = 0.05
WINDOW_SECONDS_FOR_HMM = 5.0

INTENSITY_MULTIPLIERS: Tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 10.0)
DUTY_CYCLES: Tuple[float, ...] = (0.20, 0.33, 0.50)
CHANNEL_SETS: Tuple[str, ...] = ("DEFAULT", "BROADER")


def _ensure_simulator_on_path() -> None:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


# ---------------------------------------------------------------------------
# Step 1 — load + verify FROZEN 07b inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenInputs:
    spread0: float
    seed: int
    use_exp_approx: bool
    inspread_bid_over_ask_mu_ratio: float
    per_native_type_mu_multipliers: Tuple[Tuple[str, float], ...]
    hmm_default_recovery_method: str
    hmm_n_states: int
    hmm_emission_features: Tuple[str, ...]
    runtime_patch_targets: Tuple[str, ...]


def load_frozen_inputs() -> FrozenInputs:
    """Load 07b winning_candidate + dissertation HMM default + runtime
    patches. Anything mismatched is a 07C BLOCKED_RUNTIME signal."""
    if not INPUT_07B_JSON.exists():
        raise FileNotFoundError(
            f"07b artefact not found at {INPUT_07B_JSON}; cannot freeze inputs."
        )
    d = json.loads(INPUT_07B_JSON.read_text())
    wc = d["winning_candidate"]["parameters"]
    spread0 = float(wc["spread0"])
    seed = int(wc["seed"])
    use_exp_approx = bool(wc["use_exp_approx"])
    inspread = float(wc["inspread_bid_over_ask_mu_ratio"])
    multipliers = tuple(
        (str(name), float(mult))
        for name, mult in wc["per_native_type_mu_multipliers"]
    )
    # HMM dissertation default (load + verify)
    _ensure_simulator_on_path()
    from simulator.experiment import ExperimentConfig
    from simulator.recovery_hmm import HMMRecoveryConfig
    from simulator.d2_konark_latent_book import KONARK_RUNTIME_PATCHES

    exp_cfg = ExperimentConfig(
        config_path="<unused>",
        seeds=(0,),
        output_root="<unused>",
    )
    if exp_cfg.recovery_method != "hmm":
        raise RuntimeError(
            f"ExperimentConfig.recovery_method == {exp_cfg.recovery_method!r}; "
            "must be 'hmm' (07c frozen-input check)."
        )
    hmm_default = HMMRecoveryConfig()
    if hmm_default.n_states != 2:
        raise RuntimeError(
            f"HMMRecoveryConfig n_states = {hmm_default.n_states}; "
            "dissertation default is 2."
        )
    if hmm_default.emission_features != (
        "recent_event_rate",
        "directional_imbalance",
    ):
        raise RuntimeError(
            f"HMMRecoveryConfig emission_features = "
            f"{hmm_default.emission_features}; dissertation default mismatched."
        )
    patch_targets = tuple(p["target"] for p in KONARK_RUNTIME_PATCHES)
    if not any("thinningOgataIS2" in t and "335" in t for t in patch_targets):
        raise RuntimeError(
            "Patch 1 (numpy-2.x scalar conversion) not registered in "
            "KONARK_RUNTIME_PATCHES."
        )
    if not any("truncation" in t.lower() or "left" in t.lower() for t in patch_targets):
        raise RuntimeError(
            "Patch 2 (self.left reset post-truncation) not registered in "
            "KONARK_RUNTIME_PATCHES."
        )
    return FrozenInputs(
        spread0=spread0,
        seed=seed,
        use_exp_approx=use_exp_approx,
        inspread_bid_over_ask_mu_ratio=inspread,
        per_native_type_mu_multipliers=multipliers,
        hmm_default_recovery_method=str(exp_cfg.recovery_method),
        hmm_n_states=int(hmm_default.n_states),
        hmm_emission_features=tuple(hmm_default.emission_features),
        runtime_patch_targets=patch_targets,
    )


# ---------------------------------------------------------------------------
# Step 2 — build the 30-cell grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCell:
    cell_id: int
    intensity_multiplier: float
    duty_cycle: float
    channel_set: str  # "DEFAULT" or "BROADER"
    period_seconds: float
    affected_channels_template: Tuple[str, ...]


def build_sweep_grid() -> Tuple[SweepCell, ...]:
    cells: List[SweepCell] = []
    cell_id = 0
    for intensity in INTENSITY_MULTIPLIERS:
        for duty in DUTY_CYCLES:
            for cs in CHANNEL_SETS:
                cell_id += 1
                period = WINDOW_WIDTH_SECONDS / float(duty)
                if cs == "DEFAULT":
                    template = ("mo_<side>", "lo_inspread_<side>")
                else:
                    template = (
                        "mo_<side>",
                        "lo_inspread_<side>",
                        "lo_top_<side>",
                        "co_top_<side>",
                    )
                cells.append(
                    SweepCell(
                        cell_id=cell_id,
                        intensity_multiplier=float(intensity),
                        duty_cycle=float(duty),
                        channel_set=str(cs),
                        period_seconds=float(period),
                        affected_channels_template=template,
                    )
                )
    assert len(cells) == 30, f"expected 30 cells, got {len(cells)}"
    return tuple(cells)


def cell_meta_order_windows(cell: SweepCell, horizon: float):
    """Build the MetaOrderRegimeWindow tuple for one cell, alternating
    Bid/Ask starting with Bid, with 30s width per the cell's period."""
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import MetaOrderRegimeWindow

    rng = np.random.default_rng(SEED_REGIME)  # reserved for future placement noise
    _ = rng
    windows = []
    wid = 0
    start = 0.0
    while start + WINDOW_WIDTH_SECONDS <= horizon:
        wid += 1
        side = "Bid" if (wid % 2 == 1) else "Ask"
        affected = tuple(
            t.replace("<side>", side) for t in cell.affected_channels_template
        )
        windows.append(
            MetaOrderRegimeWindow(
                window_id=wid,
                start_t=float(start),
                end_t=float(start + WINDOW_WIDTH_SECONDS),
                direction=side,
                intensity_multiplier=float(cell.intensity_multiplier),
                affected_native_types=affected,
            )
        )
        start += cell.period_seconds
    return tuple(windows)


def build_d2_config_for_cell(cell: SweepCell, frozen: FrozenInputs, horizon: float):
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import D2LatentBookConfig

    return D2LatentBookConfig(
        spread0=frozen.spread0,
        seed=frozen.seed,
        use_exp_approx=frozen.use_exp_approx,
        inspread_bid_over_ask_mu_ratio=frozen.inspread_bid_over_ask_mu_ratio,
        per_native_type_mu_multipliers=frozen.per_native_type_mu_multipliers,
        meta_order_regime_windows=cell_meta_order_windows(cell, horizon),
    )


# ---------------------------------------------------------------------------
# Step 3 — per-cell HMM evaluation on latent + observed
# ---------------------------------------------------------------------------


def _build_observed_trace_from_d2(d2_result, *, observed_only: bool):
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import KONARK_TYPE_TO_INDEX, FX_TOPOFBOOK_TYPES, PHI_BETA_PROJECTION
    from simulator.observation import ObservedTrace

    if observed_only:
        n = int(d2_result.n_observed)
        type_to_idx = {t: i for i, t in enumerate(FX_TOPOFBOOK_TYPES)}
        times = np.asarray(
            [ev.time for ev in d2_result.observed_events], dtype=np.float64
        )
        types = np.asarray(
            [type_to_idx[ev.observed_type] for ev in d2_result.observed_events],
            dtype=np.int64,
        )
    else:
        # Latent diagnostic stream: full 12-type Konark events mapped onto FX
        # alphabet (deep events folded in same-side as inspread for diagnostic
        # only; this is NOT the production projection — it's the uncensored
        # comparison stream the spec asks for in Step 3c).
        latent_fx_map = {
            "lo_top_Bid": 0, "co_top_Bid": 1, "lo_top_Ask": 3, "co_top_Ask": 2,
            "mo_Ask": 2, "mo_Bid": 1,
            "lo_inspread_Bid": 0, "lo_inspread_Ask": 3,
            "lo_deep_Bid": 0, "co_deep_Bid": 1,
            "lo_deep_Ask": 3, "co_deep_Ask": 2,
        }
        times = np.asarray(
            [ev.time for ev in d2_result.latent_events], dtype=np.float64
        )
        types = np.asarray(
            [latent_fx_map[ev.konark_native_type] for ev in d2_result.latent_events],
            dtype=np.int64,
        )
        n = int(times.size)
    return ObservedTrace(
        times=times,
        event_types=types,
        bid_ticks=np.zeros(n, dtype=np.int64),
        ask_ticks=np.ones(n, dtype=np.int64),
        spread_ticks=np.ones(n, dtype=np.int64),
        initial_bid_ticks=0,
        initial_ask_ticks=1,
        min_spread_ticks=1,
        crossing_policy="censor",
        tick_size=1e-5,
        num_latent_events=int(d2_result.n_latent),
        num_censored_events=int(d2_result.n_latent - int(d2_result.n_observed)),
    )


def _viterbi_to_intervals(viterbi, window_starts, window_seconds, active_state):
    intervals = []
    if viterbi.size == 0:
        return intervals
    is_active = (viterbi == active_state).astype(np.int8)
    diff = np.diff(is_active, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        intervals.append(
            (
                float(window_starts[s]),
                float(window_starts[e - 1] + window_seconds),
                1.0,
            )
        )
    return intervals


def hmm_evaluate(
    d2_result, *, observed_only: bool, horizon: float, seed: int
) -> dict:
    _ensure_simulator_on_path()
    from simulator.config import MetaOrderWindow
    from simulator.recovery import compute_window_features
    from simulator.recovery_eval import evaluate_meta_order_recovery
    from simulator.recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm

    obs = _build_observed_trace_from_d2(d2_result, observed_only=observed_only)
    features = compute_window_features(
        obs,
        window_seconds=WINDOW_SECONDS_FOR_HMM,
        hop_seconds=WINDOW_SECONDS_FOR_HMM,
        horizon=horizon,
    )
    if len(features) < 4:
        return {
            "available": False,
            "reason": "fewer than 4 windows",
            "f1": None,
            "precision": None,
            "recall": None,
        }
    cfg = HMMRecoveryConfig(
        n_states=2,
        emission_features=("recent_event_rate", "directional_imbalance"),
        max_iter=200,
        tol_relative_ll=1e-6,
        seed=int(seed),
    )
    res = recover_regimes_hmm(features, cfg)
    active_state = int(np.argmax(res.emission_means[:, 0]))
    intervals = _viterbi_to_intervals(
        res.viterbi_labels, res.window_starts, WINDOW_SECONDS_FOR_HMM, active_state
    )
    true_windows = [
        MetaOrderWindow(
            start_time=float(w.start_t),
            end_time=float(w.end_t),
            alpha=float(w.intensity_multiplier - 1.0),
            target_event_types=tuple(range(4)),
        )
        for w in d2_result.meta_order_regime_windows
    ]
    eval_ = evaluate_meta_order_recovery(
        true_windows=true_windows,
        inferred_intervals=intervals,
        horizon=horizon,
        time_resolution=0.1,
    )
    return {
        "available": True,
        "stream": "observed" if observed_only else "latent",
        "f1": float(eval_.f1),
        "precision": float(eval_.precision),
        "recall": float(eval_.recall),
        "n_inferred_intervals": int(len(intervals)),
        "n_true_windows": int(len(true_windows)),
        "active_state_index": int(active_state),
        "hmm_converged": bool(res.converged),
        "hmm_n_iter": int(res.n_iter),
    }


def run_one_cell(cell: SweepCell, frozen: FrozenInputs) -> dict:
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import simulate_d2_latent_book

    cfg = build_d2_config_for_cell(cell, frozen, T_SIM_SECONDS)
    t0 = time.time()
    d2_result = simulate_d2_latent_book(cfg, horizon_seconds=T_SIM_SECONDS)
    sim_wall = time.time() - t0

    t1 = time.time()
    hmm_latent = hmm_evaluate(
        d2_result,
        observed_only=False,
        horizon=T_SIM_SECONDS,
        seed=SIM_SEED,
    )
    hmm_lat_wall = time.time() - t1

    t2 = time.time()
    hmm_observed = hmm_evaluate(
        d2_result,
        observed_only=True,
        horizon=T_SIM_SECONDS,
        seed=SIM_SEED,
    )
    hmm_obs_wall = time.time() - t2

    f1_latent = hmm_latent.get("f1")
    f1_observed = hmm_observed.get("f1")
    cens_cost = (
        float(f1_observed) - float(f1_latent)
        if (f1_latent is not None and f1_observed is not None)
        else None
    )
    return {
        "cell_id": int(cell.cell_id),
        "intensity_multiplier": float(cell.intensity_multiplier),
        "duty_cycle": float(cell.duty_cycle),
        "channel_set": str(cell.channel_set),
        "period_seconds": float(cell.period_seconds),
        "affected_channels_template": list(cell.affected_channels_template),
        "n_latent_events": int(d2_result.n_latent),
        "n_observed_events": int(d2_result.n_observed),
        "censoring_fraction": float(d2_result.censoring_fraction),
        "n_meta_order_windows": int(len(d2_result.meta_order_regime_windows)),
        "f1_latent": f1_latent,
        "precision_latent": hmm_latent.get("precision"),
        "recall_latent": hmm_latent.get("recall"),
        "f1_observed": f1_observed,
        "precision_observed": hmm_observed.get("precision"),
        "recall_observed": hmm_observed.get("recall"),
        "censoring_cost_cell": cens_cost,
        "hmm_latent_converged": bool(hmm_latent.get("hmm_converged", False)),
        "hmm_observed_converged": bool(hmm_observed.get("hmm_converged", False)),
        "wall_seconds_total": float(sim_wall + hmm_lat_wall + hmm_obs_wall),
        "wall_seconds_breakdown": {
            "simulate": float(sim_wall),
            "hmm_latent": float(hmm_lat_wall),
            "hmm_observed": float(hmm_obs_wall),
        },
    }


# ---------------------------------------------------------------------------
# Step 4 — threshold-finding logic
# ---------------------------------------------------------------------------


def find_threshold_cell(cell_results: List[dict]) -> Optional[dict]:
    """Lowest-intensity cell with F1_latent >= 0.7, breaking ties by
    smallest duty cycle, then smallest channel set (DEFAULT < BROADER),
    then smallest |F1_latent - 0.7| margin."""
    qualifying = [
        c for c in cell_results
        if c.get("f1_latent") is not None and c["f1_latent"] >= HMM_F1_THRESHOLD
    ]
    if not qualifying:
        return None
    channel_rank = {"DEFAULT": 0, "BROADER": 1}
    return sorted(
        qualifying,
        key=lambda c: (
            c["intensity_multiplier"],
            c["duty_cycle"],
            channel_rank.get(c["channel_set"], 99),
            abs(c["f1_latent"] - HMM_F1_THRESHOLD),
        ),
    )[0]


def sensitivity_neighbourhood(
    threshold: dict, cell_results: List[dict]
) -> dict:
    """Cells within +/- 1 grid step on each dimension of the threshold."""
    intensities = list(INTENSITY_MULTIPLIERS)
    duties = list(DUTY_CYCLES)
    channels = list(CHANNEL_SETS)
    int_idx = intensities.index(threshold["intensity_multiplier"])
    duty_idx = duties.index(threshold["duty_cycle"])
    chan_idx = channels.index(threshold["channel_set"])
    int_band = intensities[max(0, int_idx - 1) : min(len(intensities), int_idx + 2)]
    duty_band = duties[max(0, duty_idx - 1) : min(len(duties), duty_idx + 2)]
    chan_band = channels[max(0, chan_idx - 1) : min(len(channels), chan_idx + 2)]
    nbhd = [
        c
        for c in cell_results
        if c["intensity_multiplier"] in int_band
        and c["duty_cycle"] in duty_band
        and c["channel_set"] in chan_band
    ]
    f1_lat = [c["f1_latent"] for c in nbhd if c.get("f1_latent") is not None]
    f1_obs = [c["f1_observed"] for c in nbhd if c.get("f1_observed") is not None]
    return {
        "neighbourhood_cell_ids": [int(c["cell_id"]) for c in nbhd],
        "n_cells": int(len(nbhd)),
        "f1_latent_mean": float(np.mean(f1_lat)) if f1_lat else None,
        "f1_latent_std": float(np.std(f1_lat)) if f1_lat else None,
        "f1_observed_mean": float(np.mean(f1_obs)) if f1_obs else None,
        "f1_observed_std": float(np.std(f1_obs)) if f1_obs else None,
    }


# ---------------------------------------------------------------------------
# Step 5 — classification
# ---------------------------------------------------------------------------


def classify(cell_results: List[dict], threshold: Optional[dict]) -> Tuple[str, str]:
    f1s = [
        c["f1_latent"]
        for c in cell_results
        if c.get("f1_latent") is not None
    ]
    if not f1s:
        return ("07C_BLOCKED_RUNTIME", "no F1_latent values produced for any cell")
    f1_max = max(f1s)
    f1_min = min(f1s)
    f1_var = f1_max - f1_min
    if threshold is not None:
        return (
            "07C_COMPLETE",
            f"threshold cell {threshold['cell_id']} achieves "
            f"F1_latent={threshold['f1_latent']:.3f} >= {HMM_F1_THRESHOLD}",
        )
    if f1_var > F1_VARIATION_FLOOR_FOR_CONFIG_INCOMPLETE:
        return (
            "07C_PARTIAL",
            f"max F1_latent = {f1_max:.3f} < {HMM_F1_THRESHOLD}; "
            f"variation across grid = {f1_var:.3f} > "
            f"{F1_VARIATION_FLOOR_FOR_CONFIG_INCOMPLETE} -> HMM transferability finding",
        )
    return (
        "07C_CONFIGURATION_INCOMPLETE",
        f"max F1_latent = {f1_max:.3f}; variation across grid = {f1_var:.3f} <= "
        f"{F1_VARIATION_FLOOR_FOR_CONFIG_INCOMPLETE} -> regime knobs not moving "
        "the latent process; surface for spec amendment",
    )


# ---------------------------------------------------------------------------
# Step 6 — promote-07b clause
# ---------------------------------------------------------------------------


def _fit_layer1_em(times, types, T) -> dict:
    _ensure_simulator_on_path()
    from simulator.layer1_em import Layer1EMConfig, fit_layer1_em

    cfg = Layer1EMConfig(betas=(2.0,), max_iter=200, tol_relative_ll=1.0e-5)
    t0 = time.time()
    res = fit_layer1_em(times=times, types=types, T=T, M=4, config=cfg)
    return {
        "rho_spec": float(res.spectral_radius),
        "log_likelihood": float(res.log_likelihood),
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        "wall_seconds": float(time.time() - t0),
    }


def promote_07b(threshold_cell: dict, frozen: FrozenInputs) -> dict:
    """Re-run 07b headline at the threshold cell's regime configuration.
    Update the 07b artefact in place."""
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import (
        FX_TOPOFBOOK_TYPES,
        simulate_d2_latent_book,
    )

    # Reconstruct cell + config
    cell = SweepCell(
        cell_id=int(threshold_cell["cell_id"]),
        intensity_multiplier=float(threshold_cell["intensity_multiplier"]),
        duty_cycle=float(threshold_cell["duty_cycle"]),
        channel_set=str(threshold_cell["channel_set"]),
        period_seconds=float(threshold_cell["period_seconds"]),
        affected_channels_template=tuple(threshold_cell["affected_channels_template"]),
    )
    cfg = build_d2_config_for_cell(cell, frozen, T_SIM_SECONDS)
    t0 = time.time()
    d2 = simulate_d2_latent_book(cfg, horizon_seconds=T_SIM_SECONDS)
    print(
        f"  [promote-07b] re-ran D2 at threshold cell {cell.cell_id}: "
        f"n_latent={d2.n_latent}, n_observed={d2.n_observed}"
    )
    # Layer-1 EM on observed
    type_to_idx = {t: i for i, t in enumerate(FX_TOPOFBOOK_TYPES)}
    times_obs = np.asarray([ev.time for ev in d2.observed_events], dtype=np.float64)
    types_obs = np.asarray(
        [type_to_idx[ev.observed_type] for ev in d2.observed_events], dtype=np.int64
    )
    em_d2 = _fit_layer1_em(times_obs, types_obs, T_SIM_SECONDS)
    # HMM on observed AND latent at threshold cell
    hmm_obs = hmm_evaluate(d2, observed_only=True, horizon=T_SIM_SECONDS, seed=SIM_SEED)
    hmm_lat = hmm_evaluate(d2, observed_only=False, horizon=T_SIM_SECONDS, seed=SIM_SEED)
    cens_cost = hmm_obs["f1"] - hmm_lat["f1"]
    # Promote 07b classification
    if hmm_obs["f1"] >= HMM_F1_THRESHOLD:
        new_07b = "07B_COMPLETE"
    else:
        new_07b = f"07B_PARTIAL_with_quantified_censoring_cost={cens_cost:+.3f}"
    headline = {
        "threshold_cell_id": int(cell.cell_id),
        "threshold_cell_parameters": {
            "intensity_multiplier": float(cell.intensity_multiplier),
            "duty_cycle": float(cell.duty_cycle),
            "channel_set": str(cell.channel_set),
            "period_seconds": float(cell.period_seconds),
            "affected_channels_template": list(cell.affected_channels_template),
        },
        "n_latent_events": int(d2.n_latent),
        "n_observed_events": int(d2.n_observed),
        "censoring_fraction": float(d2.censoring_fraction),
        "rho_d2_observed_at_threshold": em_d2["rho_spec"],
        "log_likelihood_d2_at_threshold": em_d2["log_likelihood"],
        "n_iter_em_at_threshold": em_d2["n_iter"],
        "converged_em_at_threshold": em_d2["converged"],
        "f1_observed_at_threshold": float(hmm_obs["f1"]),
        "f1_latent_at_threshold": float(hmm_lat["f1"]),
        "precision_observed_at_threshold": float(hmm_obs["precision"]),
        "recall_observed_at_threshold": float(hmm_obs["recall"]),
        "censoring_cost_at_threshold": float(cens_cost),
        "new_07b_classification": new_07b,
        "wall_seconds": float(time.time() - t0),
    }

    # Mutate the 07b artefact in place
    d = json.loads(INPUT_07B_JSON.read_text())
    # Preserve original under default_regime_configuration
    if "default_regime_configuration" not in d:
        d["default_regime_configuration"] = {
            "production_run": d.get("production_run"),
            "layer1_em_observed_d2": d.get("layer1_em_observed_d2"),
            "three_way_rho_table": d.get("three_way_rho_table"),
            "hmm_layer2_observed_d2": d.get("hmm_layer2_observed_d2"),
            "hmm_layer2_latent_d2": d.get("hmm_layer2_latent_d2"),
            "censoring_cost_f1": d.get("censoring_cost_f1"),
            "classification": d.get("classification"),
            "note": (
                "Original 07b headline at the default regime configuration; "
                "preserved here when 07c COMPLETE promoted the headline to "
                "the threshold-cell regime configuration. Original "
                "F1_observed = 0.291; F1_latent = 0.254."
            ),
        }
    d["threshold_cell_regime"] = {
        "cell_id": int(cell.cell_id),
        "parameters": headline["threshold_cell_parameters"],
        "selected_via": "07c sweep COMPLETE classification",
    }
    d["headline_at_threshold"] = headline
    d["classification_after_07c_promotion"] = new_07b
    INPUT_07B_JSON.write_text(json.dumps(d, indent=2))
    print(
        f"  [promote-07b] updated {INPUT_07B_JSON.name}: "
        f"new 07b status = {new_07b}"
    )
    return headline


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(cell_results: List[dict], threshold: Optional[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: F1_latent vs intensity, one curve per (duty, channel)
    ax = axes[0]
    for duty in DUTY_CYCLES:
        for cs in CHANNEL_SETS:
            xs = []
            ys = []
            for c in cell_results:
                if c["duty_cycle"] == duty and c["channel_set"] == cs:
                    xs.append(c["intensity_multiplier"])
                    ys.append(c["f1_latent"])
            order = np.argsort(xs)
            ax.plot(
                np.array(xs)[order],
                np.array(ys)[order],
                marker="o",
                label=f"duty={duty:.2f}, {cs}",
                linewidth=1.4,
            )
    ax.axhline(HMM_F1_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.5,
               label=f"F1 = {HMM_F1_THRESHOLD} threshold")
    ax.set_xlabel("intensity_multiplier")
    ax.set_ylabel(r"F1$_\mathrm{latent}$")
    ax.set_title("F1 (latent) by regime parameters")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Panel 2: F1_observed vs intensity
    ax = axes[1]
    for duty in DUTY_CYCLES:
        for cs in CHANNEL_SETS:
            xs = []
            ys = []
            for c in cell_results:
                if c["duty_cycle"] == duty and c["channel_set"] == cs:
                    xs.append(c["intensity_multiplier"])
                    ys.append(c["f1_observed"])
            order = np.argsort(xs)
            ax.plot(
                np.array(xs)[order],
                np.array(ys)[order],
                marker="s",
                label=f"duty={duty:.2f}, {cs}",
                linewidth=1.4,
            )
    ax.axhline(HMM_F1_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.5,
               label=f"F1 = {HMM_F1_THRESHOLD} threshold")
    ax.set_xlabel("intensity_multiplier")
    ax.set_ylabel(r"F1$_\mathrm{observed}$")
    ax.set_title("F1 (observed) by regime parameters")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Panel 3: censoring-cost heatmap (intensity x duty), DEFAULT only
    # plus annotation of BROADER cell deltas
    ax = axes[2]
    H = np.zeros((len(INTENSITY_MULTIPLIERS), len(DUTY_CYCLES)))
    for i, intensity in enumerate(INTENSITY_MULTIPLIERS):
        for j, duty in enumerate(DUTY_CYCLES):
            cell = next(
                (c for c in cell_results
                 if c["intensity_multiplier"] == intensity
                 and c["duty_cycle"] == duty
                 and c["channel_set"] == "DEFAULT"),
                None,
            )
            H[i, j] = cell["censoring_cost_cell"] if cell else 0.0
    vmax = max(0.05, np.max(np.abs(H)))
    im = ax.imshow(
        H, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xticks(range(len(DUTY_CYCLES)))
    ax.set_xticklabels([f"{d:.2f}" for d in DUTY_CYCLES])
    ax.set_yticks(range(len(INTENSITY_MULTIPLIERS)))
    ax.set_yticklabels([f"{m:.1f}" for m in INTENSITY_MULTIPLIERS])
    ax.set_xlabel("duty cycle")
    ax.set_ylabel("intensity_multiplier")
    ax.set_title("Censoring cost (F1_obs - F1_lat), DEFAULT channels")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i,j]:+.2f}",
                    ha="center", va="center",
                    color=("black" if abs(H[i,j]) < vmax * 0.5 else "white"),
                    fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title = "07c regime detectability sweep on D2 latent full-book"
    if threshold:
        title += (
            f"  (threshold: cell {threshold['cell_id']}, "
            f"intensity={threshold['intensity_multiplier']:.0f}, "
            f"duty={threshold['duty_cycle']:.2f}, {threshold['channel_set']})"
        )
    else:
        max_f1 = max(c["f1_latent"] for c in cell_results if c.get("f1_latent") is not None)
        title += f"  (no threshold; max F1_latent = {max_f1:.3f})"
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _ensure_simulator_on_path()
    print("=== 07c regime detectability sweep ===")

    print("\n[Step 1] Loading and verifying frozen 07b inputs ...")
    frozen = load_frozen_inputs()
    print(f"  spread0 = {frozen.spread0}")
    print(f"  seed = {frozen.seed}")
    print(f"  use_exp_approx = {frozen.use_exp_approx}")
    print(f"  inspread_bid_over_ask_mu_ratio = {frozen.inspread_bid_over_ask_mu_ratio}")
    print(f"  per_native_type_mu_multipliers = {frozen.per_native_type_mu_multipliers}")
    print(f"  HMM default recovery_method = {frozen.hmm_default_recovery_method}")
    print(f"  HMM default n_states = {frozen.hmm_n_states}")
    print(f"  HMM default emission_features = {frozen.hmm_emission_features}")
    print(f"  runtime patch targets = {frozen.runtime_patch_targets}")

    print("\n[Step 2] Building 30-cell sweep grid ...")
    cells = build_sweep_grid()
    print(f"  built {len(cells)} cells")

    print("\n[Step 3] Per-cell execution ...")
    cell_results: List[dict] = []
    t_global = time.time()
    for cell in cells:
        rec = run_one_cell(cell, frozen)
        cell_results.append(rec)
        print(
            f"  cell {rec['cell_id']:>2d}/30: int={rec['intensity_multiplier']:>4.1f} "
            f"duty={rec['duty_cycle']:.2f} ch={rec['channel_set']:<7s} "
            f"n_lat={rec['n_latent_events']:>5d} n_obs={rec['n_observed_events']:>5d} "
            f"F1_lat={(rec['f1_latent'] if rec['f1_latent'] is not None else float('nan')):.3f} "
            f"F1_obs={(rec['f1_observed'] if rec['f1_observed'] is not None else float('nan')):.3f} "
            f"cost={(rec['censoring_cost_cell'] if rec['censoring_cost_cell'] is not None else float('nan')):+.3f} "
            f"wall={rec['wall_seconds_total']:>4.1f}s"
        )
    print(f"\n  total sweep wall: {time.time() - t_global:.1f}s")

    print("\n[Step 4] Threshold-finding ...")
    threshold = find_threshold_cell(cell_results)
    if threshold is not None:
        print(
            f"  threshold cell: {threshold['cell_id']} "
            f"(int={threshold['intensity_multiplier']}, duty={threshold['duty_cycle']}, "
            f"ch={threshold['channel_set']}); "
            f"F1_latent={threshold['f1_latent']:.3f}, "
            f"F1_observed={threshold['f1_observed']:.3f}, "
            f"cost={threshold['censoring_cost_cell']:+.3f}"
        )
        nbhd = sensitivity_neighbourhood(threshold, cell_results)
        print(f"  sensitivity neighbourhood: {nbhd}")
    else:
        print("  no threshold (no cell achieves F1_latent >= 0.7)")
        nbhd = None

    print("\n[Step 5] Classifying ...")
    classification, why = classify(cell_results, threshold)
    print(f"  classification: {classification}")
    print(f"  reason: {why}")

    promote_payload = None
    new_07b_status = None
    if classification == "07C_COMPLETE":
        print("\n[Step 6] Promote-07b clause firing ...")
        promote_payload = promote_07b(threshold, frozen)
        new_07b_status = promote_payload["new_07b_classification"]
    else:
        print(f"\n[Step 6] Promote-07b clause NOT firing (classification = {classification})")

    payload = {
        "schema_version": "1.0",
        "prompt": "07c_latent_regime_detectability_calibration",
        "config": {
            "T_sim_seconds": float(T_SIM_SECONDS),
            "sim_seed": int(SIM_SEED),
            "regime_placement_seed": int(SEED_REGIME),
            "window_width_seconds": float(WINDOW_WIDTH_SECONDS),
            "hmm_f1_threshold": float(HMM_F1_THRESHOLD),
            "f1_variation_floor_for_config_incomplete": float(F1_VARIATION_FLOOR_FOR_CONFIG_INCOMPLETE),
            "intensity_multipliers": list(INTENSITY_MULTIPLIERS),
            "duty_cycles": list(DUTY_CYCLES),
            "channel_sets": list(CHANNEL_SETS),
        },
        "frozen_inputs": {
            "from_07b_artefact": str(INPUT_07B_JSON.name),
            "spread0": float(frozen.spread0),
            "seed": int(frozen.seed),
            "use_exp_approx": bool(frozen.use_exp_approx),
            "inspread_bid_over_ask_mu_ratio": float(frozen.inspread_bid_over_ask_mu_ratio),
            "per_native_type_mu_multipliers": [
                [n, float(m)] for n, m in frozen.per_native_type_mu_multipliers
            ],
            "hmm_default_recovery_method": str(frozen.hmm_default_recovery_method),
            "hmm_n_states": int(frozen.hmm_n_states),
            "hmm_emission_features": list(frozen.hmm_emission_features),
            "runtime_patch_targets": list(frozen.runtime_patch_targets),
        },
        "sweep_cells": cell_results,
        "threshold": (
            None
            if threshold is None
            else {
                "cell_id": int(threshold["cell_id"]),
                "intensity_multiplier": float(threshold["intensity_multiplier"]),
                "duty_cycle": float(threshold["duty_cycle"]),
                "channel_set": str(threshold["channel_set"]),
                "f1_latent": float(threshold["f1_latent"]),
                "f1_observed": float(threshold["f1_observed"]),
                "censoring_cost": float(threshold["censoring_cost_cell"]),
                "sensitivity_neighbourhood": nbhd,
            }
        ),
        "max_f1_latent": (
            max(c["f1_latent"] for c in cell_results if c["f1_latent"] is not None)
        ),
        "min_f1_latent": (
            min(c["f1_latent"] for c in cell_results if c["f1_latent"] is not None)
        ),
        "f1_latent_variation_max_minus_min": (
            max(c["f1_latent"] for c in cell_results if c["f1_latent"] is not None)
            - min(c["f1_latent"] for c in cell_results if c["f1_latent"] is not None)
        ),
        "classification": classification,
        "classification_reason": why,
        "promote_07b": {
            "fired": bool(promote_payload is not None),
            "new_07b_classification": new_07b_status,
            "headline_at_threshold": promote_payload,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_JSON}")
    make_figure(cell_results, threshold)
    print(f"Wrote {OUT_PDF}")
    print(f"\nFINAL CLASSIFICATION: {classification}")
    if promote_payload is not None:
        print(f"PROMOTED 07b -> {new_07b_status}")


if __name__ == "__main__":
    main()
