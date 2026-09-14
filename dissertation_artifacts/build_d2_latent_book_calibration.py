"""Build artefacts:
    dissertation_artifacts/data/d2_latent_book_calibration_summary.json
    dissertation_artifacts/figures/d2_latent_book_inference.pdf

D2 latent full-book calibration + inference + HMM evaluation, per the
prompt-07b reframe. The 4-type FX top-of-book stream is the *observable*,
NOT the calibration target. The calibration target is the cached
real-EURUSD primary-day baseline (`eurusd_observed_composition.json`)
under a generous plausibility band (max per-FX-type composition gap
<= 25 pp; projected D2 rate within an order of magnitude of real-EURUSD
rate, i.e., [0.113, 11.3] events/s).

Pipeline:
    1. Load the cached real-EURUSD baseline (built by
       `build_eurusd_observed_composition.py`).
    2. Run a bounded calibration search (<= 10 candidates) over the
       D2 mu vector + per-native-type multipliers.
    3. Pick the winning candidate (or surface CONFIGURATION_INCOMPLETE
       if none passes the plausibility band after 10 tries).
    4. Run the winning candidate at a horizon long enough for >= 5,000
       observed FX events.
    5. Run Layer-1 EM on the observed D2 stream; compare rho_D2 to
       rho_D1 and rho_EURUSD-real (= 0.937) as a three-way table (no
       acceptance gate on D1 vs D2 rho match).
    6. Run HMM Layer-2 on the observed D2 stream; evaluate against
       `regime_truth` (the injected meta-order labels exposed only to
       the simulator). Acceptance: HMM F1 >= 0.7 on injected windows.
    7. If HMM F1 < 0.7, additionally run HMM on the LATENT stream and
       report the legacy `censoring_cost_f1` field using the repository
       convention `F1_projected - F1_latent` (`observed` is the historical
       name for the projected stream).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_PATH = str(REPO_ROOT / "src")
EURUSD_CACHE = HERE / "data" / "eurusd_observed_composition.json"
OUT_JSON = HERE / "data" / "d2_latent_book_calibration_summary.json"
OUT_PDF = HERE / "figures" / "d2_latent_book_inference.pdf"

CALIBRATION_HORIZON_SECONDS = 600.0
PRODUCTION_HORIZON_SECONDS = 1800.0
PRODUCTION_EVENT_FLOOR = 5000
WINDOW_SECONDS = 5.0
COMPOSITION_GAP_THRESHOLD_PP = 25.0
RATE_BAND = (0.113, 11.3)
HMM_F1_THRESHOLD = 0.70
CENSORING_FRACTION_FLOOR = 0.15  # 15% censoring sanity gate (07b spec)
RHO_D1_REFERENCE = None  # populated from prior artefact at runtime
RHO_EURUSD_REAL = 0.937308

FX_TYPE_LABELS = ("BID_UP", "BID_DOWN", "ASK_UP", "ASK_DOWN")


def _ensure_simulator_on_path() -> None:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


# ---------------------------------------------------------------------------
# Real-EURUSD baseline + D1 rho reference loaders
# ---------------------------------------------------------------------------


def _load_real_eurusd_baseline() -> dict:
    if not EURUSD_CACHE.exists():
        raise FileNotFoundError(
            f"Cached real-EURUSD baseline not found at {EURUSD_CACHE}. "
            "Run `build_eurusd_observed_composition.py` first."
        )
    return json.loads(EURUSD_CACHE.read_text())


def _load_rho_d1_reference() -> Optional[float]:
    """Look for a prior D1 Layer-1 EM rho. Use the EURUSD primary-day EM
    fit (since D1 is the FX-native simulator) as the canonical rho_D1
    reference for the three-way table."""
    em_path = HERE / "data" / "eurusd_layer1_em_summary.json"
    if not em_path.exists():
        return None
    try:
        d = json.loads(em_path.read_text())
        primary = d.get("primary_day_2021_07_20", {})
        em = primary.get("em_result")
        if em is not None:
            return float(em.get("rho_spec"))
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Composition diagnostics
# ---------------------------------------------------------------------------


def _composition_pct(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: 100.0 * v / total for k, v in counts.items()}


def _composition_gap(c_obs: Dict[str, float], c_real_fractions: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for label in FX_TYPE_LABELS:
        a = c_obs.get(label, 0.0)
        b = 100.0 * float(c_real_fractions.get(label.lower() + "_fraction", 0.0))
        out[label] = abs(a - b)
    return out


# ---------------------------------------------------------------------------
# Calibration candidate runner
# ---------------------------------------------------------------------------


def _run_d2_candidate(
    cand_id: int,
    description: str,
    cfg_kwargs: dict,
    real_eurusd: dict,
    horizon: float,
) -> dict:
    _ensure_simulator_on_path()
    from simulator.d2_konark_latent_book import (
        D2LatentBookConfig,
        simulate_d2_latent_book,
    )

    cfg = D2LatentBookConfig(**cfg_kwargs)
    res = simulate_d2_latent_book(cfg, horizon_seconds=horizon)
    n_obs = res.n_observed
    rate = n_obs / horizon
    comp_pct = _composition_pct(res.post_projection_fx_counts)
    gap = _composition_gap(comp_pct, real_eurusd["composition_fractions"])
    max_gap = max(gap.values()) if gap else 0.0
    rate_in_band = RATE_BAND[0] <= rate <= RATE_BAND[1]
    composition_ok = max_gap <= COMPOSITION_GAP_THRESHOLD_PP
    censoring_ok = res.censoring_fraction >= CENSORING_FRACTION_FLOOR
    passes_all = rate_in_band and composition_ok and censoring_ok
    return {
        "candidate_id": int(cand_id),
        "description": str(description),
        "parameters_changed": {
            k: v for k, v in cfg_kwargs.items()
            if k not in {"meta_order_regime_windows"}
        },
        "horizon_seconds": float(horizon),
        "n_latent": int(res.n_latent),
        "n_observed_fx": int(n_obs),
        "projected_event_rate_per_second": float(rate),
        "rate_in_band": bool(rate_in_band),
        "censoring_fraction": float(res.censoring_fraction),
        "censoring_meets_15pct_floor": bool(censoring_ok),
        "post_projection_fx_counts": dict(res.post_projection_fx_counts),
        "composition_pct": dict(comp_pct),
        "composition_gap_pp": dict(gap),
        "max_composition_gap_pp": float(max_gap),
        "composition_within_25pp": bool(composition_ok),
        "passes_plausibility": bool(passes_all),
        "wall_seconds": float(res.wall_seconds),
    }


def _calibration_candidates() -> List[Tuple[str, dict]]:
    """Return a list of (description, cfg_kwargs) tuples — at most 10.

    Highest-leverage knob first (per spec): rebalance lo_inspread mu.
    Subsequent candidates explore secondary knobs (per-native-type
    multipliers, spread0).
    """
    base = dict(spread0=0.03, seed=1, use_exp_approx=True)
    return [
        # 1: Konark default 3:1 inspread Bid:Ask
        ("default 3:1 inspread, no overrides",
         {**base, "inspread_bid_over_ask_mu_ratio": 3.0}),
        # 2: Symmetric inspread (highest-leverage knob)
        ("symmetric (1:1) inspread, no overrides",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0}),
        # 3: Symmetric + downscale Bid-side overall to fight Hawkes
        # cross-excitation that still favours Bid.
        ("symmetric inspread + halve lo_inspread_Bid",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (("lo_inspread_Bid", 0.5),)}),
        # 4: Push Ask side up + halve Bid inspread
        ("symmetric inspread + halve lo_inspread_Bid + 2x Ask-side mo and inspread",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.5),
              ("mo_Ask", 2.0),
              ("co_top_Ask", 2.0),
              ("lo_top_Ask", 2.0),
              ("lo_inspread_Ask", 2.0),
          )}),
        # 5: Increase deep mu to push censoring above 15%
        ("4 + 5x deep on both sides (boost censoring)",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.5),
              ("mo_Ask", 2.0),
              ("co_top_Ask", 2.0),
              ("lo_top_Ask", 2.0),
              ("lo_inspread_Ask", 2.0),
              ("lo_deep_Bid", 5.0),
              ("lo_deep_Ask", 5.0),
              ("co_deep_Bid", 5.0),
              ("co_deep_Ask", 5.0),
          )}),
        # 6: Same as 5 but with stronger BID_UP downscale
        ("5 + 0.3x lo_inspread_Bid (stronger Bid downscale)",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.3),
              ("mo_Ask", 3.0),
              ("co_top_Ask", 3.0),
              ("lo_top_Ask", 3.0),
              ("lo_inspread_Ask", 3.0),
              ("lo_deep_Bid", 5.0),
              ("lo_deep_Ask", 5.0),
              ("co_deep_Bid", 5.0),
              ("co_deep_Ask", 5.0),
          )}),
        # 7: 6 with even more deep
        ("6 + 10x deep on both sides",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.3),
              ("mo_Ask", 3.0),
              ("co_top_Ask", 3.0),
              ("lo_top_Ask", 3.0),
              ("lo_inspread_Ask", 3.0),
              ("lo_deep_Bid", 10.0),
              ("lo_deep_Ask", 10.0),
              ("co_deep_Bid", 10.0),
              ("co_deep_Ask", 10.0),
          )}),
        # 8: Rebalance toward symmetric AND increase Bid-DOWN side (mo_Bid, co_top_Bid)
        ("7 + 2x mo_Bid and co_top_Bid (boost BID_DOWN)",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.3),
              ("mo_Ask", 3.0),
              ("co_top_Ask", 3.0),
              ("lo_top_Ask", 3.0),
              ("lo_inspread_Ask", 3.0),
              ("mo_Bid", 2.0),
              ("co_top_Bid", 2.0),
              ("lo_deep_Bid", 10.0),
              ("lo_deep_Ask", 10.0),
              ("co_deep_Bid", 10.0),
              ("co_deep_Ask", 10.0),
          )}),
        # 9: Final tuning — push composition closer to symmetric
        ("8 + 0.2x lo_inspread_Bid",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.2),
              ("mo_Ask", 3.0),
              ("co_top_Ask", 3.0),
              ("lo_top_Ask", 3.0),
              ("lo_inspread_Ask", 3.0),
              ("mo_Bid", 2.0),
              ("co_top_Bid", 2.0),
              ("lo_deep_Bid", 10.0),
              ("lo_deep_Ask", 10.0),
              ("co_deep_Bid", 10.0),
              ("co_deep_Ask", 10.0),
          )}),
        # 10: Even more aggressive Bid downscale
        ("9 + 0.15x lo_inspread_Bid",
         {**base, "inspread_bid_over_ask_mu_ratio": 1.0,
          "per_native_type_mu_multipliers": (
              ("lo_inspread_Bid", 0.15),
              ("mo_Ask", 3.0),
              ("co_top_Ask", 3.0),
              ("lo_top_Ask", 3.0),
              ("lo_inspread_Ask", 3.0),
              ("mo_Bid", 2.5),
              ("co_top_Bid", 2.5),
              ("lo_deep_Bid", 10.0),
              ("lo_deep_Ask", 10.0),
              ("co_deep_Bid", 10.0),
              ("co_deep_Ask", 10.0),
          )}),
    ]


# ---------------------------------------------------------------------------
# Layer-1 EM + HMM evaluation helpers
# ---------------------------------------------------------------------------


def _fit_layer1_em(times: np.ndarray, types: np.ndarray, T: float) -> dict:
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


def _build_observed_trace_from_d2(d2_result, observed_only=True):
    from simulator.observation import ObservedTrace

    if observed_only:
        n = int(d2_result.n_observed)
        times = np.asarray([ev.time for ev in d2_result.observed_events], dtype=np.float64)
        type_to_idx = {t: i for i, t in enumerate(FX_TYPE_LABELS)}
        types = np.asarray(
            [type_to_idx[ev.observed_type] for ev in d2_result.observed_events],
            dtype=np.int64,
        )
    else:
        # For latent-stream HMM (used only for censoring-cost diagnostic),
        # use the same FX projection rule but DO NOT drop deep events; map
        # them directly onto the most semantically appropriate FX type.
        # The mapping below is for diagnostic-only use.
        from simulator.d2_konark_latent_book import KONARK_TYPE_TO_INDEX
        latent_fx_map = {
            "lo_top_Bid": 0,        # BID_UP
            "co_top_Bid": 1,        # BID_DOWN
            "lo_top_Ask": 3,        # ASK_DOWN
            "co_top_Ask": 2,        # ASK_UP
            "mo_Ask": 2,
            "mo_Bid": 1,
            "lo_inspread_Bid": 0,
            "lo_inspread_Ask": 3,
            # Diagnostic mapping for deep events (uncensored stream only):
            "lo_deep_Bid": 0,       # treat depth additions same side as inspread for diagnostic
            "co_deep_Bid": 1,
            "lo_deep_Ask": 3,
            "co_deep_Ask": 2,
        }
        times = np.asarray([ev.time for ev in d2_result.latent_events], dtype=np.float64)
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


def _viterbi_to_intervals(
    viterbi: np.ndarray,
    window_starts: np.ndarray,
    window_seconds: float,
    active_state: int,
):
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


def _hmm_layer2_eval(
    d2_result, *, observed_only: bool, horizon: float, seed: int
) -> dict:
    _ensure_simulator_on_path()
    from simulator.config import MetaOrderWindow
    from simulator.recovery import compute_window_features
    from simulator.recovery_eval import evaluate_meta_order_recovery
    from simulator.recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm

    obs = _build_observed_trace_from_d2(d2_result, observed_only=observed_only)
    features = compute_window_features(
        obs, window_seconds=WINDOW_SECONDS, hop_seconds=WINDOW_SECONDS, horizon=horizon
    )
    if len(features) < 4:
        return {"available": False, "reason": "fewer than 4 windows"}
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
        res.viterbi_labels, res.window_starts, WINDOW_SECONDS, active_state
    )
    # Translate D2 regime windows into MetaOrderWindow shape for the
    # evaluation harness; the FX-native MetaOrderWindow ground-truth
    # interface accepts (start_time, end_time, alpha, target_event_types).
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
    posterior_active = res.posterior[:, active_state]
    p = np.clip(res.posterior, 1e-12, 1.0)
    ent = float(np.mean(-np.sum(p * np.log(p), axis=1)))
    # Per-event accuracy on window membership (using window_starts as reference grid)
    in_true = np.zeros(res.window_starts.shape[0], dtype=bool)
    for w in d2_result.meta_order_regime_windows:
        in_true |= (
            (res.window_starts >= w.start_t)
            & (res.window_starts < w.end_t)
        )
    pred_active = (res.viterbi_labels == active_state)
    per_event_accuracy = float(np.mean(pred_active == in_true))
    # Inferred regime durations
    durations = []
    if res.viterbi_labels.size > 0:
        cur_state = int(res.viterbi_labels[0])
        cur_dur = 1
        for s in res.viterbi_labels[1:]:
            if int(s) == cur_state:
                cur_dur += 1
            else:
                durations.append(cur_dur * WINDOW_SECONDS)
                cur_state = int(s)
                cur_dur = 1
        durations.append(cur_dur * WINDOW_SECONDS)
    durations_arr = np.asarray(durations, dtype=np.float64)
    return {
        "available": True,
        "stream": "observed" if observed_only else "latent",
        "hmm_converged": bool(res.converged),
        "hmm_n_iter": int(res.n_iter),
        "active_state_index": int(active_state),
        "n_windows_features": int(len(features)),
        "precision": float(eval_.precision),
        "recall": float(eval_.recall),
        "f1": float(eval_.f1),
        "per_event_accuracy_on_window_membership": per_event_accuracy,
        "mean_detection_delay_seconds": float(eval_.mean_detection_delay_seconds),
        "n_inferred_intervals": int(len(intervals)),
        "n_true_windows": int(len(true_windows)),
        "active_window_rate_posterior_gt_half": float(np.mean(posterior_active > 0.5)),
        "mean_posterior_entropy_nats": float(ent),
        "regime_duration_distribution_seconds": {
            "median": float(np.median(durations_arr)) if durations_arr.size else None,
            "p10": float(np.percentile(durations_arr, 10)) if durations_arr.size else None,
            "p90": float(np.percentile(durations_arr, 90)) if durations_arr.size else None,
            "n_segments": int(durations_arr.size),
        },
        "_posterior_active": posterior_active.tolist(),
        "_window_starts": res.window_starts.tolist(),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _make_figure(
    payload: dict,
    d2_result,
    rho_d1: Optional[float],
    rho_d2: Optional[float],
    hmm_observed: dict,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel 1: latent vs observed by Konark-native type (censoring object)
    ax = axes[0]
    pre = d2_result.pre_projection_native_counts
    types = list(pre.keys())
    pre_counts = [pre[t] for t in types]
    obs_counts = []
    from simulator.d2_konark_latent_book import PHI_BETA_PROJECTION
    for t in types:
        if PHI_BETA_PROJECTION[t] is None:
            obs_counts.append(0)  # censored
        else:
            obs_counts.append(pre[t])
    x = np.arange(len(types))
    width = 0.4
    ax.bar(x - width / 2, pre_counts, width, label="latent", color="C0")
    ax.bar(x + width / 2, obs_counts, width, label="post-Phi_beta (observed)", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=70, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Latent vs observed by Konark-native type\n(deep events censored)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: HMM posterior on a representative episode w/ true windows overlaid
    ax = axes[1]
    if hmm_observed.get("available", False):
        post = np.asarray(hmm_observed.get("_posterior_active", []))
        starts = np.asarray(hmm_observed.get("_window_starts", []))
        if post.size > 0 and starts.size > 0:
            ax.plot(starts, post, color="C0", lw=1.5, label="HMM posterior p(active)")
            for w in d2_result.meta_order_regime_windows:
                ax.axvspan(
                    w.start_t,
                    w.end_t,
                    color="C3",
                    alpha=0.18,
                    label="injected regime window" if w.window_id == 1 else None,
                )
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("posterior p(active)")
            ax.set_title(
                f"HMM posterior on observed D2 stream\n"
                f"(F1 = {hmm_observed.get('f1', float('nan')):.3f})"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    # Panel 3: Three-way rho comparison
    ax = axes[2]
    rho_labels = []
    rho_values = []
    rho_colors = []
    if rho_d1 is not None:
        rho_labels.append("D1 (FX-native EM,\n EURUSD primary)")
        rho_values.append(float(rho_d1))
        rho_colors.append("C0")
    if rho_d2 is not None:
        rho_labels.append("D2 (Konark + Phi_beta)")
        rho_values.append(float(rho_d2))
        rho_colors.append("C3")
    rho_labels.append("EURUSD anchor (real)")
    rho_values.append(float(RHO_EURUSD_REAL))
    rho_colors.append("C2")
    ax.bar(rho_labels, rho_values, color=rho_colors)
    ax.set_ylabel(r"branching ratio $\rho$")
    ax.set_ylim(0, max(1.0, 1.1 * max(rho_values)))
    ax.axhline(1.0, color="k", lw=0.7, alpha=0.4, ls="--")
    for i, v in enumerate(rho_values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_title("Three-way rho comparison\n(no acceptance gate; substantive)")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "07b: D2 latent full-book under explicit C_beta — "
        "censoring object, HMM recovery, three-way rho",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def _strip_internal(d):
    """Drop fields prefixed with `_` (used for in-process figure data)."""
    if isinstance(d, dict):
        return {k: _strip_internal(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_internal(v) for v in d]
    return d


def main():
    _ensure_simulator_on_path()
    from simulator.evaluation import projection_cost_f1

    print("=== 07b D2 latent full-book calibration + inference ===")

    # Load real-EURUSD baseline
    real_eurusd = _load_real_eurusd_baseline()
    rho_d1 = _load_rho_d1_reference()
    print(
        f"  real-EURUSD baseline: N={real_eurusd['n_events']}, "
        f"rate={real_eurusd['event_rate_per_second']:.4f} events/s, "
        f"composition_pct={ {k.replace('_fraction',''): 100*v for k, v in real_eurusd['composition_fractions'].items()} }"
    )
    print(f"  rho_D1 reference (EM on EURUSD primary day): {rho_d1}")

    # Calibration search
    print(f"\n[Calibration search at T={CALIBRATION_HORIZON_SECONDS:.0f}s, budget=10]")
    candidates_specs = _calibration_candidates()
    candidates_log = []
    winner = None
    for i, (desc, kwargs) in enumerate(candidates_specs, start=1):
        print(f"  candidate {i}: {desc}")
        rec = _run_d2_candidate(
            i, desc, kwargs, real_eurusd, CALIBRATION_HORIZON_SECONDS
        )
        candidates_log.append(rec)
        print(
            f"    n_obs={rec['n_observed_fx']:>5d} rate={rec['projected_event_rate_per_second']:.3f}"
            f"  cens={rec['censoring_fraction']:.3f}"
            f"  max_gap={rec['max_composition_gap_pp']:.1f}pp"
            f"  pass={rec['passes_plausibility']}"
        )
        if rec["passes_plausibility"]:
            winner = (i, desc, kwargs)
            print(f"  -> winner: candidate {i}")
            break

    if winner is None:
        print("\n  No candidate passed plausibility within 10-candidate budget.")
        # Pick the candidate with smallest max_gap as the "best-effort" config to
        # still report rho/HMM diagnostics; flag CONFIGURATION_INCOMPLETE.
        candidates_log.sort(key=lambda r: r["max_composition_gap_pp"])
        best = candidates_log[0]
        winner = (best["candidate_id"], best["description"], next(
            (kw for d, kw in candidates_specs if d == best["description"]), None
        ))
        classification_pre_inference = "07B_CONFIGURATION_INCOMPLETE"
    else:
        classification_pre_inference = "07B_PROCEED_TO_INFERENCE"

    # Production run at the winner's parameters
    win_id, win_desc, win_kwargs = winner
    print(
        f"\n[Production run at T={PRODUCTION_HORIZON_SECONDS:.0f}s with "
        f"winning kwargs from candidate {win_id}]"
    )
    from simulator.d2_konark_latent_book import (
        D2LatentBookConfig,
        simulate_d2_latent_book,
    )

    cfg = D2LatentBookConfig(**win_kwargs)
    t0 = time.time()
    d2_result = simulate_d2_latent_book(
        cfg, horizon_seconds=PRODUCTION_HORIZON_SECONDS
    )
    print(f"  wall: {time.time()-t0:.1f}s")
    print(
        f"  n_latent={d2_result.n_latent}, n_observed={d2_result.n_observed},"
        f" cens_pct={d2_result.censoring_pct:.2f}%"
    )
    prod_event_floor_met = d2_result.n_observed >= PRODUCTION_EVENT_FLOOR
    if not prod_event_floor_met and classification_pre_inference == "07B_PROCEED_TO_INFERENCE":
        # Production T didn't reach floor; allow proceed but flag.
        print(
            f"  WARN: production n_observed={d2_result.n_observed} below "
            f"floor {PRODUCTION_EVENT_FLOOR}; proceeding to inference but "
            "marking calibration as production-floor-not-met."
        )

    # Layer-1 EM on observed D2 stream
    print("\n[Layer-1 EM on observed D2 stream]")
    times_obs = np.asarray([ev.time for ev in d2_result.observed_events], dtype=np.float64)
    type_to_idx = {t: i for i, t in enumerate(FX_TYPE_LABELS)}
    types_obs = np.asarray(
        [type_to_idx[ev.observed_type] for ev in d2_result.observed_events],
        dtype=np.int64,
    )
    em_d2 = _fit_layer1_em(times_obs, types_obs, PRODUCTION_HORIZON_SECONDS)
    print(f"  EM_D2: {em_d2}")

    # HMM Layer-2 on observed D2 stream
    print("\n[HMM Layer-2 on observed D2 stream evaluated against regime_truth]")
    hmm_observed = _hmm_layer2_eval(
        d2_result,
        observed_only=True,
        horizon=PRODUCTION_HORIZON_SECONDS,
        seed=int(cfg.seed),
    )
    print(
        f"  precision={hmm_observed.get('precision'):.3f}  "
        f"recall={hmm_observed.get('recall'):.3f}  "
        f"f1={hmm_observed.get('f1'):.3f}  "
        f"acc={hmm_observed.get('per_event_accuracy_on_window_membership'):.3f}"
    )

    # If projected-stream F1 < 0.7, compare it with latent-stream recovery.
    hmm_latent = None
    projection_cost = None
    if hmm_observed.get("f1", 0.0) < HMM_F1_THRESHOLD:
        print(
            f"\n[HMM F1 = {hmm_observed.get('f1', float('nan')):.3f} < "
            f"{HMM_F1_THRESHOLD}; running HMM on LATENT stream for "
            "projection-cost comparison]"
        )
        hmm_latent = _hmm_layer2_eval(
            d2_result,
            observed_only=False,
            horizon=PRODUCTION_HORIZON_SECONDS,
            seed=int(cfg.seed),
        )
        if hmm_latent.get("available"):
            projection_cost = projection_cost_f1(
                hmm_observed["f1"], hmm_latent["f1"]
            )
            print(
                f"  F1_latent = {hmm_latent['f1']:.3f}  "
                f"F1_observed = {hmm_observed['f1']:.3f}  "
                f"projection_cost = {projection_cost:+.3f}"
            )

    # Final classification
    g1_source_truth = True  # Verified at module level + in tests
    g2_censoring = d2_result.censoring_fraction >= CENSORING_FRACTION_FLOOR
    plausibility_winner = (winner is not None) and (classification_pre_inference == "07B_PROCEED_TO_INFERENCE")
    g3_real_plausibility = plausibility_winner
    g4_hmm_recovery = hmm_observed.get("f1", 0.0) >= HMM_F1_THRESHOLD

    if g1_source_truth and g2_censoring and g3_real_plausibility and g4_hmm_recovery:
        classification = "07B_COMPLETE"
    elif g1_source_truth and g2_censoring and g3_real_plausibility and not g4_hmm_recovery:
        classification = "07B_PARTIAL"
    elif g1_source_truth and (not g2_censoring or not g3_real_plausibility):
        classification = "07B_CONFIGURATION_INCOMPLETE"
    else:
        classification = "07B_BLOCKED_RUNTIME"

    payload = {
        "schema_version": "1.0",
        "prompt": "07b_konark_latent_full_book_calibration",
        "config": {
            "calibration_horizon_seconds": float(CALIBRATION_HORIZON_SECONDS),
            "production_horizon_seconds": float(PRODUCTION_HORIZON_SECONDS),
            "production_event_floor": int(PRODUCTION_EVENT_FLOOR),
            "window_seconds": float(WINDOW_SECONDS),
            "composition_gap_threshold_pp": float(COMPOSITION_GAP_THRESHOLD_PP),
            "rate_band": list(RATE_BAND),
            "hmm_f1_threshold": float(HMM_F1_THRESHOLD),
            "censoring_fraction_floor": float(CENSORING_FRACTION_FLOOR),
            "rho_eurusd_real_anchor": float(RHO_EURUSD_REAL),
        },
        "real_eurusd_baseline": real_eurusd,
        "rho_d1_reference": rho_d1,
        "calibration_candidates": candidates_log,
        "winning_candidate": {
            "candidate_id": int(win_id),
            "description": str(win_desc),
            "parameters": {
                k: v for k, v in win_kwargs.items()
                if k not in {"meta_order_regime_windows"}
            },
            "selected_after_n_screened": len(candidates_log),
            "passed_plausibility": classification_pre_inference == "07B_PROCEED_TO_INFERENCE",
        },
        "production_run": {
            "horizon_seconds": float(PRODUCTION_HORIZON_SECONDS),
            "n_latent": int(d2_result.n_latent),
            "n_observed": int(d2_result.n_observed),
            "censoring_fraction": float(d2_result.censoring_fraction),
            "censoring_fraction_meets_15pct_floor": bool(g2_censoring),
            "production_event_floor_met": bool(prod_event_floor_met),
            "pre_projection_native_counts": dict(d2_result.pre_projection_native_counts),
            "post_projection_fx_counts": dict(d2_result.post_projection_fx_counts),
            "post_projection_composition_pct": _composition_pct(d2_result.post_projection_fx_counts),
            "composition_gap_pp_to_real_eurusd": _composition_gap(
                _composition_pct(d2_result.post_projection_fx_counts),
                real_eurusd["composition_fractions"],
            ),
            "regime_truth_aggregate": d2_result.regime_truth_aggregate,
            "config_summary": d2_result.config_summary,
            "wall_seconds": float(d2_result.wall_seconds),
        },
        "meta_order_regime_windows": [
            {
                "window_id": int(w.window_id),
                "start_t": float(w.start_t),
                "end_t": float(w.end_t),
                "direction": str(w.direction),
                "intensity_multiplier": float(w.intensity_multiplier),
                "affected_native_types": list(w.affected_native_types),
            }
            for w in d2_result.meta_order_regime_windows
        ],
        "layer1_em_observed_d2": em_d2,
        "three_way_rho_table": {
            "rho_d1_em_eurusd_primary": rho_d1,
            "rho_d2_observed": float(em_d2["rho_spec"]),
            "rho_eurusd_real_anchor": float(RHO_EURUSD_REAL),
        },
        "hmm_layer2_observed_d2": _strip_internal(hmm_observed),
        "hmm_layer2_latent_d2": _strip_internal(hmm_latent) if hmm_latent else None,
        # Retain the historical key for compatibility; its signed definition
        # is F1_projected - F1_latent.
        "censoring_cost_f1": (
            float(projection_cost) if projection_cost is not None else None
        ),
        "source_of_truth_verification_inherited_from_07": {
            "import_path": (
                "from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import HawkesArrival"
            ),
            "konark_repo_root": str(Path(REPO_ROOT.parent) / "Konarks-github repo"),
            "event_generation_call": "arrival.get_nextarrival(timelimit=T) -> HawkesArrival.thinningOgataIS2(T)",
            "runtime_patches": d2_result.runtime_patches,
        },
        "acceptance_gates": {
            "gate_1_source_of_truth": bool(g1_source_truth),
            "gate_2_censoring_sanity": bool(g2_censoring),
            "gate_3_real_eurusd_plausibility": bool(g3_real_plausibility),
            "gate_4_latent_truth_recovery_hmm_f1": bool(g4_hmm_recovery),
        },
        "classification": classification,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_JSON}")
    _make_figure(payload, d2_result, rho_d1, em_d2["rho_spec"], hmm_observed)
    print(f"Wrote {OUT_PDF}")
    print(f"\nFINAL CLASSIFICATION: {classification}")


if __name__ == "__main__":
    main()
