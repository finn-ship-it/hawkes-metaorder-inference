"""Build artefacts:
    dissertation_artifacts/data/layer3_d1_skew_markout_comparison.json
    dissertation_artifacts/figures/layer3_d1_markout_distribution.pdf

Layer-3 toxicity-aware SkewAgent + MarkoutHarness on D1.

For each seed in `seeds = range(1, 31)`:
    - Generate D1 simulator state at horizon T = 600 s, kernel-blind
      meta-order configuration loaded from
      `My repo/configs/meta_order_smoke.json` (T overridden).
    - Mid trace: derived from `ObservedTrace.bid_ticks` and
      `ObservedTrace.ask_ticks`, with the initial mid prepended at t=0.
    - Episode A (Hawkes-blind baseline): SkewAgent with
      `posterior_sensitivity = 0.0`, all other defaults.
    - Episode B (HMM-fed): same simulator seed and observation; HMM
      posterior `p(z = active | O_{0:T})` extracted via
      `recover_regimes_hmm`; SkewAgent with
      `posterior_sensitivity = 1.0`, all other defaults equal to A.
    - Same matched-RNG `(u_bid, u_ask)` draws across A and B at every
      market event, ensuring the only difference between episodes is
      the per-event skew.

Acceptance gates (per spec):
    1. Episode B mean toxicity at the 5 s horizon < Episode A; sign
       stable across >= 70% of the 30 seeds.
    2. Episode B fill rate >= 50% of Episode A.
    3. HMM mean-over-active-windows posterior > 0.6.

The runner emits the JSON aggregate and the per-side KDE figure
regardless of the gate outcome; classification is reported by the
caller (the prompt-10msc V1-V6 re-emit).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_PATH = str(REPO_ROOT / "src")
CONFIG_PATH = REPO_ROOT / "configs" / "meta_order_smoke.json"
OUT_JSON = HERE / "data" / "layer3_d1_skew_markout_comparison.json"
OUT_PDF = HERE / "figures" / "layer3_d1_markout_distribution.pdf"

SEEDS: Tuple[int, ...] = tuple(range(1, 31))
HORIZON_SECONDS: float = 600.0
WINDOW_SECONDS: float = 5.0
HORIZONS: Tuple[float, ...] = (1.0, 5.0, 30.0)
HEADLINE_HORIZON: float = 5.0

PLUMBING_RECONSTRUCTIONS = {
    "mid_trace": {
        "source": "ObservedTrace.bid_ticks and ObservedTrace.ask_ticks",
        "formula": "mid(t) = 0.5 * (bid_ticks(t) + ask_ticks(t)) * tick_size",
        "interpolation": "right-continuous step lookup between observed events",
        "initial_mid": (
            "0.5 * (initial_bid_ticks + initial_ask_ticks) * tick_size, "
            "prepended at t = 0"
        ),
        "implementation": "_build_mid_trace + MarkoutHarness.mid_at",
    }
}


def _ensure_simulator_on_path() -> None:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


def _load_base_config():
    """Load the meta_order_smoke config and override the horizon to
    `HORIZON_SECONDS`. Returns a SimulatorConfig."""
    from simulator.config import SimulatorConfig, config_from_json

    cfg = config_from_json(str(CONFIG_PATH))
    cfg = SimulatorConfig(**{**cfg.__dict__, "horizon": float(HORIZON_SECONDS)})
    return cfg


def _project_with_seed(base_cfg, seed: int):
    """Run the simulator at the given seed and project to an ObservedTrace.

    Returns (run_trace, observed_trace, simulator_config_with_seed).
    """
    from simulator.config import SimulatorConfig
    from simulator.hawkes_core import HawkesSimulator
    from simulator.observation import ObservationOperator, default_observation_config

    cfg = SimulatorConfig(**{**base_cfg.__dict__, "seed": int(seed)})
    sim = HawkesSimulator(cfg)
    sim.reset(seed=int(seed))
    run = sim.simulate()

    op = ObservationOperator(default_observation_config())
    observed = op.project(run)
    return run, observed, cfg


def _build_mid_trace(observed) -> Tuple[np.ndarray, np.ndarray]:
    """Construct (mid_times, mid_values) in price units.

    A point is added at t = 0 holding the initial mid; every observed
    event updates the mid. The trace is right-continuous.
    """
    tick_size = float(observed.tick_size)
    init_bid = float(observed.initial_bid_ticks)
    init_ask = float(observed.initial_ask_ticks)
    n = int(observed.num_observed_events)
    times = np.zeros(n + 1, dtype=np.float64)
    mids = np.zeros(n + 1, dtype=np.float64)
    times[0] = 0.0
    mids[0] = 0.5 * (init_bid + init_ask) * tick_size
    if n > 0:
        times[1:] = np.asarray(observed.times, dtype=np.float64)
        mids[1:] = (
            0.5
            * (
                np.asarray(observed.bid_ticks, dtype=np.float64)
                + np.asarray(observed.ask_ticks, dtype=np.float64)
            )
            * tick_size
        )
    return times, mids


def _build_window_features(observed, horizon: float):
    from simulator.recovery import compute_window_features

    return compute_window_features(
        observed,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=WINDOW_SECONDS,
        horizon=horizon,
    )


def _fit_hmm(features, seed: int):
    from simulator.recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm

    cfg = HMMRecoveryConfig(
        n_states=2,
        emission_features=("recent_event_rate", "directional_imbalance"),
        max_iter=200,
        tol_relative_ll=1e-6,
        seed=int(seed),
    )
    res = recover_regimes_hmm(features, cfg)
    active_state = int(np.argmax(res.emission_means[:, 0]))
    return res, active_state


def _posterior_at_event_times(
    times: np.ndarray, window_starts: np.ndarray, posterior: np.ndarray
) -> np.ndarray:
    """Right-continuous step lookup of the per-window posterior at the
    given event times. Defaults to `posterior[0]` for `t < window_starts[0]`.
    """
    if window_starts.size == 0:
        return np.full(times.shape, 0.5, dtype=np.float64)
    idx = np.searchsorted(window_starts, times, side="right") - 1
    idx = np.clip(idx, 0, window_starts.size - 1)
    return posterior[idx]


def _mean_posterior_over_true_windows(
    window_starts: np.ndarray,
    posterior: np.ndarray,
    true_windows,
) -> Optional[float]:
    """Mean of `posterior[t]` over windows that fall inside any true
    meta-order window. Used for the precondition gate (>= 0.6)."""
    if window_starts.size == 0 or len(true_windows) == 0:
        return None
    mask = np.zeros(window_starts.size, dtype=bool)
    for w in true_windows:
        mask |= (window_starts >= float(w.start_time)) & (
            window_starts < float(w.end_time)
        )
    if not np.any(mask):
        return None
    return float(np.mean(posterior[mask]))


def _run_episode(
    *,
    observed,
    mid_times: np.ndarray,
    mid_values: np.ndarray,
    posteriors_at_events: np.ndarray,
    skew_cfg,
    rng_seed: int,
):
    """Replay the observed event stream through a SkewAgent + matched RNG.

    Returns
    -------
    fills : list of Fill
    inventory_trace : np.ndarray
    skew_trace : np.ndarray (n_events, 2) — (bid_skew, ask_skew) per event.
    fill_prob_trace : np.ndarray (n_events, 2) — (p_bid, p_ask) per event.
    """
    from simulator.markout import MarkoutHarness
    from simulator.skew_agent import SkewAgent

    agent = SkewAgent(skew_cfg)
    harness_for_mid = MarkoutHarness(mid_times, mid_values)

    n_events = int(observed.num_observed_events)
    rng = np.random.default_rng(int(rng_seed))
    u_bid_all = rng.uniform(size=n_events)
    u_ask_all = rng.uniform(size=n_events)

    fills = []
    inv = 0.0
    inventory_trace = np.zeros(n_events, dtype=np.float64)
    skew_trace = np.zeros((n_events, 2), dtype=np.float64)
    fill_prob_trace = np.zeros((n_events, 2), dtype=np.float64)

    for i in range(n_events):
        t = float(observed.times[i])
        p = float(posteriors_at_events[i])
        mid_at_t = harness_for_mid.mid_at(t)
        bid_skew, ask_skew = agent.quote(t, p, inv)
        skew_trace[i, 0] = bid_skew
        skew_trace[i, 1] = ask_skew
        fill_prob_trace[i, 0] = agent.fill_probability(bid_skew)
        fill_prob_trace[i, 1] = agent.fill_probability(ask_skew)
        bid_fill, ask_fill, inv = agent.step(
            t=t,
            posterior_at_t=p,
            inventory=inv,
            mid_at_t=mid_at_t,
            u_bid=float(u_bid_all[i]),
            u_ask=float(u_ask_all[i]),
        )
        if bid_fill is not None:
            fills.append(bid_fill)
        if ask_fill is not None:
            fills.append(ask_fill)
        inventory_trace[i] = inv
    return fills, inventory_trace, skew_trace, fill_prob_trace


def _evaluate_seed(seed: int):
    _ensure_simulator_on_path()
    from simulator.markout import MarkoutHarness
    from simulator.skew_agent import SkewAgentConfig

    base_cfg = _load_base_config()
    run, observed, cfg = _project_with_seed(base_cfg, seed)
    if observed.num_observed_events == 0:
        return None

    mid_times, mid_values = _build_mid_trace(observed)
    features = _build_window_features(observed, HORIZON_SECONDS)
    if len(features) < 4:
        return None
    hmm_res, active_state = _fit_hmm(features, seed)

    posterior_active = hmm_res.posterior[:, active_state]
    posteriors_at_events = _posterior_at_event_times(
        np.asarray(observed.times, dtype=np.float64),
        hmm_res.window_starts,
        posterior_active,
    )

    # Precondition: mean posterior over true active windows
    true_windows = list(cfg.meta_order.windows)
    mean_post_active = _mean_posterior_over_true_windows(
        hmm_res.window_starts, posterior_active, true_windows
    )

    # Episode A: posterior_sensitivity = 0.0 (Hawkes-blind baseline; the
    # posterior input is irrelevant under sensitivity=0). All other
    # config parameters are spec defaults.
    cfg_a = SkewAgentConfig(posterior_sensitivity=0.0)
    cfg_b = SkewAgentConfig(posterior_sensitivity=1.0)

    # Matched RNG: identical (u_bid, u_ask) draws for both episodes.
    rng_seed_for_fills = int(seed) * 100003 + 7
    fills_a, inv_a, skew_a, p_a = _run_episode(
        observed=observed,
        mid_times=mid_times,
        mid_values=mid_values,
        posteriors_at_events=posteriors_at_events,
        skew_cfg=cfg_a,
        rng_seed=rng_seed_for_fills,
    )
    fills_b, inv_b, skew_b, p_b = _run_episode(
        observed=observed,
        mid_times=mid_times,
        mid_values=mid_values,
        posteriors_at_events=posteriors_at_events,
        skew_cfg=cfg_b,
        rng_seed=rng_seed_for_fills,
    )

    harness = MarkoutHarness(mid_times, mid_values)
    res_a = harness.compute(fills_a, horizons_seconds=HORIZONS)
    res_b = harness.compute(fills_b, horizons_seconds=HORIZONS)

    n_events = int(observed.num_observed_events)
    fill_rate_a = float(len(fills_a)) / max(1.0, 2.0 * n_events)
    fill_rate_b = float(len(fills_b)) / max(1.0, 2.0 * n_events)

    return {
        "seed": int(seed),
        "horizon": float(HORIZON_SECONDS),
        "n_events": int(n_events),
        "n_fills_episode_a": int(len(fills_a)),
        "n_fills_episode_b": int(len(fills_b)),
        "fill_rate_episode_a": fill_rate_a,
        "fill_rate_episode_b": fill_rate_b,
        "fill_rate_ratio_b_over_a": (
            float(fill_rate_b / fill_rate_a) if fill_rate_a > 0 else None
        ),
        "hmm_active_state": int(active_state),
        "hmm_converged": bool(hmm_res.converged),
        "hmm_n_iter": int(hmm_res.n_iter),
        "mean_posterior_over_true_active_windows": (
            float(mean_post_active) if mean_post_active is not None else None
        ),
        "aggregates_episode_a": res_a.aggregates,
        "aggregates_episode_b": res_b.aggregates,
        "_per_fill_b_minus_a_at_headline_horizon": _stratified_b_minus_a(
            res_a, res_b, HEADLINE_HORIZON
        ),
        # Persist per-fill markouts at the headline horizon for the figure.
        "per_fill_signed_markout_5s_episode_a": [
            {"side": r.side, "signed_markout": r.signed_markout}
            for r in res_a.per_fill
            if r.horizon_seconds == HEADLINE_HORIZON
        ],
        "per_fill_signed_markout_5s_episode_b": [
            {"side": r.side, "signed_markout": r.signed_markout}
            for r in res_b.per_fill
            if r.horizon_seconds == HEADLINE_HORIZON
        ],
    }


def _stratified_b_minus_a(res_a, res_b, h: float) -> Dict[str, Optional[float]]:
    a = res_a.aggregate_for(h)
    b = res_b.aggregate_for(h)
    out: Dict[str, Optional[float]] = {}
    for k in ("mean_signed_markout", "mean_toxicity", "median_toxicity"):
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            out[k] = None
        else:
            out[k] = float(vb - va)
    return out


def _aggregate_across_seeds(per_seed: List[dict]) -> dict:
    n = len(per_seed)
    if n == 0:
        return {}

    def _seed_metric(rec, ep_key, h, metric):
        agg = rec[f"aggregates_episode_{ep_key}"].get(h) or rec[
            f"aggregates_episode_{ep_key}"
        ].get(float(h))
        return None if agg is None else agg.get(metric)

    out: Dict[str, dict] = {}
    for h in HORIZONS:
        a_means = [
            _seed_metric(r, "a", h, "mean_toxicity") for r in per_seed
        ]
        b_means = [
            _seed_metric(r, "b", h, "mean_toxicity") for r in per_seed
        ]
        a_signed = [
            _seed_metric(r, "a", h, "mean_signed_markout") for r in per_seed
        ]
        b_signed = [
            _seed_metric(r, "b", h, "mean_signed_markout") for r in per_seed
        ]
        a_means_arr = np.array(
            [v for v in a_means if v is not None], dtype=np.float64
        )
        b_means_arr = np.array(
            [v for v in b_means if v is not None], dtype=np.float64
        )
        diffs = []
        sign_b_lower_than_a = 0
        for va, vb in zip(a_means, b_means):
            if va is None or vb is None:
                continue
            diffs.append(vb - va)
            if vb < va:
                sign_b_lower_than_a += 1
        diffs_arr = np.array(diffs, dtype=np.float64)
        n_paired = int(diffs_arr.size)
        out[str(float(h))] = {
            "n_seeds_paired": n_paired,
            "mean_toxicity_episode_a": (
                float(np.mean(a_means_arr)) if a_means_arr.size > 0 else None
            ),
            "mean_toxicity_episode_b": (
                float(np.mean(b_means_arr)) if b_means_arr.size > 0 else None
            ),
            "mean_b_minus_a": (
                float(np.mean(diffs_arr)) if n_paired > 0 else None
            ),
            "sign_stable_fraction_b_lower_than_a": (
                float(sign_b_lower_than_a) / float(n_paired) if n_paired > 0 else None
            ),
            "median_b_minus_a": (
                float(np.median(diffs_arr)) if n_paired > 0 else None
            ),
            "p10_b_minus_a": (
                float(np.percentile(diffs_arr, 10)) if n_paired > 0 else None
            ),
            "p90_b_minus_a": (
                float(np.percentile(diffs_arr, 90)) if n_paired > 0 else None
            ),
            "mean_signed_markout_episode_a": (
                float(np.mean([v for v in a_signed if v is not None]))
                if any(v is not None for v in a_signed)
                else None
            ),
            "mean_signed_markout_episode_b": (
                float(np.mean([v for v in b_signed if v is not None]))
                if any(v is not None for v in b_signed)
                else None
            ),
        }
    return out


def _gate_classify(across: dict, per_seed: List[dict]) -> dict:
    headline = across.get(str(float(HEADLINE_HORIZON)), {})
    mean_post_vals = [
        r["mean_posterior_over_true_active_windows"]
        for r in per_seed
        if r.get("mean_posterior_over_true_active_windows") is not None
    ]
    mean_posterior_mean = (
        float(np.mean(mean_post_vals)) if mean_post_vals else None
    )

    fill_rates_b_over_a = [
        r["fill_rate_ratio_b_over_a"]
        for r in per_seed
        if r.get("fill_rate_ratio_b_over_a") is not None
    ]
    median_fill_rate_ratio = (
        float(np.median(fill_rates_b_over_a)) if fill_rates_b_over_a else None
    )

    # Gate 1: Episode B mean toxicity at headline horizon < Episode A,
    # sign stable >= 70%
    g1_mean_b_minus_a = headline.get("mean_b_minus_a")
    g1_sign_stable = headline.get("sign_stable_fraction_b_lower_than_a")
    gate_1 = (
        g1_mean_b_minus_a is not None
        and g1_mean_b_minus_a < 0.0
        and g1_sign_stable is not None
        and g1_sign_stable >= 0.70
    )

    # Gate 2: Episode B fill rate >= 50% of Episode A
    gate_2 = (
        median_fill_rate_ratio is not None
        and median_fill_rate_ratio >= 0.50
    )

    # Gate 3: HMM mean-over-active-windows posterior > 0.6
    gate_3 = (
        mean_posterior_mean is not None
        and mean_posterior_mean > 0.60
    )

    if gate_1 and gate_2 and gate_3:
        classification = "10MSC_COMPLETE"
    elif gate_1 and gate_3 and not gate_2:
        # Per spec post-run amendment 2: the rule reduces per-fill
        # toxicity but at unacceptable fill-rate cost. The closed-form
        # MSc-bounded rule cannot extract NET economic value from the
        # HMM posterior; the gap motivates the optimal-control PhD
        # thread named in chapter 8 future directions.
        classification = "10MSC_NEGATIVE_RESULT_RULE_TOO_COARSE"
    elif not gate_3:
        # Signal precondition unmet on this grid.
        classification = "10MSC_NEGATIVE_RESULT_SIGNAL_PRECONDITION_UNMET"
    elif (
        g1_mean_b_minus_a is not None
        and g1_mean_b_minus_a >= 0.0
    ):
        # Episode B does not even reduce per-fill toxicity in mean.
        classification = "10MSC_NEGATIVE_RESULT_NO_TOXICITY_REDUCTION"
    elif (
        g1_sign_stable is not None and g1_sign_stable < 0.70
    ):
        # Sign of the reduction is not stable across the grid.
        classification = "10MSC_PARTIAL_REDUCTION_SIGN_UNSTABLE"
    else:
        classification = "10MSC_PARTIAL"

    return {
        "gate_1_mean_toxicity_reduction_at_headline_horizon": bool(gate_1),
        "gate_2_fill_rate_retention": bool(gate_2),
        "gate_3_signal_precondition": bool(gate_3),
        "headline_horizon_seconds": float(HEADLINE_HORIZON),
        "headline_mean_b_minus_a": g1_mean_b_minus_a,
        "headline_sign_stable_fraction_b_lower_than_a": g1_sign_stable,
        "median_fill_rate_ratio_b_over_a": median_fill_rate_ratio,
        "mean_posterior_over_true_active_windows_avg": mean_posterior_mean,
        "classification": classification,
    }


def _make_figure(per_seed: List[dict]) -> None:
    """Per-fill 5 s signed markout, signal-blind versus signal-aware, per side."""
    bid_a = []
    bid_b = []
    ask_a = []
    ask_b = []
    for r in per_seed:
        for rec in r["per_fill_signed_markout_5s_episode_a"]:
            if rec["side"] == "bid":
                bid_a.append(float(rec["signed_markout"]))
            else:
                ask_a.append(float(rec["signed_markout"]))
        for rec in r["per_fill_signed_markout_5s_episode_b"]:
            if rec["side"] == "bid":
                bid_b.append(float(rec["signed_markout"]))
            else:
                ask_b.append(float(rec["signed_markout"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    def _plot_side(ax, vals_a, vals_b, title):
        if not vals_a and not vals_b:
            ax.set_title(f"{title} (no fills)")
            return
        all_vals = np.concatenate(
            [np.asarray(vals_a), np.asarray(vals_b)]
        ) if (vals_a or vals_b) else np.array([0.0])
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        rng = max(hi - lo, 1e-9)
        bins = np.linspace(lo - 0.05 * rng, hi + 0.05 * rng, 41)
        if vals_a:
            ax.hist(
                vals_a,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.4,
                label="Signal-blind baseline",
                color="C0",
            )
        if vals_b:
            ax.hist(
                vals_b,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.4,
                label="Signal-aware arm",
                color="C3",
            )
        ax.axvline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("signed markout @ 5 s (price units)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        # Cosmetic (P6b): the markout scale is ~1e-5, so default full-precision
        # tick labels collide. Use a shared scientific offset and few ticks.
        # Data and bins are unchanged.
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(8)

    _plot_side(axes[0], bid_a, bid_b, "Bid-side fills")
    _plot_side(axes[1], ask_a, ask_b, "Ask-side fills")
    axes[0].set_ylabel("density")
    fig.suptitle(
        "Signed markout distribution at 5 s across 30 matched simulations",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    _ensure_simulator_on_path()
    print("=== Layer-3 D1 SkewAgent + MarkoutHarness 30-seed comparison ===")
    print(
        f"  config: {CONFIG_PATH.relative_to(REPO_ROOT)}, "
        f"horizon = {HORIZON_SECONDS:.0f} s, seeds = 1..{SEEDS[-1]}"
    )

    per_seed: List[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        out = _evaluate_seed(seed)
        if out is None:
            print(f"  seed {seed}: skipped (too few events / windows)")
            continue
        per_seed.append(out)
        h_a_5s = out["aggregates_episode_a"].get(HEADLINE_HORIZON, {}) or {}
        h_b_5s = out["aggregates_episode_b"].get(HEADLINE_HORIZON, {}) or {}
        a_tox = h_a_5s.get("mean_toxicity")
        b_tox = h_b_5s.get("mean_toxicity")
        delta = (
            None if (a_tox is None or b_tox is None) else (b_tox - a_tox)
        )
        print(
            f"  seed {seed:2d}: "
            f"events={out['n_events']:>5d}  "
            f"fills A/B = {out['n_fills_episode_a']:>4d}/{out['n_fills_episode_b']:>4d}  "
            f"mean tox 5s A/B = "
            f"{(a_tox if a_tox is not None else float('nan')):.5f}/"
            f"{(b_tox if b_tox is not None else float('nan')):.5f}  "
            f"B-A={(delta if delta is not None else float('nan')):+.5f}  "
            f"post={out['mean_posterior_over_true_active_windows']}"
        )
    dt = time.time() - t0
    print(f"\n  total wall: {dt:.1f}s, n_seeds = {len(per_seed)}")

    across = _aggregate_across_seeds(per_seed)
    gate = _gate_classify(across, per_seed)

    payload = {
        "schema_version": "1.0",
        "horizon_seconds": float(HORIZON_SECONDS),
        "window_seconds": float(WINDOW_SECONDS),
        "horizons_evaluated_seconds": list(HORIZONS),
        "headline_horizon_seconds": float(HEADLINE_HORIZON),
        "plumbing_reconstructions": PLUMBING_RECONSTRUCTIONS,
        "n_seeds_requested": len(SEEDS),
        "n_seeds_evaluated": len(per_seed),
        "across_seeds": across,
        "gate_summary": gate,
        "per_seed": per_seed,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_JSON}")

    _make_figure(per_seed)
    print(f"Wrote {OUT_PDF}")

    print("\n--- Gate summary ---")
    for k, v in gate.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
