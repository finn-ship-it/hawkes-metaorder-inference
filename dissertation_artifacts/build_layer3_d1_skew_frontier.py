"""Build artefact: Layer-3 D1 posterior-sensitivity skew frontier.

Extends the prompt-10msc matched-seed Layer-3 experiment from a single
Episode-B sensitivity (posterior_sensitivity = 1.0) to a SWEEP over
posterior_sensitivity, holding everything else fixed. The scientific
question is the constrained market-making trade-off: as the quoting rule
consumes the HMM posterior more aggressively, how much per-fill toxicity
does it remove and how much fill volume does it give up?

Protocol (reused verbatim from build_layer3_d1_skew_markout.py via import,
so the endpoints reproduce the committed 10msc artefact exactly):
  - config My repo/configs/meta_order_smoke.json, horizon T = 600 s
  - seeds 1..30
  - same HMM posterior construction (recover_regimes_hmm, 2 states)
  - same MarkoutHarness, horizons {1, 5, 30} s, headline 5 s
  - same matched fill-RNG seed per seed (seed*100003 + 7), identical
    (u_bid, u_ask) draws across every sensitivity cell
  - Episode A baseline = posterior_sensitivity 0.0, default SkewAgentConfig
  - SWEEP posterior_sensitivity in {0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00}
  - max_skew_ticks = 5.0 and all fill-logistic parameters fixed at the
    SkewAgentConfig defaults (the clean question is posterior sensitivity,
    not a confounded quote/fill-model change).

Per-seed cost: one simulate + one HMM fit, then a cheap episode replay +
markout per sensitivity. The s = 0.0 cell IS Episode A, so it is computed
once and reused as the baseline for every ratio.

This run computes NEW diagnostics on the EXISTING D1 Layer-3 setup; it
changes no model, no source module, no test, and does not touch the
committed layer3_d1_skew_markout_comparison.json. Cross-check gates assert
the s = 0.0 and s = 1.0 cells reproduce the committed endpoints.

Outputs:
  data/layer3_d1_skew_frontier.json
  figures/layer3_d1_skew_frontier.pdf
  tables/tab_layer3_d1_skew_frontier.tex
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_layer3_d1_skew_markout as base

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "data" / "layer3_d1_skew_frontier.json"
OUT_PDF = HERE / "figures" / "layer3_d1_skew_frontier.pdf"
OUT_TEX = HERE / "tables" / "tab_layer3_d1_skew_frontier.tex"
COMMITTED_10MSC = HERE / "data" / "layer3_d1_skew_markout_comparison.json"

SENSITIVITIES: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
HEADLINE = base.HEADLINE_HORIZON  # 5.0
FILL_RETENTION_CONSTRAINT = 0.50

# Cross-check tolerances against the committed 10msc endpoints.
TOL_TOX = 1e-12
TOL_RATIO = 1e-9


def _seed_cell(seed: int):
    """Return, for one seed, the Episode-A baseline metrics and a dict
    sensitivity -> per-seed metrics, sharing one simulate + one HMM fit and
    the matched fill RNG. Returns None if the seed yields too few windows.
    """
    base._ensure_simulator_on_path()
    from simulator.skew_agent import SkewAgentConfig

    base_cfg = base._load_base_config()
    run, observed, cfg = base._project_with_seed(base_cfg, seed)
    if observed.num_observed_events == 0:
        return None

    mid_times, mid_values = base._build_mid_trace(observed)
    features = base._build_window_features(observed, base.HORIZON_SECONDS)
    if len(features) < 4:
        return None
    hmm_res, active_state = base._fit_hmm(features, seed)
    posterior_active = hmm_res.posterior[:, active_state]
    posteriors_at_events = base._posterior_at_event_times(
        np.asarray(observed.times, dtype=np.float64),
        hmm_res.window_starts,
        posterior_active,
    )
    true_windows = list(cfg.meta_order.windows)
    mean_post_active = base._mean_posterior_over_true_windows(
        hmm_res.window_starts, posterior_active, true_windows
    )

    from simulator.markout import MarkoutHarness

    harness = MarkoutHarness(mid_times, mid_values)
    n_events = int(observed.num_observed_events)
    rng_seed_for_fills = int(seed) * 100003 + 7

    def run_sensitivity(s: float):
        skew_cfg = SkewAgentConfig(posterior_sensitivity=float(s))
        fills, _inv, _skew, _p = base._run_episode(
            observed=observed,
            mid_times=mid_times,
            mid_values=mid_values,
            posteriors_at_events=posteriors_at_events,
            skew_cfg=skew_cfg,
            rng_seed=rng_seed_for_fills,
        )
        res = harness.compute(fills, horizons_seconds=base.HORIZONS)
        agg = res.aggregate_for(HEADLINE)
        fill_rate = float(len(fills)) / max(1.0, 2.0 * n_events)
        return {
            "n_fills": int(len(fills)),
            "fill_rate": fill_rate,
            "mean_toxicity_5s": (None if agg is None else agg.get("mean_toxicity")),
            "mean_signed_markout_5s": (None if agg is None else agg.get("mean_signed_markout")),
        }

    cells = {f"{s:.2f}": run_sensitivity(s) for s in SENSITIVITIES}
    return {
        "seed": int(seed),
        "n_events": n_events,
        "hmm_converged": bool(hmm_res.converged),
        "mean_posterior_over_true_active_windows": (
            float(mean_post_active) if mean_post_active is not None else None
        ),
        "cells": cells,
    }


def _aggregate(per_seed: List[dict]) -> List[dict]:
    """Across-seed aggregation per sensitivity, with Episode A (s=0.00) as
    the matched baseline within each seed."""
    rows = []
    for s in SENSITIVITIES:
        key = f"{s:.2f}"
        tox_s, tox_a, deltas, ratios, signs = [], [], [], [], []
        n_fills_s = []
        for r in per_seed:
            cs = r["cells"][key]
            ca = r["cells"]["0.00"]
            if cs["mean_toxicity_5s"] is None or ca["mean_toxicity_5s"] is None:
                continue
            tox_s.append(cs["mean_toxicity_5s"])
            tox_a.append(ca["mean_toxicity_5s"])
            deltas.append(cs["mean_toxicity_5s"] - ca["mean_toxicity_5s"])
            signs.append(1 if cs["mean_toxicity_5s"] < ca["mean_toxicity_5s"] else 0)
            if ca["fill_rate"] > 0:
                ratios.append(cs["fill_rate"] / ca["fill_rate"])
            n_fills_s.append(cs["n_fills"])
        tox_s = np.asarray(tox_s)
        tox_a = np.asarray(tox_a)
        deltas = np.asarray(deltas)
        ratios = np.asarray(ratios)
        mean_tox_s = float(np.mean(tox_s))
        mean_tox_a = float(np.mean(tox_a))
        mean_delta = float(np.mean(deltas))
        # % toxicity reduction vs A (positive = B less toxic than A)
        pct_reduction = (
            float(100.0 * (mean_tox_a - mean_tox_s) / mean_tox_a)
            if mean_tox_a != 0
            else None
        )
        median_retention = float(np.median(ratios)) if ratios.size else None
        sign_stable = float(np.mean(signs)) if signs else None
        rows.append(
            {
                "posterior_sensitivity": float(s),
                "n_seeds_paired": int(deltas.size),
                "mean_toxicity_5s": mean_tox_s,
                "mean_toxicity_5s_episode_a": mean_tox_a,
                "mean_b_minus_a_5s": mean_delta,
                "pct_toxicity_reduction_vs_a": pct_reduction,
                "median_fill_rate_retention_vs_a": median_retention,
                "sign_stable_fraction_b_lower_than_a": sign_stable,
                "mean_fills_per_seed": float(np.mean(n_fills_s)) if n_fills_s else None,
                "total_fills": int(np.sum(n_fills_s)) if n_fills_s else 0,
            }
        )
    return rows


def _cross_check_endpoints(rows: List[dict], per_seed: List[dict]) -> dict:
    """Assert s=0.00 is a no-op vs A, and s=1.00 reproduces the committed
    10msc Episode-B endpoints."""
    by_s = {r["posterior_sensitivity"]: r for r in rows}
    checks = {}

    # s = 0.0 must be identical to Episode A: zero delta, retention 1.0.
    r0 = by_s[0.0]
    checks["s0_zero_delta"] = abs(r0["mean_b_minus_a_5s"]) <= TOL_TOX
    checks["s0_retention_unity"] = abs(r0["median_fill_rate_retention_vs_a"] - 1.0) <= TOL_RATIO

    # s = 1.0 must reproduce the committed 10msc gate numbers.
    committed = json.loads(COMMITTED_10MSC.read_text())
    gs = committed["gate_summary"]
    r1 = by_s[1.0]
    checks["s1_mean_b_minus_a"] = abs(r1["mean_b_minus_a_5s"] - gs["headline_mean_b_minus_a"]) <= TOL_TOX
    checks["s1_median_retention"] = abs(
        r1["median_fill_rate_retention_vs_a"] - gs["median_fill_rate_ratio_b_over_a"]
    ) <= TOL_RATIO
    checks["s1_sign_stable"] = abs(
        r1["sign_stable_fraction_b_lower_than_a"] - gs["headline_sign_stable_fraction_b_lower_than_a"]
    ) <= TOL_RATIO
    # mean posterior over true windows (seed-level, sensitivity-independent)
    mp = float(np.mean([r["mean_posterior_over_true_active_windows"] for r in per_seed]))
    checks["mean_posterior_matches_10msc"] = abs(
        mp - gs["mean_posterior_over_true_active_windows_avg"]
    ) <= 1e-9
    checks["_committed_gate_summary"] = {
        "headline_mean_b_minus_a": gs["headline_mean_b_minus_a"],
        "median_fill_rate_ratio_b_over_a": gs["median_fill_rate_ratio_b_over_a"],
        "headline_sign_stable_fraction_b_lower_than_a": gs["headline_sign_stable_fraction_b_lower_than_a"],
        "mean_posterior_over_true_active_windows_avg": gs["mean_posterior_over_true_active_windows_avg"],
    }
    checks["_reconstructed_mean_posterior_avg"] = mp
    ok = all(v for k, v in checks.items() if not k.startswith("_"))
    if not ok:
        raise SystemExit(f"STOP: frontier endpoints do not reproduce committed 10msc: {checks}")
    return checks


def _feasible_region(rows: List[dict]) -> dict:
    feasible = [
        r
        for r in rows
        if r["pct_toxicity_reduction_vs_a"] is not None
        and r["pct_toxicity_reduction_vs_a"] > 0.0
        and r["median_fill_rate_retention_vs_a"] is not None
        and r["median_fill_rate_retention_vs_a"] >= FILL_RETENTION_CONSTRAINT
    ]
    best = None
    if feasible:
        best = max(feasible, key=lambda r: r["pct_toxicity_reduction_vs_a"])
    return {
        "constraint": f"toxicity_reduction > 0 AND fill_retention >= {FILL_RETENTION_CONSTRAINT}",
        "feasible_sensitivities": [r["posterior_sensitivity"] for r in feasible],
        "n_feasible_cells": len(feasible),
        "best_feasible_sensitivity": (best["posterior_sensitivity"] if best else None),
        "best_feasible_pct_reduction": (best["pct_toxicity_reduction_vs_a"] if best else None),
        "best_feasible_fill_retention": (best["median_fill_rate_retention_vs_a"] if best else None),
    }


def _make_figure(rows: List[dict], feasible: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    xs = [r["median_fill_rate_retention_vs_a"] for r in rows]
    ys = [r["pct_toxicity_reduction_vs_a"] for r in rows]
    ss = [r["posterior_sensitivity"] for r in rows]
    ax.plot(xs, ys, "-", color="0.6", lw=1.0, zorder=1)
    feas_set = set(feasible["feasible_sensitivities"])
    for x, y, s in zip(xs, ys, ss):
        feasible_pt = s in feas_set
        ax.scatter(
            [x], [y], s=70, zorder=3,
            color=("#2A7F3E" if feasible_pt else "#B0431F"),
            edgecolor="black", linewidth=0.6,
        )
        ax.annotate(
            f"$s={s:.2f}$", (x, y), textcoords="offset points", xytext=(6, 5),
            fontsize=8,
        )
    ax.axvline(FILL_RETENTION_CONSTRAINT, color="C0", ls="--", lw=1.0,
               label=f"fill retention = {FILL_RETENTION_CONSTRAINT:.2f}")
    ax.axhline(
        0.0, color="k", ls=":", lw=0.9,
        label="zero absolute-markout reduction",
    )
    # shade the feasible region (retention >= 0.5, reduction > 0) in data
    # coordinates: freeze the final axis limits first, then add a rectangle
    # spanning retention >= 0.5 and reduction >= 0 up to the axis edges.
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.add_patch(plt.Rectangle(
        (FILL_RETENTION_CONSTRAINT, 0.0),
        ax.get_xlim()[1] - FILL_RETENTION_CONSTRAINT,
        ax.get_ylim()[1],
        color="#2A7F3E", alpha=0.06, zorder=0,
    ))
    ax.set_xlabel("Median fill-rate retention vs signal-blind baseline")
    ax.set_ylabel("Absolute-markout reduction at 5 s vs baseline (%)")
    ax.set_title("Posterior-sensitivity fill-markout frontier (30 matched simulations)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"  wrote {OUT_PDF}")


def _make_table(rows: List[dict]) -> None:
    lines = [
        "% Auto-generated by build_layer3_d1_skew_frontier.py (P6c). Do not edit by hand.",
        "\\begin{tabular}{rrrrr}",
        "\\hline\\hline",
        "$s$ & Abs.\\ markout red.\\ (\\%) & Fill retention & Sign-stable & Mean fills/seed \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['posterior_sensitivity']:.2f} & "
            f"{r['pct_toxicity_reduction_vs_a']:+.2f} & "
            f"{r['median_fill_rate_retention_vs_a']:.3f} & "
            f"{r['sign_stable_fraction_b_lower_than_a']:.2f} & "
            f"{r['mean_fills_per_seed']:.0f} \\\\"
        )
    lines += ["\\hline\\hline", "\\end{tabular}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_TEX}")


def main():
    print("=== Layer-3 D1 posterior-sensitivity skew frontier (30 seeds) ===")
    t0 = time.time()
    per_seed = []
    for seed in base.SEEDS:
        cell = _seed_cell(seed)
        if cell is None:
            print(f"  seed {seed}: skipped (too few windows)")
            continue
        per_seed.append(cell)
    print(f"  evaluated {len(per_seed)} seeds in {time.time() - t0:.1f}s")

    rows = _aggregate(per_seed)
    checks = _cross_check_endpoints(rows, per_seed)
    print(f"  endpoint cross-check vs committed 10msc: PASS")
    feasible = _feasible_region(rows)

    print("\n  sensitivity  tox_reduction%%  fill_retention  sign_stable")
    for r in rows:
        print(
            f"    s={r['posterior_sensitivity']:.2f}    "
            f"{r['pct_toxicity_reduction_vs_a']:+7.3f}        "
            f"{r['median_fill_rate_retention_vs_a']:.3f}         "
            f"{r['sign_stable_fraction_b_lower_than_a']:.2f}"
        )
    print(
        f"\n  feasible (reduction>0 & retention>={FILL_RETENTION_CONSTRAINT}): "
        f"{feasible['feasible_sensitivities']}  best s={feasible['best_feasible_sensitivity']}"
    )

    payload = {
        "artifact": "layer3_d1_skew_frontier",
        "description": (
            "Posterior-sensitivity skew frontier for the D1 Layer-3 experiment. "
            "Reuses the prompt-10msc protocol verbatim (config meta_order_smoke, "
            "T=600 s, seeds 1..30, same HMM posterior, MarkoutHarness, matched "
            "fill RNG); sweeps posterior_sensitivity with max_skew_ticks=5.0 and "
            "the fill-logistic fixed at SkewAgentConfig defaults. Episode A "
            "(s=0.0) is the Hawkes-blind matched baseline within each seed."
        ),
        "config": "configs/meta_order_smoke.json",
        "horizon_seconds": base.HORIZON_SECONDS,
        "headline_horizon_seconds": HEADLINE,
        "seeds": list(base.SEEDS),
        "n_seeds_evaluated": len(per_seed),
        "posterior_sensitivity_grid": list(SENSITIVITIES),
        "fixed_skew_agent_config": {
            "base_spread_ticks": 2.0,
            "max_skew_ticks": 5.0,
            "inventory_penalty": 0.05,
            "fill_probability_p_at_zero": 0.4,
            "fill_probability_p_at_max": 0.05,
            "tick_size_in_pips": 0.1,
            "note": "only posterior_sensitivity is swept; all else default",
        },
        "fill_retention_constraint": FILL_RETENTION_CONSTRAINT,
        "frontier": rows,
        "feasible_region": feasible,
        "endpoint_cross_check_vs_10msc": checks,
        "per_seed": per_seed,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {OUT_JSON}")

    _make_figure(rows, feasible)
    _make_table(rows)


if __name__ == "__main__":
    main()
