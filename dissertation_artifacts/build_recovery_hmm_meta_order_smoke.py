"""Build artefacts: dissertation_artifacts/data/recovery_hmm_meta_order_smoke.json
                  dissertation_artifacts/figures/recovery_hmm_posterior.pdf.

Runs the Layer-2 HMM recovery on the existing meta_order_smoke 50-seed
ensemble (`runs/ensembles_2026-05-06/meta_order_smoke_50seeds/`), evaluates
F1 / detection-delay against the simulator's true meta-order windows, and
compares against the threshold-detector summary in
`dissertation_artifacts/data/recovery_summary.csv`.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ENSEMBLE_DIR = (
    REPO_ROOT
    / "runs"
    / "ensembles_2026-05-06"
    / "meta_order_smoke_50seeds"
    / "ensemble_meta_order_smoke_50seeds"
)
THRESHOLD_CSV = HERE / "data" / "recovery_summary.csv"
OUT_JSON = HERE / "data" / "recovery_hmm_meta_order_smoke.json"
OUT_PDF = HERE / "figures" / "recovery_hmm_posterior.pdf"
SRC_PATH = str(REPO_ROOT / "src")


def _ensure_simulator_on_path() -> None:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


def _viterbi_to_intervals(
    viterbi: np.ndarray,
    window_starts: np.ndarray,
    window_seconds: float,
    active_state: int,
) -> List[Tuple[float, float, float]]:
    """Map Viterbi labels into (start, end, score) active intervals.

    `score` is set to 1.0 for compatibility with `RecoveryEvaluation`.
    """
    intervals: List[Tuple[float, float, float]] = []
    if viterbi.size == 0:
        return intervals
    is_active = (viterbi == active_state).astype(np.int8)
    diff = np.diff(is_active, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        t_start = float(window_starts[s])
        # End is the right edge of the last active window
        t_end = float(window_starts[e - 1] + window_seconds)
        intervals.append((t_start, t_end, 1.0))
    return intervals


def _evaluate_seed(seed_dir: Path, window_seconds: float = 5.0):
    _ensure_simulator_on_path()
    from simulator.artifact import load_run
    from simulator.config import MetaOrderWindow
    from simulator.recovery import compute_window_features
    from simulator.recovery_eval import evaluate_meta_order_recovery
    from simulator.recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm

    run = load_run(str(seed_dir))
    horizon = float(run.config.horizon)

    features = compute_window_features(
        run.observed,
        window_seconds=window_seconds,
        hop_seconds=window_seconds,
        horizon=horizon,
    )
    if len(features) < 4:
        return None

    cfg = HMMRecoveryConfig(
        n_states=2,
        emission_features=("recent_event_rate", "directional_imbalance"),
        max_iter=200,
        tol_relative_ll=1e-6,
        seed=int(getattr(run, "seed", 0) or 0),
    )
    res = recover_regimes_hmm(features, cfg)

    # Pick the active state as the one with higher emission mean on rate
    active_state = int(np.argmax(res.emission_means[:, 0]))
    intervals = _viterbi_to_intervals(
        res.viterbi_labels, res.window_starts, window_seconds, active_state
    )

    # True meta-order windows from the run
    true_windows = [
        MetaOrderWindow(
            start_time=float(w.start_time),
            end_time=float(w.end_time),
            alpha=float(w.alpha),
            target_event_types=tuple(int(t) for t in w.target_event_types),
        )
        for w in run.config.meta_order.windows
    ]
    eval_ = evaluate_meta_order_recovery(
        true_windows=true_windows,
        inferred_intervals=intervals,
        horizon=horizon,
        time_resolution=0.1,
    )

    return {
        "seed_dir": str(seed_dir.name),
        "horizon": horizon,
        "n_windows": int(res.n_windows),
        "hmm_converged": bool(res.converged),
        "hmm_n_iter": int(res.n_iter),
        "active_state_index": active_state,
        "transition_matrix": res.transition_matrix.tolist(),
        "emission_means": res.emission_means.tolist(),
        "emission_stds": res.emission_stds.tolist(),
        "log_likelihood": float(res.log_likelihood),
        "precision": float(eval_.precision),
        "recall": float(eval_.recall),
        "f1": float(eval_.f1),
        "mean_detection_delay_seconds": float(eval_.mean_detection_delay_seconds),
        "fp_intervals": int(len(eval_.false_positive_intervals)),
        "fn_intervals": int(len(eval_.false_negative_intervals)),
        "n_inferred_intervals": int(len(intervals)),
        "n_true_windows": int(len(true_windows)),
        "viterbi_labels": res.viterbi_labels.tolist(),
        "window_starts": res.window_starts.tolist(),
        "posterior": res.posterior.tolist(),
    }


def main():
    _ensure_simulator_on_path()
    print("=== Layer-2 HMM recovery on meta_order_smoke 50-seed ensemble ===")
    seed_dirs = sorted(
        [p for p in ENSEMBLE_DIR.iterdir() if p.is_dir() and p.name.startswith("meta_order_smoke_50seeds_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    print(f"  {len(seed_dirs)} seed runs found in {ENSEMBLE_DIR}")

    per_seed = []
    t0 = time.time()
    for sdir in seed_dirs:
        out = _evaluate_seed(sdir)
        if out is None:
            print(f"  {sdir.name}: skipped (too few windows)")
            continue
        per_seed.append(out)
        print(
            f"  {sdir.name}: F1={out['f1']:.3f} prec={out['precision']:.3f} "
            f"recall={out['recall']:.3f} delay={out['mean_detection_delay_seconds']:.2f}s "
            f"conv={out['hmm_converged']}"
        )
    dt = time.time() - t0
    print(f"\n  total wall: {dt:.1f}s")

    # Aggregate
    arr_f1 = np.array([s["f1"] for s in per_seed])
    arr_prec = np.array([s["precision"] for s in per_seed])
    arr_recall = np.array([s["recall"] for s in per_seed])
    arr_delay = np.array([s["mean_detection_delay_seconds"] for s in per_seed])

    # Threshold-detector comparator (existing CSV)
    thresh = pd.read_csv(THRESHOLD_CSV)
    aggregate = {
        "ensemble": "meta_order_smoke_50seeds",
        "n_seeds": int(len(per_seed)),
        "window_seconds": 5.0,
        "hmm_default_recovery": True,
        "hmm_aggregate": {
            "f1_mean": float(np.mean(arr_f1)),
            "f1_std": float(np.std(arr_f1)),
            "precision_mean": float(np.mean(arr_prec)),
            "precision_std": float(np.std(arr_prec)),
            "recall_mean": float(np.mean(arr_recall)),
            "recall_std": float(np.std(arr_recall)),
            "detection_delay_mean": float(np.mean(arr_delay)),
            "detection_delay_std": float(np.std(arr_delay)),
        },
        "threshold_aggregate": {
            "f1_mean": float(thresh["f1"].mean()),
            "f1_std": float(thresh["f1"].std()),
            "precision_mean": float(thresh["precision"].mean()),
            "precision_std": float(thresh["precision"].std()),
            "recall_mean": float(thresh["recall"].mean()),
            "recall_std": float(thresh["recall"].std()),
            "detection_delay_mean": float(thresh["mean_detection_delay_seconds"].mean()),
            "detection_delay_std": float(thresh["mean_detection_delay_seconds"].std()),
        },
        "per_seed": per_seed,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(aggregate, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(
        f"\n--- HMM aggregate (50 seeds) ---\n"
        f"  F1 mean = {aggregate['hmm_aggregate']['f1_mean']:.3f}  "
        f"std = {aggregate['hmm_aggregate']['f1_std']:.3f}\n"
        f"  delay mean = {aggregate['hmm_aggregate']['detection_delay_mean']:.2f}s\n"
        f"--- Threshold (50 seeds, baseline) ---\n"
        f"  F1 mean = {aggregate['threshold_aggregate']['f1_mean']:.3f}  "
        f"std = {aggregate['threshold_aggregate']['f1_std']:.3f}"
    )

    # Figure: posterior trace for one representative seed (the one with median F1)
    _make_posterior_figure(per_seed)


def _make_posterior_figure(per_seed) -> None:
    """Draw the HMM posterior trace for the median-F1 seed, overlaying its
    known pressure periods. Reads only per-seed records already computed
    (posterior, window_starts, active_state_index, seed_dir, f1) and writes
    only OUT_PDF; it touches no JSON, so it can regenerate the figure from a
    saved summary without re-running the HMM.
    """
    arr_f1 = np.array([s["f1"] for s in per_seed])
    median_seed_idx = int(np.argmin(np.abs(arr_f1 - np.median(arr_f1))))
    rep = per_seed[median_seed_idx]
    posterior = np.array(rep["posterior"])
    window_starts = np.array(rep["window_starts"])
    active_state = rep["active_state_index"]
    p_active = posterior[:, active_state]

    # Re-load the seed to get its true windows for overlay
    seed_dir = ENSEMBLE_DIR / rep["seed_dir"]
    from simulator.artifact import load_run

    run = load_run(str(seed_dir))
    true_windows = run.config.meta_order.windows

    seed_label = rep["seed_dir"].split("_")[-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_starts, p_active, color="C0", lw=1.6, label="HMM posterior p(active | O)")
    ax.fill_between(window_starts, 0, p_active, alpha=0.2)
    for w in true_windows:
        ax.axvspan(w.start_time, w.end_time, alpha=0.18, color="C3",
                   label="known pressure period" if w == true_windows[0] else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("posterior p(active state)")
    ax.set_title(
        f"HMM posterior on a known pressure period\n"
        f"(seed {seed_label}, $F_1 = {rep['f1']:.3f}$)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
