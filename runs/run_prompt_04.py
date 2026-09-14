"""Driver for prompt 04 — recovery baseline at production scale.

Runs meta_order_smoke.json with seeds 1..50 through experiment.run_experiment
with MetaOrderBaselineConfig set, so per-member recovery is computed.
Idempotent.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

from simulator.experiment import ExperimentConfig, run_experiment
from simulator.recovery import MetaOrderBaselineConfig

DATE = "2026-05-06"
RECOVERY_ROOT = os.path.join(REPO, "runs", f"recovery_meta_order_smoke_{DATE}")
CONFIG = os.path.join(REPO, "configs", "meta_order_smoke.json")
ENVELOPE = os.path.join(REPO, "targets", "real_fx_envelope.json")
LABEL = "recovery_meta_order_smoke_50seeds"
SEEDS = tuple(range(1, 51))

# Detector config — anchored to the dissertation's worked example. DO NOT change.
BASELINE = MetaOrderBaselineConfig(
    window_seconds=5.0,
    score_threshold=0.5,
    direction="up",
    min_active_seconds=5.0,
)


def _final_dir() -> str:
    return RECOVERY_ROOT


def main():
    if os.path.isdir(_final_dir()):
        # Find the experiment dir inside; if it already contains 50 members, reuse.
        existing = [
            d for d in os.listdir(_final_dir())
            if os.path.isdir(os.path.join(_final_dir(), d)) and d.startswith(LABEL)
        ]
        if existing:
            experiment_dir = os.path.join(_final_dir(), existing[0])
            print(f"reusing experiment dir {experiment_dir}")
        else:
            experiment_dir = None
    else:
        experiment_dir = None

    if experiment_dir is None:
        os.makedirs(_final_dir(), exist_ok=True)
        cfg = ExperimentConfig(
            config_path=CONFIG,
            seeds=SEEDS,
            output_root=_final_dir(),
            envelope_path=ENVELOPE,
            meta_order_baseline_cfg=BASELINE,
            experiment_label=LABEL,
        )
        t0 = time.time()
        experiment_dir = run_experiment(cfg)
        elapsed = time.time() - t0
        print(f"experiment_dir = {experiment_dir} ({elapsed:.1f} s)")

    # Aggregate per-member recovery.
    eval_summary = json.load(
        open(os.path.join(experiment_dir, "evaluation_summary.json"))
    )

    rows = []
    for member in eval_summary.get("members", []):
        seed = member["seed"]
        ev = member.get("meta_order_evaluation")
        if ev is None:
            continue
        rows.append({
            "seed": seed,
            "precision": ev.get("precision"),
            "recall": ev.get("recall"),
            "f1": ev.get("f1"),
            "mean_detection_delay_seconds": ev.get("mean_detection_delay_seconds"),
            "fp_intervals": len(ev.get("false_positive_intervals", [])),
            "fn_intervals": len(ev.get("false_negative_intervals", [])),
        })

    csv_path = os.path.join(_final_dir(), "recovery_summary.csv")
    if rows:
        headers = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"recovery_summary.csv -> {csv_path} ({len(rows)} rows)")

    def _stat(field):
        xs = [r[field] for r in rows if r[field] is not None]
        return {
            "n": len(xs),
            "mean": statistics.mean(xs) if xs else None,
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
        }

    aggregate = {
        "n_seeds": len(rows),
        "precision": _stat("precision"),
        "recall": _stat("recall"),
        "f1": _stat("f1"),
        "mean_detection_delay_seconds": _stat("mean_detection_delay_seconds"),
        "fp_intervals": _stat("fp_intervals"),
        "fn_intervals": _stat("fn_intervals"),
    }
    agg_path = os.path.join(_final_dir(), "recovery_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump({"experiment_dir": experiment_dir, "aggregate": aggregate}, f, indent=2)
    print(f"recovery_aggregate.json -> {agg_path}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
