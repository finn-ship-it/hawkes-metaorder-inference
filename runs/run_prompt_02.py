"""Driver for prompt 02 — three 50-seed production ensembles.

Idempotent: if a target dir already exists, it is skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

from simulator.experiment import ExperimentConfig, run_experiment
from simulator.artifact import load_run

DATE = "2026-05-06"
ENSEMBLE_ROOT = os.path.join(REPO, "runs", f"ensembles_{DATE}")
ENVELOPE = os.path.join(REPO, "targets", "real_fx_envelope.json")
SEEDS = tuple(range(1, 51))

CONFIGS = [
    ("first_milestone", "first_milestone.json"),
    ("two_regime_smoke", "two_regime_smoke.json"),
    ("meta_order_smoke", "meta_order_smoke.json"),
]


def _final_dir(name: str) -> str:
    return os.path.join(ENSEMBLE_ROOT, f"{name}_50seeds")


def run_one(name: str, config_basename: str) -> dict:
    """Run a single 50-seed ensemble. Returns aggregate stats."""
    final_dir = _final_dir(name)
    if os.path.isdir(final_dir):
        print(f"[{name}] already complete — reusing {final_dir}")
    else:
        config_path = os.path.join(REPO, "configs", config_basename)
        os.makedirs(ENSEMBLE_ROOT, exist_ok=True)
        cfg = ExperimentConfig(
            config_path=config_path,
            seeds=SEEDS,
            output_root=ENSEMBLE_ROOT,
            envelope_path=ENVELOPE,
            experiment_label=f"{name}_50seeds",
        )
        t0 = time.time()
        experiment_dir = run_experiment(cfg)
        elapsed = time.time() - t0
        print(f"[{name}] experiment_dir = {experiment_dir} ({elapsed:.1f} s)")
        # Rename to drop the _<timestamp> suffix so directory matches prompt-02 acceptance.
        os.rename(experiment_dir, final_dir)

    # Inspect: ensemble subdir lives inside final_dir, with members inside it.
    inner = next(
        os.path.join(final_dir, d)
        for d in os.listdir(final_dir)
        if os.path.isdir(os.path.join(final_dir, d)) and d != "recovery"
    )
    member_dirs = sorted(
        os.path.join(inner, d) for d in os.listdir(inner)
        if os.path.isdir(os.path.join(inner, d)) and d.startswith(name)
    )

    rho_list = []
    crossed_total = 0
    total_rate_list = []
    for md in member_dirs:
        try:
            run = load_run(md)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"failed to load member {md}: {exc}") from exc
        s = json.load(open(os.path.join(md, "summary.json")))
        rho = s["endogeneity"]["spectral_radius"]
        crossed = s["observation"]["num_crossed_observations"]
        total_rate = s["latent"]["total_rate"]
        rho_list.append(rho)
        crossed_total += int(crossed)
        total_rate_list.append(total_rate)

    ec_path = os.path.join(inner, "ensemble_comparison.json")
    es_path = os.path.join(inner, "ensemble_summary.json")
    ec = json.load(open(ec_path)) if os.path.exists(ec_path) else None
    es = json.load(open(es_path)) if os.path.exists(es_path) else None

    return {
        "name": name,
        "final_dir": final_dir,
        "inner_ensemble_dir": inner,
        "n_members": len(member_dirs),
        "rho_mean": statistics.mean(rho_list),
        "rho_std": statistics.pstdev(rho_list),
        "rho_min": min(rho_list),
        "rho_max": max(rho_list),
        "crossed_total": crossed_total,
        "total_rate_mean": statistics.mean(total_rate_list),
        "total_rate_std": statistics.pstdev(total_rate_list),
        "envelope_overall_status": (ec or {}).get("overall_status"),
        "envelope_status_counts": (ec or {}).get("status_counts"),
        "ensemble_summary_keys": list((es or {}).keys()),
    }


def main():
    print(f"ENSEMBLE_ROOT = {ENSEMBLE_ROOT}")
    print(f"SEEDS = 1..50 ({len(SEEDS)} seeds)")
    results = []
    for name, basename in CONFIGS:
        results.append(run_one(name, basename))
    out = os.path.join(ENSEMBLE_ROOT, "run_prompt_02_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
