"""
experiment.py — reproducible end-to-end dissertation experiment runner.

Glues together: config load -> ensemble simulate -> observation -> summary
-> single-run comparison -> ensemble-level comparison -> per-member regime
and meta-order recovery -> recovery evaluation -> manifest. Outputs land in
a single experiment directory whose name is the user-chosen `experiment_label`.

Public surface
--------------
- ExperimentConfig dataclass
- run_experiment(cfg) -> str (path of the experiment directory)
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .artifact import _utc_timestamp_compact, default_runs_root, load_run
from .comparison import COMPARISON_FILENAME
from .config import SimulatorConfig, MetaOrderWindow, config_from_json, config_to_json
from .ensemble import (
    ENSEMBLE_SUMMARY_FILENAME,
    run_ensemble,
)
from .ensemble_comparison import (
    ENSEMBLE_COMPARISON_FILENAME,
    save_ensemble_comparison,
)
from .evaluation import SUMMARY_FILENAME
from .recovery import (
    MetaOrderBaselineConfig,
    RegimeBaselineConfig,
    compute_window_features,
    recover_regimes_baseline,
    score_meta_order_activity,
)
from .recovery_eval import RecoveryEvaluation, evaluate_meta_order_recovery
from .recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm


EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
EXPERIMENT_RECOVERY_DIRNAME = "recovery"
EXPERIMENT_EVALUATION_FILENAME = "evaluation_summary.json"
EXPERIMENT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one dissertation experiment.

    Fields
    ------
    config_path : str
        Path to the SimulatorConfig JSON.
    seeds : sequence of int
        Pairwise-distinct integer seeds.
    output_root : str
        Directory under which the experiment directory is created.
    envelope_path : str or None
        Optional envelope path. If supplied, both single-run comparison and
        ensemble-level comparison artefacts are produced.
    regime_baseline_cfg : RegimeBaselineConfig or None
        If supplied, the regime baseline is run on each member.
    meta_order_baseline_cfg : MetaOrderBaselineConfig or None
        If supplied AND `recovery_method == "threshold"`, the meta-order
        threshold detector is run on each member; the recovery evaluation
        is computed if the simulator config has at least one MetaOrderWindow.
    recovery_method : str, default "hmm".
        Layer-2 meta-order recovery method. Accepted values: "hmm"
        (the primary, probabilistic recovery method, default) and
        "threshold" (the deterministic comparison baseline). When "hmm",
        the threshold detector and `meta_order_baseline_cfg` are not used
        for meta-order recovery; the HMM is run on each member's window
        features instead.
    hmm_recovery_cfg : HMMRecoveryConfig or None
        Optional configuration override for the HMM recovery. When
        `recovery_method == "hmm"` and this field is None, sensible
        defaults are applied (n_states=2, max_iter=200, emission features
        on (recent_event_rate, directional_imbalance)). Ignored when
        `recovery_method == "threshold"`.
    hmm_window_seconds : float, default 5.0
        Sliding-window length (seconds) used to compute the
        WindowFeatures fed to the HMM. Only consulted when
        `recovery_method == "hmm"`.
    experiment_label : str
        Subdirectory name. Used verbatim — caller must avoid collisions.
    """
    config_path: str
    seeds: Sequence[int]
    output_root: str
    envelope_path: Optional[str] = None
    regime_baseline_cfg: Optional[RegimeBaselineConfig] = None
    meta_order_baseline_cfg: Optional[MetaOrderBaselineConfig] = None
    recovery_method: str = "hmm"
    hmm_recovery_cfg: Optional[HMMRecoveryConfig] = None
    hmm_window_seconds: float = 5.0
    experiment_label: str = "experiment"


def _windows_to_dicts(windows: Sequence[MetaOrderWindow]) -> list:
    return [
        {
            "start_time": w.start_time,
            "end_time": w.end_time,
            "alpha": w.alpha,
            "target_event_types": list(w.target_event_types),
        }
        for w in windows
    ]


def _viterbi_to_intervals(
    viterbi, window_starts, window_seconds, active_state
):
    """Map an HMM Viterbi label sequence to (start, end, score) active intervals.

    `score` is set to 1.0 for compatibility with `RecoveryEvaluation`.
    Helper retained as a free function so callers (including the
    dissertation-artefact runners) can reuse it.
    """
    import numpy as np

    intervals = []
    if viterbi.size == 0:
        return intervals
    is_active = (viterbi == active_state).astype(np.int8)
    diff = np.diff(is_active, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        t_start = float(window_starts[s])
        t_end = float(window_starts[e - 1] + window_seconds)
        intervals.append((t_start, t_end, 1.0))
    return intervals


def _run_hmm_meta_order_recovery(
    observed,
    window_seconds: float,
    cfg: Optional[HMMRecoveryConfig],
):
    """Run the HMM Layer-2 recovery on a single observed trace.

    Returns
    -------
    intervals : list of (start, end, score) active intervals.
    hmm_summary : dict with the per-member HMM diagnostics (transition
        matrix, emission means/stds, log-likelihood, n_iter, converged).
    """
    import numpy as np

    features = compute_window_features(
        observed,
        window_seconds=window_seconds,
        hop_seconds=window_seconds,
    )
    if not features:
        return [], {
            "n_windows": 0,
            "converged": False,
            "log_likelihood": None,
        }
    if cfg is None:
        cfg = HMMRecoveryConfig(n_states=2, max_iter=200, tol_relative_ll=1e-6)
    if len(features) < cfg.n_states:
        return [], {
            "n_windows": int(len(features)),
            "converged": False,
            "log_likelihood": None,
            "skipped_reason": "fewer windows than states",
        }
    res = recover_regimes_hmm(features, cfg)
    active_state = int(np.argmax(res.emission_means[:, 0]))
    intervals = _viterbi_to_intervals(
        res.viterbi_labels, res.window_starts, window_seconds, active_state
    )
    summary = {
        "n_windows": int(res.n_windows),
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        "converged_reason": str(res.converged_reason),
        "log_likelihood": float(res.log_likelihood),
        "active_state_index": active_state,
        "transition_matrix": res.transition_matrix.tolist(),
        "emission_means": res.emission_means.tolist(),
        "emission_stds": res.emission_stds.tolist(),
    }
    return intervals, summary


def _evaluation_to_dict(ev: RecoveryEvaluation) -> dict:
    import math
    delay = ev.mean_detection_delay_seconds
    return {
        "precision": ev.precision,
        "recall": ev.recall,
        "f1": ev.f1,
        "mean_detection_delay_seconds": (
            None if delay is None or (isinstance(delay, float) and math.isnan(delay))
            else delay
        ),
        "false_positive_intervals": ev.false_positive_intervals,
        "false_negative_intervals": ev.false_negative_intervals,
        "confusion_table": ev.confusion_table,
    }


def run_experiment(cfg: ExperimentConfig) -> str:
    """Execute the full experiment and return the experiment directory path.

    Side effects: writes one ensemble directory and one experiment manifest.
    """
    seeds = list(int(s) for s in cfg.seeds)
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be pairwise distinct")
    if cfg.recovery_method not in {"hmm", "threshold"}:
        raise ValueError(
            f"recovery_method must be 'hmm' or 'threshold'; got {cfg.recovery_method!r}"
        )

    base_cfg: SimulatorConfig = config_from_json(cfg.config_path)
    timestamp = _utc_timestamp_compact()
    experiment_dir = os.path.join(
        cfg.output_root, f"{cfg.experiment_label}_{timestamp}"
    )
    os.makedirs(experiment_dir, exist_ok=False)

    # Persist the resolved config alongside the experiment for reproducibility.
    config_to_json(base_cfg, os.path.join(experiment_dir, "resolved_config.json"))

    # Run the ensemble (members go inside experiment_dir).
    ensemble_dir = run_ensemble(
        base_cfg,
        seeds=seeds,
        runs_root=experiment_dir,
        observe=True,
        envelope_path=cfg.envelope_path,
        timestamp_prefix=cfg.experiment_label,
    )

    # Optional ensemble-level comparison.
    ensemble_comparison_path = None
    if cfg.envelope_path is not None:
        ensemble_comparison_path = str(
            save_ensemble_comparison(ensemble_dir, cfg.envelope_path, mode="mean")
        )

    # Per-member recovery + evaluation.
    recovery_dir = os.path.join(experiment_dir, EXPERIMENT_RECOVERY_DIRNAME)
    os.makedirs(recovery_dir, exist_ok=True)

    per_member_evaluations = []
    for seed in seeds:
        member_dir = os.path.join(
            ensemble_dir, f"{cfg.experiment_label}_{seed}"
        )
        run = load_run(member_dir)
        member_record: dict = {"seed": int(seed), "member_dir": member_dir}

        if cfg.regime_baseline_cfg is not None:
            transitions = recover_regimes_baseline(
                run.observed,
                cfg.regime_baseline_cfg,
                horizon=base_cfg.horizon,
            )
            member_record["regime_transitions"] = transitions

        # Layer-2 meta-order recovery: HMM (default) or threshold (baseline)
        intervals = None
        recovery_ran = False
        if cfg.recovery_method == "hmm":
            intervals, hmm_summary = _run_hmm_meta_order_recovery(
                run.observed,
                window_seconds=cfg.hmm_window_seconds,
                cfg=cfg.hmm_recovery_cfg,
            )
            member_record["meta_order_inferred_intervals"] = intervals
            member_record["hmm_summary"] = hmm_summary
            recovery_ran = True
        elif cfg.recovery_method == "threshold":
            if cfg.meta_order_baseline_cfg is not None:
                intervals = score_meta_order_activity(
                    run.observed,
                    cfg.meta_order_baseline_cfg,
                    horizon=base_cfg.horizon,
                )
                member_record["meta_order_inferred_intervals"] = intervals
                recovery_ran = True

        if recovery_ran and base_cfg.meta_order is not None and base_cfg.meta_order.windows:
            ev = evaluate_meta_order_recovery(
                base_cfg.meta_order.windows,
                intervals,
                horizon=base_cfg.horizon,
            )
            member_record["meta_order_evaluation"] = _evaluation_to_dict(ev)

        per_member_evaluations.append(member_record)
        with open(
            os.path.join(recovery_dir, f"member_{seed}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(member_record, f, indent=2)

    # Aggregate recovery evaluation summary. Aggregates are produced
    # whenever any meta-order recovery (HMM or threshold) ran on a
    # simulator config that has at least one true MetaOrderWindow.
    eval_summary: dict = {
        "num_members": len(seeds),
        "members": per_member_evaluations,
        "recovery_method": cfg.recovery_method,
    }
    has_evaluations = any(
        "meta_order_evaluation" in m for m in per_member_evaluations
    )
    if has_evaluations and base_cfg.meta_order and base_cfg.meta_order.windows:
        precisions = [m["meta_order_evaluation"]["precision"]
                      for m in per_member_evaluations
                      if "meta_order_evaluation" in m]
        recalls = [m["meta_order_evaluation"]["recall"]
                   for m in per_member_evaluations
                   if "meta_order_evaluation" in m]
        f1s = [m["meta_order_evaluation"]["f1"]
               for m in per_member_evaluations
               if "meta_order_evaluation" in m]
        delays = [m["meta_order_evaluation"]["mean_detection_delay_seconds"]
                  for m in per_member_evaluations
                  if "meta_order_evaluation" in m
                  and m["meta_order_evaluation"]["mean_detection_delay_seconds"] is not None]
        if precisions:
            import statistics
            eval_summary["meta_order_aggregate"] = {
                "precision_mean": statistics.mean(precisions),
                "recall_mean": statistics.mean(recalls),
                "f1_mean": statistics.mean(f1s),
                "mean_detection_delay_seconds_mean": (
                    statistics.mean(delays) if delays else None
                ),
                "n": len(precisions),
            }
    with open(
        os.path.join(experiment_dir, EXPERIMENT_EVALUATION_FILENAME),
        "w", encoding="utf-8",
    ) as f:
        json.dump(eval_summary, f, indent=2)

    # Manifest.
    manifest = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_label": cfg.experiment_label,
        "experiment_timestamp_utc": timestamp,
        "experiment_dir": experiment_dir,
        "config_path": os.path.abspath(cfg.config_path),
        "envelope_path": (
            os.path.abspath(cfg.envelope_path) if cfg.envelope_path else None
        ),
        "seeds": seeds,
        "ensemble_dir": ensemble_dir,
        "ensemble_summary_file": os.path.join(ensemble_dir, ENSEMBLE_SUMMARY_FILENAME),
        "ensemble_comparison_file": ensemble_comparison_path,
        "recovery_dir": recovery_dir,
        "evaluation_summary_file": os.path.join(
            experiment_dir, EXPERIMENT_EVALUATION_FILENAME
        ),
        "true_meta_order_windows": _windows_to_dicts(
            base_cfg.meta_order.windows if base_cfg.meta_order else ()
        ),
        "regime_baseline_used": cfg.regime_baseline_cfg is not None,
        # `meta_order_baseline_used` is True iff *any* meta-order recovery
        # (HMM or threshold) ran AND produced an evaluation. Preserved for
        # back-compat; new code should consult `recovery_method`.
        "meta_order_baseline_used": bool(has_evaluations),
        "recovery_method": cfg.recovery_method,
    }
    with open(
        os.path.join(experiment_dir, EXPERIMENT_MANIFEST_FILENAME),
        "w", encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    return experiment_dir


__all__ = [
    "ExperimentConfig",
    "EXPERIMENT_MANIFEST_FILENAME",
    "EXPERIMENT_RECOVERY_DIRNAME",
    "EXPERIMENT_EVALUATION_FILENAME",
    "EXPERIMENT_SCHEMA_VERSION",
    "run_experiment",
]
