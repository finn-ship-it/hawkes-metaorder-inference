"""
runner.py — single public entry point for one simulator run.

Responsibilities:
    1. Build or load a validated first-milestone SimulatorConfig.
    2. Instantiate the simulator core.
    3. Execute one simulation at the fixed seed.
    4. Persist the result through the persistence layer in artifact.py.

This module deliberately does not implement an agent loop, an observation
operator, or any form of ensemble orchestration; those are outside the scope
of the first coding milestone.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional
import numpy as np

from .config import (
    SimulatorConfig,
    ExponentialKernelParams,
    BaselineParams,
    KernelFamily,
    validate,
    validate_first_milestone,
    config_from_json,
)
from .hawkes_core import HawkesSimulator, RunTrace
from .artifact import save_run, default_runs_root
from .observation import (
    ObservationConfig,
    ObservationOperator,
    default_observation_config,
)
from .evaluation import save_summary
from .comparison import save_comparison


# ---------------------------------------------------------------------------
# Default configuration file location
# ---------------------------------------------------------------------------


DEFAULT_FIRST_MILESTONE_CONFIG_RELPATH = os.path.join(
    "configs", "first_milestone.json"
)


def default_first_milestone_config_path() -> str:
    """Absolute path to `<repo_root>/configs/first_milestone.json`.

    Resolved from this module's location so the path does not depend on the
    process working directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    repo_root = os.path.dirname(src_dir)
    return os.path.join(repo_root, DEFAULT_FIRST_MILESTONE_CONFIG_RELPATH)


# ---------------------------------------------------------------------------
# Default first-milestone configuration
# ---------------------------------------------------------------------------


def build_default_first_milestone_config(
    seed: Optional[int] = None,
    horizon: Optional[float] = None,
    config_path: Optional[str] = None,
) -> SimulatorConfig:
    """Return a validated SimulatorConfig satisfying the first-milestone gate.

    The canonical parameter values live in
    `<repo_root>/configs/first_milestone.json` and are loaded from that file
    through `config_from_json`. If the file is absent, a programmatic fallback
    reproducing the original parameter set is used so that the simulator core
    remains runnable in environments where the configuration directory has
    not been provisioned.

    Parameters
    ----------
    seed : int, optional
        Override for the seed recorded in the loaded configuration.
    horizon : float, optional
        Override for the horizon recorded in the loaded configuration.
    config_path : str, optional
        Override for the configuration file path. Defaults to
        `default_first_milestone_config_path()`.
    """
    path = (
        config_path
        if config_path is not None
        else default_first_milestone_config_path()
    )

    if os.path.isfile(path):
        cfg = config_from_json(path)
    else:
        cfg = _programmatic_first_milestone_fallback()

    if seed is not None or horizon is not None:
        cfg = SimulatorConfig(
            **{
                **cfg.__dict__,
                "seed": int(seed) if seed is not None else cfg.seed,
                "horizon": float(horizon) if horizon is not None else cfg.horizon,
            }
        )

    # Gate by scope: single-regime configurations pass the first-milestone
    # gate; multi-regime configurations are validated under the general rules.
    if cfg.latent_regime.num_regimes == 1 and not cfg.meta_order.enabled:
        validate_first_milestone(cfg)
    else:
        validate(cfg)
    return cfg


def _programmatic_first_milestone_fallback() -> SimulatorConfig:
    """Programmatic fallback used only when the canonical configuration file
    is absent. Parameter values match the original on-disk configuration
    bit-for-bit.
    """
    d = 4
    alpha = 0.1 * np.ones((d, d))
    beta = 2.0 * np.ones((d, d))
    mu = 0.5 * np.ones(d)
    return SimulatorConfig(
        dimension=d,
        horizon=60.0,
        seed=20260423,
        kernel_family=KernelFamily.EXPONENTIAL,
        kernel_params=ExponentialKernelParams(alpha=alpha, beta=beta),
        baseline=BaselineParams(mu=mu),
    )


# ---------------------------------------------------------------------------
# Single-run entry point
# ---------------------------------------------------------------------------


def run_once(
    cfg: Optional[SimulatorConfig] = None,
    runs_root: Optional[str] = None,
    simulator_version: str = "0.1.0",
    observation_config: Optional[ObservationConfig] = None,
    write_summary: bool = False,
    compare_envelope_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Execute one simulation and persist the result.

    Parameters
    ----------
    cfg : SimulatorConfig, optional
        Configuration. If None, the default first-milestone configuration is used.
    runs_root : str, optional
        Base directory for the run artefact. If None, the repository's
        default runs root (<repo_root>/runs) is used.
    simulator_version : str
        Version string recorded in the run metadata.
    observation_config : ObservationConfig, optional
        When provided, the latent run is deterministically projected through
        an ObservationOperator and the resulting ObservedTrace is persisted
        alongside the latent artefact. When None, no observation layer is
        attached.
    write_summary : bool, default False
        When True, compute and persist `summary.json` inside the run
        directory after the latent (and observation, if any) artefacts have
        been saved.
    compare_envelope_path : str, optional
        When supplied, the run is compared against the envelope JSON at
        this path and the resulting report is persisted as comparison.json
        inside the run directory. Implies write_summary=True.
    timestamp : str, optional
        Override for the UTC timestamp used to name the run directory.
        Forwarded to save_run unchanged. Intended for ensemble harnesses
        and reproducibility checks that need predictable paths.

    Returns
    -------
    str
        Absolute path of the persisted run directory.
    """
    if cfg is None:
        cfg = build_default_first_milestone_config()
    else:
        # Accept single-regime (first-milestone) and multi-regime configurations.
        if cfg.latent_regime.num_regimes == 1 and not cfg.meta_order.enabled:
            validate_first_milestone(cfg)
        else:
            validate(cfg)

    simulator = HawkesSimulator(cfg)
    simulator.reset(seed=cfg.seed)
    trace: RunTrace = simulator.simulate()

    observed = None
    if observation_config is not None:
        observed = ObservationOperator(observation_config).project(trace)

    run_dir = save_run(
        cfg=cfg,
        trace=trace,
        runs_root=runs_root,
        simulator_version=simulator_version,
        observed=observed,
        timestamp=timestamp,
    )
    # --compare implies --summarise; summary.json must exist before
    # comparison.json can be written.
    need_summary = bool(write_summary) or (compare_envelope_path is not None)
    if need_summary:
        save_summary(run_dir)
    if compare_envelope_path is not None:
        save_comparison(run_dir, compare_envelope_path)
    return run_dir


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run one simulation of the first-milestone FX-native Hawkes "
            "simulator core and persist the result."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON SimulatorConfig produced by "
            "config.config_to_json. If omitted, the default first-milestone "
            "configuration is used."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the RNG seed recorded in the loaded configuration.",
    )
    p.add_argument(
        "--horizon",
        type=float,
        default=None,
        help=(
            "Override the simulation horizon in seconds recorded in the "
            "loaded configuration."
        ),
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help=(
            "Base directory for the run artefact. Defaults to "
            "<repo_root>/runs."
        ),
    )
    p.add_argument(
        "--observe",
        action="store_true",
        help=(
            "Project the latent trace through the default observation "
            "operator and persist the resulting observable event stream "
            "alongside the latent artefact."
        ),
    )
    p.add_argument(
        "--summarise",
        action="store_true",
        help=(
            "After the run is persisted, compute summary.json inside the "
            "run directory using the deterministic evaluation harness."
        ),
    )
    p.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="ENVELOPE_PATH",
        help=(
            "After the run is summarised, compare the summary against the "
            "envelope JSON at the given path and persist comparison.json in "
            "the run directory. Implies --summarise."
        ),
    )
    p.add_argument(
        "--experiment",
        action="store_true",
        help=(
            "Dispatch to the dissertation experiment runner instead of the "
            "single-run path. Requires --config (configuration JSON) and uses "
            "the supplied --seed list (comma-separated, via --experiment-seeds), "
            "--runs-root, and optional --compare. Adds --experiment-label."
        ),
    )
    p.add_argument(
        "--experiment-seeds",
        type=str,
        default="1,2,3",
        help=(
            "Comma-separated integer seeds used by --experiment. Ignored "
            "without --experiment."
        ),
    )
    p.add_argument(
        "--experiment-label",
        type=str,
        default="experiment",
        help=(
            "Subdirectory label for --experiment output. Ignored without "
            "--experiment."
        ),
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.experiment:
        # Lazy import: experiment.py depends on recovery and ensemble modules
        # that are not needed for the single-run path.
        from .experiment import ExperimentConfig, run_experiment
        if args.config is None:
            print(
                "--experiment requires --config to be supplied",
                file=sys.stderr,
            )
            return 2
        seeds = [int(s.strip()) for s in args.experiment_seeds.split(",") if s.strip()]
        out_root = args.runs_root if args.runs_root is not None else default_runs_root()
        exp_cfg = ExperimentConfig(
            config_path=args.config,
            seeds=tuple(seeds),
            output_root=out_root,
            envelope_path=args.compare,
            experiment_label=args.experiment_label,
        )
        out = run_experiment(exp_cfg)
        print(out)
        return 0

    cfg = build_default_first_milestone_config(
        seed=args.seed,
        horizon=args.horizon,
        config_path=args.config,
    )
    observation_config = default_observation_config() if args.observe else None
    run_dir = run_once(
        cfg=cfg,
        runs_root=args.runs_root,
        observation_config=observation_config,
        write_summary=bool(args.summarise),
        compare_envelope_path=args.compare,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
