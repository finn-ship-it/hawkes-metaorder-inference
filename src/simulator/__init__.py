"""
FX-native latent simulator — D1 simulator core.

This package provides the minimal multivariate Hawkes simulator targeted at the
FX top-of-book event types defined in the dissertation blueprint. The first
coding milestone restricts the configuration to four event types, one latent
regime, no meta-order windows, and the exponential kernel family.

Public surface is intentionally small:
    - config:       typed configuration dataclasses and validation
    - kernels:      kernel evaluation, integrals, and upper bounds
    - hawkes_core:  Ogata thinning simulator

Downstream modules (observation operator, estimator) are out of scope for this
milestone.
"""

from .config import (
    EventType,
    KernelFamily,
    ExponentialKernelParams,
    BaselineParams,
    LatentRegimeSpec,
    MetaOrderWindow,
    MetaOrderSpec,
    NumericalSettings,
    SimulatorConfig,
    validate,
    validate_first_milestone,
    config_to_json,
    config_from_json,
)
from .kernels import (
    Kernel,
    ExponentialKernel,
    build_kernel,
    branching_matrix,
    spectral_radius,
)
from .latent import (
    RegimeProcess,
    baseline_for_regime,
    MetaOrderSchedule,
)
from .hawkes_core import (
    Event,
    RunTrace,
    SimulatorState,
    HawkesSimulator,
)
from .observation import (
    ALLOWED_CROSSING_POLICIES,
    ObservationConfig,
    ObservedTrace,
    ObservationOperator,
    default_observation_config,
)
from .artifact import (
    RunMetadata,
    LoadedRun,
    save_run,
    load_run,
    default_runs_root,
    run_directory_name,
    CONFIG_FILENAME,
    EVENTS_FILENAME,
    METADATA_FILENAME,
    OBSERVATION_FILENAME,
    ARTEFACT_FORMAT_VERSION,
)
from .evaluation import (
    summarise,
    save_summary,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA_VERSION,
)
from .comparison import (
    EnvelopeError,
    compare_to_envelope,
    load_envelope,
    save_comparison,
    COMPARISON_FILENAME,
    COMPARISON_SCHEMA_VERSION,
)
from .runner import (
    build_default_first_milestone_config,
    default_first_milestone_config_path,
    run_once,
)
from .ensemble import (
    run_ensemble,
    ENSEMBLE_SCHEMA_VERSION,
    ENSEMBLE_SUMMARY_FILENAME,
)
from .ensemble_comparison import (
    compare_ensemble_to_envelope,
    save_ensemble_comparison,
    ENSEMBLE_COMPARISON_FILENAME,
    ENSEMBLE_COMPARISON_SCHEMA_VERSION,
    ALLOWED_MODES as ENSEMBLE_COMPARISON_MODES,
)
from .envelope_bridge import (
    load_empirical_envelope_draft,
    merge_envelopes,
    ALLOWED_PREFER as ENVELOPE_BRIDGE_MODES,
)
from .recovery import (
    WindowFeatures,
    compute_window_features,
    RegimeBaselineConfig,
    recover_regimes_baseline,
    MetaOrderBaselineConfig,
    score_meta_order_activity,
    true_regime_at,
)
from .recovery_eval import (
    RecoveryEvaluation,
    evaluate_meta_order_recovery,
)
from .experiment import (
    ExperimentConfig,
    run_experiment,
    EXPERIMENT_MANIFEST_FILENAME,
    EXPERIMENT_RECOVERY_DIRNAME,
    EXPERIMENT_EVALUATION_FILENAME,
    EXPERIMENT_SCHEMA_VERSION,
)

__all__ = [
    "EventType",
    "KernelFamily",
    "ExponentialKernelParams",
    "BaselineParams",
    "LatentRegimeSpec",
    "MetaOrderWindow",
    "MetaOrderSpec",
    "NumericalSettings",
    "SimulatorConfig",
    "validate",
    "validate_first_milestone",
    "config_to_json",
    "config_from_json",
    "Kernel",
    "ExponentialKernel",
    "build_kernel",
    "branching_matrix",
    "spectral_radius",
    "RegimeProcess",
    "baseline_for_regime",
    "MetaOrderSchedule",
    "Event",
    "RunTrace",
    "SimulatorState",
    "HawkesSimulator",
    "ALLOWED_CROSSING_POLICIES",
    "ObservationConfig",
    "ObservedTrace",
    "ObservationOperator",
    "default_observation_config",
    "RunMetadata",
    "LoadedRun",
    "save_run",
    "load_run",
    "default_runs_root",
    "run_directory_name",
    "CONFIG_FILENAME",
    "EVENTS_FILENAME",
    "METADATA_FILENAME",
    "OBSERVATION_FILENAME",
    "ARTEFACT_FORMAT_VERSION",
    "summarise",
    "save_summary",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "EnvelopeError",
    "compare_to_envelope",
    "load_envelope",
    "save_comparison",
    "COMPARISON_FILENAME",
    "COMPARISON_SCHEMA_VERSION",
    "build_default_first_milestone_config",
    "default_first_milestone_config_path",
    "run_once",
    "run_ensemble",
    "ENSEMBLE_SCHEMA_VERSION",
    "ENSEMBLE_SUMMARY_FILENAME",
    "compare_ensemble_to_envelope",
    "save_ensemble_comparison",
    "ENSEMBLE_COMPARISON_FILENAME",
    "ENSEMBLE_COMPARISON_SCHEMA_VERSION",
    "ENSEMBLE_COMPARISON_MODES",
    "load_empirical_envelope_draft",
    "merge_envelopes",
    "ENVELOPE_BRIDGE_MODES",
    "WindowFeatures",
    "compute_window_features",
    "RegimeBaselineConfig",
    "recover_regimes_baseline",
    "MetaOrderBaselineConfig",
    "score_meta_order_activity",
    "true_regime_at",
    "RecoveryEvaluation",
    "evaluate_meta_order_recovery",
    "ExperimentConfig",
    "run_experiment",
    "EXPERIMENT_MANIFEST_FILENAME",
    "EXPERIMENT_RECOVERY_DIRNAME",
    "EXPERIMENT_EVALUATION_FILENAME",
    "EXPERIMENT_SCHEMA_VERSION",
]

__version__ = "0.1.0"
