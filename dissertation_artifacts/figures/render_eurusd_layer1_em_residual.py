"""Plot-only renderer for the EUR/USD time-rescaling residual figure.

The renderer reads only ``eurusd_layer1_em_residual_plot_data.npz``.  It cannot
load empirical events or parameters and cannot launch fitting, optimisation,
simulation, recovery, calibration, or event construction.

The default design shows pooled unit-exponential quantile--quantile coordinates
on linear and logarithmic axes.  ``--design event-types`` is retained solely to
reproduce the dissertation-size candidate reviewed before the pooled design was
selected; it uses the same plot-ready artifact and performs no research
calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PLOT_DATA = HERE.parent / "data" / "eurusd_layer1_em_residual_plot_data.npz"
DEFAULT_OUTPUT = HERE / "eurusd_layer1_em_residual_qq.pdf"

EXPECTED_PLOT_DATA_SHA256 = "79241f2249ec8590ed4acd9178b34ec4a84b477794a22cfa9712082fb6992c85"
EVENT_TYPES = ("bid_up", "bid_down", "ask_up", "ask_down")
EVENT_LABELS = {
    "bid_up": "Bid up",
    "bid_down": "Bid down",
    "ask_up": "Ask up",
    "ask_down": "Ask down",
}
DATA_COLOR = "#0072B2"
REFERENCE_COLOR = "#8C2D04"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plot_data(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    actual = sha256(path)
    if EXPECTED_PLOT_DATA_SHA256 != "PENDING" and actual != EXPECTED_PLOT_DATA_SHA256:
        raise SystemExit(f"STOP: unexpected plot-data hash: {actual}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"pooled_theoretical", "pooled_empirical", "metadata_json"}
    required.update(f"{name}_{kind}" for name in EVENT_TYPES for kind in ("theoretical", "empirical"))
    missing = required.difference(arrays)
    if missing:
        raise SystemExit(f"STOP: plot-data artifact lacks {sorted(missing)}")

    metadata = json.loads(str(arrays.pop("metadata_json")))
    if metadata["artifact"] != "eurusd_layer1_em_residual_plot_data":
        raise SystemExit("STOP: unexpected plot-data identity")
    if tuple(metadata["event_types"]) != EVENT_TYPES:
        raise SystemExit("STOP: unexpected plot-data event ordering")
    if int(metadata["validation"]["pooled"]["count"]) != 97_948:
        raise SystemExit("STOP: unexpected pooled residual count")
    if float(metadata["validation"]["maximum_absolute_scalar_delta"]) > float(
        metadata["validation"]["absolute_tolerance"]
    ):
        raise SystemExit("STOP: stored validation exceeds its declared tolerance")

    for prefix in ("pooled", *EVENT_TYPES):
        theoretical = np.asarray(arrays[f"{prefix}_theoretical"], dtype=np.float64)
        empirical = np.asarray(arrays[f"{prefix}_empirical"], dtype=np.float64)
        if theoretical.ndim != 1 or empirical.shape != theoretical.shape:
            raise SystemExit(f"STOP: unexpected {prefix} coordinate dimensions")
        if np.any(theoretical <= 0.0) or np.any(empirical <= 0.0):
            raise SystemExit(f"STOP: {prefix} coordinates are not positive")
        if np.any(np.diff(theoretical) <= 0.0) or np.any(np.diff(empirical) < 0.0):
            raise SystemExit(f"STOP: {prefix} quantile coordinates are not sorted")
        expected = -np.log1p(
            -(np.arange(1, theoretical.size + 1, dtype=np.float64) - 0.5)
            / theoretical.size
        )
        if not np.array_equal(theoretical, expected):
            raise SystemExit(f"STOP: {prefix} theoretical quantiles changed")
    return arrays, metadata


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_quantile_points(ax, theoretical: np.ndarray, empirical: np.ndarray, label: str):
    return ax.scatter(
        theoretical,
        empirical,
        s=1.2,
        color=DATA_COLOR,
        edgecolors="none",
        alpha=0.50,
        rasterized=True,
        label=label,
        zorder=2,
    )


def add_reference(ax, lower: float, upper: float, label: str | None = None):
    return ax.plot(
        [lower, upper],
        [lower, upper],
        color=REFERENCE_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=1.2,
        label=label,
        zorder=3,
    )[0]


def pooled_design(arrays: dict[str, np.ndarray], metadata: dict):
    theoretical = arrays["pooled_theoretical"]
    empirical = arrays["pooled_empirical"]
    fig, (linear_ax, log_ax) = plt.subplots(
        1, 2, figsize=(7.2, 3.45), constrained_layout=True
    )
    fig.suptitle("Time-rescaling residuals of the EUR/USD EM fit", fontsize=9.5)

    central_limit = float(-np.log1p(-0.999))
    central = theoretical <= central_limit
    points = add_quantile_points(
        linear_ax,
        theoretical[central],
        empirical[central],
        "Pooled fitted-compensator increments",
    )
    reference = add_reference(
        linear_ax, 0.0, 7.1, "Unit-exponential reference"
    )
    linear_ax.set_xlim(0.0, 7.1)
    linear_ax.set_ylim(0.0, 7.1)
    linear_ax.set_aspect("equal", adjustable="box")
    linear_ax.set_title("(a) Central 99.9% (linear axes)")
    linear_ax.set_xlabel("Unit-exponential quantile (dimensionless)")
    linear_ax.set_ylabel("Compensator-increment quantile (dimensionless)")
    linear_ax.grid(True, color="0.88", linewidth=0.45, zorder=0)
    linear_ax.legend(
        handles=(points, reference),
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handlelength=2.3,
    )

    add_quantile_points(log_ax, theoretical, empirical, "Pooled residuals")
    common_lower = 4.0e-6
    common_upper = 45.0
    add_reference(log_ax, common_lower, common_upper)
    log_ax.set_xscale("log")
    log_ax.set_yscale("log")
    log_ax.set_xlim(common_lower, common_upper)
    log_ax.set_ylim(common_lower, common_upper)
    log_ax.set_aspect("equal", adjustable="box")
    log_ax.set_title("(b) Complete range (logarithmic axes)")
    log_ax.set_xlabel("Unit-exponential quantile (dimensionless)")
    log_ax.set_ylabel("Compensator-increment quantile (dimensionless)")
    log_ax.grid(True, which="major", color="0.88", linewidth=0.45, zorder=0)

    pooled_summary = metadata["validation"]["pooled"]
    observed_fraction = float(pooled_summary["left_tail_observed_frac"])
    expected_fraction = float(pooled_summary["left_tail_expected_frac"])
    short_rank = int(np.searchsorted(empirical, 0.1, side="left"))
    short_rank = min(max(short_rank, 0), empirical.size - 1)
    log_ax.annotate(
        f"{100.0 * observed_fraction:.1f}% below 0.1\n"
        f"unit exponential: {100.0 * expected_fraction:.1f}%",
        xy=(theoretical[short_rank], empirical[short_rank]),
        xytext=(2.2e-4, 0.34),
        textcoords="data",
        fontsize=6.9,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.65", "lw": 0.5},
        zorder=4,
    )
    log_ax.annotate(
        "Rare upper-tail excess",
        xy=(theoretical[-1], empirical[-1]),
        xytext=(0.55, 19.0),
        fontsize=6.9,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.65", "lw": 0.5},
        zorder=4,
    )
    return fig


def event_type_design(arrays: dict[str, np.ndarray], metadata: dict):
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 4.65), constrained_layout=True, sharex=True, sharey=True
    )
    lower, upper = 4.0e-6, 45.0
    left_tail = metadata["pooling_check"]["left_tail_fraction_below_0_1_by_type"]
    expected = metadata["pooling_check"][
        "left_tail_fraction_expected_under_unit_exponential"
    ]
    for panel, (ax, name) in enumerate(zip(axes.flat, EVENT_TYPES, strict=True)):
        theoretical = arrays[f"{name}_theoretical"]
        empirical = arrays[f"{name}_empirical"]
        add_quantile_points(ax, theoretical, empirical, "Fitted-compensator increments")
        add_reference(
            ax,
            lower,
            upper,
            "Unit-exponential reference" if panel == 0 else None,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(EVENT_LABELS[name])
        ax.grid(True, which="major", color="0.88", linewidth=0.45, zorder=0)
        ax.text(
            0.03,
            0.94,
            f"{100.0 * float(left_tail[name]):.1f}% below 0.1\n"
            f"unit exponential: {100.0 * float(expected):.1f}%",
            transform=ax.transAxes,
            fontsize=6.8,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.65", "lw": 0.5},
        )
        if panel == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.92, borderpad=0.3)
    fig.supxlabel("Unit-exponential quantile (dimensionless)", fontsize=8.0)
    fig.supylabel("Compensator-increment quantile (dimensionless)", fontsize=8.0)
    fig.suptitle(
        "Time-rescaling residuals by empirical event type (logarithmic axes)",
        fontsize=9.0,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        choices=("pooled", "event-types"),
        default="pooled",
        help="pooled is the selected dissertation design",
    )
    parser.add_argument("--plot-data", type=Path, default=PLOT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arrays, metadata = load_plot_data(args.plot_data)
    style()
    fig = (
        pooled_design(arrays, metadata)
        if args.design == "pooled"
        else event_type_design(arrays, metadata)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=300,
        metadata={
            "Title": "Time-rescaling residuals of the EUR/USD EM fit",
            "Author": "Dissertation artifact renderer",
            "Creator": "render_eurusd_layer1_em_residual.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"SHA-256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
