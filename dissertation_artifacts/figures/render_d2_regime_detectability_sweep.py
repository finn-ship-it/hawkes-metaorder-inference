"""Plot-only renderer for the latent/projected recovery sweep.

The only scientific input is the unchanged 30-cell JSON summary written by the
historical sweep.  This module imports no simulator, fitter, HMM, recovery, or
evaluation code.  It validates the source hash, grid identity, stored summaries,
and the equality ``censoring_cost_cell = f1_observed - f1_latent`` before drawing.

Two layouts are available for documented Stage 4 candidate review:

``scatter``
    Latent-stream versus projected-stream F1, faceted by channel set.  Duty cycle
    is marker shape and intensity multiplier is grayscale fill.

``configuration`` (default and authoritative output)
    Channel-set facets over the five tested intensity multipliers.  Each cell is
    a short connector between latent-stream and projected-stream F1; marker shape
    identifies duty cycle and marker fill identifies the stream.  The x-axis is
    categorical so the seven-dimensional experiment is not presented as a
    continuous fitted relationship.

Each cell is one simulation at the stored seeds.  Neither layout represents a
repeated-seed estimate or uncertainty interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "data" / "d2_regime_detectability_sweep.json"
OUTPUT = HERE / "d2_regime_detectability_sweep.pdf"
EXPECTED_INPUT_SHA256 = "db0f55ada60be6fc11a07c18bfd7d1e637b0420fc20fa00319ede3d188d95546"
LOCKED_MATPLOTLIB = "3.10.9"

EXPECTED_INTENSITIES = (2.0, 3.0, 5.0, 8.0, 10.0)
EXPECTED_DUTIES = (0.20, 0.33, 0.50)
EXPECTED_CHANNELS = ("DEFAULT", "BROADER")
PANEL_TITLES = (
    "(a) Pressure on market orders\nand in-spread limit orders",
    "(b) Also including top-of-book\nlimit orders and cancellations",
)
MARKERS = {0.20: "o", 0.33: "s", 0.50: "^"}
BLUE = "#0072B2"
DARK_GREY = "#303030"
MID_GREY = "#777777"
LIGHT_GREY = "#D9D9D9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> tuple[dict, list[dict]]:
    if matplotlib.__version__ != LOCKED_MATPLOTLIB:
        raise RuntimeError(
            f"matplotlib {LOCKED_MATPLOTLIB} is required; found {matplotlib.__version__}"
        )
    actual_hash = sha256(INPUT)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Unexpected sweep-data hash: {actual_hash}")

    payload = json.loads(INPUT.read_text())
    config = payload["config"]
    if tuple(map(float, config["intensity_multipliers"])) != EXPECTED_INTENSITIES:
        raise ValueError("Unexpected intensity grid")
    if tuple(map(float, config["duty_cycles"])) != EXPECTED_DUTIES:
        raise ValueError("Unexpected duty-cycle grid")
    if tuple(config["channel_sets"]) != EXPECTED_CHANNELS:
        raise ValueError("Unexpected channel-set grid")
    if float(config["hmm_f1_threshold"]) != 0.7:
        raise ValueError("Unexpected recovery criterion")
    if int(config["sim_seed"]) != 1 or int(config["regime_placement_seed"]) != 42:
        raise ValueError("Unexpected stored sweep seeds")

    cells = payload["sweep_cells"]
    if len(cells) != 30:
        raise ValueError(f"Expected 30 sweep cells, found {len(cells)}")
    expected_grid = set(product(EXPECTED_INTENSITIES, EXPECTED_DUTIES, EXPECTED_CHANNELS))
    actual_grid = {
        (float(cell["intensity_multiplier"]), float(cell["duty_cycle"]), cell["channel_set"])
        for cell in cells
    }
    if actual_grid != expected_grid:
        raise ValueError("Sweep cells do not form the expected complete grid")
    if [int(cell["cell_id"]) for cell in cells] != list(range(1, 31)):
        raise ValueError("Unexpected cell identifiers or ordering")

    for cell in cells:
        latent = float(cell["f1_latent"])
        projected = float(cell["f1_observed"])
        if not (0.0 <= latent <= 1.0 and 0.0 <= projected <= 1.0):
            raise ValueError(f"F1 outside [0,1] in cell {cell['cell_id']}")
        if not bool(cell["hmm_latent_converged"]) or not bool(cell["hmm_observed_converged"]):
            raise ValueError(f"Stored recovery did not converge in cell {cell['cell_id']}")
        if not np.isclose(
            float(cell["censoring_cost_cell"]), projected - latent, rtol=0.0, atol=5e-16
        ):
            raise ValueError(f"Stored projected-minus-latent difference fails in cell {cell['cell_id']}")

    latent_values = np.asarray([float(cell["f1_latent"]) for cell in cells])
    if not np.isclose(latent_values.min(), float(payload["min_f1_latent"]), rtol=0.0, atol=1e-15):
        raise ValueError("Stored minimum latent F1 does not reproduce")
    if not np.isclose(latent_values.max(), float(payload["max_f1_latent"]), rtol=0.0, atol=1e-15):
        raise ValueError("Stored maximum latent F1 does not reproduce")
    if not np.isclose(
        np.ptp(latent_values),
        float(payload["f1_latent_variation_max_minus_min"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Stored latent F1 range does not reproduce")

    threshold = payload["threshold"]
    threshold_cell = next(cell for cell in cells if int(cell["cell_id"]) == 14)
    for key in ("intensity_multiplier", "duty_cycle", "channel_set"):
        if threshold[key] != threshold_cell[key]:
            raise ValueError(f"Threshold identity fails for {key}")
    for threshold_key, cell_key in (
        ("f1_latent", "f1_latent"),
        ("f1_observed", "f1_observed"),
        ("censoring_cost", "censoring_cost_cell"),
    ):
        if not np.isclose(
            float(threshold[threshold_key]),
            float(threshold_cell[cell_key]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(f"Threshold summary fails for {threshold_key}")
    return payload, cells


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _cell(cells: list[dict], channel: str, duty: float, intensity: float) -> dict:
    matches = [
        cell
        for cell in cells
        if cell["channel_set"] == channel
        and np.isclose(float(cell["duty_cycle"]), duty)
        and np.isclose(float(cell["intensity_multiplier"]), intensity)
    ]
    if len(matches) != 1:
        raise ValueError("Configuration does not identify exactly one sweep cell")
    return matches[0]


def render_scatter(cells: list[dict], output: Path) -> None:
    """Candidate A: direct latent-versus-projected comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.82), sharex=True, sharey=True)
    grey_levels = dict(zip(EXPECTED_INTENSITIES, np.linspace(0.90, 0.30, 5)))

    for index, (ax, channel) in enumerate(zip(axes, EXPECTED_CHANNELS)):
        ax.plot([0.2, 1.0], [0.2, 1.0], color=MID_GREY, lw=0.9, ls="--", zorder=1)
        ax.axvline(0.7, color="#999999", lw=0.75, ls=":", zorder=0)
        ax.axhline(0.7, color="#999999", lw=0.75, ls=":", zorder=0)
        for cell in (item for item in cells if item["channel_set"] == channel):
            intensity = float(cell["intensity_multiplier"])
            duty = float(cell["duty_cycle"])
            ax.scatter(
                float(cell["f1_latent"]),
                float(cell["f1_observed"]),
                s=43,
                marker=MARKERS[duty],
                facecolor=str(grey_levels[intensity]),
                edgecolor=DARK_GREY,
                linewidth=0.75,
                zorder=3,
            )
        ax.set_title(PANEL_TITLES[index])
        ax.set_xlim(0.2, 1.01)
        ax.set_ylim(0.2, 1.01)
        ax.set_xticks(np.arange(0.2, 1.01, 0.2))
        ax.set_yticks(np.arange(0.2, 1.01, 0.2))
        ax.set_xlabel(r"$F_1$ on the latent simulated stream")
        ax.grid(True, color="#DDDDDD", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$F_1$ on the projected simulated stream")

    intensity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=str(grey_levels[intensity]),
            markeredgecolor=DARK_GREY,
            markersize=5.5,
            label=f"{intensity:g}",
        )
        for intensity in EXPECTED_INTENSITIES
    ]
    duty_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[duty],
            linestyle="none",
            markerfacecolor="0.65",
            markeredgecolor=DARK_GREY,
            markersize=5.5,
            label=f"{duty:.2f}",
        )
        for duty in EXPECTED_DUTIES
    ]
    fig.legend(
        handles=intensity_handles,
        title="Intensity multiplier (marker fill)",
        loc="lower left",
        bbox_to_anchor=(0.07, 0.005),
        ncol=5,
        frameon=False,
        title_fontsize=7,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.legend(
        handles=duty_handles,
        title="Fraction of time under imposed pressure (marker shape)",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.005),
        ncol=3,
        frameon=False,
        title_fontsize=7,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.suptitle(
        "Stronger pressure is easier to detect",
        fontsize=9.5,
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.79, wspace=0.17)
    save(fig, output, "latent-versus-projected recovery candidate")


def render_configuration(cells: list[dict], output: Path) -> None:
    """Candidate B: paired stream results over the tested configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.72), sharey=True)
    x_positions = np.arange(len(EXPECTED_INTENSITIES), dtype=float)
    duty_offsets = {0.20: -0.19, 0.33: 0.0, 0.50: 0.19}
    pair_offset = 0.026

    for index, (ax, channel) in enumerate(zip(axes, EXPECTED_CHANNELS)):
        for duty in EXPECTED_DUTIES:
            for x, intensity in zip(x_positions, EXPECTED_INTENSITIES):
                cell = _cell(cells, channel, duty, intensity)
                centre = x + duty_offsets[duty]
                latent = float(cell["f1_latent"])
                projected = float(cell["f1_observed"])
                ax.plot(
                    [centre - pair_offset, centre + pair_offset],
                    [latent, projected],
                    color=MID_GREY,
                    lw=0.75,
                    zorder=2,
                )
                ax.scatter(
                    centre - pair_offset,
                    latent,
                    s=31,
                    marker=MARKERS[duty],
                    facecolor="white",
                    edgecolor=DARK_GREY,
                    linewidth=0.8,
                    zorder=3,
                )
                ax.scatter(
                    centre + pair_offset,
                    projected,
                    s=31,
                    marker=MARKERS[duty],
                    facecolor=BLUE,
                    edgecolor=DARK_GREY,
                    linewidth=0.65,
                    zorder=4,
                )
        ax.axhline(0.7, color="#555555", lw=0.85, ls="--", zorder=1)
        ax.text(
            4.40,
            0.708,
            r"$F_1=0.7$ criterion",
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="#444444",
        )
        ax.set_title(PANEL_TITLES[index])
        ax.set_xlim(-0.48, 4.48)
        ax.set_ylim(0.2, 1.015)
        ax.set_xticks(x_positions, [f"{value:g}" for value in EXPECTED_INTENSITIES])
        ax.set_xlabel(r"Rate multiplier, $1+\delta_p$")
        ax.grid(True, axis="y", color="#D9D9D9", lw=0.55, zorder=0)
    axes[0].set_ylabel(r"Recovery score, $F_1$")

    stream_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=DARK_GREY,
            markersize=5.5,
            label="Latent simulated stream",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor=DARK_GREY,
            markersize=5.5,
            label="Projected simulated stream",
        ),
    ]
    duty_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[duty],
            linestyle="none",
            markerfacecolor=LIGHT_GREY,
            markeredgecolor=DARK_GREY,
            markersize=5.5,
            label=f"{duty:.2f}",
        )
        for duty in EXPECTED_DUTIES
    ]
    fig.legend(
        handles=stream_handles,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.015),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    fig.legend(
        handles=duty_handles,
        title="Fraction of time under imposed pressure",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.005),
        ncol=3,
        frameon=False,
        title_fontsize=7,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.suptitle(
        "Stronger pressure is easier to detect",
        fontsize=9.5,
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.79, wspace=0.12)
    save(fig, output, "Stronger pressure is easier to detect")


def save(fig: plt.Figure, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        metadata={
            "Title": title,
            "Author": "Dissertation artifact renderer",
            "Creator": "render_d2_regime_detectability_sweep.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        choices=("scatter", "configuration"),
        default="configuration",
        help="Candidate layout; the default is the authoritative design.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="PDF destination (defaults to the authoritative figure path).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, cells = load_and_validate()
    configure_style()
    if args.design == "scatter":
        render_scatter(cells, args.output)
    else:
        render_configuration(cells, args.output)

    latent = np.asarray([float(cell["f1_latent"]) for cell in cells])
    projected = np.asarray([float(cell["f1_observed"]) for cell in cells])
    differences = projected - latent
    print(f"Input SHA-256 {sha256(INPUT)}")
    print(f"Validated cells {len(cells)}")
    print(f"Latent/projected correlation {np.corrcoef(latent, projected)[0, 1]:.12f}")
    print(f"Projected-minus-latent range {differences.min():+.12f} to {differences.max():+.12f}")
    print(f"Recovery criterion {float(payload['config']['hmm_f1_threshold']):.1f}")
    print(f"Wrote {args.output}")
    print(f"SHA-256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
