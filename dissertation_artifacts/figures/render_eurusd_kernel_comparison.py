"""Plot-only renderer for dissertation Figure 4.1.

The renderer reads two stored fit outputs and cannot launch event construction,
calibration, or fitting:

* ``data/eurusd_konark_cls_kernel.json`` supplies the saved conditional-
  least-squares kernel on its original non-uniform lag bins.
* ``data/plot_inputs/eurusd_em_parameters.json`` supplies the coefficients
  of the endorsed converged EM fit, extracted without refitting.

Rows are subsequent (child) event types and columns are preceding (parent)
event types. The heatmaps show the full stored branching matrices: CLS over
its 0--500 second fitted support, and EM integrated to infinity.
The two temporal panels display the first ten seconds of the saved kernels.

The two temporal panels are selected mechanically from the full matrices:

1. the off-diagonal interaction maximising the smaller mass across estimates;
2. the remaining interaction with the largest absolute mass discrepancy.

The selection describes the displayed estimates, without assuming agreement.
CLS bin widths and both full-support spectral radii are checked against
the frozen results before plotting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
REPO_ROOT = ARTIFACTS.parent
NONPARAM_PATH = ARTIFACTS / "data" / "eurusd_konark_cls_kernel.json"
EM_PATH = ARTIFACTS / "data" / "plot_inputs" / "eurusd_em_parameters.json"
OUTPUT = HERE / "eurusd_nonparametric_kernel.pdf"

EXPECTED_NONPARAM_SHA256 = "428b791bbe1cf3da298df940a27f7e1f0225fcc217d49fdc3fd73afc1f9157a6"
EXPECTED_EM_SHA256 = "69beb05b6c2487b1e44f6bd9e1b2652552c26280f8238352bb705da799cecc4b"
EVENT_KEYS = ("bid_up", "bid_down", "ask_up", "ask_down")
EVENT_LABELS = ("Bid up", "Bid down", "Ask up", "Ask down")
BLUE = "#0072B2"
ORANGE = "#D55E00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def load_and_validate():
    actual_nonparam = sha256(NONPARAM_PATH)
    actual_em = sha256(EM_PATH)
    if actual_nonparam != EXPECTED_NONPARAM_SHA256:
        raise ValueError(f"Unexpected nonparametric source hash: {actual_nonparam}")
    if actual_em != EXPECTED_EM_SHA256:
        raise ValueError(f"Unexpected EM source hash: {actual_em}")

    nonparam = json.loads(NONPARAM_PATH.read_text())
    em = json.loads(EM_PATH.read_text())

    if nonparam["primary_day"] != "2021-07-20":
        raise ValueError("Unexpected nonparametric fit day")
    if int(nonparam["n_events"]) != 97_952:
        raise ValueError("Unexpected nonparametric event count")
    if float(nonparam["fit_window_seconds"]) != 86_400.0:
        raise ValueError("Unexpected nonparametric fit window")
    cfg = nonparam["config"]
    if (cfg["min_lag"], cfg["max_lag"], cfg["n_bins"], cfg["fine_bin_seconds"], cfg["solver"]) != (0.5, 500.0, 42, 0.5, "osqp"):
        raise ValueError("Unexpected CLS configuration")
    grid = np.asarray(nonparam["result"]["lag_grid"], dtype=float)
    phi = np.asarray(nonparam["result"]["theta"], dtype=float)
    if grid.shape != (43,) or phi.shape != (4, 4, 42):
        raise ValueError("Unexpected CLS array dimensions")
    if not np.all(np.diff(grid) > 0) or not np.all(np.isfinite(phi)):
        raise ValueError("Invalid CLS grid or coefficients")
    full_mass = np.sum(phi * np.diff(grid), axis=2)
    stored_nonparam_mass = np.asarray(nonparam["result"]["branching_matrix"], dtype=float)
    if not np.allclose(full_mass, stored_nonparam_mass, rtol=0.0, atol=5e-13):
        raise ValueError("Nonparametric mass does not reproduce its stored matrix")
    if not np.isclose(
        spectral_radius(full_mass),
        float(nonparam["result"]["spectral_radius"]),
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError("Nonparametric spectral radius does not reproduce")

    if tuple(em["alphabet"]) != EVENT_KEYS:
        raise ValueError("Unexpected EM event ordering")
    if em["alpha_all_orientation"] != (
        "alpha_all[r, child_type_i, parent_type_j]: index r over betas_per_s, "
        "i = child/effect type, j = parent/cause type; Phi[i,j] = sum_r "
        "alpha_all[r,i,j] / betas_per_s[r] = expected number of type-i offspring "
        "per type-j event."
    ):
        raise ValueError("Unexpected EM coefficient orientation")
    if em["fit_date"] != "2021-07-20" or float(em["T_seconds"]) != 86_400.0:
        raise ValueError("Unexpected EM fit identity")
    if int(em["em_metadata"]["n_events"]) != 97_952 or not em["em_metadata"]["converged"]:
        raise ValueError("The stored EM fit is not the endorsed converged fit")

    betas = np.asarray(em["betas_per_s"], dtype=float)
    alpha = np.asarray(em["alpha_all"], dtype=float)
    if betas.shape != (4,) or alpha.shape != (4, 4, 4):
        raise ValueError("Unexpected EM array dimensions")
    if not np.array_equal(betas, np.asarray([0.1, 1.0, 10.0, 100.0])):
        raise ValueError("Unexpected EM decay grid")

    em_full_mass = np.sum(alpha / betas[:, None, None], axis=0)
    if not np.allclose(
        em_full_mass,
        np.asarray(em["branching_matrix"], dtype=float),
        rtol=0.0,
        atol=5e-14,
    ):
        raise ValueError("EM coefficients do not reproduce the stored branching matrix")
    if not np.isclose(
        spectral_radius(em_full_mass), float(em["spectral_radius_rho"]), rtol=0.0, atol=5e-13
    ):
        raise ValueError("EM spectral radius does not reproduce")

    curve_grid = np.geomspace(grid[1] / 2, 10.0, 600)
    em_curves = np.sum(
        alpha[:, :, :, None]
        * np.exp(-betas[:, None] * curve_grid[None, :])[:, None, None, :],
        axis=0,
    )
    return grid, phi, curve_grid, em_curves, stored_nonparam_mass, np.asarray(em["branching_matrix"], dtype=float)


def select_representative_pairs(nonparam_mass: np.ndarray, em_mass: np.ndarray):
    off_diagonal = [(i, j) for i in range(4) for j in range(4) if i != j]
    strongest_joint = max(off_diagonal, key=lambda ij: min(nonparam_mass[ij], em_mass[ij]))
    remaining = [(i, j) for i in range(4) for j in range(4) if (i, j) != strongest_joint]
    largest_mismatch = max(remaining, key=lambda ij: abs(nonparam_mass[ij] - em_mass[ij]))
    selected = (strongest_joint, largest_mismatch)
    return selected


def add_heatmap(ax, values: np.ndarray, title: str, vmax: float):
    image = ax.imshow(values, cmap="cividis", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(4), EVENT_LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), EVENT_LABELS)
    ax.set_xlabel("Preceding quote move")
    ax.set_ylabel("Responding quote move")
    ax.set_title(title, pad=5)
    midpoint = vmax * 0.53
    for i in range(4):
        for j in range(4):
            color = "white" if values[i, j] < midpoint else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    return image


def main() -> None:
    grid, phi, curve_grid, em_curves, nonparam_mass, em_mass = load_and_validate()
    pairs = select_representative_pairs(nonparam_mass, em_mass)
    vmax = max(float(nonparam_mass.max()), float(em_mass.max()))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.4))
    fig.suptitle("Hawkes excitation kernels for EUR/USD", x=0.52, y=0.985, fontsize=9.5)
    # Both rows share the same two column edges. The horizontal colour scale
    # avoids taking space from one matrix and shifting the visual centre.
    lefts = (0.12, 0.60)
    width = 0.32
    matrix_height = width * 7.2 / 6.4
    heatmap_left = fig.add_axes((lefts[0], 0.54, width, matrix_height))
    heatmap_right = fig.add_axes((lefts[1], 0.54, width, matrix_height))
    image = add_heatmap(
        heatmap_left, nonparam_mass,
        "(a) Conditional least squares\n" + rf"Spectral radius $\rho = {spectral_radius(nonparam_mass):.4f}$",
        vmax,
    )
    add_heatmap(
        heatmap_right, em_mass,
        "(b) Converged EM fit\n" + rf"Spectral radius $\rho = {spectral_radius(em_mass):.4f}$",
        vmax,
    )
    scale = fig.add_axes((0.30, 0.415, 0.44, 0.017))
    colorbar = fig.colorbar(image, cax=scale, orientation="horizontal")
    colorbar.set_label("Integrated excitation strength", labelpad=2)
    colorbar.ax.tick_params(labelsize=7, pad=1)

    curve_axes = []
    for panel, (i, j) in enumerate(pairs):
        ax = fig.add_axes((lefts[panel], 0.12, width, 0.21))
        curve_axes.append(ax)
        visible = grid[:-1] < 10.0
        edges = np.r_[grid[:-1][visible], 10.0]
        ax.stairs(
            phi[i, j][visible],
            edges,
            baseline=None,
            color=BLUE,
            linewidth=1.0,
            label="Conditional least squares",
        )
        ax.plot(
            curve_grid,
            em_curves[i, j],
            color=ORANGE,
            linewidth=1.35,
            linestyle="--",
            label="Converged EM fit",
        )
        ax.set_xscale("log")
        ax.set_xlim(curve_grid[0], 10.0)
        ymax = max(float(phi[i, j][visible].max()), float(em_curves[i, j].max()))
        ax.set_ylim(0.0, 1.08 * ymax if ymax > 0.0 else 1.0)
        ax.grid(True, alpha=0.22)
        ax.set_title(f"({chr(99 + panel)}) {EVENT_LABELS[j]} → {EVENT_LABELS[i]}", fontsize=9)
        ax.set_xlabel("Time after preceding move (seconds)", fontsize=8)
        if panel == 0:
            ax.set_ylabel(r"Kernel value $\phi_{ij}(t)$ (s$^{-1}$)")
            handles, labels = ax.get_legend_handles_labels()

    # Centre the heading, colour scale and legend on the complete labelled
    # panels, including their y-axis labels, rather than on the bare axes.
    fig.canvas.draw()
    panel_box = Bbox.union([
        ax.get_tightbbox(fig.canvas.get_renderer())
        for ax in (heatmap_left, heatmap_right, *curve_axes)
    ]).transformed(fig.transFigure.inverted())
    centre = 0.5 * (panel_box.x0 + panel_box.x1)
    fig._suptitle.set_x(centre)
    scale.set_position((centre - 0.22, 0.415, 0.44, 0.017))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(centre, 0.008),
               ncol=2, frameon=False)

    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={
            "Title": "Hawkes excitation kernels for EUR/USD",
            "Author": "Dissertation artifact renderer",
            "Creator": "render_eurusd_kernel_comparison.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT}")
    print(f"SHA-256 {sha256(OUTPUT)}")
    print("Selected (child, parent) pairs:", pairs)
    print("Stored CLS branching matrix:", nonparam_mass.tolist())
    print("Stored EM branching matrix:", em_mass.tolist())


if __name__ == "__main__":
    main()
