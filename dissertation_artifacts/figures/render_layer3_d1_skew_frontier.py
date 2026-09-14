#!/usr/bin/env python3
"""Render the revised Chapter 6 seven-setting figure from stored results.

This confined plotting path reads only
``data/layer3_d1_skew_frontier.json`` and writes only
``figures/layer3_d1_skew_frontier.pdf``. It does not import the simulator,
HMM, fill harness, markout code, calibration code, or any fitting pathway.

For each of the 30 matched seeds and seven tested posterior-sensitivity
settings, the upper panel displays the stored fill proportion divided by the
same seed's value at s=0. The stored fill proportion is the number of fills
divided by the two side-specific quote opportunities evaluated at every
projected simulated market event. The diamond is the median across seeds.

The lower panel displays, in native ticks, the difference between the stored
seed-level mean absolute five-second post-fill price movement at a tested
setting and the same seed's value at s=0. The diamond is the arithmetic mean
of those 30 within-seed differences. Negative values therefore mean that the
absolute movement is smaller than at s=0. Absolute magnitude discards the
direction of movement and is not evidence of adverse selection by itself.

The seven x positions are discrete tested settings. No interpolation,
continuity, confidence interval, or model uncertainty is implied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCKED_MATPLOTLIB = "3.10.9"
LOCKED_NUMPY = "2.4.6"
EXPECTED_SOURCE_SHA256 = "6cdd45120fa96c725f8d97a1270fd96de0858fc93eaea96958d3a2bc6c39bffa"
EXPECTED_SENSITIVITIES = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
NATIVE_TICK_SIZE = 1e-5
FILL_RETENTION_CRITERION = 0.50

REPO = Path(__file__).resolve().parents[2]
SOURCE_JSON = REPO / "dissertation_artifacts" / "data" / "layer3_d1_skew_frontier.json"
OUTPUT_PDF = Path(__file__).resolve().parent / "layer3_d1_skew_frontier.pdf"

SEED_EDGE = "#676767"
GRID = "#D8D8D8"
SUMMARY = "#111111"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_locked_inputs() -> None:
    if matplotlib.__version__ != LOCKED_MATPLOTLIB:
        raise RuntimeError(
            f"matplotlib {LOCKED_MATPLOTLIB} is required; found {matplotlib.__version__}"
        )
    if np.__version__ != LOCKED_NUMPY:
        raise RuntimeError(f"numpy {LOCKED_NUMPY} is required; found {np.__version__}")
    actual = _sha256(SOURCE_JSON)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Stored input hash mismatch for {SOURCE_JSON}: "
            f"expected {EXPECTED_SOURCE_SHA256}, found {actual}"
        )


def _load_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with SOURCE_JSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("artifact") != "layer3_d1_skew_frontier":
        raise ValueError("Unexpected stored frontier artifact")
    if float(payload.get("headline_horizon_seconds", -1.0)) != 5.0:
        raise ValueError("The stored headline horizon must be five seconds")
    if int(payload.get("n_seeds_evaluated", -1)) != 30:
        raise ValueError("The stored result must contain 30 evaluated seeds")
    sensitivities = tuple(float(value) for value in payload["posterior_sensitivity_grid"])
    if sensitivities != EXPECTED_SENSITIVITIES:
        raise ValueError("Unexpected posterior-sensitivity settings")

    config = payload["fixed_skew_agent_config"]
    tick_size = float(config["tick_size_in_pips"]) * 1e-4
    if not np.isclose(tick_size, NATIVE_TICK_SIZE, atol=1e-16, rtol=0.0):
        raise ValueError("The stored quote tick does not equal 1e-5 price units")

    per_seed = sorted(payload["per_seed"], key=lambda record: int(record["seed"]))
    seeds = np.asarray([int(record["seed"]) for record in per_seed], dtype=int)
    if len(per_seed) != 30 or set(seeds.tolist()) != set(range(1, 31)):
        raise ValueError("The stored result must contain unique seeds 1 through 30")

    retention = np.empty((30, len(sensitivities)), dtype=float)
    movement_change_ticks = np.empty_like(retention)
    mean_movement = np.empty_like(retention)
    fills = np.empty_like(retention)

    for seed_index, record in enumerate(per_seed):
        n_events = int(record["n_events"])
        if n_events <= 0:
            raise ValueError("Each seed must contain at least one projected simulated event")
        baseline = record["cells"]["0.00"]
        baseline_fill_proportion = float(baseline["fill_rate"])
        baseline_movement = float(baseline["mean_toxicity_5s"])
        if baseline_fill_proportion <= 0.0 or not np.isfinite(baseline_movement):
            raise ValueError("Invalid s=0 seed-level baseline")

        for setting_index, sensitivity in enumerate(sensitivities):
            key = f"{sensitivity:.2f}"
            cell = record["cells"][key]
            n_fills = int(cell["n_fills"])
            fill_proportion = float(cell["fill_rate"])
            movement = float(cell["mean_toxicity_5s"])
            if not np.isclose(
                fill_proportion,
                n_fills / (2.0 * n_events),
                atol=1e-15,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Seed {record['seed']} at s={sensitivity:.2f} has an "
                    "inconsistent quote-opportunity fill proportion"
                )
            if not np.isfinite(movement):
                raise ValueError("Non-finite seed-level absolute movement")
            retention[seed_index, setting_index] = (
                fill_proportion / baseline_fill_proportion
            )
            movement_change_ticks[seed_index, setting_index] = (
                movement - baseline_movement
            ) / tick_size
            mean_movement[seed_index, setting_index] = movement
            fills[seed_index, setting_index] = n_fills

    frontier_by_setting = {
        float(row["posterior_sensitivity"]): row for row in payload["frontier"]
    }
    if set(frontier_by_setting) != set(sensitivities):
        raise ValueError("The aggregate frontier does not match the tested settings")

    for setting_index, sensitivity in enumerate(sensitivities):
        row = frontier_by_setting[sensitivity]
        ratios = retention[:, setting_index]
        changes_price = movement_change_ticks[:, setting_index] * tick_size
        setting_movements = mean_movement[:, setting_index]
        baseline_movements = mean_movement[:, 0]

        checks = (
            np.isclose(
                np.median(ratios),
                float(row["median_fill_rate_retention_vs_a"]),
                atol=1e-15,
                rtol=0.0,
            ),
            np.isclose(
                np.mean(changes_price),
                float(row["mean_b_minus_a_5s"]),
                atol=1e-16,
                rtol=0.0,
            ),
            np.isclose(
                np.mean(setting_movements),
                float(row["mean_toxicity_5s"]),
                atol=1e-16,
                rtol=0.0,
            ),
            np.isclose(
                np.mean(baseline_movements),
                float(row["mean_toxicity_5s_episode_a"]),
                atol=1e-16,
                rtol=0.0,
            ),
            np.isclose(
                np.mean(movement_change_ticks[:, setting_index] < 0.0),
                float(row["sign_stable_fraction_b_lower_than_a"]),
                atol=1e-15,
                rtol=0.0,
            ),
            np.isclose(
                np.mean(fills[:, setting_index]),
                float(row["mean_fills_per_seed"]),
                atol=1e-12,
                rtol=0.0,
            ),
            int(np.sum(fills[:, setting_index])) == int(row["total_fills"]),
            int(row["n_seeds_paired"]) == 30,
        )
        mean_baseline = float(np.mean(baseline_movements))
        recomputed_percentage = 100.0 * (
            mean_baseline - float(np.mean(setting_movements))
        ) / mean_baseline
        if not np.isclose(
            recomputed_percentage,
            float(row["pct_toxicity_reduction_vs_a"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("Stored aggregate percentage reduction is inconsistent")
        if not all(checks):
            raise ValueError(f"Stored aggregate checks failed at s={sensitivity:.2f}")

    if not np.allclose(retention[:, 0], 1.0, atol=1e-15, rtol=0.0):
        raise ValueError("The s=0 retention ratios must equal one")
    if not np.allclose(
        movement_change_ticks[:, 0], 0.0, atol=1e-15, rtol=0.0
    ):
        raise ValueError("The s=0 movement changes must equal zero")

    return seeds, retention, movement_change_ticks


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.3,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def render() -> Path:
    """Validate the stored result and write the revised Figure 6.2 PDF only."""
    _assert_locked_inputs()
    seeds, retention, movement_change_ticks = _load_results()
    _set_style()

    positions = np.arange(len(EXPECTED_SENSITIVITIES), dtype=float)
    jitter = (((seeds * 17) % 30) - 14.5) / 14.5 * 0.13
    medians = np.median(retention, axis=0)
    mean_changes = np.mean(movement_change_ticks, axis=0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.20, 4.45),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    for setting_index, position in enumerate(positions):
        axes[0].scatter(
            position + jitter,
            retention[:, setting_index],
            s=13,
            marker="o",
            facecolor="white",
            edgecolor=SEED_EDGE,
            linewidth=0.55,
            zorder=2,
            label="Individual seed" if setting_index == 0 else None,
        )
    axes[0].scatter(
        positions,
        medians,
        s=38,
        marker="D",
        facecolor=SUMMARY,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
        label="Median across seeds",
    )
    axes[0].axhline(
        FILL_RETENTION_CRITERION,
        color="#333333",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )
    axes[0].text(
        0.02,
        FILL_RETENTION_CRITERION + 0.017,
        "0.50 criterion",
        transform=axes[0].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.0,
    )
    axes[0].set_ylim(
        max(0.0, min(0.15, float(retention.min()) - 0.08)),
        max(1.08, float(retention.max()) + 0.08),
    )
    axes[0].set_ylabel("Fill retention relative to $s=0$")
    axes[0].set_title("(a) Fill retention", loc="left", pad=4)
    axes[0].legend(loc="lower left", frameon=False, ncol=2, columnspacing=1.0)
    axes[0].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    for setting_index, position in enumerate(positions):
        axes[1].scatter(
            position + jitter,
            movement_change_ticks[:, setting_index],
            s=13,
            marker="o",
            facecolor="white",
            edgecolor=SEED_EDGE,
            linewidth=0.55,
            zorder=2,
            label="Individual seed" if setting_index == 0 else None,
        )
    axes[1].scatter(
        positions,
        mean_changes,
        s=38,
        marker="D",
        facecolor=SUMMARY,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
        label="Equal-seed mean",
    )
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=0.9, zorder=1)
    axes[1].text(
        0.02,
        0.018,
        "No change",
        transform=axes[1].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.0,
    )
    movement_low = min(0.0, float(movement_change_ticks.min()))
    movement_high = max(0.0, float(movement_change_ticks.max()))
    movement_padding = max(0.02, (movement_high - movement_low) * 0.12)
    axes[1].set_ylim(movement_low - movement_padding, movement_high + movement_padding)
    axes[1].set_ylabel(
        "Change in mean absolute\nmovement (ticks)"
    )
    axes[1].set_title("(b) Change in absolute post-fill movement", loc="left", pad=4)
    axes[1].legend(loc="lower left", frameon=False, ncol=2, columnspacing=1.0)
    axes[1].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    axes[1].set_xlim(-0.52, 6.52)
    axes[1].set_xticks(
        positions,
        ["0", "0.10", "0.20", "0.35", "0.50", "0.75", "1.00"],
    )
    axes[1].set_xlabel("Tested posterior-sensitivity setting ($s$)")

    fig.suptitle(
        "Execution-value frontier\n"
        "Seven posterior-sensitivity settings; 30 matched simulations",
        y=0.995,
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.55, h_pad=0.8)
    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "Execution-value frontier",
            "Author": "Finn Smith",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    print("setting  retention median [IQR]  seeds>=0.50  movement change mean (ticks)  seeds<0")
    for index, sensitivity in enumerate(EXPECTED_SENSITIVITIES):
        q1, q3 = np.quantile(retention[:, index], [0.25, 0.75])
        print(
            f"s={sensitivity:.2f}  {medians[index]:.6f} "
            f"[{q1:.6f}, {q3:.6f}]  "
            f"{int(np.sum(retention[:, index] >= FILL_RETENTION_CRITERION))}/30  "
            f"{mean_changes[index]:+.8f}  "
            f"{int(np.sum(movement_change_ticks[:, index] < 0.0))}/30"
        )
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Output SHA-256: {_sha256(OUTPUT_PDF)}")
    return OUTPUT_PDF


if __name__ == "__main__":
    render()
