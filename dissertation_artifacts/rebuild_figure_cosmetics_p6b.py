"""P6b Phase 3 cosmetic figure regeneration (no headline artifact recompute).

Redraws two dissertation figures from their EXISTING committed summary JSONs,
applying cosmetic-only fixes already made in the source builders:
  - eurusd... recovery_hmm_posterior.pdf : title no longer prints the raw
    `meta_order_smoke` config token (now "meta-order benchmark").
  - layer3_d1_markout_distribution.pdf   : crowded ~1e-5 x-tick labels are
    de-crowded with a shared scientific offset and few ticks.

Crucially this reads the committed JSON summaries and calls the builders'
figure functions directly; it re-runs no HMM and no 30-seed simulation and
writes no JSON, so every headline artifact (F1, gates, etc.) is untouched.
Data, bins, and numbers are identical; only tick formatting and one title
string change.
"""

from __future__ import annotations

import json
from pathlib import Path

import build_recovery_hmm_meta_order_smoke as rec
import build_layer3_d1_skew_markout as l3

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    rec._ensure_simulator_on_path()

    rec_json = json.loads((DATA / "recovery_hmm_meta_order_smoke.json").read_text())
    rec._make_posterior_figure(rec_json["per_seed"])
    print(f"  regenerated {rec.OUT_PDF.name}")

    l3_json = json.loads((DATA / "layer3_d1_skew_markout_comparison.json").read_text())
    l3._make_figure(l3_json["per_seed"])
    print(f"  regenerated {l3.OUT_PDF.name}")


if __name__ == "__main__":
    main()
