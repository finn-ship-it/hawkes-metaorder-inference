"""
envelope_bridge.py — non-invasive bridge between the placeholder envelope
and the empirical-draft envelope produced from archive_empirical/.

Design goals
------------
1. Loading the draft envelope must not modify it.
2. Merging the placeholder and the draft must produce a NEW envelope dict
   without mutating either input. This keeps targets/real_fx_envelope.json
   safe under any merge mode.
3. The merge is field-level. When both envelopes provide a target for the
   same dotted path, the 'prefer' argument decides which one wins. Fields
   that exist only in one envelope are passed through.
4. The merged envelope passes comparison.load_envelope so it can be used
   directly with the existing comparison infrastructure.

This module never reaches into archive_empirical/ at runtime; it operates
only on JSON envelopes that already live under My repo/targets/.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from .comparison import load_envelope


ALLOWED_PREFER = ("draft", "placeholder")


def load_empirical_envelope_draft(path: str) -> Dict[str, Any]:
    """Read and validate a draft empirical envelope.

    The validation contract is identical to comparison.load_envelope's. The
    function additionally checks that every field in the draft carries a
    'source' key documenting its provenance, since unsourced empirical
    targets defeat the purpose of the draft.
    """
    envelope = load_envelope(path)
    fields = envelope["fields"]
    missing_source = [
        name for name, spec in fields.items() if "source" not in spec
    ]
    if missing_source:
        raise ValueError(
            "draft envelope fields lack 'source' provenance: "
            + ", ".join(sorted(missing_source))
        )
    return envelope


def merge_envelopes(
    placeholder: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    prefer: str = "draft",
) -> Dict[str, Any]:
    """Merge two envelopes without mutating either input.

    Parameters
    ----------
    placeholder : dict
        The current placeholder envelope (e.g. real_fx_envelope.json).
    draft : dict
        The empirical draft envelope (e.g. empirical_envelope_draft.json).
    prefer : str
        'draft' (default) takes the draft's spec for any field that appears
        in both; 'placeholder' takes the placeholder's spec.

    Returns
    -------
    dict
        A new envelope dict, deep-copied from its inputs, with merged
        'fields' and a 'merge_provenance' top-level block listing which
        envelope each field came from.
    """
    if prefer not in ALLOWED_PREFER:
        raise ValueError(
            f"prefer must be one of {ALLOWED_PREFER!r}, got {prefer!r}"
        )

    merged: Dict[str, Any] = copy.deepcopy(placeholder)
    merged_fields: Dict[str, Any] = merged.setdefault("fields", {})
    provenance: Dict[str, str] = {}

    # Fields present only in the placeholder pass through.
    for name in merged_fields:
        provenance[name] = "placeholder"

    # Walk the draft's fields and merge.
    for name, draft_spec in draft.get("fields", {}).items():
        if name in merged_fields:
            if prefer == "draft":
                merged_fields[name] = copy.deepcopy(draft_spec)
                provenance[name] = "draft (preferred over placeholder)"
            else:
                provenance[name] = "placeholder (preferred over draft)"
        else:
            merged_fields[name] = copy.deepcopy(draft_spec)
            provenance[name] = "draft (placeholder has no entry)"

    merged["merge_provenance"] = provenance
    merged["merge_mode"] = prefer
    merged.setdefault("notes", []).append(
        f"Envelope produced by simulator.envelope_bridge.merge_envelopes "
        f"with prefer='{prefer}'. Field-level provenance is recorded under "
        f"the 'merge_provenance' top-level block."
    )
    return merged


__all__: List[str] = [
    "ALLOWED_PREFER",
    "load_empirical_envelope_draft",
    "merge_envelopes",
]
