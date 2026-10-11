"""InteractBind (public HF release) parsing. Residue-level labels only (DECISIONS 2026-10-11)."""

from __future__ import annotations

import numpy as np

HF_REPO = "Zhaohan-Meng/InteractBind"
CONFIGS = ("affinity", "p_ood_25", "p_ood_28", "p_ood_31", "p_ood_33")

# Raw column name -> short type key used everywhere else in the code.
SITE_COLUMNS = {
    "Hydrogen bonding_binding_site": "hbond",
    "Salt Bridges_binding_site": "salt",
    "Van der Waals_binding_site": "vdw",
    "Hydrophobic_binding_site": "hydrophobic",
    "π–π Stacking_binding_site": "pipi",
    "Cation–π_binding_site": "cationpi",
    "Overall_binding_site": "overall",
}
TYPES = ("hbond", "salt", "vdw", "hydrophobic", "pipi", "cationpi")


def parse_site_string(s: str) -> np.ndarray:
    """Parse a binding-site string like ``"[0,1,0]"`` into a uint8 0/1 vector.

    Index j of the result corresponds to residue j (0-based) of the row's FASTA sequence.

    Args:
        s: The raw cell value.

    Returns:
        (L,) uint8 array of 0/1.

    Raises:
        ValueError: If the string holds anything other than single-digit 0/1 entries.
    """
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"not a list literal: {s[:40]!r}")
    body = s[1:-1].replace(" ", "")
    if body == "":
        return np.zeros(0, dtype=np.uint8)
    raw = np.frombuffer(body.encode("ascii"), dtype=np.uint8)
    # Fast path: "d,d,d" -> digits at even positions, commas at odd positions.
    if len(raw) % 2 == 1 and (raw[1::2] == ord(",")).all():
        vals = raw[0::2] - ord("0")
    else:
        vals = np.array([int(float(x)) for x in body.split(",")], dtype=np.int64)
    if not np.isin(vals, (0, 1)).all():
        raise ValueError(f"non-binary values: {np.unique(vals)[:10]}")
    return vals.astype(np.uint8)
