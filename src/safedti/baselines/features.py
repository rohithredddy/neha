"""Cheap CPU features for learned pair models: Morgan bits (ligand), AAC + DPC (protein)."""

from __future__ import annotations

from itertools import product

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {a: i for i, a in enumerate(AA)}
DIPEPTIDES = ["".join(p) for p in product(AA, repeat=2)]


def aac(seq: str) -> np.ndarray:
    """Amino-acid composition: (20,) fractions over the 20 standard residues (others ignored)."""
    v = np.zeros(20, dtype=np.float32)
    for c in seq:
        i = _AA_IDX.get(c)
        if i is not None:
            v[i] += 1
    return v / max(v.sum(), 1.0)


def dpc(seq: str) -> np.ndarray:
    """Dipeptide composition: (400,) fractions of consecutive standard pairs, row-major AA x AA."""
    v = np.zeros(400, dtype=np.float32)
    for a, b in zip(seq, seq[1:], strict=False):
        i, j = _AA_IDX.get(a), _AA_IDX.get(b)
        if i is not None and j is not None:
            v[20 * i + j] += 1
    return v / max(v.sum(), 1.0)


def morgan_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    """(n_bits,) uint8 Morgan fingerprint, or None if RDKit cannot parse the SMILES."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
