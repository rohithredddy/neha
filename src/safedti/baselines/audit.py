"""Shortcut-audit helpers for the binary task (scripts/run_ligand_audit.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def purity(df: pd.DataFrame) -> dict:
    """Label purity per ligand (`smiles`): how often one ligand has a single label."""
    g = df.groupby("smiles")["label"].agg(["mean", "size"])
    pure = (g["mean"] == 0) | (g["mean"] == 1)
    multi = g["size"] >= 2
    return {
        "n_ligands": len(g),
        "frac_ligands_multi_row": float(multi.mean()),
        "frac_multi_row_ligands_pure": float(pure[multi].mean()) if multi.any() else None,
        "frac_rows_on_multi_row_pure_ligands": float((g["size"] * (pure & multi)).sum() / len(df)),
    }


def size_matched(te: pd.DataFrame, col: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Positional indices of a subset with equal positives/negatives in every exact `col` bin."""
    keep = []
    for _, idx in te.groupby(col.to_numpy(), sort=True).indices.items():
        lab = te["label"].to_numpy()[idx]
        pos, neg = idx[lab == 1], idx[lab == 0]
        n = min(len(pos), len(neg))
        if n:
            keep += [*rng.choice(pos, n, replace=False), *rng.choice(neg, n, replace=False)]
    return np.sort(np.asarray(keep, dtype=np.int64))
