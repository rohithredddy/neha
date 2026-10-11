"""Group-aware K-fold splitting (ID subset; DECISIONS 2026-10-11)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_kfold(
    groups: pd.Series, n_folds: int, seed: int, val_frac: float = 0.1
) -> pd.DataFrame:
    """Assign every row to a fold so that no group spans two folds.

    Groups are shuffled with `seed`, then assigned largest-first to the currently smallest fold
    (balances row counts). For fold k: test = rows in fold k; val = a seeded `val_frac` of the
    remaining *groups*; train = the rest.

    Args:
        groups: (n_rows,) group label per row (e.g. UniProt ID).
        n_folds: Number of folds.
        seed: Random seed.
        val_frac: Fraction of non-test groups held out for validation.

    Returns:
        DataFrame indexed like `groups` with column ``fold`` and, for each k,
        a column ``split_k`` in {"train", "val", "test"}.
    """
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    names = sizes.index.to_numpy()
    names = names[rng.permutation(len(names))]
    order = sorted(names, key=lambda g: -sizes[g])  # stable: ties keep shuffled order
    load = np.zeros(n_folds, dtype=np.int64)
    fold_of: dict = {}
    for g in order:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += sizes[g]
    out = pd.DataFrame({"fold": groups.map(fold_of).astype(int)}, index=groups.index)
    for k in range(n_folds):
        rest = np.array(sorted(g for g, f in fold_of.items() if f != k))
        rest = rest[rng.permutation(len(rest))]
        val_groups = set(rest[: int(round(val_frac * len(rest)))])
        split = np.where(out["fold"] == k, "test",
                         np.where(groups.isin(val_groups), "val", "train"))
        out[f"split_{k}"] = split
    return out
