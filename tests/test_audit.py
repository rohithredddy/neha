"""Hand-checked tests for the binary shortcut-audit helpers."""

import numpy as np
import pandas as pd

from safedti.baselines.audit import purity, size_matched


def test_size_matched_balances_every_bin():
    # bin 10: 3 pos, 1 neg -> keep 1+1 ; bin 20: 0 pos, 2 neg -> keep 0 ; bin 30: 2 pos, 2 neg -> 4
    te = pd.DataFrame({"label": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]})
    size = pd.Series([10, 10, 10, 10, 20, 20, 30, 30, 30, 30])
    idx = size_matched(te, size, np.random.default_rng(0))
    assert len(idx) == 6
    kept = pd.DataFrame({"size": size.to_numpy()[idx], "label": te["label"].to_numpy()[idx]})
    counts = kept.groupby(["size", "label"]).size().to_dict()
    assert counts == {(10, 0): 1, (10, 1): 1, (30, 0): 2, (30, 1): 2}
    assert 3 in idx  # the only bin-10 negative must be kept


def test_purity_hand_example():
    df = pd.DataFrame({"smiles": ["a", "a", "b", "b", "c"], "label": [1, 1, 0, 1, 0]})
    r = purity(df)
    assert r["n_ligands"] == 3
    assert r["frac_multi_row_ligands_pure"] == 0.5  # a pure, b mixed
    assert r["frac_rows_on_multi_row_pure_ligands"] == 2 / 5
