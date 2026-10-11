import numpy as np
import pandas as pd

from safedti.data.splits import grouped_kfold


def _groups():
    rng = np.random.default_rng(1)
    return pd.Series(rng.choice([f"P{i}" for i in range(200)], size=3000))


def test_no_group_crosses_splits_and_all_rows_assigned():
    g = _groups()
    out = grouped_kfold(g, n_folds=5, seed=0)
    assert out["fold"].between(0, 4).all()
    assert (g.groupby(out["fold"]).nunique().sum()) == g.nunique()  # each group in one fold
    for k in range(5):
        s = out[f"split_{k}"]
        sets = {name: set(g[s == name]) for name in ("train", "val", "test")}
        assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"])
        assert not (sets["val"] & sets["test"])
        assert (s == "test").sum() == (out["fold"] == k).sum()


def test_balanced_and_deterministic():
    g = _groups()
    a, b = grouped_kfold(g, 5, seed=0), grouped_kfold(g, 5, seed=0)
    assert a.equals(b)
    counts = a["fold"].value_counts()
    assert counts.max() - counts.min() <= 30
    assert not a.equals(grouped_kfold(g, 5, seed=1))
