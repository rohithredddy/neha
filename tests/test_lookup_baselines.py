"""Hand-checked tests for the training-free lookup baselines (CONTRIBUTING.md rule 4)."""

import numpy as np
import pandas as pd
import pytest

from safedti.baselines.lookup import (
    key_prior,
    localization_metrics,
    lookup_scores,
    site_count_table,
)
from safedti.data.interactbind import parse_site_string


def _sites(L, idx):
    return parse_site_string("[" + ",".join("1" if j in idx else "0" for j in range(L)) + "]")


def test_site_counts_stay_on_the_same_residue_index():
    fasta = "ACDEFGHIKL"  # residue 7 (0-based) is 'I', residue 2 is 'D'
    table = site_count_table(
        [fasta, fasta, "MNPQ"], [_sites(10, {2, 7}), _sites(10, {7}), _sites(4, {0})]
    )
    counts, n = table[fasta]
    assert n == 2
    assert counts.tolist() == [0, 0, 1, 0, 0, 0, 0, 2, 0, 0]
    assert fasta[int(np.argmax(counts))] == "I"
    assert table["MNPQ"][0].tolist() == [1, 0, 0, 0]


def test_lookup_ranks_by_count_and_noise_never_reorders_counts():
    fasta = "ACDEFGHIKL"
    table = site_count_table([fasta, fasta], [_sites(10, {2, 7}), _sites(10, {7})])
    rng = np.random.default_rng(0)
    for _ in range(50):
        (s,), matched = lookup_scores([fasta], [10], table, rng, tie_noise=0.5)
        assert matched.tolist() == [True]
        order = np.argsort(-s, kind="stable")
        assert order[0] == 7 and order[1] == 2


def test_unmatched_protein_gets_pure_noise_in_range():
    (s,), matched = lookup_scores(["XYZ"], [3], {}, np.random.default_rng(0), tie_noise=0.5)
    assert matched.tolist() == [False]
    assert ((s >= 0) & (s < 0.5)).all()


def test_localization_metrics_hand_example():
    # pair 0: top residue 7 is the true site -> hit@1.
    # pair 1: true site ranked 2nd -> miss@1, hit@3.
    scores = [np.array([0, 0, 1, 0, 0, 0, 0, 2, 0, 0.0]), np.array([3.0, 2.0, 1.0])]
    labels = [_sites(10, {7}), _sites(3, {1})]
    m = localization_metrics(scores, labels, ks=(1, 3), chunk=1)  # chunk=1 exercises padding
    assert m["brhr@1"] == 0.5 and m["brhr@3"] == 1.0 and m["n_pairs"] == 2
    assert m["precision@npos"] == pytest.approx(0.5)


def test_key_prior_smoothing_and_unseen_fallback():
    tr_k = pd.Series(["a", "a", "b"])
    tr_y = pd.Series([1, 1, 0])
    p, seen = key_prior(tr_k, tr_y, pd.Series(["a", "b", "c"]), smoothing=1.0)
    p0 = 2 / 3
    assert p.tolist() == pytest.approx([(2 + p0) / 3, (0 + p0) / 2, p0])
    assert seen.tolist() == [True, True, False]
