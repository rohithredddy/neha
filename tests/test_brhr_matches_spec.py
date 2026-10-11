"""Hand-checked BRHR test (docs/02_METHOD_SPEC.md §8). Expected values computed by hand below."""

import pytest
import torch

from safedti.metrics.localization import (
    brhr,
    precision_at_npos,
    random_brhr_expected,
    residue_auprc,
    residue_scores,
)

P = 9.0  # value placed in cells that must be ignored (padding / sink)


def _batch():
    # Pair A: atoms 0-2 valid, atom 3 padded; residues 0-4 valid, residue 5 padded; col 6 = sink.
    S_a = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.0, 0.3, P, P],
            [0.5, 0.1, 0.8, 0.0, 0.2, P, P],
            [0.0, 0.0, 0.1, 0.7, 0.6, P, P],
            [P, P, P, P, P, P, P],
        ]
    )
    # max over valid atoms -> r0=.5 r1=.9 r2=.8 r3=.7 r4=.6 ; ranking r1 r2 r3 r4 r0
    # Pair B: all atoms/residues valid; residue maxima r0..r5 = .2 .1 .3 .4 .5 .6
    S_b = torch.tensor(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, P],
            [0.0, 0.1, 0.3, 0.0, 0.0, 0.0, P],
            [0.0, 0.0, 0.0, 0.4, 0.5, 0.0, P],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, P],
        ]
    )
    # ranking r5 r4 r3 r2 r0 r1
    S = torch.stack([S_a, S_b])
    atom_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    res_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    # A: binding residue r3 (label on padded r5 must be ignored). B: binding residue r0.
    labels = torch.tensor([[0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    return S, atom_mask, res_mask, labels


def test_residue_scores_max_over_valid_atoms_without_sink():
    S, am, rm, _ = _batch()
    s = residue_scores(S, am, rm, has_sink=True)
    assert s.shape == (2, 6)
    torch.testing.assert_close(s[0, :5], torch.tensor([0.5, 0.9, 0.8, 0.7, 0.6]))
    assert torch.isinf(s[0, 5]) and s[0, 5] < 0
    torch.testing.assert_close(s[1], torch.tensor([0.2, 0.1, 0.3, 0.4, 0.5, 0.6]))


def test_brhr_hand_example():
    S, am, rm, y = _batch()
    s = residue_scores(S, am, rm, has_sink=True)
    out = brhr(s, y, rm, ks=(1, 3, 5))
    # A: top1 r1 miss, top3 {r1,r2,r3} hit, top5 hit. B: top1 r5 miss, top3 miss, top5 hit.
    assert out == {1: 0.0, 3: 0.5, 5: 1.0}


def test_precision_and_auprc_hand_example():
    S, am, rm, y = _batch()
    s = residue_scores(S, am, rm, has_sink=True)
    assert precision_at_npos(s, y, rm) == 0.0
    # A: true residue at rank 3 -> AP 1/3 ; B: rank 5 -> AP 1/5
    assert residue_auprc(s, y, rm) == pytest.approx((1 / 3 + 1 / 5) / 2)


def test_k_larger_than_valid_residues_is_a_hit():
    s = torch.tensor([[0.1, 0.2, float("-inf")]])
    rm = torch.tensor([[True, True, False]])
    y = torch.tensor([[True, False, False]])
    assert brhr(s, y, rm, ks=(5,)) == {5: 1.0}


def test_random_ranking_matches_analytic():
    g = torch.Generator().manual_seed(0)
    B, L, npos, k = 20000, 50, 3, 5
    s = torch.rand(B, L, generator=g)
    y = torch.zeros(B, L, dtype=torch.bool)
    y[:, :npos] = True
    rm = torch.ones(B, L, dtype=torch.bool)
    emp = brhr(s, y, rm, ks=(k,))[k]
    assert emp == pytest.approx(random_brhr_expected(L, npos, k), abs=0.01)
