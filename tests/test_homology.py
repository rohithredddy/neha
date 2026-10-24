"""Hand-checked alignment -> residue-index tests for homology transfer (CONTRIBUTING.md rule 4)."""

import numpy as np
import pandas as pd
import pytest

from safedti.baselines.homology import aligned_positions, check_alignment, transfer_scores


def test_aligned_positions_with_gaps_on_both_sides():
    # query  = "MKACDEFG"  (0-based: M0 K1 A2 C3 D4 E5 F6 G7); alignment starts at A (qstart=3)
    # target = "XXACDYEFG" (X0 X1 A2 C3 D4 Y5 E6 F7 G8);   alignment starts at A (tstart=3)
    #   qaln: A C D - E F
    #   taln: A C D Y E F
    q_idx, t_idx = aligned_positions("ACD-EF", "ACDYEF", qstart=3, tstart=3)
    assert q_idx.tolist() == [2, 3, 4, 5, 6]
    assert t_idx.tolist() == [2, 3, 4, 6, 7]
    query, target = "MKACDEFG", "XXACDYEFG"
    assert [query[i] for i in q_idx] == [target[j] for j in t_idx] == list("ACDEF")


def test_gap_in_target():
    # qaln: A C W D ; taln: A C - D  -> W (query 2) has no partner
    q_idx, t_idx = aligned_positions("ACWD", "AC-D", qstart=1, tstart=5)
    assert q_idx.tolist() == [0, 1, 3]
    assert t_idx.tolist() == [4, 5, 6]


def test_check_alignment_catches_off_by_one():
    row = pd.Series(
        {"qaln": "ACD-EF", "taln": "ACDYEF", "qstart": 3, "qend": 7, "tstart": 3, "tend": 8}
    )
    check_alignment(row, "MKACDEFG", "XXACDYEFG")
    bad = row.copy()
    bad["qstart"] = 2  # one residue too early
    bad["qend"] = 6
    with pytest.raises(AssertionError):
        check_alignment(bad, "MKACDEFG", "XXACDYEFG")


def test_transfer_scores_moves_site_to_the_aligned_query_residue():
    # target binding residue is Y? no: target residue 6 ('E') binds; it aligns to query residue 5.
    q_idx, t_idx = aligned_positions("ACD-EF", "ACDYEF", qstart=3, tstart=3)
    f = np.zeros(9)
    f[6] = 1.0
    s = transfer_scores(8, [(q_idx, t_idx, 0.8, f)])
    assert np.flatnonzero(s).tolist() == [5]
    assert "MKACDEFG"[5] == "E" and s[5] == pytest.approx(0.8)
    # target residue 5 ('Y') is in a query gap -> transfers nowhere
    f2 = np.zeros(9)
    f2[5] = 1.0
    assert transfer_scores(8, [(q_idx, t_idx, 1.0, f2)]).sum() == 0


def test_link_edges_uses_either_direction_coverage():
    from safedti.baselines.homology import link_edges

    hits = pd.DataFrame(
        {
            # frag (len 50) fully inside full (len 500): qcov 1.0, tcov 0.1 -> linked
            # low identity pair -> not linked; self hit -> ignored; short local hit -> not linked
            "query": ["frag", "a", "x", "p"],
            "target": ["full", "b", "x", "q"],
            "fident": [0.95, 0.20, 1.0, 0.9],
            "qstart": [1, 1, 1, 1],
            "qend": [50, 300, 100, 20],
            "tstart": [101, 1, 1, 1],
            "tend": [150, 300, 100, 20],
            "qlen": [50, 300, 100, 400],
            "tlen": [500, 300, 100, 400],
        }
    )
    assert link_edges(hits, 0.3, 0.8) == [("frag", "full")]


def test_connected_components_are_transitive():
    from safedti.baselines.homology import connected_components

    comp = connected_components(["a", "b", "c", "d"], [("c", "b"), ("b", "a")])
    assert comp["a"] == comp["b"] == comp["c"] == "a"
    assert comp["d"] == "d"


def test_global_identity_hand_example():
    from safedti.baselines.homology import global_aligner, global_identity

    al = global_aligner(-10.0, -0.5)
    # identical sequences -> 1.0 under both definitions
    assert global_identity("ACDEFGHIK", "ACDEFGHIK", al) == {
        "identity_aln": 1.0,
        "identity_shorter": 1.0,
    }
    # "ACDEF" inside "WWACDEFWW": 5 identities, alignment length 9 (4 free end gaps)
    r = global_identity("ACDEF", "WWACDEFWW", al)
    assert r["identity_aln"] == pytest.approx(5 / 9)
    assert r["identity_shorter"] == pytest.approx(1.0)
    # non-standard residue U is mapped to X instead of raising
    assert 0 < global_identity("ACUEF", "ACDEF", al)["identity_aln"] < 1
