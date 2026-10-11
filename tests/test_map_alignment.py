"""Residue indexing tests (docs/02_METHOD_SPEC.md §8, residue-level version)."""

import numpy as np
import pytest

from safedti.data.interactbind import parse_site_string


def test_position_7_is_residue_7_zero_based():
    fasta = "ACDEFGHIKL"  # A0 C1 D2 E3 F4 G5 H6 I7 K8 L9 -> residue 7 (0-based) is 'I'
    s = "[" + ",".join("1" if j == 7 else "0" for j in range(len(fasta))) + "]"
    y = parse_site_string(s)
    assert len(y) == len(fasta)
    assert np.flatnonzero(y).tolist() == [7]
    assert fasta[np.flatnonzero(y)[0]] == "I"


def test_spaces_and_float_like_entries():
    assert parse_site_string("[0, 1, 0]").tolist() == [0, 1, 0]
    assert parse_site_string("[0.0,1.0]").tolist() == [0, 1]


def test_rejects_non_binary():
    with pytest.raises(ValueError):
        parse_site_string("[0,2,0]")
