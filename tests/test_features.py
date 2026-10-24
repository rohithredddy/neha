"""Hand-checked tests for the pair-model features."""

import numpy as np
import pytest

from safedti.baselines.features import AA, DIPEPTIDES, aac, dpc, morgan_bits


def test_aac_hand_example():
    v = aac("AACX")  # X is ignored: A=2/3, C=1/3
    assert v[AA.index("A")] == pytest.approx(2 / 3)
    assert v[AA.index("C")] == pytest.approx(1 / 3)
    assert v.sum() == pytest.approx(1.0)


def test_dpc_hand_example():
    v = dpc("ACAC")  # pairs AC, CA, AC -> AC=2/3, CA=1/3
    assert v[DIPEPTIDES.index("AC")] == pytest.approx(2 / 3)
    assert v[DIPEPTIDES.index("CA")] == pytest.approx(1 / 3)
    assert dpc("AXA").sum() == 0  # pairs touching a non-standard residue are skipped


def test_morgan_bits_shape_and_parse_failure():
    fp = morgan_bits("CCO", 2, 1024)
    assert fp.shape == (1024,) and fp.dtype == np.uint8 and fp.sum() > 0
    assert morgan_bits("not_a_smiles(", 2, 1024) is None
