import numpy as np

from safedti.data.esm_cache import merge_windows, seq_key, window_spans


def test_short_sequence_single_window():
    assert window_spans(300) == [(0, 300)]
    assert window_spans(1022) == [(0, 1022)]


def test_long_sequence_covers_everything_right_aligned():
    for L in (1023, 1600, 2044, 3109):
        spans = window_spans(L)
        assert spans[0][0] == 0 and spans[-1][1] == L
        assert all(e - s == 1022 for s, e in spans)
        covered = np.zeros(L, bool)
        for s, e in spans:
            covered[s:e] = True
        assert covered.all()


def test_merge_windows_averages_overlap_and_keeps_alignment():
    L, D = 1500, 4
    truth = np.arange(L, dtype=np.float32)[:, None].repeat(D, 1)  # residue j -> value j
    spans = window_spans(L)
    embs = [truth[s:e] for s, e in spans]
    np.testing.assert_allclose(merge_windows(L, spans, embs), truth)


def test_seq_key_stable():
    assert seq_key("ACDE") == seq_key("ACDE") != seq_key("ACDF")
