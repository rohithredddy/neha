"""ESM-2 residue-embedding cache: keys, paths and long-sequence windowing.

One file per unique sequence: ``{root}/{model}/{key[:2]}/{key}.npy``, shape (L, D), float16, where
``key = sha1(sequence)``. Row j is residue j (0-based) of the sequence — the same axis as the
InteractBind binding-site vectors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ESM_MAX_RESIDUES = 1022  # 1024 positions minus BOS/EOS
DEFAULT_STRIDE = 511


def seq_key(seq: str) -> str:
    """Stable cache key for a protein sequence."""
    return hashlib.sha1(seq.encode("ascii")).hexdigest()


def cache_path(root: str | Path, model: str, seq: str) -> Path:
    """Where the embedding of `seq` lives."""
    k = seq_key(seq)
    return Path(root) / model / k[:2] / f"{k}.npy"


def window_spans(
    L: int, max_len: int = ESM_MAX_RESIDUES, stride: int = DEFAULT_STRIDE
) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows covering 0..L, each at most `max_len` long.

    The last window is right-aligned to end exactly at L, so every residue is covered and no
    window is shorter than necessary.
    """
    assert L > 0 and 0 < stride <= max_len
    if L <= max_len:
        return [(0, L)]
    spans = []
    start = 0
    while start + max_len < L:
        spans.append((start, start + max_len))
        start += stride
    spans.append((L - max_len, L))
    return spans


def merge_windows(L: int, spans: list[tuple[int, int]], embs: list[np.ndarray]) -> np.ndarray:
    """Average per-window embeddings where windows overlap. Returns (L, D) float32."""
    D = embs[0].shape[1]
    acc = np.zeros((L, D), dtype=np.float32)
    cnt = np.zeros((L, 1), dtype=np.float32)
    for (s, e), x in zip(spans, embs, strict=True):
        assert x.shape == (e - s, D), (x.shape, (e - s, D))
        acc[s:e] += x
        cnt[s:e] += 1
    assert (cnt > 0).all(), "windows must cover every residue"
    return acc / cnt


def load_embedding(root: str | Path, model: str, seq: str) -> np.ndarray:
    """Memory-map a cached embedding and check its length against the sequence."""
    x = np.load(cache_path(root, model, seq), mmap_mode="r")
    assert x.shape[0] == len(seq), (x.shape, len(seq))
    return x
