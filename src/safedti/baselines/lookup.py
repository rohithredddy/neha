"""Training-free lookup baselines (no learned parameters).

Residue localization — *same-sequence lookup*: for a test pair, residue j is scored by how many
annotated train pairs with the **identical FASTA** have j as a binding residue. Identical
sequences make residue j the same residue in both rows, so no alignment is involved. Ties
(including every residue of an unmatched protein) are broken by seeded uniform noise in [0, 0.5),
which can never reorder two different integer counts (DECISIONS 2026-10-11: degenerate baselines
use random scores, never constants). BRHR itself is computed by `safedti.metrics.localization`.

Binary — *key prior*: P(label=1) of a test pair = smoothed positive rate of train pairs sharing
its key (FASTA, UniProt ID, or SMILES); unseen keys get the global train rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from safedti.metrics.localization import brhr_per_pair, precision_at_npos


def site_count_table(
    keys: Iterable[str], sites: Iterable[np.ndarray]
) -> dict[str, tuple[np.ndarray, int]]:
    """Sum binding-site vectors per key.

    Args:
        keys: Per-row key (FASTA). Rows sharing a key must have equal-length site vectors.
        sites: Per-row (L,) 0/1 site vectors aligned to that row's FASTA.

    Returns:
        key -> ((L,) int64 count of rows where residue j binds, number of rows).
    """
    table: dict[str, tuple[np.ndarray, int]] = {}
    for k, y in zip(keys, sites, strict=True):
        if k in table:
            c, n = table[k]
            assert len(c) == len(y), f"site length mismatch for one key: {len(c)} vs {len(y)}"
            table[k] = (c + y, n + 1)
        else:
            table[k] = (y.astype(np.int64), 1)
    return table


def lookup_scores(
    keys: Sequence[str],
    lengths: Sequence[int],
    table: dict[str, tuple[np.ndarray, int]],
    rng: np.random.Generator,
    tie_noise: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Residue scores = train site counts for the same key + tie-break noise.

    Args:
        keys: Test keys.
        lengths: Test sequence lengths (must equal the table entry's length when matched).
        table: Output of `site_count_table` on train rows.
        rng: Seeded generator for tie-breaking.
        tie_noise: Upper bound of the uniform noise; must be < 1 so integer counts keep order.

    Returns:
        (list of (L_i,) float64 scores, (n,) bool matched mask).
    """
    assert 0 < tie_noise < 1
    scores, matched = [], np.zeros(len(keys), dtype=bool)
    for i, (k, L) in enumerate(zip(keys, lengths, strict=True)):
        noise = rng.uniform(0.0, tie_noise, size=L)
        if k in table:
            c, _ = table[k]
            assert len(c) == L
            scores.append(c.astype(np.float64) + noise)
            matched[i] = True
        else:
            scores.append(noise)
    return scores, matched


def pad_batch(
    scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length per-pair vectors to (B, L_max) tensors.

    Returns:
        scores (B, L) float64 (-inf at padding), labels (B, L) bool, res_mask (B, L) bool.
    """
    B, L = len(scores), max(len(s) for s in scores)
    S = torch.full((B, L), float("-inf"), dtype=torch.float64)
    Y = torch.zeros((B, L), dtype=torch.bool)
    M = torch.zeros((B, L), dtype=torch.bool)
    for i, (s, y) in enumerate(zip(scores, labels, strict=True)):
        assert len(s) == len(y)
        S[i, : len(s)] = torch.from_numpy(np.asarray(s, dtype=np.float64))
        Y[i, : len(y)] = torch.from_numpy(np.asarray(y).astype(bool))
        M[i, : len(s)] = True
    return S, Y, M


def localization_metrics(
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    ks: Sequence[int],
    chunk: int,
) -> dict[str, float]:
    """BRHR@K and Precision@n⁺ over pairs, computed in chunks to bound memory.

    Args:
        scores: Per-pair (L_i,) residue scores.
        labels: Per-pair (L_i,) 0/1 binding residues (each with >= 1 positive).
        ks: BRHR cutoffs.
        chunk: Pairs per padded batch.

    Returns:
        {"brhr@k": mean hit rate, "precision@npos": mean, "n_pairs": count}.
    """
    hits = {k: [] for k in ks}
    prec_sum = 0.0
    for a in range(0, len(scores), chunk):
        S, Y, M = pad_batch(scores[a : a + chunk], labels[a : a + chunk])
        for k, v in brhr_per_pair(S, Y, M, tuple(ks)).items():
            hits[k].append(v)
        prec_sum += precision_at_npos(S, Y, M) * len(S)
    out = {f"brhr@{k}": torch.cat(hits[k]).mean().item() for k in ks}
    out["precision@npos"] = prec_sum / len(scores)
    out["n_pairs"] = len(scores)
    return out


def key_prior(
    train_keys: pd.Series, train_labels: pd.Series, test_keys: pd.Series, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed per-key positive rate: (pos + a·p₀) / (n + a), p₀ = global train rate.

    Returns:
        ((n_test,) float scores, (n_test,) bool mask of keys seen in train).
    """
    p0 = float(train_labels.mean())
    g = train_labels.groupby(train_keys.values).agg(["sum", "count"])
    rate = (g["sum"] + smoothing * p0) / (g["count"] + smoothing)
    mapped = test_keys.map(rate)
    return mapped.fillna(p0).to_numpy(dtype=np.float64), mapped.notna().to_numpy()
