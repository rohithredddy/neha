"""Residue-level localization metrics. BRHR@K is implemented here and ONLY here (rule 3).

Definition (InteractBind, arXiv 2605.24045 §BRHR; docs/02_METHOD_SPEC.md §3):
    s_j = max_i S_ij over valid ligand atoms i (sink column excluded);
    rank residues by s_j; BRHR@K = 1 if any of the top-K residues is a true binding residue;
    the dataset score is the mean over evaluated pairs.

Tie-breaking: descending stable sort, so equal scores are ordered by residue index. Model maps
are continuous and essentially never tie; degenerate baselines must use random scores, not
constants (see docs/DECISIONS.md).
"""

from __future__ import annotations

import math

import torch

NEG_INF = float("-inf")


def residue_scores(
    S: torch.Tensor,
    atom_mask: torch.Tensor,
    res_mask: torch.Tensor,
    has_sink: bool,
) -> torch.Tensor:
    """Aggregate an atom x residue interaction map to residue scores by max over atoms.

    Args:
        S: Interaction map logits, shape (B, N, L) or (B, N, L+1) if `has_sink`
            (sink is the LAST column).
        atom_mask: (B, N) bool, True for real atoms.
        res_mask: (B, L) bool, True for real residues.
        has_sink: Whether S carries a trailing sink column to drop.

    Returns:
        (B, L) residue scores, -inf at padded residues.
    """
    if has_sink:
        S = S[..., :-1]
    B, N, L = S.shape
    assert atom_mask.shape == (B, N), (atom_mask.shape, (B, N))
    assert res_mask.shape == (B, L), (res_mask.shape, (B, L))
    assert atom_mask.any(dim=1).all(), "every pair needs at least one valid atom"
    masked = S.masked_fill(~atom_mask[:, :, None], NEG_INF)
    s = masked.max(dim=1).values
    return s.masked_fill(~res_mask, NEG_INF)


def _ranking(scores: torch.Tensor) -> torch.Tensor:
    """Residue indices sorted by descending score, ties by ascending index. (B, L)."""
    return torch.sort(scores, dim=1, descending=True, stable=True).indices


def brhr_per_pair(
    scores: torch.Tensor,
    labels: torch.Tensor,
    res_mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[int, torch.Tensor]:
    """Per-pair BRHR@K hits.

    Args:
        scores: (B, L) residue scores (padded residues must be -inf, as from `residue_scores`).
        labels: (B, L) bool, True for annotated binding residues.
        res_mask: (B, L) bool, True for real residues.
        ks: Cutoffs.

    Returns:
        {k: (B,) float tensor of 0/1 hits}.
    """
    assert scores.shape == labels.shape == res_mask.shape
    labels = labels & res_mask
    assert labels.any(dim=1).all(), "BRHR is only defined for pairs with >=1 binding residue"
    order = _ranking(scores.masked_fill(~res_mask, NEG_INF))
    hits_sorted = torch.gather(labels, 1, order)
    n_valid = res_mask.sum(dim=1)
    out = {}
    for k in ks:
        top = (
            torch.arange(scores.shape[1], device=scores.device)[None, :]
            < torch.clamp(n_valid, max=k)[:, None]
        )
        out[k] = (hits_sorted & top).any(dim=1).float()
    return out


def brhr(
    scores: torch.Tensor,
    labels: torch.Tensor,
    res_mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[int, float]:
    """Dataset-level BRHR@K (mean of per-pair hits)."""
    return {k: v.mean().item() for k, v in brhr_per_pair(scores, labels, res_mask, ks).items()}


def precision_at_npos(scores: torch.Tensor, labels: torch.Tensor, res_mask: torch.Tensor) -> float:
    """Mean over pairs of |top-n⁺ ∩ true| / n⁺ (equals recall@n⁺)."""
    labels = labels & res_mask
    order = _ranking(scores.masked_fill(~res_mask, NEG_INF))
    hits_sorted = torch.gather(labels, 1, order).float()
    npos = labels.sum(dim=1)
    top = torch.arange(scores.shape[1], device=scores.device)[None, :] < npos[:, None]
    return ((hits_sorted * top).sum(dim=1) / npos.clamp(min=1)).mean().item()


def residue_auprc(scores: torch.Tensor, labels: torch.Tensor, res_mask: torch.Tensor) -> float:
    """Mean over pairs of average precision over that pair's valid residues."""
    labels = labels & res_mask
    order = _ranking(scores.masked_fill(~res_mask, NEG_INF))
    hits = torch.gather(labels, 1, order).float()
    valid = torch.gather(res_mask, 1, order).float()
    hits = hits * valid
    cum_hits = hits.cumsum(dim=1)
    rank = torch.arange(1, scores.shape[1] + 1, device=scores.device)[None, :].float()
    ap = (cum_hits / rank * hits).sum(dim=1) / labels.sum(dim=1).clamp(min=1)
    return ap.mean().item()


def random_brhr_expected(L: int, n_pos: int, k: int) -> float:
    """Analytic BRHR@K of a uniformly random ranking: 1 - C(L-n⁺, K) / C(L, K)."""
    if k >= L:
        return 1.0
    return 1.0 - math.comb(L - n_pos, k) / math.comb(L, k)


def ligand_specificity_gain(
    scores: list,
    labels: list,
    groups: list,
    k: int,
) -> dict[str, float]:
    """Ligand-Specificity Gain LSG@K: does a ranking follow *this* ligand's sites?

    For pair i of protein p (with m >= 2 annotated ligands), using the pair's own ranking r_i:
        LSG_i = hit@K(r_i, y_i) - mean_{j != i, j in p} hit@K(r_i, y_j)
    where hit@K is BRHR@K of one pair (computed by `brhr_per_pair`). A ligand-blind method gives
    every ligand of p the same ranking, so the LSG_i sum to exactly 0 over each protein. Positive
    values need rankings tailored to the ligand. Pairs whose protein has one ligand are skipped.

    Args:
        scores: Per-pair (L_p,) residue scores (numpy or tensor).
        labels: Per-pair (L_p,) 0/1 binding sites.
        groups: Per-pair protein key; pairs sharing a key share the residue axis.
        k: BRHR cutoff.

    Returns:
        {"lsg": mean over pairs, "own": mean own hit@K, "other": mean other-ligand hit@K,
         "n_pairs": pairs used, "n_proteins": proteins used}.
    """
    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    own_all, other_all, n_prot = [], [], 0
    for idx in by_group.values():
        if len(idx) < 2:
            continue
        n_prot += 1
        S = torch.stack([torch.as_tensor(scores[i], dtype=torch.float64) for i in idx])
        Y = torch.stack([torch.as_tensor(labels[i]).bool() for i in idx])
        m, L = Y.shape
        assert S.shape == (m, L), "pairs of one protein must share the residue axis"
        mask = torch.ones(m, L, dtype=torch.bool)
        # hits[a, b] = hit@K of ranking a against sites of ligand b
        hits = torch.stack(
            [brhr_per_pair(S, Y[b].expand(m, L), mask, (k,))[k] for b in range(m)], dim=1
        )
        own = hits.diagonal()
        other = (hits.sum(dim=1) - own) / (m - 1)
        own_all.append(own)
        other_all.append(other)
    if not own_all:
        return {"lsg": float("nan"), "own": float("nan"), "other": float("nan"),
                "n_pairs": 0, "n_proteins": 0}  # fmt: skip
    own, other = torch.cat(own_all), torch.cat(other_all)
    return {
        "lsg": (own - other).mean().item(),
        "own": own.mean().item(),
        "other": other.mean().item(),
        "n_pairs": int(own.numel()),
        "n_proteins": n_prot,
    }
