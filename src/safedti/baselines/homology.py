"""Homology-transfer residue baseline: copy binding sites from aligned train sequences.

For a test sequence q and its MMseqs2 hits h (train sequences), residue j of q is scored

    s_j = sum_h  w_h * f_h[t_h(j)]

where t_h(j) is the train residue aligned to j (match/mismatch columns only; gaps contribute
nothing), f_h is the per-residue binding frequency of train sequence h (possibly ligand-weighted)
and w_h = fident_h. Positions are 0-based and index the FASTA, the same axis as the
InteractBind site vectors. `aligned_positions` is the only place alignment columns become residue
indices; it is hand-tested and `check_alignment` verifies every hit against the real sequences.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

MMSEQS_FORMAT = "query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,bits,qaln,taln"


def aligned_positions(
    qaln: str, taln: str, qstart: int, tstart: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map alignment columns to 0-based residue indices of query and target.

    Args:
        qaln: Aligned query segment with '-' for gaps (MMseqs2 `qaln`).
        taln: Aligned target segment, same length as `qaln`.
        qstart: 1-based query start of the alignment (MMseqs2 convention).
        tstart: 1-based target start.

    Returns:
        (q_idx, t_idx): equal-length int64 arrays; column c with residues on both sides yields
        q_idx[c'] = query residue index, t_idx[c'] = target residue index.
    """
    assert len(qaln) == len(taln), (len(qaln), len(taln))
    q = np.frombuffer(qaln.encode("ascii"), dtype=np.uint8) != ord("-")
    t = np.frombuffer(taln.encode("ascii"), dtype=np.uint8) != ord("-")
    qi = np.cumsum(q) - 1 + (qstart - 1)
    ti = np.cumsum(t) - 1 + (tstart - 1)
    both = q & t
    return qi[both].astype(np.int64), ti[both].astype(np.int64)


def check_alignment(row: pd.Series, qseq: str, tseq: str) -> None:
    """Assert an MMseqs2 hit is consistent with the real sequences (catches off-by-one)."""
    assert row.qaln.replace("-", "") == qseq[row.qstart - 1 : row.qend], "qaln != query slice"
    assert row.taln.replace("-", "") == tseq[row.tstart - 1 : row.tend], "taln != target slice"


def write_fasta(seqs: Mapping[str, str], path: Path) -> None:
    """Write {id: sequence} as FASTA."""
    with path.open("w", encoding="ascii") as f:
        for k, s in seqs.items():
            f.write(f">{k}\n{s}\n")


def mmseqs_search(
    query: Mapping[str, str],
    target: Mapping[str, str],
    workdir: Path,
    mmseqs: str,
    sensitivity: float,
    max_seqs: int,
    max_accept: int,
    threads: int,
    fmt: str = MMSEQS_FORMAT,
) -> pd.DataFrame:
    """Run `mmseqs easy-search` (query vs target) with alignment strings.

    Args:
        query / target: {id: sequence}.
        workdir: Scratch directory (recreated). Must not contain spaces (MMseqs2 on Windows).
        mmseqs: Path to the mmseqs executable/wrapper.
        sensitivity: `-s` value.
        max_seqs: `--max-seqs` prefilter cap.
        max_accept: `--max-accept` cap on reported alignments per query.
        threads: `--threads`.
        fmt: `--format-output` columns (default includes alignment strings).

    Returns:
        Hits with columns `fmt` (fident in [0, 1], 1-based starts), best bit score first.
    """
    assert " " not in str(workdir), f"no spaces allowed in {workdir}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    qf, tf, out = workdir / "q.fasta", workdir / "t.fasta", workdir / "hits.tsv"
    write_fasta(query, qf)
    write_fasta(target, tf)
    cmd = [
        mmseqs, "easy-search", str(qf), str(tf), str(out), str(workdir / "tmp"),
        "-s", str(sensitivity), "--max-seqs", str(max_seqs), "--max-accept", str(max_accept),
        "-e", "10",
        "--threads", str(threads), "--format-output", fmt, "-v", "1",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    # The Windows .bat wrapper can hide a crash's exit code: require the output file.
    if not out.exists():
        raise RuntimeError(f"mmseqs easy-search produced no output file: {out}")
    cols = fmt.split(",")
    if out.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(out, sep="\t", header=None, names=cols, dtype={"query": str, "target": str})
    return df.sort_values(["query", "bits"], ascending=[True, False]).reset_index(drop=True)


def transfer_scores(
    L: int, hits: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]
) -> np.ndarray:
    """Accumulate weighted binding frequencies through alignments.

    Args:
        L: Query length.
        hits: (q_idx, t_idx, weight, target_freq) per hit; target_freq is (L_t,) in [0, 1].

    Returns:
        (L,) float64 scores.
    """
    s = np.zeros(L, dtype=np.float64)
    for q_idx, t_idx, w, f in hits:
        np.add.at(s, q_idx, w * f[t_idx])
    return s


def tanimoto_weights(query_fp, fps: list, floor: float) -> np.ndarray:
    """Tanimoto similarity of one fingerprint to many, plus `floor` (weights never all vanish)."""
    from rdkit import DataStructs

    return np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, fps), dtype=np.float64) + floor


def link_edges(
    hits: pd.DataFrame, min_identity: float, min_cov_either: float
) -> list[tuple[str, str]]:
    """Edges between sequences similar enough to leak: identity >= `min_identity` and the
    alignment covers >= `min_cov_either` of the query OR of the target (so a fragment links to
    its full-length parent). Needs columns query, target, fident, qstart, qend, tstart, tend,
    qlen, tlen.
    """
    qcov = (hits["qend"] - hits["qstart"] + 1) / hits["qlen"]
    tcov = (hits["tend"] - hits["tstart"] + 1) / hits["tlen"]
    keep = (hits["fident"] >= min_identity) & ((qcov >= min_cov_either) | (tcov >= min_cov_either))
    e = hits.loc[keep & (hits["query"] != hits["target"]), ["query", "target"]]
    return list(e.itertuples(index=False, name=None))


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    """Union-find. Returns node -> component id (the smallest node id in its component)."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in nodes}


_STD = set("ACDEFGHIKLMNPQRSTVWYBZX")


def global_aligner(open_gap: float, extend_gap: float):
    """Needleman–Wunsch aligner, EMBOSS-needle defaults: BLOSUM62, free end gaps."""
    from Bio import Align
    from Bio.Align import substitution_matrices

    a = Align.PairwiseAligner()
    a.mode = "global"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score = open_gap, extend_gap
    a.end_gap_score = 0.0
    return a


def global_identity(s1: str, s2: str, aligner) -> dict[str, float]:
    """Identity of one optimal global alignment.

    Returns:
        {"identity_aln": identical / alignment length (EMBOSS needle "Identity", end gaps
        included), "identity_shorter": identical / length of the shorter sequence}.
    Residues outside BLOSUM62's alphabet are mapped to X.
    """
    s1 = "".join(c if c in _STD else "X" for c in s1)
    s2 = "".join(c if c in _STD else "X" for c in s2)
    aln = next(iter(aligner.align(s1, s2)))
    c = aln.counts()
    length = c.identities + c.mismatches + c.gaps
    return {
        "identity_aln": c.identities / length,
        "identity_shorter": c.identities / min(len(s1), len(s2)),
    }
