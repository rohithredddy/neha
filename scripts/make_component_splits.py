"""Leak-defined ID split: connected components of the "would leak" graph (N2 audit, fix v2).

MMseqs2's own clustering (make_cluster_splits.py) still left 34% of test rows with a >=90%
identity, >=80% coverage train neighbour, because it requires coverage of BOTH sequences
(a fragment never joins its full-length parent) and is not transitive under set cover. Here the
grouping is built from the same definition the audit measures: all-vs-all MMseqs2 search, an
edge when identity >= min_identity and coverage of EITHER sequence >= min_cov_either, then
connected components, then grouped_kfold on components. Residual leakage is still measured by
run_homology_baselines.py on the resulting splits.

Writes data/processed/interactbind/{out}.parquet (+ .json stats).

    python scripts/make_component_splits.py --config configs/baselines/component_split.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from safedti.baselines.homology import connected_components, link_edges, mmseqs_search
from safedti.data.esm_cache import seq_key
from safedti.data.splits import grouped_kfold
from safedti.utils.config import load_config

RAW = Path("data/raw/interactbind/affinity_train.parquet")
FMT = "query,target,fident,qstart,qend,tstart,tend,qlen,tlen,bits"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/component_split.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-hits", action="store_true", help="reuse workdir hits.tsv if complete")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()

    df = pd.read_parquet(RAW, columns=["uniprot_id", "fasta", "label"])
    df = df[df["label"].isin([0, 1]) & df["fasta"].notna()].copy()
    df["row_id"] = df.index
    seqs = {seq_key(s): s for s in df["fasta"].unique()}

    work = Path(cfg["workdir"]) / f"allvsall_{Path(cfg['out']).stem}"
    cached = work / "hits.tsv"
    if args.reuse_hits and cached.exists():
        hits = pd.read_csv(
            cached,
            sep="\t",
            header=None,
            names=FMT.split(","),
            dtype={"query": str, "target": str},
        )
        # the cached run must have searched exactly these sequences
        prev = (work / "q.fasta").read_text(encoding="ascii").splitlines()[0::2]
        assert {h[1:] for h in prev} == set(seqs), "cached hits were built from other input"
    else:
        hits = mmseqs_search(
            seqs,
            seqs,
            work,
            cfg["mmseqs"],
            cfg["sensitivity"],
            cfg["max_seqs"],
            cfg["max_accept"],
            cfg["threads"],
            fmt=FMT,
        )
    assert len(hits), "all-vs-all search returned no hits"
    edges = link_edges(hits, cfg["min_identity"], cfg["min_cov_either"])
    comp = connected_components(list(seqs), edges)
    df["component"] = df["fasta"].map(lambda s: comp[seq_key(s)])

    sizes = df.groupby("component").size().sort_values(ascending=False)
    largest = float(sizes.iloc[0] / len(df))
    stats = {
        "n_rows": len(df),
        "n_sequences": len(seqs),
        "n_hits": len(hits),
        "n_edges": len(edges),
        "n_components": int(len(sizes)),
        "largest_component_rows": int(sizes.iloc[0]),
        "largest_component_frac": largest,
        "top5_component_fracs": [float(x / len(df)) for x in sizes.iloc[:5]],
        "max_fold_frac": 1 / cfg["folds"],
    }
    if largest > 1 / cfg["folds"]:
        stats["error"] = "largest component exceeds one fold: no balanced grouped split exists"
        print(json.dumps(stats, indent=2))
        Path(cfg["out"]).with_suffix(".json").write_text(json.dumps(stats, indent=2))
        return 1

    sp = grouped_kfold(df["component"], cfg["folds"], args.seed)
    out = pd.concat([df[["row_id", "uniprot_id", "label", "component"]], sp], axis=1)
    dest = Path(cfg["out"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    stats["folds"] = {
        f"fold{k}": {n: int((out[f"split_{k}"] == n).sum()) for n in ("train", "val", "test")}
        for k in range(cfg["folds"])
    }
    stats["wall_time_s"] = time.time() - t0
    dest.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
