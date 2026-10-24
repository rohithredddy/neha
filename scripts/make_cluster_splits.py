"""Proposed fix: InteractBind-ID 5-fold split grouped by MMseqs2 sequence clusters (N2 audit).

Clusters every unique ID FASTA with `mmseqs easy-cluster` (identity/coverage from the config),
then assigns whole clusters to folds with `grouped_kfold`. Residual leakage is not assumed away:
`run_homology_baselines.py` on the resulting `id_c30_fold*` splits measures it.

Writes data/processed/interactbind/id_c30_folds.parquet (same layout as id_folds.parquet) and a
.json with cluster/fold statistics.

    python scripts/make_cluster_splits.py --config configs/baselines/cluster_split.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from safedti.baselines.homology import write_fasta
from safedti.data.esm_cache import seq_key
from safedti.data.splits import grouped_kfold
from safedti.utils.config import load_config

RAW = Path("data/raw/interactbind/affinity_train.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/cluster_split.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = pd.read_parquet(RAW, columns=["uniprot_id", "fasta", "label"])
    df = df[df["label"].isin([0, 1]) & df["fasta"].notna()].copy()
    df["row_id"] = df.index
    seqs = {seq_key(s): s for s in df["fasta"].unique()}

    work = Path(cfg["workdir"]) / f"cluster_{Path(cfg['out']).stem}"
    assert " " not in str(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    write_fasta(seqs, work / "id.fasta")
    # Step by step: `easy-cluster` segfaults in result2flat on Windows (18-8cc5c), and the .bat
    # wrapper hides the exit code, so every step's output is checked explicitly.
    m, db = cfg["mmseqs"], str(work / "db")
    cmds = [
        [m, "createdb", str(work / "id.fasta"), db, "-v", "1"],
        [m, "cluster", db, str(work / "clu"), str(work / "tmp"),
         "--min-seq-id", str(cfg["min_seq_id"]), "-c", str(cfg["coverage"]),
         "--cov-mode", str(cfg["cov_mode"]), "--cluster-mode", str(cfg["cluster_mode"]),
         "-s", str(cfg["sensitivity"]),
         "--threads", str(cfg["threads"]), "-v", "1"],
        [m, "createtsv", db, db, str(work / "clu"), str(work / "clu.tsv"), "-v", "1"],
    ]  # fmt: skip
    for c in cmds:
        subprocess.run(c, check=True, capture_output=True)
    tsv = work / "clu.tsv"
    if not tsv.exists() or tsv.stat().st_size == 0:
        raise RuntimeError(f"mmseqs clustering produced no output: {tsv}")
    clu = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
    assert set(clu["member"]) == set(seqs), "every sequence must be in exactly one cluster"
    cluster_of = dict(zip(clu["member"], clu["rep"], strict=True))
    df["cluster"] = df["fasta"].map(lambda s: cluster_of[seq_key(s)])

    sp = grouped_kfold(df["cluster"], cfg["folds"], args.seed)
    out = pd.concat([df[["row_id", "uniprot_id", "label", "cluster"]], sp], axis=1)
    dest = Path(cfg["out"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)

    sizes = df.groupby("cluster").size().sort_values(ascending=False)
    stats = {
        "mmseqs_cmds": [" ".join(c[1:]) for c in cmds],
        "n_rows": len(df),
        "n_sequences": len(seqs),
        "n_clusters": int(len(sizes)),
        "largest_cluster_rows": int(sizes.iloc[0]),
        "largest_cluster_frac": float(sizes.iloc[0] / len(df)),
        "folds": {
            f"fold{k}": {n: int((out[f"split_{k}"] == n).sum()) for n in ("train", "val", "test")}
            for k in range(cfg["folds"])
        },
    }
    dest.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
