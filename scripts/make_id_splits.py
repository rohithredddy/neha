"""Build the InteractBind-ID 5-fold split, grouped by exact protein sequence (DECISIONS 2026-10-11).

`uniprot_id` is NOT used for grouping: one FASTA carries up to 222 UniProt IDs in this table, and
linking rows by UniProt-or-FASTA collapses 92.6% of rows into one component.

ID = `affinity` rows with label in {0, 1} and a non-missing FASTA (84,809 rows).
Writes data/processed/interactbind/id_folds.parquet: row_id (row in affinity_train.parquet),
uniprot_id, label, fold, split_0..split_4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from safedti.data.splits import grouped_kfold

RAW = Path("data/raw/interactbind/affinity_train.parquet")
OUT = Path("data/processed/interactbind/id_folds.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(RAW, columns=["uniprot_id", "fasta", "smiles", "label"])
    n_in = len(df)
    df = df[df["label"].isin([0, 1]) & df["fasta"].notna()].copy()
    df["row_id"] = df.index
    sp = grouped_kfold(df["fasta"], args.folds, args.seed)
    out = pd.concat([df[["row_id", "uniprot_id", "label"]], sp], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    stats = {"rows_in": n_in, "rows_id": len(df), "seed": args.seed}
    for k in range(args.folds):
        s = out[f"split_{k}"]
        train_fasta = set(df["fasta"][s == "train"])
        stats[f"fold{k}"] = {
            **{n: int((s == n).sum()) for n in ("train", "val", "test")},
            "test_pos_frac": round(float(out["label"][s == "test"].mean()), 4),
            "test_rows_with_fasta_in_train": round(
                float(df["fasta"][s == "test"].isin(train_fasta).mean()), 4),
            "test_smiles_in_train": round(float(df["smiles"][s == "test"].isin(
                set(df["smiles"][s == "train"])).mean()), 4),
        }
    print(json.dumps(stats, indent=2))
    OUT.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
