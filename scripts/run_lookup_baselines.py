"""Training-free lookup baselines on InteractBind (benchmark audit, N2).

For each split, the train portion builds the lookup tables and the test portion is scored:
  * localization — same-sequence site-count lookup (random fallback for unseen FASTA) vs a
    random ranking, BRHR@K and Precision@n⁺ on annotated positives;
  * binary — target-only (FASTA, UniProt) and drug-only (SMILES) smoothed positive-rate priors,
    AUROC / AUPRC on all test rows.
Validation rows are unused. Writes results/lookup_{split}_{seed}.json per split.

    python scripts/run_lookup_baselines.py --config configs/baselines/lookup.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from safedti.baselines.lookup import (
    key_prior,
    localization_metrics,
    lookup_scores,
    site_count_table,
)
from safedti.data.interactbind import parse_site_string
from safedti.metrics.localization import random_brhr_expected
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

RAW = Path("data/raw/interactbind")
ID_FOLDS = Path("data/processed/interactbind/id_folds.parquet")
COLS = ["uniprot_id", "fasta", "smiles", "label"]
RDLogger.DisableLog("rdApp.*")


def load_split(
    name: str, site_col: str, extra_cols: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) frames for an OOD config or an ID fold.

    Rows with a missing value in the base columns are dropped; `extra_cols` are only carried.
    """
    base = [*COLS, site_col]
    cols = [*base, *extra_cols]
    if name.startswith("p_ood_"):
        tr = pd.read_parquet(RAW / f"{name}_train.parquet", columns=cols)
        te = pd.read_parquet(RAW / f"{name}_test.parquet", columns=cols)
    elif name.startswith("id_"):
        # id_fold{k}: exact-FASTA grouping; id_{tag}_fold{k}: data/.../id_{tag}_folds.parquet
        stem, _, k = name.rpartition("_fold")
        k = int(k)
        path = ID_FOLDS if stem == "id" else ID_FOLDS.with_name(f"{stem}_folds.parquet")
        df = pd.read_parquet(RAW / "affinity_train.parquet", columns=cols)
        folds = pd.read_parquet(path, columns=["row_id", f"split_{k}"])
        df = df.loc[folds["row_id"].to_numpy()]
        s = folds[f"split_{k}"].to_numpy()
        tr, te = df[s == "train"], df[s == "test"]
    else:
        raise ValueError(f"unknown split {name}")
    # row_id = row index in the source parquet (affinity_train for ID folds), kept for release.
    tr = tr.dropna(subset=base).rename_axis("row_id").reset_index()
    te = te.dropna(subset=base).rename_axis("row_id").reset_index()
    for d in (tr, te):
        d["sites"] = [parse_site_string(x) for x in d[site_col]]
        d["L"] = d["fasta"].str.len()
        assert (d["sites"].map(len) == d["L"]).all(), "site vector length != len(fasta)"
        d["n_pos"] = d["sites"].map(lambda y: int(y.sum()))
    return tr, te


def heavy_atoms(smiles: str) -> float:
    """RDKit heavy-atom count; NaN if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    return float(mol.GetNumHeavyAtoms()) if mol is not None else float("nan")


def random_expected(te: pd.DataFrame, ks: list[int]) -> dict[str, float]:
    return {
        f"brhr@{k}": float(
            np.mean(
                [random_brhr_expected(L, n, k) for L, n in zip(te["L"], te["n_pos"], strict=True)]
            )
        )
        for k in ks
    }


def run_split(name: str, cfg: dict, rng: np.random.Generator) -> dict:
    ks, chunk = cfg["brhr_k"], cfg["chunk"]
    tr, te = load_split(name, cfg["site_column"])
    out: dict = {"n_train": len(tr), "n_test": len(te)}

    # ---- localization -------------------------------------------------------------------------
    tab_rows = tr[tr["label"].isin(cfg["table_labels"]) & (tr["n_pos"] > 0)]
    table = site_count_table(tab_rows["fasta"], tab_rows["sites"])
    ev = te[te["label"].isin(cfg["eval_labels"]) & (te["n_pos"] > 0)].reset_index(drop=True)
    sc, matched = lookup_scores(
        ev["fasta"].tolist(), ev["L"].tolist(), table, rng, cfg["tie_noise"]
    )
    rnd = [rng.uniform(0.0, 1.0, size=L) for L in ev["L"]]
    y = ev["sites"].tolist()
    pair_keys = set(zip(tr["fasta"], tr["smiles"], strict=True))
    m_idx = np.flatnonzero(matched)
    loc = {
        "n_eval_pairs": len(ev),
        "coverage_fasta_in_train_table": float(matched.mean()),
        "frac_pair_in_train": float(
            np.mean([p in pair_keys for p in zip(ev["fasta"], ev["smiles"], strict=True)])
        ),
        "median_L": float(ev["L"].median()),
        "lookup_all": localization_metrics(sc, y, ks, chunk),
        "random_all": localization_metrics(rnd, y, ks, chunk),
        "random_all_analytic": random_expected(ev, ks),
    }
    if len(m_idx):
        loc["lookup_matched"] = localization_metrics(
            [sc[i] for i in m_idx], [y[i] for i in m_idx], ks, chunk
        )
        loc["random_matched_analytic"] = random_expected(ev.iloc[m_idx], ks)
    out["localization"] = loc

    # ---- binary ---------------------------------------------------------------------------------
    tr_b, te_b = tr[tr["label"].isin([0, 1])], te[te["label"].isin([0, 1])]
    binary = {"n_test": len(te_b), "test_pos_frac": float(te_b["label"].mean())}
    for key in cfg["binary_keys"]:
        p, seen = key_prior(tr_b[key], tr_b["label"], te_b[key], cfg["binary_smoothing"])
        binary[f"{key}_prior"] = {
            "auroc": float(roc_auc_score(te_b["label"], p)),
            "auprc": float(average_precision_score(te_b["label"], p)),
            "coverage": float(seen.mean()),
        }
    # Ligand size alone: no train data used. Docked affinity scales with size.
    ha = te_b["smiles"].map(heavy_atoms)
    binary["ligand_heavy_atoms"] = {
        "auroc": float(roc_auc_score(te_b["label"], ha.fillna(ha.median()))),
        "auprc": float(average_precision_score(te_b["label"], ha.fillna(ha.median()))),
        "n_rdkit_parse_failures": int(ha.isna().sum()),
    }
    out["binary"] = binary
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/lookup.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="*", help="override cfg splits")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    for name in args.splits or cfg["splits"]:
        t0 = time.time()
        rng = np.random.default_rng([args.seed, cfg["splits"].index(name)])
        metrics = run_split(name, cfg, rng)
        p = write_result(
            f"lookup_{name}_{args.seed}",
            {**cfg, "split": name},
            args.seed,
            metrics,
            time.time() - t0,
        )
        lo, b = metrics["localization"], metrics["binary"]
        print(
            f"{name:9s} cov={lo['coverage_fasta_in_train_table']:.3f} "
            f"lookup@1/5={lo['lookup_all']['brhr@1']:.3f}/{lo['lookup_all']['brhr@5']:.3f} "
            f"random@1/5={lo['random_all']['brhr@1']:.3f}/{lo['random_all']['brhr@5']:.3f} "
            f"| AUROC fasta={b['fasta_prior']['auroc']:.3f} "
            f"uniprot={b['uniprot_id_prior']['auroc']:.3f} "
            f"smiles={b['smiles_prior']['auroc']:.3f} "
            f"heavy_atoms={b['ligand_heavy_atoms']['auroc']:.3f} -> {p}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
