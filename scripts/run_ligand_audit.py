"""Binary-task shortcut audit: how much of the InteractBind label is explained by the ligand alone.

Per split (train fits, test scores; validation unused):
  * label purity per ligand: fraction of rows whose SMILES has one label across all its proteins;
  * single RDKit descriptors as scores (no training), AUROC on test;
  * ligand-only model: HistGradientBoosting on all RDKit 2D descriptors (no protein input);
  * size-matched test subset: within each exact heavy-atom bin keep equally many positives and
    negatives (seeded). Heavy-atom AUROC there is 0.5 by construction; the ligand-only model's
    AUROC on it measures the shortcut left beyond size. The subset's row ids are saved for reuse.
ID only (OOD tables carry no `affinity`): Spearman of docked affinity vs heavy atoms.
Writes results/ligand_audit_{split}_{seed}.json.

    python scripts/run_ligand_audit.py --config configs/baselines/ligand_audit.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from safedti.baselines.audit import purity, size_matched
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_lookup_baselines import load_split  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DESC_NAMES = [n for n, _ in Descriptors._descList]
_DESC_CACHE: dict[str, np.ndarray] = {}


def descriptors(smiles: pd.Series) -> pd.DataFrame:
    """RDKit 2D descriptors per row (NaN row if the SMILES does not parse), cached by SMILES."""
    for s in smiles.unique():
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                _DESC_CACHE[s] = np.full(len(DESC_NAMES), np.nan)
            else:
                d = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
                _DESC_CACHE[s] = np.array([d[n] for n in DESC_NAMES], dtype=np.float64)
    X = np.stack([_DESC_CACHE[s] for s in smiles])
    X[~np.isfinite(X)] = np.nan
    return pd.DataFrame(X, columns=DESC_NAMES, index=smiles.index)


def binary_scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def run_split(name: str, cfg: dict, seed: int, rng: np.random.Generator) -> dict:
    tr, te = load_split(name, cfg["site_column"])
    tr = tr[tr["label"].isin([0, 1])].reset_index(drop=True)
    te = te[te["label"].isin([0, 1])].reset_index(drop=True)
    Xtr, Xte = descriptors(tr["smiles"]), descriptors(te["smiles"])
    y_tr, y_te = tr["label"].to_numpy(), te["label"].to_numpy()
    out: dict = {
        "n_train": len(tr),
        "n_test": len(te),
        "test_pos_frac": float(y_te.mean()),
        "n_test_rdkit_failures": int(Xte.isna().all(axis=1).sum()),
        "purity_train": purity(tr),
        "purity_test": purity(te),
    }
    out["single_descriptor_auroc"] = {
        d: float(roc_auc_score(y_te, Xte[d].fillna(Xte[d].median())))
        for d in cfg["single_descriptors"]
    }
    gbm = HistGradientBoostingClassifier(**cfg["gbm"], random_state=seed).fit(Xtr, y_tr)
    p = gbm.predict_proba(Xte)[:, 1]
    out["ligand_only_gbm"] = binary_scores(y_te, p)

    col = Xte[cfg["size_match_column"]].fillna(-1)
    idx = size_matched(te, col, rng)
    out["size_matched"] = {
        "n": int(len(idx)),
        "frac_of_test": float(len(idx) / len(te)),
        "heavy_atom_auroc": float(roc_auc_score(y_te[idx], col.to_numpy()[idx])),
        "ligand_only_gbm": binary_scores(y_te[idx], p[idx]),
    }
    d = Path(cfg["size_matched_dir"])
    d.mkdir(parents=True, exist_ok=True)
    te.iloc[idx][["row_id", "label"]].to_parquet(d / f"{name}_seed{seed}.parquet", index=False)
    return out


def affinity_vs_size() -> dict:
    """ID table only: docked affinity vs heavy-atom count (label 0/1 rows)."""
    df = pd.read_parquet("data/raw/interactbind/affinity_train.parquet",
                         columns=["smiles", "label", "affinity", "fasta"])  # fmt: skip
    df = df[df["label"].isin([0, 1]) & df["fasta"].notna()]
    ha = descriptors(df["smiles"])["HeavyAtomCount"]
    ok = ha.notna() & df["affinity"].notna()
    rho = spearmanr(ha[ok], df["affinity"][ok])
    return {
        "n": int(ok.sum()),
        "spearman_heavy_atoms_vs_affinity": float(rho.statistic),
        "label_rule_check_auroc_neg_affinity": float(
            roc_auc_score(df["label"][ok], -df["affinity"][ok])
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/ligand_audit.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="*", help="override cfg splits")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    t0 = time.time()
    m = affinity_vs_size()
    write_result(f"ligand_audit_id_affinity_{args.seed}", cfg, args.seed, m, time.time() - t0)
    print("ID affinity vs size:", m, flush=True)
    for name in args.splits or cfg["splits"]:
        t0 = time.time()
        rng = np.random.default_rng([args.seed, cfg["splits"].index(name)])
        m = run_split(name, cfg, args.seed, rng)
        write_result(
            f"ligand_audit_{name}_{args.seed}",
            {**cfg, "split": name},
            args.seed,
            m,
            time.time() - t0,
        )
        sm = m["size_matched"]
        print(
            f"{name:9s} HA={m['single_descriptor_auroc']['HeavyAtomCount']:.3f} "
            f"gbm={m['ligand_only_gbm']['auroc']:.3f} | size-matched n={sm['n']} "
            f"HA={sm['heavy_atom_auroc']:.3f} gbm={sm['ligand_only_gbm']['auroc']:.3f} "
            f"| pure_rows={m['purity_test']['frac_rows_on_multi_row_pure_ligands']:.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
