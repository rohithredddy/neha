"""Learned pair models on CPU: does a model given BOTH drug and protein just learn the ligand?

Per split, three HistGradientBoosting models with identical hyperparameters, trained on train and
scored on test (validation unused, no tuning):
  ligand_only   Morgan bits
  protein_only  amino-acid + dipeptide composition
  pair          both, concatenated
AUROC/AUPRC on all test rows and on the size-matched test subset saved by run_ligand_audit.py
(same split and seed), where heavy-atom count carries no signal by construction.
Writes results/pair_models_{split}_{seed}.json.

    python scripts/run_pair_models.py --config configs/baselines/pair_models.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from safedti.baselines.features import aac, dpc, morgan_bits
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_lookup_baselines import load_split  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_LIG: dict[str, np.ndarray] = {}
_PROT: dict[str, np.ndarray] = {}


def ligand_matrix(smiles: pd.Series, radius: int, n_bits: int) -> tuple[np.ndarray, int]:
    """(n, n_bits) Morgan bits (zeros where RDKit fails) and the number of failures."""
    fails = 0
    for s in smiles.unique():
        if s not in _LIG:
            fp = morgan_bits(s, radius, n_bits)
            _LIG[s] = fp if fp is not None else np.zeros(n_bits, dtype=np.uint8)
            fails += fp is None
    return np.stack([_LIG[s] for s in smiles]).astype(np.float32), fails


def protein_matrix(fasta: pd.Series) -> np.ndarray:
    """(n, 420) AAC + DPC."""
    for s in fasta.unique():
        if s not in _PROT:
            _PROT[s] = np.concatenate([aac(s), dpc(s)])
    return np.stack([_PROT[s] for s in fasta])


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def run_split(name: str, cfg: dict, seed: int) -> dict:
    tr, te = load_split(name, cfg["site_column"])
    tr = tr[tr["label"].isin([0, 1])].reset_index(drop=True)
    te = te[te["label"].isin([0, 1])].reset_index(drop=True)
    fp = cfg["ligand_fp"]
    Ltr, f1 = ligand_matrix(tr["smiles"], fp["radius"], fp["n_bits"])
    Lte, f2 = ligand_matrix(te["smiles"], fp["radius"], fp["n_bits"])
    Ptr, Pte = protein_matrix(tr["fasta"]), protein_matrix(te["fasta"])
    X = {
        "ligand_only": (Ltr, Lte),
        "protein_only": (Ptr, Pte),
        "pair": (np.hstack([Ltr, Ptr]), np.hstack([Lte, Pte])),
    }
    sm_path = Path(cfg["size_matched_dir"]) / f"{name}_seed{seed}.parquet"
    sm_ids = set(pd.read_parquet(sm_path)["row_id"]) if sm_path.exists() else None
    sm = te["row_id"].isin(sm_ids).to_numpy() if sm_ids is not None else None
    y_tr, y_te = tr["label"].to_numpy(), te["label"].to_numpy()
    out: dict = {
        "n_train": len(tr),
        "n_test": len(te),
        "rdkit_failures": f1 + f2,
        "size_matched_n": int(sm.sum()) if sm is not None else None,
    }
    for v in cfg["variants"]:
        t0 = time.time()
        Xtr, Xte = X[v]
        clf = HistGradientBoostingClassifier(**cfg["gbm"], random_state=seed).fit(Xtr, y_tr)
        p = clf.predict_proba(Xte)[:, 1]
        out[v] = {"all": scores(y_te, p), "fit_s": time.time() - t0}
        if sm is not None and sm.sum():
            out[v]["size_matched"] = scores(y_te[sm], p[sm])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/pair_models.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="*", help="override cfg pair_splits")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    for name in args.splits or cfg["pair_splits"]:
        t0 = time.time()
        m = run_split(name, cfg, args.seed)
        write_result(
            f"pair_models_{name}_{args.seed}",
            {**cfg, "split": name},
            args.seed,
            m,
            time.time() - t0,
        )
        line = " ".join(
            f"{v}={m[v]['all']['auroc']:.3f}/"
            f"{m[v].get('size_matched', {}).get('auroc', float('nan')):.3f}"
            for v in cfg["variants"]
        )
        print(f"{name:14s} AUROC all/size-matched: {line} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
