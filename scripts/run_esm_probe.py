"""Ligand-agnostic learned residue model: linear probe on frozen ESM-2 residue embeddings (N2).

The probe never sees the ligand, so every ligand of a protein gets the same residue ranking.
Per split: a seeded sample of annotated train FASTAs; residue label = binding in >= 1 train pair
of that FASTA; all positive residues + `neg_per_pos` sampled negatives per sequence; standardised
logistic regression. Test: annotated positives, residue score = probe logit, BRHR via the
project metric. Embeddings are precomputed (CONTRIBUTING.md rule 8):

    python scripts/run_esm_probe.py --seed 0 --select      # writes cfg seqs_file
    python scripts/precompute_esm.py --model esm2_t6_8M_UR50D --seqs-file <seqs_file>
    python scripts/run_esm_probe.py --seed 0               # results/esm_probe_{split}_{seed}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from safedti.baselines.lookup import localization_metrics
from safedti.data.esm_cache import load_embedding
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_lookup_baselines import load_split, random_expected  # noqa: E402


def split_tables(name: str, cfg: dict, seed: int):
    """(train site table {fasta: binary sites}, sampled train FASTAs, eval frame)."""
    tr, te = load_split(name, cfg["site_column"])
    tab = tr[tr["label"].isin(cfg["table_labels"]) & (tr["n_pos"] > 0)]
    sites = {f: (np.stack(g.to_list()).sum(axis=0) > 0) for f, g in tab.groupby("fasta")["sites"]}
    pool = sorted(sites)
    rng = np.random.default_rng([seed, cfg["probe_splits"].index(name)])
    n = min(cfg["n_train_seqs"], len(pool))
    sample = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
    ev = te[te["label"].isin(cfg["eval_labels"]) & (te["n_pos"] > 0)].reset_index(drop=True)
    return sites, sample, ev


def run_split(name: str, cfg: dict, seed: int) -> dict:
    sites, sample, ev = split_tables(name, cfg, seed)
    rng = np.random.default_rng([seed, 1000 + cfg["probe_splits"].index(name)])
    root, model = cfg["esm_root"], cfg["esm_model"]
    X, y = [], []
    for f in sample:
        E = load_embedding(root, model, f).astype(np.float32)
        lab = sites[f]
        assert E.shape[0] == len(lab) == len(f), "embedding rows must equal residues"
        pos = np.flatnonzero(lab)
        neg = np.flatnonzero(~lab)
        neg = rng.choice(neg, min(len(neg), cfg["neg_per_pos"] * len(pos)), replace=False)
        idx = np.concatenate([pos, neg])
        X.append(E[idx])
        y.append(lab[idx])
    X, y = np.concatenate(X), np.concatenate(y)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(**cfg["logreg"], random_state=seed).fit(scaler.transform(X), y)

    cache: dict[str, np.ndarray] = {}
    scores = []
    for f in ev["fasta"]:
        if f not in cache:
            E = load_embedding(root, model, f).astype(np.float32)
            assert E.shape[0] == len(f)
            cache[f] = clf.decision_function(scaler.transform(E))
        scores.append(cache[f])
    ks = cfg["brhr_k"]
    return {
        "n_train_seqs": len(sample),
        "n_train_residues": int(len(y)),
        "train_pos_frac": float(y.mean()),
        "n_eval_pairs": len(ev),
        "probe": localization_metrics(scores, ev["sites"].tolist(), ks, cfg["chunk"]),
        "random_analytic": random_expected(ev, ks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/esm_probe.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--select", action="store_true", help="only write the sequence list")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    if args.select:
        need: set[str] = set()
        for name in cfg["probe_splits"]:
            _, sample, ev = split_tables(name, cfg, args.seed)
            need |= set(sample) | set(ev["fasta"])
        Path(cfg["seqs_file"]).write_text("\n".join(sorted(need)) + "\n", encoding="ascii")
        print(f"{len(need)} sequences, {sum(map(len, need)):,} residues -> {cfg['seqs_file']}")
        return 0
    for name in cfg["probe_splits"]:
        t0 = time.time()
        m = run_split(name, cfg, args.seed)
        write_result(
            f"esm_probe_{name}_{args.seed}", {**cfg, "split": name}, args.seed, m, time.time() - t0
        )
        p, r = m["probe"], m["random_analytic"]
        print(
            f"{name:14s} probe@1/5={p['brhr@1']:.3f}/{p['brhr@5']:.3f} "
            f"rand@1/5={r['brhr@1']:.3f}/{r['brhr@5']:.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
