"""Homology-transfer localization baselines and sequence-identity split audit (N2).

Per split (train builds tables, test is scored; validation unused):
  * MMseqs2 search of every unique test FASTA against every unique train FASTA;
  * split audit: nearest-train identity of test pairs (all rows), with and without a coverage
    filter;
  * residue baselines on annotated test positives:
      nn1        sites of the single best annotated hit,
      knn        top-m annotated hits weighted by identity,
      knn_lig    as knn, each target's site frequency weighted by ligand Tanimoto similarity;
    each with random fallback when there is no hit, reported overall and per identity stratum;
  * ligand-agnostic oracle ceiling (uses TEST labels, an upper bound, not a baseline): rank
    residues by site frequency over the test pairs of the same FASTA, in-sample and
    leave-one-out.
Writes results/homology_{split}_{seed}.json.

    python scripts/run_homology_baselines.py --config configs/baselines/homology.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from safedti.baselines.homology import (
    aligned_positions,
    check_alignment,
    mmseqs_search,
    tanimoto_weights,
    transfer_scores,
)
from safedti.baselines.lookup import localization_metrics
from safedti.data.esm_cache import seq_key
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_lookup_baselines import load_split, random_expected  # noqa: E402

RDLogger.DisableLog("rdApp.*")


class Fingerprints:
    """Morgan fingerprints cached by SMILES (None if RDKit cannot parse it)."""

    def __init__(self, radius: int, n_bits: int):
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self.cache: dict[str, object] = {}

    def __call__(self, smi: str):
        if smi not in self.cache:
            mol = Chem.MolFromSmiles(smi)
            self.cache[smi] = self.gen.GetFingerprint(mol) if mol is not None else None
        return self.cache[smi]


def build_targets(tab: pd.DataFrame, fps: Fingerprints) -> dict[str, dict]:
    """Per annotated train FASTA: stacked site rows, ligand fingerprints, mean frequency."""
    out = {}
    for fasta, g in tab.groupby("fasta", sort=False):
        Y = np.stack(g["sites"].to_list()).astype(np.float64)
        f = [fps(s) for s in g["smiles"]]
        ok = [x is not None for x in f]
        out[fasta] = {
            "Y": Y,
            "freq": Y.mean(axis=0),
            "fps": [x for x in f if x is not None],
            "Y_fp": Y[np.asarray(ok)],
        }
    return out


def ligand_freq(tgt: dict, qfp, floor: float) -> np.ndarray:
    if qfp is None or not tgt["fps"]:
        return tgt["freq"]
    w = tanimoto_weights(qfp, tgt["fps"], floor)
    return (w @ tgt["Y_fp"]) / w.sum()


def add_noise(scores: np.ndarray, rng: np.random.Generator, tie_noise: float) -> np.ndarray:
    pos = scores[scores > 0]
    assert pos.size == 0 or pos.min() > tie_noise, "tie noise could reorder real scores"
    return scores + rng.uniform(0.0, tie_noise, size=scores.shape)


def identity_audit(best: pd.Series, thresholds: list[float]) -> dict:
    """Distribution of nearest-train identity over test rows (0 when no hit)."""
    v = best.to_numpy()
    return {
        "n": int(v.size),
        "frac_no_hit": float((v == 0).mean()),
        "quantiles": {str(q): float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "frac_ge": {str(t): float((v >= t).mean()) for t in thresholds},
    }


def run_split(name: str, cfg: dict, rng: np.random.Generator) -> dict:
    ks, chunk, noise = cfg["brhr_k"], cfg["chunk"], cfg["tie_noise"]
    tr, te = load_split(name, cfg["site_column"])
    fps = Fingerprints(cfg["ligand_fp"]["radius"], cfg["ligand_fp"]["n_bits"])

    train_seqs = {seq_key(s): s for s in tr["fasta"].unique()}
    test_seqs = {seq_key(s): s for s in te["fasta"].unique()}
    hits = mmseqs_search(
        test_seqs, train_seqs, Path(cfg["workdir"]) / name, cfg["mmseqs"],
        cfg["sensitivity"], cfg["max_seqs"], cfg["max_accept"], cfg["threads"],
    )  # fmt: skip
    for _, h in hits.iterrows():
        check_alignment(h, test_seqs[h.query], train_seqs[h.target])
    hits["qcov"] = (hits["qend"] - hits["qstart"] + 1) / hits["query"].map(
        lambda k: len(test_seqs[k])
    )

    # ---- split audit (all test rows) ---------------------------------------------------------
    best_any = hits.groupby("query")["fident"].max()
    best_cov = hits[hits["qcov"] >= cfg["qcov_min"]].groupby("query")["fident"].max()
    te_keys = te["fasta"].map(seq_key)
    thr = cfg["identity_thresholds"]
    audit = {
        "n_hits": len(hits),
        "all_test_rows": identity_audit(te_keys.map(best_any).fillna(0.0), thr),
        "all_test_rows_qcov": identity_audit(te_keys.map(best_cov).fillna(0.0), thr),
    }

    # ---- residue baselines (annotated positives) --------------------------------------------
    tab = tr[tr["label"].isin(cfg["table_labels"]) & (tr["n_pos"] > 0)]
    targets = build_targets(tab, fps)
    annotated = {seq_key(s) for s in targets}
    ann_hits = hits[hits["target"].isin(annotated)]
    by_query = {q: g.head(cfg["top_m"]) for q, g in ann_hits.groupby("query", sort=False)}

    ev = te[te["label"].isin(cfg["eval_labels"]) & (te["n_pos"] > 0)].reset_index(drop=True)
    variants = {"nn1": [], "knn": [], "knn_lig": []}
    nearest = np.zeros(len(ev))
    for i, (fasta, smi, L) in enumerate(zip(ev["fasta"], ev["smiles"], ev["L"], strict=True)):
        g = by_query.get(seq_key(fasta))
        per = {v: [] for v in variants}
        if g is not None:
            qfp = fps(smi)
            for rank, h in enumerate(g.itertuples()):
                q_idx, t_idx = aligned_positions(h.qaln, h.taln, h.qstart, h.tstart)
                tgt = targets[train_seqs[h.target]]
                if rank == 0:
                    per["nn1"].append((q_idx, t_idx, 1.0, tgt["freq"]))
                per["knn"].append((q_idx, t_idx, h.fident, tgt["freq"]))
                lf = ligand_freq(tgt, qfp, cfg["ligand_weight_floor"])
                per["knn_lig"].append((q_idx, t_idx, h.fident, lf))
        for v in variants:
            variants[v].append(add_noise(transfer_scores(L, per[v]), rng, noise))
        nearest[i] = g["fident"].iloc[0] if g is not None else 0.0

    y = ev["sites"].tolist()
    loc: dict = {
        "n_eval_pairs": len(ev),
        "frac_with_annotated_hit": float((nearest > 0).mean()),
        "random_analytic": random_expected(ev, ks),
        **{v: localization_metrics(s, y, ks, chunk) for v, s in variants.items()},
        "by_identity": {},
    }
    edges = cfg["identity_bins"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = np.flatnonzero((nearest >= lo) & (nearest < hi))
        if lo == 0.0:
            idx = np.flatnonzero(nearest < hi)  # includes "no hit" (nearest == 0)
        key = f"[{lo},{hi})"
        if len(idx) == 0:
            loc["by_identity"][key] = {"n_pairs": 0}
            continue
        loc["by_identity"][key] = {
            "random_analytic": random_expected(ev.iloc[idx], ks),
            **{v: localization_metrics([s[i] for i in idx], [y[i] for i in idx], ks, chunk)
               for v, s in variants.items()},
        }  # fmt: skip

    # ---- ligand-agnostic oracle (TEST labels; ceiling for any ligand-blind ranking) ----------
    multi = ev.groupby("fasta")["sites"].transform("size") >= 2
    sub = ev[multi]
    ins, loo = [], []
    for _, g in sub.groupby("fasta", sort=False):
        Y = np.stack(g["sites"].to_list()).astype(np.float64)
        tot = Y.sum(axis=0)
        for r in range(len(Y)):
            ins.append(add_noise(tot.copy(), rng, 0.5))  # integer counts: noise < 1 keeps order
            loo.append(add_noise(tot - Y[r], rng, 0.5))
    ys = [y for _, g in sub.groupby("fasta", sort=False) for y in g["sites"]]
    oracle = {"n_pairs": len(sub), "n_proteins": int(sub["fasta"].nunique())}
    if len(sub):
        oracle |= {
            "in_sample": localization_metrics(ins, ys, ks, chunk),
            "leave_one_out": localization_metrics(loo, ys, ks, chunk),
            "random_analytic": random_expected(sub, ks),
        }
    return {"split_audit": audit, "localization": loc, "ligand_agnostic_oracle": oracle}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/homology.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="*", help="override cfg splits")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    for name in args.splits or cfg["splits"]:
        t0 = time.time()
        rng = np.random.default_rng([args.seed, cfg["splits"].index(name)])
        m = run_split(name, cfg, rng)
        p = write_result(
            f"homology_{name}_{args.seed}", {**cfg, "split": name}, args.seed, m, time.time() - t0
        )
        lo, au = m["localization"], m["split_audit"]["all_test_rows"]
        print(
            f"{name:9s} hit={lo['frac_with_annotated_hit']:.2f} "
            f"median_id={au['quantiles']['0.5']:.2f} ge90={au['frac_ge']['0.9']:.2f} | "
            + " ".join(
                f"{v}@1/5={lo[v]['brhr@1']:.3f}/{lo[v]['brhr@5']:.3f}"
                for v in ("nn1", "knn", "knn_lig")
            )
            + " rand@1/5={brhr@1:.3f}/{brhr@5:.3f}".format(**lo["random_analytic"])
            + f" ({time.time() - t0:.0f}s) -> {p}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
