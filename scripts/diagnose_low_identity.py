"""Diagnostic: why does homology kNN beat random on pairs whose best hit is < 30% identity?

Recomputes the kNN scores of run_homology_baselines.py for one split and, for test positives in
the lowest identity stratum (top-bit-score hit fident < 0.3, or no hit), separates:
  * no hit at all (scores are pure tie noise -> must equal random; a code sanity check);
  * binning artefact: another of the top-m hits has fident >= 0.3;
  * genuinely remote hits, by alignment coverage of the test sequence.
Adds a generic residue-type prior (train P(binding | amino acid), no alignment) on the same
pairs, to see how much of the signal is "binding residues are often W/F/Y/...".
Writes results/diagnostic_lowid_{split}_{seed}.json.

    python scripts/diagnose_low_identity.py --split id_cc30_fold0 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from safedti.baselines.homology import aligned_positions, mmseqs_search, transfer_scores
from safedti.baselines.lookup import localization_metrics
from safedti.data.esm_cache import seq_key
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_homology_baselines import Fingerprints, add_noise, build_targets  # noqa: E402
from run_lookup_baselines import load_split, random_expected  # noqa: E402


def residue_type_prior(tab) -> dict[str, float]:
    """P(binding | amino acid) over annotated train rows."""
    pos, tot = {}, {}
    for fasta, y in zip(tab["fasta"], tab["sites"], strict=True):
        for aa, v in zip(fasta, y, strict=True):
            tot[aa] = tot.get(aa, 0) + 1
            pos[aa] = pos.get(aa, 0) + int(v)
    return {aa: pos[aa] / tot[aa] for aa in tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/homology.yaml")
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    rng = np.random.default_rng([args.seed, 99])
    t0 = time.time()
    ks, chunk, noise = cfg["brhr_k"], cfg["chunk"], cfg["tie_noise"]

    tr, te = load_split(args.split, cfg["site_column"])
    train_seqs = {seq_key(s): s for s in tr["fasta"].unique()}
    test_seqs = {seq_key(s): s for s in te["fasta"].unique()}
    hits = mmseqs_search(
        test_seqs, train_seqs, Path(cfg["workdir"]) / f"diag_{args.split}", cfg["mmseqs"],
        cfg["sensitivity"], cfg["max_seqs"], cfg["max_accept"], cfg["threads"],
    )  # fmt: skip
    tab = tr[tr["label"].isin(cfg["table_labels"]) & (tr["n_pos"] > 0)]
    targets = build_targets(tab, Fingerprints(**cfg["ligand_fp"]))
    annotated = {seq_key(s) for s in targets}
    ann = hits[hits["target"].isin(annotated)]
    by_query = {q: g.head(cfg["top_m"]) for q, g in ann.groupby("query", sort=False)}
    prior = residue_type_prior(tab)

    ev = te[te["label"].isin(cfg["eval_labels"]) & (te["n_pos"] > 0)].reset_index(drop=True)
    rows = []
    for fasta, L, y in zip(ev["fasta"], ev["L"], ev["sites"], strict=True):
        g = by_query.get(seq_key(fasta))
        top = g["fident"].iloc[0] if g is not None else 0.0
        if top >= 0.3:
            continue
        per = []
        if g is not None:
            for h in g.itertuples():
                q_idx, t_idx = aligned_positions(h.qaln, h.taln, h.qstart, h.tstart)
                per.append((q_idx, t_idx, h.fident, targets[train_seqs[h.target]]["freq"]))
        s_knn = add_noise(transfer_scores(L, per), rng, noise)
        s_aa = np.array([prior.get(a, 0.0) for a in fasta]) + rng.uniform(0, 1e-9, L)
        rows.append({
            "y": y, "knn": s_knn, "aa": s_aa, "L": L, "n_pos": int(y.sum()),
            "has_hit": g is not None,
            "max_fident_topm": float(g["fident"].max()) if g is not None else 0.0,
            "top_qcov": (g["qend"].iloc[0] - g["qstart"].iloc[0] + 1) / L if g is not None else 0.0,
        })  # fmt: skip

    def summarize(sel: list[dict]) -> dict:
        if not sel:
            return {"n_pairs": 0}
        import pandas as pd

        y = [r["y"] for r in sel]
        return {
            "n_pairs": len(sel),
            "knn": localization_metrics([r["knn"] for r in sel], y, ks, chunk),
            "residue_type_prior": localization_metrics([r["aa"] for r in sel], y, ks, chunk),
            "random_analytic": random_expected(pd.DataFrame(sel)[["L", "n_pos"]], ks),
        }

    hit = [r for r in rows if r["has_hit"]]
    m = {
        "split": args.split,
        "n_low_identity_pairs": len(rows),
        "all_low_identity": summarize(rows),
        "no_hit": summarize([r for r in rows if not r["has_hit"]]),
        "hit_but_other_topm_ge_0.3": summarize([r for r in hit if r["max_fident_topm"] >= 0.3]),
        "hit_all_topm_lt_0.3": summarize([r for r in hit if r["max_fident_topm"] < 0.3]),
        "hit_all_topm_lt_0.3_by_qcov": {
            f"[{a},{b})": summarize(
                [r for r in hit if r["max_fident_topm"] < 0.3 and a <= r["top_qcov"] < b]
            )
            for a, b in [(0.0, 0.3), (0.3, 0.8), (0.8, 1.01)]
        },
    }
    p = write_result(
        f"diagnostic_lowid_{args.split}_{args.seed}", {**cfg, "split": args.split}, args.seed, m,
        time.time() - t0,
    )  # fmt: skip
    for k in ("all_low_identity", "no_hit", "hit_but_other_topm_ge_0.3", "hit_all_topm_lt_0.3"):
        d = m[k]
        if d["n_pairs"]:
            print(f"{k:28s} n={d['n_pairs']:5d} knn@1={d['knn']['brhr@1']:.3f} "
                  f"aa@1={d['residue_type_prior']['brhr@1']:.3f} "
                  f"rand@1={d['random_analytic']['brhr@1']:.3f}")  # fmt: skip
    for b, d in m["hit_all_topm_lt_0.3_by_qcov"].items():
        if d["n_pairs"]:
            print(f"  remote, qcov {b:11s} n={d['n_pairs']:5d} knn@1={d['knn']['brhr@1']:.3f} "
                  f"aa@1={d['residue_type_prior']['brhr@1']:.3f}")  # fmt: skip
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
