"""Global-identity re-check of the protein-OOD splits (Needleman–Wunsch, as InteractBind uses).

InteractBind caps train–test protein similarity at 25/28/31/33% using global NW alignment but
does not state the identity formula, so two are reported: identical / alignment length (EMBOSS
needle) and identical / shorter length. Exhaustive NW over all test x train pairs is too slow on
CPU, so each test sequence is aligned to its top `n_candidates` MMseqs2 hits (by bit score); the
max over those is a lower bound on the true max. Test sequences with no MMseqs2 hit are reported
separately. A random test x train background shows what unrelated proteins score under NW.
Writes results/global_identity_{split}_{seed}.json.

    python scripts/run_global_identity_audit.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from safedti.baselines.homology import global_aligner, global_identity, mmseqs_search
from safedti.data.esm_cache import seq_key
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from run_lookup_baselines import load_split  # noqa: E402

FMT = "query,target,fident,qstart,qend,tstart,tend,qlen,tlen,bits"
DEFS = ("identity_aln", "identity_shorter")


def frac_ge(v: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    return {str(t): float((v >= t).mean()) for t in thresholds}


def quantiles(v: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


def run_split(name: str, cfg: dict, rng: np.random.Generator) -> dict:
    tr, te = load_split(name, cfg["site_column"])
    train = {seq_key(s): s for s in tr["fasta"].unique()}
    test = {seq_key(s): s for s in te["fasta"].unique()}
    hits = mmseqs_search(
        test, train, Path(cfg["workdir"]) / f"global_{name}", cfg["mmseqs"], cfg["sensitivity"],
        cfg["max_seqs"], cfg["max_accept"], cfg["threads"], fmt=FMT,
    )  # fmt: skip
    al = global_aligner(cfg["gap_open"], cfg["gap_extend"])
    best = {d: {} for d in DEFS}
    for q, g in hits.groupby("query", sort=False):
        ids = [
            global_identity(test[q], train[t], al) for t in g["target"].head(cfg["n_candidates"])
        ]
        for d in DEFS:
            best[d][q] = max(r[d] for r in ids)

    keys = te["fasta"].map(seq_key)
    has_hit = keys.isin(set(hits["query"])).to_numpy()
    thr = cfg["identity_thresholds"]
    nominal = float(name[-2:]) / 100
    out: dict = {
        "n_test_rows": len(te),
        "n_test_sequences": len(test),
        "frac_test_rows_with_mmseqs_hit": float(has_hit.mean()),
        "nominal_max_similarity": nominal,
    }
    for d in DEFS:
        v = keys[has_hit].map(best[d]).to_numpy(dtype=float)
        out[d] = {
            "rows_with_hit": {
                "quantiles": quantiles(v),
                "frac_ge": frac_ge(v, thr),
                "frac_ge_nominal": float((v >= nominal).mean()),
            },
            # rows without any MMseqs2 hit count as below every threshold (lower bound)
            "frac_all_test_rows_ge_nominal_lower_bound": float((v >= nominal).sum() / len(te)),
        }

    tk, qk = list(train), list(test)
    bg = {d: [] for d in DEFS}
    for _ in range(cfg["background_pairs"]):
        r = global_identity(test[qk[rng.integers(len(qk))]], train[tk[rng.integers(len(tk))]], al)
        for d in DEFS:
            bg[d].append(r[d])
    out["background_random_pairs"] = {
        d: {
            "quantiles": quantiles(np.asarray(v)),
            "frac_ge_nominal": float(np.mean(np.asarray(v) >= nominal)),
        }
        for d, v in bg.items()
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baselines/global_identity.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="*", help="override cfg global_splits")
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    for name in args.splits or cfg["global_splits"]:
        t0 = time.time()
        rng = np.random.default_rng([args.seed, cfg["global_splits"].index(name)])
        m = run_split(name, cfg, rng)
        write_result(
            f"global_identity_{name}_{args.seed}", {**cfg, "split": name}, args.seed, m,
            time.time() - t0,
        )  # fmt: skip
        a, b = m["identity_aln"], m["background_random_pairs"]["identity_aln"]
        print(
            f"{name}: rows>=nominal (NW aln-identity, lower bound) "
            f"{a['frac_all_test_rows_ge_nominal_lower_bound']:.3f}; median best "
            f"{a['rows_with_hit']['quantiles']['0.5']:.3f}; random-pair median "
            f"{b['quantiles']['0.5']:.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
