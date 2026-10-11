"""Precompute frozen ESM-2 residue embeddings for every unique InteractBind sequence.

Resumable: sequences whose .npy already exists are skipped; files are written atomically.
Order: OOD-split sequences first (test/val, then train), then the rest of the ID table, so
OOD experiments can start before the run finishes.

    python scripts/precompute_esm.py --model esm2_t30_150M_UR50D --out data/processed/esm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from safedti.data.esm_cache import cache_path, merge_windows, seq_key, window_spans
from safedti.utils.seeding import seed_everything

RAW = Path("data/raw/interactbind")
ORDER = [
    *[f"p_ood_{p}_{s}" for s in ("test", "validation") for p in (25, 28, 31, 33)],
    *[f"p_ood_{p}_train" for p in (25, 28, 31, 33)],
    "affinity_train",
]


def ordered_unique_sequences() -> list[tuple[int, str]]:
    """(priority group, sequence) for each unique sequence, first occurrence wins."""
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for rank, name in enumerate(ORDER):
        fasta = pd.read_parquet(RAW / f"{name}.parquet", columns=["fasta"])["fasta"].dropna()
        for s in fasta:
            if s not in seen:
                seen.add(s)
                out.append((rank, s))
    return out


def clean(seq: str, valid: set[str]) -> tuple[str, int]:
    """Uppercase and map tokens unknown to ESM to 'X'. Length is preserved (alignment!)."""
    s = seq.upper()
    bad = sum(c not in valid for c in s)
    return ("".join(c if c in valid else "X" for c in s) if bad else s), bad


@torch.inference_mode()
def embed(model, converter, layer: int, seqs: list[str]) -> list[np.ndarray]:
    _, _, toks = converter([(str(i), s) for i, s in enumerate(seqs)])
    rep = model(toks, repr_layers=[layer])["representations"][layer]
    return [rep[i, 1 : len(s) + 1].float().numpy() for i, s in enumerate(seqs)]


def save_atomic(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, x.astype(np.float16))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--out", default="data/processed/esm")
    ap.add_argument("--token-budget", type=int, default=4096, help="residues per batch")
    ap.add_argument("--limit", type=int, default=0, help="only first N sequences (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    import esm

    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval()
    layer = model.num_layers
    converter = alphabet.get_batch_converter()
    valid = {t for t in alphabet.all_toks if len(t) == 1 and t.isalpha()}

    ranked = ordered_unique_sequences()
    if args.limit:
        ranked = ranked[: args.limit]
    seqs = [s for _, s in ranked]
    # Within a priority group, sort by length so batches carry little padding.
    todo = [s for _, s in sorted(ranked, key=lambda t: (t[0], len(t[1])))
            if not cache_path(args.out, args.model, s).exists()]
    total_res = sum(len(s) for s in todo)
    print(f"{len(seqs)} unique sequences, {len(seqs) - len(todo)} cached, {len(todo)} to do "
          f"({total_res:,} residues), threads={torch.get_num_threads()}", flush=True)

    n_bad_seqs, n_bad_res = 0, 0
    t0, done_res, last_log = time.time(), 0, 0.0
    short = [s for s in todo if len(s) <= 1022]
    long_ = [s for s in todo if len(s) > 1022]

    def progress(n: int) -> None:
        nonlocal done_res, last_log
        done_res += n
        if time.time() - last_log > 60:
            el = time.time() - t0
            eta = el / done_res * (total_res - done_res) / 3600
            print(f"[{time.strftime('%H:%M:%S')}] {done_res:,}/{total_res:,} residues "
                  f"({100 * done_res / total_res:.1f}%) elapsed {el / 3600:.2f} h, "
                  f"ETA {eta:.2f} h", flush=True)
            last_log = time.time()

    # Short sequences: keep the given priority order, batch consecutive ones by token budget.
    batch: list[str] = []
    for s in short + [None]:  # type: ignore[list-item]
        if s is not None and (not batch or (len(batch) + 1) * max(map(len, batch + [s]))
                              <= args.token_budget):
            batch.append(s)
            continue
        if batch:
            cleaned = []
            for b in batch:
                c, bad = clean(b, valid)
                n_bad_seqs += bad > 0
                n_bad_res += bad
                cleaned.append(c)
            for b, x in zip(batch, embed(model, converter, layer, cleaned), strict=True):
                assert x.shape[0] == len(b)
                save_atomic(cache_path(args.out, args.model, b), x)
            progress(sum(map(len, batch)))
        batch = [s] if s is not None else []

    # Long sequences: overlapping windows, averaged.
    for s in long_:
        c, bad = clean(s, valid)
        n_bad_seqs += bad > 0
        n_bad_res += bad
        spans = window_spans(len(c))
        embs = [embed(model, converter, layer, [c[a:b]])[0] for a, b in spans]
        x = merge_windows(len(c), spans, embs)
        save_atomic(cache_path(args.out, args.model, s), x)
        progress(len(s))

    manifest = {
        "model": args.model, "layer": layer, "dim": int(model.embed_dim), "dtype": "float16",
        "n_sequences": len(seqs), "seqs_with_unknown_tokens": int(n_bad_seqs),
        "unknown_residues_mapped_to_X": int(n_bad_res), "n_long_windowed": len(long_),
        "window": {"max_len": 1022, "stride": 511, "merge": "mean"},
        "wall_time_s_this_run": time.time() - t0, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "keys_sample": [seq_key(s) for s in seqs[:3]],
    }
    Path(args.out, args.model).mkdir(parents=True, exist_ok=True)
    Path(args.out, args.model, "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
