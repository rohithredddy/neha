"""M1: download InteractBind (public HF release), verify it, print a report, write DATA_CARD.md.

Network access happens here only (CLAUDE.md rule 8). Outputs:
    data/raw/interactbind/{config}_{split}.parquet   raw tables
    data/raw/interactbind/verify_report.json         every number printed below
    data/processed/DATA_CARD.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from safedti.data.interactbind import CONFIGS, HF_REPO, SITE_COLUMNS, TYPES, parse_site_string
from safedti.metrics.localization import random_brhr_expected
from safedti.utils.seeding import seed_everything

RAW = Path("data/raw/interactbind")
PROCESSED = Path("data/processed")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> tuple[dict[str, Path], str]:
    """Download every config/split to parquet; return paths and the dataset revision sha."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(HF_REPO).sha
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cfg in CONFIGS:
        dsd = load_dataset(HF_REPO, cfg, revision=revision, cache_dir=str(RAW / "hf_cache"))
        for split, ds in dsd.items():
            p = RAW / f"{cfg}_{split}.parquet"
            if not p.exists():
                ds.to_pandas().to_parquet(p, index=False)
            paths[f"{cfg}/{split}"] = p
    return paths, revision


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def analyse_table(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """All checks for one table. Returns JSON-serialisable stats."""
    r: dict = {"rows": len(df)}
    req = ["fasta", "smiles", "label", *SITE_COLUMNS]
    missing = df[req].isna()
    r["missing_values_per_column"] = {c: int(v) for c, v in missing.sum().items() if v}
    r["rows_dropped_missing"] = int(missing.any(axis=1).sum())
    df = df[~missing.any(axis=1)].reset_index(drop=True)
    r["rows_after_drop"] = len(df)
    r["unique_proteins_uniprot"] = int(df["uniprot_id"].nunique())
    r["unique_fasta"] = int(df["fasta"].nunique())
    r["unique_ligands_smiles"] = int(df["smiles"].nunique())
    r["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().items()}
    r["duplicate_fasta_smiles_pairs"] = int(df.duplicated(["fasta", "smiles"]).sum())
    L = df["fasta"].str.len().to_numpy()
    r["L"] = {
        "min": int(L.min()),
        "median": float(np.median(L)),
        "max": int(L.max()),
        "frac_gt_1024": float((L > 1024).mean()),
    }

    # Parse all site columns; check lengths against FASTA.
    sites = {}
    len_mismatch = Counter()
    parse_fail = Counter()
    for col, key in SITE_COLUMNS.items():
        arrs = []
        for s, n in zip(df[col].to_numpy(), L, strict=True):
            try:
                a = parse_site_string(s)
            except ValueError:
                parse_fail[key] += 1
                a = np.zeros(n, dtype=np.uint8)
            if len(a) != n:
                len_mismatch[key] += 1
            arrs.append(a)
        sites[key] = arrs
    r["site_parse_failures"] = dict(parse_fail)
    r["site_len_mismatch_vs_fasta"] = dict(len_mismatch)

    ok = [len(sites["overall"][i]) == L[i] for i in range(len(df))]
    overall_n = np.array([int(a.sum()) for a in sites["overall"]])
    lab = df["label"].to_numpy()
    for lv in (0, 1):
        m = lab == lv
        r[f"label{lv}_frac_with_overall_site"] = (
            float((overall_n[m] > 0).mean()) if m.any() else None
        )
    pos_annot = (lab == 1) & (overall_n > 0)
    r["n_pos_annotated"] = int(pos_annot.sum())
    npos = overall_n[pos_annot]
    if len(npos):
        r["n_binding_residues_pos"] = {
            "min": int(npos.min()),
            "p10": float(np.percentile(npos, 10)),
            "median": float(np.median(npos)),
            "p90": float(np.percentile(npos, 90)),
            "max": int(npos.max()),
        }
    # Overall == OR of six types?
    or_mismatch = 0
    for i in range(len(df)):
        if not ok[i]:
            continue
        u = np.zeros(L[i], dtype=bool)
        for t in TYPES:
            if len(sites[t][i]) == L[i]:
                u |= sites[t][i].astype(bool)
        or_mismatch += int((u != sites["overall"][i].astype(bool)).any())
    r["overall_ne_union_of_types"] = or_mismatch
    # Per-type prevalence among annotated positives (for inverse-frequency weights).
    r["type_frac_of_annotated_pos_with_any"] = {
        t: float(np.mean([sites[t][i].sum() > 0 for i in np.flatnonzero(pos_annot)])) for t in TYPES
    }
    r["type_mean_residues_annotated_pos"] = {
        t: float(np.mean([sites[t][i].sum() for i in np.flatnonzero(pos_annot)])) for t in TYPES
    }
    # Binding residues beyond the 1024 crop window (spec §7 truncation risk).
    beyond = [
        int(np.flatnonzero(sites["overall"][i]).max()) >= 1024 for i in np.flatnonzero(pos_annot)
    ]
    r["pos_with_binding_residue_index_ge_1024"] = int(np.sum(beyond))
    # Random-ranking BRHR (analytic), averaged over annotated positives.
    r["random_brhr_expected"] = {
        k: float(
            np.mean(
                [
                    random_brhr_expected(int(L[i]), int(overall_n[i]), k)
                    for i in np.flatnonzero(pos_annot)
                ]
            )
        )
        for k in (1, 3, 5)
    }
    # att_site format.
    pat = df["att_site"].str.replace(r"^[0-9A-Za-z]{4}_", "PDB_", regex=True)
    pat = pat.str.replace(
        r"_[A-Za-z0-9]+$", lambda m: "_" + re.sub(r"\d", "#", m.group(0)[1:]), regex=True
    )
    r["att_site_patterns_top"] = dict(Counter(pat).most_common(8))
    r["att_site_examples"] = df["att_site"].head(5).tolist()

    # N2 early signal: how much do binding sites vary across ligands of the SAME protein?
    # Mean pairwise Jaccard of overall sites between different annotated positives sharing FASTA.
    by_prot: dict[str, list[int]] = {}
    for i in np.flatnonzero(pos_annot):
        by_prot.setdefault(df["fasta"].iat[i], []).append(i)
    multi = [ix for ix in by_prot.values() if len(ix) >= 2]
    js = []
    for ix in multi:
        ix = rng.permutation(ix)[:10]
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                js.append(
                    jaccard(
                        sites["overall"][ix[a]].astype(bool), sites["overall"][ix[b]].astype(bool)
                    )
                )
    r["proteins_with_ge2_annotated_ligands"] = len(multi)
    r["frac_annotated_pos_in_multi_ligand_proteins"] = float(
        sum(len(ix) for ix in multi) / max(1, pos_annot.sum())
    )
    if js:
        js = np.array(js)
        r["same_protein_site_jaccard"] = {
            "n_pairs": int(len(js)),
            "mean": float(js.mean()),
            "median": float(np.median(js)),
            "frac_identical": float((js == 1.0).mean()),
            "frac_gt_0.5": float((js > 0.5).mean()),
        }
    return r


def ood_overlap(tables: dict[str, pd.DataFrame], cfg: str) -> dict:
    tr = tables.get(f"{cfg}/train")
    te = tables.get(f"{cfg}/test")
    if tr is None or te is None:
        return {}
    return {
        "test_fasta_in_train": float(te["fasta"].isin(set(tr["fasta"])).mean()),
        "test_smiles_in_train": float(te["smiles"].isin(set(tr["smiles"])).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    paths, revision = download()
    tables = {k: pd.read_parquet(p) for k, p in paths.items()}
    report: dict = {
        "hf_repo": HF_REPO,
        "revision": revision,
        "download_date": time.strftime("%Y-%m-%d"),
        "seed": args.seed,
        "files": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
        "columns": list(next(iter(tables.values())).columns),
        "tables": {},
        "ood_overlap": {},
    }
    for k, df in tables.items():
        print(f"analysing {k} ({len(df)} rows) ...", flush=True)
        report["tables"][k] = analyse_table(df, rng)
    for cfg in CONFIGS[1:]:
        report["ood_overlap"][cfg] = ood_overlap(tables, cfg)
    report["wall_time_s"] = time.time() - t0

    (RAW / "verify_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    write_data_card(report)
    return 0


def write_data_card(rep: dict) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DATA_CARD — InteractBind (public release)",
        "",
        f"- Source: https://huggingface.co/datasets/{rep['hf_repo']}",
        f"- Revision (commit sha): `{rep['revision']}`",
        f"- Download date: {rep['download_date']}",
        "- License: CC-BY-4.0",
        "- Labels: residue-level binary binding sites per NCI type (no atom axis).",
        "",
        "| table | rows | proteins (fasta) | ligands | labels | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for k, t in rep["tables"].items():
        lines.append(
            f"| {k} | {t['rows']} | {t['unique_fasta']} | {t['unique_ligands_smiles']} "
            f"| {t['label_counts']} | `{rep['files'][k]['sha256'][:16]}…` |"
        )
    lines += [
        "",
        "Full statistics: `data/raw/interactbind/verify_report.json` "
        "(generated by `scripts/verify_interactbind.py`).",
    ]
    (PROCESSED / "DATA_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
