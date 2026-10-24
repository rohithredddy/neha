"""Build paper tables from results/*.json (never copy a number by hand; docs/05_EXPERIMENTS.md).

Reads lookup_*, homology_*, ligand_audit_*, pair_models_*, esm_probe_*, global_identity_* results
for one seed and writes Markdown and LaTeX to results/tables/. ID folds are shown as mean ± std
over folds. Published numbers come from
configs/published.yaml and are labelled as such.

    python scripts/make_tables.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

RES = Path("results")
OOD = ["p_ood_25", "p_ood_28", "p_ood_31", "p_ood_33"]
ID = [f"id_fold{k}" for k in range(5)]
C30 = [f"id_c30_fold{k}" for k in range(5)]
CC30 = [f"id_cc30_fold{k}" for k in range(5)]
ID_EXACT = "ID, FASTA-grouped (5 folds)"
ID_C30 = "ID, 30% set-cover clusters (5 folds)"
ID_CC30 = "ID, 30% connected components (5 folds)"
LC30 = [f"id_lc30_fold{k}" for k in range(4)]
ID_LC30 = "ID, leak-free components (4 folds, proposed)"


def load(prefix: str, split: str, seed: int) -> dict | None:
    p = RES / f"{prefix}_{split}_{seed}.json"
    return json.loads(p.read_text(encoding="utf-8"))["metrics"] if p.exists() else None


def get(d: dict | None, path: str) -> float | None:
    for k in path.split("/"):
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def cell(prefix: str, path: str, splits: list[str], seed: int, scale: float = 100) -> str:
    """Formatted value for one split, or mean ± std over several (ID folds)."""
    vals = [get(load(prefix, s, seed), path) for s in splits]
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "—"
    if len(splits) == 1:
        return f"{vals[0]:.1f}"
    return (
        f"{np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"
    )


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def latex(header: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: s.replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_")  # noqa: E731
    out = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule"]
    out.append(" & ".join(esc(h) for h in header) + r" \\ \midrule")
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def groups() -> list[tuple[str, list[str]]]:
    return (
        [(ID_EXACT, ID)]
        + [(s, [s]) for s in OOD]
        + [
            (ID_C30, C30),
            (ID_CC30, CC30),
            (ID_LC30, LC30),
        ]
    )


def published(name: str, id_value: float | None, ood: dict, key: str) -> float | None:
    """Published value for a row; their ID folds are closest to our FASTA-grouped ID."""
    if name == ID_EXACT:
        return id_value
    return None if name in (ID_C30, ID_CC30, ID_LC30) else get(ood, f"{name}/{key}")


def t_binary(seed: int, pub: dict) -> tuple[list[str], list[list[str]]]:
    h = ["split", "FASTA prior", "UniProt prior", "SMILES prior", "heavy atoms",
         "ligand-only GBM", "GBM, size-matched", "FusionDTI (published)"]  # fmt: skip
    fus_id = pub["interactbind_table2_id"]["FusionDTI"]["auroc"]
    fus_ood = pub["interactbind_fusiondti_ood"]
    rows = []
    for name, sp in groups():
        p = published(name, fus_id, fus_ood, "auroc")
        rows.append([
            name,
            cell("lookup", "binary/fasta_prior/auroc", sp, seed),
            cell("lookup", "binary/uniprot_id_prior/auroc", sp, seed),
            cell("lookup", "binary/smiles_prior/auroc", sp, seed),
            cell("ligand_audit", "single_descriptor_auroc/HeavyAtomCount", sp, seed),
            cell("ligand_audit", "ligand_only_gbm/auroc", sp, seed),
            cell("ligand_audit", "size_matched/ligand_only_gbm/auroc", sp, seed),
            f"{p:.1f}" if p is not None else "—",
        ])  # fmt: skip
    return h, rows


def t_localization(seed: int, k: int, pub: dict) -> tuple[list[str], list[list[str]]]:
    h = ["split", "random", "exact lookup", "homology NN", "homology kNN", "kNN + ligand",
         "oracle LOO*", "FusionDTI (published)"]  # fmt: skip
    fus_id = pub["interactbind_table2_id"]["FusionDTI"].get(f"brhr{k}")
    fus_ood = pub["interactbind_fusiondti_ood"]
    rows = []
    for name, sp in groups():
        p = published(name, fus_id, fus_ood, f"brhr{k}")
        rows.append([
            name,
            cell("homology", f"localization/random_analytic/brhr@{k}", sp, seed),
            cell("lookup", f"localization/lookup_all/brhr@{k}", sp, seed),
            cell("homology", f"localization/nn1/brhr@{k}", sp, seed),
            cell("homology", f"localization/knn/brhr@{k}", sp, seed),
            cell("homology", f"localization/knn_lig/brhr@{k}", sp, seed),
            cell("homology", f"ligand_agnostic_oracle/leave_one_out/brhr@{k}", sp, seed),
            f"{p:.1f}" if p is not None else "—",
        ])  # fmt: skip
    return h, rows


def t_split_audit(seed: int) -> tuple[list[str], list[list[str]]]:
    h = ["split", "nominal max identity", "median nearest identity", "≥ nominal",
         "≥ 50%", "≥ 90%", "identical", "≥ nominal (cov ≥ 0.8)", "≥ 90% (cov ≥ 0.8)"]  # fmt: skip
    rows = []
    for name, sp in groups():
        nom = "—" if name.startswith("ID") else f"{name[-2:]}%"
        thr = "—" if name.startswith("ID") else f"0.{name[-2:]}"
        a = "split_audit/all_test_rows"
        rows.append([
            name, nom,
            cell("homology", f"{a}/quantiles/0.5", sp, seed),
            cell("homology", f"{a}/frac_ge/{thr}", sp, seed) if thr != "—" else "—",
            cell("homology", f"{a}/frac_ge/0.5", sp, seed),
            cell("homology", f"{a}/frac_ge/0.9", sp, seed),
            cell("homology", f"{a}/frac_ge/0.999", sp, seed),
            cell("homology", f"{a}_qcov/frac_ge/{thr}", sp, seed) if thr != "—" else "—",
            cell("homology", "split_audit/all_test_rows_qcov/frac_ge/0.9", sp, seed),
        ])  # fmt: skip
    return h, rows


def t_by_identity(seed: int, k: int) -> tuple[list[str], list[list[str]]]:
    first = load("homology", OOD[0], seed)
    bins = list(get(first, "localization/by_identity") or {})
    h = ["split", *[f"{b} kNN / random (n)" for b in bins]]
    rows = []
    for name, sp in groups():
        r = [name]
        for b in bins:
            base = f"localization/by_identity/{b}"
            n = [get(load("homology", s, seed), f"{base}/knn/n_pairs") for s in sp]
            n = int(sum(x for x in n if x)) if any(n) else 0
            if n == 0:
                r.append("— (0)")
                continue
            r.append(
                f"{cell('homology', f'{base}/knn/brhr@{k}', sp, seed)} / "
                f"{cell('homology', f'{base}/random_analytic/brhr@{k}', sp, seed)} ({n})"
            )
        rows.append(r)
    return h, rows


def t_pair_models(seed: int) -> tuple[list[str], list[list[str]]]:
    vs = ["ligand_only", "protein_only", "pair"]
    h = ["split", *[f"{v} AUROC" for v in vs], *[f"{v}, size-matched" for v in vs]]
    rows = []
    for name, sp in [(ID_EXACT, ID), (ID_LC30, LC30), *[(s, [s]) for s in OOD]]:
        rows.append([
            name,
            *[cell("pair_models", f"{v}/all/auroc", sp, seed) for v in vs],
            *[cell("pair_models", f"{v}/size_matched/auroc", sp, seed) for v in vs],
        ])  # fmt: skip
    return h, rows


def t_esm_probe(seed: int, pub: dict) -> tuple[list[str], list[list[str]]]:
    h = ["split", "ESM probe @1", "ESM probe @5", "homology kNN @1", "homology kNN @5",
         "random @1", "random @5", "FusionDTI @1 (publ.)", "FusionDTI @5 (publ.)"]  # fmt: skip
    fus = pub["interactbind_table2_id"]["FusionDTI"]
    fus_ood = pub["interactbind_fusiondti_ood"]
    rows = []
    for name, sp in [(ID_EXACT, ID), (ID_LC30, LC30), *[(o, [o]) for o in OOD]]:
        p1 = fus["brhr1"] if sp is ID else None
        p5 = fus["brhr5"] if sp is ID else get(fus_ood, f"{sp[0]}/brhr5")
        rows.append([
            name,
            cell("esm_probe", "probe/brhr@1", sp, seed),
            cell("esm_probe", "probe/brhr@5", sp, seed),
            cell("homology", "localization/knn/brhr@1", sp, seed),
            cell("homology", "localization/knn/brhr@5", sp, seed),
            cell("esm_probe", "random_analytic/brhr@1", sp, seed),
            cell("esm_probe", "random_analytic/brhr@5", sp, seed),
            f"{p1:.1f}" if p1 is not None else "—",
            f"{p5:.1f}" if p5 is not None else "—",
        ])  # fmt: skip
    return h, rows


def t_lsg(seed: int) -> tuple[list[str], list[list[str]]]:
    h = ["split", "pairs (proteins)", "kNN LSG@1", "kNN+ligand LSG@1", "ceiling LSG@1",
         "kNN+ligand LSG@5", "ceiling LSG@5"]  # fmt: skip
    rows = []
    for name, sp in [(ID_EXACT, ID), (ID_LC30, LC30), *[(o, [o]) for o in OOD]]:
        L = "ligand_specificity"
        n = [get(load("homology", s, seed), f"{L}/@1/knn") for s in sp]
        n = [x for x in n if x]
        counts = f"{sum(x['n_pairs'] for x in n)} ({sum(x['n_proteins'] for x in n)})" if n else "—"
        rows.append([
            name, counts,
            cell("homology", f"{L}/@1/knn/lsg", sp, seed),
            cell("homology", f"{L}/@1/knn_lig/lsg", sp, seed),
            cell("homology", f"{L}/@1/own_sites_ceiling/lsg", sp, seed),
            cell("homology", f"{L}/@5/knn_lig/lsg", sp, seed),
            cell("homology", f"{L}/@5/own_sites_ceiling/lsg", sp, seed),
        ])  # fmt: skip
    return h, rows


def seed_cell(prefix: str, path: str, splits: list[str], seeds: list[int]) -> str:
    """Mean ± std over seeds of the fold-averaged value."""
    per_seed = []
    for sd in seeds:
        v = [get(load(prefix, s, sd), path) for s in splits]
        v = [x for x in v if x is not None]
        if len(v) == len(splits):
            per_seed.append(100 * float(np.mean(v)))
    if len(per_seed) < 2:
        return "—"
    return f"{np.mean(per_seed):.1f} ± {np.std(per_seed, ddof=1):.1f} (n={len(per_seed)})"


def t_size_matched_seeds(seeds: list[int]) -> tuple[list[str], list[list[str]]]:
    h = ["split", "size-matched ligand-only GBM AUROC (mean ± std over sampling seeds)"]
    rows = []
    for name, sp in [(ID_EXACT, ID), (ID_LC30, LC30), *[(o, [o]) for o in OOD]]:
        rows.append(
            [name, seed_cell("ligand_audit", "size_matched/ligand_only_gbm/auroc", sp, seeds)]
        )
    return h, rows


def t_global_identity(seed: int) -> tuple[list[str], list[list[str]]]:
    h = ["split", "nominal max similarity", "rows ≥ nominal (NW, /aln length)",
         "rows ≥ nominal (NW, /shorter)", "median best NW identity", "random pairs: median",
         "random pairs ≥ nominal"]  # fmt: skip
    rows = []
    for s in OOD:
        g = "global_identity"
        rows.append([
            s, f"{s[-2:]}%",
            cell(g, "identity_aln/frac_all_test_rows_ge_nominal_lower_bound", [s], seed),
            cell(g, "identity_shorter/frac_all_test_rows_ge_nominal_lower_bound", [s], seed),
            cell(g, "identity_aln/rows_with_hit/quantiles/0.5", [s], seed),
            cell(g, "background_random_pairs/identity_aln/quantiles/0.5", [s], seed),
            cell(g, "background_random_pairs/identity_aln/frac_ge_nominal", [s], seed),
        ])  # fmt: skip
    return h, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    pub = yaml.safe_load(Path("configs/published.yaml").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "T_binary_auroc": t_binary(args.seed, pub),
        "T_localization_brhr1": t_localization(args.seed, 1, pub),
        "T_localization_brhr5": t_localization(args.seed, 5, pub),
        "T_split_audit": t_split_audit(args.seed),
        "T_brhr1_by_identity": t_by_identity(args.seed, 1),
        "T_pair_models_auroc": t_pair_models(args.seed),
        "T_esm_probe_brhr": t_esm_probe(args.seed, pub),
        "T_global_identity": t_global_identity(args.seed),
        "T_ligand_specificity_gain": t_lsg(args.seed),
        "T_size_matched_seeds": t_size_matched_seeds([0, 1, 2]),
    }
    md = []
    for name, (h, rows) in specs.items():
        (out / f"{name}.tex").write_text(latex(h, rows) + "\n", encoding="utf-8")
        md += [f"### {name}", "", table(h, rows), ""]
    md.append("*oracle LOO uses test labels of other ligands of the same protein (a ceiling).")
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
