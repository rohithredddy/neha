"""Compact IEEE two-column tables for paper/ from results/*.json (no number typed by hand).

Writes paper/tables/{binary,localization,leakage_ood,leakage_id}.tex (booktabs, tabular body
only; captions live in paper/main.tex). Published numbers come from configs/published.yaml.

    python scripts/make_paper_tables.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from make_tables import C30, CC30, ID, LC30, OOD, cell, get  # noqa: E402

GROUPS = [
    ("ID (FASTA-grouped)", ID),
    ("ID (leak-free)", LC30),
    *[(f"OOD-{s[-2:]}", [s]) for s in OOD],
]


def tex(s: str) -> str:
    return s.replace("±", r"$\pm$").replace("%", r"\%").replace("—", "--")


def tabular(cols: str, header: list[str], rows: list[list[str]]) -> str:
    out = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def fmt(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def binary(seed: int, pub: dict) -> str:
    fus = pub["interactbind_fusiondti_ood"]
    rows = []
    for name, sp in GROUPS:
        p = pub["interactbind_table2_id"]["FusionDTI"]["auroc"] if sp is ID else (
            get(fus, f"{sp[0]}/auroc") if sp[0] in OOD else None)  # fmt: skip
        rows.append([
            name,
            cell("ligand_audit", "single_descriptor_auroc/HeavyAtomCount", sp, seed),
            cell("pair_models", "ligand_only/all/auroc", sp, seed),
            cell("pair_models", "protein_only/all/auroc", sp, seed),
            cell("pair_models", "pair/all/auroc", sp, seed),
            cell("pair_models", "pair/size_matched/auroc", sp, seed),
            fmt(p),
        ])  # fmt: skip
    head = ["Split", "Heavy atoms$^a$", "Ligand", "Protein", "Ligand+Prot.", "L+P size-m.$^b$",
            "FusionDTI$^c$"]  # fmt: skip
    return tabular("lcccccc", head, rows)


def localization(seed: int, pub: dict) -> str:
    fus = pub["interactbind_fusiondti_ood"]
    t2 = pub["interactbind_table2_id"]["FusionDTI"]
    rows = []
    for name, sp in GROUPS:
        p1 = t2["brhr1"] if sp is ID else None
        p5 = t2["brhr5"] if sp is ID else (get(fus, f"{sp[0]}/brhr5") if sp[0] in OOD else None)
        rows.append([
            name,
            cell("homology", "localization/random_analytic/brhr@1", sp, seed),
            cell("homology", "localization/random_analytic/brhr@5", sp, seed),
            cell("homology", "localization/knn/brhr@1", sp, seed),
            cell("homology", "localization/knn/brhr@5", sp, seed),
            cell("esm_probe", "probe/brhr@1", sp, seed),
            cell("esm_probe", "probe/brhr@5", sp, seed),
            cell("homology", "ligand_agnostic_oracle/leave_one_out/brhr@1", sp, seed),
            fmt(p1), fmt(p5),
        ])  # fmt: skip
    head = ["Split", "Rand.@1", "Rand.@5", "Hom.@1", "Hom.@5", "Probe@1", "Probe@5", "Oracle@1$^e$",
            "FDTI@1$^c$", "FDTI@5$^c$"]  # fmt: skip
    return tabular("lccccccccc", head, rows)


def leakage_ood(seed: int) -> str:
    rows = []
    for s in OOD:
        thr, g, lb = f"0.{s[-2:]}", "global_identity", "frac_all_test_rows_ge_nominal_lower_bound"
        rows.append([
            f"OOD-{s[-2:]}",
            cell("homology", f"split_audit/all_test_rows_qcov/frac_ge/{thr}", [s], seed),
            cell(g, f"identity_aln/{lb}", [s], seed),
            cell(g, f"identity_shorter/{lb}", [s], seed),
            cell(g, "background_random_pairs/identity_aln/frac_ge_nominal", [s], seed),
        ])  # fmt: skip
    head = ["Split", "Local, cov$\\geq$0.8", "NW / aln.", "NW / shorter", "Random pairs"]
    return tabular("lcccc", head, rows)


def leakage_id(seed: int) -> str:
    rows = []
    variants = [("FASTA-grouped", ID), ("MMseqs2 30%, set cover", C30),
                ("MMseqs2 30%, conn. comp.", CC30), ("Leak-defined (ours)", LC30)]  # fmt: skip
    for name, sp in variants:
        rows.append([
            name,
            cell("homology", "split_audit/all_test_rows_qcov/frac_ge/0.9", sp, seed),
            cell("homology", "split_audit/all_test_rows/quantiles/0.5", sp, seed),
            cell("homology", "localization/knn/brhr@1", sp, seed),
            cell("homology", "localization/knn/brhr@5", sp, seed),
            cell("pair_models", "pair/all/auroc", sp, seed)
            if sp in (ID, LC30) else cell("ligand_audit", "ligand_only_gbm/auroc", sp, seed),
        ])  # fmt: skip
    head = ["ID split", "Leak$^f$", "Med. id.", "Hom.@1", "Hom.@5", "Bin. AUROC$^g$"]
    return tabular("lccccc", head, rows)


def lsg(seed: int) -> str:
    rows = []
    for name, sp in GROUPS:
        L = "ligand_specificity"
        rows.append([
            name,
            cell("homology", f"{L}/@1/knn/lsg", sp, seed),
            cell("homology", f"{L}/@1/knn_lig/lsg", sp, seed),
            cell("homology", f"{L}/@1/own_sites_ceiling/lsg", sp, seed),
            cell("homology", f"{L}/@5/knn_lig/lsg", sp, seed),
            cell("homology", f"{L}/@5/own_sites_ceiling/lsg", sp, seed),
        ])  # fmt: skip
    head = ["Split", "kNN@1", "kNN+lig.@1", "Ceiling@1", "kNN+lig.@5", "Ceiling@5"]
    return tabular("lccccc", head, rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()
    pub = yaml.safe_load(Path("configs/published.yaml").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, body in {
        "binary": binary(args.seed, pub),
        "localization": localization(args.seed, pub),
        "leakage_ood": leakage_ood(args.seed),
        "leakage_id": leakage_id(args.seed),
        "lsg": lsg(args.seed),
    }.items():
        (out / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"tables -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
