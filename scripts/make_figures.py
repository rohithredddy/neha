"""Paper figures from results/*.json (plots only; every value is read from a results file or
configs/published.yaml). Writes PDF + PNG to results/figures/.

  F1  published models: ID AUROC vs BRHR@1, with training-free reference lines
  F2  BRHR@1 of homology kNN vs random, by nearest-train identity stratum, per OOD split
  F3  binary AUROC per split: ligand-only / protein-only / pair models vs published FusionDTI
  F4  BRHR@5 per split: ligand-blind ESM probe and homology transfer vs published FusionDTI

    python scripts/make_figures.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_tables import ID, LC30, OOD, get, load  # noqa: E402

# Categorical slots 1-4 of the dataviz reference palette (validated for adjacent bars, light)
# + neutral reference ink.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF, INK, MUTED, GRID = "#8f8e88", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})  # fmt: skip


def mean_over(prefix: str, path: str, splits: list[str], seed: int) -> float | None:
    v = [get(load(prefix, s, seed), path) for s in splits]
    v = [x for x in v if x is not None]
    return 100 * float(np.mean(v)) if v else None


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def f1(seed: int, pub: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    models = {k: v for k, v in pub["interactbind_table2_id"].items() if k != "source"}
    x = [v["brhr1"] for v in models.values()]
    y = [v["auroc"] for v in models.values()]
    ax.scatter(x, y, s=36, color=REF, edgecolor="white", linewidth=1.5, zorder=3,
               label="published DTI models (InteractBind Table 2)")  # fmt: skip
    for name, xi, yi in zip(models, x, y, strict=True):
        if name in ("FusionDTI", "DrugBAN", "MolTrans"):
            ax.annotate(name, (xi, yi), xytext=(4, -9), textcoords="offset points", color=MUTED)
    eb = pub["explainbind_id"]["ExplainBind"]
    ax.scatter([eb["brhr1"]], [eb["auroc"]], s=36, marker="D", color=REF, edgecolor="white",
               linewidth=1.5, zorder=3, label="ExplainBind (preprint, map-supervised)")  # fmt: skip
    ha = mean_over("ligand_audit", "single_descriptor_auroc/HeavyAtomCount", ID, seed)
    knn = mean_over("homology", "localization/knn/brhr@1", ID, seed)
    if ha is not None:
        ax.axhline(ha, color=C1, lw=2)
        ax.text(ax.get_xlim()[0] + 1, ha + 0.25, f"ligand heavy-atom count, no training: {ha:.1f}",
                color=INK, fontsize=8)  # fmt: skip
    if knn is not None:
        ax.axvline(knn, color=C2, lw=2)
        ax.text(knn - 0.8, 96.0, f"homology transfer,\nno training: {knn:.1f}", color=INK,
                fontsize=8, ha="right")  # fmt: skip
    ax.set_xlabel("BRHR@1 (%)")
    ax.set_ylabel("AUROC (%)")
    ax.set_ylim(94, 100)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("ID split: published models vs training-free references", fontsize=9, loc="left")
    save(fig, out, "F1_auroc_vs_brhr")


def f2(seed: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6), sharey=True)
    for ax, split in zip(axes, OOD, strict=True):
        strata = get(load("homology", split, seed), "localization/by_identity") or {}
        labels, knn, rnd, ns = [], [], [], []
        for b, d in strata.items():
            if not d.get("knn"):
                continue
            lo, hi = (100 * float(x) for x in b.strip("[)").split(","))
            rng_label = f"≥{lo:.1f}" if hi > 100 else f"{lo:.0f}–{hi:.1f}".replace(".0", "")
            labels.append(f"{rng_label}\nn={d['knn']['n_pairs']}")
            knn.append(100 * d["knn"]["brhr@1"])
            rnd.append(100 * d["random_analytic"]["brhr@1"])
            ns.append(d["knn"]["n_pairs"])
        xs = np.arange(len(labels))
        ax.plot(xs, rnd, color=REF, lw=2, ls="--", marker="o", ms=4, label="random ranking")
        ax.plot(xs, knn, color=C1, lw=2, marker="o", ms=5, label="homology kNN")
        ax.set_xticks(xs, labels, fontsize=6.5)
        ax.set_title(split.replace("p_ood_", "protein-OOD ") + "%", fontsize=9, loc="left")
        ax.set_xlabel("nearest-train identity (%)", fontsize=8)
    axes[0].set_ylabel("BRHR@1 (%)")
    axes[0].legend(loc="upper left", fontsize=7)
    save(fig, out, "F2_brhr_by_identity")


def f3(seed: int, pub: dict, out: Path) -> None:
    groups = [
        ("ID", ID),
        ("ID leak-free", LC30),
        *[(s.replace("p_ood_", "OOD-"), [s]) for s in OOD],
    ]
    series = [
        (
            "heavy-atom count (no training)",
            C1,
            "ligand_audit",
            "single_descriptor_auroc/HeavyAtomCount",
        ),
        ("ligand-only GBM", C2, "pair_models", "ligand_only/all/auroc"),
        ("ligand + protein GBM", C3, "pair_models", "pair/all/auroc"),
        ("protein-only GBM", C4, "pair_models", "protein_only/all/auroc"),
    ]
    ood_pub = [get(pub["interactbind_fusiondti_ood"], f"{s}/auroc") for s in OOD]
    fus = [pub["interactbind_table2_id"]["FusionDTI"]["auroc"], None, *ood_pub]
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    w, gap = 0.16, 0.02
    xs = np.arange(len(groups))
    n = len(series) + 1
    for i, (lab, col, prefix, path) in enumerate(series):
        vals = [mean_over(prefix, path, sp, seed) for _, sp in groups]
        pos = xs + (i - (n - 1) / 2) * (w + gap)
        ax.bar(pos, [v or 0 for v in vals], width=w, color=col, label=lab)
        for p, v in zip(pos, vals, strict=True):
            if v is not None:
                ax.text(p, v + 0.6, f"{v:.0f}", ha="center", fontsize=6, color=INK)
    pos = xs + ((n - 1) / 2) * (w + gap)
    ax.bar(pos, [f or 0 for f in fus], width=w, color=REF, label="FusionDTI (published)")
    for p, f in zip(pos, fus, strict=True):
        if f is not None:
            ax.text(p, f + 0.6, f"{f:.0f}", ha="center", fontsize=6, color=INK)
        else:
            ax.text(p, 41, "n/r", ha="center", fontsize=6, color=MUTED)
    ax.axhline(50, color=MUTED, lw=0.8)
    ax.set_ylim(40, 103)
    ax.set_xticks(xs, [g for g, _ in groups])
    ax.set_ylabel("AUROC (%)")
    ax.legend(ncols=3, fontsize=7, loc="upper right", bbox_to_anchor=(1, 1.25))
    save(fig, out, "F3_binary_shortcut")


def f4(seed: int, pub: dict, out: Path) -> None:
    """BRHR@5 per split; ID values are means over folds (5 FASTA-grouped, 4 leak-free)."""
    splits = [
        ("ID", ID),
        ("ID leak-free", LC30),
        *[(s.replace("p_ood_", "OOD-"), [s]) for s in OOD],
    ]
    series = [
        ("random ranking", REF, "esm_probe", "random_analytic/brhr@5"),
        ("ESM-2 probe (no ligand input)", C1, "esm_probe", "probe/brhr@5"),
        ("homology transfer (no training)", C2, "homology", "localization/knn/brhr@5"),
    ]
    fus = pub["interactbind_fusiondti_ood"]
    pub5 = [pub["interactbind_table2_id"]["FusionDTI"]["brhr5"], None,
            *[get(fus, f"{s}/brhr5") for s in OOD]]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    w, gap = 0.19, 0.02
    xs = np.arange(len(splits))
    for i, (lab, col, prefix, path) in enumerate(series):
        vals = [mean_over(prefix, path, sp, seed) for _, sp in splits]
        pos = xs + (i - 1.5) * (w + gap)
        ax.bar(pos, [v or 0 for v in vals], width=w, color=col, label=lab)
        for p_, v in zip(pos, vals, strict=True):
            if v is not None:
                ax.text(p_, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    pos = xs + 1.5 * (w + gap)
    ax.bar(pos, [v or 0 for v in pub5], width=w, color=C3, label="FusionDTI (published)")
    for p_, v in zip(pos, pub5, strict=True):
        label = f"{v:.0f}" if v is not None else "n/r"
        ax.text(p_, (v or 0) + 0.8, label, ha="center", fontsize=6.5, color=INK if v else MUTED)
    ax.set_xticks(xs, [n for n, _ in splits])
    ax.set_ylabel("BRHR@5 (%)")
    ax.set_ylim(0, 90)
    ax.legend(ncols=2, fontsize=7, loc="upper right", bbox_to_anchor=(1, 1.25))
    save(fig, out, "F4_localization_without_ligand")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    pub = yaml.safe_load(Path("configs/published.yaml").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f1(args.seed, pub, out)
    f2(args.seed, out)
    f3(args.seed, pub, out)
    f4(args.seed, pub, out)
    print(f"figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
