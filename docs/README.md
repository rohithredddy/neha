# SAFE-DTI — documentation set (v2)

Supervision-transferred, Adaptive and Faithfully Explainable Drug–Target Interaction prediction.
M.Tech thesis → 1 research paper.

| File | Goes in | Read when |
|---|---|---|
| `CLAUDE.md` | **repo root** | auto-loaded by Claude Code each session |
| `00_PROJECT_BRIEF.md` | `docs/` | first — novelty, contributions, risks, scope |
| `01_RELATED_WORK.md` | `docs/` | writing Section 2; checking prior art |
| `02_METHOD_SPEC.md` | `docs/` | implementing the model (authoritative) |
| `03_DATA_PIPELINE.md` | `docs/` | building the data |
| `04_IMPLEMENTATION_PLAN.md` | `docs/` | start of each milestone |
| `05_EXPERIMENTS.md` | `docs/` | running experiments, tables, figures |

Also create empty `docs/DECISIONS.md` and `docs/refs.bib`.
The Claude Code usage guide is a separate file outside this bundle.

---

## The gap, in one table

A 2026 benchmark (InteractBind) scored eight DTI models on binding prediction *and* binding-site
localization:

| | best model | typical |
|---|---|---|
| Binary AUROC | 98.3 | 95–98 |
| BRHR@1 (top residue correct) | 21.6 | 11–16 |

Models are right about *whether* a drug binds almost always, and right about *where* it binds
about a fifth of the time. The benchmark authors attribute this to training on binary labels
with nothing supervising the interaction map — and they released ~100k pairs of residue–atom
contact annotations across six non-covalent interaction types.

## What SAFE-DTI does about it

- **C1 — Grounded Interaction Maps.** Supervise the cross-attention map with a loss built for
  extreme sparsity, positive-unlabeled noise, and interaction-type imbalance: listwise residue
  ranking + sampled atom–residue contrastive + six type-specific heads with inverse-frequency
  weighting + an attention sink.
- **C2 — Grounding transfer.** Pretrain grounded, then fine-tune on BindingDB/BioSNAP/Human with
  binary labels only and an EMA-teacher consistency regulariser. Measure *grounding retention*
  on a leakage-filtered held-out set. **This is the research question.**
- **C3 — Structure-Adaptive Gate.** Optional AlphaFold branch gated by per-residue pLDDT, with
  exact sequence-only fallback. Hypothesis: structure helps localization far more than accuracy.
- **C4 — Localization-aware selective prediction.** A grounding margin from the map feeds a
  conformal nonconformity score; confident-but-ungrounded predictions are the dangerous ones.

## Why this is a sound thesis bet

The dataset, splits, metric and eight baseline numbers already exist, so the usual
data-engineering risk is gone. The gap is quantified, so success is measurable. And C2 is
interesting whether the answer is yes or no — a careful negative result on grounding transfer is
still a paper.

## Start here

1. Read `00_PROJECT_BRIEF.md`.
2. Set up the repo per the Claude Code guide.
3. Do **M1 (InteractBind verification)** before anything else — especially the PDB→RDKit
   atom-order alignment. Everything downstream depends on it.
