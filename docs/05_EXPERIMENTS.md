# Experiments, Evaluation and Paper Artifacts (v2)

`scripts/make_tables.py` reads `results/*.json` and emits LaTeX. Never copy a number by hand.

---

## 1. Metrics

**Binary:** AUROC, AUPRC, Accuracy, F1 (match InteractBind's reporting so your rows slot into
their Table 2).

**Localization:** BRHR@1 / @3 / @5 (primary, their definition exactly), plus Precision@n⁺,
Recall@n⁺, residue AUPRC. Per-interaction-type BRHR@5 across all six types. Atom-side
localization if atom alignment succeeded (new measurement — flag it as such).

**Controls, mandatory:** random ranking, a hydrophobicity-only heuristic, and a
sequence-conservation heuristic. Without these your BRHR numbers mean nothing.

**Transfer:** grounding retention `GR@K = BRHR@K(θ₂)/BRHR@K(θ₁)` on the leakage-filtered
retention set.

**Uncertainty:** ECE (15 bins), Brier, conformal coverage per band, average set size, AURC,
accuracy at 80% / 50% coverage.

**Statistics:** 5 seeds (or 5 folds for ID), mean ± std, paired Wilcoxon, BH correction.

---

## 2. Experiment matrix

| ID | Question | Setup |
|---|---|---|
| **E1** | Does map supervision improve localization? | InteractBind-ID, 5-fold: SAFE-DTI(GIM) vs SAFE-DTI(no grounding) vs FusionDTI vs DrugBAN vs ArkDTA-style |
| **E2** | At what cost to binary accuracy? | λ_rank sweep {0, 0.25, 0.5, 1, 2, 4} → AUROC/BRHR trade-off curve |
| **E3** | Do type-specific heads fix the hard interaction types? | per-type BRHR@5 with/without `L_type`; focus on π–π and cation–π |
| **E4** | Does grounding help OOD? | four protein-similarity bands (25/28/31/33%), both tasks |
| **E5** | **Does grounding survive binary-only transfer?** | stage 2 on BindingDB/BioSNAP/Human; `λ_cons ∈ {0, 0.1, 0.5, 1}`; report GR + AUROC |
| **E6** | Does grounding improve downstream binary prediction? | stage-2 AUROC, grounded-init vs random-init vs binary-pretrained-init, cluster splits |
| **E7** | Does structure help localization more than classification? | SAG on/off; ΔAUROC vs ΔBRHR; also SaProt-tokens vs explicit geometry vs sequence-only |
| **E8** | Does grounding-aware abstention beat probability alone? | risk–coverage, AURC, per-band conformal coverage |
| **E9** | Are confident-but-ungrounded predictions error-enriched? | 2×2 of (high/low p̂) × (high/low margin); error rate per cell |
| **E10** | Bias audit | drug-only and target-only ablations; TAPB-style per-target prior test; does a target-only model score well? |
| **E11** | Ablations | see below |
| **E12** | Sensitivity | τ ∈ {50,70,90}; K negatives ∈ {8,32,128}; T_c; ESM size; L_max |

**E1, E3 and E5 are the paper.** Everything else is support. If time collapses, run those three
properly with 5 seeds rather than all twelve badly.

---

## 3. Ablations (E11)

| Variant | Removes |
|---|---|
| SAFE-DTI (full) | — |
| − L_rank | listwise ranking |
| − L_con | PU contrastive |
| − L_type | six type heads (→ pooled supervision, ≈ ArkDTA's signal) |
| − sink | attention sink column |
| − SAG | sequence only |
| − pLDDT weighting | `r ≡ 1` |
| − modality dropout | `p_drop = 0` |
| − EMA consistency | `λ_cons = 0` (stage 2) |
| dense BCE instead of (a)+(b) | the naive supervision anyone would try first |

That last row matters: showing the naive dense-BCE map loss fails (it will, because of sparsity
and PU noise) is what justifies your loss design.

---

## 4. Tables and figures

**Tables**
- T1 — dataset statistics, structure coverage, map sparsity, binding-residue count distribution.
- T2 — ID: binary + localization, ours vs baselines (same shape as InteractBind's Table 2).
- T3 — per-interaction-type BRHR@5, six types × models.
- T4 — OOD across four similarity bands.
- T5 — transfer: λ_cons × {stage-2 AUROC, GR@1, GR@5}.
- T6 — ablations.
- T7 — calibration and conformal coverage.

**Figures**
- F1 — architecture, with the grounding loss and SAG gate equations inset.
- F2 — **the motivating plot**: AUROC vs BRHR@1 scatter for all published baselines, with
  SAFE-DTI moving up and right. One glance tells the whole story.
- F3 — AUROC/BRHR trade-off curve over λ_rank.
- F4 — per-type radar or heatmap, before vs after type-aware supervision.
- F5 — grounding retention vs λ_cons, with the un-regularised collapse visible.
- F6 — risk–coverage, probability-only vs grounding-aware.
- F7 — case study: one target with a co-crystal; predicted residue map on the 3D structure
  (py3Dmol) beside the atom-level map on the 2D molecule (RDKit similarity maps), with true
  contacts marked. Pick a target in the retention set so it is a genuine held-out example.

---

## 5. Limitations to state

Residue graph is a distance-threshold GCN, not equivariant · AlphaFold models are apo ·
InteractBind negatives are docking-derived, so a docking prior is baked in · binding-residue
annotations come from rule-based NCI detection (PLIP/GetContacts geometric cutoffs), not from
direct measurement · a model may legitimately attend to allosteric or selectivity residues that
score as false positives · proteins truncated at 1024 residues · single-conformation ground
truth ignores binding-site flexibility · BRHR@K is a hit-rate and saturates; report Precision@n⁺
alongside it.

---

## 6. Reproducibility package

`scripts/run_all.sh` end to end · pinned environment export · released: split files, retention
set, atom-alignment permutations, one checkpoint per stage, every `results/*.json` ·
`REPRODUCE.md` with expected wall times and expected numbers ± tolerance.
