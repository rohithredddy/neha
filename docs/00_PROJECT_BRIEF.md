# SAFE-DTI — Project Brief (v2)

**Title (working):** SAFE-DTI: Supervision-transferred, Adaptive and Faithfully Explainable
Drug–Target Interaction Prediction

**Type:** M.Tech thesis + 1 research paper
**Duration:** ~6 months part-time · **Compute:** 1 GPU (12–16 GB)

---

## 1. The finding that defines this project

A 2026 benchmark called **InteractBind** evaluated eight interaction-aware DTI models on two
things at once: can they predict *whether* a drug binds, and can they localize *where* it binds.
The result is stark.

| Model | AUROC (binary) | BRHR@1 (localization) | BRHR@5 |
|---|---|---|---|
| FusionDTI | **98.3** | **21.6** | **35.6** |
| GraphBAN | 97.8 | 16.1 | 30.4 |
| DrugBAN | 97.6 | 15.7 | 31.4 |
| CAT-DTI | 96.8 | 12.2 | 22.1 |
| MolTrans | 95.1 | 10.9 | 22.4 |

BRHR@K = fraction of pairs where at least one of the top-K ranked residues is a true binding
residue. **The best model is right about binding 98% of the time and right about the top binding
residue 22% of the time.** Every "interpretable" DTI paper of the last five years has been
showing attention maps that are, by this measure, mostly wrong.

The benchmark authors diagnose the cause themselves: models are trained only on binary labels,
with nothing guiding their interaction maps. And they now supply the missing ingredient —
~100k pairs with residue–atom contact maps for six non-covalent interaction types.

**SAFE-DTI is the model that uses that supervision, and the study of what happens when you do.**

---

## 2. Why this is a better M.Tech project than the original proposal

The original plan (GNN + protein LM + cross-attention + Integrated Gradients + confidence) was
a reasonable system in which every component was already published. This version:

- has a **quantified gap** to close (22% → ?), not a vague claim of "better interpretability";
- builds on a **ready-made dataset with ready-made splits and ready-made baseline numbers**,
  removing the single biggest risk in a time-boxed thesis (data engineering);
- asks a question nobody has answered yet, because the data only just appeared;
- and degrades gracefully: even a modest improvement is a publishable, well-framed result,
  while the transfer study (C2) is interesting whether the answer is yes or no.

---

## 3. Contributions

**C1 — Grounded Interaction Maps (GIM): interaction-type-aware map supervision.**
Supervise the cross-attention map directly against annotated residue–atom contacts, with a loss
designed for the three things that make this hard: extreme sparsity (a handful of true cells in
an N×L matrix), positive-unlabeled structure (an unannotated cell is not a confirmed non-contact),
and severe type imbalance (van der Waals contacts are everywhere; cation-π is rare and is exactly
where every benchmarked model fails worst, at 16.1% BRHR@5). Three components: a residue-level
listwise ranking loss, an atom–residue contrastive term over sampled negatives, and six
type-specific auxiliary heads with inverse-frequency weighting.

*Differentiation from ArkDTA/MONN, which also used NCI labels:* they regularise attention on
PDBbind for affinity **regression**, train and test on the same annotated corpus, treat all
contacts as one undifferentiated signal, and report qualitative case studies. We work at
100k-pair scale on a public localization benchmark, decompose supervision by interaction type,
and — critically — study what survives transfer to datasets that have no annotations at all.

**C2 — Grounding transfer to label-poor datasets (the core research question).**
In real screening you have binary activity data and no co-crystal structure. So: pretrain with
map supervision on annotated complexes, then fine-tune on BindingDB / BioSNAP / Human using
**binary labels only**, with an EMA-teacher map-consistency regulariser to resist "grounding
collapse". Then measure *grounding retention* — the localization ability of the transferred
model, evaluated on held-out annotated pairs it never saw in stage 2. Two sub-questions:
(a) does localization survive binary-only fine-tuning? (b) does being grounded make the binary
predictions themselves more robust under protein-similarity-controlled distribution shift?

**C3 — Structure-Adaptive Gate (SAG).**
An optional structure branch over an AlphaFold/PDB residue graph, fused through a residual gate
weighted by per-residue pLDDT, with an exact sequence-only fallback. Reframed hypothesis, which
is what makes it worth testing: **structure should improve localization far more than it improves
binary accuracy.** If true, that is a clean result that explains why the field's structure-free
models score 98% AUROC while failing at localization.

**C4 — Localization-aware selective prediction.**
A prediction that is confident but ungrounded is the dangerous one. We define a *grounding
margin* from the interaction map, combine it with predicted probability into a nonconformity
score, and run Mondrian conformal prediction over the benchmark's protein-similarity bands.
Claim to test: grounding-aware abstention gives a better risk–coverage curve than probability
alone, and high-confidence/low-grounding pairs are enriched in errors.

> C1 + C3 are the model. C2 is the research question. C4 is the practical payoff.
> If C1 gives only a small gain, C2 and C4 still carry a paper.

---

## 4. Explicitly out of scope

Affinity regression · docking and pose prediction · SE(3)-equivariant networks · heterogeneous
knowledge graphs · fine-tuning the protein language model (frozen, precomputed) · building a new
dataset (we consume InteractBind) · wet-lab validation · LLM agents or generated natural-language
explanations · multi-site / allosteric modelling.

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| InteractBind download/format issues | medium | verify on day 1; fall back to building a smaller set from PDBbind + PLIP, as ArkDTA did |
| Grounding loss hurts AUROC | medium | tune λ; report the trade-off curve — it is itself a result |
| Grounding does not survive transfer (C2 negative) | medium | a well-measured negative result with the consistency-regulariser ablation is publishable; frame the paper as a study, not a product claim |
| FusionDTI baseline hard to reproduce | low | code is public; their Table 2 numbers are citable for ID; re-run only what you must |
| Structure coverage too low for SAG | medium | AlphaFold DB covers most UniProt entries; report coverage honestly and scope SAG to the covered subset |
| Scope creep across four contributions | **high** | C1 and C2 are mandatory. C3 and C4 are cut in that order if week 18 arrives without results. |

---

## 6. Success criteria

**Must hit (thesis):**
- Reproduce a published baseline's ID AUROC within ±0.01 and its BRHR@5 within ±3 points.
- GIM-trained SAFE-DTI improves BRHR@1 by ≥ 5 absolute points over the best reported baseline
  with AUROC loss ≤ 0.01.
- Grounding retention after binary-only transfer measured and reported with its ablation.

**Paper-worthy:**
- BRHR@1 ≥ 35% (vs 21.6% current best), i.e. a step change, not a nudge.
- Statistically significant per-interaction-type gains on π–π and cation–π.
- A clear answer to C2, either direction, with the consistency ablation isolating the cause.
- C4 risk–coverage improvement over probability-only abstention.

**Venues:** *Briefings in Bioinformatics*, *Bioinformatics*, *Journal of Cheminformatics*,
*Nature Communications* (ambitious), or ML workshops (NeurIPS AI4Science, ICLR MLDD) for speed.
