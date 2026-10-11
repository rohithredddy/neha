# Research Update — October 2026 (literature re-check + novelty re-positioning)

Date of check: 2026-10-11. Every claim below links to the source it was read from. Items marked
**[unverified]** could not be confirmed from a primary source and must be checked before use.

---

## 0. TL;DR — two findings change the plan

1. **C1 is no longer novel as stated.** The InteractBind authors themselves released
   **ExplainBind** (bioRxiv, March 2026), which already aligns cross-attention with the
   InteractBind interaction maps using **six per-type supervised heads** + a KL alignment loss,
   and reports **BRHR@1 55.6% / BRHR@5 74.6%**. Our brief's "paper-worthy" target (BRHR@1 ≥ 35%)
   is already exceeded by prior work. *"First to supervise maps on InteractBind" and "type
   decomposition" cannot be claimed.*
2. **The public InteractBind release is residue-level only.** The Hugging Face CSVs store, per
   interaction type, a **binary list aligned to the FASTA residues** — not the m×n residue×atom
   matrices described in the paper. Atom-level supervision (method spec §4b contrastive term),
   atom-side localization, and the PDB→RDKit atom-alignment task (M1) are **not possible from the
   public data** unless the authors release the full maps.

What is still open (checked, nothing found): **grounding retention after binary-only transfer
(C2)** and **grounding-aware conformal / selective prediction (C4)**. These become the paper.

---

## 1. Verified facts about InteractBind

Source: arXiv 2605.24045 v1, submitted 21 May 2026, *under review, NeurIPS 2026 Datasets &
Benchmarks (Evaluations and Datasets) track* — https://arxiv.org/abs/2605.24045 ·
full text https://arxiv.org/html/2605.24045v1

The brief's numbers were checked against the paper's Table 2 and **match exactly**
(ID, 5-fold CV, mean ± SD, %):

| Model | AUROC | BRHR@1 | BRHR@3 | BRHR@5 |
|---|---|---|---|---|
| MolTrans | 95.1 ± 2.2 | 10.9 ± 0.5 | 17.7 ± 0.2 | 22.4 ± 0.8 |
| TransformerCPI | 95.4 ± 0.3 | 10.9 ± 0.2 | 16.7 ± 0.7 | 21.7 ± 0.4 |
| HyperAttentionDTI | 95.5 ± 0.6 | 12.3 ± 0.4 | 17.7 ± 0.6 | 20.8 ± 0.3 |
| PerceiverCPI | 96.9 ± 0.2 | 10.6 ± 0.1 | 13.9 ± 0.5 | 19.9 ± 0.9 |
| CAT-DTI | 96.8 ± 0.4 | 12.2 ± 0.7 | 14.8 ± 0.2 | 22.1 ± 0.6 |
| DrugBAN | 97.6 ± 1.2 | 15.7 ± 0.6 | 23.9 ± 0.1 | 31.4 ± 0.4 |
| GraphBAN | 97.8 ± 0.2 | 16.1 ± 0.5 | 22.9 ± 0.2 | 30.4 ± 0.6 |
| FusionDTI | 98.3 ± 0.2 | 21.6 ± 0.1 | 29.6 ± 0.3 | 35.6 ± 0.2 |

- BRHR@K definition confirmed: residue score = **max over ligand units**, rank, hit if top-K
  overlaps true binding residues, averaged over pairs. (Matches spec §3.)
- Dataset sizes confirmed: 99,391 pairs / 11,473 proteins / 9,017 ligands; ID 84,825; four
  protein-OOD splits of 13,126 pairs each (8:1:1).
- Per-type FusionDTI BRHR@5 confirmed: vdW 33.2, H-bond 28.9, hydrophobic 28.2, salt 27.9,
  π–π 18.2, cation–π 16.1.
- OOD: FusionDTI AUROC 81.8 → 91.8 and BRHR@5 28.4 → 30.1 from 25% → 33% similarity. Confirmed.
- Labels are **docking-derived** (positives ≤ −7, negatives > −5 kcal/mol, Vina global docking
  of random ligands). Labels are not experimental activity. This is a reviewer target.
- Note: in this benchmark paper the maps are used **only for evaluation**; the paper says
  supervision is the missing ingredient. ExplainBind (below) is where they use it for training.

**Public data format** — https://huggingface.co/datasets/Zhaohan-Meng/InteractBind (CC-BY-4.0)
and https://github.com/ZhaohanM/InteractBind (README + assets only, no training code):
- Configs: `affinity` (~99.4k rows), `p_ood_25/28/31/33` (train/validation/test).
- Columns: `pubchem_id, uniprot_id, fasta, saprot, selfies, smiles, att_site, label, affinity`,
  plus seven `*_binding_site` columns (six NCI types + `Overall`).
- Each `*_binding_site` cell is a **string-encoded binary list, one entry per FASTA residue**.
- No atom axis and no continuous strengths in the public release. `att_site` is undocumented; inspect it in M1.
- The ID 5-fold split assignment is not obviously exposed as a config. **[unverified]**: check in M1.

---

## 2. New prior work since the docs were written

### 2.1 ExplainBind — direct overlap with C1 (must be cited and compared against)
*Explainable Physicochemical Determinants of Protein–Ligand Binding via Non-Covalent
Interactions.* Meng, Bai, Yuan, …, Loscalzo (Glasgow / Broad / Harvard). bioRxiv, March 2026,
**not peer-reviewed**. https://www.biorxiv.org/content/10.64898/2026.03.03.707476v1.full ·
Broad page: https://www.broadinstitute.org/publications/broad1375566

What they did:
- Frozen ESM-2 (protein, residue tokens) + SELFormer (ligand, **SELFIES tokens**, not atoms);
  optional SaProt structure-aware tokens.
- 8 cross-attention heads: **6 supervised per NCI type**, 2 free. Hidden 512.
- Loss: `L_cls + λ·KL-alignment` (λ = 0.3). Predicted attention and ground-truth maps are
  each normalised into a distribution and matched with a weighted KL.
- InteractBind ID + 4 protein-OOD + 4 ligand-OOD splits.
- **BRHR@1 55.6%, BRHR@5 74.6%** (strict); ±1-residue relaxed window adds ~6–9 points.
  ID AUROC 0.993.
- Also trained on BindingDB / BioSNAP / Human, but **with binary labels only, from scratch**.
  On those datasets they report AUROC only and no localization.
- Wet-lab: retrospective ACE ranking; single-concentration L2HGDH screen.

What they did **not** do (from the visible text; the supplement was partly truncated):
- No pretrain-on-maps → fine-tune-on-binary transfer, and no measure of whether grounding survives.
- No uncertainty, calibration metric, conformal or selective prediction ("calibrated" is claimed
  with no ECE or similar).
- No handling of sparsity or positive-unlabeled structure is described. KL over the full
  normalised map treats every unannotated cell as a true zero.
- No ligand-agnostic control: they never test whether a model that ignores the ligand can reach
  the same BRHR.
- No code released.

### 2.2 Other relevant 2025–2026 work
| Work | Relevance | Source |
|---|---|---|
| **ProMoNet** (J. Cheminform., May 2026) | Sequence-based binding-site **pretraining → fine-tuning** for affinity/kinetics. Closest to C2's *mechanics*, but never measures whether site prediction survives fine-tuning. Cite as the pretrain→transfer precedent. | https://link.springer.com/article/10.1186/s13321-026-01227-0 |
| *Where Black-box DTI Models Look* (arXiv 2606.14245, Jun 2026) | Audit of DTI attributions. Finds padding/special-token artefacts; says explanations need structural ground truth. Supports our motivation. | https://arxiv.org/abs/2606.14245 |
| Hierarchical sequential cross-attention DTI (arXiv 2609.34921, Sep 2026) | Argues attention maps should be validated against binding residues before interpretability claims. Supports motivation. | https://arxiv.org/abs/2609.34921 |
| Conformal prediction for DTI (arXiv 2505.18890) | Cluster-conditioned CP for **regression** (KIBA). Still the closest CP-for-DTI work, with no map-derived score. | https://arxiv.org/abs/2505.18890 |
| Conformal Selective Prediction with General Risk Control (arXiv 2603.24704) | General selective-CP machinery that C4 can build on. | https://arxiv.org/abs/2603.24704 |
| DVNDTA (BMC Bioinf., Jun 2026) | Affinity model motivated by NCI learning. Minor, cite in related work only. | https://pmc.ncbi.nlm.nih.gov/articles/PMC13520400/ |
| ArkDTA (Bioinformatics 2023) | Still the attention-sink / NCI-regularisation precedent. | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10311339/ |

Searches for *grounding retention / interpretability after fine-tuning in DTI* and for
*conformal prediction using interaction-map signals* returned nothing matching (searched
2026-10-11). Absence of evidence is not proof, so re-search before submission.

---

## 3. Re-positioned novelty (recommendation)

Stop pitching "a model that supervises maps"; ExplainBind owns that. Pitch **a study of what map
supervision actually buys you**, with three contributions, each checked as open:

**N1 (core) — Grounding retention under binary-only transfer** (old C2, unchanged in substance).
Pretrain with residue-level supervision on InteractBind, then fine-tune on BindingDB / BioSNAP /
Human with binary labels only. Measure GR@K on the leakage-filtered retention set. Compare against
an EMA-consistency regulariser. ExplainBind trained on these datasets *without* maps and never
measured localization there, so this question is open, and it is the one that matters for real
screening. A negative result is still publishable.

**N2 (new, cheap, high-impact) — Is BRHR measuring interaction or just pocket detection?**
Most proteins in InteractBind have one dominant pocket. A **ligand-agnostic** residue scorer (for
example a linear probe on ESM-2 residue embeddings predicting `Overall_binding_site`, with no
ligand input) may already reach a high BRHR@K. Report:
- the ligand-agnostic BRHR ceiling;
- a **ligand-sensitivity score**: how much the predicted residue ranking changes for different
  ligands binding the same protein, compared with how much the ground truth changes.
If supervised maps barely beat the ligand-agnostic probe, that reframes the whole benchmark,
including ExplainBind's 55.6%. This is the kind of result that gets a mid-tier paper accepted
and cited. It needs about a week of compute.

**N3 — Grounding-aware selective prediction** (old C4). A map-derived grounding margin and entropy
in a Mondrian-conformal nonconformity score over similarity bands. Claim: better risk–coverage
than probability-only, and confident-but-ungrounded pairs are error-enriched. Open (§2.2).

**Supporting method (not a headline claim) — PU/sparsity-aware residue loss.** Listwise ranking
+ type heads with inverse-frequency weights + attention sink, compared head-to-head against an
**ExplainBind-style KL alignment** baseline that we re-implement (no code is public). Defensible
angle: KL over the full map treats unlabeled residues as hard zeros; ranking does not.

**Deprioritise:** C3 (SAG). ExplainBind already shows SaProt structure tokens help, and the
public data has no structure-level benchmark. Keep it only if N1–N3 are finished by week 18.

---

## 4. Required changes to the existing docs (proposed, not yet applied to the spec)

| Doc | Change |
|---|---|
| `00_PROJECT_BRIEF.md` | Replace C1 novelty claim. New success bar: beat an ExplainBind-style KL baseline that **we** run (their 55.6% is unreviewed and unreproducible without code). Drop "BRHR@1 ≥ 35%" as a paper-worthy criterion. Add N2. |
| `01_RELATED_WORK.md` | Add ExplainBind as the closest prior work, plus ProMoNet and 2606.14245. Remove "type decomposition" and "atom-level maps" from our differentiators. |
| `02_METHOD_SPEC.md` | §4b contrastive term and atom-side metrics depend on atom-level maps that are **not public**. Either (a) email the authors for the full m×n maps, or (b) switch to residue-level losses only (4a, 4c, 4d). Log the decision in `DECISIONS.md`. Add an ExplainBind-KL baseline loss and the ligand-agnostic probe. |
| `03_DATA_PIPELINE.md` | Map format is residue-level binary strings in CSV, not sparse residue×atom COO. The PDB→RDKit alignment task in §1 step 2 is moot unless full maps are obtained. |
| `04_IMPLEMENTATION_PLAN.md` | M1 becomes: parse the string lists, verify `len == len(fasta)`, inspect `att_site`, locate the ID fold assignments. Add the ligand-agnostic probe as an early milestone (it is also a sanity baseline). |
| `05_EXPERIMENTS.md` | Promote E5 (transfer) and E8/E9 (selective prediction) to headline. Add E13: ligand-agnostic ceiling + ligand-sensitivity. |

**Decision (2026-10-11): no contact with the authors; use only public data.** Consequences:
- Residue-level supervision only: losses 4a (rank), 4c (type), 4d (sink). 4b dropped.
- No atom-side metric; no PDB→RDKit alignment task.
- If the ID fold assignment is not in the public files, build our own seeded protein-grouped
  5-fold split and re-run all baselines on it. Lead comparisons with the **public OOD splits**,
  which are directly comparable.
- ExplainBind has no code, so its 55.6% is cited as reported. Our comparison is against our own
  re-implementation of its KL-alignment loss under identical encoders.
This narrows the method but leaves N1–N3 untouched. None of them needs atom-level maps.

---

## 5. Target venues (mid-tier)

Dates marked [typ.] are the usual annual window, **not yet confirmed for 2027**. Check each CFP.

| Venue | Tier / fit | Deadline | Fit for this paper |
|---|---|---|---|
| **IEEE BIBM 2027** | CCF-B, main mid-tier bioinformatics conf. | [typ.] Aug 2027 | **Primary target.** Full study (N1+N2+N3). |
| **ACM BCB 2027** | Mid-tier, ACM proceedings | [typ.] Apr 2027 | Good fit if N1+N2 are done by spring. |
| **ICLR 2027 workshops** (GEM / MLDD-style) | Workshop, non-archival, fast | [typ.] Jan–Feb 2027 | N2 alone (ligand-agnostic finding) as an early, citable preprint. |
| PAKDD 2027 / ECML-PKDD 2027 | CCF-C / B, ML venues with applied tracks | [typ.] Nov 2026 / Mar 2027 | PAKDD is too soon. ECML applied-DS track is a possibility. |
| ISBRA 2027 · ICONIP 2027 | Mid/lower-tier, LNCS proceedings | [typ.] mid-2027 | Safe fallback. |
| RECOMB 2027 | Top-tier | 28 Oct 2026 (confirmed: https://recomb.org/recomb2027/keydates_and_deadlines) | Too soon; not realistic. |
| Journal fallback: BMC Bioinformatics, J. Cheminformatics | Q1/Q2 journals | rolling | J. Cheminform. published ProMoNet, so the topic fits. |

Suggested path: workshop paper on N2 in Feb 2027 → full paper (N1+N2+N3) to ACM BCB (Apr) or
IEEE BIBM (Aug) 2027 → thesis.

Suggested working title: *Does Grounding Survive? Binding-Site Supervision, Transfer, and
Grounding-Aware Abstention in Drug–Target Interaction Prediction.*

---

## 6. Where every number in this file came from
- InteractBind paper, Table 2 and text: https://arxiv.org/html/2605.24045v1
- InteractBind data card/format: https://huggingface.co/datasets/Zhaohan-Meng/InteractBind
- InteractBind repo: https://github.com/ZhaohanM/InteractBind
- ExplainBind preprint: https://www.biorxiv.org/content/10.64898/2026.03.03.707476v1.full
- ProMoNet: https://link.springer.com/article/10.1186/s13321-026-01227-0
- DTI explainability audit: https://arxiv.org/abs/2606.14245
- CP for DTI: https://arxiv.org/abs/2505.18890
- RECOMB 2027 dates: https://recomb.org/recomb2027/keydates_and_deadlines
- Secondary only (not used for numbers): https://blog.pebblous.ai/blog/protein-ligand-ai-benchmark-illusion/en/
