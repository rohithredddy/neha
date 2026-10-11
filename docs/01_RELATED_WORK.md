# Related Work and Positioning

Everything below was verified against a real source. URLs given so you can fetch and cite
properly rather than retyping.

---

## 1. The benchmark that defines the problem

**InteractBind** — *A Large-Scale Dataset and Benchmark: Do Protein-Ligand Models Learn Binding
Sites or Just Binding Likelihood?* (Meng et al., Glasgow / Harvard Medical School, 2026)
arXiv: https://arxiv.org/abs/2605.24045 · data: https://huggingface.co/datasets/Zhaohan-Meng/InteractBind

- ~99,391 pairs, 11,473 proteins, 9,017 ligands, curated from PDB complexes.
- Per pair: protein FASTA **and** SaProt structure-aware sequence, ligand SMILES/SELFIES,
  binary label, docking affinity, and **six non-covalent interaction maps** (hydrogen bond,
  salt bridge, van der Waals, hydrophobic, π–π stacking, cation–π) plus an aggregated map.
  Contacts detected with PLIP and GetContacts, with distance-decay strength in [1e-6, 1].
- Splits: affinity-aware ID subset (84,825 pairs, 5-fold CV; positives ≤ −7 kcal/mol, negatives
  > −5, ambiguous middle discarded) and **four protein-similarity-controlled OOD splits** at
  25 / 28 / 31 / 33% max train–test identity (Needleman–Wunsch), plus ligand-similarity splits.
- Metric: **BRHR@K** — aggregate the model's interaction map along the ligand axis by max,
  rank residues, score 1 if any of the top-K is a true binding residue.
- Headline results: best model FusionDTI = 98.3 AUROC / 21.6 BRHR@1 / 35.6 BRHR@5. Per-type,
  van der Waals is easiest (33.2 BRHR@5) and cation–π hardest (16.1).
- Under OOD, FusionDTI's AUROC falls from 91.8 (33% identity) to 81.8 (25%); BRHR@5 moves only
  28.4 → 30.1, i.e. localization is uniformly poor regardless of similarity.
- Ligand-side shift matters much less than protein-side shift.
- Stated limitation: no structure-level benchmark yet, because pose ambiguity makes ground truth
  hard. **This is the opening for our C3.**

**Use this verbatim as the paper's motivation.** Their Table 2 also gives you eight baseline
numbers for free, so you only need to re-run the ones you modify.

---

## 2. Prior attempts at supervising attention with contacts

| Work | Task | Supervision | Scale | Limits relevant to us |
|---|---|---|---|---|
| **MONN** (2020) | affinity + pairwise interaction | PDBbind non-covalent contacts | small | joint prediction, no type decomposition, no transfer study |
| **ArkDTA** (Bioinformatics 2023, ISMB; https://github.com/dmis-lab/ArkDTA) | affinity regression | residue-wise NCI labels on PDBbind, attention-regularisation loss with a pseudo-embedding sink for no-NCI residues | small | one corpus, all contacts pooled into one signal, qualitative evaluation, no OOD, no uncertainty |
| **AttentionSiteDTI / FragXsiteDTI** | DTI | pocket inputs (structure required) | moderate | require structure at inference; pocket-level, not residue-level |

So *attention supervision from contacts is not itself novel* — say so plainly in Related Work.
Our differences, which are the ones to defend:
1. **Scale and evaluation**: 100k pairs, a public localization metric, protein-similarity OOD.
2. **Type decomposition**: six separate supervision signals with imbalance correction, targeting
   the interaction types the benchmark shows are worst.
3. **Atom-level maps**: we keep a molecular graph, so our map axis is *atoms*, matching the
   ground truth exactly. SELFIES/substructure-token models cannot be scored atom-side fairly —
   the benchmark projects everything onto the residue axis for this reason. We can report
   atom-side localization as a new measurement.
4. **Transfer**: nobody has asked whether grounding survives fine-tuning on unannotated data.

The ArkDTA "attention sink" trick (a learned pseudo-embedding absorbing attention from residues
with no NCI) is worth adopting and citing — it is a clean solution to the sparsity problem.

---

## 3. Baselines

| Model | Why include |
|---|---|
| **FusionDTI** (EMNLP Findings 2025; https://github.com/ZhaohanM/FusionDTI) | current best on both tasks; SaProt SA-vocabulary protein tokens + SELFIES ligand tokens + token-level fusion, frozen PLM encoders. **Primary baseline.** |
| **DrugBAN** (Nat. Mach. Intell. 2023; https://github.com/peizhenbai/DrugBAN) | standard interpretable baseline; bilinear attention + CDAN domain adaptation; ships BindingDB/BioSNAP/Human with cluster splits we need for C2 |
| **GraphBAN** (Nat. Commun. 2025) | inductive graph variant of BAN, strong localization |
| **ArkDTA** | the attention-supervision ablation: "supervision, but not ours" |
| **SAFE-DTI (no grounding)** | our own architecture trained with binary labels only — isolates C1 |
| concat-MLP(ESM-mean, ECFP4) | trivial sanity baseline; reviewers ask for it |

---

## 4. Representation choices, with justification

- **ESM-2** residue embeddings are the standard PLM choice for DTI (DTIAM uses 650M; reviews
  confirm PLM embeddings improve both DTI and target classification:
  https://www.sciencedirect.com/science/article/pii/S2001037025005719). Frozen, precomputed.
- **SaProt** structure-aware vocabulary (used by FusionDTI, Foldseek 3Di tokens) is an
  alternative way to inject structure. Worth one comparison row: SAG (explicit geometry) vs
  SaProt (tokenised structure) vs sequence-only.
- **AlphaFold DB** gives broad structural coverage but the models are apo and may miss
  binding-competent conformations, and AlphaFold confidence pLDDT < 50 typically indicates
  disorder while > 70 is the conventional confidence threshold
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC9226352/). This grounds the SAG gate at τ = 70.
- **Never use ESMFold structures.** ESMFold is built on ESM-2, so the "structure" modality would
  not be independent of the sequence modality. Say this explicitly in the paper; SpliceBind
  (https://arxiv.org/pdf/2606.04020) raises exactly this coupling concern.

---

## 5. Dataset bias — the critique you must pre-empt

Reviewers in this area will ask whether you are measuring real learning or dataset artefacts.

- **Hidden ligand bias**: TransformerCPI showed DUD-E-trained models learn ligand patterns
  instead of interactions; DrugBAN removed drugs appearing in only one class for this reason.
- **Target prior bias**: TAPB (Nat. Commun. 2025, https://www.nature.com/articles/s41467-025-66915-1)
  argues that on BioSNAP and BindingDB *target* prior bias — not drug bias — is the dominant
  confounder, letting models memorise per-target label tendencies.
- **Train–test leakage** inflates binding-affinity benchmarks severely; CleanSplit retraining
  caused large performance drops (Nat. Mach. Intell. 2025,
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12552120/).
- InteractBind's negatives come from **docking** rather than random pairing, which is better but
  introduces a docking-score prior of its own. Acknowledge it.

**Your defence is structural, and it is one of the quiet strengths of this project:** a model
exploiting target prior bias cannot localize binding residues. Localization metrics are far
harder to game than AUROC. Make this argument explicitly — it turns a benchmark critique into
an argument for your evaluation.

Practical requirements: report per-target label balance; run the TAPB-style null test; include a
drug-only and a target-only ablation (if a target-only model scores well, the split is leaky).

---

## 6. Uncertainty

Cluster-conditioned conformal prediction for DTI exists (https://arxiv.org/pdf/2505.18890) but
targets regression and conditions on drug/protein clusters. Mondrian CP over the benchmark's
**similarity bands** is a natural, defensible adaptation for binary classification. The novel
element in C4 is not the conformal machinery — it is using a *map-derived* grounding signal in
the nonconformity score.

---

## 7. Positioning paragraph for the abstract

> Drug–target interaction models now exceed 98% AUROC while identifying the correct top binding
> residue in only about one fifth of cases, because they are trained on binary labels that say
> nothing about where binding occurs. Using newly released residue–atom contact annotations, we
> train interaction maps directly with an interaction-type-aware grounding objective, and then
> ask the question that matters for deployment: does this grounding survive fine-tuning on the
> large binary-only datasets that real screening relies on? We find [result], and show that
> grounding-aware abstention identifies the confident-but-unlocalized predictions that are most
> likely to be wrong.

---

## 8. BibTeX to collect

InteractBind · FusionDTI · DrugBAN · GraphBAN · ArkDTA · MONN · MolTrans · TransformerCPI ·
HyperAttentionDTI · PerceiverCPI · CAT-DTI · GraphDTA · DeepConv-DTI · ESM-2 (Lin 2023) ·
SaProt · Foldseek · AlphaFold2 · AlphaFold DB · PLIP 2025 · GetContacts · AutoDock Vina ·
SELFIES · ECFP · Integrated Gradients · ERASER faithfulness · Vovk conformal prediction ·
Mondrian CP · MC-dropout · TAPB · CleanSplit · cluster-conditioned CP for DTI · AI-Bind ·
RDKit · PyTorch Geometric. Save to `docs/refs.bib`, fetched from DOIs.
