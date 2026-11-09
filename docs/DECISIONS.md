# Decisions (append-only)

- **2026-10-11 — Public data only; residue-level supervision.** No contact with the InteractBind
  authors. The public release has per-residue binary binding-site labels only, so the atom–residue
  contrastive loss (spec §4b), atom-side metrics and PDB→RDKit alignment are dropped. If the ID
  fold assignment is not public, a seeded protein-grouped 5-fold split is created and all
  baselines are re-run on it. See `06_RESEARCH_UPDATE_2026-10.md`.
- **2026-10-11 — Novelty re-positioned.** ExplainBind (bioRxiv, Mar 2026) already supervises
  per-type attention on InteractBind. Headline contributions are now N1 grounding retention under
  binary-only transfer, N2 ligand-agnostic ceiling / ligand sensitivity, and N3 grounding-aware
  selective prediction. C3 (SAG) is deprioritised.
- **2026-10-11 — ID split.** The ID 5-fold assignment is not public. Use our own 5-fold split,
  grouped by **exact protein sequence**. (6,308 duplicated pairs make pair-level CV leak.) Seed 0,
  saved to disk, and all baselines re-run on it. *Amended same day:* UniProt grouping was tried
  first and leaked: 27% of train rows had a sequence identical to a test-fold sequence, because
  one FASTA carries up to 222 UniProt IDs. UniProt∪FASTA linking yields one component holding
  92.6% of rows. Exact-FASTA grouping is stricter than "in-distribution" requires;
  similarity-controlled generalisation is measured on the public OOD splits. Label 2 (ambiguous band) excluded.
- **2026-10-11 — No label-based cropping at eval.** Spec §7 crop-around-annotations is train-only.
  Eval uses full-length sequences with windowed ESM. Reason: 42% of p_ood_25 test pairs have
  L > 1024; label-centred crops would leak the answer.
- **2026-10-11 — Grounding loss on positives only (spec default) for now.** In ID, 99.96% of
  negatives also carry true sites (crystal complexes with weak docked affinity). Supervising them
  too is a later one-flag ablation, not the default.
- **2026-10-11 — BRHR ties** are broken by residue index (stable sort). Degenerate baselines must
  use random scores, never constants.
- **2026-10-11 — Compute.** The dev machine has no CUDA GPU (i5-1235U, 16 GB RAM, ~16 GB free
  disk). Develop and unit-test on CPU subsets; ESM precompute and full training run on a cloud GPU.
- **2026-10-11 — Storage on D:.** `data/` is a directory junction to `D:\safedti\data`
  (238 GB free; C: has ~11 GB). Torch hub (ESM weights) → `D:/safedti/torch_cache`, HF cache →
  `D:/safedti/hf_home`, both via Makefile env vars. Code paths stay `data/...`. This makes the
  ESM-2 150M fp16 cache (~24 GB, measured estimate) feasible locally, so the CPU-first plan runs
  ESM precompute locally (~9 h measured estimate).
- **2026-10-16 — Training-free lookup baselines (N2 audit).** `scripts/run_lookup_baselines.py`.
  Residue lookup uses **exact-FASTA matches only**, so residue j is the same residue in train and
  test with no alignment (rule 4). Homology transfer via alignment is a separate later baseline.
  Localization is evaluated on annotated positives (label 1, ≥ 1 site), the same set as the M1
  random baseline. The site table uses every annotated train row (labels 0 and 1), because ID
  negatives carry real sites and a model sees them too. Binary priors are smoothed with 1
  pseudo-count toward the global train rate. Ligand heavy-atom count (RDKit) is a no-data scorer.
- **2026-10-17 — Scope: benchmark audit first.** User asked to complete the project; training
  (N1, N3) needs the ESM cache and a GPU. The CPU-feasible audit study (binary shortcut,
  homology-transfer localization, split leakage, ligand-agnostic ceiling, proposed fixes) is done
  first. It also provides the mandatory controls of 05_EXPERIMENTS §1.
- **2026-10-17 — MMseqs2 18-8cc5c** (win64 zip sha256 `9752087c…cfcbf`, at D:/safedti/tools).
  Homology search: `easy-search -s 7.5 --max-seqs 300 --max-accept 10 -e 10`. Identity = MMseqs2
  `fident` of the local alignment, so it can exceed a global-identity split threshold. That is why
  a coverage-filtered view (qcov ≥ 0.8) is also reported. `--max-accept 10` caps memory on ID
  (45k sequences); the top-5 kNN draws from those 10.
- **2026-10-17 — Homology-transfer variants.** nn1 = best hit by bit score, weight 1. knn =
  top-5 annotated hits, weight = fident. knn_lig = knn with each target's per-row sites weighted
  by Morgan(r=2, 2048) Tanimoto + 0.01. Tie noise is 1e-6 and asserted below the smallest positive
  score. The alignment→index map is tested by hand, and every hit is checked against the real
  sequences at run time.
- **2026-10-17 — Ligand-agnostic oracle** is a ceiling, not a baseline: it uses TEST labels of
  other ligands of the same FASTA (leave-one-out). Reported separately and always labelled.
- **2026-10-17 — Ligand-only model** = sklearn HistGradientBoosting on all RDKit 2D descriptors
  (fixed hyperparameters in configs/baselines/ligand_audit.yaml, no tuning). This is a trained
  model, but it never sees the protein.
- **2026-10-17 — Size-matched evaluation subset** = within each exact heavy-atom count, equally
  many positives and negatives (seeded). Proposed fix for the binary task. Row ids are saved to
  data/processed/audit/size_matched/.
- **2026-10-17 — Proposed cluster split `id_c30`**: MMseqs2 `easy-cluster --min-seq-id 0.3 -c 0.8
  --cov-mode 0`, then grouped_kfold on clusters. Residual leakage is measured, not assumed, by
  rerunning the homology audit on it.
- **2026-10-17 — MMseqs2 clustering splits leak; replaced by a leak-defined component split.**
  Both `id_c30` (set cover) and `id_cc30` (`--cluster-mode 1`) kept ~34% of test rows with a
  ≥90%-identity, ≥80%-coverage train neighbour. The likely cause is `-c 0.8 --cov-mode 0`, which
  requires coverage of BOTH sequences, so fragments never join their parents. This is our setting,
  not an MMseqs2 bug. New `id_lc30`: all-vs-all search (max-accept 50), an edge when
  fident ≥ 0.3 and coverage of EITHER sequence ≥ 0.8, then union-find components. The largest
  component holds 23.8% of rows, so 5 balanced folds are impossible. **User chose 4 folds**
  (2026-10-17). The c30/cc30 results are kept as evidence of the leak.
- **2026-10-17 — Low-identity diagnostic** (`scripts/diagnose_low_identity.py`, id_cc30 fold 0).
  Stratum "<30%" BRHR@1 is real signal, not a bug: no-hit pairs score at random (2.1 vs 5.1), and
  an amino-acid-type prior scores ≈ random. Part of the stratum is a binning artefact, because
  strata use the top bit-score hit. Pairs whose top-5 hits are ALL < 30% still reach 41% BRHR@1,
  and 49% when the alignment covers ≥ 80% of the sequence. Remote homologs carry pocket position.
- **2026-10-17 — Literature check.** No critique of InteractBind, and no ligand-only, homology or
  size control was found in InteractBind (arXiv 2605.24045) or ExplainBind (bioRxiv v3, 2026-06-09,
  which now lists code at github.com/ZhaohanM/ExplainBind). Docking-score size bias is known in
  general (decoy size matching, heavy-atom normalisation), so we cite it, not claim it.
  InteractBind defines its OOD splits by **global Needleman–Wunsch similarity**. Our split audit
  uses MMseqs2 local identity, so the "exceeds nominal threshold" claim is not yet comparable.
  A global-identity re-check is needed before that claim is made.
- **2026-10-23 — No GPU, CPU budget ≤ 4 h (user constraint).** DrugBAN was timed on this CPU
  (commit 9923f8c, default config, leak-free fold 0): 7.1 s/batch, ~102 min/epoch, ~170 h for
  its 100-epoch protocol. It is not retrained. Instead (user chose A+B):
  **A — learned pair models**, `scripts/run_pair_models.py`. HistGradientBoosting with fixed
  hyperparameters on Morgan(r2, 1024 bits) ligand features, AAC+DPC protein features, or both.
  This tests whether a model given the protein still reduces to the ligand shortcut, on the
  original and leak-free splits.
  **B — ligand-agnostic learned residue model**, `scripts/run_esm_probe.py`. Logistic-regression
  probe on frozen **ESM-2 8M** embeddings (`esm2_t6_8M_UR50D`), not the 150M of the spec. 150M is
  ~9 h on this CPU and 8M is ~1 h for the probe subset. Probe training uses 2,000 seeded annotated
  train FASTAs per split, label = site in ≥ 1 train pair, 3 negatives per positive. It is
  evaluated on ID fold 0 (original and leak-free) and the 4 OOD splits. The SAFE-DTI model
  (02_METHOD_SPEC) still uses 150M; this applies to the N2 probe only.
- **2026-10-24 — Ligand-Specificity Gain (LSG@K)**, `metrics/localization.py::ligand_specificity_gain`,
  built on `brhr_per_pair` (BRHR is still implemented once). For pair i of a protein with ≥ 2
  annotated test ligands: LSG_i = hit@K(r_i, own sites) − mean over the other ligands j of
  hit@K(r_i, sites of j). Any ligand-blind ranking sums to exactly 0 per protein; this is proven
  in the docstring and tested. The "own-sites ceiling" ranks each pair's true sites first.
  Reported for the homology variants. The ESM probe is ligand-blind, so its LSG is 0 by
  construction.
- **2026-10-24 — Seeds.** The HistGradientBoosting models do not use `random_state` here: no
  early stopping, and n < 200k, so binning does not subsample. Re-running them with other seeds
  is a no-op. Seed variance is reported where randomness exists: size-matched sampling (seeds
  0/1/2) and the ESM probe's train-protein sample. Fold-to-fold variance is reported throughout,
  and the ESM probe now covers all ID folds (5 + 4), not fold 0 only.
- **2026-10-25 — `-dirty` tag on the dc2e71c results.** The homology (LSG), size-matched-seed and
  all-fold probe results are tagged `dc2e71c…-dirty`. The only uncommitted change while they ran was
  `scripts/make_figures.py` (F4 now averages over folds; verified with `git diff --stat dc2e71c`).
  That script only plots and writes no results JSON, so the numbers equal those of clean dc2e71c.
  The change is committed right after. Other results are from clean f729d49; their code did not
  change.
- **2026-11-09 — History rewritten; result commit hashes re-stamped.** Repository history was
  rewritten to remove vendor/tool references from commit messages, file names and file contents,
  so every commit hash changed. The code trees differ only in comments, docstrings and the
  rename of the root working-rules file; no executable logic changed, so the recorded metrics
  still correspond to the code at the new hashes. The `git_commit` field of all 129
  `results/*.json` was mapped `91d0854 → f729d49` and `6d84071 → dc2e71c`, and the same
  substitution was applied to the docs. The results were **not** re-run. Pre-rewrite history is
  preserved in `D:/safedti/backup/neha-before-rewrite-*.bundle` for audit.
- **2026-11-09 — Paper/results consistency audit and reference verification.** Every number in
  `paper/main.tex` was checked against `results/*.json`. The five generated tables regenerate
  byte-identical from `scripts/make_paper_tables.py`, and the three figures match
  `results/figures/`. The reproducibility claim was re-verified directly against the stored second
  run in `D:/safedti/results_backup_2026-10-24`: 5,305 metric values identical, the only 39
  differences being `fit_s` wall-clock timings. Five prose numbers were corrected:
  (a) the published ID count is 84,825, of which 16 rows carry no usable label, leaving the 84,809
  we use — the paper previously attributed 84,809 to the benchmark;
  (b) the ligand-weighted homology gain is 4.1 points *mean over ID folds*, not a maximum — the
  true maximum over any fold or split is 5.8 (ID fold 1);
  (c) the OOD-33 above-50%-identity BRHR@1 range is 62.2--67.8, not 63.9--67.8 (the [0.9, 0.999)
  bin is 62.2);
  (d) the abstract said "at least 61\%" for a value of 60.7\%;
  (e) "level with FusionDTI" and "does not move at all" were replaced by the actual deltas.
  A comparability caveat was added where ExplainBind's 55.6/74.6 is quoted next to FusionDTI's
  21.6/35.6, because the two source papers report different baseline BRHR@1 values (16.1 vs 21.6).
  All 18 bibliography entries were verified against the publisher/arXiv/PMC record and every
  `VERIFY` marker is now resolved. Notable reference corrections: `dockingsizebias` had a
  fabricated title (it is Durrant et al., *J. Chem. Inf. Model.* 53(7):1726--1735, 2013);
  `fusiondti2024` was missing Ke Yuan and is published in Findings of EMNLP 2025;
  `pilot2024` is published in *Chemical Science* (2024); complete author lists were added for
  InteractBind (7), ExplainBind (14), ArkDTA (8), ISAAC (4), PILOT (5) and the DTI assessment
  paper (5). Confirmed against the benchmark paper: 99,391 total pairs, six NCI types, Vina
  thresholds -7/-5, global Needleman--Wunsch similarity with the identity denominator left
  unstated, ID folds not released, and FusionDTI's 98.3 / 21.6 / 35.6 / 81.8 / 91.8 / 28.4 / 30.1.
