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
