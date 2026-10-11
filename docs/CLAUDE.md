# CLAUDE.md — working rules for this repository

**SAFE-DTI** is an M.Tech research project that will become a publication. Correctness and
reproducibility matter more than speed. Read `docs/00_PROJECT_BRIEF.md` and
`docs/02_METHOD_SPEC.md` before writing any model code.

## Project in three sentences
Drug–target interaction models reach ~98% AUROC but identify the correct top binding residue
only ~22% of the time, because they are trained on binary labels alone. SAFE-DTI supervises the
cross-attention interaction map directly against residue–atom contact annotations from the
InteractBind dataset, using an interaction-type-aware, sparsity- and PU-robust grounding loss,
with an optional pLDDT-gated structural branch. The central research question is whether that
grounding survives fine-tuning on the large binary-only datasets used in real screening.

## Source of truth
- `docs/02_METHOD_SPEC.md` — architecture, losses, hyperparameters. **Code must match it.**
- `docs/03_DATA_PIPELINE.md` — all data handling.
- `docs/04_IMPLEMENTATION_PLAN.md` — milestones and acceptance criteria.
- `docs/05_EXPERIMENTS.md` — what gets measured.
- `docs/DECISIONS.md` — append-only; log every non-trivial choice with a one-line reason.

## Hard rules
1. **No fabricated results.** Never write placeholder metrics or example output that looks like
   a real result. If something cannot be run, say so.
2. **No silent spec deviation.** Changing a hyperparameter or architecture detail requires
   editing `docs/02_METHOD_SPEC.md` in the same commit.
3. **BRHR is implemented exactly once**, in `src/safedti/metrics/localization.py`, exactly as
   the spec defines it (max over atoms, rank residues, hit if any of top-K is a true binding
   residue, sink column excluded). Never reimplement or "improve" it inline — the numbers must
   stay comparable to the published benchmark.
4. **Residue and atom indexing is the project's most dangerous bug.** Any code touching the
   ground-truth map needs a test with a hand-checked example.
5. **Tests before training.** Especially `test_atom_align`, `test_brhr_matches_spec`, and
   `test_overfit`.
6. **Determinism.** Every entry point takes `--seed`; seed torch, numpy, random; set
   `cudnn.deterministic = True`.
7. **Every run writes a JSON** to `results/` with git commit, full config, seed, metrics, wall
   time. No number may enter the paper without one.
8. **No network calls in training or dataloading.** Downloads live in `scripts/`. ESM-2
   embeddings are precomputed; never instantiate ESM inside a training loop.
9. **Maps are stored sparse** (COO) and densified in the collate function. Never store dense.
10. One config change per experiment. Ablations differ by exactly one flag.
11. No logic in notebooks; they read `results/*.json` and plot.

## Style
Python 3.11, type hints on public functions, Google docstrings. `ruff format` + `ruff check`
before committing. Config in YAML, no magic numbers in code. Document tensor shapes as
`(B, N, L)` and assert them at module boundaries. Conventional commits
(`feat:`, `fix:`, `test:`, `docs:`, `exp:`).

## Commands
```bash
make setup
make verify-data                 # InteractBind sanity report (run this first, always)
make test
make train STAGE=1 CFG=configs/model/safedti.yaml DATA=ib_id SEED=0
make train STAGE=2 CFG=configs/stage/stage2_lam05.yaml DATA=biosnap SEED=0
make eval RUN=results/ib_id_safedti_0.json
make tables
```

## When you are unsure
Ask. Specifically: ambiguous dataset fields, two plausible ways to compute a metric, a missing
external tool, or anything the spec is silent about. Never guess on anything affecting a
reported number.

## Current status
<!-- Update at the end of every session. -->
- Milestone: M0 not started
- Last run: none
- Known issues: none
- Open questions: PDB→RDKit atom-order alignment unverified (M1)
