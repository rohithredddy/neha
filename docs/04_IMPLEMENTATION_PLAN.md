# Implementation Plan (v2)

One milestone ≈ one Claude Code session. Each has a definition of done you can *run*, not read.

---

## Repository layout

```
safe-dti/
├── CLAUDE.md
├── README.md · pyproject.toml · Makefile
├── docs/            00_..05_, DECISIONS.md, refs.bib
├── configs/
│   ├── base.yaml
│   ├── model/{safedti,no_grounding,no_type,no_sink,seq_only,no_gate}.yaml
│   ├── data/{ib_id,ib_ood25,ib_ood28,ib_ood31,ib_ood33,bindingdb,biosnap,human}.yaml
│   └── stage/{stage1,stage2_lam0,stage2_lam05,stage2_lam1}.yaml
├── src/safedti/
│   ├── data/      interactbind.py transfer.py maps.py atom_align.py structures.py splits.py
│   ├── models/    drug_gin.py protein_seq.py protein_struct.py sag.py cross_attn.py safedti.py
│   ├── losses/    classification.py grounding.py consistency.py
│   ├── metrics/   binary.py localization.py   # BRHR lives here, implemented once
│   ├── uncertainty/ mcdropout.py conformal.py score.py
│   ├── train.py eval.py
│   └── utils/     seeding.py logging.py io.py
├── scripts/       verify_interactbind.py precompute_esm.py fetch_structures.py
│                  build_retention_set.py make_tables.py run_all.sh
├── tests/ · results/ · notebooks/   # notebooks plot only, no logic
```

---

## M0 — Scaffold *(half session)*
Pinned deps (torch, torch-geometric, rdkit-pypi, fair-esm, datasets, biopython, scikit-learn,
pandas, pyarrow, pyyaml, captum, pytest, matplotlib). Config loading, deterministic seeding,
JSON results writer, `tests/test_smoke.py`.
**Done:** `pytest` green; `python -m safedti.train --config configs/base.yaml --dry-run` exits 0.

## M1 — InteractBind verification *(the highest-risk session; do it first)*
`scripts/verify_interactbind.py`: load, print real shapes/dtypes, check residue axis vs FASTA
length, parse residue-level binding-site strings (no atom alignment — public data has no atom
axis, see DECISIONS.md 2026-10-11), fix the ID fold assignment, compute
a random-ranking BRHR baseline, write `DATA_CARD.md`.
**Done:** a report printed with real numbers from real data, and `tests/test_map_alignment.py`
passing on ≥ 50 random pairs. Also train the ligand-agnostic ESM linear probe (N2) here — it is
the earliest real result and a sanity baseline. (Old text: if atom alignment fails, record and proceed
residue-only.

## M2 — Metrics first
Implement `metrics/localization.py` (BRHR@K exactly as specified, plus Precision@k, Recall@k,
AUPRC over residues) and `metrics/binary.py`. Unit-test BRHR on hand-built matrices.
**Done:** `test_brhr_matches_spec` passes; a random-baseline BRHR on real data is printed and
is close to the analytic `1 − C(L−n⁺, K)/C(L, K)`.

> Metrics before models. If BRHR is wrong, every subsequent number is wrong and you won't notice.

## M3 — Data loading, sparse maps, ESM cache, structures
`interactbind.py`, sparse COO map storage with on-the-fly densify in collate, `precompute_esm.py`,
`fetch_structures.py`, `build_retention_set.py` with the 40% identity leakage filter.
**Done:** one batch loads in < 100 ms; `tests/test_stage2_no_leak.py` passes; structure coverage
printed per dataset.

## M4 — Model
Encoders, SAG, cross-attention with sink, type heads, classifier. All tests from
`02_METHOD_SPEC.md` §8 written **first**.
**Done:** every §8 test passes, including `test_overfit` reaching BRHR@1 > 0.8 on 200 pairs.

## M5 — Grounding losses (C1)
`losses/grounding.py`: listwise rank, PU-sampled contrastive, type-aware auxiliary, sink
regulariser, with the ramp schedule.
**Done:** stage-1 run on InteractBind-ID beats the reported FusionDTI BRHR@1 of 21.6, or you
have a documented diagnosis of why not.

## M6 — Baselines
Re-run FusionDTI and DrugBAN on the same splits (both public). Implement ArkDTA-style
"supervision, pooled, no types" as the key ablation, and the concat-MLP sanity baseline. Cite
InteractBind's Table 2 for the models you don't re-run.
**Done:** `results/baselines/` has 5-seed results per model × split.

## M7 — Transfer (C2)
Stage-2 fine-tuning with the EMA-teacher consistency loss; the `λ_cons` sweep; grounding
retention measured on the clean held-out set.
**Done:** a table of `AUROC(stage2)` × `BRHR(retention)` × `λ_cons`, 5 seeds.

## M8 — Uncertainty (C4)
MC-dropout, grounding margin and map entropy, Mondrian conformal over similarity bands,
risk–coverage curves, the confident-but-ungrounded error analysis.
**Done:** per-band coverage within ±2% of 90%; AURC with and without the grounding term.

## M9 — Full matrix, figures, writing
Everything in `05_EXPERIMENTS.md`, 5 seeds, significance tests, LaTeX tables, paper draft.

---

## Engineering rules

1. Every run writes a JSON to `results/` with git hash, full config, seed, all metrics, wall
   time, data-card version. **No number enters the paper that isn't in a results file.**
2. 5 seeds, mean ± std, paired Wilcoxon for every headline claim, Benjamini–Hochberg across the
   ablation family.
3. One config change per experiment.
4. Append to `docs/DECISIONS.md` for any choice not in the spec. This becomes your methodology
   chapter.
5. Branch per milestone; commit after each.
6. Long runs go in your own terminal under `tmux`, not inside the agent.

---

## Timeline (26 weeks)

| Weeks | Work |
|---|---|
| 1 | M0 + **M1** — verify the data before committing to the project |
| 2 | M2 metrics + reproduce FusionDTI as published |
| 3–4 | M3 data pipeline |
| 5–7 | M4 model |
| 8–11 | M5 grounding losses, tuning, first real result |
| 12–13 | M6 baselines |
| 14–17 | M7 transfer study |
| 18–20 | M8 uncertainty |
| 21–23 | M9 full matrix, ablations, figures |
| 24–26 | Paper, thesis, repo release |

**Decision gate at week 11.** If grounding has not improved BRHR@1 by at least 5 absolute points
by then, stop adding contributions: drop C3 and C4, and reframe the thesis as a rigorous
negative/diagnostic study of *why* interaction-map supervision does not transfer into
localization. That is still a thesis and still a paper. Do not spend months on C3/C4 bolted onto
a broken C1.
