# Data Pipeline (v2)

Runs once, writes `data/processed/`, then frozen. Training never touches raw files or the network.

---

## 1. Primary dataset — InteractBind

`https://huggingface.co/datasets/Zhaohan-Meng/InteractBind`

| Subset | Pairs | Proteins | Ligands | Use |
|---|---|---|---|---|
| Complete | 99,391 | 11,473 | 9,017 | — |
| **ID** (affinity-based) | 84,825 | 10,942 | 6,911 | stage 1, 5-fold CV |
| OOD P-25% | 13,126 | 3,278 | 9,527 | hardest OOD |
| OOD P-28% | 13,126 | 2,966 | 8,957 | |
| OOD P-31% | 13,126 | 3,202 | 10,083 | |
| OOD P-33% | 13,126 | 4,807 | 11,084 | easiest OOD |

OOD splits are 8:1:1 train/val/test, 5 seeds. ID is 5-fold CV. **Use their splits unmodified** —
that is what makes your numbers comparable to their Table 2.

Per pair you get: protein FASTA, protein SaProt (SA) sequence, ligand SMILES and SELFIES, binary
label, docking affinity, six interaction maps, aggregated map.

**Day-1 verification task — do this before anything else:**
1. Download and load 10 samples. Print the actual shapes and dtypes of the maps.
2. Confirm the map's residue axis matches `len(FASTA)` and the atom axis matches the RDKit atom
   count of the SMILES *in RDKit's canonical atom order*. It very likely will not match
   automatically — the maps were built from PDB atom order. **Write an explicit
   PDB-atom-order → RDKit-atom-order permutation, with a test.** If you cannot resolve the atom
   correspondence, fall back to residue-axis-only supervision (losses 4a, 4c, 4d in the method
   spec still work) and record the decision in `DECISIONS.md`.
3. Recompute BRHR@5 for a random-ranking baseline and sanity-check it against `K/L`.
4. Write `DATA_CARD.md` with the download date and file hashes.

Known caveats to state in the paper: negatives are **docking-derived** (Vina global docking,
> −5 kcal/mol), so there is a docking-score prior; metal-containing complexes were excluded;
multi-ligand PDB entries were split by a 0.5 nm proximity rule.

---

## 2. Transfer datasets (stage 2, binary labels only)

From the DrugBAN repo (https://github.com/peizhenbai/DrugBAN), which ships preprocessed CSVs and
both random and cluster splits:

| Dataset | Drugs | Targets | Pairs | Notes |
|---|---|---|---|---|
| BindingDB | ~7.1k | ~1.4k | ~32.6k | imbalanced |
| BioSNAP | ~4.5k | ~2.2k | ~27.5k | balanced, random negatives |
| Human | ~2.7k | ~2.0k | ~6.7k | known hidden-ligand-bias issues |

Build `target_map.csv` mapping each target sequence to a UniProt accession: exact hash match
against SwissProt first, then MMseqs2 at ≥ 98% identity / ≥ 95% coverage. You need accessions
for structures (C3) and for the leakage filter (C2).

**Leakage filter for C2 — mandatory.** Cluster all stage-2 targets together with the held-out
InteractBind retention targets at 40% identity (MMseqs2 `easy-cluster --min-seq-id 0.4`). Any
retention target sharing a cluster with a stage-2 target is dropped from the retention set.
Without this, "grounding retention" just measures memorisation and the result is worthless.

---

## 3. Structures (for C3)

Priority: experimental PDB via SIFTS (`plddt := 100` on resolved residues) → AlphaFold DB
`AF-{acc}-F1-model_v4.cif` (pLDDT in the B-factor column) → none (`a = 0`).

**Never ESMFold** — it is built on ESM-2, so the structural modality would not be independent of
the sequence modality. State this in the paper.

Save `data/processed/structures/{acc}.npz` with `coords (L,3) f32`, `plddt (L,) f32`,
`mask (L,) bool`; missing residues get NaN coords and `r_i = 0`. Manifest:
`acc, source, coverage, mean_plddt, frac_plddt_gt70, n_residues`.

InteractBind proteins come from PDB complexes, so experimental structures should be available
for most of them — good for C3's clean case. The stage-2 datasets are where AlphaFold coverage
matters; measure and report it (it becomes a figure).

---

## 4. Precomputed ESM-2 embeddings

```
scripts/precompute_esm.py --model esm2_t30_150M_UR50D --fasta data/interim/all_targets.fasta \
    --out data/processed/esm/ --dtype float16 --max-len 1024
```
One memory-mapped `.npy` per target hash, shape `(L, 640)`. Roughly 1.2 KB/residue in fp16.
Deduplicate across all datasets first — many targets recur.

Optionally also precompute SaProt embeddings for the FusionDTI-style comparison row.

---

## 5. Drug graphs

RDKit → PyG `Data`, cached in a single `.pt` keyed by canonical SMILES, **storing the
atom-order permutation** needed to align with the ground-truth map axis (see §1 step 2).
Log and drop sanitisation failures with counts.

---

## 6. Final artifacts

```
data/processed/
├── interactbind/{id_fold{0..4},ood_p{25,28,31,33}}/{train,val,test}.parquet
├── interactbind/maps/            # sparse COO per pair: (atom_idx, res_idx, type, strength)
├── retention_set.parquet         # held-out, identity-filtered, used in neither stage
├── transfer/{bindingdb,biosnap,human}/{train,val,test}_{random,cluster}.csv
├── target_map.csv  ·  structure_manifest.csv  ·  structures/*.npz
├── esm/*.npy  ·  drug_graphs.pt
└── DATA_CARD.md
```

Store maps **sparse**. A dense 128×1024 float array per pair × 85k pairs is ~45 GB; the same
data as COO triples is a few hundred MB. Densify on the fly in the collate function.

`DATA_CARD.md` must record: InteractBind version/commit and download date, DrugBAN commit,
UniProt release, AlphaFold DB version, and every filtering count as `n in → n out`.
