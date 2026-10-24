#!/usr/bin/env bash
# Reproduce every N2 benchmark-audit result, table and figure (CPU only, ~2-3 h on an i5 laptop).
# Assumes `make verify-data` has downloaded InteractBind. Set SPLITS=1 to also rebuild the split
# files (MMseqs2), otherwise the existing data/processed/interactbind/*_folds.parquet are used.
#   SEED=0 SPLITS=0 bash scripts/run_audit_all.sh
set -euo pipefail
PY=${PY:-.venv/Scripts/python}
SEED=${SEED:-0}
export TORCH_HOME=${TORCH_HOME:-D:/safedti/torch_cache}

if [[ "${SPLITS:-0}" == "1" ]]; then
  $PY scripts/make_id_splits.py --seed "$SEED"
  $PY scripts/make_cluster_splits.py --config configs/baselines/cluster_split.yaml --seed "$SEED"
  $PY scripts/make_cluster_splits.py --config configs/baselines/cluster_split_cc.yaml --seed "$SEED"
  $PY scripts/make_component_splits.py --config configs/baselines/component_split.yaml --seed "$SEED"
fi

$PY scripts/run_lookup_baselines.py --seed "$SEED"
$PY scripts/run_ligand_audit.py --seed "$SEED"          # writes size-matched subsets used below
# extra sampling seeds for the size-matched subsets (GBMs themselves are seed-independent)
SM="id_fold0 id_fold1 id_fold2 id_fold3 id_fold4 id_lc30_fold0 id_lc30_fold1 id_lc30_fold2 id_lc30_fold3 p_ood_25 p_ood_28 p_ood_31 p_ood_33"
for s in 1 2; do $PY scripts/run_ligand_audit.py --seed "$s" --splits $SM; done
$PY scripts/run_homology_baselines.py --seed "$SEED"
$PY scripts/run_pair_models.py --seed "$SEED"
$PY scripts/run_esm_probe.py --seed "$SEED" --select
$PY scripts/precompute_esm.py --model esm2_t6_8M_UR50D --out data/processed/esm \
  --seqs-file data/processed/esm/probe_sequences.txt   # resumable; skips cached sequences
$PY scripts/run_esm_probe.py --seed "$SEED"
$PY scripts/diagnose_low_identity.py --split id_cc30_fold0 --seed "$SEED"
$PY scripts/run_global_identity_audit.py --seed "$SEED"
$PY scripts/make_tables.py --seed "$SEED"
$PY scripts/make_figures.py --seed "$SEED"
$PY scripts/make_paper_tables.py --seed "$SEED"
cp results/figures/F2_brhr_by_identity.pdf results/figures/F3_binary_shortcut.pdf \
   results/figures/F4_localization_without_ligand.pdf paper/figures/
echo "audit complete"
