### T_binary_auroc

| split | FASTA prior | UniProt prior | SMILES prior | heavy atoms | ligand-only GBM | GBM, size-matched | FusionDTI (published) |
|---|---|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | 50.0 ± 0.0 | 57.5 ± 0.2 | 98.7 ± 0.1 | 98.7 ± 0.1 | 99.9 ± 0.0 | 95.0 ± 0.4 | 98.3 |
| p_ood_25 | 50.0 | 49.3 | 69.9 | 85.2 | 85.7 | 66.9 | 81.8 |
| p_ood_28 | 50.0 | 48.5 | 68.7 | 81.2 | 87.9 | 75.0 | — |
| p_ood_31 | 50.0 | 49.0 | 70.6 | 82.6 | 89.8 | 74.1 | — |
| p_ood_33 | 50.7 | 50.0 | 63.5 | 79.3 | 88.5 | 75.2 | 91.8 |
| ID, 30% set-cover clusters (5 folds) | 50.0 ± 0.0 | 52.8 ± 0.8 | 98.2 ± 0.1 | 98.7 ± 0.3 | 99.8 ± 0.0 | 92.6 ± 1.2 | — |
| ID, 30% connected components (5 folds) | 50.0 ± 0.0 | 52.6 ± 0.3 | 98.2 ± 0.2 | 98.7 ± 0.2 | 99.8 ± 0.0 | 92.9 ± 0.5 | — |
| ID, leak-free components (4 folds, proposed) | 50.0 ± 0.0 | 51.8 ± 0.2 | 98.1 ± 0.1 | 98.7 ± 0.2 | 99.8 ± 0.1 | 91.8 ± 1.4 | — |

### T_localization_brhr1

| split | random | exact lookup | homology NN | homology kNN | kNN + ligand | oracle LOO* | FusionDTI (published) |
|---|---|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | 4.5 ± 0.1 | 4.4 ± 0.1 | 42.6 ± 2.0 | 57.5 ± 0.8 | 61.6 ± 1.2 | 67.8 ± 2.2 | 21.6 |
| p_ood_25 | 1.5 | 0.9 | 7.4 | 7.4 | 7.1 | 60.2 | — |
| p_ood_28 | 4.7 | 5.3 | 32.5 | 34.7 | 38.3 | 69.3 | — |
| p_ood_31 | 4.4 | 4.0 | 31.2 | 34.7 | 36.2 | 64.7 | — |
| p_ood_33 | 4.4 | 10.2 | 39.8 | 46.8 | 47.3 | 69.2 | — |
| ID, 30% set-cover clusters (5 folds) | 4.5 ± 0.3 | 4.4 ± 0.5 | 20.4 ± 2.1 | 32.7 ± 2.2 | 33.9 ± 2.7 | 67.4 ± 3.0 | — |
| ID, 30% connected components (5 folds) | 4.4 ± 0.4 | 4.6 ± 0.6 | 19.6 ± 2.7 | 31.4 ± 2.5 | 32.9 ± 2.4 | 67.4 ± 5.3 | — |
| ID, leak-free components (4 folds, proposed) | 4.5 ± 0.4 | 4.3 ± 0.5 | 16.0 ± 4.1 | 21.7 ± 5.2 | 22.3 ± 5.0 | 67.2 ± 1.6 | — |

### T_localization_brhr5

| split | random | exact lookup | homology NN | homology kNN | kNN + ligand | oracle LOO* | FusionDTI (published) |
|---|---|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | 20.0 ± 0.4 | 19.7 ± 0.8 | 66.5 ± 1.3 | 77.1 ± 0.7 | 79.4 ± 0.8 | 85.5 ± 1.5 | 35.6 |
| p_ood_25 | 7.3 | 8.2 | 15.0 | 14.3 | 15.3 | 73.3 | 28.4 |
| p_ood_28 | 20.9 | 19.8 | 56.4 | 56.2 | 55.9 | 88.1 | — |
| p_ood_31 | 19.8 | 21.1 | 59.1 | 57.6 | 57.4 | 87.2 | — |
| p_ood_33 | 19.7 | 25.8 | 64.4 | 70.4 | 68.7 | 92.3 | 30.1 |
| ID, 30% set-cover clusters (5 folds) | 20.0 ± 1.3 | 20.0 ± 1.4 | 40.7 ± 3.2 | 55.5 ± 2.7 | 56.9 ± 2.6 | 85.7 ± 3.5 | — |
| ID, 30% connected components (5 folds) | 19.9 ± 1.5 | 20.1 ± 1.5 | 39.7 ± 4.0 | 53.9 ± 3.8 | 54.9 ± 3.7 | 85.6 ± 4.5 | — |
| ID, leak-free components (4 folds, proposed) | 20.0 ± 1.6 | 20.0 ± 1.4 | 37.4 ± 9.4 | 43.5 ± 7.8 | 44.2 ± 8.0 | 85.4 ± 2.0 | — |

### T_split_audit

| split | nominal max identity | median nearest identity | ≥ nominal | ≥ 50% | ≥ 90% | identical | ≥ nominal (cov ≥ 0.8) | ≥ 90% (cov ≥ 0.8) |
|---|---|---|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | — | 99.8 ± 0.0 | — | 87.6 ± 0.4 | 80.2 ± 0.7 | 48.3 ± 0.9 | — | 75.5 ± 0.5 |
| p_ood_25 | 25% | 33.1 | 80.9 | 18.8 | 6.9 | 3.2 | 0.0 | 0.0 |
| p_ood_28 | 28% | 38.4 | 81.6 | 16.8 | 1.4 | 0.8 | 25.0 | 0.6 |
| p_ood_31 | 31% | 36.3 | 71.3 | 15.9 | 3.9 | 1.7 | 32.4 | 1.0 |
| p_ood_33 | 33% | 58.2 | 81.2 | 58.6 | 41.8 | 21.5 | 58.2 | 35.5 |
| ID, 30% set-cover clusters (5 folds) | — | 70.7 ± 7.8 | — | 58.1 ± 2.0 | 45.7 ± 2.1 | 29.3 ± 3.3 | — | 34.0 ± 1.3 |
| ID, 30% connected components (5 folds) | — | 73.3 ± 15.0 | — | 59.0 ± 3.3 | 45.7 ± 4.0 | 29.2 ± 3.4 | — | 34.0 ± 4.1 |
| ID, leak-free components (4 folds, proposed) | — | 30.9 ± 0.8 | — | 6.4 ± 0.6 | 0.5 ± 0.3 | 0.2 ± 0.1 | — | 0.0 ± 0.0 |

### T_brhr1_by_identity

| split | [0.0,0.3) kNN / random (n) | [0.3,0.5) kNN / random (n) | [0.5,0.9) kNN / random (n) | [0.9,0.999) kNN / random (n) | [0.999,1.01) kNN / random (n) |
|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | 33.7 ± 3.6 / 4.4 ± 0.3 (685) | 49.5 ± 3.5 / 4.6 ± 0.2 (1655) | 53.0 ± 2.9 / 4.5 ± 0.2 (1775) | 59.5 ± 1.7 / 4.1 ± 0.1 (11612) | 59.0 ± 2.1 / 4.8 ± 0.2 (10002) |
| p_ood_25 | 4.9 / 1.5 (370) | 9.4 / 1.5 (181) | 20.6 / 1.5 (63) | 3.8 / 1.5 (26) | 0.0 / 1.8 (18) |
| p_ood_28 | 18.7 / 4.2 (252) | 49.2 / 5.2 (327) | 25.7 / 4.9 (70) | 40.0 / 4.0 (5) | 0.0 / 5.1 (4) |
| p_ood_31 | 23.3 / 4.4 (283) | 44.4 / 4.5 (322) | 33.3 / 3.7 (33) | 26.7 / 3.4 (15) | 80.0 / 3.8 (5) |
| p_ood_33 | 20.2 / 4.3 (168) | 39.3 / 4.6 (168) | 63.9 / 4.1 (72) | 62.2 / 4.3 (135) | 67.8 / 4.2 (115) |
| ID, 30% set-cover clusters (5 folds) | 33.0 ± 2.3 / 4.1 ± 0.2 (4236) | 40.9 ± 3.5 / 5.1 ± 0.6 (6294) | 30.9 ± 4.0 / 4.5 ± 0.3 (2899) | 30.6 ± 5.0 / 4.4 ± 0.4 (5752) | 26.2 ± 9.8 / 4.2 ± 0.5 (6548) |
| ID, 30% connected components (5 folds) | 32.4 ± 3.8 / 4.2 ± 0.1 (4167) | 39.7 ± 4.7 / 4.8 ± 0.4 (6191) | 28.4 ± 2.0 / 4.5 ± 0.7 (3011) | 29.3 ± 3.3 / 4.3 ± 0.5 (5936) | 25.9 ± 5.2 / 4.3 ± 0.5 (6424) |
| ID, leak-free components (4 folds, proposed) | 21.9 ± 5.3 / 4.2 ± 0.3 (19446) | 21.9 ± 5.7 / 5.1 ± 0.8 (5827) | 15.6 ± 4.7 / 5.6 ± 0.8 (387) | 46.8 ± 50.3 / 3.4 ± 1.2 (49) | 0.0 ± 0.0 / 5.8 ± 1.9 (20) |

### T_pair_models_auroc

| split | ligand_only AUROC | protein_only AUROC | pair AUROC | ligand_only, size-matched | protein_only, size-matched | pair, size-matched |
|---|---|---|---|---|---|---|
| ID, FASTA-grouped (5 folds) | 99.8 ± 0.0 | 81.9 ± 0.8 | 99.9 ± 0.0 | 94.9 ± 0.4 | 80.1 ± 1.5 | 97.1 ± 0.3 |
| ID, leak-free components (4 folds, proposed) | 99.7 ± 0.1 | 64.1 ± 1.8 | 99.7 ± 0.1 | 92.4 ± 1.1 | 60.2 ± 6.8 | 93.1 ± 1.3 |
| p_ood_25 | 81.3 | 49.8 | 84.4 | 61.6 | 51.7 | 64.5 |
| p_ood_28 | 85.9 | 49.6 | 87.7 | 74.8 | 47.4 | 75.0 |
| p_ood_31 | 88.7 | 50.7 | 90.6 | 73.6 | 51.5 | 79.8 |
| p_ood_33 | 87.6 | 53.9 | 89.4 | 75.2 | 53.9 | 79.8 |

### T_esm_probe_brhr

| split | ESM probe @1 | ESM probe @5 | homology kNN @1 | homology kNN @5 | random @1 | random @5 | FusionDTI @1 (publ.) | FusionDTI @5 (publ.) |
|---|---|---|---|---|---|---|---|---|
| ID, FASTA-grouped, fold 0 | 23.4 | 57.8 | 58.2 | 76.5 | 4.4 | 19.8 | 21.6 | 35.6 |
| ID, leak-free, fold 0 | 13.1 | 40.6 | 29.3 | 54.9 | 4.9 | 21.7 | — | — |
| p_ood_25 | 8.5 | 26.6 | 7.4 | 14.3 | 1.5 | 7.3 | — | 28.4 |
| p_ood_28 | 20.2 | 49.5 | 34.7 | 56.2 | 4.7 | 20.9 | — | — |
| p_ood_31 | 23.7 | 59.9 | 34.7 | 57.6 | 4.4 | 19.8 | — | — |
| p_ood_33 | 21.1 | 56.5 | 46.8 | 70.4 | 4.4 | 19.7 | — | 30.1 |

### T_global_identity

| split | nominal max similarity | rows ≥ nominal (NW, /aln length) | rows ≥ nominal (NW, /shorter) | median best NW identity | random pairs: median | random pairs ≥ nominal |
|---|---|---|---|---|---|---|
| p_ood_25 | 25% | 3.8 | 56.6 | 7.1 | 4.0 | 0.0 |
| p_ood_28 | 28% | 18.0 | 59.5 | 21.3 | 6.6 | 0.0 |
| p_ood_31 | 31% | 29.0 | 46.2 | 24.4 | 7.8 | 0.1 |
| p_ood_33 | 33% | 60.7 | 67.4 | 49.3 | 8.5 | 0.5 |

*oracle LOO uses test labels of other ligands of the same protein (a ceiling).
