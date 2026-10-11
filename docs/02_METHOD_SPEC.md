# SAFE-DTI — Method Specification (v2)

Authoritative. Code must match this file; a deliberate deviation requires editing this file in
the same commit.

Notation: pair `(d, p)`; drug graph with `N` atoms (`N_max = 128`); protein of `L` residues
(`L_max = 1024`); hidden `H = 128`. Ground-truth contact map `M ∈ [0,1]^{N×L}` with per-type
maps `M^(t)`, `t ∈ {hbond, salt, vdw, hydrophobic, pipi, cationpi}`.

---

## 1. Encoders

**Drug — GIN over the molecular graph.** Atom features: RDKit canonical set (74-dim: atom type,
degree, implicit valence, formal charge, radical electrons, hybridisation, aromaticity, total H).
3 GIN layers, hidden 128, BatchNorm + ReLU, no pooling yet.
`H_d ∈ R^{N×128}`.

> Keeping **atoms** (not SELFIES tokens) is still a deliberate choice for representation quality,
> but the public InteractBind release has **residue-level labels only** (no atom axis). All
> supervision and evaluation are therefore on the residue axis via `s_j = max_i S_ij`. No
> atom-side supervision or metrics. See DECISIONS.md, 2026-10-11.

**Protein sequence — frozen ESM-2.** `esm2_t30_150M_UR50D` (640-d), precomputed to disk in fp16.
`Linear(640→128)` then a 3-layer 1D CNN (kernels 3/6/9, residual) for local motif context.
`H_seq ∈ R^{L×128}`.

**Protein structure — optional.** Residue graph from Cα coordinates: edge if `‖C_i − C_j‖ < 8 Å`,
degree capped at 20, plus sequential neighbours; RBF-expanded distance (16 bins, 2–8 Å) as edge
weights. Node input `[H_seq_i ; pLDDT_i/100 ; i/L]` → `Linear(130→128)` → 2 GCN layers.
`H_str ∈ R^{L×128}`. Deliberately not SE(3)-equivariant; the question under test is gating, not
geometric expressiveness. State this in Limitations.

---

## 2. Structure-Adaptive Gate (C3)

```
r_i = σ((pLDDT_i − τ)/T)            τ = 70, T = 10      # =1 for experimental structures
g_i = σ(W_g [H_seq_i ; H_str_i] + b_g)                   # elementwise, R^128
H_p_i = H_seq_i + a · r_i · (g_i ⊙ H_str_i)              # a ∈ {0,1} availability
```

*Property 1 (exact fallback).* `a = 0` ⟹ `H_p ≡ H_seq`, and no structure parameter touches the
output. The sequence-only ablation is literally the same weights, not a retrained model.
*Property 2 (confidence bound).* `‖H_p_i − H_seq_i‖ ≤ r_i‖H_str_i‖`, so a pLDDT-30 residue
(`r ≈ 0.018`) can shift its representation by at most ~2% of the structural signal.

Training uses **modality dropout**: for proteins with structure, `a ~ Bernoulli(0.7)`.

---

## 3. Interaction map and prediction head

Multi-head cross-attention, `n_heads = 4`, head dim 32.

```
S^(h) = (W_Q^(h) H_d)(W_K^(h) H_p)^T / √32        ∈ R^{N×L}
S     = mean_h S^(h)                               # THE interaction map (logits)
```
Mask padded atoms/residues with −inf.

**Attention sink (after ArkDTA).** Append one learned pseudo-residue column `s_sink` to `S`
before any softmax. Residues with no contact can dump their mass there instead of being forced
onto real residues. Cite ArkDTA; ablate it.

Contextualisation:
`H_d' = LN(H_d + softmax_L(S) · W_V H_p)` ; `H_p' = LN(H_p + softmax_N(Sᵀ) · W_U H_d)`,
each followed by a 2-layer FFN.

Pooling and classifier:
```
α = softmax(w_dᵀ tanh(W_d H_d')) ; z_d = Σ α_i H_d'_i
β = softmax(w_pᵀ tanh(W_p H_p')) ; z_p = Σ β_j H_p'_j
z = [z_d ; z_p ; z_d ⊙ z_p]  → MLP(384→512→128→1), ReLU, Dropout 0.2 (kept on for MC-dropout)
```

**Residue score for BRHR** (match the benchmark exactly): `s_j = max_i S_ij` over valid atoms,
sink column excluded. Do not substitute your own aggregation, or your numbers stop being
comparable to their Table 2. You may *additionally* report a logsumexp variant.

**Six type heads.** `S^(t) = (W_Q^(t) H_d)(W_K^(t) H_p)ᵀ / √32` sharing `H_d, H_p`, one small
projection pair per type. These exist only to shape the representation; the reported map stays `S`.

---

## 4. Grounding loss (C1) — the core contribution

Applied **only to positive pairs with annotations**; masked elsewhere.

Let `y⁺_j = 1` if residue `j` is a binding residue, `n⁺ = Σ_j y⁺_j` (typically 5–30 of hundreds).

**(a) Residue listwise ranking.** Target distribution `q = y⁺ / n⁺`; prediction
`p = softmax_j(s_j / T_rank)`, `T_rank = 1.0`.
`L_rank = CE(q, p) = −Σ_j q_j log p_j`.
Robust to sparsity and to which atom contacts which residue — it only asks that true binding
residues rank high. This is the term most directly aligned with BRHR.

**(b) ~~Atom–residue contrastive~~ — DROPPED (2026-10-11): needs atom-level maps, which are not
public.** Kept here for the record only; do not implement.

**(b, original text)**
For each annotated contact `(i,j)` with strength `M_ij`, sample `K = 32` negative residues `j⁻`
uniformly from non-annotated residues of the same protein:
```
L_con = − Σ_{(i,j) ∈ Ω} M_ij · log [ exp(S_ij/T_c) / (exp(S_ij/T_c) + Σ_{j⁻} exp(S_ij⁻/T_c)) ]
```
`T_c = 0.1`, normalised by `|Ω|`. Sampling rather than dense BCE is the positive-unlabeled
mitigation: unannotated cells are treated as *probable* negatives, never as hard zeros.

**(c) Type-aware auxiliary supervision.** For each type `t` with residue-level targets `y^(t)`,
the same listwise loss on `s^(t)_j = max_i S^(t)_ij`, weighted by inverse frequency:
```
L_type = Σ_t w_t · CE(q^(t), softmax(s^(t)))      w_t ∝ 1 / sqrt(freq_t),  Σ_t w_t = 1
```
This is what attacks the π–π (18.2) and cation–π (16.1) BRHR@5 floor. Ablating `L_type` and
showing per-type deltas is a required experiment.

**(d) Sink regulariser.** Encourage non-binding residues toward the sink:
`L_sink = mean_{j : y⁺_j = 0} softmax_sink_mass_j`, small weight.

**Total:**
```
L_stage1 = L_cls + λ_r L_rank + λ_t L_type + λ_s L_sink
λ_r = 1.0, λ_t = 0.5, λ_s = 0.1        # tune λ_r first; it dominates
```
Ramp grounding terms in linearly over the first 3 epochs so the map is not shaped before the
encoders are meaningful.

---

## 5. Grounding transfer (C2)

**Stage 1** — train on InteractBind-ID with `L_stage1`. Save weights `θ₁`.

**Stage 2** — fine-tune on BindingDB / BioSNAP / Human, which have **binary labels only**:
```
L_stage2 = L_cls + λ_cons · L_cons
```
where `L_cons` prevents grounding collapse using an EMA teacher `θ_ema` (decay 0.999,
initialised from `θ₁`, no gradient):
```
L_cons = KL( softmax_j(s_j^teacher / T_m) ‖ softmax_j(s_j^student / T_m) )   T_m = 2.0
```
plus an optional entropy-matching term holding the student's map entropy near the stage-1
average. `λ_cons ∈ {0, 0.1, 0.5, 1.0}` is the central ablation of C2.

**Grounding retention (new metric, define it carefully in the paper):**
```
GR = BRHR@K(θ₂) / BRHR@K(θ₁),  measured on a held-out InteractBind test split
                                that is used in NEITHER stage
```
Report `GR@1` and `GR@5` alongside absolute BRHR. Guard against leakage: targets in that held-out
split must not appear in the stage-2 datasets — filter by sequence identity ≥ 40% using MMseqs2.

---

## 6. Uncertainty and grounding margin (C4)

**Epistemic:** MC-dropout, `T = 20` passes, `p̂ = mean σ(logit)`, `u = var σ(logit)`.

**Grounding margin** from the interaction map:
```
m = ( mean of top-5 residue scores − median residue score ) / (IQR of residue scores + ε)
```
Plus normalised map entropy `h = H(softmax(s)) / log L`. Both computed from `S` at no extra cost.
Define once in `uncertainty/score.py` and never redefine.

**Nonconformity score:** `σ_nc = (1 − p̂_y) · (1 + γ·h − γ·m̃)` with `m̃` the min-max-normalised
margin and `γ` tuned on validation. Compare against the plain `1 − p̂_y`.

**Mondrian conformal:** calibration sets partitioned by protein-similarity band (25/28/31/33%)
and by structure availability. Threshold `q_k` at `α = 0.1`. Report per-band empirical coverage,
average set size, and the risk–coverage curve (AURC) with and without the grounding term.
Baselines: marginal CP, temperature scaling, raw sigmoid.

---

## 7. Hyperparameters

```yaml
batch_size: 32            # maps are N×L; memory-bound
optimizer: AdamW
lr: 1e-4                  # stage 1
lr_stage2: 2e-5
weight_decay: 1e-5
scheduler: cosine, warmup 500 steps
epochs: 40 (stage 1), 15 (stage 2)
early_stopping: val AUPRC + val BRHR@5 (sum of ranks), patience 8
grad_clip: 5.0
dropout: 0.2
seeds: [0,1,2,3,4]        # 5 seeds, mean ± std — non-negotiable
amp: true
N_max: 128                # ligands are small; check the dataset's distribution first
L_max: 1024
hidden: 128 ; heads: 4
tau_plddt: 70 ; T_plddt: 10 ; p_drop_structure: 0.3
lambda: {rank: 1.0, type: 0.5, sink: 0.1, cons: 0.5}
T_rank: 1.0 ; T_c: 0.1 ; T_m: 2.0
ema_decay: 0.999
```

Truncation (**training only**): never truncate a protein in a way that drops annotated binding
residues. If `L > L_max`, crop a 1024-residue window centred on the annotated contacts.
**At evaluation, never use labels to choose a window** — that leaks the answer. Eval uses the full
sequence (ESM embeddings computed in overlapping windows for L > 1022; the map is N×L at any L).
(DECISIONS 2026-10-11; 42% of p_ood_25 test pairs have L > 1024.) Original text continued: (and for unannotated
proteins, centre on the highest-pLDDT contiguous window). **Log every truncation; a silent crop
that removes the answer will quietly destroy your BRHR numbers.**

Trainable parameters ≈ 3–4 M (ESM frozen). A stage-1 epoch over ~85k pairs should take minutes,
not hours. If it doesn't, ESM is being run inside the loop or maps are being rebuilt per batch.

---

## 8. Required unit tests before any training run

- `test_sag_fallback` — `a=0` output bitwise-identical to sequence-only; structure grads None.
- `test_brhr_matches_spec` — hand-built 4×6 map with a known answer gives the expected BRHR@K.
- `test_map_alignment` — a synthetic pair whose binding-site string has a 1 at position 7 yields
  `y⁺[7] = 1` (0-based, aligned to FASTA) and `len(y⁺) == len(fasta)`; **off-by-one in residue indexing is the single most likely bug in this
  project and would silently invalidate every number.**
- `test_mask_invariance` — padding does not change the logit (tol 1e-5) or the residue ranking.
- `test_grounding_loss_zero` — a perfect map gives `L_rank ≈ 0`.
- `test_pu_sampling` — negatives are never drawn from annotated residues.
- `test_overfit` — 200 pairs, train AUROC > 0.95 and BRHR@1 > 0.8 within 300 steps.
- `test_stage2_no_leak` — no stage-2 target has ≥ 40% identity to a held-out retention target.
