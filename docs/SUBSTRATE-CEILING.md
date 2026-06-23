# Is ~506 the A10G hardware limit, or a monoculture? — substrate study 2026-06-23

A from-scratch study (53-agent workflow, bandwidth-physics-anchored, adversarially red-teamed) of the
question: *is the verified board's ~506–507 flatline the A10G hardware limit, or just everyone copying one
shared stack?* Short answer: **a real monoculture on unused bandwidth — but the unused bandwidth is
unreachable because gemma-4-E4B can't be quantized below int4.**

## 1. The flatline is the int4 *substrate* ceiling, NOT the silicon ceiling

Every verified top entry (sparkgemma 506.63 / us 506.74 / vidraft 506.94 / firfir 507.00) serves the
**identical** chiku-inu osoi5-37 int4-g128 weights + firfir hayai kernels + 12k head + kenyan-duma drafter.
The only inter-agent variation is byte-level config (w188/w192, ctk48/49). It's a genuine monoculture.

Backed out from the measured anchor (506.74 tps, ~594 GB/s effective HBM, STEPTIME = ~99% GPU-bound on int4
weight bandwidth, ~77% of the M=1 GEMV roofline = near-optimal for int4-GEMV-at-M=1), the **A10G has
~+100–200 tok/s of unused HBM bandwidth** above 506. Physical ceilings *if* a fewer-bytes substrate were
reachable at valid ppl:

| substrate | system byte ratio | ceiling tps |
|---|---|---|
| int4 g128 (osoi5 baseline) | 1.00 | 506.7 |
| int3 | 0.84 | ~605 (+19%) |
| int4 + 2:4 (realistic, 3 bits/wt) | 0.84 | ~605 |
| int2 | 0.68 | ~751 (+48%) |
| MatFormer width-slice ×0.75 | 0.75 | ~676 |

**So the board is NOT at the silicon limit.** The instinct that "scores flatline because everyone copies
one researched stack" is correct.

## 2. The real wall: gemma-4-E4B can't go below int4

The spare bandwidth is unreachable because **gemma-4-E4B is catastrophically quant-fragile below int4** —
measured ([llama.cpp #22407](https://github.com/ggml-org/llama.cpp/issues/22407)): **Q4_K_M ppl ≈ 23,
Q3_K_M ppl ≈ 919.** Its Per-Layer-Embedding (PLE) architecture has a hard sub-4-bit cliff. Combined with
osoi5's razor margin (ppl 2.394, only **0.026** under the 2.42 cap), every fewer-bytes substrate busts the
gate. The monoculture is *forced by the model*, not by laziness — int4 is the floor for this architecture.

## 3. Candidate graveyard (36 candidates → why each fewer-bytes path dies)

| path | why it's dead |
|---|---|
| int3 / int2 on osoi5 | the PPL cliff (Q3=919); osoi5's 0.026 margin busts instantly |
| rotation (QuaRot/SpinQuant) → int3 | rotation fixes *activation* outliers, not the PLE *embedding-grid* cliff; and no W3A16 GEMV kernel exists on sm_86 (BitBLAS=W2/W4, GemLite=1/2/4/8, ITQ3=Blackwell-only) |
| 2:4 structured sparsity | keeps int4 (dodges the cliff) BUT batch=1 marginal gain over int4 is only ~1.1–1.2× (the 3–5× Sparse-Marlin headline is the batch≥16 *compute* regime); needs sparse *pretrain* (13B tok, multi-H100) to recover accuracy |
| codebook/VQ (QuIP#, AQLM, QTIP) | *invert* at M=1 — gather-bound, ~30–48% BW vs Marlin's 77% → run **below** 506 |
| MatFormer width-slice | gemma-4-E4B is not a config-array MatFormer; E2B is a separate model, not a nested slice |
| faithful-base layer-drop to close the gap | KV-sharing forecloses it (measured ppl 28.67 / 5.14); 334→506 needs ~22 drops, headroom funds ~5 |

## 4. The one live probe (low EV, but free)

**2:4-MLP SparseGPT on the faithful QAT base** (333.91 tps, ppl 1.979, 0.44 headroom — the *only* base in
the field with ppl budget) is the single lever that spends headroom *without* hitting the int3 cliff
(Sparse-Marlin runs on sm_86). But: batch=1 gain ~1.1–1.2×, and even a stacked best-case physics ceiling is
**~499 — just short of 506**. **P(beats 506 gate-safe) ~5–10%.**

Worth running ONLY because the gate is free: **offline 2:4-MLP ppl check on the 3080** (no A10G slot).
Sequence: one-shot SparseGPT 2:4 the *MLP weights only* of the faithful base → measure token-level micro
ppl. **If ppl > 2.35, dead.** Only if ≤2.35: microbench Sparse-Marlin M=1 GEMV (confirm ≥90% of dense-Marlin
GB/s — its M=1 path is undocumented). Only if both pass: one A10G slot for served tps + greedy-identity.
Do **not** touch the osoi5 base (0.026 margin = instant bust). Do **not** pursue int3/int2/codebook/width-slice.

## 5. Honest conclusion

**Not hardware-saturated — but effectively at the limit for *this model*, for a PPL+architecture reason
(the sub-4-bit cliff), not a bandwidth one.** The monoculture hypothesis is right that the field sits in one
local optimum on unused bandwidth; it's wrong if it implies a *reachable* better optimum exists for
gemma-4-E4B on the A10G at ppl ≤ 2.42. **Hold the verified 506.74.** Spend zero paid slots; run only the
free offline 2:4-MLP ppl probe to definitively close-or-open the last live lever. The faithful quality stack
(333.91, capability-neutral) remains the hedge if the board ever re-gates on capability.

Sources: llama.cpp #22407 (gemma-4 sub-4-bit ppl cliff); [2:4 Sparse-Llama, Red Hat/Neural Magic 2025](https://developers.redhat.com/articles/2025/02/28/24-sparse-llama-smaller-models-efficient-gpu-inference);
[GemLite low-bit Triton GEMV](https://dropbox.github.io/gemlite_blogpost/); any4/tinygemm low-batch Ampere
(arXiv:2507.04610); SparseGPT (arXiv:2301.00774).
