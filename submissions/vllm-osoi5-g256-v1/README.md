# `vllm-osoi5-g256-v1` — coarser-quant body (g128 → g256)

**Status: ❌ negative.** Benched **461.01 tok/s** (vs our verified **506.74**) at PPL **2.5628**
(over the ~2.42 cap → invalid). Regresses on *both* axes. See the result record
[`results/vllm-osoi5-g256-v1/`](../../results/vllm-osoi5-g256-v1) and the writeup
[`drafts/2026-06-23-g256-result.md`](../../drafts/2026-06-23-g256-result.md).

## What this submission is

The **identical serve stack** to our verified SOTA [`vllm-hayai-repro-v1`](../vllm-hayai-repro-v1)
— same custom vLLM wheel, split-KV verify, FA-sliding (w192), ONEGRAPH, fused-sparse-argmax,
CENTROID_TOP_K=48, 16k→12k lm_head re-prune, kenyan-duma MTP K=7 drafter — served on **one changed
input**: the osoi5 int4 body re-grouped from **group_size=128 to group_size=256**.

So the only diff from `vllm-hayai-repro-v1/manifest.json` is the weights:

```diff
- "WEIGHTS_BUCKET": "hf://buckets/gemma-challenge/gemma-chiku-inu/weights/osoi5-v0-baked",
+ "WEIGHTS_BUCKET": "hf://buckets/gemma-challenge/gemma-mikasa-inbound/weights/osoi5-g256",
```

For the full serve.py + all patch files, see [`../vllm-hayai-repro-v1/`](../vllm-hayai-repro-v1)
(copied verbatim). The complete submission as run is on the bucket:
`hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/osoi5-g256-clean/`.

## The recipe — `regroup_g256.py`

Re-groups an osoi5-style int4 pack-quantized checkpoint: de-quantizes each g128 body Linear,
re-quantizes at g256 (halves the bf16 scale-byte overhead, ~3% → ~1.5% of body bytes — a
*prompt-invariant* decode-bandwidth lever), and re-packs. The lm_head (channel-wise) and all
non-quantized tensors are copied untouched. Self-verifies the de-quant round-trip error.

```bash
python regroup_g256.py <src_osoi5_dir> <dst_g256_dir>
```

> ⚠️ **Aux gotcha (cost us a failed run):** the regroup must copy `chat_template.jinja` into the
> destination, or the head-pruned serve dir fails `ChatTemplateResolutionError` at warmup. Fixed
> in this version (it's in the aux-copy list).

## Why it failed (both axes)

1. **Speed: 461 < 506.74.** The kenyan-duma drafter is tuned to the **g128** target argmaxes.
   The coarser g256 body shifts the target's greedy tokens, so MTP acceptance **collapses**
   (accept ratio ≈ **0.66** vs g128's ≈ **0.82** — 461 output / 698 total tok/s). The acceptance
   loss dominates any body-GEMM byte saving → net **slower**.
2. **Validity: PPL 2.5628 > 2.42.** Coarser g256 scales raise body PPL **+0.17** over g128's
   2.394 — far more than the hoped ~+0.02. Invalid.

GPTQ-calibrated g256 would not rescue it: calibration buys ~0.1 PPL (→ ~2.45, still
borderline-invalid) and does **nothing** for the drafter-acceptance collapse. Logged as a
dead-end so the field doesn't re-try it.
