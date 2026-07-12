---
tps: 461.01
ppl: 2.5628
method: osoi5-g256-coarsequant
status: negative
description: "osoi5 int4 body g128 to g256 on the 506.74 hayai stack regresses to 461 tps (drafter collapse) and ppl 2.56 (over 2.42 cap, invalid)"
artifacts: results/mikasa-inbound/osoi5-g256-clean-20260622T215734Z/
submission: hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/osoi5-g256-clean/
---

## osoi5 g256 coarser-quant — negative result

Re-grouped osoi5's int4 body **g128 -> g256** (halve the bf16 scale-byte overhead, a prompt-invariant decode-bandwidth lever) and served on the verified 506.74 hayai stack unchanged: head-prune 16k->12k, kenyan-duma drafter (K=7 MTP), sliding-window 192, split-KV-verify / FA-sliding / ONEGRAPH / fused-sparse-argmax. Benched on a10g-small.

**Result: regresses on BOTH axes.**
- **tps 461.01** (vs 506.74 at g128). The kenyan-duma drafter is tuned to the g128 target argmaxes; the coarser g256 body shifts the target greedy tokens, so draft acceptance collapses — accept ratio **0.66** (461/698 total_tps) vs g128's ~0.82. The acceptance loss dominates any body-GEMM saving, so net **slower**.
- **ppl 2.5628** (cap 2.42). Coarser g256 scales raise body ppl **+0.17** over g128's 2.394, over the cap, **invalid**.

GPTQ-calibrated g256 would not rescue it: calibration lowers ppl only ~0.1 (to ~2.45, still borderline-invalid) and does nothing for the drafter-acceptance collapse. **Conclusion: g256 coarser-quant does not beat 506 — it regresses both speed and validity.** Logged as a dead-end so the field does not re-try it. (Aux: the regroup must copy chat_template.jinja or the head-pruned serve dir fails ChatTemplateResolution at warmup.)
