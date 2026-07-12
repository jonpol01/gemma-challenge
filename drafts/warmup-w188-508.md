---
tps: 508.25
ppl: 2.3934
method: w188-ctk49-n64-warmup
status: agent-run
artifacts: noise-507-20260624T050657Z-a
submission: hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/vllm-warmup-w188-ctk49-v1/
description: int4 (W4A16) osoi5-v0-baked + untied 12k pck04 lm_head + kenyan-duma MTP K=7 + N64 synthetic warmup bridge (prompt-agnostic kernel pre-JIT, private-stable) + split-KV verify + FA-sliding window=188 + CENTROID_TOP_K=49 + ONEGRAPH. Output-neutral greedy verify, full multimodal intact. Single-stream a10g-small, 128 prompts x 512 tokens.
---
Output token throughput 508.25 tok/s, ppl 2.3934 (token-level aggregate from summary.json, within the 2.42 validity cap), 128/128 requests, A10G a10g-small, concurrency 1. job 6a3b65f3d530f3857e66e100, run_prefix noise-507-20260624T050657Z-a.
