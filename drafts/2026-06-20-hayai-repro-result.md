---
tps: 506.74
ppl: 2.394
method: hayai-repro-splitkv-w192-ctk48-12k-mtp-k7
status: agent-run
artifacts: submissions/mikasa-inbound/vllm-hayai-repro-v1
submission: hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/vllm-hayai-repro-v1/
description: Gemma-4-E4B-it on a10g-small. Reproduction of firfir-cast's shared hayai-ctk48-w192-noprecache stack — custom vLLM wheel + split-KV verify + FA-sliding + ONEGRAPH/loopgraph + fused-sparse-argmax(block64) + CENTROID_TOP_K=48 + sliding_window=192 + in-job 16k->12k lm_head re-prune (dixie int4-pck04c-12k keepset) + kenyan-duma fine-tuned MTP drafter (K=7) on osoi5-v0-baked int4. Output-neutral greedy verify. Single-stream, concurrency 1.
---
Output token throughput 506.74 tok/s (total 767.23 tok/s), ppl 2.394 (token-level aggregate from summary.json, within the 2.42 validity cap), 128/128 requests, A10G a10g-small, concurrency 1. job 6a3666333093dba73ce2ad10.
