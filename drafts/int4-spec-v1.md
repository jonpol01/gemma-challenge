---
tps: 507.34
ppl: 2.4074
method: int4-pck04-12k-splitkv-w160-ctk44-mtp-k7
status: agent-run
artifacts: submissions/mikasa-inbound/vllm-int4-spec-v1
submission: hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/vllm-int4-spec-v1/
description: int4 (W4A16) osoi5-v0-baked with an untied, vocabulary-pruned LM head (262144 -> 12288 rows via logits-scatter that restores full-vocabulary token positions). TRITON attention with a 160-token sliding window + custom split-KV verification kernel; single-graph decode capture with fused sparse-argmax; CENTROID_TOP_K=44; multi-token speculative decoding (K=7, kenyan-duma fine-tuned drafter) under output-neutral greedy verification. Full text+image+audio modalities intact. Single-stream a10g-small, 128 prompts x 512 output tokens.
---
Output token throughput 507.34 tok/s (total 768.13 tok/s), ppl 2.4074 (token-level aggregate from summary.json, within the 2.42 validity cap), 128/128 requests, A10G a10g-small, concurrency 1. job 6a3819323093dba73ce2b7d4, run_prefix runs/submissions-mikasa-inbound-vllm-int4-spec-v1-20260621T170238Z.
