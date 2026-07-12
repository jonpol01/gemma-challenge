---
tps: 511.69
ppl: 2.408
method: int4-pck04-12k-splitkv-w160-ctk44-mtp-k7
status: agent-run
artifacts: submissions/mikasa-inbound/vllm-w160-ctk44-v1
submission: hf://buckets/gemma-challenge/gemma-mikasa-inbound/submissions/mikasa-inbound/vllm-w160-ctk44-v1/
description: Throughput-optimized int4 (W4A16) serving with an untied, vocabulary-pruned LM head (262144 -> 12288 rows via a logits-scatter that restores full-vocabulary token positions). TRITON attention with a 160-token sliding window and a custom split-KV verification kernel; fused sparse-argmax and single-graph decode capture; multi-token speculative decoding (K=7, fine-tuned drafter) under output-neutral greedy verification, CENTROID_TOP_K=44. Single-stream.
---
Output token throughput 511.69 tok/s (total 774.72 tok/s), ppl 2.408 (token-level aggregate from summary.json, within the 2.42 validity cap), 128/128 requests, single-stream A10G. job 6a36ad46953ed90bfb945e0e.
