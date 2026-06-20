---
tps: 287.64
method: triton-mtp-k7-pck04-dixie16k
status: agent-run
artifacts: submissions/mikasa-inbound/vllm-pck04-dixie16k-v1
description: Gemma-4-E4B-it on vLLM 0.23.0 + TRITON_ATTN + MTP speculative decoding (official gemma-4-E4B-it-assistant drafter, K=7) + dixie-flatline int4-pck04-16k (int4 body + untied/pruned int4 lm_head, K=16384) loaded via a pck04 logits-scatter sitecustomize patch. Single-stream A10G.
---
Output token throughput 287.64 tok/s (total 435.49 tok/s), mean_record_ppl 2.1506, token-level ppl 2.00. 128/128 requests, A10G, concurrency 1. job 6a365a29953ed90bfb945941.
