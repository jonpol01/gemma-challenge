---
tps: 224.04
method: triton-mtp-k7-w4a16-v23
status: agent-run
artifacts: submissions/mikasa-inbound/vllm-mtp-w4a16-v23
description: Gemma-4-E4B-it on vLLM 0.23.0 + TRITON_ATTN + MTP speculative decoding (official gemma-4-E4B-it-assistant drafter, K=7) + official W4A16 (gemma-4-E4B-it-qat-w4a16-ct, compressed-tensors). Single-stream A10G.
---
Output token throughput 224.0 tok/s (total 339.2, input 115.2), mean TTFT 2285 ms, ppl 2.159, mean spec-decode acceptance length 4.09. 128/128 requests, A10G, concurrency 1.
