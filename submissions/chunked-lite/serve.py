#!/usr/bin/env python
from __future__ import annotations
import os
import sys

def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.92")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS", "1024")
    attention_backend = os.environ.get("ATTENTION_BACKEND", "FLASHINFER")
    enable_prefix_caching = os.environ.get("ENABLE_PREFIX_CACHING", "1")

    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", port,
        "--dtype", "bfloat16",
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", max_num_batched_tokens,
        "--attention-backend", attention_backend,
    ]
    if enable_prefix_caching == "1":
        args += ["--enable-prefix-caching"]
    os.execvpe(args[0], args, os.environ)

if __name__ == "__main__":
    main()
