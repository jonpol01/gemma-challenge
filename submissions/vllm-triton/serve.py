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
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    attention_backend = os.environ.get("ATTENTION_BACKEND")
    enable_prefix_caching = os.environ.get("ENABLE_PREFIX_CACHING")

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]
    if attention_backend:
        args += ["--attention-backend", attention_backend]
    if enable_prefix_caching:
        args += ["--enable-prefix-caching"]
    # enforce_eager disables CUDA-graph capture — skips the warm-up step where
    # FLASHINFER crashes with 'Unsupported max_mma_kv: 0' on this model.
    if os.environ.get("ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        args += ["--enforce-eager"]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
