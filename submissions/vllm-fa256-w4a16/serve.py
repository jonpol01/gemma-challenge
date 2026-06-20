#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    # Put this submission dir on PYTHONPATH so our sitecustomize.py (the per-layer
    # FlashAttention patch) is auto-imported by EVERY interpreter spawned from here —
    # crucially the vLLM EngineCore subprocess, where attention-backend selection runs.
    subdir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = subdir + os.pathsep + os.environ.get("PYTHONPATH", "")

    model_id = (
        os.environ.get("SERVE_MODEL_OVERRIDE")
        or os.environ.get("MODEL_ID")
        or "google/gemma-4-E4B-it"
    )
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS")
    attention_backend = os.environ.get("ATTENTION_BACKEND")
    enable_prefix_caching = os.environ.get("ENABLE_PREFIX_CACHING")
    speculative_config = os.environ.get("SPECULATIVE_CONFIG")
    quantization = os.environ.get("QUANTIZATION")

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
    if max_num_seqs:
        args += ["--max-num-seqs", max_num_seqs]
    if attention_backend:
        args += ["--attention-backend", attention_backend]
    if enable_prefix_caching:
        args += ["--enable-prefix-caching"]
    if quantization:
        args += ["--quantization", quantization]
    if speculative_config:
        args += ["--speculative-config", speculative_config]
    if os.environ.get("ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        args += ["--enforce-eager"]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
