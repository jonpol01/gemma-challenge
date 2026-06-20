#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    # SERVE_MODEL_OVERRIDE wins over MODEL_ID so the manifest can keep the
    # canonical model_id (google/gemma-4-E4B-it) for identity/validation while we
    # actually load an optimized *form* of it (e.g. the official W4A16 checkpoint).
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
    # MTP speculative decoding — pass the JSON straight through as one argv
    # element (execvpe, not a shell, so no quoting issues). e.g.
    # {"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":7}
    if speculative_config:
        args += ["--speculative-config", speculative_config]
    # enforce_eager disables CUDA-graph capture (only needed for FLASHINFER, which
    # is unusable here anyway). Left off by default so TRITON_ATTN keeps graphs.
    if os.environ.get("ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        args += ["--enforce-eager"]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
