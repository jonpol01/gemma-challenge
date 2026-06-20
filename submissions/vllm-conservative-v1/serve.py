#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")

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
        os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.88"),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    max_batched = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if max_batched:
        args += ["--max-num-batched-tokens", max_batched]

    if os.environ.get("ATTENTION_BACKEND"):
        args += ["--attention-backend", os.environ["ATTENTION_BACKEND"]]

    if os.environ.get("ENABLE_PREFIX_CACHING"):
        args += ["--enable-prefix-caching"]

    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()