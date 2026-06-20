#!/usr/bin/env python
"""Serve chiku-inu's shared osoi5-v0-baked (int4 g128 + untied-int4 lm_head) with our
TRITON_ATTN + MTP config. Syncs the 9GB model in-job from its bucket, and puts this dir on
PYTHONPATH so sitecustomize.py (the lm_head loader patch) loads in the EngineCore subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _log(m: str) -> None:
    print("[serve] " + str(m), file=sys.stderr, flush=True)


def main() -> None:
    subdir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = subdir + os.pathsep + os.environ.get("PYTHONPATH", "")

    weights_bucket = os.environ.get("WEIGHTS_BUCKET")
    model_id = os.environ.get("SERVE_MODEL_OVERRIDE") or os.environ.get("MODEL_ID") or "google/gemma-4-E4B-it"
    if weights_bucket:
        local = "/tmp/osoi5-model"
        _log("syncing weights %s -> %s" % (weights_bucket, local))
        rc = 1
        for cli in (["hf", "buckets", "sync"], ["huggingface-cli", "buckets", "sync"]):
            try:
                rc = subprocess.run(cli + [weights_bucket, local]).returncode
            except FileNotFoundError:
                continue
            if rc == 0:
                break
        if rc != 0:
            _log("weight sync FAILED rc=%s — aborting" % rc)
            sys.exit(1)
        model_id = local
        _log("model set to %s" % model_id)

    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", served_model_name,
        "--host", os.environ.get("HOST", "0.0.0.0"),
        "--port", os.environ.get("PORT", "8000"),
        "--dtype", "bfloat16",
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if os.environ.get("MAX_NUM_BATCHED_TOKENS"):
        args += ["--max-num-batched-tokens", os.environ["MAX_NUM_BATCHED_TOKENS"]]
    if os.environ.get("MAX_NUM_SEQS"):
        args += ["--max-num-seqs", os.environ["MAX_NUM_SEQS"]]
    if os.environ.get("ATTENTION_BACKEND"):
        args += ["--attention-backend", os.environ["ATTENTION_BACKEND"]]
    if os.environ.get("ENABLE_PREFIX_CACHING"):
        args += ["--enable-prefix-caching"]
    if os.environ.get("SPECULATIVE_CONFIG"):
        args += ["--speculative-config", os.environ["SPECULATIVE_CONFIG"]]
    _log("launching vllm: model=%s backend=%s (PYTHONPATH has loader patch)" % (model_id, os.environ.get("ATTENTION_BACKEND")))
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
