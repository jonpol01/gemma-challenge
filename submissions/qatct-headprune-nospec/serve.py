#!/usr/bin/env python
"""Phase-1 quality-safe speed push: faithful QAT-ct body + IN-MEMORY head-prune.

The QAT-ct checkpoint is served completely UNMODIFIED (tied lm_head, full vocab) —
exactly the stack that loads cleanly at 229 tps. serve_patch_inmem (auto-imported
by sitecustomize) then, *after* the tied weights are resident on GPU, builds a
separate ParallelLMHead(K) from embed_tokens.weight[keep_ids] and scatters the
[M,K] logits back to full vocab. The faithful 42-layer int4 body is untouched
(capability); only the output head shrinks 262k->12k (~26% of decode) for speed.
Greedy-lossless MTP speculation via the Google QAT-matched assistant drafter.
"""
import json
import os
import sys

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
DRAFTER = os.environ.get(
    "DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
)
SRC = os.environ.get("LOCAL_MODEL_DIR", "/tmp/qatct")


def prep() -> None:
    """Download the UNMODIFIED model — no untie, no shard, no config edits."""
    if os.path.exists(os.path.join(SRC, "config.json")):
        print(f"[prep] reusing {SRC}", flush=True)
        return
    from huggingface_hub import snapshot_download
    print(f"[prep] downloading {MODEL} -> {SRC} (unmodified)", flush=True)
    snapshot_download(MODEL, local_dir=SRC)


def main() -> None:
    prep()
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PCK04_KEEPSET"] = os.path.join(here, "pck04_keepset.json")
    os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")

    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6") or "0")
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", SRC,
        "--served-model-name", os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host", os.environ.get("HOST", "0.0.0.0"),
        "--port", os.environ.get("PORT", "8000"),
        "--dtype", "bfloat16",
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "1"),
        "--trust-remote-code", "--no-enable-log-requests",
    ]
    mnbt = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if mnbt:
        args += ["--max-num-batched-tokens", mnbt]
    if num_spec > 0:
        args += ["--speculative-config", json.dumps({"model": DRAFTER, "num_speculative_tokens": num_spec})]
    print("[serve] launching:", " ".join(args), flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
