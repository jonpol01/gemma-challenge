#!/usr/bin/env python
"""A10G bake Job 1: download bf16 base -> prune 42->37 (osoi5 structure) -> bf16 PPL -> upload.

Runs in vllm/vllm-openai with /bake mounted read-only (this script + prune_bake.py +
ppl_quick.py + osoi5-config.json + ppl_ground_truth_tokens.jsonl). Diagnostic: the bf16
PPL on the 128-record set should land BELOW osoi5's 2.394 (no quant error yet); a sane
value (~2.0-2.4) confirms the layer selection {1,2,3,37,38} holds. Uploads the bf16-37L
checkpoint so GPTQ (Job 2) pulls it from the bucket without re-downloading the 15 GB base.
"""
import os
import subprocess
import sys

BAKE = "/bake"
BASE = "/tmp/base"
D37 = "/tmp/bf16-37L"
OUT_BUCKET = "hf://buckets/gemma-challenge/gemma-mikasa-inbound/weights/bf16-37L-v1"


def run(cmd, **kw):
    print("::RUN::", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


run([sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers>=5.9", "safetensors", "hf_xet"])

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import snapshot_download  # noqa: E402

print("::STAGE:: download base google/gemma-4-E4B-it", flush=True)
snapshot_download("google/gemma-4-E4B-it", local_dir=BASE, max_workers=8)

print("::STAGE:: prune 42->37", flush=True)
run([sys.executable, f"{BAKE}/prune_bake.py"],
    env=dict(os.environ, SRC=BASE, DST=D37, OSOI5_CFG=f"{BAKE}/osoi5-config.json"))

# upload FIRST so the prune is banked even if the ppl step is tight on T4 VRAM
print("::STAGE:: upload bf16-37L ->", OUT_BUCKET, flush=True)
run(["hf", "buckets", "sync", D37, OUT_BUCKET])

print("::STAGE:: bf16 PPL (target: well below osoi5's 2.394)", flush=True)
run([sys.executable, f"{BAKE}/ppl_quick.py"],
    env=dict(os.environ, MODEL=D37, PPL_SET=f"{BAKE}/ppl_ground_truth_tokens.jsonl"))
print("::DONE:: BAKE_JOB1_COMPLETE", flush=True)
