#!/usr/bin/env bash
# Capture int4 argmax targets on A10G. dixie int4 mounted at /dixie, our bucket
# (scripts/corpus/sitecustomize) at /bucket (read-write → write traces straight back).
set -e
export VLLM_USE_V1=1
INT4=/dixie/weights/int4-pck04-16k
PROMPTS="${PROMPTS:-/bucket/killtest/corpus_heldout.jsonl}"
OUT="${OUT:-/bucket/killtest/traces_derisk.jsonl}"
echo "===CHECK-MOUNT==="
ls -la "$INT4" 2>&1 | head -6
echo "===CAPTURE=== prompts=$PROMPTS out=$OUT limit=${CAP_LIMIT:-20}"
PYTHONPATH=/bucket/killtest PCK04_KEEPSET="$INT4/pck04_keepset.json" \
  python3 /bucket/killtest/capture_argmax.py --model "$INT4" \
  --prompts "$PROMPTS" --out "$OUT" --limit "${CAP_LIMIT:-20}" \
  2>&1 | grep -avE "Loading weights|Compressing|it/s\]"
echo "===SAMPLE==="
head -c 220 /bucket/killtest/traces_derisk.jsonl; echo
echo "===CAPTURE DONE==="
