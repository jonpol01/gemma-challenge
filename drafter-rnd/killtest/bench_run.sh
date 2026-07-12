#!/usr/bin/env bash
# tok/s A/B: serve int4 (vLLM, fast Marlin) + each drafter via spec-decode, measure tok/s.
# Each drafter in a fresh python3 process = clean vLLM/GPU state. pck04 patch for int4 lm_head.
set -e
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
INT4=/dixie/weights/int4-pck04-16k
PROMPTS="${PROMPTS:-/bucket/killtest/corpus_heldout.jsonl}"
LIMIT="${LIMIT:-64}"; MAXTOK="${MAXTOK:-256}"; K="${K:-7}"
DRAFTERS="${DRAFTERS:-stock=google/gemma-4-E4B-it-assistant mine=/bucket/killtest/drafter-eagle/final}"
echo "===CHECK==="; ls "$INT4/pck04_keepset.json" "$PROMPTS" 2>&1 | head
for pair in $DRAFTERS; do
  tag="${pair%%=*}"; path="${pair#*=}"
  echo "===BENCH $tag ($path) k=$K limit=$LIMIT==="
  PYTHONPATH=/bucket/killtest PCK04_KEEPSET="$INT4/pck04_keepset.json" \
    python3 /bucket/killtest/bench_tps.py --int4 "$INT4" --drafter "$path" \
    --prompts "$PROMPTS" --tag "$tag" --limit "$LIMIT" --max-tokens "$MAXTOK" --k "$K" \
    2>&1 | grep -aE "RESULTJSON|\[bench\]|Error|Traceback|CUDA|OOM|out of memory|ReplicatedLinear" || echo "  (bench $tag failed)"
done
echo "===BENCH DONE==="