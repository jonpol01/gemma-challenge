#!/usr/bin/env bash
# EAGLE drafter: smoke (match-rate) or train. bf16 target loads fast in HF (int4 ≈ bf16
# by quant design); same EAGLE wiring. Target downloaded in-job, drafter too.
set -e
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>/dev/null || true
TARGET="${TARGET:-google/gemma-4-E4B-it}"
DRAFTER="${DRAFTER:-google/gemma-4-E4B-it-assistant}"
TRACES="${TRACES:-/bucket/killtest/traces_full.jsonl}"
MODE="${MODE:---smoke}"; [ "$MODE" = "train" ] && MODE=""   # MODE=train -> no --smoke flag -> training
LIMIT="${LIMIT:-10}"; MAXPOS="${MAXPOS:-20}"; OUT="${OUT:-}"; ACCUM="${ACCUM:-16}"; LR="${LR:-2e-4}"
EXTRA=""; [ -n "$OUT" ] && EXTRA="--out $OUT"
echo "===CHECK==="; ls -la "$TRACES" 2>&1 | head -2; echo "target=$TARGET drafter=$DRAFTER"
echo "===EAGLE $MODE=== limit=$LIMIT maxpos=$MAXPOS out='$OUT' accum=$ACCUM"
python3 /bucket/killtest/train_eagle.py --int4 "$TARGET" --drafter "$DRAFTER" \
  --tokenizer "$TARGET" --traces "$TRACES" $MODE --limit "$LIMIT" --max-pos "$MAXPOS" \
  --accum "$ACCUM" --lr "$LR" $EXTRA \
  2>&1 | grep --line-buffered -avE "it/s\]|[0-9]+%\|"
echo "===EAGLE DONE==="