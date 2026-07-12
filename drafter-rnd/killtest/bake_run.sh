#!/usr/bin/env bash
# Bake a 36-layer osoi5 (drop one sliding layer) and upload to our bucket. CPU surgery.
set -e
export PYTHONUNBUFFERED=1
OSOI=/osoi5
DST=/tmp/osoi5-36
DROP="${DROP:-33}"
OUTBUCKET="hf://buckets/gemma-challenge/gemma-mikasa-inbound/weights/osoi5-36L-drop${DROP}"
echo "===CHECK==="; ls "$OSOI/model.safetensors" "$OSOI/config.json" 2>&1 | head
echo "===BAKE drop=$DROP==="
python3 /bucket/killtest/bake_36.py "$OSOI" "$DST" "$DROP" 2>&1 | grep -avE "it/s\]|%\|"
echo "===local result==="; ls -la "$DST" 2>&1 | grep -iE "safetensors|config|index|keepset" | head
echo "===CLEAN stale sharded files in bucket==="
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json; do
  hf buckets rm "$OUTBUCKET/$f" 2>/dev/null && echo "  removed $f" || true
done
echo "===UPLOAD -> $OUTBUCKET==="
hf buckets sync "$DST" "$OUTBUCKET" 2>&1 | tail -3
echo "===VERIFY uploaded==="
hf buckets ls "$OUTBUCKET" 2>&1 | grep -iE "safetensors|config|index" | head
echo "===BAKE+UPLOAD DONE==="