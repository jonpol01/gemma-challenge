#!/usr/bin/env bash
# Build CE traces from captured int4 trajectories, then train the MTP drafter.
# /dixie :ro (int4 tokenizer + chat template). /bucket rw (traces in, drafter out).
set -e
DIXIE_INT4=/dixie/weights/int4-pck04-16k
TRACES_IN="${TRACES_IN:-/bucket/killtest/traces_full.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-/bucket/killtest/train_traces.jsonl}"
DRAFT_INIT="${DRAFT_INIT:-google/gemma-4-E4B-it-assistant}"
DRAFT_REV="${DRAFT_REV:-}"
OUT="${OUT:-/bucket/killtest/drafter-ce}"
LIMIT="${LIMIT:-0}"; MAXPOS="${MAXPOS:-0}"; BSZ="${BSZ:-32}"; LR="${LR:-2e-4}"
GC="${GC:---grad-checkpoint}"; FREEZE="${FREEZE:-}"

echo "===CHECK-MOUNT==="
ls "$DIXIE_INT4/tokenizer_config.json" "$TRACES_IN" 2>&1 | head
echo "===BUILD-TRACES=== limit=$LIMIT max_pos=$MAXPOS"
python3 /bucket/killtest/build_traces.py --tokenizer "$DIXIE_INT4" \
  --in-file "$TRACES_IN" --out "$TRAIN_JSONL" --limit "$LIMIT" --max-pos "$MAXPOS"
wc -l "$TRAIN_JSONL"
REVARG=""; [ -n "$DRAFT_REV" ] && REVARG="--init-revision $DRAFT_REV"
echo "===TRAIN=== init=$DRAFT_INIT rev='${DRAFT_REV:-main}' bsz=$BSZ lr=$LR gc='$GC' freeze='$FREEZE'"
python3 /bucket/killtest/train_ce.py --init "$DRAFT_INIT" $REVARG \
  --corpus "$TRAIN_JSONL" --out "$OUT" --batch-size "$BSZ" --lr "$LR" \
  --workers 2 --log-every 50 $GC $FREEZE \
  2>&1 | grep -avE "Loading checkpoint|Downloading|it/s\]|[0-9]+%\|"
echo "===TRAIN DONE==="
ls -la "$OUT/final" 2>&1 | head