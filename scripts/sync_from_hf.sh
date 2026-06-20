#!/usr/bin/env bash
# Re-pull our HF bucket (gemma-challenge/gemma-mikasa-inbound) into this repo:
#   submissions/  results/ (minus bulky raw dumps)  drafts/  data/ snapshots
# Uses a local `hf` CLI if on PATH, else the `hermes` docker container.
set -euo pipefail

BUCKET="hf://buckets/gemma-challenge/gemma-mikasa-inbound"
API="https://gemma-challenge-gemma-bucket-sync.hf.space/v1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if command -v hf >/dev/null 2>&1; then
  HF() { hf "$@"; }
  STAGE="$(mktemp -d)"; HFSTAGE="$STAGE"
  trap 'rm -rf "$STAGE"' EXIT
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx hermes; then
  # container can only write under /opt/data, which is bind-mounted to ~/.hermes
  HF() { docker exec hermes env HOME=/opt/data/profiles/local/home /opt/hermes/.venv/bin/hf "$@"; }
  STAGE="$HOME/.hermes/tmp/mikasa-sync-$$"; HFSTAGE="/opt/data/tmp/mikasa-sync-$$"
  mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
else
  echo "error: need a local 'hf' CLI or a running 'hermes' container" >&2; exit 1
fi

echo ">> pulling $BUCKET"
HF buckets sync "$BUCKET" "$HFSTAGE" >/dev/null

echo ">> laying out submissions/ results/ drafts/"
mkdir -p "$REPO/submissions" "$REPO/results" "$REPO/drafts"
rsync -a "$STAGE/submissions/mikasa-inbound/" "$REPO/submissions/"
[ -d "$STAGE/drafts" ] && rsync -a "$STAGE/drafts/" "$REPO/drafts/"
# benchmark-output dirs live at the bucket root and under results/mikasa-inbound/
for d in "$STAGE"/*/; do
  b="$(basename "$d")"
  case "$b" in submissions|results|drafts) continue;; esac
  if [ -f "${d}job_status.json" ] || [ -f "${d}summary.json" ]; then
    rsync -a --exclude 'decode_outputs.jsonl' --exclude 'benchmark.jsonl' "$d" "$REPO/results/$b/"
  fi
done
[ -d "$STAGE/results/mikasa-inbound" ] && \
  rsync -a --exclude 'decode_outputs.jsonl' --exclude 'benchmark.jsonl' "$STAGE/results/mikasa-inbound/" "$REPO/results/"

echo ">> refreshing data/ leaderboard snapshots"
if T="$(HF auth token 2>/dev/null)" && [ -n "$T" ]; then
  for q in "verification=valid&best_per_agent=true&limit=100:leaderboard_valid" "best_per_agent=true&limit=100:leaderboard_all"; do
    url="${q%%:*}"; name="${q##*:}"
    curl -s "$API/leaderboard?$url" -H "Authorization: Bearer $T" -o "$REPO/data/$name.json" || true
  done
fi

echo ">> done — review with: git -C \"$REPO\" status"
