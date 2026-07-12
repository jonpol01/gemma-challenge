#!/usr/bin/env bash
# Greedy-identity gate for fp8_e4m3 KV cache: does it change int4 greedy tokens vs default KV?
# Reference = traces_derisk.jsonl (int4 + default/bf16 KV greedy on heldout[:20]).
set -e
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
INT4=/dixie/weights/int4-pck04-16k
PROMPTS=/bucket/killtest/corpus_heldout.jsonl
echo "===CAPTURE fp8_e4m3 KV (heldout[:20]) ==="
PYTHONPATH=/bucket/killtest PCK04_KEEPSET="$INT4/pck04_keepset.json" \
  python3 /bucket/killtest/capture_argmax.py --model "$INT4" \
  --prompts "$PROMPTS" --out /tmp/fp8.jsonl --limit 20 --kv-cache-dtype fp8_e4m3 \
  2>&1 | grep -avE "it/s\]|[0-9]+%\|" | tail -5
echo "===DIFF vs default-KV reference (traces_derisk) ==="
python3 - <<'PY'
import json
ref=[json.loads(l) for l in open("/bucket/killtest/traces_derisk.jsonl")]
fp8=[json.loads(l) for l in open("/tmp/fp8.jsonl")]
n=min(len(ref),len(fp8)); ident=0; first=None
for i in range(n):
    a=ref[i]["target_token_ids"]; b=fp8[i]["target_token_ids"]
    if a==b: ident+=1
    elif first is None:
        for j in range(min(len(a),len(b))):
            if a[j]!=b[j]: first=(i,j,len(a),len(b)); break
        else: first=(i,"len_diff",len(a),len(b))
print(f"IDENTITYJSON {json.dumps({'prompts':n,'identical':ident,'divergent':n-ident,'first_divergence':first})}")
PY
echo "===FP8 IDENTITY DONE==="