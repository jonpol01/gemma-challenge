#!/usr/bin/env bash
# Kill-test on A10G. Models are DOWNLOADED in-job (mounting the gated 16GB model
# hung container init), scripts/corpus come from the small bucket mount at /bucket.
set -e
export VLLM_USE_V1=1
echo "===SETUP==="
pip install -q llmcompressor 2>&1 | tail -1
echo "===DOWNLOAD-MODELS==="
hf download google/gemma-4-E4B-it --local-dir /tmp/bf16 2>&1 | tail -1
hf download google/gemma-4-E4B-it-assistant --local-dir /tmp/drafter 2>&1 | tail -1
echo "===QUANT==="
python3 /bucket/killtest/quantize_job.py /tmp/bf16 /tmp/int4 2>&1 | grep -avE "Loading weights|Compressing model|it/s\]"
echo "===INT4==="
python3 /bucket/killtest/measure_accept.py --target /tmp/int4 --drafter /tmp/drafter \
  --prompts /bucket/killtest/corpus_heldout.jsonl --chat --limit 150 --max-tokens 256 \
  --gpu-mem 0.9 --tag INT4 2>&1 | grep -aE "RESULTJSON|\] RESULT|Error|error:|Traceback|NO spec"
echo "===BF16==="
python3 /bucket/killtest/measure_accept.py --target /tmp/bf16 --drafter /tmp/drafter \
  --prompts /bucket/killtest/corpus_heldout.jsonl --chat --limit 150 --max-tokens 256 \
  --gpu-mem 0.9 --tag BF16 2>&1 | grep -aE "RESULTJSON|\] RESULT|Error|error:|Traceback|NO spec"
echo "===KILLTEST DONE==="
