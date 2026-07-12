#!/usr/bin/env python
"""env-driven vLLM server: QAT-ct + qat-assistant MTP drafter on our hayai wheel (cheap-lever sweep)."""
import json, os, sys
def main():
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6") or "0")
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id, "--served-model-name", os.environ.get("SERVED_MODEL_NAME","gemma-4-e4b-it"),
            "--host", os.environ.get("HOST","0.0.0.0"), "--port", os.environ.get("PORT","8000"),
            "--dtype","bfloat16", "--max-model-len", os.environ.get("MAX_MODEL_LEN","4096"),
            "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION","0.90"),
            "--max-num-seqs", os.environ.get("MAX_NUM_SEQS","1"),
            "--trust-remote-code","--no-enable-log-requests"]
    mnbt=os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if mnbt: args+=["--max-num-batched-tokens",mnbt]
    pm=os.environ.get("PERFORMANCE_MODE")
    if pm: args+=["--performance-mode",pm]
    if os.environ.get("DISABLE_LOG_STATS")=="1": args+=["--disable-log-stats"]
    if os.environ.get("DISABLE_UVICORN_ACCESS_LOG")=="1": args+=["--disable-uvicorn-access-log"]
    ull=os.environ.get("UVICORN_LOG_LEVEL")
    if ull: args+=["--uvicorn-log-level",ull]
    if num_spec>0 and drafter: args+=["--speculative-config", json.dumps({"model":drafter,"num_speculative_tokens":num_spec})]
    if os.environ.get("ENFORCE_EAGER")=="1": args+=["--enforce-eager"]
    here=os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"]=here+os.pathsep+os.environ.get("PYTHONPATH","")
    print("[serve] launching:"," ".join(args),flush=True)
    os.execvpe(args[0],args,os.environ)
if __name__=="__main__": main()
