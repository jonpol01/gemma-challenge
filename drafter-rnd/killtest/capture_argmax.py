#!/usr/bin/env python
"""Greedy-decode prompts on the int4 model (pck04 patch auto-loaded via PYTHONPATH
sitecustomize) and save the target argmax token trajectory per prompt — the drafter
training targets. Chat-templated to match the bench/serve distribution."""
import json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--kv-cache-dtype", default="auto")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    ps = []
    for line in open(a.prompts):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ps.append(d["prompt"] if isinstance(d, dict) else str(d))
    if a.limit:
        ps = ps[:a.limit]
    print(f"[cap] {len(ps)} prompts | model={a.model}", flush=True)

    print(f"[cap] kv_cache_dtype={a.kv_cache_dtype}", flush=True)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=a.gpu_mem, trust_remote_code=True, enforce_eager=True,
              kv_cache_dtype=a.kv_cache_dtype)
    outs = llm.chat([[{"role": "user", "content": p}] for p in ps],
                    SamplingParams(temperature=0, max_tokens=a.max_tokens))
    n = 0
    with open(a.out, "w") as f:
        for p, o in zip(ps, outs):
            f.write(json.dumps({"prompt": p, "target_token_ids": list(o.outputs[0].token_ids)}) + "\n")
            n += 1
    print(f"[cap] CAPTURED {n} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
