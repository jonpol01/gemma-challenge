#!/usr/bin/env python3
"""Single-stream served-tok/s A/B for a spec-decode MTP drafter. Offline vLLM LLM()
with speculative_config over the int4 target; greedy-decode the eval prompts and report
output tok/s. Relative measure to compare drafters before spending an official run.
pck04 patch auto-loads via PYTHONPATH sitecustomize (lm_head K=16384)."""
import json, argparse, time


def load_prompts(path, limit):
    ps = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ps.append(d["prompt"] if isinstance(d, dict) else str(d))
    return ps[:limit] if limit else ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", default="X")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    ps = load_prompts(a.prompts, a.limit)
    print(f"[bench] tag={a.tag} drafter={a.drafter} n={len(ps)} k={a.k} eager={a.eager}", flush=True)
    spec = {"method": "mtp", "model": a.drafter, "num_speculative_tokens": a.k}
    llm = LLM(model=a.int4, speculative_config=spec, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=a.gpu_mem, trust_remote_code=True, max_num_seqs=1,
              enforce_eager=a.eager)
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    msgs = [[{"role": "user", "content": p}] for p in ps]
    llm.chat(msgs[:1], sp)                       # warmup (graph capture / settle)
    t0 = time.time()
    outs = llm.chat(msgs, sp)
    dt = time.time() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"tag": a.tag, "drafter": a.drafter, "k": a.k,
           "tps": round(tot / dt, 2), "tokens": tot, "sec": round(dt, 2), "n": len(ps)}
    print("RESULTJSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
