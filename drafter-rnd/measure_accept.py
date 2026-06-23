#!/usr/bin/env python
"""Measure spec-decode mean acceptance length for a (target, e1-drafter) pair.

The pck04 lm_head loader patch is applied via sitecustomize.py auto-import:
run with  PYTHONPATH=/mnt/e/gemma/code/pck04  PCK04_KEEPSET=<keepset.json>
for the int4-pck04 target (auto-loads in the main proc AND every spawned vLLM
worker). For the bf16 base target, run WITHOUT those two so no patch applies.

Result line:  [<tag>] RESULT ... mean_accept_len=Y
"""
from __future__ import annotations
import argparse, json, os


def load_prompts(path: str, limit: int) -> list[str]:
    ps: list[str] = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ps.append(d["prompt"] if isinstance(d, dict) else str(d))
        else:
            ps = [l.rstrip("\n") for l in f if l.strip()]
    return ps[:limit] if limit else ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--chat", action="store_true", help="apply chat template (matches bench)")
    ap.add_argument("--quantization", default=None,
                    help="e.g. bitsandbytes for load-time int4; omit for bf16")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = load_prompts(a.prompts, a.limit)
    patch = "on" if os.environ.get("PCK04_KEEPSET") else "off"
    print(f"[{a.tag}] {len(prompts)} prompts | target={a.target} | drafter={a.drafter} "
          f"| quant={a.quantization or 'bf16'} | pck04={patch} | K={a.k} "
          f"max_tok={a.max_tokens} chat={a.chat}", flush=True)

    llm_kwargs = dict(
        model=a.target, dtype="bfloat16", max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem, trust_remote_code=True, enforce_eager=True,
        speculative_config={"method": "mtp", "model": a.drafter, "num_speculative_tokens": a.k},
    )
    if a.quantization:
        llm_kwargs["quantization"] = a.quantization
        if a.quantization == "bitsandbytes":
            llm_kwargs["load_format"] = "bitsandbytes"
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    if a.chat:
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    else:
        outs = llm.generate(prompts, sp)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    vals = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None)
                if v is None:
                    v = getattr(m, "count", None)
                vals[m.name] = v
    except Exception as e:
        print(f"[{a.tag}] get_metrics() error: {e!r}", flush=True)

    if not vals:
        print(f"[{a.tag}] NO spec_decode metrics found; all metric names:", flush=True)
        try:
            for m in llm.get_metrics():
                print("   ", m.name, flush=True)
        except Exception:
            pass

    nd = vals.get("vllm:spec_decode_num_drafts")
    na = vals.get("vllm:spec_decode_num_accepted_tokens")
    ndt = vals.get("vllm:spec_decode_num_draft_tokens")
    Y = (na / nd + 1.0) if nd else float("nan")
    ar = (na / ndt) if ndt else float("nan")
    print(f"[{a.tag}] RESULT prompts={len(prompts)} gen_tokens={gen_tokens} "
          f"num_drafts={nd} accepted={na} draft_tokens={ndt} "
          f"accept_rate={ar:.4f} mean_accept_len={Y:.4f}", flush=True)
    print("[RESULTJSON] " + json.dumps({
        "tag": a.tag, "prompts": len(prompts), "gen_tokens": gen_tokens,
        "num_drafts": nd, "accepted": na, "draft_tokens": ndt,
        "accept_rate": ar, "mean_accept_len": Y,
    }), flush=True)


if __name__ == "__main__":
    main()
