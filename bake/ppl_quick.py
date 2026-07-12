#!/usr/bin/env python
"""Quick token-level PPL of a (multimodal) gemma-4 bake on the 128-record ppl set.

Loads ONLY the text submodule: builds Gemma4ForCausalLM from text_config and loads
the renamed `model.language_model.*` -> `model.*` weights (the validated text path),
so it works regardless of the multimodal wrapper / whether GPTQ ran. Token-level
PPL = exp(sum NLL / sum scored tokens), matching the harness's `summary.json.ppl`.

Usage: MODEL=<dir> [PPL_SET=ppl_ground_truth_tokens.jsonl] python ppl_quick.py
"""
import json
import math
import os

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, Gemma4TextConfig

MODEL = os.environ["MODEL"]
PPL_SET = os.environ.get("PPL_SET", "ppl_ground_truth_tokens.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(os.path.join(MODEL, "config.json")))
tcfg = Gemma4TextConfig(**cfg["text_config"])
tcfg.tie_word_embeddings = cfg.get("tie_word_embeddings", True)
model = AutoModelForCausalLM.from_config(tcfg).to(torch.bfloat16)

LM = "model.language_model."
sd = {}
with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith(LM):
            sd["model." + k[len(LM):]] = f.get_tensor(k)
miss, unexp = model.load_state_dict(sd, strict=False)
miss = [m for m in miss if "lm_head" not in m]  # lm_head is tied -> expected absent
print("LOAD miss=%d unexp=%d" % (len(miss), len(unexp)))
if miss:
    print("  MISSING:", miss[:8])
if unexp:
    print("  UNEXPECTED:", unexp[:8])
model = model.to(DEV).eval()

tot_nll = 0.0
tot_tok = 0
with open(PPL_SET) as fh:
    for line in fh:
        r = json.loads(line)
        ctx, tgt = r["context_token_ids"], r["target_token_ids"]
        ids = torch.tensor([ctx + tgt], device=DEV)
        with torch.no_grad():
            logits = model(ids).logits[0]
        ls = len(ctx)
        lp = torch.log_softmax(logits[ls - 1: ls - 1 + len(tgt)].float(), dim=-1)
        tg = torch.tensor(tgt, device=DEV)
        tot_nll += -lp.gather(-1, tg[:, None]).squeeze(-1).sum().item()
        tot_tok += len(tgt)

print("PPL %.4f  tokens %d  records-scored" % (math.exp(tot_nll / tot_tok), tot_tok))
