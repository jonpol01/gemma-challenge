#!/usr/bin/env python
"""Prune google/gemma-4-E4B-it (bf16 multimodal) -> 37-layer text-pruned bf16.

Drops 5 sliding text layers (default {1,2,3,37,38}), reproducing osoi5's exact
layer_types (full-attention at {2,8,14,20,26,32,36}) and num_kv_shared_layers=16,
with NO source<->sharer status flip. Validated on a tiny random gemma-4: pruned
checkpoint reloads with 0 missing / 0 unexpected keys and forwards cleanly.

gemma-4 KV-share recap (modeling_gemma4.py): first_kv_shared = num_hidden_layers
- num_kv_shared_layers; layers >= it are SHARERS (no k/v proj — reuse the last
source layer's K/V, keyed by attention type). Dropping 3 front sources (idx<24)
takes sources 24->21; dropping 2 back sharers takes sharers 18->16. Sharers keep
only q_proj/q_norm/o_proj; we strip their k_proj/v_proj/k_norm.

Towers: kept as-is (bf16), exactly like osoi5 (whose quant ignore-list excludes
every vision/audio Linear). PLE tensors (embed_tokens_per_layer,
per_layer_model_projection) are sliced on the layer axis 42->37.

Output config = osoi5's config minus quantization_config (this artifact is bf16;
GPTQ adds quant later), with tie_word_embeddings taken from the base (the base is
tied -> the served 12k head is built in-memory from embed_tokens, as in the 333 stack).

Usage:
  SRC=<base dir> DST=<out dir> OSOI5_CFG=<osoi5 config.json> [DROP=1,2,3,37,38] python prune_bake.py
"""
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ["SRC"]
DST = os.environ["DST"]
OSOI5_CFG = os.environ["OSOI5_CFG"]
DROP = sorted(int(x) for x in os.environ.get("DROP", "1,2,3,37,38").split(","))
DROPSET = set(DROP)

LM = "model.language_model."
N_ORIG = 42
KEEP = [i for i in range(N_ORIG) if i not in DROPSET]
N_NEW = len(KEEP)
NEW_OF = {o: n for n, o in enumerate(KEEP)}

# new structure from osoi5 config (the proven template)
cfg = json.load(open(OSOI5_CFG))
tc = cfg["text_config"]
HSPL = tc["hidden_size_per_layer_input"]
NEW_NKVS = tc["num_kv_shared_layers"]
NEW_FIRST_SHARED = N_NEW - NEW_NKVS
STRIP = (".self_attn.k_proj.", ".self_attn.v_proj.",
         ".self_attn.k_norm.", ".self_attn.v_norm.")

assert N_NEW == tc["num_hidden_layers"], (N_NEW, tc["num_hidden_layers"])
assert tc["layer_types"] == [json.load(open(OSOI5_CFG))["text_config"]["layer_types"][i]
                             for i in range(N_NEW)], "osoi5 cfg sanity"

os.makedirs(DST, exist_ok=True)
src_st = os.path.join(SRC, "model.safetensors")
out = {}
n_layer = n_strip = n_other = 0
with safe_open(src_st, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith(LM + "layers."):
            rest = k[len(LM + "layers."):]
            li = int(rest.split(".")[0])
            if li in DROPSET:
                continue
            ni = NEW_OF[li]
            sub = rest[len(str(li)) + 1:]
            if ni >= NEW_FIRST_SHARED and any(s in "." + sub for s in STRIP):
                n_strip += 1
                continue
            out[f"{LM}layers.{ni}.{sub}"] = f.get_tensor(k)
            n_layer += 1
        elif k == LM + "embed_tokens_per_layer.weight":
            t = f.get_tensor(k)               # [vocab, N_ORIG*HSPL]
            v = t.shape[0]
            out[k] = t.view(v, N_ORIG, HSPL)[:, KEEP, :].reshape(v, N_NEW * HSPL).contiguous()
        elif k == LM + "per_layer_model_projection.weight":
            t = f.get_tensor(k)               # [N_ORIG*HSPL, hidden]
            h = t.shape[1]
            out[k] = t.view(N_ORIG, HSPL, h)[KEEP, :, :].reshape(N_NEW * HSPL, h).contiguous()
        else:
            out[k] = f.get_tensor(k)          # towers, embed_tokens, norm, per_layer_projection_norm
            n_other += 1

save_file(out, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})

# output config: osoi5 minus quant; bf16; tie taken from the base (these weights are tied)
base_cfg = json.load(open(os.path.join(SRC, "config.json")))
btie = base_cfg.get("text_config", {}).get(
    "tie_word_embeddings", base_cfg.get("tie_word_embeddings", True))
cfg.pop("quantization_config", None)
cfg["dtype"] = "bfloat16"
cfg["tie_word_embeddings"] = btie
cfg["text_config"]["tie_word_embeddings"] = btie
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)

for aux in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
            "chat_template.jinja", "processor_config.json", "special_tokens_map.json"):
    p = os.path.join(SRC, aux)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(DST, aux))

print(f"DONE: {N_NEW} layers (dropped {DROP}); kept {n_layer} layer tensors, "
      f"stripped {n_strip} sharer-kv, copied {n_other} other; total {len(out)} tensors; tie={btie}")
