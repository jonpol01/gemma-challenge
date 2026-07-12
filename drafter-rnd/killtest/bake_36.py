#!/usr/bin/env python3
"""Drop ONE language_model decoder layer from a Gemma4 int4 checkpoint (key surgery).
- single-file model.safetensors output (serve.py's lm_head re-prune reads that path)
- drop layers.<DROP>.* tensors, renumber layers.<i>.* (i>DROP) down by one
- slice any GLOBAL (non-decoder-layer) tensor that carries a dim == num_layers (the
  per-layer embeddings, embed_tokens_per_layer) at index DROP so it matches 36 layers
- fix config (num_hidden_layers, layer_types)"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST, DROP = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(DST, exist_ok=True)
st = os.path.join(SRC, "model.safetensors")

with safe_open(st, framework="pt", device="cpu") as f:
    keys = list(f.keys())

lp = next((re.search(r'(.*language_model\.layers\.)\d+\.', k).group(1) for k in keys
           if re.search(r'.*language_model\.layers\.\d+\.', k)), None)
assert lp, "no language_model decoder-layer prefix"
NL = max(int(re.search(re.escape(lp) + r'(\d+)\.', k).group(1)) for k in keys
         if re.search(re.escape(lp) + r'\d+\.', k)) + 1
print(f"[bake] prefix={lp!r} num_layers={NL} drop={DROP}", flush=True)
assert 0 <= DROP < NL
_tc0 = json.load(open(os.path.join(SRC, "config.json"))).get("text_config", {})
H = int(_tc0.get("hidden_size_per_layer_input", 256))
print(f"[bake] hidden_per_layer H={H}; flattened per-layer dim = NL*H = {NL*H}", flush=True)


def newkey(k):
    m = re.match(re.escape(lp) + r'(\d+)\.(.*)', k)
    if not m:
        return k
    i = int(m.group(1))
    if i == DROP:
        return None
    return f"{lp}{i-1}.{m.group(2)}" if i > DROP else k


def fix_tensor(nk, t):
    # decoder-layer tensors are handled by drop/renumber, not here
    if "language_model.layers." in nk:
        return t
    dims = [d for d, s in enumerate(t.shape) if s == NL]
    if len(dims) == 1:
        idx = torch.tensor([i for i in range(NL) if i != DROP])
        t2 = t.index_select(dims[0], idx).contiguous()
        print(f"[bake] sliced {nk} dim{dims[0]} {tuple(t.shape)} -> {tuple(t2.shape)}", flush=True)
        return t2
    if len(dims) > 1:
        print(f"[bake] WARN {nk} has {len(dims)} dims=={NL} {tuple(t.shape)} — NOT sliced", flush=True)
    # flattened per-layer tensor (e.g. embed_tokens_per_layer): a dim == NL*H, layer-major
    if "per_layer" in nk:
        for d, s in enumerate(t.shape):
            if s == NL * H:
                idx = torch.tensor([i for i in range(NL) if i != DROP])
                v = t.unflatten(d, (NL, H)).index_select(d, idx)
                t2 = v.flatten(d, d + 1).contiguous()
                print(f"[bake] sliced(flat) {nk} dim{d} {tuple(t.shape)} -> {tuple(t2.shape)}", flush=True)
                return t2
    return t


newt = {}
with safe_open(st, framework="pt", device="cpu") as f:
    for k in keys:
        nk = newkey(k)
        if nk is None:
            continue
        newt[nk] = fix_tensor(nk, f.get_tensor(k))
print(f"[bake] saving single model.safetensors ({len(newt)} tensors)", flush=True)
save_file(newt, os.path.join(DST, "model.safetensors"))

cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg.get("text_config", cfg)
tc["num_hidden_layers"] = int(tc["num_hidden_layers"]) - 1
if "layer_types" in tc:
    del tc["layer_types"][DROP]
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print(f"[bake] config num_hidden_layers -> {tc['num_hidden_layers']}", flush=True)

for fn in os.listdir(SRC):
    if fn == "model.safetensors" or fn == "config.json" or fn.endswith(".index.json"):
        continue
    s = os.path.join(SRC, fn)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(DST, fn))
print("[bake] DONE -> " + DST, flush=True)
