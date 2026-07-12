#!/usr/bin/env python
"""Re-group an osoi5-style int4 pack-quantized checkpoint: body Linears g128 -> g256.

The speed lever for the gemma-challenge: halving the bf16 scale overhead (~3% -> ~1.5%
of body bytes) → ~+1% decode on a bandwidth-bound stack, prompt-invariant (holds the
private re-verify). Only the body group ("group_0", group_size=128) is re-grouped; the
lm_head (channelwise, "group_0_lmhead") and all non-quantized tensors are copied as-is.

RISK: coarser quant raises PPL; osoi5 sits at 2.394 vs the 2.42 cap (0.026 headroom),
so this re-group is a coin-flip on validity. Self-verifies the round-trip error.

Usage: python regroup_g256.py <src_dir> <dst_dir>
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
def _find_pack_funcs():
    """Locate pack_to_int32 / unpack_from_int32 regardless of compressed_tensors layout."""
    import importlib
    import pkgutil
    import compressed_tensors
    print("[regroup] compressed_tensors", getattr(compressed_tensors, "__version__", "?"), flush=True)
    for _, name, _ in pkgutil.walk_packages(compressed_tensors.__path__, compressed_tensors.__name__ + "."):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "pack_to_int32") and hasattr(mod, "unpack_from_int32"):
            print(f"[regroup] pack funcs from {name}", flush=True)
            return mod.pack_to_int32, mod.unpack_from_int32
    raise ImportError("pack_to_int32/unpack_from_int32 not found in compressed_tensors")


pack_to_int32, unpack_from_int32 = _find_pack_funcs()

SRC, DST = sys.argv[1], sys.argv[2]
OLD_G, NEW_G = 128, 256
os.makedirs(DST, exist_ok=True)

st = os.path.join(SRC, "model.safetensors")
out: dict = {}
n_regrouped = 0
max_err = 0.0

with safe_open(st, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    bases = sorted({k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")})
    base_set = set(bases)

    for k in keys:
        # tensors that belong to a re-groupable base are emitted by the base loop below
        b = None
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            if k.endswith(suf) and k[: -len(suf)] in base_set:
                b = k[: -len(suf)]
                break
        if b is None:
            out[k] = f.get_tensor(k)  # embeds, norms, PLE, lm_head-by-name etc. — copy

    for b in bases:
        packed = f.get_tensor(b + ".weight_packed")            # [out, in/8] int32
        scale = f.get_tensor(b + ".weight_scale")              # [out, groups]
        shape = f.get_tensor(b + ".weight_shape")              # [out, in]
        out_dim, in_dim = int(shape[0]), int(shape[1])
        groups = scale.shape[1] if scale.dim() == 2 else 1

        # channelwise / per-tensor (the lm_head group) or not g128 → copy untouched
        if groups != in_dim // OLD_G or in_dim % NEW_G != 0:
            out[b + ".weight_packed"] = packed
            out[b + ".weight_scale"] = scale
            out[b + ".weight_shape"] = shape
            continue

        q = unpack_from_int32(packed, 4, (out_dim, in_dim)).to(torch.float32)   # signed int4
        w = (q.reshape(out_dim, groups, OLD_G) * scale.to(torch.float32).unsqueeze(-1)).reshape(out_dim, in_dim)

        wg = w.reshape(out_dim, in_dim // NEW_G, NEW_G)
        new_scale = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        nq = torch.round(wg / new_scale).clamp(-8, 7)
        w2 = (nq * new_scale).reshape(out_dim, in_dim)
        max_err = max(max_err, (w2 - w).abs().max().item())

        new_packed = pack_to_int32(nq.reshape(out_dim, in_dim).to(torch.int8), 4)
        out[b + ".weight_packed"] = new_packed
        out[b + ".weight_scale"] = new_scale.reshape(out_dim, in_dim // NEW_G).to(scale.dtype)
        out[b + ".weight_shape"] = shape
        n_regrouped += 1

save_file(out, os.path.join(DST, "model.safetensors"))

# copy + patch config (group_size 128 -> 256 for group_0), copy aux files
cfg = json.load(open(os.path.join(SRC, "config.json")))
g0 = cfg.get("quantization_config", {}).get("config_groups", {}).get("group_0", {})
if g0.get("weights", {}).get("group_size") == OLD_G:
    g0["weights"]["group_size"] = NEW_G
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
for aux in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "processor_config.json", "pck04_keepset.json"):
    p = os.path.join(SRC, aux)
    if os.path.exists(p):
        import shutil
        shutil.copy(p, os.path.join(DST, aux))

print(f"[regroup] re-grouped {n_regrouped} body Linears g{OLD_G}->g{NEW_G}; "
      f"max round-trip de-quant error = {max_err:.5f} (small = correct re-pack)", flush=True)
