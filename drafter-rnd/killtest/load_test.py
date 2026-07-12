#!/usr/bin/env python3
"""Isolated int4 load + one text forward. Times the load, prints shared_kv_states shapes."""
import time, torch
from transformers import Gemma4ForConditionalGeneration
P = "/dixie/weights/int4-pck04-16k"
t0 = time.time()
print(f"[lt] START load {P}", flush=True)
m = Gemma4ForConditionalGeneration.from_pretrained(
    P, dtype=torch.bfloat16, trust_remote_code=True,
    ignore_mismatched_sizes=True, device_map="cuda").eval()
print(f"[lt] LOADED in {time.time()-t0:.0f}s; GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# locate text decoder
lm = None
for path in ("model.language_model", "language_model", "model"):
    o = m; ok = True
    for a in path.split("."):
        if hasattr(o, a): o = getattr(o, a)
        else: ok = False; break
    if ok and hasattr(o, "layers") and hasattr(o, "embed_tokens"):
        lm = o; print(f"[lt] text decoder at '{path}' layers={len(o.layers)}", flush=True); break
assert lm is not None, "no text decoder"

ids = torch.tensor([[2, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
t1 = time.time()
with torch.no_grad():
    out = lm(input_ids=ids, return_shared_kv_states=True, use_cache=False)
print(f"[lt] fwd {time.time()-t1:.1f}s; last_hidden={tuple(out.last_hidden_state.shape)}", flush=True)
skv = out.shared_kv_states
print(f"[lt] skv keys={list(skv.keys()) if skv else None}", flush=True)
if skv:
    for k, (K, V) in skv.items():
        print(f"[lt]   {k}: K={tuple(K.shape)} V={tuple(V.shape)}", flush=True)
print("[lt] DONE", flush=True)
