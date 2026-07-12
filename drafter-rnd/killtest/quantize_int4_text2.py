#!/usr/bin/env python
"""Text-only int4 (W4A16-g128) of gemma-4-E4B for the 3080.
Loads the full multimodal model correctly, transplants the text backbone
(m.model.language_model) + head (m.lm_head) into a clean Gemma4ForCausalLM
(reassign, not copy → RAM-safe), frees the vision/audio towers, then RTN-quantizes."""
import sys, time, gc
import torch
from transformers import Gemma4ForConditionalGeneration, Gemma4ForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL, OUT = sys.argv[1], sys.argv[2]
t = time.time()
print(f"[q] loading full multimodal: {MODEL}", flush=True)
full = Gemma4ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
print(f"[q] full loaded ({time.time()-t:.0f}s)", flush=True)

cfg = full.config.text_config
with torch.device("meta"):                        # zero real memory — avoids the OOM double-alloc
    causal = Gemma4ForCausalLM(cfg)
causal.model = full.model.language_model          # transplant real text backbone (reference)
causal.lm_head = full.lm_head                     # transplant real head
causal.config.architectures = ["Gemma4ForCausalLM"]
_meta = [n for n, p in causal.named_parameters() if p.is_meta]
print(f"[q] leftover meta params after transplant: {len(_meta)} {_meta[:4]}", flush=True)
if _meta:
    print("[q] ABORT: meta params remain"); sys.exit(1)
# free the non-text towers
for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
    if hasattr(full.model, attr):
        setattr(full.model, attr, None)
gc.collect()

n = sum(p.numel() for p in causal.parameters()) / 1e9
print(f"[q] text-only {type(causal).__name__}: {n:.2f}B params; quantizing W4A16 RTN ...", flush=True)
recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=causal, recipe=recipe, output_dir=OUT)
print(f"[q] DONE -> {OUT}  ({time.time()-t:.0f}s total)", flush=True)
