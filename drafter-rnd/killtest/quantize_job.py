#!/usr/bin/env python
"""In-job full-model int4 quant on the A10G GPU (24GB fits the ~16GB bf16 + quant).
Quantizes the mounted multimodal bf16 W4A16, copies tokenizer/processor files."""
import sys, time, os, shutil
import torch
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL, OUT = sys.argv[1], sys.argv[2]
t = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
print(f"[q] loaded {type(model).__name__} {time.time()-t:.0f}s", flush=True)
recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe, output_dir=OUT)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "chat_template.jinja", "generation_config.json", "processor_config.json",
          "preprocessor_config.json"):
    s = os.path.join(MODEL, f)
    if os.path.exists(s):
        shutil.copy(s, OUT)
print(f"[q] DONE -> {OUT} ({time.time()-t:.0f}s)", flush=True)
