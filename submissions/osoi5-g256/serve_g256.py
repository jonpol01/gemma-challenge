#!/usr/bin/env python
"""Wrapper: download osoi5 -> re-group body g128->g256 -> hand off to the hayai serve.

The hayai serve.py downloads WEIGHTS_BUCKET into LOCAL_MODEL_DIR (skipping if already
present) then head-prunes + serves. We pre-stage a g256 copy at LOCAL_MODEL_DIR so its
download is skipped and the rest of its pipeline (head-prune 16k->12k, engine, serve)
runs on the coarser-quant body.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/tmp/osoi5-src"
G256 = "/tmp/osoi5-g256"
WB = os.environ["WEIGHTS_BUCKET"]

if not os.path.exists(os.path.join(G256, "config.json")):
    if not os.path.exists(os.path.join(SRC, "config.json")):
        print(f"[g256-wrap] downloading {WB} -> {SRC}", flush=True)
        subprocess.run(["hf", "buckets", "sync", WB, SRC], check=True)
    print(f"[g256-wrap] re-grouping body g128->g256: {SRC} -> {G256}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "regroup_g256.py"), SRC, G256], check=True)
    # persist the g256 weights so they can be benched via the clean hayai serve (no wrapper)
    print("[g256-wrap] uploading g256 -> weights/osoi5-g256 (persist for clean re-bench)", flush=True)
    subprocess.run(
        ["hf", "buckets", "sync", G256,
         "hf://buckets/gemma-challenge/gemma-mikasa-inbound/weights/osoi5-g256"],
        check=False,
    )
else:
    print(f"[g256-wrap] reusing {G256}", flush=True)

# point the hayai serve at the g256 copy (its ensure_weights will skip the download)
os.environ["LOCAL_MODEL_DIR"] = G256
print(f"[g256-wrap] handoff -> serve.py (LOCAL_MODEL_DIR={G256})", flush=True)
os.execvpe(sys.executable, [sys.executable, os.path.join(HERE, "serve.py")], os.environ)
