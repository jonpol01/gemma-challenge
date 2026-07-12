#!/usr/bin/env python
"""Body-layer prune + in-memory head-prune on the faithful QAT-ct.

Serve the QAT-ct checkpoint with config.num_hidden_layers + config.layer_types
edited to drop the layers in DROP_LAYERS (e.g. "2,3,4"); serve_patch_inmem
(auto-imported by sitecustomize) remaps the weight stream to match (skip+renumber
dropped layers, slice the global per-layer-embedding) and builds the 12k head
in-memory. Faithful int4 body otherwise untouched. Capability-gated.
"""
import json
import os
import sys

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
DRAFTER = os.environ.get(
    "DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
)
SRC = os.environ.get("LOCAL_MODEL_DIR", "/tmp/qatct")


def _parse_drop() -> list:
    s = os.environ.get("DROP_LAYERS", "").strip()
    if not s:
        return []
    return sorted({int(x) for x in s.replace(" ", "").split(",") if x != ""})


def prep() -> None:
    if not os.path.exists(os.path.join(SRC, "config.json")):
        from huggingface_hub import snapshot_download
        print(f"[prep] downloading {MODEL} -> {SRC} (unmodified)", flush=True)
        snapshot_download(MODEL, local_dir=SRC)
    else:
        print(f"[prep] reusing {SRC}", flush=True)


def _edit_config_for_drop() -> None:
    drop = _parse_drop()
    if not drop:
        return
    dropset = set(drop)
    cfgp = os.path.join(SRC, "config.json")
    cfg = json.load(open(cfgp))
    tc = cfg.get("text_config", cfg)
    n = tc.get("num_hidden_layers")
    if n is None:
        print("[layerdrop] WARN: no num_hidden_layers in config", flush=True)
        return
    new_n = n - len(drop)
    if tc.get("num_hidden_layers") == new_n:
        print(f"[layerdrop] config already at {new_n} layers", flush=True)
        return
    tc["num_hidden_layers"] = new_n
    lt = tc.get("layer_types")
    if isinstance(lt, list) and len(lt) == n:
        tc["layer_types"] = [t for i, t in enumerate(lt) if i not in dropset]
    # gemma-4 KV-sharing: the last num_kv_shared_layers layers reuse earlier layers' KV.
    # Reduce it by however many dropped layers fall in that (original) sharing region,
    # so we only drop layers that feed no one (avoids the orphaned-KV breakage).
    nkvs = tc.get("num_kv_shared_layers")
    if nkvs:
        share_start = n - nkvs
        dropped_sharers = sum(1 for d in drop if d >= share_start)
        if dropped_sharers:
            tc["num_kv_shared_layers"] = nkvs - dropped_sharers
            print(f"[layerdrop] num_kv_shared_layers {nkvs}->{tc['num_kv_shared_layers']} ({dropped_sharers} sharers dropped)", flush=True)
    json.dump(cfg, open(cfgp, "w"), indent=2)
    print(
        f"[layerdrop] config edited: num_hidden_layers {n}->{new_n}, dropped {drop}; "
        f"layer_types now {len(tc.get('layer_types', []))} entries",
        flush=True,
    )


def main() -> None:
    prep()
    _edit_config_for_drop()
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PCK04_KEEPSET"] = os.path.join(here, "pck04_keepset.json")
    os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")

    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6") or "0")
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", SRC,
        "--served-model-name", os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host", os.environ.get("HOST", "0.0.0.0"),
        "--port", os.environ.get("PORT", "8000"),
        "--dtype", "bfloat16",
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "1"),
        "--trust-remote-code", "--no-enable-log-requests",
    ]
    mnbt = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if mnbt:
        args += ["--max-num-batched-tokens", mnbt]
    if num_spec > 0:
        args += ["--speculative-config", json.dumps({"model": DRAFTER, "num_speculative_tokens": num_spec})]
    print("[serve] launching:", " ".join(args), flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
