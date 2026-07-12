"""In-memory head-prune for the UNMODIFIED, TIED QAT-ct checkpoint.

serve_patch_pck04 rebuilds lm_head at __init__ and expects a pruned K-row head
*in the checkpoint* (osoi5 was baked that way). QAT-ct ships TIED, and untying it
on disk fights vLLM's loader (embed gets tied into the K-head -> shape assert).

This variant never touches the checkpoint:

  1. The model loads normally, TIED (lm_head shares the full 262144-row embed).
     Identical to the proven stock serve (229 tps) -> no weight-loader assert.
  2. AFTER load_weights returns (embed resident on GPU), build a SEPARATE
     ParallelLMHead(K) and copy embed_tokens.weight[keep_ids] into it, then
     replace self.lm_head. The decode logits GEMM is now over K=12k rows, not
     262k -> the speed win. embed_tokens stays full (input gather is cheap).
  3. compute_logits scatters [M, K] -> [M, full_vocab] with -inf padding
     (identical to serve_patch_pck04, the proven path).

Because the K-head rows ARE embed[keep_ids] verbatim, logits at kept positions
are bit-identical to the tied full head -> greedy-token-identity holds wherever
the greedy token is in the keepset.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

PCK04_KEEPSET_PATH = os.environ.get("PCK04_KEEPSET", "")
_TARGET_MODULE = "vllm.model_executor.models.gemma4"
_TARGET_CLASS = "Gemma4ForCausalLM"

_state: dict[str, Any] = {"device_cache": {}}


def _load_keepset(path: str) -> tuple[list[int], int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[inmem] PCK04_KEEPSET={path!r} does not exist")
    d = json.loads(p.read_text())
    keep = d["keep_ids"]
    full = int(d.get("full_vocab") or d.get("vocab_size") or 0)
    if not full:
        raise ValueError(f"[inmem] keepset {path!r} has no full_vocab/vocab_size")
    return keep, full


def _scatter_to_full_vocab(pruned_logits: Any, keep_ids: list[int], full_vocab: int) -> Any:
    import torch  # type: ignore

    device = pruned_logits.device
    cache = _state["device_cache"].setdefault(str(device), {})
    keep_idx = cache.get("keep_idx")
    if keep_idx is None:
        keep_idx = torch.tensor(keep_ids, dtype=torch.long, device=device)
        cache["keep_idx"] = keep_idx
    K = len(keep_ids)
    M = pruned_logits.shape[0]
    assert pruned_logits.shape[-1] == K, (
        f"[inmem] FINGERPRINT FAIL: expected pruned logits [M,{K}], got {list(pruned_logits.shape)}"
    )
    if M <= 16:
        oc = cache.setdefault("out_cache", {})
        key = (M, pruned_logits.dtype)
        out = oc.get(key)
        if out is None:
            out = torch.full((M, full_vocab), float("-inf"), dtype=pruned_logits.dtype, device=device)
            oc[key] = out
    else:
        out = torch.full((M, full_vocab), float("-inf"), dtype=pruned_logits.dtype, device=device)
    out.index_copy_(1, keep_idx, pruned_logits)
    return out


def _apply_inmem_patch(module: Any) -> None:
    import torch  # type: ignore

    if not PCK04_KEEPSET_PATH:
        print("[inmem] PCK04_KEEPSET not set — in-memory head-prune INACTIVE", file=sys.stderr, flush=True)
        return

    keep_ids, full_vocab = _load_keepset(PCK04_KEEPSET_PATH)
    K = len(keep_ids)

    cls = getattr(module, _TARGET_CLASS, None)
    assert cls is not None, f"[inmem] FINGERPRINT FAIL: {_TARGET_CLASS} not in {module.__name__}"
    original_load_weights = getattr(cls, "load_weights", None)
    original_compute_logits = getattr(cls, "compute_logits", None)
    assert original_load_weights is not None, "[inmem] FINGERPRINT FAIL: load_weights not found"
    assert original_compute_logits is not None, "[inmem] FINGERPRINT FAIL: compute_logits not found"

    ParallelLMHead = getattr(module, "ParallelLMHead", None)
    if ParallelLMHead is None:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore

    def _find_embed(self_model: Any) -> Any:
        m = getattr(self_model, "model", None)
        emb = getattr(m, "embed_tokens", None) if m is not None else None
        if emb is not None and hasattr(emb, "weight"):
            return emb
        for name, sub in self_model.named_modules():
            if name.endswith("embed_tokens") and "per_layer" not in name and hasattr(sub, "weight"):
                return sub
        raise RuntimeError("[inmem] could not locate main embed_tokens")

    @functools.wraps(original_load_weights)
    def load_weights_inmem(self_model: Any, *args: Any, **kwargs: Any) -> Any:
        ret = original_load_weights(self_model, *args, **kwargs)
        emb = _find_embed(self_model)
        w = emb.weight
        device, dtype, hidden = w.device, w.dtype, w.shape[1]
        existing_prefix = getattr(getattr(self_model, "lm_head", None), "prefix", None) or "lm_head"
        idx = torch.tensor(keep_ids, dtype=torch.long, device=device)
        new_head = ParallelLMHead(
            K, hidden, bias=False, params_dtype=dtype,
            org_num_embeddings=K, quant_config=None, prefix=existing_prefix,
        ).to(device)
        with torch.no_grad():
            src = w.data.index_select(0, idx).to(dtype)
            assert new_head.weight.shape == src.shape, (
                f"[inmem] head {tuple(new_head.weight.shape)} != src {tuple(src.shape)}"
            )
            new_head.weight.data.copy_(src)
        self_model.lm_head = new_head
        print(
            f"[inmem] built pruned lm_head {tuple(new_head.weight.shape)} from "
            f"{tuple(w.shape)} embed[keep] on {device} (K={K}) pid {os.getpid()}",
            file=sys.stderr, flush=True,
        )
        return ret

    @functools.wraps(original_compute_logits)
    def compute_logits_inmem(self_model: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        pruned = original_compute_logits(self_model, hidden_states, *args, **kwargs)
        if pruned is None:
            return None
        return _scatter_to_full_vocab(pruned, keep_ids, full_vocab)

    cls.load_weights = load_weights_inmem
    cls.compute_logits = compute_logits_inmem
    print(
        f"[inmem] patched {_TARGET_CLASS}.load_weights + compute_logits "
        f"(K={K}, full_vocab={full_vocab}, keepset={PCK04_KEEPSET_PATH!r}) pid {os.getpid()}",
        file=sys.stderr, flush=True,
    )


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: Any, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn: Any) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._patch_fn)
        return spec


sys.meta_path.insert(0, _TargetFinder(_TARGET_MODULE, _apply_inmem_patch))
