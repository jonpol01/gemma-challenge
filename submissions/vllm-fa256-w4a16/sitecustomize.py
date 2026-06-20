# sitecustomize.py — auto-imported by EVERY Python interpreter on PYTHONPATH,
# including vLLM's spawned EngineCore subprocess (which is where attention-backend
# selection actually runs). Goal: per-layer attention dispatch for gemma-4-E4B —
# upgrade the head_size=256 sliding layers to FlashAttention (exact, lossless: the
# model has no attn_logit_softcapping), leave the head_size=512 global layers on
# whatever was selected (TRITON_ATTN). Fully fail-safe: any error -> passthrough
# (stock TRITON, == our 224 baseline), never a hard crash. Logs every decision so
# the first job's logs reveal the AttentionSelectorConfig shape if extraction misses.
import sys


def _log(m):
    try:
        print("[fa_patch] " + str(m), file=sys.stderr, flush=True)
    except Exception:
        pass


try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as _BE
    from vllm.platforms import cuda as _cuda

    _orig = _cuda.CudaPlatform.__dict__["get_attn_backend_cls"].__func__

    def _head_size(cfg):
        # AttentionSelectorConfig: try attribute then namedtuple/_asdict
        for a in ("head_size", "head_dim"):
            v = getattr(cfg, a, None)
            if isinstance(v, int):
                return v
        try:
            d = cfg._asdict()
            for k in ("head_size", "head_dim"):
                if isinstance(d.get(k), int):
                    return d[k]
        except Exception:
            pass
        return None

    _seen = set()

    def _patched(cls, selected_backend, attn_selector_config, num_heads=None):
        try:
            hs = _head_size(attn_selector_config)
            if hs not in _seen:
                _seen.add(hs)
                _log("head_size=%s incoming_backend=%s" % (hs, selected_backend))
            if hs == 256:
                if selected_backend != _BE.FLASH_ATTN:
                    _log("upgrading head_size=256 -> FLASH_ATTN")
                selected_backend = _BE.FLASH_ATTN
        except Exception as e:  # noqa: BLE001
            _log("dispatch error (passthrough): %r" % e)
        return _orig(cls, selected_backend, attn_selector_config, num_heads)

    _cuda.CudaPlatform.get_attn_backend_cls = classmethod(_patched)
    _log("installed: head_size 256 -> FlashAttention (512 stays as-selected)")
except Exception as e:  # noqa: BLE001
    _log("NOT installed, running stock backend: %r" % e)
