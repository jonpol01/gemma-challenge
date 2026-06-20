# sitecustomize.py — auto-imported by every interpreter on PYTHONPATH (incl. vLLM's
# EngineCore subprocess). Goal: let chiku-inu's shared `osoi5-v0-baked` (int4 W4A16 g128 +
# UNTIED int4 lm_head) LOAD on stock vLLM 0.23.0, which otherwise asserts:
#   vocab_parallel_embedding.py weight_loader:
#     assert loaded_weight.shape[output_dim] == self.org_vocab_size   -> AssertionError
# (firfir-cast patches model_loader/utils.py + gemma4.py to handle this lm_head.)
#
# This patch is DIAGNOSTIC-FIRST + best-effort:
#   - logs the real shapes (param name, loaded vocab dim, org_vocab_size, output_dim) so the
#     first job reveals the exact mismatch even if the auto-fix is wrong;
#   - attempts the most likely fix: slice/pad the loaded weight along the vocab (output) dim
#     to org_vocab_size (handles a padded-storage mismatch), then calls the original loader.
# Fully fail-safe: any error -> passthrough to the original (which asserts, but we logged).
import sys


def _log(m):
    try:
        print("[osoi5_patch] " + str(m), file=sys.stderr, flush=True)
    except Exception:
        pass


try:
    import torch  # noqa: F401
    import vllm.model_executor.layers.vocab_parallel_embedding as _vpe

    _orig_wl = _vpe.VocabParallelEmbedding.weight_loader
    _logged = set()

    def _resize(w, dim, target):
        cur = w.shape[dim]
        if cur == target:
            return w
        if cur > target:
            idx = [slice(None)] * w.dim()
            idx[dim] = slice(0, target)
            return w[tuple(idx)].contiguous()
        pad = list(w.shape)
        pad[dim] = target - cur
        return torch.cat([w, w.new_zeros(pad)], dim=dim).contiguous()

    def _wl(self, param, loaded_weight):
        try:
            od = getattr(param, "output_dim", None)
            ovs = getattr(self, "org_vocab_size", None)
            nm = getattr(param, "_vllm_name", None) or getattr(param, "tensor_name", "?")
            if od is not None and ovs is not None and loaded_weight.dim() > od:
                cur = loaded_weight.shape[od]
                key = (str(nm), tuple(loaded_weight.shape), od, ovs)
                if key not in _logged:
                    _logged.add(key)
                    _log("name=%s loaded.shape=%s output_dim=%s org_vocab_size=%s num_padded=%s" % (
                        nm, tuple(loaded_weight.shape), od, ovs, getattr(self, "num_embeddings_padded", "?")))
                if cur != ovs:
                    _log("resizing vocab dim %s: %s -> %s" % (od, cur, ovs))
                    loaded_weight = _resize(loaded_weight, od, ovs)
        except Exception as e:  # noqa: BLE001
            _log("resize/diagnostic error (passthrough): %r" % e)
        return _orig_wl(self, param, loaded_weight)

    _vpe.VocabParallelEmbedding.weight_loader = _wl
    _log("installed vocab weight_loader resize+diagnostic patch")
except Exception as e:  # noqa: BLE001
    _log("NOT installed (stock loader): %r" % e)
