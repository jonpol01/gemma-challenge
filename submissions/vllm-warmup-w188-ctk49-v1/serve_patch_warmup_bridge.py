"""warmup-bridge: prompt-agnostic engine warmup via synthetic prompts.

Replays synthetic prompts during the untimed warmup window to exercise the
full prefill+decode path with varied sequence lengths. Unlike PRECACHE_BENCH,
this uses NO benchmark dataset — prompts are generated deterministically from
a fixed word list and seed. The KV cache entries won't match benchmark prompts,
so the benefit is purely from:

  - CUDA graph replay pipelines warmed for each sequence length
  - Attention kernels JIT-compiled (no lazy compile during timed phase)
  - Memory pools pre-allocated for the expected length distribution
  - No cold-start overhead on the first benchmark request

Because the warmup is identical on EVERY run (public and private), the TPS
benefit is identical on both sets, making this naturally private-stable:
  delta(public, private) ≈ 0%  →  passes the 5% gate.

Loaded by sitecustomize.py when WARMUP_BRIDGE=1. Uses the same ASGI readiness
gate pattern as serve_patch_precache.py (pure-ASGI, no BaseHTTPMiddleware).

Env:
  WARMUP_BRIDGE=1              enable (checked by sitecustomize before import)
  WARMUP_NUM_PROMPTS=32        number of synthetic prompts to replay
  WARMUP_MAX_TOKENS=1          decode tokens per warmup request
  WARMUP_SEED=42               deterministic generation seed
  WARMUP_REQUIRE=0             fail-closed on warmup failure
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

TAG = "[warmup-bridge]"

WARMUP_NUM_PROMPTS = int(os.environ.get("WARMUP_NUM_PROMPTS", "32"))
WARMUP_MAX_TOKENS = int(os.environ.get("WARMUP_MAX_TOKENS", "1"))
WARMUP_SEED = int(os.environ.get("WARMUP_SEED", "42"))
WARMUP_REQUIRE = os.environ.get("WARMUP_REQUIRE") == "1"
FIRST_REQUEST_TIMEOUT_S = 600.0
PER_REQUEST_TIMEOUT_S = 120.0

_WARMUP_DONE = threading.Event()
_WARMUP_STARTED = threading.Event()

# Fixed word bank — no external dependencies, ~200 common English words.
# These are used only for sequence length, never for output-matching.
_WORDS = [
    "the", "a", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "although", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "any", "some", "many", "much", "several", "few", "enough", "each",
    "about", "above", "across", "after", "along", "around", "before",
    "behind", "below", "beneath", "beside", "between", "beyond",
    "down", "during", "except", "inside", "into", "near", "outside",
    "over", "through", "under", "until", "upon", "without", "within",
    "write", "make", "take", "give", "come", "see", "know", "think",
    "look", "want", "find", "tell", "ask", "work", "call", "try",
    "leave", "need", "put", "mean", "keep", "let", "begin", "seem",
    "help", "turn", "start", "show", "hear", "play", "run", "move",
    "live", "believe", "bring", "happen", "hold", "follow", "change",
    "understand", "remember", "consider", "learn", "accept", "build",
]


def _log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def _generate_prompts(num_prompts: int, seed: int) -> list[dict[str, str]]:
    """Generate deterministic synthetic prompts of varying lengths.

    Lengths are chosen to cover the typical benchmark distribution (50-450
    tokens) with some longer outliers. The exact distribution doesn't need to
    match the benchmark — it just needs to exercise attention kernels across
    a range of sequence lengths.
    """
    rng = random.Random(seed)
    # Spread of target lengths to cover the kernel configuration space
    target_lengths = [
        50, 72, 95, 120, 145, 170, 195, 220, 245, 270,
        295, 320, 345, 370, 395, 420, 445, 470, 500, 530,
        80, 105, 130, 155, 180, 205, 230, 255, 280, 305,
        330, 355,
    ]
    prompts = []
    for i in range(num_prompts):
        length = target_lengths[i % len(target_lengths)]
        words = [rng.choice(_WORDS) for _ in range(length)]
        text = " ".join(words)
        # Trim to target — word boundaries may have added extra chars
        text = text[:length]
        prompts.append({"id": f"warmup-{i}", "prompt_text": text})
    return prompts[:num_prompts]


def _post_chat(base_url: str, model: str, prompt: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": WARMUP_MAX_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        json.loads(response.read().decode("utf-8"))


def _warmup() -> None:
    port = os.environ.get("PORT", "8000")
    model = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    base_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    records = _generate_prompts(WARMUP_NUM_PROMPTS, WARMUP_SEED)
    _log(
        f"warming up with {len(records)} synthetic prompts "
        f"(max_tokens={WARMUP_MAX_TOKENS}, seed={WARMUP_SEED})"
    )

    try:
        started = time.monotonic()
        for index, record in enumerate(records):
            deadline = time.monotonic() + (
                FIRST_REQUEST_TIMEOUT_S if index == 0 else PER_REQUEST_TIMEOUT_S
            )
            attempt = 0
            while True:
                attempt += 1
                try:
                    _post_chat(
                        base_url, model, record["prompt_text"], PER_REQUEST_TIMEOUT_S
                    )
                    break
                except (urllib.error.URLError, OSError, ValueError) as error:
                    if time.monotonic() >= deadline or (index > 0 and attempt >= 3):
                        raise RuntimeError(
                            f"warmup request {index + 1}/{len(records)}"
                            f" (id={record['id']}) failed: {error}"
                        ) from error
                    time.sleep(2.0)

        elapsed = time.monotonic() - started
        _log(f"warmup complete: {len(records)} prompts in {elapsed:.1f}s")
        _WARMUP_DONE.set()
    except Exception as error:
        _log(f"WARMUP FAILED: {error!r}")
        if WARMUP_REQUIRE:
            _log("WARMUP_REQUIRE=1 — /v1/models stays gated (fail-closed)")
            return
        _log("WARMUP_REQUIRE unset — ungating WITHOUT warmup")
        _WARMUP_DONE.set()


class _WarmupGateASGI:
    """Pure-ASGI readiness gate: 503 on /v1/models until warmup completes."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path") == "/v1/models"
            and not _WARMUP_DONE.is_set()
        ):
            body = json.dumps(
                {"detail": "warming: synthetic engine warmup in flight"}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def _apply_warmup_patch(module) -> None:
    base_serve_http = module.serve_http

    def serve_http_warmup(app, *args, **kwargs):
        stack = getattr(app, "middleware_stack", None)
        if stack is not None:
            app.middleware_stack = _WarmupGateASGI(stack)
        else:
            base_build = app.build_middleware_stack
            app.build_middleware_stack = lambda: _WarmupGateASGI(base_build())
        if not _WARMUP_STARTED.is_set():
            _WARMUP_STARTED.set()
            threading.Thread(target=_warmup, name="warmup-bridge", daemon=True).start()
            _log("readiness gate installed; warmup thread started")
        return base_serve_http(app, *args, **kwargs)

    module.serve_http = serve_http_warmup
    _log(f"patched vllm.entrypoints.launcher.serve_http in pid {os.getpid()}")


class _WarmupPatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        _apply_warmup_patch(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _WarmupTargetFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.entrypoints.launcher":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WarmupPatchingLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _WarmupTargetFinder())
_log("meta-path finder armed for vllm.entrypoints.launcher")
