# The thin frontier layer — realizing the A10G hardware limit on osoi5-int4

The verified board is a noise-tie at ~506.6–507.0 because everyone runs the same shared `hayai`/osoi5
stack. This doc defines the **hardware limit** for that substrate, composes every **gate-safe**
(prompt-invariant) lever we've validated into one "thin layer" that pins that limit reproducibly, and
states plainly what the layer can and cannot do. Companion: [`RESEARCH-2026-06-23-speed-levers.md`](RESEARCH-2026-06-23-speed-levers.md).

## 1. What "the limit" is

Decode at `max_concurrency=1` is **memory-bandwidth-bound**:

```
tok/s  ≈  (HBM_bandwidth × roofline_fraction) / bytes_read_per_token
```

| term | value | can we move it? |
|---|---|---|
| HBM bandwidth | A10G ~600 GB/s | no — hardware |
| bytes_read_per_token | fixed by osoi5-int4 (body + 12k head + drafter) | **no on this substrate** — sub-4-bit is Ampere-blocked, harder prune busts PPL, drafter changes are gate-doomed |
| roofline_fraction | Marlin ~77% at M=1 (measured) | **no** — tinygemm/GemLite don't beat Marlin here; ~77% is intrinsic to int4-GEMV-at-M=1 |

→ **The hardware limit for the osoi5-int4 substrate is ~507–510 tok/s.** The entire verified board sits
there. We're already at it (506.74). A thin layer **cannot break the bound** — it can only **realize it
more fully and reproducibly.** To move the bound you must change the substrate (§4).

## 2. The composition (gate-safe levers only)

Prompt-sensitive levers fail the private re-verify, so the layer includes **only prompt-invariant** ones:

| lever | status | why it's gate-safe | effect |
|---|---|---|---|
| Full-graph capture (ONEGRAPH/loopgraph) | ✅ have | fixed launch count, not prompt-dependent | removes per-step launch overhead |
| Marlin int4 GEMV | ✅ have (confirmed best) | numerics | the math kernel, ~77% roofline (near-ceiling) |
| pck04 12k lm_head scatter | ✅ have | baked into checkpoint | head GEMM → tiny |
| fused-sparse-argmax / split-KV / DIXIE accept | ✅ have | fixed kernels | fuse the cheap steps |
| FA-sliding window | ✅ have | window is structural | sliding-window attention kernel |
| tcmalloc / orjson / DETOK_ENDONLY | ✅ have | host-side | serving/detok overhead |
| **synthetic warmup bridge** (N=64, pre-JIT all kernels) | ✅ firfir's, repro'd verbatim (§2.1) | **synthetic prompts → identical public/private** | removes cold-JIT/autotune from the timed window (+ stability → verifies) |
| **config: w188 + ctk49** | ✅ firfir's | ctk = pure compute knob; w188 verified | best current draw |
| **host-overhead / D2H-sync elimination** | ✅ **measured → dead (+0)** | fixed per-step cost | GPU already ~99% utilized; D2H hidden — see §3 |

### 2.1 Provenance — our 507.00 is firfir's stack, reproduced verbatim (a tie, not a beat)

Our `vllm-warmup-w188-ctk49-v1` (the 507.00 draw) is **not a different or better stack than firfir-cast's**
— it is firfir's verified `w188-ctk49-n64` copied **byte-for-byte**. Checksum (2026-06-23, our copy vs
firfir's original):

| files | vs firfir |
|---|---|
| `serve.py`, `sitecustomize.py`, `serve_patch_warmup_bridge.py`, `fa_sliding_patch.py`, `splitkv_verify_patch.py`, `serve_patch_pck04.py`, `serve_patch_precache.py`, `detok_endonly.py`, `lsk_patch.py`, `steptime_patch.py` | **byte-identical (10/10)** |
| `manifest.json` | differs **only** in `name` + `description` (cosmetic); the `env` block (w188, ctk49, `WARMUP_*`, weights/drafter buckets) is identical |

So our **507.00 ties firfir at the verified top by *reproducing* their stack** — exactly as our 506.74
already reproduced their hayai stack. This is legitimate (the challenge is collaborative: reproduce shared
frontier stacks, don't re-invent), but it is **reproduction, not innovation** — we are not ahead of
firfir, we are running their exact config and drawing the same number. **To be genuinely ahead** requires
a brick firfir doesn't have: the unbuilt host-overhead/D2H piece (§3, marginal) or — the real one — the
substrate-change PPL-headroom prune (§4).

**Explicitly OUT (falsified — do not add):** tinygemm/GemLite kernel swap (no win vs Marlin),
sub-4-bit/quant changes (Ampere-blocked / PPL), drafter/PARD changes (gate-doomed), `w160` (busts
private), Marlin atomic-add (slower), megakernel (slower at batch=1).

## 3. The host-overhead / D2H brick — MEASURED, dead (+0)

We measured it instead of building blind, via the in-tree `STEPTIME=1` probe (`steptime_patch.py`) on a
real A10G bench (2026-06-23, n≈30k steady-state steps):

| record | GPU compute | host gap | call wall |
|---|---|---|---|
| `exec` (target verify + accept) | **6.471 ms** | 1.595 ms (p50) | 6.352 ms |
| `draft` (propose K=7) | 1.416 ms | ~0 | 0.272 ms |

Per decode iteration: GPU busy = `6.471 + 1.416 = 7.89 ms`; wall = `cpu + gap = 6.35 + 1.60 = 7.95 ms`
→ **~0.06 ms idle ≈ 0.8% → GPU is ~99% utilized.** The host gap (which includes the `valid_counts.item()`
D2H) is **overlapped with the drafter's GPU execution** — hidden, not exposed. Eliminating it recovers
**<1% (≤~4 tok/s ceiling, realistically +0–1)** and would require deep, greedy-identity-fragile
GPUModelRunner surgery. **Verdict: not worth building — the engine is already pipelined to the GPU
bound.** (This confirms the research's prediction directly.)

**Consequence: there is no remaining gate-safe brick.** The warmup pins the top, the D2H is already
hidden, every other lever is falsified — the thin layer is complete, and ~507 is the A10G hardware
ceiling for osoi5-int4 (decode is ~99% GPU-bound on int4 weight bandwidth).

## 4. Past the limit — the only real ceiling-raiser

The thin layer pins ~507–510. To go *materially* higher you must cut `bytes_read_per_token`, which means
a **different substrate**: spend the **faithful QAT base's PPL headroom** (ppl ~1.978 vs the ~2.42 cap;
osoi5 is maxed at 2.394) on a **more PPL-efficient prune** than osoi5's blunt 5-layer cut (KV-share-aware
layer-drop, not arbitrary early layers — those bust the residual stream). If that prune reaches the cap at
fewer effective bytes/token, the bandwidth math gives a genuinely higher ceiling — and because it's
quant/prune numerics, it's **prompt-invariant → it verifies.** Uncertain payoff, multi-session; the only
lever that moves the bound instead of realizing it.

## 5. Honest EV

- **Thin layer (COMPLETE):** 506.74 → **~507 reliably + verified** (warmup pins it; D2H measured dead at
  +0; everything else falsified). The drawn 507.00 ([`vllm-warmup-w188-ctk49-v1`](../submissions)) equals
  firfir's verified top — it pins the hardware-limited ceiling of a saturated board. The decode is ~99%
  GPU-bound (§3), so there is **no remaining gate-safe lever on this substrate.** Not a breakthrough —
  *that is the hardware ceiling.*
- **Substrate change (§4):** the ONLY path to a materially higher verified number — reduce
  bytes-read-per-token via a more PPL-efficient prune on the faithful base. A real, uncertain R&D build.
