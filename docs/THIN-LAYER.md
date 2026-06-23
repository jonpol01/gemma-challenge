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
| **host-overhead / D2H-sync elimination** | ⬜ **not built** | fixed per-step cost | the only unbuilt brick — see §3 |

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

## 3. The one unbuilt brick: host-overhead / D2H elimination

vLLM's per-step accept handoff still does a host-side `valid_counts.item()` D2H read (reduced but not
eliminated by `DIXIE_FUSED_ACCEPT_PREP`). **nsys-gated:** profile an A10G serve run; if there's a
>5 µs *uncovered* host stall between graph replays attributable to it, patch it (constant-stride padded
replay, with a greedy-identity check on the padding). Realistic gain ~+0–1 tok/s — most likely it's
already hidden in the host scheduling gap. Do **not** build it blind; nsys first.

## 4. Past the limit — the only real ceiling-raiser

The thin layer pins ~507–510. To go *materially* higher you must cut `bytes_read_per_token`, which means
a **different substrate**: spend the **faithful QAT base's PPL headroom** (ppl ~1.978 vs the ~2.42 cap;
osoi5 is maxed at 2.394) on a **more PPL-efficient prune** than osoi5's blunt 5-layer cut (KV-share-aware
layer-drop, not arbitrary early layers — those bust the residual stream). If that prune reaches the cap at
fewer effective bytes/token, the bandwidth math gives a genuinely higher ceiling — and because it's
quant/prune numerics, it's **prompt-invariant → it verifies.** Uncertain payoff, multi-session; the only
lever that moves the bound instead of realizing it.

## 5. Honest EV

- **Thin layer:** 506.74 → **~507–508 reliably + verified** (warmup ~+0.3, D2H ~+0–1, best config draw).
  It pins the hardware-limited top of a saturated board. Not a breakthrough — *that's the ceiling.*
- **Substrate change (§4):** the only path to a materially higher verified number; a real R&D build.
